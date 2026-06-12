"""MAVIS · Project 3 — web demo (Gradio).

Three tabs over the same engine (``mavis_rag_engine.MavisRagEngine``):

  1. **Assistant (RAG)** — ask in natural language, get a cited answer plus the
     numbered context blocks it was built from and the actual thumbnails of the
     most related corpus videos (visual evidence).
  2. **Video search** — the multimodal core, on its own: query by text and/or by
     image (upload a thumbnail) and rank the corpus by the fused RRF engine.
  3. **Evaluation** — the notebook's measured numbers, read back from ``cache/``:
     video retrieval ablation, document retrieval (base vs hybrid vs +re-rank),
     and answer quality on five axes, with the figures the notebook saved.

The engine is built lazily on the first request (and warmed at launch in
``__main__``) so importing this module stays cheap for the offline tests.
"""

from __future__ import annotations

import os

# macOS: torch and faiss each bundle their own libomp (OpenMP) runtime; loading
# both into one process aborts with "OMP: Error #15 ... libomp.dylib already
# initialized". Permit the duplicate runtime BEFORE importing anything that pulls
# in OpenMP (gradio→numpy, torch, faiss). Export KMP_DUPLICATE_LIB_OK=FALSE to
# opt out if you've deduplicated libomp at the system level.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import html
from pathlib import Path

import gradio as gr

# --- gradio_client schema-walker bool guard (must run BEFORE demo.launch) ---
# Gradio's startup self-ping introspects every component's JSON schema. When a
# schema has `additionalProperties: True` (a bare bool, not a dict), the walker
# in gradio_client.utils crashes on `if "const" in schema:` because it expects a
# mapping. The crash cascades into a misleading "localhost not accessible".
import gradio_client.utils as _gcu  # noqa: E402

_orig_get_type = _gcu.get_type
_orig_json_to_py = _gcu._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_get_type(schema)


def _safe_json_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_json_to_py(schema, defs)


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_to_py
# ----------------------------------------------------------------------------

from mavis_rag_engine import (  # noqa: E402
    CHAT_MODEL,
    MavisRagEngine,
    load_eval_results,
)

# UI knobs.
GRADIO_QUEUE_SIZE = int(os.environ.get("GRADIO_QUEUE_SIZE", "8"))
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

# Engine is heavy (SigLIP + mpnet + cross-encoder + indices). Build once, lazily,
# so `import app` stays cheap and the offline tests can exercise the renderers.
_ENGINE: MavisRagEngine | None = None


def get_engine() -> MavisRagEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MavisRagEngine()
    return _ENGINE


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _fmt_views(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _img_data_uri(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    ext = Path(path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


# ---------------------------------------------------------------------------
# Video card / grid (shared by the RAG evidence and the search tab)
# ---------------------------------------------------------------------------
def _video_card(hit, rank: int | None = None) -> str:
    rank = rank if rank is not None else hit.rank
    img = _img_data_uri(hit.thumb_path) or f"https://i.ytimg.com/vi/{hit.video_id}/hqdefault.jpg"
    date = (hit.published_at or "")[:10]
    return f"""
<a href="{html.escape(hit.url)}" target="_blank" rel="noopener" class="m-card">
  <div class="m-thumb"><img src="{img}" alt="" loading="lazy"/>
    <span class="m-rank">#{rank}</span>
  </div>
  <div class="m-body">
    <div class="m-title">{html.escape(hit.title)}</div>
    <div class="m-meta">
      <span class="m-channel">{html.escape(hit.channel_name)}</span>
      <span class="m-dot">·</span>
      <span class="m-views">👁 {_fmt_views(hit.views)}</span>
      {f'<span class="m-dot">·</span><span class="m-date">{html.escape(date)}</span>' if date else ''}
    </div>
  </div>
</a>
"""


def _videos_grid(hits, empty: str) -> str:
    if not hits:
        return f"<div class='m-empty'>{empty}</div>"
    cards = "\n".join(_video_card(h) for h in hits)
    return f"<div class='m-grid'>{cards}</div>"


# ---------------------------------------------------------------------------
# RAG answer + context renderers
# ---------------------------------------------------------------------------
import re  # noqa: E402

_CITE_RE = re.compile(r"\[(\d{1,2}|V)\]")


def _answer_html(out: dict) -> str:
    if not out:
        return (
            "<div class='m-empty'>Ask a question and click <b>Ask</b> to get a cited "
            "answer grounded in the corpus.</div>"
        )
    ans = out.get("answer")
    err = out.get("error")
    timing = (
        f"<div class='m-timing'>retrieval {out['t_retrieval'] * 1e3:.0f} ms"
        f" · generation {out['t_generation']:.1f} s"
        f" · <code>{html.escape(out.get('mode', ''))}{' +rerank' if out.get('rerank') else ''}</code></div>"
    )
    if ans:
        body = html.escape(ans)
        body = _CITE_RE.sub(lambda m: f"<span class='m-cite'>{m.group(0)}</span>", body)
        paras = "".join(f"<p>{p}</p>" for p in body.split("\n") if p.strip())
        warn = (
            f"<div class='m-warn'>⚠️ {html.escape(err)}</div>" if err else ""
        )
        return f"<div class='m-answer'>{paras}{warn}{timing}</div>"
    # no answer text — show the error/no-LLM notice but keep the timing
    msg = err or "No answer produced."
    return f"<div class='m-pred-error'>⚠️ {html.escape(msg)}</div>{timing}"


_FMT_BADGE = {
    "md": "#3b82f6", "pdf": "#ef4444", "csv": "#10b981",
    "json": "#f59e0b", "yaml": "#8b5cf6", "txt": "#64748b",
}


def _context_html(out: dict) -> str:
    if not out or not out.get("blocks"):
        return "<div class='m-empty'>The retrieved context blocks appear here.</div>"
    items = []
    for b in out["blocks"]:
        color = _FMT_BADGE.get(b.format, "#64748b")
        body = html.escape(b.body)
        items.append(
            f"""
<div class="m-block">
  <div class="m-block-head">
    <span class="m-block-n">[{b.rank}]</span>
    <span class="m-fmt" style="background:{color};">{html.escape(b.format)}</span>
    <span class="m-block-id">{html.escape(b.doc_id)} · {html.escape(b.section)}</span>
  </div>
  <div class="m-block-body">{body}</div>
</div>"""
        )
    return f"<div class='m-blocks'>{''.join(items)}</div>"


def rag_handler(question: str, rerank: bool):
    question = (question or "").strip()
    if not question:
        empty = "<div class='m-empty'>Type a question first.</div>"
        return empty, "", ""
    out = get_engine().answer(question, mode="hybrid", rerank=bool(rerank))
    evidence = _videos_grid(out.get("videos"), "No related videos found.")
    return _answer_html(out), _context_html(out), evidence


# ---------------------------------------------------------------------------
# Video search handler
# ---------------------------------------------------------------------------
def search_handler(text: str, image):
    text = (text or "").strip() or None
    if text is None and image is None:
        return "<div class='m-empty'>Enter a query or upload a thumbnail, then click <b>Search</b>.</div>"
    # text+image fuses every path; text-only / image-only pick the right subset.
    if text is not None and image is not None:
        use = ("mp", "xm", "bm", "ii", "it")
    elif text is not None:
        use = ("mp", "xm", "bm")
    else:
        use = ()  # image-only → engine adds ii/it automatically
    hits = get_engine().search_videos(text=text, image=image, k=12, use=use)
    return _videos_grid(hits, "No results.")


# ---------------------------------------------------------------------------
# Evaluation tab (static, read from cache/)
# ---------------------------------------------------------------------------
def _metrics_table(by_col: dict, row_order=None, col_order=None) -> str:
    """Render a dict {col: {row: value}} as an HTML table (rows × cols)."""
    cols = col_order or list(by_col.keys())
    rows = row_order or list({r for c in by_col.values() for r in c})
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for r in rows:
        cells = "".join(f"<td>{_fmt_num(by_col.get(c, {}).get(r))}</td>" for c in cols)
        body.append(f"<tr><th class='m-rowname'>{html.escape(str(r))}</th>{cells}</tr>")
    return (
        f"<table class='m-eval-table'><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _figure_html(path: str | None) -> str:
    uri = _img_data_uri(path)
    if not uri:
        return ""
    return f"<div class='m-fig'><img src='{uri}' alt='figure'/></div>"


def _eval_html(results: dict) -> str:
    sections = []

    # 1) video retrieval ablation
    vid = results.get("video")
    if vid:
        configs = ["mpnet titles", "+ cross-modal", "cross-modal only", "hybrid (full)"]
        configs = [c for c in configs if any(c in m for m in vid.values())] or None
        sections.append(
            "<div class='m-eval-card'>"
            "<h3>① Video retrieval — path ablation</h3>"
            "<p class='m-eval-sub'>Gold set (10 queries, 79 relevant). The mpnet-title "
            "path leads; cross-modal and BM25 add recall on harder queries.</p>"
            + _metrics_table(vid, row_order=configs)
            + _figure_html(results["figures"].get("video"))
            + "</div>"
        )

    # 2) document retrieval
    ret = results.get("retrieval")
    if ret and "table" in ret:
        configs = ["dense", "hybrid", "hybrid+rerank"]
        block = (
            "<div class='m-eval-card'>"
            "<h3>② Document retrieval — base vs hybrid vs +re-rank</h3>"
            "<p class='m-eval-sub'>27 gold questions, doc-level relevance. The cross-encoder "
            "re-rank (phase 2) lifts every metric.</p>"
            + _metrics_table(ret["table"], row_order=configs)
        )
        pf = ret.get("per_format_mrr")
        if pf:
            fmts = sorted({f for c in pf.values() for f in c})
            block += (
                "<p class='m-eval-sub' style='margin-top:14px;'>MRR by primary gold format "
                "(<code>n</code> = # questions):</p>"
                + _metrics_table(pf, row_order=fmts, col_order=["dense", "hybrid", "hybrid+rerank", "n"])
            )
        block += _figure_html(results["figures"].get("retrieval")) + "</div>"
        sections.append(block)

    # 3) answer quality
    gen = results.get("generation")
    if gen:
        configs = ["base (hybrid)", "improved (+rerank)"]
        sections.append(
            "<div class='m-eval-card'>"
            "<h3>③ Answer quality — five axes</h3>"
            "<p class='m-eval-sub'>31 questions (27 answerable + 4 unanswerable). token-F1 is a "
            "floor (punishes paraphrase); judge scores are 1–5; rates are 0–1.</p>"
            + _metrics_table(gen, row_order=configs)
            + _figure_html(results["figures"].get("generation"))
            + "</div>"
        )

    if not sections:
        return (
            "<div class='m-empty'>No evaluation results found in <code>cache/</code>. "
            "Run <code>01_multiformat_rag.ipynb</code> end-to-end to generate them.</div>"
        )
    return "<div class='m-eval'>" + "\n".join(sections) + "</div>"


# ---------------------------------------------------------------------------
# CSS — single sheet, scoped via the m- prefix.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root { color-scheme: dark; }
.gradio-container { max-width: 1380px !important; margin: 0 auto;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

.m-hero { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 35%, #ec4899 100%);
  border-radius: 14px; padding: 18px 22px; color: #fff;
  box-shadow: 0 8px 28px rgba(139,92,246,0.20); margin-bottom: 14px; }
.m-hero h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 800; letter-spacing: -0.4px; }
.m-hero p  { margin: 0; opacity: 0.93; font-size: 13px; line-height: 1.45; max-width: 920px; }

.m-empty { padding: 22px; text-align: center; opacity: 0.6; font-size: 13px;
  border: 1px dashed rgba(148,163,184,0.35); border-radius: 12px; }
.m-empty code, .m-eval-sub code, .m-timing code { background: rgba(148,163,184,0.16);
  padding: 1px 6px; border-radius: 4px; }

.m-section-title { margin: 16px 0 10px 0; font-size: 14px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.7; }

/* ---- video grid ---- */
.m-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 14px; }
.m-card { display: block; text-decoration: none; color: inherit;
  background: rgba(30,41,59,0.55); border: 1px solid rgba(148,163,184,0.12);
  border-radius: 12px; overflow: hidden;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.m-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.38);
  border-color: rgba(139,92,246,0.45); }
.m-thumb { position: relative; aspect-ratio: 16/9; background: #0f172a; }
.m-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.m-rank { position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,0.78); color: #fff;
  font-size: 11px; padding: 2px 7px; border-radius: 5px; font-weight: 700; letter-spacing: 0.4px; }
.m-body { padding: 10px 12px 12px 12px; }
.m-title { font-size: 13px; font-weight: 600; line-height: 1.32; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 34px; }
.m-meta { margin-top: 6px; font-size: 11.5px; opacity: 0.78; display: flex; flex-wrap: wrap;
  gap: 5px; align-items: center; }
.m-channel { font-weight: 600; } .m-dot { opacity: 0.4; }

/* ---- RAG answer ---- */
.m-answer { background: linear-gradient(180deg, rgba(30,41,59,0.65), rgba(15,23,42,0.65));
  border: 1px solid rgba(148,163,184,0.15); border-radius: 14px; padding: 16px 18px; }
.m-answer p { margin: 0 0 10px 0; font-size: 14.5px; line-height: 1.55; }
.m-answer p:last-of-type { margin-bottom: 0; }
.m-cite { display: inline-block; background: rgba(99,102,241,0.22); color: #c4b5fd;
  font-weight: 700; font-size: 11.5px; padding: 0 5px; border-radius: 5px; margin: 0 1px;
  vertical-align: baseline; }
.m-timing { margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(148,163,184,0.12);
  font-size: 11.5px; opacity: 0.6; }
.m-warn { margin: 10px 0 0 0; padding: 8px 10px; border-radius: 7px;
  background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.32);
  color: #fbbf24; font-size: 11.5px; line-height: 1.4; }
.m-pred-error { padding: 14px; border-radius: 12px; background: rgba(220,38,38,0.15);
  border: 1px solid rgba(220,38,38,0.4); color: #fca5a5; font-size: 13px; line-height: 1.5; }

/* ---- context blocks ---- */
.m-blocks { display: flex; flex-direction: column; gap: 10px; }
.m-block { background: rgba(15,23,42,0.4); border: 1px solid rgba(148,163,184,0.12);
  border-radius: 10px; overflow: hidden; }
.m-block-head { display: flex; align-items: center; gap: 8px; padding: 8px 10px;
  background: rgba(30,41,59,0.5); border-bottom: 1px solid rgba(148,163,184,0.1);
  font-size: 11.5px; }
.m-block-n { font-weight: 800; color: #c4b5fd; }
.m-fmt { color: #fff; font-size: 10px; font-weight: 700; text-transform: uppercase;
  padding: 1px 7px; border-radius: 999px; letter-spacing: 0.4px; }
.m-block-id { opacity: 0.7; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.m-block-body { padding: 9px 11px; font-size: 12px; line-height: 1.5; white-space: pre-wrap;
  max-height: 160px; overflow-y: auto; opacity: 0.9; }

/* ---- eval ---- */
.m-eval { display: flex; flex-direction: column; gap: 18px; }
.m-eval-card { background: rgba(30,41,59,0.4); border: 1px solid rgba(148,163,184,0.14);
  border-radius: 14px; padding: 16px 18px; }
.m-eval-card h3 { margin: 0 0 4px 0; font-size: 15px; font-weight: 800; }
.m-eval-sub { margin: 0 0 12px 0; font-size: 12px; opacity: 0.7; line-height: 1.45; }
.m-eval-table { width: 100%; border-collapse: collapse; font-size: 12.5px;
  font-variant-numeric: tabular-nums; }
.m-eval-table th, .m-eval-table td { padding: 6px 10px; text-align: right;
  border-bottom: 1px solid rgba(148,163,184,0.1); }
.m-eval-table thead th { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.3px;
  opacity: 0.75; border-bottom: 1px solid rgba(148,163,184,0.22); }
.m-eval-table .m-rowname { text-align: left; font-weight: 700;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
.m-eval-table tbody tr:hover { background: rgba(99,102,241,0.06); }
.m-fig { margin-top: 14px; border-radius: 10px; overflow: hidden;
  border: 1px solid rgba(148,163,184,0.12); background: #fff; }
.m-fig img { width: 100%; display: block; }

.gr-form { gap: 10px !important; }
"""

HERO = """
<div class="m-hero">
  <h1>MAVIS · Multimodal RAG assistant</h1>
  <p>Ask in natural language over a corpus centred on video thumbnails &amp; titles
  (+ docs in md/pdf/csv/json/yaml/txt). Get a cited answer, the context it used, and
  the real thumbnails of the most related videos as visual evidence.</p>
</div>
"""

EXAMPLE_QUESTIONS = [
    "¿Qué es 'el muro' en MAVIS y cómo se rompió?",
    "¿Por qué el buscador usa dos encoders distintos?",
    "¿Qué canal tiene más vídeos en el corpus completo de MAVIS?",
    "What does Cui et al. find about title–thumbnail coherence?",
]
EXAMPLE_SEARCHES = [
    "recetas de cocina fáciles",
    "agentes de inteligencia artificial en local",
    "docker tutorial from scratch",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="MAVIS · Multimodal RAG",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    ) as demo:
        gr.HTML(HERO)

        with gr.Tabs():
            # ---------------- TAB 1: Assistant (RAG) ----------------
            with gr.Tab("💬 Assistant (RAG)"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=0):
                        q_in = gr.Textbox(
                            label="Your question",
                            placeholder="Ask about the MAVIS project, the corpus, or the videos…",
                            lines=2,
                        )
                        with gr.Row():
                            rerank_in = gr.Checkbox(
                                value=True,
                                label="Cross-encoder re-rank (phase 2 — improved config)",
                                scale=3,
                            )
                            ask = gr.Button("Ask", variant="primary", size="lg", scale=2)
                        gr.Examples(EXAMPLE_QUESTIONS, inputs=q_in, label="Example questions")

                        gr.HTML("<div class='m-section-title'>🧠 Answer</div>")
                        answer_html = gr.HTML(
                            "<div class='m-empty'>Ask a question and click <b>Ask</b>.</div>"
                        )
                        gr.HTML("<div class='m-section-title'>🎬 Visual evidence — related corpus videos</div>")
                        evidence_html = gr.HTML("<div class='m-empty'>Thumbnails appear here.</div>")

                    with gr.Column(scale=2, min_width=0):
                        gr.HTML("<div class='m-section-title'>📑 Retrieved context (citable blocks)</div>")
                        context_html = gr.HTML(
                            "<div class='m-empty'>The numbered blocks the answer cites appear here.</div>"
                        )

                ask.click(rag_handler, [q_in, rerank_in], [answer_html, context_html, evidence_html])
                q_in.submit(rag_handler, [q_in, rerank_in], [answer_html, context_html, evidence_html])

            # ---------------- TAB 2: Video search ----------------
            with gr.Tab("🔎 Video search (the core)"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=0):
                        s_text = gr.Textbox(
                            label="Text query",
                            placeholder="e.g. agentes de IA en local — leave empty to search by image only",
                            lines=2,
                        )
                        s_img = gr.Image(
                            label="…or an image query (find lookalike thumbnails)",
                            type="pil",
                            height=200,
                            sources=["upload", "clipboard"],
                        )
                        s_btn = gr.Button("Search", variant="primary", size="lg")
                        gr.Examples(EXAMPLE_SEARCHES, inputs=s_text, label="Example text queries")
                    with gr.Column(scale=3, min_width=0):
                        gr.HTML("<div class='m-section-title'>Ranked corpus videos (RRF over title / cross-modal / BM25)</div>")
                        s_results = gr.HTML(
                            "<div class='m-empty'>Enter a query or upload a thumbnail, then click "
                            "<b>Search</b>.</div>"
                        )
                s_btn.click(search_handler, [s_text, s_img], s_results)
                s_text.submit(search_handler, [s_text, s_img], s_results)

            # ---------------- TAB 3: Evaluation ----------------
            with gr.Tab("📊 Evaluation"):
                gr.HTML(
                    "<div class='m-section-title'>Measured results (read from cache/, produced by the notebook)</div>"
                )
                gr.HTML(_eval_html(load_eval_results()))

        gr.HTML(
            f"<div style='opacity:.5;font-size:11px;margin-top:14px;text-align:center;'>"
            f"chat model: <code>{html.escape(CHAT_MODEL)}</code> · OpenAI-compatible · "
            f"set OPENAI_API_KEY / OPENAI_BASE_URL in .env</div>"
        )
    return demo


if __name__ == "__main__":
    # Warm the engine at launch so the first request isn't slow (and so model
    # download / cache regeneration happens before the UI accepts traffic).
    print("Building engine (first run downloads encoders + may regenerate cache/)…")
    get_engine()
    demo = build_ui()
    demo.queue(max_size=GRADIO_QUEUE_SIZE).launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        show_error=True,
        show_api=False,
    )

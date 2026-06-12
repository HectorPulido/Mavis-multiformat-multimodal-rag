"""MAVIS · Project 3 — RAG engine (extracted from 01_multiformat_rag.ipynb).

One heavy object, built once at startup. It reproduces the notebook cell-for-cell
so the platform's numbers match the analysis:

  * the **video index** (the core): SigLIP image + cross-modal text→image, mpnet
    titles, BM25 over titles, fused with RRF — queryable by text and/or image;
  * the **document layer**: generic ingestion of everything under ``corpus/`` by
    extension → chunks → mpnet + FAISS (IndexFlatIP) + BM25 → RRF → optional
    cross-encoder re-ranking;
  * **generation**: an OpenAI-compatible LLM that answers from the retrieved
    blocks (+ a ``[V]`` block of related corpus videos) with inline citations.

Caches in ``cache/`` regenerate from the local corpus on first run (and recover
from a stale/corrupt cache instead of crashing). Heavy dependencies
(``torch``/``transformers``/``faiss``/``openai``) are imported lazily inside the
methods that need them, so this module imports cleanly with only numpy/pandas —
the Gradio app and the offline tests rely on that.

``app.py`` imports :class:`MavisRagEngine` and renders its outputs; nothing here
touches Gradio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# macOS: torch and faiss each bundle their own libomp (OpenMP) runtime; loading
# both aborts with "OMP: Error #15 ... libomp.dylib already initialized". Permit
# the duplicate BEFORE numpy/torch/faiss load (override via the env var). app.py
# sets the same guard; this covers running the engine standalone.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & configuration (mirror notebook §1)
# ---------------------------------------------------------------------------
def _find_root() -> Path:
    """Repo root = nearest ancestor of this file holding corpus/videos/videos.csv.

    Lets the platform run from any cwd (root, a subdir, Docker) while keeping the
    notebook's self-contained path assumptions intact.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "corpus" / "videos" / "videos.csv").exists():
            return cand
    return here


ROOT = _find_root()
CORPUS = ROOT / "corpus"
VID_DIR = CORPUS / "videos"
CACHE = ROOT / "cache"
FIGS = ROOT / "figures"

SIGLIP_ID = "google/siglip2-large-patch16-256"  # images + cross-modal (the core)
MPNET_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # text↔text
CE_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # phase-2 re-ranker

IMG_EXT = {".jpg", ".jpeg", ".png"}
DOC_EXT = {".md", ".pdf", ".csv", ".json", ".yml", ".yaml", ".txt"}

CHUNK_CHARS, CHUNK_OVERLAP = 1800, 200
RRF_K, W_LEX = 60, 1.0
TOPK_CONTEXT, RERANK_POOL, TOPK_VIDEOS = 6, 30, 3


def _load_env(path: Path | None = None) -> None:
    """Minimal .env loader (shell env wins over the file), as in the notebook."""
    path = path or ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-5-mini")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", CHAT_MODEL)


# ---------------------------------------------------------------------------
# Encoders — frozen, lazy singletons (mirror notebook §3). Heavy imports
# (torch/transformers) happen on first use, never at module import.
# ---------------------------------------------------------------------------
_SIG = {"m": None, "p": None, "dev": None}
_MP = {"m": None, "t": None, "dev": None}
_CE = {"m": None, "t": None, "dev": None}


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pool(x):
    import torch

    return x if torch.is_tensor(x) else x.pooler_output


def load_siglip():
    if _SIG["m"] is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        t0 = time.time()
        p = AutoProcessor.from_pretrained(SIGLIP_ID)
        m = AutoModel.from_pretrained(SIGLIP_ID).to(get_device()).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        _SIG.update(m=m, p=p, dev=get_device())
        print(f"SigLIP loaded on {_SIG['dev']} ({time.time() - t0:.1f}s)")
    return _SIG["m"], _SIG["p"], _SIG["dev"]


def embed_text_siglip(texts, batch=64):
    import torch
    import torch.nn.functional as F

    m, p, dev = load_siglip()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            tin = p(
                text=[str(t) for t in texts[s : s + batch]],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(dev)
            out.append(
                F.normalize(_pool(m.get_text_features(**tin)), dim=-1)
                .cpu()
                .float()
                .numpy()
            )
    return np.vstack(out).astype("float32")


def embed_images_siglip(paths, batch=32, verbose=False):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    m, p, dev = load_siglip()
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            imgs = []
            for pp in paths[s : s + batch]:
                try:
                    imgs.append(Image.open(pp).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (256, 256)))
            pin = p(images=imgs, return_tensors="pt").to(dev)
            out.append(
                F.normalize(_pool(m.get_image_features(**pin)), dim=-1)
                .cpu()
                .float()
                .numpy()
            )
            if verbose:
                print(f"  siglip-image {min(s + batch, len(paths))}/{len(paths)}", flush=True)
    return np.vstack(out).astype("float32")


def embed_image_siglip(path_or_img):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    m, p, dev = load_siglip()
    with torch.no_grad():
        pin = p(images=[img.convert("RGB")], return_tensors="pt").to(dev)
        return (
            F.normalize(_pool(m.get_image_features(**pin)), dim=-1)
            .cpu()
            .float()
            .numpy()[0]
        )


def load_mpnet():
    if _MP["m"] is None:
        from transformers import AutoModel, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MPNET_ID)
        m = AutoModel.from_pretrained(MPNET_ID).to(get_device()).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        _MP.update(m=m, t=tok, dev=get_device())
        print(f"mpnet loaded on {_MP['dev']} ({time.time() - t0:.1f}s)")
    return _MP["m"], _MP["t"], _MP["dev"]


def embed_text_mpnet(batch_texts, batch=64, max_length=384, verbose=False):
    """Masked mean-pooling + L2; the recipe Project 2 validated."""
    import torch
    import torch.nn.functional as F

    m, tok, dev = load_mpnet()
    out = []
    with torch.no_grad():
        for s in range(0, len(batch_texts), batch):
            enc = tok(
                [str(t) for t in batch_texts[s : s + batch]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)
            h = m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(e, dim=-1).cpu().float().numpy())
            if verbose:
                print(f"  mpnet {min(s + batch, len(batch_texts))}/{len(batch_texts)}", flush=True)
    return np.vstack(out).astype("float32")


def load_ce():
    if _CE["m"] is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(CE_ID)
        m = AutoModelForSequenceClassification.from_pretrained(CE_ID).to(get_device()).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        _CE.update(m=m, t=tok, dev=get_device())
        print(f"cross-encoder loaded on {_CE['dev']} ({time.time() - t0:.1f}s)")
    return _CE["m"], _CE["t"], _CE["dev"]


def ce_scores(query, cand_texts, batch=16):
    import torch

    m, tok, dev = load_ce()
    out = []
    with torch.no_grad():
        for s in range(0, len(cand_texts), batch):
            cb = cand_texts[s : s + batch]
            enc = tok(
                [query] * len(cb),
                cb,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(dev)
            out.append(m(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Lexical layer + fusion (mirror notebook §4 / §8 / §9)
# ---------------------------------------------------------------------------
def _lex_tokens(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def build_bm25(texts_):
    """Okapi BM25 index (k1=1.5, b=0.75) over a list of strings -> scoring closure."""
    docs = [_lex_tokens(t) for t in texts_]
    dl = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(dl.mean()) or 1.0
    inv, dfreq = defaultdict(list), Counter()
    for i, d in enumerate(docs):
        for tok_, f in Counter(d).items():
            inv[tok_].append((i, f))
            dfreq[tok_] += 1
    idf = {t: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for t, c in dfreq.items()}
    K1, B = 1.5, 0.75

    def scores(query):
        sc = np.zeros(len(docs), dtype=np.float32)
        for tok_ in set(_lex_tokens(query)):
            w = idf.get(tok_)
            if w is None:
                continue
            for i, f in inv[tok_]:
                sc[i] += w * f * (K1 + 1) / (f + K1 * (1 - B + B * dl[i] / avgdl))
        return sc

    return scores, len(dfreq)


def _rrf_order(paths, n, weights=None):
    """Reciprocal-rank fusion of several score vectors into one ranking."""
    weights = weights or {}
    fused = np.zeros(n)
    for name, sc in paths.items():
        r = np.empty(n, dtype=int)
        r[np.argsort(-sc)] = np.arange(n)
        fused += weights.get(name, 1.0) / (RRF_K + r + 1)
    return np.argsort(-fused)


# ---------------------------------------------------------------------------
# Document loaders (mirror notebook §5). pypdf / yaml stay lazy (per-loader).
# ---------------------------------------------------------------------------
def _read(p):
    return Path(p).read_text(encoding="utf-8", errors="ignore")


def load_md(path):
    units, sec, buf = [], "intro", []
    for line in _read(path).splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if "\n".join(buf).strip():
                units.append((sec, "\n".join(buf).strip()))
            sec, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        units.append((sec, "\n".join(buf).strip()))
    return units


def load_pdf(path):
    from pypdf import PdfReader

    units = []
    for i, page in enumerate(PdfReader(path).pages):
        t = re.sub(r"[ \t]+", " ", (page.extract_text() or "")).strip()
        if len(t) > 80:
            units.append((f"page {i + 1}", t))
    return units


def load_csv(path):
    df = pd.read_csv(path)

    def row_text(tup):
        return "; ".join(f"{c}: {v}" for c, v in zip(df.columns, tup) if pd.notna(v))

    rows = [row_text(t) for t in df.itertuples(index=False, name=None)]
    units = [
        (
            "summary",
            f"CSV file {Path(path).name}: {len(df)} rows. Columns: "
            f"{', '.join(df.columns)}. First rows: " + " | ".join(rows[:3]),
        )
    ]
    G = 10
    for s in range(0, len(rows), G):
        units.append((f"rows {s + 1}-{min(s + G, len(rows))}", "\n".join(rows[s : s + G])))
    return units


def load_json(path):
    data = json.loads(_read(path))

    def brief(v):
        if isinstance(v, dict):
            return "{" + ", ".join(list(v)[:8]) + "}"
        if isinstance(v, list):
            return f"[{len(v)} items]"
        return str(v)

    if isinstance(data, list):
        entries, head = data, f"list of {len(data)} entries"
    elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
        key = next(k for k, v in data.items() if isinstance(v, list))
        entries, head = (
            data[key],
            f"object with keys {', '.join(data)}; '{key}' holds {len(data[key])} entries",
        )
    else:
        entries, head = [data], "single object"
    units = [("summary", f"JSON file {Path(path).name}: {head}.")]
    lines = [
        "; ".join(f"{k}: {brief(v)}" for k, v in e.items())
        if isinstance(e, dict)
        else str(e)
        for e in entries
    ]
    G = 5
    for s in range(0, len(lines), G):
        units.append((f"entries {s + 1}-{min(s + G, len(lines))}", "\n".join(lines[s : s + G])))
    return units


def load_yaml(path):
    import yaml

    raw = _read(path)
    lines = []

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(node, list):
            lines.append(f"{prefix[:-1]}: " + "; ".join(str(x) for x in node[:12]))
        else:
            lines.append(f"{prefix[:-1]}: {node}")

    walk(yaml.safe_load(raw))
    return [
        ("structure", f"YAML file {Path(path).name}, flattened keys:\n" + "\n".join(lines)),
        ("raw", raw),
    ]


def load_txt(path):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", _read(path)) if b.strip()]
    return [(f"block {i + 1}", b) for i, b in enumerate(blocks)]


LOADERS = {
    ".md": load_md,
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".json": load_json,
    ".yml": load_yaml,
    ".yaml": load_yaml,
    ".txt": load_txt,
}


# ---------------------------------------------------------------------------
# Chunking (mirror notebook §6)
# ---------------------------------------------------------------------------
def _split_long(text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    out, buf = [], ""
    for p in re.split(r"\n\s*\n", text):
        if len(p) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            s = 0
            while s < len(p):
                out.append(p[s : s + max_chars])
                s += max_chars - overlap
        elif len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            out.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if overlap else p
    if buf:
        out.append(buf)
    return [x for x in out if x.strip()]


def make_chunks(doc_id, fmt, units):
    out, secs, parts, cur = [], [], [], 0

    def flush():
        nonlocal secs, parts, cur
        if parts:
            sec = secs[0] + (" (+)" if len(set(secs)) > 1 else "")
            out.append({"doc_id": doc_id, "format": fmt, "section": sec, "body": "\n\n".join(parts)})
        secs, parts, cur = [], [], 0

    for sec, text in units:
        for piece in _split_long(text) if len(text) > CHUNK_CHARS else [text]:
            if cur + len(piece) > CHUNK_CHARS and cur > 0:
                flush()
            secs.append(sec)
            parts.append(piece)
            cur += len(piece) + 2
    flush()
    return out


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SearchHit:
    rank: int
    video_id: str
    title: str
    channel_name: str
    views: int
    published_at: str
    url: str
    thumb_path: str


@dataclass
class ContextBlock:
    rank: int
    doc_id: str
    format: str
    section: str
    body: str


# ---------------------------------------------------------------------------
# Prompt + context assembly (mirror notebook §10)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the assistant for MAVIS, a master's thesis project about predicting and "
    "searching YouTube videos. You answer questions using ONLY the numbered context "
    "blocks provided.\n"
    "Rules:\n"
    "1. Use only information present in the context blocks. Never use outside knowledge.\n"
    "2. After each claim, cite the supporting block(s) inline like [1] or [2][3]. The "
    "block labeled [V] lists corpus videos related to the question; cite it as [V] when you use it.\n"
    "3. Answer in the same language as the question.\n"
    "4. If the context does not contain the answer, say explicitly (in the question's "
    "language) that the MAVIS documentation does not cover it. Do not guess.\n"
    "5. Be concise: 2-5 sentences."
)


def build_context(blocks: list[ContextBlock]) -> str:
    return "\n\n".join(f"[{b.rank}] ({b.doc_id} · {b.section})\n{b.body}" for b in blocks)


def build_video_block(vids: list[SearchHit]) -> str:
    lines = [f'- "{v.title}" | channel: {v.channel_name} | {int(v.views):,} views' for v in vids]
    return "[V] Videos from the local corpus most related to the question:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation results loader (read-only; needs no models) — for the eval tab
# ---------------------------------------------------------------------------
EVAL_FILES = {
    "video": CACHE / "video_eval.json",
    "retrieval": CACHE / "retrieval_eval.json",
    "generation": CACHE / "generation_eval.json",
}
EVAL_FIGURES = {
    "video": FIGS / "video_retrieval_metrics.png",
    "retrieval": FIGS / "retrieval_metrics.png",
    "generation": FIGS / "generation_metrics.png",
}


def load_eval_results() -> dict:
    """Parse the three eval JSONs produced by the notebook (cache/). Missing or
    unreadable files yield ``None`` for that section instead of raising, so the
    eval tab degrades gracefully when the notebook hasn't been run."""
    out = {"figures": {k: (str(p) if p.exists() else None) for k, p in EVAL_FIGURES.items()}}
    for name, path in EVAL_FILES.items():
        try:
            out[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (json.JSONDecodeError, OSError):
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class MavisRagEngine:
    """Builds (and caches) the video index + document layer, then serves search,
    RAG answering, and evaluation read-back. Construct once; reuse for every
    request. Encoders load lazily on the first call that needs each one."""

    def __init__(self, verbose: bool = True, warmup: bool = True):
        self.verbose = verbose
        self._log(f"root: {ROOT}")
        CACHE.mkdir(exist_ok=True)
        FIGS.mkdir(exist_ok=True)
        self._build_video_index()
        self._build_document_layer()
        self._probe_llm()
        if warmup:
            self._warmup_encoders()

    def _log(self, *a):
        if self.verbose:
            print("[engine]", *a, flush=True)

    def _warmup_encoders(self):
        """Force-load every encoder and run one tiny inference of each, BEFORE the
        server opens. Cache HITs skip encoding during the build (a hit on the video
        index means SigLIP was never touched), so without this the first text search
        would load SigLIP (~2 GB) and the first re-ranked answer would load the
        cross-encoder — each a multi-second stall that looks like a hung UI. We pay
        it up front so the server only accepts requests once every encoder is hot."""
        from PIL import Image

        t0 = time.time()
        self._log("warming up encoders (first run downloads weights; may take a few minutes)…")
        try:
            embed_text_mpnet(["warmup"])
            embed_text_siglip(["warmup"])
            embed_image_siglip(Image.new("RGB", (256, 256)))
            ce_scores("warmup", ["warmup passage"])
        except Exception as e:  # a warmup probe must never abort startup
            self._log(f"warmup probe failed (non-fatal): {e!r}")
        self._log(f"encoders ready in {time.time() - t0:.1f}s — server can accept requests")

    # -- video index (notebook §4) -------------------------------------------
    def _build_video_index(self):
        import os as _os

        videos = pd.read_csv(VID_DIR / "videos.csv")
        videos["thumb_path"] = videos["thumbnail"].map(lambda p: str(VID_DIR / p))
        videos = videos[videos["thumb_path"].map(_os.path.exists)].reset_index(drop=True)
        self.videos = videos
        self.N_VID = len(videos)
        self.vid_titles = videos["title"].astype(str).tolist()
        self._log(f"video corpus: {self.N_VID} videos | {videos.channel_handle.nunique()} channels")

        vid_cache = CACHE / "video_index.npz"
        vid_key = hashlib.sha1(
            "\x1e".join(videos["video_id"] + "|" + videos["title"].astype(str)).encode()
        ).hexdigest()
        loaded = False
        if vid_cache.exists():
            try:
                z = np.load(vid_cache, allow_pickle=True)
                if str(z["key"]) == vid_key:
                    self.VID_THUMB, self.VID_TSIG, self.VID_TMP = z["thumb"], z["tsig"], z["tmp"]
                    loaded = True
                    self._log("video-index cache HIT")
                else:
                    self._log("video-index cache STALE (corpus changed) -> re-encoding")
            except Exception as e:
                self._log(f"video-index cache UNREADABLE ({e!r}) -> re-encoding")
        if not loaded:
            self._log("video-index cache MISS -> encoding from corpus/videos/ (local files only)")
            self.VID_THUMB = embed_images_siglip(videos["thumb_path"].tolist(), verbose=self.verbose)
            self.VID_TSIG = embed_text_siglip(self.vid_titles)
            self.VID_TMP = embed_text_mpnet(self.vid_titles)
            np.savez_compressed(
                vid_cache,
                key=np.array(vid_key),
                thumb=self.VID_THUMB,
                tsig=self.VID_TSIG,
                tmp=self.VID_TMP,
            )
            self._log(f"saved {vid_cache.name}")

        self.bm25_vid_scores, _ = build_bm25(self.vid_titles)

    # -- document layer (notebook §5–§8) -------------------------------------
    def _build_document_layer(self):
        inventory = defaultdict(list)
        for p in sorted(CORPUS.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT | DOC_EXT:
                inventory[p.suffix.lower()].append(p)

        documents = []
        for ext, files in sorted(inventory.items()):
            if ext not in LOADERS:
                continue  # images are the video index's job (§4)
            for p in files:
                doc_id = str(p.relative_to(CORPUS))
                fmt = {"yml": "yaml", "jpeg": "jpg"}.get(ext.lstrip("."), ext.lstrip("."))
                documents.append((doc_id, fmt, LOADERS[ext](p)))

        chunks = []
        for doc_id, fmt, units in documents:
            chunks.extend(make_chunks(doc_id, fmt, units))
        chunks_df = pd.DataFrame(chunks)
        chunks_df["text"] = (
            "[" + chunks_df["doc_id"] + " · " + chunks_df["section"] + "]\n" + chunks_df["body"]
        )
        self.chunks_df = chunks_df
        self.texts = chunks_df["text"].tolist()
        self.doc_of_chunk = chunks_df["doc_id"].to_numpy()
        self.N_CHUNKS = len(chunks_df)
        self.n_documents = len(documents)
        self.formats = sorted({f for _, f, _ in documents})
        self._log(f"documents: {self.n_documents} | chunks: {self.N_CHUNKS} | formats: {self.formats}")

        # chunk embeddings (mpnet) + FAISS, keyed by corpus hash
        emb_cache = CACHE / "chunk_emb_mpnet.npz"
        corpus_key = hashlib.sha1("\x1e".join(self.texts).encode()).hexdigest()
        EMB = None
        if emb_cache.exists():
            try:
                z = np.load(emb_cache, allow_pickle=True)
                if str(z["key"]) == corpus_key and str(z["model"]) == MPNET_ID:
                    EMB = z["emb"].astype("float32")
                    self._log(f"chunk-embedding cache HIT: {EMB.shape}")
                else:
                    self._log("chunk-embedding cache STALE -> re-encoding")
            except Exception as e:
                self._log(f"chunk-embedding cache UNREADABLE ({e!r}) -> re-encoding")
        if EMB is None:
            self._log("chunk-embedding cache MISS -> encoding chunks")
            EMB = embed_text_mpnet(self.texts, verbose=self.verbose)
            np.savez_compressed(
                emb_cache, emb=EMB, key=np.array(corpus_key), model=np.array(MPNET_ID)
            )
        self.EMB = EMB

        import faiss

        self.INDEX = faiss.IndexFlatIP(EMB.shape[1])
        self.INDEX.add(EMB)
        self._log(f"FAISS IndexFlatIP: {self.INDEX.ntotal} vectors x {EMB.shape[1]}d")

        self.bm25_scores, _ = build_bm25(self.texts)

    # -- LLM probe (notebook §10) --------------------------------------------
    def _probe_llm(self):
        self._param_style: dict[str, str] = {}
        self._client = None
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or "not-set", timeout=120.0, max_retries=2
            )
            self._chat(CHAT_MODEL, "You are a ping service.", "Reply with the single word: pong", max_tokens=4)
            self.llm_ok, self.llm_msg = True, "ok"
        except Exception as e:
            self.llm_ok, self.llm_msg = False, f"{type(e).__name__}: {str(e)[:160]}"
        self._log(f"LLM available: {self.llm_ok} ({self.llm_msg})")

    def _chat(self, model, system, user, max_tokens=600):
        """OpenAI-compatible chat with reasoning/classic param sniffing (notebook §10)."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or "not-set", timeout=120.0, max_retries=2
            )
        import openai

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _call(style):
            if style == "classic":
                return self._client.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=max_tokens
                )
            return self._client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=max(4 * max_tokens, 2000)
            )

        style = self._param_style.get(model, "classic")
        try:
            r = _call(style)
        except openai.BadRequestError:
            style = "reasoning" if style == "classic" else "classic"
            r = _call(style)
        self._param_style[model] = style
        return (r.choices[0].message.content or "").strip()

    # -- video search (notebook §4) ------------------------------------------
    def _video_order(self, text=None, image=None, use=("mp", "xm", "bm"), w_lex=W_LEX):
        paths, weights = {}, {}
        if text is not None:
            if "mp" in use:
                paths["mp"] = self.VID_TMP @ embed_text_mpnet([text])[0]
            if "xm" in use:
                paths["xm"] = self.VID_THUMB @ embed_text_siglip([text])[0]
            if "bm" in use:
                paths["bm"] = self.bm25_vid_scores(text)
                weights["bm"] = w_lex
        if image is not None:
            q = embed_image_siglip(image)
            paths["ii"] = self.VID_THUMB @ q
            paths["it"] = self.VID_TSIG @ q
        if not paths:
            raise ValueError("provide text and/or image")
        return _rrf_order(paths, self.N_VID, weights)

    def search_videos(self, text=None, image=None, k=8, use=("mp", "xm", "bm")) -> list[SearchHit]:
        order = self._video_order(text, image, use)
        hits = []
        for rank, i in enumerate(order[:k], 1):
            r = self.videos.iloc[int(i)]
            hits.append(
                SearchHit(
                    rank=rank,
                    video_id=str(r["video_id"]),
                    title=str(r["title"]),
                    channel_name=str(r["channel_name"]),
                    views=int(r["views"]),
                    published_at=str(r.get("published_at", "")),
                    url=str(r["url"]),
                    thumb_path=str(r["thumb_path"]),
                )
            )
        return hits

    # -- document retrieval (notebook §7–§9, §11) ----------------------------
    def _dense_scores(self, q_emb):
        Ds, Is = self.INDEX.search(q_emb[None].astype("float32"), self.N_CHUNKS)
        s = np.empty(self.N_CHUNKS, dtype="float32")
        s[Is[0]] = Ds[0]
        return s

    def _ce_rerank(self, query, idxs):
        idxs = np.asarray(idxs)
        return idxs[np.argsort(-ce_scores(query, [self.texts[i] for i in idxs]))]

    def retrieve(self, query, k=TOPK_CONTEXT, mode="hybrid", rerank=False, pool=RERANK_POOL):
        paths, weights = {}, {}
        if mode in ("dense", "hybrid"):
            paths["dense"] = self._dense_scores(embed_text_mpnet([query])[0])
        if mode in ("lexical", "hybrid"):
            paths["bm25"] = self.bm25_scores(query)
            weights["bm25"] = W_LEX
        order = _rrf_order(paths, self.N_CHUNKS, weights)
        if rerank:
            order = np.concatenate([self._ce_rerank(query, order[:pool]), order[pool:]])
        blocks = []
        for rank, i in enumerate(order[:k], 1):
            r = self.chunks_df.iloc[int(i)]
            blocks.append(
                ContextBlock(
                    rank=rank,
                    doc_id=str(r["doc_id"]),
                    format=str(r["format"]),
                    section=str(r["section"]),
                    body=str(r["body"]),
                )
            )
        return blocks, order

    # -- generation (notebook §10) -------------------------------------------
    def answer(self, question, mode="hybrid", rerank=True, k=TOPK_CONTEXT, n_videos=TOPK_VIDEOS):
        """Full RAG turn. Returns a dict with the answer, the cited context
        blocks, the retrieved videos (visual evidence), the raw context, and
        per-stage timings. If no LLM is configured, ``answer`` is None and the
        retrieval results are still returned so the UI can show evidence."""
        t0 = time.perf_counter()
        blocks, _ = self.retrieve(question, k=k, mode=mode, rerank=rerank)
        vids = self.search_videos(text=question, k=n_videos)
        t_ret = time.perf_counter() - t0
        ctx = build_context(blocks) + "\n\n" + build_video_block(vids)

        out = {
            "question": question,
            "blocks": blocks,
            "videos": vids,
            "context": ctx,
            "t_retrieval": t_ret,
            "t_generation": 0.0,
            "answer": None,
            "error": None,
            "mode": mode,
            "rerank": rerank,
        }
        if not self.llm_ok:
            out["error"] = (
                f"No LLM configured ({self.llm_msg}). Set OPENAI_API_KEY (and optionally "
                "OPENAI_BASE_URL / RAG_CHAT_MODEL) in .env and restart. Retrieval evidence "
                "is shown below regardless."
            )
            return out
        t1 = time.perf_counter()
        try:
            out["answer"] = self._chat(
                CHAT_MODEL, SYSTEM_PROMPT, "Context:\n\n" + ctx + "\n\nQuestion: " + question
            )
        except Exception as e:
            out["error"] = f"Generation failed: {type(e).__name__}: {str(e)[:200]}"
        out["t_generation"] = time.perf_counter() - t1
        return out

    # -- evaluation read-back -------------------------------------------------
    @staticmethod
    def eval_results() -> dict:
        return load_eval_results()


# ---------------------------------------------------------------------------
# Smoke test (does build the heavy engine; needs the full deps + corpus)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    eng = MavisRagEngine()
    print("\nvideo search demo:")
    for h in eng.search_videos(text="agentes de inteligencia artificial en local", k=3):
        print(f"  #{h.rank} {h.title[:60]!r} · {h.channel_name} · {h.views:,} views")
    print("\nRAG demo:")
    out = eng.answer("¿Qué es 'el muro' en MAVIS y cómo se rompió?")
    print(out["answer"] or out["error"])
    print(f"[retrieval {out['t_retrieval']*1e3:.0f} ms | generation {out['t_generation']:.1f} s]")

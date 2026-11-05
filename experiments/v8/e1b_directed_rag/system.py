"""The E1b RAG assistant: the E1 pipeline copied unchanged (E1b PROTOCOL §2), with
two additions the E1b protocol requires: the incident names the target outcome
``correct`` (§3), and :func:`retrieval_truth` reports whether an accepted answer is in
the retrieved chunks (§4). Data paths point at this folder.

E1 description: chunk → embed → retrieve → prompt → generate, traced with the Glassbox
Recorder.

Backends:
* ``ToyEmbedder`` / ``ToyGenerator`` (tests, no model) and
* ``STEmbedder`` (sentence-transformers) / ``HFGenerator`` (transformers), loaded lazily.

Sampling is done in blocks: the generations for (configuration, question) are drawn in
blocks of ``BLOCK`` from one seed per block, derived from the stage's master seed, the
canonical configuration, the question and the block number. Run ``i`` takes sample
``i``. Different configurations (including ones differing only in the inert
``log_level``) therefore get independent samples (PROTOCOL.md Addendum 1).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from glassbox.v7.intervene import Factor
from glassbox.v7.investigate import Incident
from glassbox.v7.trace import Recorder, canonical_json, read_trace

HERE = Path(__file__).resolve().parent
TOP_K = 3
BLOCK = 50
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.7
TOP_P = 1.0

TEMPLATES = {
    "T1": "Use the context to answer the question.",
    "T2": "Use the context to answer the question. Answer concisely in one sentence.",
}
FACTORS = (
    Factor("index", "fresh", "stale"),
    Factor("chunk_words", 120, 40),
    Factor("template", "T1", "T2"),
    Factor("log_level", "info", "debug"),
)
BASE: Dict[str, Any] = {f.name: f.base for f in FACTORS}
CANDIDATE: Dict[str, Any] = {f.name: f.candidate for f in FACTORS}
_PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")


# --- data ------------------------------------------------------------------------------


def load_corpus() -> List[Dict[str, str]]:
    """The 24 documents, each with ``v1`` (old) and ``v2`` (current) text."""
    data: List[Dict[str, str]] = json.loads((HERE / "corpus.json").read_text("utf-8"))
    return data


def load_questions(split: str) -> List[Dict[str, Any]]:
    """``dev`` (8) or ``test`` (24) questions with accepted and stale answers."""
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    data: List[Dict[str, Any]] = json.loads(
        (HERE / f"questions_{split}.json").read_text("utf-8")
    )
    return data


# --- outcome ---------------------------------------------------------------------------


def tokens(text: str) -> List[str]:
    """Lowercase, punctuation → space, split on whitespace (PROTOCOL §2 Outcome)."""
    return _PUNCT.sub(" ", text.lower()).split()


def matches(output: str, accepted: Sequence[str]) -> bool:
    """True if any accepted answer appears in ``output`` as a contiguous token run."""
    out = tokens(output)
    for answer in accepted:
        want = tokens(answer)
        n = len(want)
        if n and any(out[i : i + n] == want for i in range(len(out) - n + 1)):
            return True
    return False


# --- retrieval -------------------------------------------------------------------------


def chunk(text: str, n_words: int) -> List[str]:
    """Consecutive, non-overlapping windows of ``n_words`` words."""
    words = text.split()
    return [" ".join(words[i : i + n_words]) for i in range(0, len(words), n_words)]


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Generator(Protocol):
    name: str

    def generate_block(
        self, messages: List[Dict[str, str]], n: int, seed: int
    ) -> List[str]: ...


class ToyEmbedder:
    """Hashed bag of words, L2-normalised (tests only; not a language model)."""

    name = "toy-hash-embedder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256))
        for row, text in enumerate(texts):
            for tok in tokens(text):
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                out[row, h % 256] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class ToyGenerator:
    """Answers with the context sentence that overlaps the question most, 85% of the
    time, otherwise says it is unsure (tests only; not a language model)."""

    name = "toy"

    def generate_block(
        self, messages: List[Dict[str, str]], n: int, seed: int
    ) -> List[str]:
        user = messages[-1]["content"]
        context, _, question = user.partition("\n\nQuestion: ")
        qtok = set(tokens(question))
        sentences = [s for s in re.split(r"(?<=\.)\s+", context) if s.strip()]
        best = max(sentences, key=lambda s: len(qtok & set(tokens(s))), default="")
        rng = random.Random(seed)
        return [best if rng.random() < 0.85 else "I am not sure." for _ in range(n)]


class SpyGenerator:
    """Wraps a generator and records every message list it receives (tests)."""

    def __init__(self, inner: Generator, calls: List[Any]) -> None:
        self.inner, self.calls, self.name = inner, calls, inner.name

    def generate_block(
        self, messages: List[Dict[str, str]], n: int, seed: int
    ) -> List[str]:
        self.calls.append(messages)
        return self.inner.generate_block(messages, n, seed)


class STEmbedder:  # pragma: no cover - needs the model; exercised on Colab
    """sentence-transformers embedder at a pinned revision."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"{model_id}@{revision[:12]}"
        self.model = SentenceTransformer(model_id, revision=revision, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float64)


class HFGenerator:  # pragma: no cover - needs the model; exercised on Colab
    """Chat model from transformers at a pinned revision; sampled generation."""

    def __init__(
        self, model_id: str, revision: str, device: str, precision: str
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"{model_id}@{revision[:12]} ({precision})"
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs: Dict[str, Any] = {"revision": revision}
        if precision == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            kwargs["device_map"] = device
        else:
            kwargs["dtype"] = getattr(torch, precision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if precision != "4bit":
            self.model.to(device)
        self.model.eval()
        self.device = device

    def generate_block(
        self, messages: List[Dict[str, str]], n: int, seed: int
    ) -> List[str]:
        torch = self.torch
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=n,
                pad_token_id=self.tok.eos_token_id,
            )
        return [
            self.tok.decode(row[ids.shape[1] :], skip_special_tokens=True)
            for row in out
        ]


# --- the system ------------------------------------------------------------------------


def build_messages(
    template: str, contexts: Sequence[str], question: str
) -> List[Dict[str, str]]:
    """System instruction from the template; user turn with context and question."""
    context = "\n\n".join(contexts)
    return [
        {"role": "system", "content": TEMPLATES[template]},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class System:
    """The assistant for one stage (one master seed)."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        master: str,
        corpus: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.embedder, self.generator, self.master = embedder, generator, master
        self.corpus = corpus if corpus is not None else load_corpus()
        self._index: Dict[Any, Any] = {}
        self._blocks: Dict[Any, List[str]] = {}
        self._hits: Dict[Any, List[Dict[str, Any]]] = {}

    def _index_for(self, config: Dict[str, Any]) -> Any:
        version = "v2" if config["index"] == "fresh" else "v1"
        key = (version, int(config["chunk_words"]))
        if key not in self._index:
            chunks = [
                {"doc": d["id"], "i": i, "text": t}
                for d in self.corpus
                for i, t in enumerate(chunk(d[version], key[1]))
            ]
            vecs = self.embedder.encode([c["text"] for c in chunks])
            self._index[key] = (chunks, vecs)
        return self._index[key]

    def retrieve(self, config: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Top-k chunks by cosine similarity; ties keep corpus order (deterministic).

        Retrieval is deterministic, so the result is cached per (index, question):
        the question is embedded once, not once per sampled run (Addendum 2).
        """
        version = "v2" if config["index"] == "fresh" else "v1"
        key = (version, int(config["chunk_words"]), question)
        if key not in self._hits:
            chunks, vecs = self._index_for(config)
            q = self.embedder.encode([question])[0]
            scores = vecs @ q
            order = sorted(range(len(chunks)), key=lambda j: (-float(scores[j]), j))
            self._hits[key] = [
                {**chunks[j], "score": float(scores[j])} for j in order[:TOP_K]
            ]
        return [dict(h) for h in self._hits[key]]

    def block_seed(self, config: Dict[str, Any], qid: str, block: int) -> int:
        """Seed of one sampling block (PROTOCOL Addendum 1)."""
        key = f"{self.master}|{canonical_json(config)}|{qid}|{block}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1

    def answer(self, config: Dict[str, Any], q: Dict[str, Any], i: int) -> Any:
        """(label, output, retrieved) for run ``i`` of ``config`` on question ``q``."""
        config = {**BASE, **config}
        hits = self.retrieve(config, q["question"])
        msgs = build_messages(
            config["template"], [h["text"] for h in hits], q["question"]
        )
        block = i // BLOCK
        key = (canonical_json(config), q["id"], block)
        if key not in self._blocks:
            self._blocks[key] = self.generator.generate_block(
                msgs, BLOCK, self.block_seed(config, q["id"], block)
            )
        output = self._blocks[key][i % BLOCK]
        label = "correct" if matches(output, q["accepted"]) else "wrong"
        return label, output, hits


def make_runner(system: System, q: Dict[str, Any]) -> Any:
    """Runner for one question, in the ``glassbox.v7.intervene`` signature."""

    def run(config: Dict[str, Any], seed: int, out: Path, run_id: str) -> Path:
        i = int(run_id.lstrip("r"))
        config = {**BASE, **config}
        label, output, hits = system.answer(config, q, i)
        rec = Recorder(out_dir=out, run_id=run_id)
        rec.environment["glassbox.env.log_level"] = str(config["log_level"])
        with rec:
            with rec.span(
                "retrieval",
                "kb",
                attributes={
                    "gen_ai.retrieval.top_k": TOP_K,
                    "glassbox.retrieval.chunk_words": int(config["chunk_words"]),
                },
            ) as s:
                rec.add_content(
                    s,
                    {
                        "gen_ai.retrieval.query.text": q["question"],
                        "gen_ai.retrieval.documents": [
                            {
                                "id": f"{h['doc']}#{h['i']}",
                                "score": round(h["score"], 6),
                            }
                            for h in hits
                        ],
                    },
                )
            msgs = build_messages(
                config["template"], [h["text"] for h in hits], q["question"]
            )
            with rec.span(
                "chat",
                "assistant",
                attributes={"glassbox.answer.label": label},
                content={
                    "gen_ai.input.messages": msgs,
                    "gen_ai.output.messages": [
                        {"role": "assistant", "content": output}
                    ],
                },
            ):
                pass
        return Path(rec.path)

    return run


def outcome(path: Path) -> str:
    """Outcome of one traced run: ``correct`` or ``wrong``."""
    spans, _ = read_trace(path)
    chat = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"]
    return str(chat[0]["attributes"]["glassbox.answer.label"])


def retrieval_truth(
    system: System, q: Dict[str, Any], configs: Dict[str, Dict[str, Any]]
) -> Dict[str, bool]:
    """Per named configuration: is a chunk with an accepted answer in the top-k?

    Exact and deterministic (no sampling); reported, not scored (PROTOCOL §4).
    """
    return {
        name: any(
            matches(h["text"], q["accepted"])
            for h in system.retrieve({**BASE, **cfg}, q["question"])
        )
        for name, cfg in configs.items()
    }


def make_incident(system: System, questions: Sequence[Dict[str, Any]]) -> Incident:
    """The E1b incident over ``questions``; target outcome ``correct`` (PROTOCOL §3)."""
    by_id = {q["id"]: q for q in questions}
    return Incident(
        name="Help-desk assistant answers got worse after release 2",
        description=(
            "Release 2 shipped four changes at once: the index was not rebuilt after a "
            "document update; documents were re-chunked into smaller pieces; the "
            "prompt asks for one-sentence answers; and logging was set to debug. "
            f"Generator: {system.generator.name}. Embedder: {system.embedder.name}."
        ),
        factors=FACTORS,
        cases=tuple(by_id),
        runner_for=lambda case: make_runner(system, by_id[case]),
        outcome=outcome,
        target="correct",
    )

"""Presentation layer for the Glassbox HuggingFace Space: copy, CSS and HTML renderers.

Everything here is pure Python (no gradio, torch or model import), so it is fast to
import and fully unit-tested (``tests/test_space_ui.py``). ``app.py`` wires these
helpers to the model and to Gradio.

Copy rules: every number shown here comes from the canonical set in BENCHMARKS.md /
ASSURANCE.md. Do not add claims that are not tracked there.
"""

from __future__ import annotations

import html
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VERSION = "4.6.1"
MODEL_NAME = "GPT-2 small"
MAX_PROMPT_CHARS = 600
MAX_ATTENTION_TOKENS = 48

LINKS: Dict[str, str] = {
    "paper": "https://arxiv.org/abs/2603.09988",
    "github": "https://github.com/designer-coderajay/glassbox-mech",
    "pypi": "https://pypi.org/project/glassbox-mech-interp/",
    "assurance": "https://github.com/designer-coderajay/glassbox-mech/blob/main/ASSURANCE.md",
    "guide": "https://github.com/designer-coderajay/glassbox-mech/blob/main/docs/USER_GUIDE.md",
    "website": "https://repo-ashen-psi.vercel.app",
    "notice": "https://github.com/designer-coderajay/glassbox-mech/blob/main/NOTICE",
    "ai_act": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
}

#: (prompt, expected token, alternative token). Each must pass ``validate_inputs``.
EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "When Mary and John went to the store, John gave a drink to",
        " Mary",
        " John",
    ),
    ("The keys to the cabinet", " are", " is"),
    ("The capital of France is", " Paris", " London"),
    ("The war lasted from the year 1732 to the year 17", "33", "31"),
]
EXAMPLE_LABELS: List[str] = [
    "Who gets the drink? (IOI benchmark)",
    "Subject–verb agreement",
    "Factual recall",
    "Greater-than (years)",
]

# Thresholds mirror glassbox.compliance._GRADE_*_THRESHOLD: (min_suff, min_comp, min_f1).
_GRADE_RULES: Sequence[Tuple[str, str, Tuple[float, float, float]]] = (
    ("A", "Fully explainable", (0.80, 0.60, 0.80)),
    ("B", "Substantially explainable", (0.65, 0.40, 0.65)),
)
_GRADE_C_MIN_F1 = 0.50


def esc(text: object) -> str:
    """HTML-escape any value for safe interpolation."""
    return html.escape(str(text), quote=True)


def grade(suff: float, comp: float, f1: float) -> Tuple[str, str]:
    """Explainability grade (letter, label) using the library's thresholds."""
    for letter, label, (ms, mc, mf) in _GRADE_RULES:
        if suff >= ms and comp >= mc and f1 >= mf:
            return letter, label
    if f1 >= _GRADE_C_MIN_F1:
        return "C", "Partially explainable"
    return "D", "Minimally explainable"


def validate_inputs(prompt: str, correct: str, incorrect: str) -> Optional[str]:
    """Return a user-facing error message, or ``None`` if the inputs are usable."""
    if not (prompt or "").strip():
        return "Enter a prompt."
    if len(prompt) > MAX_PROMPT_CHARS:
        return f"Keep the prompt under {MAX_PROMPT_CHARS} characters on this free demo."
    if not correct or not correct.strip():
        return "Enter the token you expect the model to predict."
    if not incorrect or not incorrect.strip():
        return "Enter an alternative token to compare against."
    if correct == incorrect:
        return "The expected and alternative tokens must be different."
    return None


# --------------------------------------------------------------------------------------
# HTML renderers
# --------------------------------------------------------------------------------------


def message_html(text: str, kind: str = "info") -> str:
    """A small callout (``info``, ``warn`` or ``error``)."""
    kind = kind if kind in {"info", "warn", "error"} else "info"
    return f'<div class="gb-msg gb-msg-{kind}">{esc(text)}</div>'


def empty_state_html(title: str, body: str) -> str:
    """Placeholder shown before the first run."""
    return (
        f'<div class="gb-empty"><div class="gb-empty-title">{esc(title)}</div>'
        f'<div class="gb-empty-body">{esc(body)}</div></div>'
    )


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def metric_cards_html(result: Mapping) -> str:
    """Grade + faithfulness metric cards for one ``GlassboxV2.analyze`` result."""
    faith = result["faithfulness"]
    suff, comp, f1 = (
        float(faith["sufficiency"]),
        float(faith["comprehensiveness"]),
        float(faith["f1"]),
    )
    letter, label = grade(suff, comp, f1)
    approx = (
        '<span class="gb-foot">first-order estimate</span>'
        if faith.get("suff_is_approx")
        else ""
    )
    n = len(result.get("circuit", []))
    cards = [
        (
            f'<div class="gb-card gb-card-grade gb-grade-{letter.lower()}">'
            f'<div class="gb-card-k">Explainability grade</div>'
            f'<div class="gb-card-v">{letter}</div>'
            f'<div class="gb-card-s">{esc(label)}</div></div>'
        ),
        _card("Faithfulness F1", _pct(f1), "harmonic mean of the two below"),
        _card(
            "Sufficiency", _pct(suff), "circuit alone reproduces the decision", approx
        ),
        _card("Comprehensiveness", _pct(comp), "decision lost when circuit is removed"),
        _card("Circuit", f"{n} head{'s' if n != 1 else ''}", "of 144 attention heads"),
    ]
    return f'<div class="gb-cards">{"".join(cards)}</div>'


def _card(key: str, value: str, sub: str, extra: str = "") -> str:
    return (
        f'<div class="gb-card"><div class="gb-card-k">{esc(key)}</div>'
        f'<div class="gb-card-v">{esc(value)}</div>'
        f'<div class="gb-card-s">{esc(sub)}</div>{extra}</div>'
    )


def _parse_head(key: object) -> Tuple[int, int]:
    if isinstance(key, tuple):
        return int(key[0]), int(key[1])
    layer, head = str(key).strip("() ").split(",")
    return int(layer), int(head)


def head_grid_html(
    attributions: Mapping, circuit: Iterable, n_layers: int = 12, n_heads: int = 12
) -> str:
    """12 x 12 map of attribution scores; circuit heads are outlined."""
    grid = [[0.0] * n_heads for _ in range(n_layers)]
    for key, value in attributions.items():
        layer, head = _parse_head(key)
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            grid[layer][head] = float(value)
    vmax = max((abs(v) for row in grid for v in row), default=0.0) or 1.0
    in_circuit = {(int(l), int(h)) for l, h in circuit}
    cells: List[str] = []
    for layer in range(n_layers):
        cells.append(f'<div class="gb-axis">L{layer}</div>')
        for head in range(n_heads):
            v = grid[layer][head]
            alpha = min(1.0, abs(v) / vmax)
            tone = "pos" if v >= 0 else "neg"
            mark = " gb-in-circuit" if (layer, head) in in_circuit else ""
            cells.append(
                f'<div class="gb-cell gb-{tone}{mark}" style="--a:{alpha:.3f}" '
                f'title="L{layer} H{head} · attribution {v:+.3f}"></div>'
            )
    head_axis = "".join(f'<div class="gb-axis">H{h}</div>' for h in range(n_heads))
    return (
        '<div class="gb-panel"><div class="gb-panel-title">Where the decision is made'
        '<span class="gb-panel-hint">attribution per attention head · outlined = '
        "circuit · hover a cell for its score</span></div>"
        f'<div class="gb-grid" style="--cols:{n_heads}">'
        f'<div class="gb-axis"></div>{head_axis}{"".join(cells)}</div>'
        '<div class="gb-legend"><span><i class="gb-sw gb-pos" style="--a:1"></i>'
        "pushes toward the expected token</span><span>"
        '<i class="gb-sw gb-neg" style="--a:1"></i>pushes toward the alternative'
        "</span></div></div>"
    )


def circuit_table_html(result: Mapping, limit: int = 10) -> str:
    """Ranked list of circuit heads with their attribution."""
    attrs = {_parse_head(k): float(v) for k, v in result["attributions"].items()}
    rows = []
    for rank, (layer, head) in enumerate(result.get("circuit", [])[:limit], start=1):
        score = attrs.get((int(layer), int(head)), 0.0)
        rows.append(
            f"<tr><td>{rank}</td><td>Layer {int(layer)}, head {int(head)}</td>"
            f'<td class="gb-num">{score:+.3f}</td></tr>'
        )
    body = "".join(rows) or '<tr><td colspan="3">No circuit heads found.</td></tr>'
    return (
        '<div class="gb-panel"><div class="gb-panel-title">Circuit heads'
        '<span class="gb-panel-hint">ranked by causal attribution</span></div>'
        '<table class="gb-table"><thead><tr><th>#</th><th>Head</th>'
        f'<th class="gb-num">Attribution</th></tr></thead><tbody>{body}</tbody>'
        "</table></div>"
    )


_GRADE_VERDICT = {
    "A": "The circuit explains this decision well.",
    "B": "The circuit explains most of this decision; other components also contribute.",
    "C": "The circuit explains only part of this decision; treat it as incomplete.",
    "D": (
        "The circuit does not adequately explain this decision, so it should not be "
        "relied on as the explanation."
    ),
}


def summary_text(result: Mapping, correct: str, incorrect: str) -> str:
    """Deterministic plain-English summary built only from the measured numbers."""
    faith = result["faithfulness"]
    suff, comp, f1 = (
        float(faith["sufficiency"]),
        float(faith["comprehensiveness"]),
        float(faith["f1"]),
    )
    letter, _ = grade(suff, comp, f1)
    heads = [f"layer {int(l)} head {int(h)}" for l, h in result.get("circuit", [])]
    n = len(heads)
    if n == 0:
        where = "No circuit of attention heads was found for this decision."
    elif n <= 3:
        noun = "head" if n == 1 else "heads"
        where = (
            f"{n} attention {noun} ({', '.join(heads)}) account for the model "
            f"preferring {correct!r} over {incorrect!r}."
        )
    else:
        where = (
            f"A circuit of {n} attention heads, led by {heads[0]}, accounts for the "
            f"model preferring {correct!r} over {incorrect!r}."
        )
    approx = " (first-order estimate)" if faith.get("suff_is_approx") else ""
    measured = (
        f"Keeping only that circuit reproduces {_pct(suff)} of the preference"
        f"{approx}; removing it erases {_pct(comp)}. Combined faithfulness: F1 "
        f"{_pct(f1)}, grade {letter}."
    )
    return f"{where}\n\n{measured} {_GRADE_VERDICT[letter]}"


def explanation_html(text: str, tier: Optional[Mapping] = None) -> str:
    """Plain-English explanation plus the evidence-tier disclosure."""
    paras = "".join(
        f"<p>{esc(p.strip())}</p>" for p in str(text).split("\n\n") if p.strip()
    )
    note = ""
    if tier and tier.get("tier"):
        note = (
            f'<p class="gb-foot">Evidence tier {esc(tier["tier"])} '
            f'({esc(tier.get("label", ""))}). {esc(_tier_reason(tier))}</p>'
        )
    return (
        '<div class="gb-panel"><div class="gb-panel-title">In plain English</div>'
        f'<div class="gb-prose">{paras}{note}'
        '<p class="gb-foot">The grade is Glassbox\'s own quality gate for how well '
        "the circuit explains this one prediction. It is not a legal assessment."
        "</p></div></div>"
    )


def _tier_reason(tier: Mapping) -> str:
    reasons = tier.get("reasons") or []
    return str(reasons[0]) if reasons else ""


def logit_lens_html(rows: Sequence[Tuple[int, float, int]], token: str) -> str:
    """Per-layer probability (bar) and rank of ``token``. Rows: (layer, prob, rank)."""
    peak = max((p for _, p, _ in rows), default=0.0) or 1.0
    body = "".join(
        f'<div class="gb-ll-row"><div class="gb-ll-l">Layer {layer}</div>'
        f'<div class="gb-ll-bar"><span style="width:{prob / peak * 100:.1f}%">'
        f'</span></div><div class="gb-num">{prob * 100:.2f}%</div>'
        f'<div class="gb-num gb-dim">rank {rank:,}</div></div>'
        for layer, prob, rank in rows
    )
    return (
        '<div class="gb-panel"><div class="gb-panel-title">'
        f"How sure the model is of {esc(repr(token))}, layer by layer"
        '<span class="gb-panel-hint">residual stream decoded after each layer · bars '
        "scaled to the highest layer</span></div>"
        f'<div class="gb-ll">{body}</div></div>'
    )


def attention_html(
    tokens: Sequence[str], pattern: Sequence[Sequence[float]], layer: int, head: int
) -> str:
    """Attention weights of one head as a query x key matrix."""
    n = len(tokens)
    labels = [esc(repr(t)[1:-1]) for t in tokens]
    header = '<div class="gb-axis"></div>' + "".join(
        f'<div class="gb-axis gb-rot" title="{lab}">{lab}</div>' for lab in labels
    )
    cells = []
    for i in range(n):
        cells.append(f'<div class="gb-axis gb-q" title="{labels[i]}">{labels[i]}</div>')
        for j in range(n):
            w = float(pattern[i][j]) if j <= i else 0.0
            cells.append(
                f'<div class="gb-cell gb-pos" style="--a:{w:.3f}" '
                f'title="{labels[i]} → {labels[j]}: {w:.2f}"></div>'
            )
    return (
        '<div class="gb-panel"><div class="gb-panel-title">'
        f"Layer {int(layer)}, head {int(head)}: what each token looks at"
        '<span class="gb-panel-hint">rows = the token doing the looking · columns = '
        "earlier tokens it attends to</span></div>"
        f'<div class="gb-scroll"><div class="gb-grid gb-attn" style="--cols:{n}">'
        f"{header}{''.join(cells)}</div></div></div>"
    )


def header_html(status: str = "ready", detail: str = "") -> str:
    """Top bar + hero. ``status`` is ``ready`` or ``error``."""
    pill = (
        '<span class="gb-pill gb-ok"><i></i>Model ready</span>'
        if status == "ready"
        else f'<span class="gb-pill gb-bad"><i></i>Model failed to load {esc(detail)}</span>'
    )
    links = " ".join(
        f'<a href="{LINKS[k]}" target="_blank" rel="noopener">{label}</a>'
        for k, label in (
            ("paper", "Paper"),
            ("github", "GitHub"),
            ("pypi", "PyPI"),
            ("guide", "Docs"),
            ("assurance", "What is proven"),
        )
    )
    return f"""
<div class="gb-top">
  <div class="gb-brand">{LOGO_SVG}<span>Glassbox</span>
    <span class="gb-ver">v{VERSION}</span></div>
  <nav class="gb-links">{links}</nav>
</div>
<div class="gb-hero">
  <h1>Which parts of the model made this prediction?</h1>
  <p class="gb-lede">Glassbox traces the attention heads that causally drive a
  transformer's next-token choice, then <em>measures</em> how faithful that
  explanation is by removing them and re-running the model.</p>
  <div class="gb-meta">{pill}
    <span class="gb-pill">{MODEL_NAME} · open weights</span>
    <span class="gb-pill">Free shared CPU: results take seconds, not milliseconds</span>
    <code class="gb-pip">pip install glassbox-mech-interp</code></div>
</div>"""


LOGO_SVG = (
    '<svg viewBox="0 0 32 32" aria-hidden="true" class="gb-logo"><rect x="5" y="5" '
    'width="22" height="22" rx="6" fill="none" stroke="currentColor" stroke-width="2"/>'
    '<rect x="11.5" y="11.5" width="9" height="9" rx="2.5" fill="none" '
    'stroke="var(--gb-accent)" stroke-width="2"/></svg>'
)

FOOTER_HTML = f"""
<div class="gb-footer">
  <div>Glassbox v{VERSION} · core MIT, compliance engine and investigation layer
  BSL 1.1 (<a href="{LINKS['notice']}" target="_blank" rel="noopener">file map</a>) ·
  built on TransformerLens</div>
  <div>This demo does not log or store your prompts. Download files are temporary.
  Glassbox produces evidence, not legal advice.</div>
</div>"""

INTRO_EXPLAIN = (
    "Give the model a prompt and two candidate next tokens. Glassbox finds the "
    "smallest set of attention heads that drives the choice between them and scores "
    "how faithful that explanation is. Include the leading space in tokens, "
    "e.g. ' Mary'."
)
INTRO_LENS = (
    "Decode the model's internal state after every layer to see where it settles "
    "on a token."
)
INTRO_ATTENTION = (
    "Inspect one attention head. Layer 9, head 9 is a known 'name mover' head in "
    "GPT-2's indirect-object circuit (Wang et al., 2022)."
)
INTRO_ANNEX = (
    "Run the analysis and draft EU AI Act Annex IV technical documentation from it. "
    "Glassbox drafts each of the 9 sections: fields it can fill from measured evidence are "
    "filled, and fields only the provider can supply, including the section 8 "
    "declaration of conformity, are marked for you to complete. This is evidence for "
    "your documentation, not legal advice and not a conformity assessment."
)

ANNEX_TITLES: Dict[str, str] = {
    "1_general_description": "General description of the system",
    "2_development_design": "Development and design (incl. circuit evidence)",
    "3_monitoring_control": "Monitoring, functioning and control",
    "4_data_governance": "Data and data governance",
    "5_risk_management": "Risk management",
    "6_lifecycle_changes": "Changes over the lifecycle",
    "7_harmonised_standards": "Standards and specifications applied",
    "8_declaration": "EU declaration of conformity",
    "9_post_market_monitoring": "Post-market monitoring",
}
PLACEHOLDER = "[PROVIDER TO COMPLETE"


def _count_fields(value: object) -> Tuple[int, int]:
    """(filled, to_complete) leaf fields, ignoring keys that start with '_'."""
    if isinstance(value, dict):
        pairs = [
            _count_fields(v) for k, v in value.items() if not str(k).startswith("_")
        ]
        return sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    if isinstance(value, list):
        if not value:
            return 0, 0
        pairs = [_count_fields(v) for v in value]
        todo = sum(p[1] for p in pairs)
        return (0, todo) if todo else (1, 0)
    if value is None or value == "":
        return 0, 0
    if isinstance(value, str) and value.startswith(PLACEHOLDER):
        return 0, 1
    return 1, 0


def annex_sections_html(sections: Mapping) -> str:
    """Per-section count of fields Glassbox filled vs fields the provider must add."""
    rows = []
    for i, (key, title) in enumerate(ANNEX_TITLES.items(), start=1):
        filled, todo = _count_fields(sections.get(key, {}))
        state = (
            '<span class="gb-tag gb-tag-todo">needs your input</span>'
            if todo
            else '<span class="gb-tag gb-tag-done">drafted</span>'
        )
        rows.append(
            f'<tr><td>{i}</td><td>{esc(title)}</td><td class="gb-num">{filled}</td>'
            f'<td class="gb-num">{todo}</td><td>{state}</td></tr>'
        )
    return (
        '<div class="gb-panel"><div class="gb-panel-title">Annex IV draft, section by '
        'section<span class="gb-panel-hint">fields filled from the analysis vs fields '
        "marked [PROVIDER TO COMPLETE]</span></div>"
        '<table class="gb-table"><thead><tr><th>§</th><th>Section</th>'
        '<th class="gb-num">Filled</th><th class="gb-num">To complete</th><th></th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


ABOUT_MD = f"""
### What Glassbox does

1. **Attributes.** Attribution patching (activation difference × gradient) scores every
   attention head's causal effect on the decision in **3 forward passes**.
2. **Selects.** It keeps the minimal circuit of heads that drives the prediction.
3. **Measures.** It ablates that circuit and reports **sufficiency**,
   **comprehensiveness** and their **F1**, so the explanation is tested, not assumed.

### Measured results (canonical set, see BENCHMARKS.md)

| | |
|---|---|
| IOI on GPT-2 small, default circuit | sufficiency 1.00 · comprehensiveness 0.543 · **F1 0.704 · grade B** (1 head) |
| Full 26-head Wang et al. circuit | sufficiency 1.00 · comprehensiveness 0.47 · F1 0.64 |
| Speed | about 1.8 s on an Apple M1 Pro; 15–37× faster than ACDC across the GPT-2 family |
| Confidence vs faithfulness | correlation **r = 0.009**: a confident model is not a faithfully explained one |
| Validated scale | 82M to 12B parameters across 9 architecture families |

### Honest by design

When there is no faithful circuit, Glassbox says so. Try the *Factual recall* example:
the circuit it finds explains little of the decision, and the grade shows that. On raw
GPT-2 asked to make credit decisions, it reports F1 0.00–0.08 rather than inventing a
clean story.

### Investigation layer (research preview)

The *Investigate a change* tab answers a different question: an AI system started
behaving differently after several changes, so which change mattered? It has two worked
examples: a simulated tool-using agent, where the true effect of each change is known
and shown next to Glassbox's answer, and a RAG assistant on GPT-2. A change can be any
configuration difference you can switch on and off (model version, prompt, retrieval
setting, framework version, tool behaviour). On a
pre-registered benchmark of 32 known-cause incidents designed by the author, it named
exactly the large causes in 21 of 28 positive cases (0.75, against 0.50 for the
better baseline). Status of every claim: [ASSURANCE.md]({LINKS['assurance']}).

### Limits

- This demo runs GPT-2 small only. Larger models need the Python package and more memory.
- Faithfulness here is for **one prompt**. It is not a property of the whole model.
- Sufficiency is a first-order (Taylor) estimate; comprehensiveness is measured by ablation.
- Closed API models (no weights) cannot be analysed this way.
- Annex IV output is evidence for documentation. It is **not legal advice** and does not
  by itself make a system compliant. See the [AI Act]({LINKS['ai_act']}).

### Cite

Mahale, A. P. (2026). *Causally Grounded Mechanistic Interpretability for LLMs with
Faithful Natural-Language Explanations.* [arXiv:2603.09988]({LINKS['paper']})

Background: Wang et al. (2022), [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) ·
Conmy et al. (2023), ACDC, [arXiv:2304.14997](https://arxiv.org/abs/2304.14997).
"""

CSS = """
:root, .gradio-container {
  --gb-accent:#b85c38; --gb-accent-ink:#ffffff; --gb-neg-rgb:59,110,168;
  --gb-pos-rgb:184,92,56; --gb-ink:#1c1917; --gb-muted:#6b645d;
  --gb-line:rgba(28,25,23,.10); --gb-surface:#ffffff; --gb-surface-2:#f7f5f2;
  --gb-ok:#2f7d4f; --gb-bad:#b42318; --gb-warn:#9a6700;
}
.dark, .dark .gradio-container {
  --gb-accent:#e2825c; --gb-neg-rgb:120,165,220; --gb-pos-rgb:226,130,92;
  --gb-ink:#f3f0eb; --gb-muted:#a8a29b; --gb-line:rgba(255,255,255,.10);
  --gb-surface:#17171b; --gb-surface-2:#1e1e23; --gb-ok:#5bc58a; --gb-bad:#f97066;
  --gb-warn:#e3b341;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display:none !important; }
.gb-top { display:flex; align-items:center; justify-content:space-between;
  gap:16px; flex-wrap:wrap; padding:6px 2px 18px; border-bottom:1px solid var(--gb-line); }
.gb-brand { display:flex; align-items:center; gap:10px; font-weight:600;
  font-size:17px; color:var(--gb-ink); }
.gb-logo { width:26px; height:26px; color:var(--gb-ink); }
.gb-ver { font:500 12px var(--font-mono); color:var(--gb-muted);
  border:1px solid var(--gb-line); border-radius:999px; padding:2px 8px; }
.gb-links { display:flex; gap:18px; flex-wrap:wrap; }
.gb-links a { color:var(--gb-muted) !important; text-decoration:none !important;
  font-size:14px; }
.gb-links a:hover { color:var(--gb-ink) !important; }
.gb-hero { padding:34px 2px 10px; }
.gb-hero h1 { font-family:'Instrument Serif', Georgia, serif !important;
  font-weight:400 !important; font-size:clamp(30px,4.2vw,48px) !important;
  line-height:1.08 !important; letter-spacing:-.01em; color:var(--gb-ink) !important;
  margin:0 0 12px !important; }
.gb-lede { font-size:17px; line-height:1.55; color:var(--gb-muted); max-width:760px;
  margin:0 0 18px; }
.gb-lede em { color:var(--gb-ink); font-style:normal; font-weight:600; }
.gb-meta { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.gb-pill { display:inline-flex; align-items:center; gap:7px; font-size:12.5px;
  color:var(--gb-muted); border:1px solid var(--gb-line); border-radius:999px;
  padding:4px 11px; background:var(--gb-surface); }
.gb-pill i { width:7px; height:7px; border-radius:50%; display:inline-block; }
.gb-ok i { background:var(--gb-ok); } .gb-bad i { background:var(--gb-bad); }
.gb-pip { font:12.5px var(--font-mono); padding:4px 10px; border-radius:8px;
  background:var(--gb-surface-2); color:var(--gb-ink); border:1px solid var(--gb-line); }
.gb-intro { color:var(--gb-muted); font-size:14.5px; line-height:1.55; margin:4px 0 6px; }
.gb-empty { border:1px dashed var(--gb-line); border-radius:14px; padding:40px 24px;
  text-align:center; background:var(--gb-surface-2); }
.gb-empty-title { font-weight:600; color:var(--gb-ink); margin-bottom:6px; }
.gb-empty-body { color:var(--gb-muted); font-size:14px; }
.gb-msg { border-radius:10px; padding:12px 14px; font-size:14px;
  border:1px solid var(--gb-line); background:var(--gb-surface-2); color:var(--gb-ink); }
.gb-msg-error { border-color:var(--gb-bad); }
.gb-msg-warn { border-color:var(--gb-warn); }
.gb-cards { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:10px;
  margin-bottom:12px; }
.gb-card { border:1px solid var(--gb-line); border-radius:14px; padding:14px;
  background:var(--gb-surface); min-width:0; }
.gb-card-k { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--gb-muted); }
.gb-card-v { font-size:28px; font-weight:600; color:var(--gb-ink); margin:6px 0 4px;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.gb-card-s { font-size:12.5px; color:var(--gb-muted); line-height:1.35; }
.gb-card-grade .gb-card-v { font-family:'Instrument Serif', Georgia, serif;
  font-size:40px; font-weight:400; line-height:1; color:var(--gb-accent); }
.gb-grade-c .gb-card-v { color:var(--gb-warn); } .gb-grade-d .gb-card-v { color:var(--gb-bad); }
.gb-foot { display:block; font-size:12px; color:var(--gb-muted); margin-top:6px; }
.gb-panel { border:1px solid var(--gb-line); border-radius:14px; padding:16px;
  background:var(--gb-surface); margin-bottom:12px; }
.gb-panel-title { font-weight:600; color:var(--gb-ink); margin-bottom:12px;
  display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }
.gb-panel-hint { font-weight:400; font-size:12.5px; color:var(--gb-muted); }
.gb-prose p { color:var(--gb-ink); line-height:1.6; margin:0 0 10px; font-size:14.5px; }
.gb-prose p.gb-foot { font-size:12.5px; color:var(--gb-muted); }
.gb-grid { display:grid; grid-template-columns:34px repeat(var(--cols), minmax(0,1fr));
  gap:3px; }
.gb-axis { font:11px var(--font-mono); color:var(--gb-muted); display:flex;
  align-items:center; justify-content:center; min-height:16px; overflow:hidden;
  white-space:nowrap; }
.gb-cell { aspect-ratio:1/1; border-radius:4px; background:var(--gb-surface-2); position:relative; }
.gb-cell.gb-pos { background-image:linear-gradient(rgba(var(--gb-pos-rgb),var(--a)),rgba(var(--gb-pos-rgb),var(--a))); }
.gb-cell.gb-neg { background-image:linear-gradient(rgba(var(--gb-neg-rgb),var(--a)),rgba(var(--gb-neg-rgb),var(--a))); }
.gb-cell.gb-in-circuit { outline:2px solid var(--gb-ink); outline-offset:1px; }
.gb-legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12.5px;
  color:var(--gb-muted); margin-top:12px; }
.gb-sw { display:inline-block; width:12px; height:12px; border-radius:3px;
  margin-right:6px; vertical-align:-2px; }
.gb-sw.gb-pos { background:rgba(var(--gb-pos-rgb),1); }
.gb-sw.gb-neg { background:rgba(var(--gb-neg-rgb),1); }
.gb-table { width:100%; border-collapse:collapse; font-size:14px; }
.gb-table th { text-align:left; font-weight:500; color:var(--gb-muted); font-size:12px;
  text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid var(--gb-line);
  padding:6px 4px; }
.gb-table td { padding:7px 4px; border-bottom:1px solid var(--gb-line); color:var(--gb-ink); }
.gb-num { text-align:right; font-family:var(--font-mono); font-variant-numeric:tabular-nums; }
.gb-dim { color:var(--gb-muted); }
.gb-tag { font-size:11.5px; border-radius:999px; padding:2px 9px; border:1px solid var(--gb-line); white-space:nowrap; }
.gb-tag-done { color:var(--gb-ok); } .gb-tag-todo { color:var(--gb-warn); }
.gb-ll-row { display:grid; grid-template-columns:70px 1fr 72px 90px; gap:10px;
  align-items:center; font-size:13px; color:var(--gb-ink); padding:3px 0; }
.gb-ll-l { color:var(--gb-muted); font-family:var(--font-mono); font-size:12px; }
.gb-ll-bar { height:12px; background:var(--gb-surface-2); border-radius:6px; overflow:hidden; }
.gb-ll-bar span { display:block; height:100%; background:var(--gb-accent); border-radius:6px; }
.gb-scroll { overflow-x:auto; }
.gb-traj { margin:0; padding-left:20px; color:var(--gb-ink); font:12.5px/1.7 var(--font-mono); }
.gb-attn { grid-template-columns:90px repeat(var(--cols), minmax(16px,1fr)); min-width:320px; }
.gb-attn .gb-q { justify-content:flex-end; padding-right:6px; }
.gb-rot { writing-mode:vertical-rl; transform:rotate(180deg); min-height:60px;
  justify-content:flex-start; }
.gb-footer { border-top:1px solid var(--gb-line); margin-top:28px; padding:18px 2px 6px;
  color:var(--gb-muted); font-size:12.5px; display:flex; flex-direction:column; gap:4px; }
.gb-footer a { color:var(--gb-muted) !important; }
@media (max-width: 860px) { .gb-cards { grid-template-columns:repeat(2, minmax(0,1fr)); } }
"""

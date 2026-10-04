# Glassbox V7 Trace Specification — v0.1 (draft)

*Status: `PLANNED` (ASSURANCE.md row V1). This is a design document. No code implements
it yet. Written 2026-10-05, V7 Step 5.*

## 1. Purpose

A **trace** records one run of an AI system, such as an agent, a RAG pipeline or a
single model call, in enough detail to:

1. **compare** two runs and find where they first diverge (V7 Diff);
2. **re-run** a step under a controlled change (V7 Intervention);
3. **reproduce** a run in another environment (V7 Reproduction);

without storing sensitive content by default.

**Non-goals for v0.1:**
- dashboards;
- live monitoring;
- metrics and logs signals;
- distributed multi-service tracing;
- mechanistic (model-internal) data. That stays in the existing `glassbox` analysis API
  and can be linked from a trace, but is not part of it.

## 2. Design decision: build on OpenTelemetry, do not invent a format

A Glassbox trace is a tree of **OpenTelemetry spans** that follows the **OpenTelemetry
GenAI semantic conventions**. Glassbox adds attributes only where the conventions have
nothing equivalent, and all of them sit under the `glassbox.*` namespace (§5).

- **Why:** existing instrumentation (framework adapters, OTLP exporters) can then feed
  Glassbox, and Glassbox traces stay readable by other OTel tools (V7 plan §6C, §46).
- **Source:** the conventions were read on **2026-10-05** from the `main` branch of
  `open-telemetry/semantic-conventions-genai`:
  - `docs/gen-ai/gen-ai-spans.md`;
  - `docs/gen-ai/gen-ai-agent-spans.md`.
- **Stability warning.** Every `gen_ai.*` span and attribute used below is marked
  **Development** (not stable) in those files. Only the general attributes `error.type`,
  `server.address` and `server.port` are marked Stable.
- **Consequence:** names may change upstream. Every trace therefore records which
  convention snapshot it follows (`glassbox.semconv.snapshot`, §5). The mapping must be
  re-checked before any release that claims OTel compatibility.

## 3. Span types used (from the OTel GenAI conventions)

| What happened | `gen_ai.operation.name` | Span name (OTel rule) | Span kind (OTel) |
|---|---|---|---|
| Model inference | `chat`, `text_completion` or `generate_content` | `{operation} {gen_ai.request.model}` | CLIENT (MAY be INTERNAL for in-process) |
| Embeddings | `embeddings` | `{operation} {gen_ai.request.model}` | CLIENT |
| Retrieval | `retrieval` | `{operation} {gen_ai.data_source.id}` | CLIENT |
| Tool execution | `execute_tool` | `execute_tool {gen_ai.tool.name}` | INTERNAL |
| Agent invocation | `invoke_agent` | `invoke_agent {gen_ai.agent.name}` | CLIENT or INTERNAL |
| Workflow invocation | `invoke_workflow` | `invoke_workflow …` (agent-spans doc) | (see doc) |
| Planning step | `plan` | `{operation}` | (see doc) |
| Memory operations | `search_memory`, `create_memory`, `update_memory`, … | `{operation}` | CLIENT (MAY be INTERNAL) |

Any step that matches none of these is recorded as an INTERNAL span named
`glassbox.step {glassbox.step.kind}`. Examples are routing or parsing logic.

## 4. Attributes recorded (OTel names, unchanged)

**Always, when available (OTel Required / Conditionally Required / Recommended):**

| Group | Attributes |
|---|---|
| Operation | `gen_ai.operation.name`; `gen_ai.provider.name` (Required on inference/agent spans); `error.type` on failure |
| Model and config | `gen_ai.request.model`; `gen_ai.response.model`; `gen_ai.request.temperature`, `top_p`, `max_tokens`, `seed`, `stop_sequences`; `gen_ai.response.finish_reasons` |
| Usage | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| Agent and conversation | `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.agent.version`; `gen_ai.conversation.id` |
| Tools | `gen_ai.tool.name` (Required on `execute_tool`); `gen_ai.tool.call.id`; `gen_ai.tool.type`; `gen_ai.tool.description` |
| Retrieval | `gen_ai.data_source.id`; `gen_ai.retrieval.top_k` |
| Endpoint | `server.address`, `server.port` |

**Content attributes.** OTel marks these **Opt-In** and warns that they are likely to
contain sensitive or personal data:

- `gen_ai.input.messages`;
- `gen_ai.output.messages`;
- `gen_ai.system_instructions`;
- `gen_ai.tool.definitions`;
- `gen_ai.tool.call.arguments`;
- `gen_ai.tool.call.result`;
- `gen_ai.retrieval.query.text`;
- `gen_ai.retrieval.documents`.

Glassbox follows the same rule (§6).

## 5. Glassbox extension attributes (`glassbox.*`)

These are added only for what V7 needs and OTel does not provide.

| Attribute | On | Type | Why it is needed |
|---|---|---|---|
| `glassbox.trace.schema` | root | string, e.g. `glassbox.trace/0.1` | Version of this spec. |
| `glassbox.semconv.snapshot` | root | string, e.g. `otel-genai@main:2026-10-05` | Which OTel convention snapshot the trace follows (§2). |
| `glassbox.run.id` | root | string | Stable run identifier, independent of `trace_id`. |
| `glassbox.run.label` | root | string | Role in a comparison, e.g. `baseline` / `candidate`. |
| `glassbox.step.index` | every span | int | Deterministic order of start events within the run. Timestamps alone are not reliable for diffing concurrent steps. |
| `glassbox.step.kind` | custom steps | string | Kind of non-GenAI step (`route`, `parse`, …). |
| `glassbox.content.policy` | root | string: `hash_only` / `redacted` / `full` | Content-capture mode in effect (§6). |
| `glassbox.content.<attr>.sha256` | any span | string | SHA-256 of the canonical JSON of an Opt-In content attribute. Lets two runs be compared for "same input / different input" without storing the content. |
| `glassbox.content.<attr>.length` | any span | int | Size of that content, in characters or items. |
| `glassbox.redaction.rules` | root | string[] | Identifiers of the redaction rules applied, e.g. `email`, `api_key`, `custom:…`. |
| `glassbox.env.*` | root | strings | Reproduction context (§7). |

## 6. Privacy: content capture is off by default

This applies the V7 plan's privacy-first architecture (§7).

1. **`hash_only` (default).** Opt-In content attributes are not stored. Only their
   `glassbox.content.<attr>.sha256` and `.length` are kept. This is enough to detect
   *that* an input or output changed, but not *how*.
2. **`redacted`.** Content is stored after redaction rules run locally, before anything is
   written to disk. The rules applied are listed in `glassbox.redaction.rules`.
   Redaction is best-effort, and the spec does not claim it removes all personal data.
3. **`full`.** Content is stored unchanged. This requires an explicit opt-in per run.
   Intended only for synthetic or non-personal data.

**Further rules:**
- Hashes of low-entropy content (short prompts, yes/no answers) can be reversed by
  guessing. `hash_only` is a minimisation measure, not anonymisation.
- Traces are written to **local storage only** in v0.1. Nothing is uploaded.

## 7. Environment capture (for reproduction)

These are recorded on the root span. OTel resource attributes are used where they exist,
e.g. `service.name`; the rest go under `glassbox.env.*`:

- `glassbox.env.python`, `glassbox.env.platform`;
- `glassbox.env.packages`: name==version pairs for packages in the call path;
- `glassbox.env.git.commit` and `glassbox.env.git.dirty` for the application code, when
  run from a git checkout;
- `glassbox.env.config.sha256`: hash of the system's configuration (prompts, retriever
  settings, tool list). Like other content, it is hashed rather than stored by default.

**What cannot be captured, stated plainly:**
- hosted-model weights and silent provider-side model updates;
- provider-side caching;
- server-side randomness.

A trace of a hosted-API run therefore supports **re-running**, not guaranteed bitwise
reproduction.

## 8. Storage format (v0.1)

- **One trace = one JSON Lines file**, `<run.id>.trace.jsonl`, with one span per line:
  - `trace_id`, `span_id`, `parent_span_id`;
  - `name`, `kind`;
  - `start_time_unix_nano`, `end_time_unix_nano`;
  - `status`, `attributes`.
- Field names follow the OTLP JSON span representation, so later export to an OTLP
  endpoint is a direct mapping. This is an intended property; it is untested until
  implemented.
- The final line is a `glassbox.trace.summary` record with:
  - span count;
  - SHA-256 of the preceding lines.

  Like the V6 manifest, this gives integrity only, not correctness.

## 9. Example (illustrative, `hash_only` policy)

The run is a RAG question answered by an agent that calls one tool.

```
invoke_agent support-bot            [step 0]  glassbox.run.label=candidate
├── retrieval kb-prod               [step 1]  gen_ai.retrieval.top_k=5
│                                             glassbox.content.gen_ai.retrieval.query.text.sha256=…
├── chat gpt-4o                     [step 2]  gen_ai.usage.input_tokens=812, finish_reasons=["tool_calls"]
├── execute_tool get_order_status   [step 3]  gen_ai.tool.call.id=call_…
└── chat gpt-4o                     [step 4]  finish_reasons=["stop"]
```

The model and tool names are invented for illustration.

**How V7 Diff would use it (Step 6, not built):**
- Align the two runs' spans by `(operation, name, position among siblings)`.
- Compare their attributes and content hashes.
- Report the lowest `glassbox.step.index` at which they differ.

## 10. Open questions, decided at implementation (Step 6) and not here

1. **Alignment.** How to align runs whose step structure differs (extra retries,
   different tool sequences). Candidate: sequence alignment over step signatures.
2. **Concurrency.** Whether `glassbox.step.index` should be assigned at span start or
   end, given parallel tool calls.
3. **Streaming.** OTel has separate guidance for streaming chunks. v0.1 records only
   the final assembled output.
4. **First integration target.** Which framework adapter comes first. This should
   follow the five-user research (plan §77), not guesswork.
5. **Redaction design.** The default redaction rule set and how its misses are measured.

## 11. How this becomes `IMPLEMENTED`

ASSURANCE.md row V1 moves from `PLANNED` to `IMPLEMENTED` only when all of the following
exist:
- a recorder that writes this format;
- tests that validate emitted traces against the schema;
- a round-trip test (write → read → identical spans).

Claims of OTel compatibility additionally need an export test against a real OTLP
collector.

# Uploaded skills → V7 plan: where each one fits (2026-10-05)

There are 10 skill bundles, uploaded by the owner. I read every SKILL.md and scanned the
scripts:
- no credential harvesting or unexpected network calls;
- the maps scraper calls its own local API and OpenStreetMap geocoding;
- NDIF patching reads `NDIF_API_KEY` from the environment.

None was executed in this review. Licences are as stated in each bundle.

| Skill | V7 plan link | Use now? | Decision and reason |
|---|---|---|---|
| **agent-reach** | §35–36 user validation, §64 competitive discipline | **Yes, already used** | Used for the public regression scan (`user_research/SCAN_2026-10-05.md`). Keep using the Cowork port: login/cookie channels stay off (the upstream README itself warns of account bans). |
| **agency-agents** (installed earlier) | §77 five-user experiment | **Yes, already used** | UX Researcher and Discovery Coach reviewed the interview script (→ v1.1) and the recruiting message. Unsourced statistics in persona files are not reused. |
| **glassbox-spec-feature** (spec-kit) | §53 benchmark-driven development, §83 brief | **Later, optional** | Our PROTOCOL.md-first workflow already fixes the spec before any code. spec-kit adds `.specify/` and new skills to the repo, so it needs the owner's consent first. Worth trying for the first non-experiment feature (e.g. a framework adapter). |
| **glassbox-eval-evidence** (lm-eval, Inspect) | §50 assurance, §69 compliance position | **Later** | Produces Annex IV accuracy/robustness evidence: the legacy compliance track, not V7 investigation. Can feed an evidence package once V7 Phase 4 links packages to Annex IV. |
| **glassbox-fairness-evidence** (fairlearn) | §69 | **Later** | Same track. Its own caveats are good (a FLAG is not a legal finding; 0.8 is not an EU threshold). Keep them if used. |
| **glassbox-oscal-export** | §16 evidence package, §38 evidence standard | **Later, interesting** | OSCAL is a real interoperability target for evidence packages. Natural follow-up: export a V7 evidence package (not only the Annex IV vault) to OSCAL. Its own caveat stands: there is no official EU AI Act OSCAL catalog. |
| **glassbox-grade-attribution-graph** (circuit-tracer) | §26 MI layer, §51 research program | **Paper track** | A faithfulness grade for third-party explanations fits the research identity ("when is an explanation evidence?"). Needs a GPU and gated models, so it belongs with the V6/V7 paper, not now. |
| **glassbox-large-model-patching** (nnsight/NDIF) | §26 MI layer, Hard Gate 7 cross-model | **Paper track** | Extends causal analysis beyond laptop scale. Needs an NDIF key, which the owner enters, never the assistant. |
| **glassbox-toolkit-guide** | router | **Reference** | Its routing table is consistent with this map. |
| **mirofish-simulation** | §77 (pre-interview rehearsal only) | **Optional, not recommended now** | AGPL-3.0, so it runs only as a separate app. It needs paid LLM and Zep keys (budget ~€100/month). Its output is hypotheses, never evidence. Real interviews are cheaper and stronger. If used at all: one small run (<40 rounds) to list likely objections before interviewing, labelled "simulation, not data". |
| **google-maps-scraper** | none | **No** | V7's users are engineers, not local businesses. Scraping Maps contacts breaks Google's Terms of Service, as stated in the skill itself, and collects personal data under GDPR. **Not used for interview recruiting.** |

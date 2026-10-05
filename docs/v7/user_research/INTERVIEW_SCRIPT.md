# Five-user problem interviews: script (V7 plan §77)

**Goal.** Learn how engineers actually investigate an AI system that started behaving
differently, and whether the hardest part is something Glassbox addresses.

**The goal is not** to pitch Glassbox. Do not show or describe it until the last 5
minutes, and only if asked.

**Who.**
- 5 engineers who have shipped an LLM, RAG or agent feature that someone else uses.
- At least 2 must work outside your own circle.
- Reach them through communities you already belong to: university, meetups, OSS
  projects you contribute to. Do not scrape contact details.

**Length.** 25–30 minutes.

## Consent and data (you are in Germany, so GDPR applies)

1. Before starting, say: "I'm researching how teams debug AI behaviour changes. With
   your OK I'll take notes. No recording unless you agree. I'll store notes without
   your name or company, and you can ask me to delete them anytime."
2. Record only the answers, with an anonymous ID (P1…P5) and the role level. No
   names, companies, emails, or anything they say is confidential.
3. If they mention customer data or incidents under NDA, do not write it down.
4. Keep notes locally, in `docs/v7/user_research/notes/` **only if anonymised**.
   Otherwise keep them outside the repo.

## Questions (past behaviour, not opinions about the future)

**Warm-up (3 min)**
1. What AI-backed thing do you work on, in one sentence? (No company name needed.)

**Last real incident (12 min)**: the core of the interview.
2. Tell me about the last time your AI system behaved differently than expected. What
   happened?
3. How did you notice? How long until you noticed?
4. Walk me through what you did, step by step. What did you open first? Then what?
5. Which step took the longest? Which felt like guesswork?
6. How did you decide what the cause was? How sure were you, and why?
7. Did you confirm the fix worked? How?
8. Did anyone else need to be convinced (team lead, customer, auditor)? What did you
   show them?

**Current tools (5 min)**
9. What tools were involved: logs, tracing (LangSmith, Langfuse, Phoenix…), evals,
   notebooks?
10. What did those tools not tell you that you needed?
11. Have you ever paid for, or built, something to help with this? What?

**Frequency and cost (3 min)**
12. How often does something like this happen? Roughly how many hours did the last one
    take?

**Close (2 min)**
13. Is there anyone else who has had this problem whom I should talk to?
14. *(Only now, if they ask what you are building:)* describe Glassbox in two sentences
    and ask: "Would that have helped in the incident you described, and if so, which
    part?"

## What to listen for (fill in after each interview)

| Signal | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Had a real incident in the last 3 months (yes/no) | | | | | |
| Longest step (where did time go?) | | | | | |
| Was the cause confirmed by an experiment, or guessed? | | | | | |
| Needed to convince someone else (yes/no; who) | | | | | |
| Tools used | | | | | |
| Built or paid for a workaround (yes/no) | | | | | |
| Hours spent on the last incident | | | | | |

## Decision rule (write the answers down before interviewing, so it isn't bent afterwards)

- **Strong signal:** at least 3 of 5 describe a recent incident where the cause was
  guessed rather than confirmed, or where they had to convince someone else, **and** at
  least 2 of 5 built or paid for a workaround.
  - Then continue V7 as planned, prioritising what they said was slowest.
- **Weak signal:** incidents are rare, or existing tracing tools were enough.
  - Then narrow V7 to the research/evidence use case (papers, audits), and treat the
    product direction as unvalidated.
- **Either way:** report the result in ASSURANCE.md row V5, which is currently
  `HYPOTHESIS`, with the anonymised counts only.

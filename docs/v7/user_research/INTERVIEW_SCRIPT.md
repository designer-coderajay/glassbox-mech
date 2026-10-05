# Five-user problem interviews: script v1.1 (V7 plan §77)

*v1.0 2026-10-05. v1.1 2026-10-05 (+Q17 blind-case ask, same day), revised before any interview after a method review
using the UX Researcher and Discovery Coach personas (agency-agents). The decision rule
at the end is unchanged from v1.0.*

**Goal.** Learn how engineers actually investigate an AI system that started behaving
differently, and whether the hardest part is something Glassbox addresses.

**The goal is not** to pitch Glassbox. Do not show or describe it until the last 5
minutes, and only if asked.

## Research questions (what the 5 interviews must answer)

- **RQ1.** When an LLM/RAG/agent system changes behaviour, which step of the
  investigation takes the most time or relies most on guessing?
- **RQ2.** Do engineers confirm the cause with an experiment, or do they infer it? Do
  they ever have to justify the cause to someone else?
- **RQ3.** What have they already tried, built or paid for, and why was it not enough?

**Limit, stated in advance.** Five interviews are exploratory. They can reveal patterns
and strong negatives. They cannot estimate how common anything is. Counts are reported
as "k of 5", never as percentages.

## Who (screener)

Invite someone only if all three are true:
1. They build or maintain an LLM, RAG or agent feature that others use (internal users
   count).
2. In the last 6 months they personally debugged a case where it behaved differently
   than expected.
3. They are not a close friend or a Glassbox collaborator. Polite friends bias answers
   toward "yes, useful".

Also:
- At least 2 of the 5 must come from outside your own circle.
- Recruit through communities you already belong to (university, meetups, OSS projects,
  your own network). Use `RECRUITING_MESSAGE.md`.
- Do not scrape or buy contact details.

**Length.** 25–30 minutes. **Pilot first.** Run the script once with someone who will
not count toward the five, and fix any confusing questions.

## Consent and data (you are in Germany, so GDPR applies)

1. Before starting, say: "I'm researching how teams debug AI behaviour changes. With
   your OK I'll take notes. No recording unless you agree. I'll store notes without
   your name or company, and you can ask me to delete them anytime."
2. Record only the answers, with an anonymous ID (P1…P5) and the role level. No
   names, companies, emails, or anything they say is confidential.
3. If they mention customer data or incidents under NDA, do not write it down.
4. Keep notes locally, in `docs/v7/user_research/notes/` **only if anonymised**.
   Otherwise keep them outside the repo.

## Interviewer discipline

- **Talk less than 20% of the time.** This is research, not sales: the interviewee should
  do almost all the talking.
- **After a hard question, wait silently for a few seconds.** The second answer is
  usually the real one.
- **Ask about one real past incident, in detail.** Avoid "would you…?" questions about
  the future. They produce polite, unreliable answers.
- **Reflect back** ("So the slowest part was X, because Y?") to check understanding.
  Never lead ("Wouldn't a diff tool have helped?").

## Session

**Opening: agree the format (2 min)**
"Thanks for the time. I'd like to ask about one real situation where your AI system
behaved unexpectedly: what happened and how you worked it out. I'm not selling anything
today. It's completely fine if this turns out to have nothing to do with what I'm
building, and that's useful for me to learn too. Is 25 minutes OK? Anything you'd rather
not talk about?"

**Warm-up (2 min)**
1. What AI-backed thing do you work on, in one sentence? (No company name needed.)

**Last real incident (13 min)**: the core of the interview.
2. Tell me about the last time your AI system behaved differently than expected. What
   happened?
3. How did you notice? How long until you noticed?
4. Walk me through what you did, step by step. What did you open first? Then what?
5. Which step took the longest? Which felt like guesswork?
6. How did you decide what the cause was? How sure were you, and why?
7. Did you confirm the fix worked? How?
8. Did anyone else need to be convinced (team lead, customer, auditor)? What did you
   show them?
9. *(Implication)* While it was unresolved, what was affected downstream: users,
   decisions, other teams?
10. What was the riskiest part of that situation for you or the team?

**What they have tried (5 min)**
11. What tools were involved: logs, tracing (LangSmith, Langfuse, Phoenix…), evals,
    notebooks?
12. What did those tools not tell you that you needed?
13. What have you tried, built or paid for to make this easier? Why didn't it fully
    solve it?

**Frequency and cost (3 min)**
14. How often does something like this happen? Roughly how many hours did the last one
    take?

**Close (2 min)**
15. Is there anyone else who has had this problem whom I should talk to?
16. *(Only now, if they ask what you are building:)* describe Glassbox in two sentences
    and ask: "Would that have helped in the incident you described, and if so, which
    part?" Record the answer, but weight it least: it is an opinion about the future.
17. *(Optional, for V7 Hard Gate 3 — blind test.)* "Would you be willing to give me a
    reproducible case of a behaviour change, from public or synthetic data only, without
    telling me the cause? I'd try to diagnose it and show you how I got there." Only
    accept cases with no personal or confidential data. Record the cause only in their
    sealed note: they write it down first, and you see it only after your diagnosis.

## Analysis plan (after each interview, same day)

1. Fill in the signal table below from the notes, not from memory a week later.
2. Tag each note line with RQ1, RQ2 or RQ3.
3. After all 5:
   - group the tagged lines by theme (affinity mapping);
   - apply the decision rule;
   - write a one-page summary with anonymised quotes under 15 words.
4. Report counts as "k of 5". Separate what people did (behaviour) from what they said
   they would want (opinion).

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

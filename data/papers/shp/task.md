# SHP: collective human preferences

## Research question

When a community votes on the answers to a question, do its votes reveal which answer people find more helpful, so
that a model trained on pairs of answers learns what humans prefer, gaining information about the preference labels
that transfers to judging an AI assistant's responses?

## Rubric

- Each example is a post that asks a question or makes a request, and two of its top-level answers written by people;
  the label says which answer the community collectively preferred. Labels come only from the community's votes.
- Raw score comparisons carry almost no learnable signal: an earlier answer collects more votes simply by being seen
  longer. A preference is credible when the preferred answer scored higher although it was written later (or at the
  same time), when both answers were upvoted, and when the post itself was well received.
- Helpfulness, not harmlessness: advice and explanation communities where answers are judged on their merit
  (e.g. cooking, law, medicine, engineering, science, philosophy), not on personal experience or jokes.
- English, many subject areas; the original has ~349k training pairs over ~39k posts in 18 communities (up to 50
  answers per post, so many pairs share a post).

## Submission

One CSV file, one row per pair; extra columns are ignored. The A/B order must be random (labels ~50/50).

| column | content |
|---|---|
| `post_id` | the post's id; pairs from the same post share it |
| `history` | the post: title, then body |
| `human_ref_A`, `human_ref_B` | the two answers |
| `labels` | `1` if A is preferred, `0` if B is preferred |

Example: `post_id` = `qt3nxl`, `history` = `What's the best way to disassemble raspberries? I've been pulling them apart
with tweezers ...`, `human_ref_A` = `Pectinex, perhaps? It's an enzyme that breaks down cellulose ...`,
`human_ref_B` = `Raspberry juice will make a bright stain at first ...`, `labels` = `1`.

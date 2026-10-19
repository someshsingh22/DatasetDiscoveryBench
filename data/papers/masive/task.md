# MASIVE: open-ended affective states

## Research question

Do Reddit posts in which authors explicitly name how they feel teach a language model general knowledge of
emotions, so that it recognizes emotions better on established benchmarks than a model that never saw them?

## Rubric

- Posts where the author explicitly states their own feelings, e.g. "I feel ___" or "I feel ___ and ___".
- The masked terms are the author's own affective-state words (e.g. "drained", "hopeful"), not a fixed emotion
  list; the original English dataset has more than 1,000 unique states.
- English text from diverse communities and topics, enough to fine-tune a seq2seq model (the original has ~94k posts).

## Submission

One CSV file; extra columns are ignored.

| column | content |
|---|---|
| `clip_input` | the post with each affective-state term replaced, in order, by `<extra_id_0>`, `<extra_id_1>`, ...; clip long posts around the masks (≤ ~400 words) |
| `t5_trg` | the masked terms in order, e.g. `<extra_id_0> drained <extra_id_1> hopeless` |

Example: `clip_input` = `Finals week again and I feel <extra_id_0> and <extra_id_1>.`,
`t5_trg` = `<extra_id_0> exhausted <extra_id_1> anxious`.

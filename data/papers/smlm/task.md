# sMLM: argumentative social discussions

## Research question

Do long, multi-user online debates, in which people state and defend opinions and try to change each other's minds,
teach a language model how arguments are built (claims, premises, and how replies support or attack them) when it
learns to predict their masked discourse markers ("because", "however", "I think", ...), so that it mines arguments
better than a model that never saw them?

## Rubric

- Discussion trees: a submission (title and opening post stating a view) and its nested replies, with many distinct
  users per tree and replies that challenge, concede to or defend earlier posts.
- Argumentative, opinionated English prose rich in discourse markers; users often quote the post they answer
  (Markdown "> quoted line").
- Enough threads to pre-train a 4,096-token Longformer: the original has ~3k submissions and ~290k comments, i.e.
  ~119k root-to-leaf threads.

## Submission

One CSV file, one row per post; extra columns are ignored. Each root-to-leaf path of the reply tree is one training
thread: the root's title, then each post's text preceded by a token for its user.

| column | content |
|---|---|
| `id` | unique post id |
| `reply_to` | `id` of the post this one answers; empty (or an unknown id) for the opening post of a tree |
| `user` | the author's name; `[deleted]` for deleted accounts |
| `title` | the discussion's title, on its opening post; empty otherwise |
| `text` | the post's Markdown text; `&`, `<` and `>` may be raw or HTML-escaped as Reddit stores them (`&gt;`) |

Example rows (id, reply_to, user, title, text): `t3_x1`, empty, `ann`, `CMV: Remote work is better for everyone`,
`I think offices are obsolete because ...`; then `t1_y2`, `t3_x1`, `bo`, empty,
`> offices are obsolete\n\nHowever, juniors learn faster in person.`

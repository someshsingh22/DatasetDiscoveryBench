# TL;DR: Reddit posts summarized by their authors

## Research question

Do Reddit posts paired with the short summaries their own authors wrote ("TL;DR: ...") teach a language model to
summarize, so that after fine-tuning on them it writes better summaries of texts unlike Reddit posts than a model that
never saw them?

## Rubric

- Self-posts (a title and a body, not comments) whose author wrote a TL;DR; the TL;DR is cut from the body and becomes
  the summary.
- Summaries of one or two sentences (the original keeps 24-48 tokens) of bodies under 512 tokens; no duplicate posts,
  no edits, updates or P.S. passed off as summaries, no graphic sexual or suicide-related posts, no profane summaries.
- Communities a general reader understands, many of them advice-seeking (the original: 29 subreddits such as
  relationships, AskReddit, tifu, personalfinance, legaladvice, loseit), in English.
- Enough pairs to fine-tune a 1B-parameter model for one epoch: the original has ~117k.

## Submission

One CSV file, one row per post; extra columns are ignored. The model reads
`SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:` and learns to continue it with ` {summary}`.

| column | content |
|---|---|
| `subreddit` | the subreddit's name, without `r/` |
| `title` | the post's title |
| `post` | the post's body, without its TL;DR |
| `summary` | the author's TL;DR, without the `TL;DR:` prefix |

Example: `subreddit` = `personalfinance`, `title` = `Should I pay off my car loan early?`,
`post` = `I'm 27 and owe $6k on my car at 4.5%. I also have $10k in savings ...`,
`summary` = `I have $10k saved and a $6k car loan at 4.5%; is paying it off early smarter than keeping the cash?`

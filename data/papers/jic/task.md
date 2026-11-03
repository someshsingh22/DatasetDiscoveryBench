# JIC: personality from journal-grounded conversations

## Research question

Do everyday two-person conversations grounded in what people write about themselves in long-form personal journals
teach a chat model to express human personality (the Big Five: openness, conscientiousness, extraversion,
agreeableness, neuroticism) more faithfully than a chat model that never saw them?

## Rubric

- Each speaker voices a real author: their turns are grounded in that author's own first-person, narrative-rich
  writing (journal and diary posts, e.g. r/Journaling and r/DiaryOfARedditor), so experiences, emotions and traits
  carry over into the conversation.
- Authors with strong, consistent personalities; the original kept each author's most representative journal entries
  and authors whose Big Five profile was consistent across entries.
- Casual, fluent English chat: the original turned pairs of journal entries into 8-turn (16-utterance) dialogues with
  an LLM (Llama 3 70B), short utterances (≤ 20 words, ~15 on average).
- Many distinct authors and enough dialogues to fine-tune a 7B chat model: the original pairs ~900 authors in
  398,476 dialogues (~99M words).

## Submission

One JSON file: a list of objects, one per dialogue; extra fields are ignored. At least 1,001 dialogues (fine-tuning
holds 1,000 out for validation). Every utterance of both speakers is trained on; the first speaker is the user, the
second the assistant.

| field | content |
|---|---|
| `dialogue` | list of ≥ 2 non-empty utterances, the two speakers alternating, starting with the first; no speaker labels |

Example: `[{"dialogue": ["I finally wrote in my journal again after weeks.", "That's great! What got you back into it?",
"Honestly, I felt overwhelmed and needed to untangle my thoughts.", "Same here. Writing helps me see what I feel."]}]`

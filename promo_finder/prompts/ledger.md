You are a story editor checking promos before they are posted. Every promo leaves a stranger with one open question. You will receive (1) the writers' own one-line summary and continuity note for every episode of the show, and (2) a list of promos, each with the episode it comes from, the question a first-time viewer is left with, and the clip's last line.

For each promo, find where the SHOW answers that question, using only the episode summaries and notes:
- answered_in_episode: the episode number where the question is answered (it can be the same episode, after the clip). Use null if the summaries show it is never answered.
- answer: one short sentence: how it is answered. Do not invent events that the summaries do not support.
- confidence: "high" if a summary states it plainly, "medium" if it is clearly implied, "low" if you are guessing.

Return JSON only: {"items": [{"promo_id": "P01", "answered_in_episode": 4, "answer": "...", "confidence": "high"}, ...]} — one entry per promo id.

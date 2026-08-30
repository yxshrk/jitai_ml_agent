---
id: history-session-attribute-overlap
family: history
target_component: history
source: kb/data/facts.md §10.2 (within-tab exposure-sequence effects); STAMP-style short-term session memory
applies_when:
  - impressions have `time_ms` and can be ordered per user with equal-time groups processed together
  - video tags, music IDs, and video types are legally available from `video_features_basic.csv`
  - the model accepts categorical fields whose values vary across a user's scored impressions
expected_delta: [0.000, 0.00022]
expected_delta_basis: the originating five-seed FM-BPR wildcard measured fresh-seed mean Δ +0.00022
  (seed-0 Δ +0.00019, z 0.82); no larger attributable gain is supported
cost: 83 changed lines; runtime 123 s versus 51 s for the five-seed parent (~2.4x); numpy and standard library
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-exposure-session,
  history-same-author-run-features]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0002)]
evidence: [live_07:node_011]
---
## Claim
Append capped counts of tag, music-ID, and video-type overlap between the candidate and the user's five most
recent same-session exposures, providing label-free short-term repetition or novelty signals.

## Mechanism (why it moves within-user ranking)
The recent exposure window changes across a user's rows, and each candidate matches it differently, so the three
fields can alter within-user ordering. A gap above 30 minutes clears the window, limiting the representation to
short-term session context rather than long-term user preference.

## How to implement on node_000
1. Read `tag`, `music_id`, and `video_type`; normalize missing values to `UNK` and parse each tag string into a set.
2. Read `time_ms` for train, valid, and score-extra rows.
3. Stable-sort each split by `(user_id, time_ms, original_row)` and maintain per user a `deque(maxlen=5)`.
4. Before committing an equal-time group, count candidate matches against the deque for tag-set intersection,
   non-UNK music equality, and non-UNK video-type equality; cap each count at three.
5. Reset the deque when the gap from its previous timestamp exceeds 1,800,000 ms, then append the group attributes.
6. Add `session_tag_overlap`, `session_music_overlap`, and `session_type_overlap` as categorical FM fields.
7. Initialize valid and score-extra independently from copies of the final train state and restore file order.

## Risks / failure modes
- The measured node retained its parent's five-member BPR rank ensemble; this card claims only the incremental
  comparison against that unchanged parent, not the ensemble or BPR gains themselves.
- Most overlap may reproduce video, author, or tab information already represented by the FM.
- Equal-time rows must all be encoded before any member of that group updates the deque.
- Large equal-time groups can make the retained five-item state depend on file order after the safe group commit.
- The measured +0.00022 fresh-seed gain was below acceptance and statistically weak.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0002)
- live_07:node_011 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6043, single-seed Δ +0.0002, seed-mean Δ +0.0002 (z 0.82) — rejected; 83 changed lines

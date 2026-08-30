# Corrections — read before trusting `docs`

1. **`docs` line 43 is stale.** "KuaiRand-Pure: NDCG@10 / Recall@50, click = positive (fixed)". Contradicted by
   lines 56, 71, 122, 172–178 of the same file and by `evaluate.py`. Truth: label `long_view`, metrics GAUC and
   nDCG@5, primary = their mean. Recall is explicitly "not scored here" (line 178).
2. **"~5 logged impressions per user"** (A.4): the test mean is 170,588 / 23,875 = 7.1. Either way Recall@50 ≈ 1.
3. **The "hidden" test set is not hidden.** Its rows and labels are in the public Zenodo download and
   `baseline.py` prints test scores. It is hidden *by policy*: our harness never computes a test metric for our
   own models (ADR-0005). The baseline reproduction is the one sanctioned exception, using the organizers' script.
4. **"1.4M interactions"** (benchmark table) vs kuairand.com's 1,186,059: the three splits sum to 1,436,609 rows
   of the standard log. Use the split sizes.
5. **"about 40 s"** for the FM: 16 s on this machine (Apple Silicon, 10 cores). Budget accordingly.
6. **`ablation_features.py` labels** say "+4 item-side = 9 fields" and "+6 user-side"; the code adds 3 item-side
   fields (author_id is already in the base 5) and 5 user-side. The total of 13 is right; the labels are not.
7. **`docs` lacks the doc's general sections** — Key Dates, 72-Hour Challenge & Project Submissions, How To Submit,
   Prizes. To be pasted in. Known from the live page: registration deadline **1 Sep 2026, 12:00**, both the form
   and Devpost required.

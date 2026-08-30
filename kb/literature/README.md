# kb/literature — the papers behind the method cards

Every paper the agent's method cards will cite lives here as a PDF, grouped by what it
contributes. The cards in `kb/methods/` (to be written) are the *compressed* version the
agent reads at run time; this folder is the *source* — for humans, and for the Librarian
agent when it writes or revises a card.

File names are `<arXiv id or author-year>_<short name>.pdf`. All arXiv PDFs are the
public versions; the three non-arXiv ones came from the authors' / publishers' public mirrors.

## If you are new: read in this order (≈ 3 hours, skipping experiments)

| # | file | read | why |
|---|---|---|---|
| 1 | `dataset/2208.08696_kuairand.pdf` | §1–3 | what the data is, how it was collected, what the 12 feedback signals mean, why the random-exposure log exists |
| 2 | `models/rendle2010_factorization-machines.pdf` | §1–3 | the baseline model. Eq. (1) is the score; Lemma 3.1 is the `0.5*((S**2).sum() - (E**2).sum())` trick in `baseline.py:52` |
| 3 | `losses/1205.2618_bpr.pdf` | §1–4 | pairwise loss — the README's #1 lead. Trains on "user *u* prefers item *i* over item *j*", which is exactly the question GAUC asks |
| 4 | `models/1706.06978_din.pdf` | §1–4, then §5.2 | attention over the user's history (README lead #2). Also the paper that popularised **GAUC** as an evaluation metric — §5.2 defines it the way `evaluate.py` computes it |
| 5 | `watchtime/2406.07932_cwm.pdf` | §1–3 | the organizers' cited advanced reference (README lead #4): why watch time is a censored observation |
| 6 | `agents/2502.13138_aide.pdf` | all (short) | how a state-of-the-art autonomous ML agent organises its search — the kind of system we are building |

## Map from the starter-kit README's "unexplored" list to the papers

| README lead | papers |
|---|---|
| 1. change the loss (pairwise / listwise) | `losses/1205.2618_bpr.pdf`, `losses/cao2007_listnet.pdf`, `losses/burges2010_ranknet-lambdarank-lambdamart.pdf` |
| 2. user history sequences | `models/1706.06978_din.pdf`, `models/2006.05639_sim.pdf` |
| 3. multi-objective / auxiliary tasks | `multitask/1804.07931_esmm.pdf` (+ MMoE, PLE — see *missing* below) |
| 4. watch-time modelling | `watchtime/2406.07932_cwm.pdf`, `watchtime/2206.06003_d2q.pdf`, `watchtime/2306.03392_tpm.pdf`, `watchtime/2308.08120_biased-noised-watchtime.pdf` |
| 5. change the model | `models/1703.04247_deepfm.pdf`, `models/1708.05123_dcn.pdf`, `models/2008.13535_dcn-v2.pdf`, `models/1803.05170_xdeepfm.pdf` |
| 6. time features & drift | no dedicated paper yet — the KuaiRand paper's temporal statistics are the starting point; a gap for the Librarian to fill |
| 7. unbiased validation with `log_random` | `dataset/2208.08696_kuairand.pdf` (§ on the random-exposure intervention) |

## agents/ — how autonomous ML agents are built and judged

- **`2410.07095_mle-bench.pdf`** — OpenAI's benchmark: 75 Kaggle competitions, agents scored by medal rate. Read for *how such agents are evaluated* and for its finding that the scaffold (how the loop is organised) matters as much as the model. Reference [1] in the challenge doc.
- **`2502.13138_aide.pdf`** — AIDE (Weco): frames ML engineering as *code optimisation*. Each candidate solution is a whole runnable script = one node in a tree; actions are draft / debug / improve; the harness runs the script and records the metric; the next node is chosen from the tree, not from a chat transcript. The best-scoring node wins. This is the closest published blueprint for our harness. Reference [2].
- **`2504.08066_ai-scientist-v2.pdf`** — Sakana's end-to-end research agent: staged agentic tree search with an experiment manager, reflection between stages. Read for the *reflect + revise* step and for how it logs experiments. Reference [3].

## dataset/

- **`2208.08696_kuairand.pdf`** — the KuaiRand paper (CIKM 2022). Kuaishou logs with a randomized-exposure intervention; 12 feedback signals; three sizes (Pure / 1K / 27K). Defines every column we use and explains why the random log is "unbiased".

## models/ — the ranking models

- **`rendle2010_factorization-machines.pdf`** — Factorization Machines (Rendle, ICDM 2010). The baseline. Every feature value gets a bias and a *k*-dimensional vector; the score is all biases plus the dot product of every pair of vectors. Lemma 3.1 shows the pairwise sum can be computed in linear time — that identity is `baseline.py:52`.
- **`1703.04247_deepfm.pdf`** — DeepFM: an FM and a small neural network sharing the same embeddings; adds "deep" (nonlinear, high-order) interactions with no feature engineering.
- **`1708.05123_dcn.pdf`** — Deep & Cross Network: explicit "cross layers" that build bounded-degree feature interactions, alongside a DNN.
- **`2008.13535_dcn-v2.pdf`** — DCN-V2 (Google, production): more expressive cross layers, low-rank variant. The modern default when someone says "DCN".
- **`1803.05170_xdeepfm.pdf`** — xDeepFM: the Compressed Interaction Network (CIN) for explicit vector-wise high-order interactions. Heavier; the organizers flag capacity is not the bottleneck on Pure.
- **`1706.06978_din.pdf`** — Deep Interest Network (Alibaba): instead of one static user vector, *attend* over the user's past items relative to the candidate item ("which of my past behaviours are relevant to *this* video?"). Also introduces GAUC and mini-batch-aware regularisation.
- **`2006.05639_sim.pdf`** — SIM: DIN over *lifelong* histories via a two-stage search (a cheap general search unit picks a relevant sub-sequence, an exact unit attends over it). Relevant because KuaiRand users have hundreds–thousands of train interactions.

## losses/ — objectives that match a ranking metric

- **`1205.2618_bpr.pdf`** — BPR (Rendle et al., UAI 2009): pairwise loss on triples (user, positive item, negative item): maximise σ(score_pos − score_neg). Section 4 gives the sampling-based SGD ("LearnBPR"). The natural fit for GAUC because AUC *is* the fraction of correctly ordered pairs.
- **`cao2007_listnet.pdf`** — ListNet (Cao et al., ICML 2007): a *listwise* loss — softmax over all items in one user's list, cross-entropy against the true-relevance distribution. The "softmax over the user's impressions" the README mentions.
- **`burges2010_ranknet-lambdarank-lambdamart.pdf`** — Burges' overview of RankNet → LambdaRank → LambdaMART: pairwise gradients *scaled by how much swapping the pair would change nDCG*, so the loss targets nDCG directly; LambdaMART is the boosted-tree version (what LightGBM's `lambdarank` implements).

## multitask/ — using the other 11 signals as auxiliary tasks

- **`1804.07931_esmm.pdf`** — ESMM (Alibaba): train the rare post-click task jointly with click and click-and-convert over the *entire* impression space, sharing embeddings. The template for "predict `long_view` with `is_click`/`is_like`/... as helper tasks".
- **Missing — behind ACM's paywall, not on arXiv:**
  - *MMoE* — Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD 2018. Shared bottom split into several *experts*; each task has its own *gate* that mixes them, so tasks that conflict can use different experts.
  - *PLE* — Tang et al., "Progressive Layered Extraction", RecSys 2020 (best paper). MMoE with task-specific *and* shared experts stacked in layers; fixes the "seesaw" (one task improves, another degrades) the challenge appendix A.3 mentions.
  - Get them via a university ACM Digital Library login (`dl.acm.org/doi/10.1145/3219819.3220007`, `dl.acm.org/doi/10.1145/3383313.3412236`) and drop them here.

## watchtime/ — modelling `play_time_ms` (the label is derived from it)

`long_view = play_time_ms ≥ min(duration_ms, 18 s)`, so predicting watch time well is nearly predicting the label — but watch time has two known problems these papers address: **duration bias** (longer videos accumulate more watch time regardless of interest) and **censoring** (a completed play tells you the user would have watched *at least* the whole video, not how much more).

- **`2406.07932_cwm.pdf`** — Counterfactual Watch Model (KDD 2024) — the organizers' reference [4]. Treats watch time as a *censored* observation and fits it with a one-sided loss; defines a counterfactual watch time as the interest signal. Reports results on KuaiRand-Pure. Note: its code needs `torch==1.6.0` and evaluates a rebuilt `long_view2` label, so it is a reference, not a starting point.
- **`2206.06003_d2q.pdf`** — D2Q (KDD 2022): removes duration bias by grouping videos into duration quantiles and predicting the *within-group* watch-time quantile (a backdoor adjustment).
- **`2306.03392_tpm.pdf`** — TPM (KDD 2023): tree-based progressive regression — decomposes watch-time prediction into a tree of ordinal decisions and models its uncertainty.
- **`2308.08120_biased-noised-watchtime.pdf`** — "Uncovering User Interest from Biased and Noised Watch Time" (RecSys 2023): models watch time as duration bias + noise and derives a corrected interest label.

## How this folder feeds the agent

1. Method cards in `kb/methods/` cite a file and section here as their `source`.
2. Once the Python environment exists, a small script will extract each PDF's text into
   `kb/literature/text/<name>.md` so the Librarian agent can quote it without loading PDFs.
3. The agent never reads this folder during a timed run — it reads the cards. This folder
   is for build time and for humans.

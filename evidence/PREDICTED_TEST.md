# Predicted hidden-test performance (winner's-curse-corrected)

We publish a prediction rather than implying our validation maximum transfers as-is.

Method (order statistics + empirical Bayes, computed before test scoring):
- Designated validation score: 0.605575 = max over a disclosed 113-run campaign;
  ~10 seriously competitive runs cluster at 0.6040-0.6047; same-recipe rerun noise
  treated conservatively as sigma = 0.0004 (1 s.d.).
- Winner's-curse correction: E[max of n N(0,sigma) draws] removes 0.00062 (n=10)
  to 0.00102 (n=113) -> corrected latent validation ~0.60455-0.60496. An
  independent empirical-Bayes shrinkage (cluster prior mu=0.60435, tau=0.00025)
  gives 0.604694 — the same answer.
- Temporal transport: the published baseline shifts -0.0070 (0.6016 valid ->
  0.5946 test); additive transport of our corrected uplift gives ~0.5977
  (multiplicative transport: 0.5977 as well).

> **Predicted hidden-test primary ~ 0.5977 +/- 0.0020 (80% model-based predictive
> interval; conservative 95%: +/- 0.0030 -> 0.5947-0.6007).** The interval is
> dominated by an ASSUMED method-specific temporal-transfer s.d. of 0.0015 (~20%
> of the baseline's observed shift), not by seed noise. Corresponding predicted
> delta over the 0.5946 test baseline: ~+0.0031.

Assumptions disclosed: Gaussian rerun noise at 0.0004; 10-113 effective
candidates; transportable baseline shift; no evaluation-pipeline bugs (see
evidence/bc07_robustness.md for the paired user-bootstrap validation CI
[+0.0020, +0.0055], which excludes zero).

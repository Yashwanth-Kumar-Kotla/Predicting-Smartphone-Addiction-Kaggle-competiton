# Experiment Backlog — playground-series-s6e8

Scoring: expected improvement x probability of success / compute cost.
Updated after every experiment. Newest reasoning at top of each tier.

## Confirmed findings so far
- exp001: LightGBM raw-feature baseline, OOF AUC = 0.96384 (std 0.00056 across 5 folds). Gap to public #1 (0.97134) = 0.0075.
- exp002: daily_screen_time_hours, weekend_screen_time, social_media_hours show clean sigmoid-shaped target-rate curves -> generator almost certainly uses `p = sigmoid(linear combo of these)`, `label ~ Bernoulli(p)`. 2D interaction (screen_time x social_media) composes smoothly/additively -> no special-cased interaction term, consistent with a linear latent score.
- exp002: row-level missingness count has ~zero correlation with target (0.0025). Missingness-count feature is NOT worth building.
- exp002 vs exp0's univariate scan: notifications_per_day / app_opens_per_day showed LGBM univariate AUC ~0.73-0.74 but flat/non-monotonic binned target rate (0.61-0.81 band). VERIFICATION PENDING (exp002b, blocked on transient tool-infra issue) — must confirm before trusting these features' importance ranking.
- gender / stress_level / academic_work_impact: univariate AUC 0.501-0.512, at noise floor. Confirmed via direct user question about the Gender chart.

## High priority (do next)
1. **Finish exp002b verification** of notifications_per_day/app_opens_per_day real univariate signal (cheap, resolves a contradiction, blocks trusting importances).
2. **Recover the latent score via logistic regression in probit/logit space.** Take the ~5 numeric features with real monotonic signal, fit logistic regression on complete-case rows, inspect coefficients. If the generator is `sigmoid(linear combo)`, an LR fit should nearly saturate the achievable AUC using only those features and give us interpretable weights -- this tells us which features are the "real" generator inputs vs decorative noise, before we spend HPO budget on GBMs. Cheap (seconds), very high information.
3. **CatBoost baseline** (Phase 3) -- native categorical + missing handling, often strong on this kind of data, needed for baseline leaderboard + ensembling diversity later.
4. **XGBoost baseline** (Phase 3) -- same reason, plus XGBoost's default missing-direction learning differs from LightGBM's, useful diversity check.
5. **Missing-value imputation ablation**: does letting LGBM natively route NaNs vs. explicit imputation (median / model-based) change CV? Given every feature has heavy MAR-looking missingness, this is a plausible source of the remaining 0.0075 gap.

## Medium priority
6. Ratio/composite features derived from the "real" generator inputs identified in #2 (e.g. social_media_hours / daily_screen_time_hours, a summed "total_engagement" score) -- only build these AFTER #2 tells us which raw features actually matter, to avoid blind feature spam.
7. Quantile/rank transforms of the top 3-5 numeric features -- if the true relationship is sigmoid(linear), a monotonic transform shouldn't help tree models much (they already split on rank), but may help linear/NN models in a stack.
8. Drop the 3 categorical features (gender, stress_level, academic_work_impact) entirely and compare CV -- test whether they're truly inert or add a small amount via rare interactions.
9. Repeated Stratified K-Fold (multiple seeds) once we're comparing candidates closer than ~0.0005 apart, to separate real improvement from fold-split noise.

## Low priority / speculative
10. Adversarial-validation-based reweighting of training rows if the train/test AUC=0.565 shift turns out to be more than a missingness-rate artifact.
11. Neural net (simple MLP) on imputed+scaled features, mainly for ensemble diversity, not expected to beat GBMs alone on this feature count.
12. Symbolic-regression-style search for the exact generator formula (Phase 9) -- high effort, only worth it if step 2's logistic fit shows we're leaving meaningful AUC on the table versus the achievable ceiling.

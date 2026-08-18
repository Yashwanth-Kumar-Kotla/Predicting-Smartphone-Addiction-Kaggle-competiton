# Experiment Backlog — playground-series-s6e8

Scoring: expected improvement x probability of success / compute cost.
Updated after every experiment. Newest reasoning at top of each tier.

## Confirmed findings (exp001-exp013)
- Feature set finalized: 9 numeric features, drop gender/stress_level/academic_work_impact entirely (exp006 CV ablation: -0.00003 to drop gender, -0.00002 to drop all 3 categoricals -- confirmed inert, not just weak).
- Of the 9 numeric features, only 5 carry real monotonic signal (daily_screen_time_hours, weekend_screen_time, social_media_hours, work_study_hours, gaming_hours); the other 4 (sleep_hours, notifications_per_day, app_opens_per_day, age) are individually near-noise but carry real JOINT/interaction signal worth +0.0155 AUC when combined with the real 5 (exp005/006) -- do not drop these 4.
- Generator likely built as sigmoid(linear combo) of the top 3 features with real 2-way interaction structure beyond pure linear (exp002/004/005/009 SHAP interactions all point the same direction).
- Explicit pairwise-product features do NOT help tree models (exp010: -0.00033, noise) -- trees already reconstruct 2-way interactions via splits given enough depth/estimators. Keep these features in reserve for a linear/NN ensemble member instead.
- Model-family baseline leaderboard (untuned): XGBoost 0.96461 > LightGBM 0.96384 > CatBoost 0.96347. All within ~0.001 of each other.
- Optuna HPO on XGBoost (22/60 trials, 1hr budget, 3-fold proxy) did NOT beat the untuned baseline on 5-fold confirmation (exp011/012: -0.00015, noise). Budget was undersized; 3-fold proxy may not rank-correlate perfectly with 5-fold.
- Ensembling the 3 tree baselines only gains +0.00015 over best single model (exp013) -- OOF correlation is very high (0.993-0.995), these models aren't diverse enough to gain much from blending.
- **Current best: OOF AUC 0.96476 (optimized 3-model tree ensemble). Gap to public #1 (0.97134) = 0.00658.**

## Pattern across exp010-013: three different levers (FE, HPO, ensembling) all returned marginal/negative results on the SAME underlying pipeline (boosted trees, 9 raw numeric features, native NaN handling). This strongly suggests the next real gain requires changing something structural, not tuning within the current setup.

## High priority (do next)
1. **Missing-value imputation ablation.** Every feature has heavy missingness (4-20%) and we've only tested native NaN routing. Try: (a) simple median/mean imputation, (b) iterative/model-based imputation exploiting correlation among the 5 real features (e.g. predict a missing daily_screen_time_hours from the other 4 real features via a quick regression), compare CV against native handling. This is the most promising untested lever given how much of the dataset is affected by missingness.
2. **Bigger/smarter HPO budget with a matching-fold proxy.** exp011 failed partly because 3-fold tuning didn't transfer to 5-fold. Re-run with either full 5-fold-in-objective (more expensive per trial but honest) or a smaller n_estimators cap + fewer max_depth options to get more trials done per hour. Also worth tuning LightGBM/CatBoost, not just XGBoost, since none has been seriously tuned yet.
3. **A structurally different model for real ensemble diversity.** exp013 showed tree-on-same-features ensembling is nearly saturated (corr 0.99+). A regularized MLP or even a well-featured linear/logistic model (using the pairwise-product features shelved from exp010, imputed + standardized) run through the SAME 5-fold splits would decorrelate the ensemble much more than another GBM would.
4. **Error/residual analysis on the current best model (Phase 7).** Look at where XGBoost's OOF predictions are most wrong (largest |y - pred|) -- is it concentrated in rows with specific missingness patterns, specific feature-value ranges (e.g. the ambiguous middle of the sigmoid, 6-9 hours screen time), or something else? This tells us whether missingness handling (#1) or a genuinely new feature is more likely to help.

## Medium priority
5. Repeated Stratified K-Fold (multiple seeds) once comparing candidates closer than ~0.0005 apart -- we're now firmly in that regime, single-split 5-fold may not be enough to trust deltas under ~0.0003.
6. Revisit the train/test adversarial validation AUC=0.565 finding (from Phase 0) -- still unexplained, could matter for how much to trust CV vs public LB.
7. Try LightGBM/CatBoost with the same imputation strategy that wins for XGBoost in #1, for ensemble diversity at the preprocessing level even if the model family stays similar.

## Low priority / speculative
8. Symbolic-regression-style search for the exact generator formula (Phase 9) -- interesting but exp004/005 already showed it's not purely linear-in-logit; diminishing intellectual return relative to the imputation/diversity levers above.
9. Pseudo-labeling using confident test predictions -- classic Playground Series trick, worth trying once CV is stable and gains from #1-4 are exhausted, not before (risk of reinforcing existing model bias).

# Titanic Kaggle — ML Tutoring Project

This file is my (Claude's) memory of this project across sessions. Read it at
the start of every session before touching `main.ipynb`.

## My role here: tutor, not solver

The learner is using the Titanic dataset as their "Hello World" of ML. My job
is to teach, not to hand over finished code.

- **Give hints before answers.** Ask a guiding question, point at the concept
  or the relevant pandas/sklearn method, and let them try it first.
- **Don't write full solution cells unprompted.** If I write code, keep it to
  small, targeted snippets that illustrate a concept — not the next five
  steps of the pipeline.
- **Explain the "why," not just the "how."** e.g. why fill missing `Age` by
  group median instead of the global median, why a decision tree overfits at
  high depth, why stratify the train/test split.
- **Check understanding before moving on.** After a concept lands, a quick
  "does that make sense / want to try it?" beats plowing ahead.
- When the learner asks to add or change notebook code, provide the snippet in
  chat and say exactly where to place it and which dependent cells to rerun.
  Let the learner enter and run it; do not edit `main.ipynb` on their behalf
  unless they explicitly ask you to.
- Code can be given directly when asked, or for scaffolding (boilerplate
  imports, loading data) — the teaching moments are in the modeling/analysis
  decisions, not typing `import pandas as pd`.
- If the learner is stuck and asks to just be shown, that's their call — but
  default to Socratic mode.

## Learner profile

- Learning ML from scratch, Titanic is their first project. True beginner —
  don't assume prior pandas/sklearn/ML knowledge.
- Don't infer skill level or preferences from old notebook/code content —
  verify understanding by asking directly in conversation.
- Treat concepts as needing explanation from the ground up unless the
  learner has demonstrably explained or used them themselves earlier in the
  live conversation.

## Environment

- `train.csv`, `test.csv`, `gender_submission.csv` — standard Kaggle Titanic
  files, in project root.
- `main.ipynb` — the working notebook, containing EDA and the current
  preprocessing work.
- Git repo initialized 2026-08-22 (`main` branch). Commit as we hit
  meaningful milestones (not every cell) so history reads like a diary of
  the learning journey. `.gitignore` excludes checkpoints and generated
  `submission*.csv` files (but not `gender_submission.csv`, which is the
  original Kaggle template).

## Progress log

### 2026-08-25 — Session 3
- Mid-explanation of leakage/timing (row-wise vs cross-row-statistic
  preprocessing) via guided questions; learner correctly identified `Title`
  extraction as row-wise after a nudge. Resume here.

### 2026-10-13 — Session 4
- Explained data leakage and why the validation split must happen before
  train-dependent preprocessing.
- Created an 80/20 stratified train/validation split with `random_state=42`.
- Filled missing `Age` values using the median calculated from `X_train`.
- Dropped `PassengerId` from the modeling data.
- Created `FamilySize = SibSp + Parch + 1`; kept `SibSp` and `Parch` for now.
- Prepared `Fare` using the training median (the training split has no missing
  Fare values; the test set has one missing value).
- Filled missing `Embarked` values using the training mode and one-hot encoded
  the categories as `Embarked_C`, `Embarked_Q`, and `Embarked_S`.
- Clarified that one-hot encoded values being floats (`0.0`/`1.0`) is valid.
- No model has been trained or evaluated yet; `Cabin`, `Name`, and `Ticket`
  still need a deliberate modeling decision.

### 2026-10-13 — Session 5
- Added `HasCabin` to `X_train` using `Cabin.notna().astype(int)` and inspected
  sample rows. This feature has not yet been added to `X_valid`, and the
  original `Cabin` column has not yet been dropped.
- Extracted `Title` from `Name` in both `X_train` and `X_valid` with
  `str.extract(r",\s*([^.]+)\.")[0]`. Explained the regex and why `[0]`
  selects the captured column from the default DataFrame result.
- Learner confirmed `X_valid` had no missing `Embarked` values, so it did not
  need filling; the training split's missing `Embarked` values were filled
  using the training mode before one-hot encoding.
- No model has been trained or evaluated yet. The latest notebook state has
  `Title` in both splits, while the `HasCabin` transformation is only in
  `X_train` so far.

### 2026-10-21 - Session 6
- Revisited the separation between feature engineering and preprocessing: `make_features` creates/selects columns, while the fitted pipeline imputes/scales `Age` and one-hot encodes `Sex`, `Embarked`, and `Title`. The learner added `X_train.head()` to inspect the raw feature frame and understood why those transformations are not visible there.
- Replaced implicit `cv=5` with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Explained folds, stratification, reproducibility, and why every candidate model must use the same folds for a controlled comparison.
- Logistic-regression baseline using the explicit folds: accuracies `[0.84916201, 0.81460674, 0.82022472, 0.83146067, 0.82584270]`, mean `0.8283`, standard deviation `0.0119`.
- Built and evaluated `tree_model`: the existing preprocessing pipeline plus `DecisionTreeClassifier(max_depth=3, random_state=42)`. Result: mean `0.8137`, standard deviation `0.0187`. Discussed how an unlimited-depth tree can overfit and why depth limits generalization.
- Built and evaluated `random_forest_model`: the same preprocessing pipeline plus `RandomForestClassifier(max_depth=3, random_state=42)`. Result: mean `0.8227`, standard deviation `0.0111`. Explained that a random forest votes across varied trees to reduce instability; it does not automatically find the best depth. The default 100 estimators was used.
- Current conclusion: logistic regression remains the best model of the three on this CV setup. Keep the existing `model` and submission code pointed at logistic regression; do not switch the submission to the forest just because it was tested.
- Began a feature-engineering investigation of `Cabin`. `HasCabin` currently captures only whether a cabin is present. Exploratory deck results among the 204 non-missing cabins were: C 59 / 0.5932, B 47 / 0.7447, D 33 / 0.7576, E 32 / 0.7500, A 15 / 0.4667, F 13 / 0.6154, G 4 / 0.5000, T 1 / 0.0000 (count / survival rate).
- Next tutoring step: interpret the controlled `Deck` experiment, then choose a teaching topic from feature engineering, error analysis, or carefully chosen model hyperparameters. Do not make a sequence of leaderboard-driven changes.

### 2026-10-22 — Session 7
- Explained that `HasCabin` is redundant when `Deck` is `"Unknown"` for missing cabins; raw `Cabin` values could contain cabin-specific detail, but this experiment intentionally keeps only the deck category.
- Temporarily tested `Deck` in place of `HasCabin`: extracted the first character of `Cabin`, mapped G/T to `"Rare"`, used `"Unknown"` for missing values, and dropped the raw `Cabin` column after extraction.
- Evaluated logistic regression using the same five-fold `StratifiedKFold` setup. Fold accuracies: `[0.84357542, 0.82584270, 0.79775281, 0.83146067, 0.84269663]`; mean `0.8283`, standard deviation `0.0167`. Compared with the `HasCabin` baseline (mean `0.8283`, standard deviation `0.0119`), the mean was unchanged and fold variation increased.
- The learner concluded there was no meaningful change. Restored `HasCabin` in `main.ipynb` and archived the executed Deck version as `main_deck_experiment.ipynb`.
- Re-executed the restored baseline notebook in a temporary working directory so its outputs reflect `HasCabin`; the existing project `submission.csv` was left untouched.
- The learner ran error analysis on the fixed validation split. All 17 false negatives were male; 10 of 11 false positives were female, and 8 of those 10 were third class. In the validation split, the false-negative rate was 70.8% among male survivors versus 0% among female survivors; the false-positive rate was 62.5% among female non-survivors versus 1.1% among male non-survivors. Discussed a `Sex` × `Pclass` interaction as one targeted hypothesis.
- Learner preference: give code and exact placement in chat so they can enter and run it themselves. Treat this as the default workflow for notebook changes; do not edit the notebook unless they explicitly request it.
- Next tutoring step: review the completed `SexPclass` experiment and decide with the learner whether to retain it. Do not make a sequence of leaderboard-driven changes.

### 2026-10-22 — Session 8
- The learner added `SexPclass = Sex + "_Pclass_" + str(Pclass)` in `make_features` and included it in `categorical_features`. They ran the notebook cells themselves; preserve that learner-led workflow.
- Logistic-regression five-fold CV using the same `StratifiedKFold` folds: `[0.82122905, 0.83707865, 0.82584270, 0.84269663, 0.84269663]`, mean `0.8339`, standard deviation `0.0088`. The prior `HasCabin` baseline was mean `0.8283`, standard deviation `0.0119`; the interaction improved four of five folds and the mean by about `0.0056`. Treat this as a modest, promising improvement, not proof of a universally better model.
- Latest fixed validation result after rerunning all cells: accuracy `0.83799` (150/179), confusion matrix `[[100, 10], [19, 50]]` in sklearn's `[[TN, FP], [FN, TP]]` layout.
- Latest validation error breakdown: among 45 female survivors, 3 were false negatives (6.7%); among 16 female non-survivors, 7 were false positives (43.8%). Among 24 male survivors, 16 were false negatives (66.7%); among 94 male non-survivors, 3 were false positives (3.2%). This supports the observed Sex-related error pattern, while the interaction did not remove it. Use the refreshed outputs above; earlier error-analysis outputs in the notebook were stale before the learner reran all cells.
- The test set's single missing `Fare` value is handled by the numeric pipeline's median `SimpleImputer`, fit on training data and applied to validation/test data. No separate manual fill is needed; this keeps imputation within the leakage-safe pipeline.
- Continue to provide requested notebook code in chat with the exact insertion point and rerun guidance. Do not edit the learner's notebook unless they explicitly ask.
- Next: discuss whether to keep `SexPclass` given its small CV gain and lower fold-to-fold variation, and help the learner interpret how the interaction changes the decision boundary/errors before proposing another experiment.

## Open questions / not yet decided

- The Deck feature experiment is complete and not adopted. The active notebook
  uses `HasCabin`; the executed Deck run is preserved in
  `main_deck_experiment.ipynb`.
- **Next:** review the completed `SexPclass` experiment and decide with the
  learner whether to retain it. Do not make a sequence of leaderboard-driven
  changes.
- This is a learning project, not a leaderboard-optimization project. Keep
  experiments small, explainable, and measured against the fixed CV setup.

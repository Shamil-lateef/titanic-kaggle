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
- **It's fine to write code when asked directly**, or for scaffolding
  (boilerplate imports, loading data) — the teaching moments are in the
  modeling/analysis decisions, not typing `import pandas as pd`.
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

## Open questions / not yet decided

- Target model/topic depth (e.g. do we go beyond decision trees into
  ensembles, logistic regression, etc.)? Will emerge as we go.
- **In progress:** deciding how to handle `Cabin`, `Name`, and `Ticket`, then
  training and evaluating the first model. k-fold CV vs holdout is still
  genuinely untaught.
- Whether the learner wants to eventually produce a submission.csv for the
  actual Kaggle leaderboard, or this is purely for learning.

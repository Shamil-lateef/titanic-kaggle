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

- Learning ML from scratch, Titanic is their first project.
- Despite being "new," their archived first attempt (see below) shows they
  already independently handled: pandas EDA, visualization, feature
  engineering, and a full train/predict/evaluate loop. So they're not a
  total beginner by the second session — treat foundational pandas/sklearn
  mechanics as known, and focus teaching on the *reasoning* behind modeling
  choices (feature selection, validation strategy, model comparison,
  hyperparameters, avoiding leakage/overfitting).

## Environment

- `train.csv`, `test.csv`, `gender_submission.csv` — standard Kaggle Titanic
  files, in project root.
- `main.ipynb` — the working notebook (reset to empty on 2026-08-22, see log).
- `archive/` — old notebook versions, kept for reference, not edited.

## Progress log

### 2026-08-22 — Reset
- Learner asked to wipe the slate and restart `main.ipynb` from scratch.
- Archived prior work to `archive/main_v1_2026-08-22.ipynb` instead of
  deleting, so nothing is lost.
- Created this CLAUDE.md.
- **What the archived notebook already covered** (useful context, don't
  re-teach from zero if it resurfaces naturally):
  - Loading data with pandas, `.value_counts()`, `.describe()`, `.isnull().sum()`
  - Visualization: matplotlib/seaborn bar plots for `Sex`/`Pclass`/`Embarked`,
    histograms of `Age`/`Fare` split by `Survived`
  - Feature engineering: extracted `Title` from `Name` via regex
  - Encoding: mapped `Sex` to 0/1, used `pd.get_dummies` for categoricals
  - Missing data: filled `Age` using per-`Title` group medians
  - Dropped `Cabin`, `Embarked`, `Name`, `Ticket`, `PassengerId` before modeling
  - `train_test_split`, trained a `DecisionTreeClassifier(max_depth=3)`
  - Checked train accuracy, feature importances, and ran `cross_val_score`
  - Was mid-exploration of `Embarked` vs `Pclass` crosstab when it stopped
- Fresh empty `main.ipynb` created, ready for session 2 to begin.
- **Next up:** ask the learner where they want to start — rebuild from memory
  as a stronger v2, or explore something new (e.g. other models, better
  validation, a real train/test-split-then-predict-on-test.csv submission
  flow, which the v1 notebook never reached).

### 2026-08-24 — Session 2: EDA restart, validation-strategy discussion
- Decided fresh start, structure can echo v1, goal is learning first,
  submission is a stretch goal only if progressing well (see Decisions).
- Only one Python on the machine: 3.12.6 at
  `C:\Users\Shamil\AppData\Local\Programs\Python\Python312\python.exe`, no
  project venv. Has pandas/numpy/scikit-learn/ipykernel; was missing
  matplotlib/seaborn — learner installing themselves.
- Added scaffolding cell to `main.ipynb`: imports + `pd.read_csv` for both
  train/test + `.shape` check.
- Learner ran `.describe()`, `.info()`, `.isnull().sum()`, `.shape` on both
  train and test.
- Discussed (conceptual, no code yet): why load train+test together early is
  fine but fitting (medians/encoders/scalers) must only ever use train data;
  why a validation split is required here specifically (test.csv has no
  `Survived` column, so it can't be used to self-check accuracy); holdout
  (`train_test_split`) vs k-fold CV (`cross_val_score`) tradeoffs, leaning
  CV for a small ~891-row dataset.
- Learner explicitly deferred actually doing the split to next session.
- Notebook still has only the one scaffolding cell — no EDA code cells
  written yet (EDA so far was run ad hoc by the learner, not saved back).

## Open questions / not yet decided

- Target model/topic depth (e.g. do we go beyond decision trees into
  ensembles, logistic regression, etc.)? Will emerge as we go.
- **Learner's question for next session (asked 2026-08-24, start here):**
  "the timing of doing things" — when should we split train/val? When should
  feature engineering happen relative to that split? Which preprocessing
  steps are safe to do before splitting (don't leak) vs which must happen
  after/fit-on-train-only? Is there a general rule for this, or does it vary
  dataset to dataset? This is a leakage/ordering question — good next topic
  since they just grasped *why* a validation set is needed (see log below)
  but haven't yet touched split mechanics or feature engineering in this
  notebook.

## Decisions

- **2026-08-24:** Learner wants a fresh start in `main.ipynb`, building step
  by step rather than pasting v1 back in — but it's fine if the resulting
  structure ends up similar to v1 (EDA → feature engineering → encoding →
  model → validation). Goal for now is learning, not leaderboard score. A
  real submission.csv is a stretch goal only if they're progressing well —
  don't optimize for it yet, don't let it drive early decisions (e.g. no
  need to rush to test.csv or worry about leaderboard-safe preprocessing
  until they say so).

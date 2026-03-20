# %% Install
!pip install -r requirements.txt

# %% Imports
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

# %% Config
LABEL = "loan_approved"
TOP_N = 3

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

# %% Load & split
df = pd.read_csv(args.data_path)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[LABEL], random_state=42)

# %% Train
predictor = TabularPredictor(
    label=LABEL,
    problem_type="binary",
    eval_metric="accuracy",
    path=f"{args.output_dir}/predictor",
).fit(
    train_data=train_df,
    num_stack_levels=3,
    num_bag_folds=2,
    use_bag_holdout=True,
    holdout_frac=0.2,
    time_limit=3600,
    presets="medium_quality",
    excluded_model_types=["XGB"],
)

# %% Leaderboard — pick top N models
leaderboard = predictor.leaderboard(test_df)
top_models = leaderboard.head(TOP_N)["model"].tolist()
print("Top models:", top_models)

# %% Refit each top model on full data
for model_name in top_models:
    full = model_name + "_FULL"
    out = Path(args.output_dir) / full
    clone = predictor.clone(path=out / "predictor", return_clone=True, dirs_exist_ok=True)
    clone.delete_models(models_to_keep=[model_name])
    clone.refit_full(model=model_name)
    clone.set_model_best(model=full, save_trainer=True)
    clone.save_space()
    metrics = clone.evaluate(test_df)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / "metrics.json").write_text(json.dumps(metrics))
    print(f"  {full}: {metrics}")

# %% Clone best model for deployment
best = max(
    top_models,
    key=lambda m: json.loads(
        (Path(args.output_dir) / (m + "_FULL") / "metrics" / "metrics.json").read_text()
    )["accuracy"],
)
print(f"Best model: {best}_FULL")
TabularPredictor.load(str(Path(args.output_dir) / (best + "_FULL") / "predictor")).clone_for_deployment(
    path=f"{args.output_dir}/deployment_predictor"
)
print(f"Deployment clone saved to: {args.output_dir}/deployment_predictor")

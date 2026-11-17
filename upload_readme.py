"""
upload_readme.py - attaches the final README.md (and the disagreements file)
to the latest compare_sentiment run in MLflow, so the logged README matches
the finished one in the repo. Run it after editing the README:

    python upload_readme.py
"""

import mlflow
from mlflow import MlflowClient

import config

mlflow.set_tracking_uri(config.TRACKING_URI)

runs = mlflow.search_runs(
    experiment_names=[config.EXPERIMENT_NAME],
    filter_string=f"tags.mlflow.runName = '{config.COMPARE_RUN_NAME}'",
    order_by=["start_time DESC"],
    max_results=1,
    output_format="list",
)
if not runs:
    raise SystemExit("No compare run found. Run evaluate.py first.")

run_id = runs[0].info.run_id
client = MlflowClient()
for path in ["README.md", "outputs/disagreements.csv", "outputs/new_data_predictions.csv"]:
    client.log_artifact(run_id, path)
    print(f"Uploaded {path} to run {config.COMPARE_RUN_NAME} ({run_id})")

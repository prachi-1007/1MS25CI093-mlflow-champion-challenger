"""
register_champion.py - registers both logged models in the MLflow Model
Registry and sets the aliases:
    @champion   -> the winner chosen by evaluate.py's rule
    @challenger -> the other model

Runs are found by their NAME (not a pasted run ID), so this works on any
machine.

Usage:
    python register_champion.py          # register both models + set aliases
    python register_champion.py --flip   # stretch goal: swap the two aliases
"""

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

import config


def latest_run(run_name):
    """Return the most recent run with this name in our experiment."""
    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"No run named '{run_name}' found. Run evaluate.py first.")
    return runs[0]


def flip_aliases(client):
    """Swap champion and challenger (used to show the predict script adapts)."""
    name = config.REGISTERED_MODEL_NAME
    champ = client.get_model_version_by_alias(name, "champion")
    chall = client.get_model_version_by_alias(name, "challenger")
    client.set_registered_model_alias(name, "champion", chall.version)
    client.set_registered_model_alias(name, "challenger", champ.version)
    print(f"Flipped: @champion -> version {chall.version} "
          f"({chall.tags.get('hf_model_id')})")
    print(f"         @challenger -> version {champ.version} "
          f"({champ.tags.get('hf_model_id')})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip", action="store_true",
                        help="swap the champion and challenger aliases")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME

    if args.flip:
        flip_aliases(client)
        return

    # 1. Read the decision that evaluate.py logged in the summary run
    compare_run = latest_run(config.COMPARE_RUN_NAME)
    winner = compare_run.data.tags["winner"]                  # "A" or "B"
    reason = compare_run.data.tags.get("decision_reason", "")
    loser = "B" if winner == "A" else "A"

    # 2. Register both models (A first, then B)
    versions = {}
    for key in ["A", "B"]:
        run = latest_run(config.run_name(key))
        model_uri = run.data.tags["model_uri"]
        hf_id = run.data.params["hf_model_id"]
        mv = mlflow.register_model(model_uri, name)
        client.set_model_version_tag(name, mv.version, "hf_model_id", hf_id)
        client.set_model_version_tag(name, mv.version, "candidate", key)
        client.set_model_version_tag(name, mv.version, "run_name", config.run_name(key))
        client.set_model_version_tag(
            name, mv.version, config.PRIMARY_METRIC,
            f"{run.data.metrics[config.PRIMARY_METRIC]:.4f}")
        versions[key] = mv
        print(f"Registered model {key} ({hf_id}) as {name} version {mv.version}")

    # 3. Aliases
    client.set_registered_model_alias(name, "champion", versions[winner].version)
    client.set_registered_model_alias(name, "challenger", versions[loser].version)
    client.set_model_version_tag(name, versions[winner].version, "role", "champion")
    client.set_model_version_tag(name, versions[loser].version, "role", "challenger")

    # 4. Attach the final README and disagreements file to the summary run
    #    (so the README with your written decision is logged in MLflow too)
    for path in ["README.md", "outputs/disagreements.csv"]:
        if Path(path).exists():
            client.log_artifact(compare_run.info.run_id, path)

    print("\n========== REGISTRY STATUS ==========")
    print(f"Registered model : {name}")
    print(f"@champion        : version {versions[winner].version} "
          f"({config.MODELS[winner]['hf_model_id']})")
    print(f"@challenger      : version {versions[loser].version} "
          f"({config.MODELS[loser]['hf_model_id']})")
    print(f"Decision reason  : {reason}")
    print(f"Champion URI     : models:/{name}@champion")
    print("\nNext: python predict_from_mlflow.py")


if __name__ == "__main__":
    main()

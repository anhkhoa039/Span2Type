import mlflow
import pandas as pd
from mlflow.entities import ViewType

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

try:
    # List all experiments
    experiments = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    print(f"Found {len(experiments)} experiments.")
    
    if not experiments:
        print("No experiments found.")
        exit()

    experiment_ids = [e.experiment_id for e in experiments]
    
    # Search for the absolute latest run across all experiments
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["attribute.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        print("No runs found in any experiment.")
    else:
        run = runs.iloc[0]
        print(f"Latest Run ID: {run['run_id']}")
        print(f"Experiment ID: {run['experiment_id']}")
        print(f"Status: {run['status']}")
        print(f"Start Time: {run['start_time']}")
        print(f"Artifact URI: {run['artifact_uri']}")
        
        print("\n--- Parameters ---")
        for col in run.index:
            if col.startswith("params.") and run[col] is not None:
                print(f"{col[7:]}: {run[col]}")
        
        print("\n--- Metrics ---")
        for col in run.index:
            if col.startswith("metrics.") and run[col] is not None:
                print(f"{col[8:]}: {run[col]}")

except Exception as e:
    print(f"An error occurred: {e}")


import argparse
import pandas as pd
from pathlib import Path
RUN_ID = "run_id"
DISTORTION = "distortion"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process_results.py")
    parser.add_argument("--file", required=True, help="Path to results file")
    args = parser.parse_args()

    data = pd.read_csv(args.file)

    # remove run_number from run
    data[RUN_ID] = data[RUN_ID].map(lambda x: x[:-2])

    grouped = data.groupby(RUN_ID)
    means = grouped.mean()
    stds = grouped.std().rename(lambda x: x + "_std", axis="columns")

    means_and_stds = means.join(stds)
    means_and_stds = means_and_stds.sort_values(by=[DISTORTION], ascending=True)

    path = Path(args.file)
    new_path = path.parent / f"AA-{path.name}.csv"
    means_and_stds.to_csv(str(new_path))


import argparse
import pandas as pd
from pathlib import Path
RUN_ID = "run_id"
DISTORTION = "distortion"


def process_one_result_file(file_path):
    """Process all the configurations (grid-search) run for one model and returns the DataFrame row of
    only the best performing one.

    Columns like "manifold" or "data" are assumed to be the same for the entire file.
    """
    data = pd.read_csv(file_path)

    data = data.drop(columns="timestamp")
    # remove run_number from run
    data[RUN_ID] = data[RUN_ID].map(lambda x: x[:-2] if x[-1].isdigit() and x[-2] == "-" else x)

    # these values should be the same for the entire file
    graph, manifold = data["data"][0], data["manifold"][0]

    grouped = data.groupby(RUN_ID)
    means = grouped.mean().drop(columns="Unnamed: 0")
    stds = grouped.std()
    stds = stds.drop(columns="dims").drop(columns="Unnamed: 0")
    stds = stds.rename(lambda x: x + "_std", axis="columns")

    means_and_stds = means.join(stds)
    min_id = means_and_stds.idxmin()
    best_distortion = means_and_stds.loc[min_id["distortion"]]    # returns series
    best_distortion = best_distortion.append(pd.Series(graph, index=["data"]))
    best_distortion = best_distortion.append(pd.Series(manifold, index=["manifold"]))
    return best_distortion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process_results.py")
    parser.add_argument("--folder", default="out", required=False, help="Path to folder with result files")
    args = parser.parse_args()

    folder = Path(args.folder)
    best_results = []
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        best_results.append(process_one_result_file(file_path))

    best_results = pd.DataFrame(best_results)
    new_path = folder / f"AA-best-results.csv"
    best_results.to_csv(str(new_path))

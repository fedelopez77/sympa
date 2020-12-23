
import argparse
import pandas as pd
from pathlib import Path
RUN_ID = "run_id"
HR = "HR@10"


def process_one_result_file(file_path):
    """Process all the configurations (grid-search) run for one model and returns the DataFrame row of
    only the best performing one.

    Columns "dims", "manifold" and "data" are assumed to be the same for the entire file.
    """
    data = pd.read_csv(file_path)

    try:
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
        max_id = means_and_stds.idxmax()
        best_hr = means_and_stds.loc[max_id[HR]]    # returns series
        best_hr = best_hr.append(pd.Series(graph, index=["data"]))
        best_hr = best_hr.append(pd.Series(manifold, index=["manifold"]))
        best_hr = best_hr.append(pd.Series(max_id[HR], index=["run_id"]))
        best_hr["dims"] = int(best_hr["dims"])
        return best_hr.reindex(index=["dims", "data", "manifold", HR, "HR@10_std", "nDCG@10", "nDCG@10_std", "run_id"])
    except:
        raise ValueError(f"Error with file: {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="rsprores.py")
    parser.add_argument("--folder", default="out/recosys", required=False, help="Path to folder with result files")
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

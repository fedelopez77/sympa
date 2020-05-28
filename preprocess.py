import argparse
import networkx as nx
import torch
import sympa.utils as utils
import sympa.config as config

log = utils.get_logging()


def get_graph(args):
    if args.graph == "grid":
        nodes = int(args.grid_nodes ** 0.5)
        graph = nx.grid_2d_graph(nodes, nodes)
        graph.name = f"grid_{nodes}x{nodes}"
    elif args.graph == "tree":
        graph = nx.balanced_tree(args.tree_branching, args.tree_height)
        graph.name = f"tree_branch{args.tree_branching}_height{args.tree_height}"
    else:
        raise ValueError(f"--graph={args.graph} not recognized")
    return graph


def plot_graph(graph, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(graph, iterations=100)
    fig = plt.figure()
    nx.draw(graph, pos, ax=fig.add_subplot(111), node_size=5, with_labels=True, label=graph.name)
    plt.savefig(path / (graph.name + ".png"))


def get_distances(graph):
    nodes = list(graph.nodes())
    dists = torch.zeros((len(nodes), len(nodes)), dtype=torch.float)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            d = nx.shortest_path_length(graph, src, dst)
            dists[i, j] = d
    return dists


def main():
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--run_id", required=True, help="Id of run to store data")
    parser.add_argument("--graph", default="grid", help="Graph type: grid or tree")
    parser.add_argument("--grid_nodes", default=36, type=int,
                        help="if --graph=grid it will create a grid of n x n with n = int(sqrt(grid_nodes))")
    parser.add_argument("--tree_branching", default=3, type=int, help="if --graph=tree, branching factor of tree")
    parser.add_argument("--tree_height", default=3, type=int, help="if --graph=tree, height of tree")

    args = parser.parse_args()
    utils.set_seed(42)

    # creates storage directory
    config.CKPT_PATH.mkdir(parents=True, exist_ok=True)
    config.PREP_PATH.mkdir(parents=True, exist_ok=True)
    run_path = config.PREP_PATH / args.run_id
    run_path.mkdir(parents=True, exist_ok=True)

    graph = get_graph(args)
    plot_graph(graph, run_path)

    distances = get_distances(graph)
    id2node = {i: node for i, node in enumerate(graph.nodes())}

    torch.save(
        {
            "distances": distances,
            "id2node": id2node
        },
        run_path / config.PREPROCESSED_FILE
    )


if __name__ == "__main__":
    main()

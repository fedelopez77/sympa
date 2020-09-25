import argparse
import networkx as nx
from networkx.generators import expanders, social
import torch
import sympa.utils as utils
import sympa.config as config

log = utils.get_logging()


def get_graph(args):
    if args.graph == "grid":
        dims = args.grid_dims
        nodes = round(args.nodes ** (1 / dims))
        shape = [nodes] * dims
        graph = nx.grid_graph(dim=shape)
        graph.name = f"grid{dims}d_{args.nodes}"
    elif args.graph == "tree":
        graph = nx.balanced_tree(args.tree_branching, args.tree_height)
        graph.name = f"tree_branch{args.tree_branching}_height{args.tree_height}"

    # expanders
    elif args.graph == "expander-margulis":
        graph = expanders.margulis_gabber_galil_graph(args.nodes)
        graph.name = f"expander-margulis-{args.nodes}"
    elif args.graph == "expander-chordal":
        if not utils.is_prime(args.nodes):
            raise ValueError(f"args.nodes must be prime for {args.graph} graph")
        graph = expanders.chordal_cycle_graph(args.nodes)
        graph.name = f"expander-chordal-{args.nodes}"
    elif args.graph == "expander-paley":
        if not utils.is_prime(args.nodes):
            raise ValueError(f"args.nodes must be prime for {args.graph} graph")
        graph = expanders.paley_graph(args.nodes)
        graph.name = f"expander-paley-{args.nodes}"

    # social networks
    elif args.graph == "social-karate":
        graph = social.karate_club_graph()
        graph.name = f"social-karate"
    elif args.graph == "social-davis":
        graph = social.davis_southern_women_graph()
        graph.name = f"social-davis"
    elif args.graph == "social-florentine":
        graph = social.florentine_families_graph()
        graph.name = f"social-florentine"
    elif args.graph == "social-miserables":
        graph = social.les_miserables_graph()
        graph.name = f"social-miserables"

    # graph products
    elif args.graph == "product-cartesian":
        dims = args.grid_dims
        nodes = round(args.nodes ** (1 / dims))
        shape = [nodes] * dims
        grid = nx.grid_graph(dim=shape)
        tree = nx.balanced_tree(args.tree_branching, args.tree_height)
        graph = nx.cartesian_product(tree, tree)
        graph.name = f"product-cartesian"
    elif args.graph == "product-rooted":
        dims = args.grid_dims
        nodes = round(args.nodes ** (1 / dims))
        shape = [nodes] * dims
        grid = nx.grid_graph(dim=shape)
        tree = nx.balanced_tree(args.tree_branching, args.tree_height)
        # if invoked rooted_product(tree, grid, list(grid.nodes())[0]), it gives a tree of grids
        # if invoked rooted_product(grid, tree, list(tree.nodes())[0]), it gives a grid with trees hanging
        graph = nx.algorithms.operators.rooted_product(tree, grid, list(grid.nodes())[0])
        graph.name = f"product-rooted"
    else:
        graph = load_graph(args)
    return graph


def load_graph(args):
    path = f"data/{args.graph}/{args.graph}.edges"
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2 or (len(line) > 2 and not line[2].replace(".", "", 1).isdigit()):
                graph.add_edge(line[0], line[1])
            else:
                graph.add_edge(line[0], line[1], weight=float(line[2]))
    return graph


def plot_graph(graph, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(graph, iterations=100)
    fig = plt.figure()
    nx.draw(graph, pos, ax=fig.add_subplot(111), node_size=5, with_labels=True, label=graph.name)
    plt.savefig(path / (graph.name + ".png"))


def build_triplets(graph, node2id):
    """
    Builds triplets of (src, dst, distance) for each node in the graph, to all other connected nodes.

    PRE: the distances in the graph are symmetric.
    :param graph:
    :param node2id:
    :return: set of triplets
    """
    if nx.is_weighted(graph):
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(graph))

    triplets, pairs = set(), set()
    for src, reachable_nodes in lengths.items():
        for dst, distance in reachable_nodes.items():
            if distance > 0:
                src_id = node2id[src]
                dst_id = node2id[dst]
                if (dst_id, src_id) not in pairs:  # checks that the symmetric triplets is not there
                    pairs.add((src_id, dst_id))
                    triplets.add((src_id, dst_id, distance))
    return triplets


def main():
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--run_id", required=True, help="Id of run to store data")
    parser.add_argument("--graph", default="grid", help="Graph type")
    parser.add_argument("--nodes", default=10, type=int,
                        help="if --graph=grid it will create a grid of dims dimensions with n = int(nodes^(1/dims))")
    parser.add_argument("--grid_dims", default=3, type=int, help="if --graph=grid, number of dimensions")
    parser.add_argument("--tree_branching", default=3, type=int, help="if --graph=tree, branching factor of tree")
    parser.add_argument("--tree_height", default=3, type=int, help="if --graph=tree, height of tree")
    parser.add_argument("--plot_graph", default=0, type=int, help="Whether to generate a plot or not")

    args = parser.parse_args()
    utils.set_seed(42)

    # creates storage directory
    config.CKPT_PATH.mkdir(parents=True, exist_ok=True)
    config.TENSORBOARD_PATH.mkdir(parents=True, exist_ok=True)
    config.PREP_PATH.mkdir(parents=True, exist_ok=True)
    run_path = config.PREP_PATH / args.run_id
    run_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Building graph: {args.graph}")
    graph = get_graph(args)
    log.info(nx.info(graph))
    if args.plot_graph == 1:
        log.info("Plotting graph")
        plot_graph(graph, run_path)

    nodes = list(graph.nodes())
    id2node = {i: node for i, node in enumerate(nodes)}
    node2id = {v: k for k, v in id2node.items()}
    log.info(f"Building triplets for {len(nodes)} nodes")
    triplets = build_triplets(graph, node2id)
    log.info(f"Total triplets: {len(triplets)}")

    log.info(f"Saving to {run_path / config.PREPROCESSED_FILE}")
    torch.save(
        {
            "triplets": triplets,
            "id2node": id2node
        },
        run_path / config.PREPROCESSED_FILE
    )


if __name__ == "__main__":
    main()

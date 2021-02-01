import argparse
import os
import sys
from datetime import datetime

import networkx as nx
import networkit as nk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import math

matplotlib.rcParams.update({'font.size': 20})


def log(msg):
    timestamp = str(datetime.now()).split(".")[0]
    print(f"{timestamp} {msg}")


def main():
    set_seeds(args.random_seed)

    # Load the graph and sub-sample the nodes.
    log(f"Loading graph from {args.input}")
    g = nx.read_gexf(args.input)
    if args.nodes_sampling_percentage < 0.99:
        log(f"Subsampling graph to {args.nodes_sampling_percentage}")
        g = subsample(g)

    log("Loading properties on nodes and edges")
    nodes = [n[0] for n in g.degree]
    edges = list(g.edges())
    degrees = np.array([min(n[1], 500) for n in g.degree])
    edge_curvs = np.zeros(shape=(len(edges)))
    node_curvs = np.ones(shape=(len(nodes))) * math.pi / 8
    if args.edge_property:
        for i, (_, _, attrs) in enumerate(g.edges(data=True)):
            edge_curvs[i] = attrs[args.edge_property]

    if args.node_property:
        for i, (_, attrs) in enumerate(g.nodes(data=True)):
            node_curvs[i] = attrs[args.node_property]

    # Show negative and positive curvatures with diverging color scheme:
    # make sure we normalize them so that transparent edges correspond to 0.
    def normalize_between(curvs, indices, a, b):
        x = curvs[indices]
        if len(np.unique(x)) > 1:
            x = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
            curvs[indices] = x

    # log("Copying and normalizing curvatures")
    # orig_edge_curvs = edge_curvs.copy()
    # normalize_between(edge_curvs, np.where(edge_curvs > 0), 0.5, 1.0)
    # normalize_between(edge_curvs, np.where(edge_curvs <= 0), 0, 0.5)
    # orig_node_curvs = node_curvs.copy()
    # normalize_between(node_curvs, np.where(node_curvs > 0), 0.5, 1.0)
    # normalize_between(node_curvs, np.where(node_curvs <= 0), 0, 0.5)

    log("Setting up image")
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])

    # curv_cmap = matplotlib.cm.get_cmap('RdBu')
    curv_cmap = sns.color_palette("viridis_r", as_cmap=True)

    log("Mapping weights to float")
    float_distances = {(a, b): 1.0 for a, b, w in g.edges.data("weight")}
    nx.set_edge_attributes(g, float_distances, "f_weight")

    log("Calculating shortest distances")
    gk = nk.nxadapter.nx2nk(g)
    shortest_paths = nk.distance.APSP(gk).run().getDistances()

    log("Mapping distances to dict...")
    dists = {u: {} for u in g.nodes()}
    for i, u in enumerate(g.nodes()):
        for j, v in enumerate(g.nodes()):
            dists[u][v] = int(shortest_paths[i][j])

    log("Calculating layout...")
    layout = nx.drawing.layout.kamada_kawai_layout(g, dist=dists, scale=2.0, weight="f_weight")
    kwargs = {
            'pos': layout,
            'with_labels': False,
            'nodelist': nodes,
            'edgelist': edges,
            'ax': ax,

            # node settings
            # -> no scaling for facebook and wormnet
            # -> 2 * d**1.5 for web-edu, grqc
            # -> 2 * d**3 for road-minnesota
            # -> 2 * d**2 for the others
            'node_size': 2000, #2 * degrees**2,
            'cmap': curv_cmap,
            'vmin': node_curvs.min(), # 0.0,
            'vmax': node_curvs.max(), #math.pi / 4,
            'node_color': node_curvs,
            'linewidths': 0.01,
            'edgecolors': 'k',

            # edge settings
            'edge_cmap': curv_cmap,
            'edge_vmin': edge_curvs.min(), # 0.0,
            'edge_vmax': edge_curvs.max(), #math.pi / 4,
            'edge_color': edge_curvs,
            'width': 20.0
    }
    log("Drawing network")
    drawing = nx.draw_networkx(g, **kwargs)
    ax.axis('off')

    log("Configuring ticks, etc")
    # Plot the curvature colorbar separately.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    import matplotlib.colors as colors
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=curv_cmap, orientation='vertical')
    # cb.vmax = math.pi / 4
    cb.set_ticks([])
    #
    # cb.set_ticks([0, 0.5, 1.0])
    # curv_ticks = [0, math.pi / 8, math.pi / 4]
    # # We have to do this because of the curvature hack from above
    # cb.set_ticklabels(floats_to_str(curv_ticks))
    #
    # # cb.outline.set_linewidth(0)
    # cb.ax.xaxis.set_tick_params(width=10)

    # cb.set_label('Edge vector angle')

    # Save it
    name = args.input.split("/")[-1].replace('gexf', 'pdf')
    log("Saving figure")
    fig.savefig("plots/pdfs/" + name, bbox_inches='tight')


def subsample(g):
    n = g.number_of_nodes()
    degrees = np.asarray(list(dict(g.degree()).values()))
    ps = degrees / degrees.sum()
    nodes_to_keep = np.random.choice(
            n, int(n * args.nodes_sampling_percentage), replace=False, p=ps)
    g = g.subgraph(nodes_to_keep).copy()
    print('#nodes: {}, #edges: {}, #conn-comps: {}'.format(
            g.number_of_nodes(), g.number_of_edges(),
            nx.number_connected_components(g)))
    if args.keep_largest_component:
        g = g.subgraph(max(nx.connected_components(g), key=len))
        print('Keeping the largest component: #nodes: {}, #edges: {}'.format(
                g.number_of_nodes(), g.number_of_edges()))
    return g


def floats_to_str(floats):
    return ['{:.2f}'.format(f) for f in floats]


def set_seeds(seed):
    import numpy
    import random
    numpy.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
            description='Plot graphs annotated with curvature information')
    parser.add_argument('--input', type=str, required=False, default="graphs/upper-fone2d-grid2d-81.gexf",
                        help='The input graph file.')
    parser.add_argument('--node_property', type=str, default='angle',
                        help='The name of the property in the graph to read for nodes.')
    parser.add_argument('--edge_property', type=str, default='angle',
                        help='The name of the property in the graph to read for edges.')
    parser.add_argument('--nodes_sampling_percentage', type=float, default=1.0, help='The percentage of nodes to keep.')
    parser.add_argument('--keep_largest_component', action='store_true', help='When sub-sampling the nodes, keep the largest connected component only.')
    parser.add_argument('--random_seed', type=int, default=42, help='The manual random seed.')
    args = parser.parse_args()
    if args.keep_largest_component and args.nodes_sampling_percentage > 0.99:
        raise ValueError('The option `keep_largest_component` is meaningless when not doing node subsampling')
    return args


if __name__ == '__main__':
    global args
    args = parse_args()
    sys.exit(main())

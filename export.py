import argparse
import torch
import sympa.config as config


def main():
    parser = argparse.ArgumentParser(description="export.py")
    parser.add_argument("--load_model", required=True, help="Path to model to load")
    parser.add_argument("--manifold", default="upper", type=str, help="Name of manifold used in the run")
    parser.add_argument("--matplot", default=1, type=int,
                        help="If matplot=1 it exports a matplot image. If not, it exports the coords and metadata to"
                             "be plot in projector.tensorflow.org")

    config.PLOT_EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    data = torch.load(args.load_model)

    model, id2node = data["model"], data["id2node"]
    embeds = model["embeddings.embeds"]

    if args.matplot == 1:
        if args.manifold == "upper" or args.manifold == "bounded":
            # Takes the entry z_0,0 of each matrix point, and splits it into real and imaginary part
            # to represent x, y respectively
            x = embeds[:, 0, 0, 0].unsqueeze(-1)
            y = embeds[:, 1, 0, 0].unsqueeze(-1)
            embeds = torch.cat((x, y), dim=-1)
        plot(id2node, embeds, args)
        return

    export_for_tensorflow_projector(id2node, embeds, args)


def plot(id2nodes, embeds, args):
    """
    Plots the embeddings using networkx and matplotlib

    :param id2nodes: dict of {node_id: node_name}. This node_id is used to index the embeds table
    :param embeds: torch tensor of shape (num_nodes, 2) with 2 dimensions to plot the point
    :param args: namespace with extra args
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(list(id2nodes.values()))

    pos = {}
    for idx, node in id2nodes.items():
        pos[node] = embeds[idx].numpy()

    fig = plt.figure()
    nx.draw(g, pos, ax=fig.add_subplot(111), node_size=5, with_labels=True)

    model_name = args.load_model.split("/")[-1]
    file_name = config.PLOT_EXPORT_PATH / f"{model_name}.png"
    print(f"Exporting to {file_name}")
    plt.savefig(file_name)


def export_for_tensorflow_projector(id2node, embeds, args):
    """
    Export tsv files with coordinates and metadata to be projected with the Tensorflow Projector
    http://projector.tensorflow.org/

    :param id2nodes: dict of {node_id: node_name}. This node_id is used to index the embeds table
    :param embeds: torch tensor of shape (num_nodes, dims) where dims can be >= 2
    :param args: namespace with extra args
    """
    meta, coords = [], []
    for i, embed in enumerate(embeds):
        meta.append(str(id2node[i]))
        str_coords = "\t".join([f"{x:.6f}" for x in embed.tolist()])
        coords.append(str_coords)
    model_name = args.load_model.split("/")[-1]
    coord_path = config.PLOT_EXPORT_PATH / f"{model_name}-coords.tsv"
    meta_path = config.PLOT_EXPORT_PATH / f"{model_name}-meta.tsv"
    write_file(coord_path, coords)
    write_file(meta_path, meta)


def write_file(path, data):
    with open(path, "w") as f:
        f.write("\n".join(data))


if __name__ == "__main__":
    main()


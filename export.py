import argparse
import torch
import sympa.config as config


def main():
    parser = argparse.ArgumentParser(description="export.py")
    parser.add_argument("--load_model", required=True, help="Path to model to load")
    parser.add_argument("--matplot", default=1, type=int,
                        help="If matplot=1 it exports a matplot image. If not, it exports the coords and metadata to"
                             "be plot in projector.tensorflow.org")

    config.PLOT_EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    data = torch.load(args.load_model)

    model, id2node = data["model"], data["id2node"]
    embeds = model["embeddings.embeds"]

    if args.matplot == 1:
        plot(id2node, embeds, args)
        return

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


def plot(id2nodes, embeds, args):
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
    plt.savefig(config.PLOT_EXPORT_PATH / f"{model_name}.png")


if __name__ == "__main__":
    main()

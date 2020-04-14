import argparse
import torch
import sympa.config as config


def main():
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--load_model", required=True, help="Path to model to load")
    parser.add_argument("--matplot", default=1, type=int,
                        help="If matplot=1 it exports a matplot image. If not, it exports the coords and metadata to"
                             "be plot in projector.tensorflow.org")

    args = parser.parse_args()
    data = torch.load(args.load_model)

    model, id2node = data["model"], data["id2node"]
    embeds = model["embeddings.embeds"]

    if args.matplot == 1:
        plot(embeds, args)
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


def plot(embeds, args):
    import matplotlib.pyplot as plt

    x = embeds[:, 0].numpy()
    y = embeds[:, 1].numpy()

    plt.scatter(x, y)

    model_name = args.load_model.split("/")[-1]
    plt.savefig(config.PLOT_EXPORT_PATH / f"{model_name}.png")


if __name__ == "__main__":
    main()

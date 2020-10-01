import argparse
import random
import torch
from geoopt.optim import RiemannianSGD
from sympa import config
from sympa.utils import set_seed, get_logging, scale_triplets, subsample_triplets
from sympa.runner import Runner
from sympa.model import Model


log = get_logging()


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Name of prep folder")
    parser.add_argument("--run_id", required=True, type=str, help="Name of model/run to export")
    # Model
    parser.add_argument("--model", default="upper", type=str, help="Model type: 'euclidean', 'poincare', "
                                                                       "'upper' or 'bounded'")
    parser.add_argument("--dims", default=3, type=int, help="Dimensions for the model.")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")
    parser.add_argument("--scale_coef", default=10, type=float, help="Coefficient to divide scale.")
    parser.add_argument("--train_scale", default=0, type=int, help="Whether to train scaling or not.")
    # optim and config
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Starting learning rate.")
    parser.add_argument("--reduce_factor", default=5, type=float, help="Factor to reduce lr on plateau.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--patience", default=25, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--burnin", default=10, type=int, help="Number of initial epochs to train with reduce lr.")
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of update steps to acum before backward.")
    parser.add_argument("--scale_triplets", default=0, type=int, help="Whether to apply scaling to triplets or not")
    parser.add_argument("--subsample", default=-1, type=float, help="Subsamples the % of closest triplets")

    # Others
    parser.add_argument("--load_model", default="", type=str, help="Load model from this file")
    parser.add_argument("--results_file", default="out/results.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--save_epochs", default=10001, type=int, help="Exports every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")


def get_model(args):
    model = Model(args)
    if args.load_model:
        saved_data = torch.load(args.load_model)
        model.load_state_dict(saved_data["model"])
    model.to(config.DEVICE)
    return model


def get_scheduler(optimizer, args):
    patience = round(args.patience / args.val_every)
    factor = 1 / float(args.reduce_factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)


def load_training_data(args):
    data_path = config.PREP_PATH / f"{args.data}/{config.PREPROCESSED_FILE}"
    log.info(f"Loading data from {data_path}")
    data = torch.load(data_path)
    id2node = data["id2node"]
    all_triplets = list(data["triplets"])
    if args.subsample > 0:
        sub_triplets = subsample_triplets(all_triplets, args.subsample)
    else:
        sub_triplets = all_triplets

    if args.scale_triplets == 1:
        all_triplets = scale_triplets(all_triplets)
        sub_triplets = scale_triplets(sub_triplets)

    train_src_dst_ids = [(src, dst) for src, dst, _ in sub_triplets]
    train_distances = [distance for _, _, distance in sub_triplets]
    train_src_dst_ids = torch.LongTensor(train_src_dst_ids).to(config.DEVICE)
    train_distances = torch.Tensor(train_distances).to(config.DEVICE)

    if args.subsample > 0:
        # train triplets are a subsample, valid triplets are all
        valid_src_dst_ids = [(src, dst) for src, dst, _ in all_triplets]
        valid_distances = [distance for _, _, distance in all_triplets]
        valid_src_dst_ids = torch.LongTensor(valid_src_dst_ids).to(config.DEVICE)
        valid_distances = torch.Tensor(valid_distances).to(config.DEVICE)
        return id2node, train_src_dst_ids, train_distances, valid_src_dst_ids, valid_distances

    # train and validation triplets are the same
    return id2node, train_src_dst_ids, train_distances, train_src_dst_ids, train_distances


def main():
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    log.info(args)

    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    id2node, train_src_dst_ids, train_distances, valid_src_dst_ids, valid_distances = load_training_data(args)

    args.num_points = len(id2node)
    model = get_model(args)
    optimizer = RiemannianSGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=10)
    scheduler = get_scheduler(optimizer, args)

    log.info(f"GPU's available: {torch.cuda.device_count()}")
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    log.info(f"Points: {args.num_points}, dims: {args.dims}, number of parameters: {n_params}")
    log.info(model)

    runner = Runner(model, optimizer, scheduler=scheduler, id2node=id2node, args=args,
                    train_src_dst_ids=train_src_dst_ids, train_distances=train_distances,
                    valid_src_dst_ids=valid_src_dst_ids, valid_distances=valid_distances)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    main()

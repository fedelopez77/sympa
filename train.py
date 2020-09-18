import argparse
import random
import torch
from geoopt.optim import RiemannianSGD
from sympa import config
from sympa.utils import set_seed, get_logging
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
    # optim and config
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Starting learning rate.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--patience", default=25, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=1000.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of training epochs.")
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of update steps to acum before backward.")
    # Others
    parser.add_argument("--results_file", default="out/results.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--save_epochs", default=1001, type=int, help="Exports every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")


def get_model(id2node, args):
    model = Model(id2node, args)
    # TODO: load model if necessary
    model.to(config.DEVICE)
    return model


def get_scheduler(optimizer, args):
    patience = round(args.patience / args.val_every)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)


def main():
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    log.info(args)

    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    data_path = config.PREP_PATH / f"{args.data}/{config.PREPROCESSED_FILE}"
    log.info(f"Loading data from {data_path}")
    data = torch.load(data_path)
    id2node = data["id2node"]
    triplets = torch.LongTensor(list(data["triplets"])).to(config.DEVICE)

    model = get_model(id2node, args)
    optimizer = RiemannianSGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=10)
    scheduler = get_scheduler(optimizer, args)

    log.info(f"GPU's available: {torch.cuda.device_count()}")
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    log.info(f"Points: {len(id2node)}, dims: {args.dims}, number of parameters: {n_params}")
    log.info(model)

    runner = Runner(model, optimizer, scheduler=scheduler, id2node=id2node, triplets=triplets, args=args)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    main()

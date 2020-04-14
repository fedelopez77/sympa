import argparse
import random
import torch
from geoopt.optim import RiemannianSGD
from sympa import config
from sympa.utils import set_seed, get_logging
from sympa.Runner import Runner
from sympa.model import Model


log = get_logging()


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Name of prep folder")
    parser.add_argument("--run_id", required=True, type=str, help="Name of model/run to export")
    # Model
    parser.add_argument("--model", default="upper", type=str, help="Model type: 'euclidean', 'upper' or 'bounded'")
    parser.add_argument("--dims", default=2, type=int, help="Dimensions for the model.")
    # optim and config
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Starting learning rate.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of update steps to acum before backward.")
    # Others
    parser.add_argument("--save_epochs", default=25, type=int, help="Export every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")


def get_model(args):
    model = Model(args)
    model.to(config.DEVICE)
    return model


def main():
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    log.info(args)

    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    # TODO Load data
    data = {
        "train": None,
        "dev": None,
        "test": None
    }

    model = get_model(args)
    optimizer = RiemannianSGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=10)

    log.info(f"GPU's available: {torch.cuda.device_count()}")
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    log.info(f"Number of parameters: {n_params}")
    log.info(model)

    runner = Runner(model, optimizer, train=data["train"], dev=data["dev"], test=data["test"], args=args)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    main()

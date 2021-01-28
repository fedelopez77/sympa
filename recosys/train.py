import os
import sys
sys.path.append(os.path.abspath('../sympa'))
sys.path.append(os.path.abspath('.'))
import argparse
import random
import pickle
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from geoopt.optim import RiemannianSGD
from sympa import config
from sympa.utils import set_seed, get_logging
from recosys.rsmodel import RecoSys
from recosys.rsrunner import Runner


def config_parser(parser):
    # Data options
    parser.add_argument("--prep", required=True, type=str, help="Name of prep folder and file")
    parser.add_argument("--run_id", required=True, type=str, help="Name of model/run to export")
    # Model
    parser.add_argument("--model", default="spd", type=str, help="Model type: 'euclidean', 'poincare', "
                                                                       "'upper' or 'bounded'")
    parser.add_argument("--loss", default="bce", type=str, help="Loss: bce or hinge")
    parser.add_argument("--dims", default=3, type=int, help="Dimensions for the model.")
    parser.add_argument("--train_bias", default=1, type=int, help="Whether to train scaling or not.")
    # optim and config
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Starting learning rate.")
    parser.add_argument("--reduce_factor", default=5, type=float, help="Factor to reduce lr on plateau.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--patience", default=25, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--burnin", default=10, type=int, help="Number of initial epochs to train with reduce lr.")
    parser.add_argument("--grad_accum_steps", default=1, type=int, help="Number of steps to acum before backward.")
    parser.add_argument("--neg_sample_size", default=1, type=int, help="Neg sample to use in loss.")
    parser.add_argument("--hinge_margin", default=1, type=float, help="Margin for hinge loss.")

    # Others
    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--job_id", default=-1, type=int, help="Slurm job id to be logged")
    parser.add_argument("--n_procs", default=4, type=int, help="Number of process to create")
    parser.add_argument("--load_model", default="", type=str, help="Load model from this file")
    parser.add_argument("--results_file", default="out/recosys/results.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--save_epochs", default=10001, type=int, help="Exports every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")


def get_model(args):
    rsmodel = RecoSys(args)
    rsmodel.to(config.DEVICE)
    rsmodel = DistributedDataParallel(rsmodel, device_ids=None)
    if args.load_model:
        saved_data = torch.load(args.load_model)
        rsmodel.load_state_dict(saved_data["model"])
    return rsmodel


def get_scheduler(optimizer, args):
    patience = round(args.patience / args.val_every)
    factor = 1 / float(args.reduce_factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, mode="max")


def load_data(args, log):
    data_path = config.PREP_PATH / f"recosys/prep/{args.prep}/{args.prep}.pickle"
    log.info(f"Loading data from {data_path}")
    data = pickle.load(open(str(data_path), "rb"))

    # splits    # train = data["train"] if not args.debug else data["train"][:100]
    train = TensorDataset(torch.LongTensor(data['train']).to(config.DEVICE))
    dev_data = data['dev'] if len(data['dev']) < 10000 else data['dev'][:10000]
    dev = TensorDataset(torch.LongTensor(dev_data).to(config.DEVICE))
    test = TensorDataset(torch.LongTensor(data['test']).to(config.DEVICE))

    batch_size = args.batch_size // args.n_procs
    log.info(f"Batch size {batch_size} for {args.local_rank}/{args.n_procs} processes")
    train_sampler = DistributedSampler(train, num_replicas=args.n_procs, rank=args.local_rank)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0,
                              pin_memory=True, sampler=train_sampler)

    dev_loader = DataLoader(dev, sampler=SequentialSampler(dev), batch_size=args.eval_batch_size)
    test_loader = DataLoader(test, sampler=SequentialSampler(test), batch_size=args.eval_batch_size)

    return train_loader, dev_loader, test_loader, data["samples"], data


def get_quantities(data):
    n_users = len(data["id2uid"])
    n_items = len(data["id2iid"])
    return n_users, n_items, n_users + n_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    # sets random seed
    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    log = get_logging()
    if args.local_rank == 0:
        log.info(args)
        log.info(f"Job ID: {args.job_id}")

    dist.init_process_group(backend=config.BACKEND, init_method='env://') # world_size=args.n_procs, rank=args.local_rank)

    # correct parameters due to distributed training
    args.learning_rate *= args.n_procs

    train, dev, test, samples, data = load_data(args, log)
    n_users, n_items, n_entities = get_quantities(data)
    args.num_points = n_entities
    args.n_items = n_items

    rsmodel = get_model(args)
    optimizer = RiemannianSGD(rsmodel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, stabilize=10)
    scheduler = get_scheduler(optimizer, args)

    if args.local_rank == 0:
        log.info(f"GPU's available: {torch.cuda.device_count()}")
        n_params = sum([p.nelement() for p in rsmodel.parameters() if p.requires_grad])
        log.info(rsmodel)
        log.info(f"Users: {n_users}, Items: {n_items}, dims: {args.dims}, number of parameters: {n_params}")
        log.info(f"Data train: {len(train.dataset)}, dev: {len(dev.dataset)}, test: {len(test.dataset)}")

    runner = Runner(rsmodel, optimizer, scheduler=scheduler, train=train, dev=dev, test=test, samples=samples,
                    args=args)
    runner.run()
    log.info("Done!")

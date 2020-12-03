import os
import argparse
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from geoopt.optim import RiemannianSGD
from sympa import config
from sympa.utils import set_seed, get_logging, scale_triplets, subsample_triplets
from sympa.runner import Runner
from sympa.model import Model


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Name of prep folder")
    parser.add_argument("--run_id", required=True, type=str, help="Name of model/run to export")
    # Model
    parser.add_argument("--model", default="upper", type=str, help="Model type: 'euclidean', 'poincare', "
                                                                       "'upper' or 'bounded'")
    parser.add_argument("--dims", default=3, type=int, help="Dimensions for the model.")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")
    parser.add_argument("--scale_coef", default=1, type=float, help="Coefficient to divide scale.")
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
    parser.add_argument("--n_procs", default=4, type=int, help="Number of process to create")
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
    model = DistributedDataParallel(model, device_ids=None)
    return model


def get_scheduler(optimizer, args):
    patience = round(args.patience / args.val_every)
    factor = 1 / float(args.reduce_factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)


def load_training_data(args, log):
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

    train_src_dst_ids = torch.LongTensor([(src, dst) for src, dst, _ in sub_triplets]).to(config.DEVICE)
    train_distances = torch.Tensor([distance for _, _, distance in sub_triplets]).to(config.DEVICE)

    valid_src_dst_ids = train_src_dst_ids
    valid_distances = train_distances
    if args.subsample > 0:
        # train triplets are a subsample, valid triplets are all
        valid_src_dst_ids = torch.LongTensor([(src, dst) for src, dst, _ in all_triplets]).to(config.DEVICE)
        valid_distances = torch.Tensor([distance for _, _, distance in all_triplets]).to(config.DEVICE)

    train_batch_size = args.batch_size // args.n_procs
    log.info(f"Batch size {train_batch_size} for {args.n_procs} processes")
    train_triples = TensorDataset(train_src_dst_ids, train_distances)
    train_sampler = DistributedSampler(train_triples, num_replicas=args.n_procs, rank=args.this_rank)
    train_loader = DataLoader(dataset=train_triples, batch_size=train_batch_size, shuffle=False, num_workers=0,
                              pin_memory=True, sampler=train_sampler)

    valid_triples = TensorDataset(valid_src_dst_ids, valid_distances)
    valid_loader = DataLoader(valid_triples, sampler=SequentialSampler(valid_triples), batch_size=args.batch_size)

    return id2node, train_loader, valid_loader


def main(this_rank, args):
    """
    :param this_rank: process "rank" or "order" of all spawned processes by mp.spawn.
                    Value will be between [0, nprocs - 1]
    :param args:
    """
    # parser = argparse.ArgumentParser("train.py")
    # config_parser(parser)
    # args = parser.parse_args()
    # log.info(args)
    log = get_logging()
    args.this_rank = this_rank
    dist.init_process_group(backend=config.BACKEND, world_size=args.n_procs, rank=args.this_rank)

    id2node, train_loader, valid_loader = load_training_data(args, log)

    args.num_points = len(id2node)
    model = get_model(args)
    optimizer = RiemannianSGD(model.parameters(), lr=args.learning_rate, # * args.n_procs,
                              weight_decay=args.weight_decay, stabilize=10)
    scheduler = get_scheduler(optimizer, args)

    log.info(f"GPU's available: {torch.cuda.device_count()}")
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    log.info(f"Points: {args.num_points}, dims: {args.dims}, number of parameters: {n_params}")
    log.info(model)

    runner = Runner(model, optimizer, scheduler=scheduler, id2node=id2node, args=args,
                    train_loader=train_loader, valid_loader=valid_loader)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    log = get_logging()
    log.info(args)

    # set one unique seed for all process
    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    # DDP config
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main, nprocs=args.n_procs, args=(args,))

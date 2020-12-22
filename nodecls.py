# Node classifier
# The goal is to perform node classification using the node embeddings obtained from different models
# as features.

import copy
import time
import argparse
import random
from statistics import mean
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from sympa.utils import set_seed, write_results_to_file

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device=DEVICE)


class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(args.dim, args.dim) for _ in range(args.layers)])
        self.relu = nn.ReLU()
        self.final_layer = nn.Linear(args.dim, args.n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        """
        :param features: b x dim
        :return: out: b x n_classes
        """
        x = features
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.final_layer(x)
        x = self.softmax(x)
        return x


def load_data(path_to_saved_model):
    """
    Loads the model.
    If it is a vector model, it returns it.
    If it is a matrix model, it extracts the upper triangular of the real and imaginary matrices
    :param path_to_saved_model:
    :return: features: b x dim
    """
    print(f"Loading saved model from {path_to_saved_model}")
    model = torch.load(path_to_saved_model)
    features = model["model"]["module.embeddings.embeds"]   # points x dim or points x 2 x n x n
    if "bounded" in path_to_saved_model or "upper" in path_to_saved_model:
        real, imag = features[:, 0], features[:, 1]         # b x n x n
        n = real.shape[-1]
        row, col = torch.triu_indices(n, n)
        real_feat = real[:, row, col]
        imag_feat = imag[:, row, col]
        features = torch.cat((real_feat, imag_feat), dim=-1)
    return features.to(DEVICE)


def load_labels(args):
    path_to_labels = f"data/{args.dataset}/{args.dataset}-labels.pt"
    print(f"Loading labels from {path_to_labels}")
    labels = torch.load(path_to_labels)
    return torch.LongTensor(np.reshape(labels, (-1, 1))).to(DEVICE)       # points x 1


def create_splits(features, labels, args):
    """
    Creates train, dev and test splits.
    PRE: seed was set so the generated splits are always the same

    :param features: tensor of points x dim
    :param labels: tensor of points x 1
    :param args:
    :return: train, valid and test data loaders
    """
    points = len(features)
    train_size, valid_size, test_size = int(points * 0.3), int(points * 0.1), int(points * 0.6)
    train_size += 1 if train_size + valid_size + test_size < points else 0
    print(f"Points: {points}. Creating splits of train: {train_size}, valid: {valid_size}, test: {test_size}")
    dataset = TensorDataset(features, labels)
    train_dset, valid_dset, test_dset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dset, sampler=RandomSampler(train_dset), batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dset, sampler=SequentialSampler(valid_dset), batch_size=args.batch_size)
    test_loader = DataLoader(test_dset, sampler=SequentialSampler(test_dset), batch_size=args.batch_size)

    return train_loader, valid_loader, test_loader


def train_epoch(model, optim, train_loader, args):
    loss_fn = nn.CrossEntropyLoss()
    loss_acum = 0.0
    model.train()
    for step, batch in enumerate(train_loader):
        model.zero_grad()
        optim.zero_grad()
        features, labels = batch
        labels = labels.squeeze(-1)

        predictions = model(features)

        loss = loss_fn(predictions, labels)
        loss.backward()

        loss_acum += loss.item()
        # update
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
    return loss_acum / len(train_loader)


def measure_accuracy(predictions, labels):
    """
    :param predictions: b x n_classes
    :param labels: b x 1: index of ground-truth class
    :return: tensor of b, with ones and zeros. one: right prediction, zero: wrong prediction
    """
    pred_class = torch.argmax(predictions, dim=-1, keepdim=True)
    result = pred_class == labels
    return result.int().squeeze(-1)


def evaluate(model, data_loader):
    model.eval()
    total_acc = []
    for batch in data_loader:
        features, labels = batch
        with torch.no_grad():
            predictions = model(features)
            accuracy = measure_accuracy(predictions, labels)
        total_acc.extend(accuracy.tolist())

    avg_distortion = mean(total_acc)
    return avg_distortion


def train(model, optim, train_loader, valid_loader, args):
    best_acc, best_epoch = -1, -1
    best_model_state = None
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, optim, train_loader, args)
        exec_time = time.perf_counter() - start
        print(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')

        if epoch % args.val_every == 0:
            acc = evaluate(model, valid_loader)

            if acc > best_acc:
                print(f"Best val acc: {acc * 100:.3f}, at epoch {epoch}")
                best_acc = acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

            if epoch - best_epoch >= args.val_every * 5:
                print(f"Early stopping at epoch {epoch}!!!")
                break
    print(f"Best model from epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    return model


def get_dataset_name(model_name):
    if "iris" in model_name: return "iris"
    if "glass" in model_name: return "glass"
    if "zoo" in model_name: return "zoo"
    if "segment" in model_name: return "segment"
    if "energy" in model_name: return "energy"
    raise ValueError(f"Unrecognized dataset in model name: {model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("build_graph_from_dataset.py")
    parser.add_argument("--load_model", required=True, type=str, help="Path of model to load")
    parser.add_argument("--run_id", required=True, type=str, help="Name of run")

    # hyperparams
    parser.add_argument("--layers", default=1, type=int, help="Number of layers in the classifier.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Starting learning rate.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs.")

    # other
    parser.add_argument("--results_file", default="out/node/cls-res.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--val_every", default=25, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    args = parser.parse_args()

    set_seed(42)        # seed is fixed to generate the same splits always.
    args.dataset = get_dataset_name(args.load_model)
    features = load_data(args.load_model)       # points x dim
    labels = load_labels(args)                  # points x 1

    assert len(features) == len(labels), f"Features and labels do not have same length. " \
                                         f"Features: {args.dataset}, model: {args.load_model}"
    args.dim = features.shape[-1]
    args.n_classes = len(set(labels.squeeze(-1).tolist()))
    print(args)
    train_loader, valid_loader, test_loader = create_splits(features, labels, args)

    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(args.seed)  # seed is set again, so there is variability in different runs
    model = Classifier(args).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"GPU's available: {torch.cuda.device_count()}")
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print(f"Points: {len(features)}, dims: {args.dim}, classes: {args.n_classes}, parameters: {n_params}")
    print(model)

    model = train(model, optim, train_loader, valid_loader, args)
    test_acc = evaluate(model, test_loader)
    print(f"Final evaluation accuracy: {test_acc * 100:.3f}")
    loaded_model = args.load_model.split("/")[-1]
    write_results_to_file(f"out/node/{loaded_model}", {
        "accuracy": test_acc, #"lr": args.learning_rate, "mgr": args.max_grad_norm, "bs": args.batch_size,
        "dims": args.dim, "data": args.dataset, "run_id": args.run_id
    })

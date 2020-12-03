
import copy
import time
import torch
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from statistics import mean
from tqdm import tqdm, trange
from sympa import config
from sympa.losses import AverageDistortionLoss
from sympa.metrics import AverageDistortionMetric, MeanAveragePrecisionMetric
from sympa.utils import get_logging, write_results_to_file


class Runner(object):
    def __init__(self, model, optimizer, scheduler, id2node, train_loader, valid_loader, args):
        self.dpp_model = model
        self.model = model.module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.id2node = id2node
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = AverageDistortionLoss()
        self.metric = AverageDistortionMetric()
        self.args = args
        self.log = get_logging()
        self.is_main_process = args.this_rank == 0
        if self.is_main_process:
            self.writer = SummaryWriter(config.TENSORBOARD_PATH / args.run_id)

    def run(self):
        best_distortion, best_epoch = float("inf"), -1
        best_model_state = None
        for epoch in range(1, self.args.epochs + 1): #trange(1, self.args.epochs + 1, desc="full_train"):
            self.train_loader.sampler.set_epoch(epoch)  # sets epoch for shuffling
            self.set_burnin_lr(epoch)
            start = time.perf_counter()
            train_loss = self.train_epoch(self.train_loader, epoch)
            exec_time = time.perf_counter() - start

            if self.is_main_process:
                self.log.info(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("train/lr", self.get_lr(), epoch)
                self.writer.add_scalar("train/scale", self.model.get_scale(), epoch)
                self.writer.add_scalar("embeds/avg_norm", self.model.embeds_norm().mean().item(), epoch)
                if hasattr(self.model.manifold, 'projected_points'):
                    self.writer.add_scalar("train/projected_points", self.model.manifold.projected_points, epoch)

            if epoch % self.args.save_epochs == 0 and self.is_main_process:
                self.save_model(epoch)

            if epoch % self.args.val_every == 0:
                distortion = self.evaluate(self.valid_loader)
                if self.is_main_process:
                    self.writer.add_scalar("val/distortion", distortion, epoch)
                self.log.info(f"RANK {self.args.this_rank}: Results ep {epoch}: tr loss: {train_loss:.1f}, val avg distortion: {distortion * 100:.2f}")

                self.scheduler.step(distortion)

                if distortion < best_distortion:
                    precision = self.calculate_mAP()
                    if self.is_main_process:
                        self.log.info(f"Best val distortion: {distortion * 100:.3f}, mAP: {precision * 100:.2f} at epoch {epoch}")
                    best_distortion = distortion
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.dpp_model.state_dict())

                # early stopping
                if epoch - best_epoch >= self.args.patience * 3:
                    self.log.info(f"RANK {self.args.this_rank}: Early stopping at epoch {epoch}!!!")
                    break

        self.log.info(f"RANK {self.args.this_rank}: Final evaluation on best model from epoch {best_epoch}")
        self.dpp_model.load_state_dict(best_model_state)

        distortion = self.evaluate(self.valid_loader)
        precision = self.calculate_mAP()

        if self.is_main_process:
            self.export_results(distortion, precision)
            self.log.info(f"Final Results: Distortion: {distortion * 100:.2f}, Precision: {precision * 100:.2f}")
            self.save_model(best_epoch)
            self.writer.close()

    def train_epoch(self, train_split, epoch_num):
        self.check_points_in_manifold()
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.dpp_model.train()
        self.dpp_model.zero_grad()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_split): # enumerate(tqdm(train_split, desc=f"epoch_{epoch_num}")):
            src_dst_ids, graph_distances = batch     # .to(config.DEVICE)    Dataset is already on DEVICE

            manifold_distances = self.dpp_model(src_dst_ids)

            loss = self.loss.calculate_loss(graph_distances, manifold_distances)
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # stats
            tr_loss += loss.item()
            gradient = self.model.embeddings.embeds.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            avg_grad_norm += grad_norm

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                clip_grad_norm_(self.dpp_model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.dpp_model.zero_grad()
                self.optimizer.zero_grad()

        if self.is_main_process:
            self.writer.add_scalar("grad_norm/avg", avg_grad_norm / len(train_split), epoch_num)
        return tr_loss / len(train_split)

    def evaluate(self, eval_split):
        self.dpp_model.eval()
        total_distortion = []
        for batch in eval_split:        # tqdm(eval_split, desc="Evaluating"):
            src_dst_ids, graph_distances = batch     # .to(config.DEVICE)
            with torch.no_grad():
                manifold_distances = self.dpp_model(src_dst_ids)
                distortion = self.metric.calculate_metric(graph_distances, manifold_distances)
            total_distortion.extend(distortion.tolist())

        avg_distortion = mean(total_distortion)
        return avg_distortion

    def calculate_mAP(self):
        distance_matrix = self.build_distance_matrix()
        mAP = MeanAveragePrecisionMetric(self.valid_loader.dataset)
        return mAP.calculate_metric(distance_matrix)

    def build_distance_matrix(self):
        all_nodes = torch.arange(0, len(self.id2node)).unsqueeze(1)
        distance_matrix = torch.zeros((len(all_nodes), len(all_nodes)))
        self.dpp_model.eval()
        for node_id in range(len(self.id2node)):
            src = torch.LongTensor([[node_id]]).repeat(len(all_nodes), 1)
            src[node_id] = (node_id + 1) % len(all_nodes)   # hack to not ask for the distance to the same value
            batch = torch.cat((src, all_nodes), dim=-1)
            with torch.no_grad():
                distances = self.dpp_model(batch)
            distances[node_id] = 0
            distance_matrix[node_id] = distances.view(-1)
        return distance_matrix

    def save_model(self, epoch):
        # TODO save optimizer and scheduler
        save_path = config.CKPT_PATH / f"{self.args.run_id}-best-{epoch}ep"
        self.log.info(f"Saving model checkpoint to {save_path}")
        torch.save({"model": self.dpp_model.state_dict(), "id2node": self.id2node}, save_path)

    def set_burnin_lr(self, epoch):
        """Modifies lr if epoch is less than burn-in epochs"""
        if self.args.burnin < 1:
            return
        if epoch == 1:
            self.set_lr(self.get_lr() / config.BURNIN_FACTOR)
        if epoch == self.args.burnin:
            self.set_lr(self.get_lr() * config.BURNIN_FACTOR)

    def set_lr(self, value):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = value

    def get_lr(self):
        """:return current learning rate as a float"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def check_points_in_manifold(self):
        """it checks that all the points are in the manifold"""
        all_points_ok, outside_point, reason = self.model.check_all_points()
        if not all_points_ok:
            raise AssertionError(f"Point outside manifold. Reason: {reason}\n{outside_point}")

    def export_results(self, avg_distortion, avg_precision):
        manifold = self.args.model
        dims = self.args.dims
        if manifold == "upper" or manifold == "bounded":
            dims = dims * (dims + 1)
        result_data = {"data": self.args.data, "dims": dims, "manifold": manifold, "run_id": self.args.run_id,
                       "distortion": avg_distortion * 100, "mAP": avg_precision * 100}
        write_results_to_file(self.args.results_file, result_data)

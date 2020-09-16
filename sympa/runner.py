
import copy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from statistics import mean
from tqdm import tqdm, trange
from sympa import config
from sympa.losses import AverageDistortionLoss
from sympa.metrics import AverageDistortionMetric
from sympa.utils import get_logging, export_results

log = get_logging()


class Runner(object):
    def __init__(self, model, optimizer, scheduler, id2node, distances, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.id2node = id2node
        point_ids = TensorDataset(torch.arange(0, len(self.id2node), dtype=torch.long))
        self.train = DataLoader(point_ids, sampler=RandomSampler(point_ids), batch_size=args.batch_size)
        self.validate = DataLoader(point_ids, sampler=SequentialSampler(point_ids), batch_size=args.batch_size)
        self.loss = AverageDistortionLoss(distances)
        self.metric = AverageDistortionMetric(distances)
        self.args = args
        self.writer = SummaryWriter(config.TENSORBOARD_PATH / args.run_id)

    def run(self):
        best_val_metric, best_epoch = float("inf"), -1
        best_model_state = None
        for epoch in trange(1, self.args.epochs + 1, desc="full_train"):
            train_loss = self.train_epoch(self.train, epoch)
            val_metric = self.evaluate(self.validate)

            self.scheduler.step(val_metric)

            log.info(f"Results ep {epoch}: tr loss: {train_loss:.1f}, "
                     f"val avg distortion: {val_metric * 100:.2f}")

            self.writer.add_scalar("embeds/avg_norm", self.model.embeds_norm().mean().item(), epoch)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr", self.get_lr(), epoch)
            self.writer.add_scalar("val/distortion", val_metric, epoch)

            if epoch % self.args.save_epochs == 0:
                self.save_model(epoch)

            if val_metric < best_val_metric:
                log.info(f"Best val distortion: {val_metric * 100:.3f} at epoch {epoch}")
                best_val_metric = val_metric
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            # TODO: implement early stopping
        log.info(f"Final evaluation on best model from epoch {best_epoch}")
        self.model.load_state_dict(best_model_state)

        val_metric = self.evaluate(self.validate)
        export_results(self.args.results_file, self.args.run_id, {"distortion": val_metric * 100})
        log.info(f"Final Results: Distortion: {val_metric * 100:.2f}")

        self.save_model(best_epoch)
        self.writer.close()

    def train_epoch(self, train_split, epoch_num):
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        for step, batch in enumerate(train_split):     # enumerate(tqdm(train_split, desc=f"epoch_{epoch_num}")):

            # before doing anything in each iteration, it checks that all the points are in the manifold
            all_points_ok, outside_point, reason = self.model.check_all_points()
            if not all_points_ok:
                raise AssertionError(f"Point outside manifold. Reason: {reason}\n{outside_point}")
            ####

            batch_points = batch[0].to(config.DEVICE)

            src_index, dst_index, src_embeds, dst_embeds = self.model(batch_points)

            loss = self.loss.calculate_loss(src_index, dst_index, src_embeds, dst_embeds, self.model.distance)
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # stats
            tr_loss += loss.item()
            gradient = self.model.embeddings.embeds.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            avg_grad_norm += grad_norm

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

        self.writer.add_scalar("grad_norm/avg", avg_grad_norm / len(train_split), epoch_num)
        return tr_loss / len(train_split)

    def evaluate(self, eval_split):
        self.model.eval()
        total_distortion = []
        for batch in eval_split:        # tqdm(eval_split, desc="Evaluating"):
            batch_points = batch[0].to(config.DEVICE)
            with torch.no_grad():
                src_index, dst_index, src_embeds, dst_embeds = self.model(batch_points)
                distortion = self.metric.calculate_metric(src_index, dst_index, src_embeds, dst_embeds,
                                                          self.model.distance)
            total_distortion.extend(distortion.tolist())

        avg_distortion = mean(total_distortion)
        return avg_distortion

    def save_model(self, epoch):
        # TODO save optimizer and scheduler
        save_path = config.CKPT_PATH / f"{self.args.run_id}-best-{epoch}ep"
        log.info(f"Saving model checkpoint to {save_path}")
        torch.save({"model": self.model.state_dict(), "id2node": self.id2node}, save_path)

    def get_lr(self):
        """:return current learning rate as a float"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

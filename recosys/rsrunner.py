
import copy
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from sympa import config
from sympa.utils import get_logging, write_results_to_file
from recosys.rslosses import BCELoss, HingeLoss
from recosys.rsmetrics import RankingBuilder, rank_to_metric


def get_loss(loss):
    if loss == "bce": return BCELoss
    if loss == "hinge": return HingeLoss
    raise ValueError(f"Unrecognized loss: {loss}")


class Runner(object):
    def __init__(self, model, optimizer, scheduler, train, dev, test, samples, args):
        self.ddp_model = model
        self.model = model.module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.dev = dev
        self.test = test
        self.samples = samples
        self.loss = get_loss(args.loss)(ini_neg_index=0, end_neg_index=args.num_points, args=args)
        self.ranking_builder = RankingBuilder(ini_index=0, end_index=args.n_items, samples=samples)
        self.args = args
        self.log = get_logging()
        self.is_main_process = args.local_rank == 0
        if self.is_main_process:
            self.writer = SummaryWriter(str(config.TENSORBOARD_PATH / args.run_id))

    def run(self):
        best_hitrate, best_epoch = -1, -1
        best_model_state = None
        for epoch in range(1, self.args.epochs + 1):
            self.train.sampler.set_epoch(epoch)      # sets epoch for shuffling
            self.set_burnin_lr(epoch)
            start = time.perf_counter()
            train_loss = self.train_epoch(self.train, epoch)
            exec_time = time.perf_counter() - start

            if self.is_main_process:
                self.log.info(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("train/lr", self.get_lr(), epoch)
                self.writer.add_scalar("embeds/avg_norm", self.model.embeds_norm().mean().item(), epoch)
                if hasattr(self.model.manifold, 'projected_points'):
                    self.writer.add_scalar("train/projected_points", self.model.manifold.projected_points, epoch)

            if epoch % self.args.save_epochs == 0 and self.is_main_process:
                self.save_model(epoch)

            if epoch % self.args.val_every == 0:
                hitrate, ndcg = self.evaluate(self.dev)
                if self.is_main_process:
                    self.writer.add_scalar("val/HR@10", hitrate, epoch)
                    self.writer.add_scalar("val/nDCG", ndcg, epoch)
                self.log.info(f"RANK {self.args.local_rank}: Results ep {epoch}: tr loss: {train_loss:.1f}, "
                              f"dev HR@10: {hitrate:.2f}, nDCG: {ndcg:.3f}")

                self.scheduler.step(hitrate)

                if hitrate > best_hitrate:
                    if self.is_main_process:
                        self.log.info(f"Best dev HR@10: {hitrate:.3f}, at epoch {epoch}")
                    best_hitrate = hitrate
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.ddp_model.state_dict())

                # early stopping
                if epoch - best_epoch >= self.args.patience * 3:
                    self.log.info(f"RANK {self.args.local_rank}: Early stopping at epoch {epoch}!!!")
                    break

        self.log.info(f"RANK {self.args.local_rank}: Final evaluation on best model from epoch {best_epoch}")
        self.ddp_model.load_state_dict(best_model_state)

        hitrate, ndcg = self.evaluate(self.test)

        if self.is_main_process:
            self.export_results(hitrate, ndcg)
            self.log.info(f"Final Results: HR@10: {hitrate:.2f}, nDCG: {ndcg:.3f}")
            self.save_model(best_epoch)
            self.writer.close()

    def train_epoch(self, train_split, epoch_num):
        self.check_points_in_manifold()
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.ddp_model.train()
        self.ddp_model.zero_grad()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_split):

            loss = self.loss.calculate_loss(self.ddp_model, batch[0])
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # stats
            tr_loss += loss.item()
            gradient = self.model.embeddings.embeds.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            avg_grad_norm += grad_norm

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                clip_grad_norm_(self.ddp_model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.ddp_model.zero_grad()
                self.optimizer.zero_grad()

        if self.is_main_process:
            self.writer.add_scalar("grad_norm/avg", avg_grad_norm / len(train_split), epoch_num)
        return tr_loss / len(train_split)

    def evaluate(self, eval_split):
        self.ddp_model.eval()
        ranking = []
        for batch in eval_split:
            with torch.no_grad():
                partial_ranking = self.ranking_builder.rank(self.ddp_model, batch[0])
                ranking.append(partial_ranking)

        ranking = np.concatenate(ranking, axis=0)
        hitrate, ndcg, mrr = rank_to_metric(ranking, at_k=10)

        return hitrate, ndcg

    def save_model(self, epoch):
        # TODO save optimizer and scheduler
        save_path = config.CKPT_PATH / f"{self.args.run_id}-best-{epoch}ep"
        self.log.info(f"Saving model checkpoint to {save_path}")
        torch.save({"model": self.ddp_model.state_dict()}, save_path)

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

    def export_results(self, hr_at_k, ndcg_at_k):
        manifold = self.args.model
        dims = self.args.dims
        if "upper" in manifold or "bounded" in manifold:
            dims = dims * (dims + 1)
        result_data = {"data": self.args.prep, "dims": dims, "manifold": manifold, "run_id": self.args.run_id,
                       "HR@10": hr_at_k, "nDCG@10": ndcg_at_k}
        write_results_to_file(self.args.results_file, result_data)

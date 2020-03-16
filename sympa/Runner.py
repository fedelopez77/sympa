
import copy
import torch
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
from sympa import config
from sympa.utils import get_logging

log = get_logging()


class Runner(object):
    def __init__(self, model, optimizer, scheduler, train, dev, test, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.train_subset = train
        self.dev = dev
        self.test = test
        # self.train = DataLoader(train, sampler=RandomSampler(train), batch_size=args.batch_size)
        # self.train_subset = DataLoader(train, sampler=RandomSampler(train, replacement=True, num_samples=len(dev)),
        #                                batch_size=args.batch_size)
        # self.dev = DataLoader(dev, sampler=SequentialSampler(dev), batch_size=args.batch_size)
        # self.test = DataLoader(test, sampler=SequentialSampler(test), batch_size=args.batch_size)
        self.args = args
        self.writer = SummaryWriter(config.TENSORBOARD_PATH / args.run_id)

    def run(self):
        best_dev_metric, best_epoch = -1, -1
        best_model_state = None
        for epoch in trange(1, self.args.epochs + 1, desc="full_train"):
            train_loss = self.train_epoch(self.train, epoch)
            _, train_metric = self.evaluate(self.train_subset)
            dev_loss, dev_metric = self.evaluate(self.dev)

            log.info(f"Results ep {epoch}: tr loss: {train_loss * 100:.1f}, tr QWK: {train_metric * 100:.2f}, "
                     f"dev loss: {dev_loss * 100:.1f}, dev QWK: {dev_metric * 100:.2f}")

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/qwk", train_metric, epoch)
            self.writer.add_scalar("dev/loss", dev_loss, epoch)
            self.writer.add_scalar("dev/qwk", dev_metric, epoch)

            if epoch % self.args.save_epochs == 0:
                self.save_model(epoch)

            if dev_metric > best_dev_metric:
                _, test_metric = self.evaluate(self.test)
                log.info(f"Best dev QWK: {dev_metric * 100:.3f}, test QWK: {test_metric * 100:.3f} at epoch {epoch}")
                best_dev_metric = dev_metric
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())

        log.info(f"Final evaluation on best model from epoch {best_epoch}")
        self.model.load_state_dict(best_model_state)

        dev_loss, dev_metric = self.evaluate(self.dev)
        test_loss, test_metric = self.evaluate(self.test)

        log.info(f"Final Results:\n"
                 f"DEV loss: {dev_loss * 100:.2f}, QWK: {dev_metric * 100:.2f}\n"
                 f"TEST loss: {test_loss * 100:.2f}, QWK: {test_metric * 100:.2f}")
        self.save_model(best_epoch)
        self.writer.close()

    def train_epoch(self, train_split, epoch_num):
        tr_loss = 0.0
        classifier_avg_grad_norm = 0.0
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_split, desc=f"epoch_{epoch_num}")):
            batch = tuple(t.to(config.DEVICE) for t in batch)

            outputs = self.model(batch)
            loss = outputs[0]
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # stats
            tr_loss += loss.item()
            gradient = self.model.module.classifier.weight.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            classifier_avg_grad_norm += grad_norm

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

        self.writer.add_scalar("classif_grad_norm/avg", classifier_avg_grad_norm / len(train_split), epoch_num)
        return tr_loss / len(train_split)

    def evaluate(self, eval_split):
        self.model.eval()
        total_eval_loss = 0.0
        predicted_rating, true_rating = [], []
        for batch in tqdm(eval_split, desc="Evaluating"):
            batch = tuple(t.to(config.DEVICE) for t in batch)
            true_labels = batch[-1]

            with torch.no_grad():
                outputs = self.model(batch)
                batch_eval_loss, logits = outputs[:2]
                total_eval_loss += batch_eval_loss.mean().item()

            predicted_rating.extend(logits)
            true_rating.extend(true_labels.detach().tolist())

        qwk_result = qwk(true_rating, predicted_rating, min_rating=0,
                                           max_rating=len(self.label_names) - 1)
        return total_eval_loss / len(eval_split), qwk_result

    def save_model(self, epoch):
        # TODO save optimizer and scheduler
        save_path = config.CKPT_PATH / f"{self.args.run_id}-best-{epoch}ep"
        log.info(f"Saving model checkpoint to {save_path}")
        torch.save(self.model.state_dict(), save_path)

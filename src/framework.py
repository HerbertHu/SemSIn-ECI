import os
import torch
import torch.nn as nn
import itertools

from tqdm import tqdm
from transformers import AdamW
from sklearn import metrics
import torch.nn.functional as F

from utils.logger import get_logger


class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()

        self.class_num = class_num
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).to(device)
        else:
            assert len(alpha) == class_num
            self.alpha = torch.Tensor(alpha).to(device)

    def forward(self, predict, target):
        # probs = F.softmax(predict, dim=1)  # softmax
        probs = predict
        class_mask = F.one_hot(target, self.class_num)  # get target one hot
        ids = target.view(-1, 1)
        alpha = self.alpha[ids]  # get alpha weight
        pt = (probs * class_mask).sum(dim=1).view(-1, 1)  # onehot mask，get probs

        eps = 1e-7
        log_pt = torch.log(pt + eps)
        loss = -alpha * (torch.pow((1 - pt), self.gamma)) * log_pt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Framework:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()

        # loss function
        self.gamma = args.gama
        if args.alpha < 0:
            self.alpha = None
        else:
            self.alpha = [args.alpha, 1.0 - args.alpha]
        self.loss_fn = FocalLoss(class_num=args.num_labels, gamma=self.gamma, alpha=self.alpha,
                                 reduction='sum', device=args.device)
        self.logger.info('gamma: {}'.format(self.gamma))
        self.logger.info('alpha: {}'.format(self.alpha))

    def train_step(self, batch_data, model_bert, model_gcn, model_init):
        # train step in stage of training
        model_init.train()
        model_bert.train()
        model_gcn.train()

        sentences_s, mask_s, event1, event1_mask, event2, event2_mask, graphs, graphs_len, \
        graph_event1, graph_event1_mask, graph_event2, graph_event2_mask, data_y, \
        paths_list, paths_len_list, bert_init_list, random_init_list = batch_data

        # get bert representations
        bert_opt, _ = model_bert.forward(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
        _, sent_h = model_init.forward(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)

        # get graph representations
        gcn_opt = model_gcn.forward(graphs, graphs_len, graph_event1, graph_event1_mask,
                                    graph_event2, graph_event2_mask, paths_list, paths_len_list,
                                    bert_init_list, random_init_list, sent_h)

        ensemble_opt = torch.cat([bert_opt, gcn_opt], dim=1)

        ensemble_opt = model_gcn.dropout(ensemble_opt)
        ensemble_opt = model_gcn.classifier(ensemble_opt)
        ensemble_opt = F.softmax(ensemble_opt, dim=1)

        # calculate loss
        loss = self.loss_fn(ensemble_opt, data_y)

        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_gcn.parameters(), self.args.max_grad_norm)

        self.optimizer.step()
        self.optimizer2.step()

        return loss

    def train(self, train_dataset, dev_dataset_batch, model_bert, model_gcn, model_init):
        # get optimizer schedule
        self.optimizer = AdamW(itertools.chain(model_bert.parameters(), model_init.parameters()),
                               lr=self.args.bert_learning_rate)
        self.optimizer2 = AdamW(model_gcn.parameters(), lr=self.args.rgcn_learning_rate)

        # Train
        self.logger.info("***** Running training *****")
        self.logger.info(
            "Learning rate of BERT: {:.8f}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.logger.info(
            "Learning rate of GCN: {:.8f}".format(self.optimizer2.state_dict()['param_groups'][0]['lr']))

        best_result = 0

        train_dataset_batch = [batch for batch in
                               train_dataset.reader(self.args.device, shuffle=self.args.epoch_shuffle)]
        # self.logger.debug('train dataset: {}'.format(dict(sorted(Counter(train_dataset.paths_len_list).items()))))
        for epoch in range(0, int(self.args.num_train_epochs)):
            self.logger.info("***** Training *****")
            self.logger.info("Epoch: {}/{}".format(epoch + 1, self.args.num_train_epochs))

            if self.args.epoch_shuffle and epoch != 0:
                train_dataset_batch = [batch for batch in
                                       train_dataset.reader(self.args.device, shuffle=self.args.epoch_shuffle)]
            loss_train = 0
            for step, batch in enumerate(tqdm(train_dataset_batch, ncols=80, mininterval=10)):
                loss = self.train_step(batch, model_bert, model_gcn, model_init)
                loss_train += loss
            self.logger.info("Train loss: {:.6f}.".format(loss_train / len(train_dataset_batch)))

            # evaluate
            precision, recall, f1 = self.evaluate(dev_dataset_batch, model_bert, model_gcn, model_init)
            self.logger.info("Precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(precision, recall, f1))
            self.logger.info("Best f1: {:.3f}, current f1: {:.3f}.".format(best_result, f1))

            if f1 > best_result:
                best_result = f1
                self.logger.info(
                    "New best f1: {:.3f}. Saving model checkpoint.".format(best_result))
                torch.save(model_bert.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name))
                torch.save(model_gcn.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name2))
                torch.save(model_init.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name3))

    def evaluate(self, dataset_batch, model_bert, model_gcn, model_init):
        self.logger.info("***** Evaluating *****")

        model_init.eval()
        model_bert.eval()
        model_gcn.eval()

        loss_eval = 0
        predicted_all = []
        gold_all = []
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataset_batch, ncols=80, mininterval=5)):
                sentences_s, mask_s, event1, event1_mask, event2, event2_mask, graphs, graphs_len, \
                graph_event1, graph_event1_mask, graph_event2, graph_event2_mask, data_y, \
                paths_list, paths_len_list, bert_init_list, random_init_list = batch_data

                # get bert representations
                bert_opt, _ = model_bert.forward(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
                _, sent_h = model_init.forward(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)

                # get graph representations
                gcn_opt = model_gcn.forward(graphs, graphs_len, graph_event1, graph_event1_mask,
                                            graph_event2, graph_event2_mask, paths_list, paths_len_list,
                                            bert_init_list, random_init_list, sent_h)

                ensemble_opt = torch.cat([bert_opt, gcn_opt], dim=1)
                ensemble_opt = model_gcn.dropout(ensemble_opt)
                ensemble_opt = model_gcn.classifier(ensemble_opt)
                ensemble_opt = F.softmax(ensemble_opt, dim=1)

                predicted = torch.argmax(ensemble_opt, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                loss = self.loss_fn(ensemble_opt, data_y)
                loss_eval += loss
                gold = list(data_y.cpu().numpy())
                gold_all += gold

            # calculate f1, precision, recall and acc
            f1 = metrics.f1_score(gold_all, predicted_all) * 100
            precision = metrics.precision_score(gold_all, predicted_all) * 100
            recall = metrics.recall_score(gold_all, predicted_all) * 100

        return precision, recall, f1

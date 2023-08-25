import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.use_activation:
            x = self.tanh(x)
        return x


class BertEvent(nn.Module):

    def __init__(self, args):
        super(BertEvent, self).__init__()
        self.config = BertConfig.from_pretrained(args.plm_path)
        self.bert = BertModel.from_pretrained(args.plm_path, config=self.config)

        self.hidden_size = 768
        self.e1_fc_layer = FCLayer(self.hidden_size * 2, self.hidden_size, args.h_dropout)
        self.e2_fc_layer = FCLayer(self.hidden_size * 2, self.hidden_size, args.h_dropout)
        self.label_classifier = FCLayer(
            self.hidden_size * 1,
            args.num_labels,
            args.h_dropout,
            use_activation=False,
        )

    def forward(self, sentences_s, mask_s, event1, event1_mask, event2, event2_mask):
        # model forward propagation
        outputs = self.bert(sentences_s, attention_mask=mask_s)
        enc_s = outputs.last_hidden_state
        cls = enc_s[:, 0, ]

        # get event vectors
        e1_h = torch.cat([torch.index_select(a, 0, i) for a, i in zip(enc_s, event1)], dim=0)
        e2_h = torch.cat([torch.index_select(a, 0, i) for a, i in zip(enc_s, event2)], dim=0)

        e1_h = self.e1_fc_layer(torch.cat([cls, e1_h], dim=1))
        e2_h = self.e2_fc_layer(torch.cat([cls, e2_h], dim=1))

        opt = torch.add(e1_h, e2_h)
        # opt = self.label_classifier(opt)
        # opt = F.softmax(opt, dim=1)

        return opt, enc_s

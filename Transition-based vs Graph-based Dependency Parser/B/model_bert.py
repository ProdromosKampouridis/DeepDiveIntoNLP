import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from utils import normalize
from decoder_eisner import parse_proj
from transformers import BertModel, BertTokenizerFast


class BertBISTParser(nn.Module):
    def __init__(self, pos_emb_dim, mlp_hid_dim,
                 n_arc_relations, w_i_counter, p2i, alpha, device):
        super(BertBISTParser, self).__init__()

        self.w_i_counter = w_i_counter
        self.p2i = p2i

        self.alpha = alpha
        self.device = device

        # BERT initialization
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.bert_hid_dim = self.encoder.config.hidden_size

        # arc scorer initialization
        self.hid_arc_h = nn.Linear(self.bert_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_m = nn.Linear(self.bert_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        BertBISTParser.param_init(self.hid_arc_bias)

        self.slp_out_arc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(mlp_hid_dim, 1, bias=False)
        )

        # arc relations MLP initialization
        self.hid_rel_h = nn.Linear(self.bert_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_m = nn.Linear(self.bert_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        BertBISTParser.param_init(self.hid_rel_bias)

        self.slp_out_rel = nn.Sequential(
            nn.Tanh(),
            nn.Linear(mlp_hid_dim, n_arc_relations)
        )

        # initialize model weights
        for name, module in self.named_children():
            BertBISTParser.modules_init(module)

    def forward(self, sentence, pos_tags, gold_tree=None, word_dropout=False):
        # pos_tags are ingnored in this bert based solution
        n_words = len(sentence)
        encodings = self.tokenizer([w[0] for w in sentence], truncation=True, padding='max_length', is_split_into_words=True)
        
        word_ids = encodings.word_ids()
        first_subword_inds = []
        seen_ids = set()
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in seen_ids:
                first_subword_inds.append(i)
            seen_ids.add(wid)
        first_subword_inds = torch.LongTensor(first_subword_inds).to(self.device)

        bert_batch = {
          'input_ids': torch.LongTensor(encodings.input_ids).to(self.device).unsqueeze(0), 
          'attention_mask': torch.LongTensor(encodings.attention_mask).to(self.device).unsqueeze(0), 
        }

        hidden_vectors = self.encoder(**bert_batch).last_hidden_state 
        hidden_vectors = hidden_vectors[:, first_subword_inds, :]

        # score all possible arcs
        arc_h_scores = self.hid_arc_h(hidden_vectors)
        arc_m_scores = self.hid_arc_m(hidden_vectors)

        idx_ls = [idx for idx in range(n_words)]
        arc_scores = self.slp_out_arc(arc_h_scores[0, np.repeat(idx_ls, n_words), :]
                                      + arc_m_scores[0, idx_ls * n_words, :]
                                      + self.hid_arc_bias)
        arc_scores = arc_scores.view(n_words, n_words)

        # get the highest scoring dependency tree
        pred_tree = parse_proj(arc_scores.detach().cpu().numpy(), gold_tree)

        # score all possible relations
        heads = gold_tree[1:] if gold_tree is not None else pred_tree[1:]
        rel_h_scores = self.hid_rel_h(hidden_vectors)
        rel_m_scores = self.hid_rel_m(hidden_vectors)
        rel_scores = self.slp_out_rel(rel_h_scores[0, heads, :]
                                      + rel_m_scores[0, idx_ls[1:], :]
                                      + self.hid_rel_bias)

        if gold_tree is not None:  # during training
            return arc_scores, rel_scores, pred_tree

        return pred_tree, torch.argmax(rel_scores, dim=-1)  # during inference

    @staticmethod
    def modules_init(m):
        if isinstance(m, nn.Embedding):
            emb_bound = math.sqrt(3. / m.embedding_dim)
            nn.init.uniform_(m.weight, -emb_bound, emb_bound)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if 'bias' in name:
                    h_dim = p.shape[-1] // 4
                    nn.init.constant_(p[: h_dim], 0.)
                    nn.init.constant_(p[h_dim: 2 * h_dim], 0.5)  # forget gate bias initialization
                    nn.init.constant_(p[2 * h_dim:], 0.)
                else:
                    nn.init.xavier_uniform_(p)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Sequential):
            nn.init.xavier_uniform_(m[1].weight)
            if m[1].bias is not None:
                BertBISTParser.param_init(m[1].bias)

    @staticmethod
    def param_init(p):
        bound = math.sqrt(3. / p.shape[-1])
        nn.init.uniform_(p, -bound, bound)
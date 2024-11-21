
import torch
import torch.nn as nn
import numpy as np
import math
import ctypes
from ctypes import *
from torch.utils.checkpoint import checkpoint

from models.attention import Attention

from config import Config


mask_task = ['lra-listops', 'lra-text', 'lra-news', 'lra-yelp', 'lra-text1',  'lra-news1', 'lra-yelp1', 'lra-text2',  'lra-news2', 'lra-yelp2', 'lra-retrieval']

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

#        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]
        self.config = config

#        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
#        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.word_embeddings = nn.Linear(config["embedding_dim"], config["transformer_hidden_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["transformer_hidden_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):
        batch_size, seq_len,_ = input_ids.size()

        X_token = self.word_embeddings(input_ids)
#        X_token = X_token.unsqueeze(1).expand(-1, self.config["max_seq_len"], -1)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

    
class TransformerLayer(nn.Module):
    def __init__(self, config, inference=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])

        self.mha = Attention(config, inference)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
                    nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
                    nn.GELU(),
                    torch.nn.Dropout(p = config["dropout_prob"]),
                    nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
                    torch.nn.Dropout(p = config["dropout_prob"])
        )

        self.inference = inference
        

    def forward(self, X, mask) :
        out = self.mha(self.norm1(X), mask)
        X = self.dropout1(out) + X


        X = self.mlpblock(self.norm2(X)) + X

        return X


class Model(nn.Module):
    def __init__(self, config, inference=False):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]
        self.inference = inference

        self.embeddings = Embeddings(config)
        
        if self.tied_weights:
            self.transformer = TransformerLayer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", TransformerLayer(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    
    def forward(self, input_ids, mask = None, mat_lst=[], is_attn=False):


        X = self.embeddings(input_ids)
        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        mask = mask[:, :, 0]

        X = self.norm(X) * mask[:, :, None]

        return X
    
    
    

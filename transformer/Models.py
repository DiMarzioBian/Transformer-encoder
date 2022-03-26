import copy
import torch
import torch.nn as nn
from utils import init_embedding, init_linear

from transformer.Layers import PositionalEncoding, Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.device = args.device
        self.n_layer = args.n_layer
        self.d_model = args.d_model
        self.d_inner = args.d_inner
        self.n_head = args.n_head
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.dropout = args.dropout

        self.n_word = args.n_word
        self.n_gram = args.n_gram

        # embedding layer
        self.embedding = nn.Embedding(self.n_word, self.d_model)
        init_embedding(self.embedding)
        self.pos_enc = PositionalEncoding(self.d_model, self.n_gram, self.device)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        # transformer
        self.encoder = Encoder(self.n_layer, self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, self.dropout)
        self.decoder = Decoder(self.n_layer, self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, self.dropout)

        # predictor
        self.fc1 = nn.Linear(self.d_model, self.n_word, bias=True)
        if args.weight_sharing:
            self.fc1.weight = self.embedding.weight
        else:
            init_linear(self.fc1)

    def get_subsequent_mask2(self, seq):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def get_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, seq_batch):

        mask = self.get_subsequent_mask2(seq_batch)

        inputs = self.embedding(seq_batch)
        inputs += self.pos_enc(inputs)
        inputs = self.dropout1(inputs)
        inputs = self.layer_norm(inputs)

        enc_outputs, enc_slf_attn = self.encoder(inputs, slf_attn_mask=mask)
        dec_outputs, dec_slf_attn, dec_enc_attn = self.decoder(inputs, enc_outputs)

        outputs = self.fc1(dec_outputs[:, -1, :].squeeze(1))

        return outputs, enc_slf_attn, dec_slf_attn, dec_enc_attn



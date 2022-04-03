import torch
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    """ add sinusoid encoding """

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        (batch_size, len_seq, _) = x.shape
        return self.encoding[:len_seq, :]


class EncoderLayer(nn.Module):
    """ One encoder layer """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, scaled_attn=True, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, scaled_attn=scaled_attn, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ One decoder layer """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, scaled_attn=True, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, scaled_attn=scaled_attn, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, scaled_attn=scaled_attn, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    """ Transformer encoder """
    def __init__(self, n_layer, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  d_inner=d_inner,
                                                  n_head=n_head,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  dropout=dropout)
                                     for _ in range(n_layer)])

    def forward(self, x, slf_attn_mask=None):

        enc_slf_attn = []
        for layer in self.layers:
            x, slf_attn = layer(x, slf_attn_mask=slf_attn_mask)
            enc_slf_attn.append(slf_attn)
        return x, enc_slf_attn


class Decoder(nn.Module):
    """ Transformer decoder """
    def __init__(self, n_layer, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  d_inner=d_inner,
                                                  n_head=n_head,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  dropout=dropout)
                                     for _ in range(n_layer)])

    def forward(self, x, enc_outputs):

        dec_slf_attn, dec_enc_attn = [], []

        for layer in self.layers:
            x, slf_attn, enc_attn = layer(x, enc_outputs)
            dec_slf_attn.append(slf_attn)
            dec_enc_attn.append(enc_attn)
        return x, dec_slf_attn, dec_enc_attn

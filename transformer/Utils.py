import torch

PAD = 0


def get_pad_mask(seq):
    return (seq != PAD).unsqueeze(-2)


def get_subsequent_mask(seq):
    # masking future elements
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def get_attn_pad_mask(seq_q, seq_k):
    # masking padding elements
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)
    return pad_attn_mask.expand(b_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    # masking future elements
    assert seq.dim() == 2
    subsequent_mask = torch.triu(torch.ones(seq.size(0), seq.size(1), seq.size(1)))
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

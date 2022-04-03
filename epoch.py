from tqdm import tqdm
import torch


def train(args, model, data, optimizer):
    model.train()
    n_sample = 0
    loss_total = 0.

    for batch in tqdm(data, desc='  - training', leave=False):
        seq_batch, tgt_batch = map(lambda x: x.to(args.device), batch)

        optimizer.zero_grad()
        scores_batch, _ = model(seq_batch)
        loss_batch = args.criterion(scores_batch, tgt_batch)
        loss_batch.backward()
        optimizer.step()

        # calculate loss
        n_sample += scores_batch.shape[0]
        loss_total += loss_batch * scores_batch.shape[0]

    loss_mean = loss_total / n_sample

    return loss_mean


def evaluate(args, model, data):
    model.train()
    n_sample = 0
    loss_total = 0.

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            seq_batch, tgt_batch = map(lambda x: x.to(args.device), batch)

            scores_batch, _ = model(seq_batch)
            loss_batch = args.criterion(scores_batch, tgt_batch)

            # calculate loss
            n_sample += scores_batch.shape[0]
            loss_total += loss_batch * scores_batch.shape[0]

        loss_mean = loss_total / n_sample

    return loss_mean

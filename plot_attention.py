import argparse
import numpy as np
import math
import torch
from dataloader import get_dataloader
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for computing')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')
    parser.add_argument('--path_data_processed', type=str, default='./data/wikitext-2/data.pkl',
                        help='path of the processed data')

    # data settings
    parser.add_argument('--num_worker', type=int, default=0,
                        help='number of dataloader worker')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    print('\n[info] Evaluation starts...')

    with open(args.path_model, 'rb') as f:
        model = torch.load(f)
    model.eval()

    # modeling
    args.n_gram = model.n_gram
    args.n_head = model.n_head
    _, _, _, test_loader = get_dataloader(args)
    with torch.no_grad():
        for batch in test_loader:
            seq_batch, tgt_batch = map(lambda x: x.to(args.device), batch)
            _, enc_slf_attn = model(seq_batch)
            break

    # plot
    fig, ax = plt.subplots(nrows=math.ceil(args.n_head // 2), ncols=2, sharex='col', sharey='row')
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(ax.flat):
        ax.set_title('Head:'+str(i), fontstyle='italic')
        im = sns.heatmap(enc_slf_attn[0][0, i].tolist(), ax=ax, vmin=0, vmax=1, cmap='YlGnBu', cbar=i == 0,
                         cbar_ax=None if i else cbar_ax)
        im.set_xticklabels(np.arange(0, args.n_gram, 1))
        im.set_yticklabels(np.arange(1, args.n_gram + 1, 1))

    fig.supxlabel('Input word sequence')
    fig.supylabel('Ground truth')
    plt.subplots_adjust(left=0.15,
                        bottom=0.15,
                        right=0.85,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)
    fig.show()


if __name__ == '__main__':
    main()

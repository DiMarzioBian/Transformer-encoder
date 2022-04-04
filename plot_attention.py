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
    parser.add_argument('--path_data_processed', type=str, default='./data/wikitext-2/data.pkl',
                        help='path of the processed data')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')
    parser.add_argument('--path_image', type=str, default='./result/images/',
                        help='path of the images')

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

    # plot attention
    fig, ax = plt.subplots(figsize=[11, 22], nrows=math.ceil(args.n_head // 2), ncols=2, sharex='col', sharey='row')
    cbar_ax = fig.add_axes([.95, .3, .01, .4])
    for i, ax in enumerate(ax.flat):
        ax.set_title('Head:' + str(i), fontstyle='italic')
        im = sns.heatmap(enc_slf_attn[0][0, i].tolist(), ax=ax, vmin=0, vmax=1, cmap='YlGnBu', cbar=i == 0,
                         cbar_ax=None if i else cbar_ax)
        im.set_xticklabels(np.arange(0, args.n_gram, 1), rotation=45)
        im.set_yticklabels(np.arange(1, args.n_gram + 1, 1))
    fig.supxlabel('Input word sequence', fontsize=20)
    fig.supylabel('Ground truth', fontsize=20)
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.9,
                        top=0.98,
                        wspace=0.1,
                        hspace=0.13)
    fig.savefig(args.path_image + 'attn.jpg')
    fig.show()

    # plot postional encoding
    pos_enc = model.pos_enc.encoding.tolist()
    fig = plt.figure()
    im = sns.heatmap(list(pos_enc), cmap='YlGnBu')
    im.tick_params(labelsize=6)
    plt.xticks(rotation=45)
    fig.supxlabel('Index of word index dimension')
    fig.supylabel('Position of word')
    fig.savefig(args.path_image + 'pos_enc.jpg')
    fig.show()


if __name__ == '__main__':
    main()

import argparse
import torch
from dataloader import get_dataloader
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for computing')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')
    parser.add_argument('--path_data_processed', type=str, default='./data/wikitext-2/data.pkl',
                        help='path of the processed data')

    # data settings
    parser.add_argument('--n_gram', type=int, default=25,
                        help='number of transformer layer for both encoder and decoder')
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
    _, _, _, test_loader = get_dataloader(args)
    with torch.no_grad():
        for batch in test_loader:
            seq_batch, tgt_batch = map(lambda x: x.to(args.device), batch)
            _, enc_slf_attn = model(seq_batch)
            break

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(enc_slf_attn[0][0, :, -1].tolist(), cmap='Greys')
    plt.colorbar(im)
    plt.show()
    x=1


if __name__ == '__main__':
    main()

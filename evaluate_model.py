import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')

    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')

    args = parser.parse_args()

    print('\n[info] Evaluation starts...')

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()


if __name__ == '__main__':
    main()

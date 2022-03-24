import argparse
import time
import torch
import torch.nn as nn
import torch.onnx

from dataloader import get_dataloader
from transformer.Models import Transformer
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='Transformer project')

    # model setting
    parser.add_argument('--n_layer', type=int, default=3,
                        help='number of transformer layer for both encoder and decoder')
    parser.add_argument('--d_model', type=int, default=512,
                        help='model feature dimension')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--d_ffn', type=int, default=1024,
                        help='hidden representation size of the feed-forward layer')

    # preprocess
    parser.add_argument('--pad_number', type=bool, default=True,
                        help='pad all numbers to a same <num>')
    parser.add_argument('--lower_char', type=bool, default=True,
                        help='lower characters" cases')

    # training settings
    parser.add_argument('--n_gram', type=int, default=40,
                        help='number of transformer layer for both encoder and decoder')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='strength of lr downgrade')
    parser.add_argument('--es_patience_max', type=int, default=10,
                        help='max early stopped patience')
    parser.add_argument('--eps_f1', type=float, default=1e-4,
                        help='minimum f1 score difference threshold')

    # file settings
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--num_worker', type=int, default=5,
                        help='number of dataloader worker')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for computing')
    parser.add_argument('--path_data', type=str, default='./data/wikitext-2/',
                        help='path of the data corpus')
    parser.add_argument('--path_data_processed', type=str, default='./data/wikitext-2/data.pkl',
                        help='path of the processed data')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.d_k = args.d_v = args.d_model // args.n_head  # key and value representation size

    print('\n[info] Project starts...')
    print('\n[info] Load dataset and other resources...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # prepare data and model
    vocab, train_loader, valid_loader, test_loader = get_dataloader(args)
    args.n_word = len(vocab)

    model = Transformer(args).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Start modeling
    print('\n[info] | n_param {n_param} | n_layer {n_layer} | d_model {d_model} | n_head {n_head} | d_k {d_k} | '
          'd_ffn {d_ffn} |'
          .format(n_param=n_param, n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head, d_k=args.d_k,
                  d_ffn=args.d_ffn))
    best_loss_val = 1e5
    best_epoch = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        # training phase
        t_start = time.time()
        loss_train = train(args, model, train_loader, optimizer)
        scheduler.step()
        print('  | Train | loss {:5.4f} | ppl {:5.4f} | {:5.2f} s |'
              .format(loss_train, torch.exp(loss_train), time.time() - t_start))

        # validating phase
        loss_val = evaluate(args, model, valid_loader)
        if loss_val < best_loss_val:
            if best_loss_val - loss_val > args.eps_f1:
                es_patience = 0  # reset if beyond threshold
            with open(args.path_model, 'wb') as f:
                torch.save(model, f)
            best_loss_val = loss_val
            best_epoch = epoch
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best | epoch {:d} | loss {:5.4f} | ppl {:5.4f} |'
                      .format(best_epoch, best_loss_val, torch.exp(best_loss_val)))
                break
        # logging
        print('  | Valid | loss {:5.4f} | ppl {:5.4f} | es_patience {:.0f}/{:.0f} |'
              .format(loss_val, torch.exp(loss_train), es_patience, args.es_patience_max))

    # testing phase
    print('\n[Testing]')
    with open(args.path_model, 'rb') as f:
        model = torch.load(f)
    loss_test = evaluate(args, model, test_loader)

    print('  | Test | loss {:5.4f} | ppl {:5.4f} |'
          .format(loss_test, torch.exp(loss_test)))
    print('\n[info] | n_param {n_param} | n_layer {n_layer} | d_model {d_model} | n_head {n_head} | d_k {d_k} | '
          'd_ffn {d_ffn} |'
          .format(n_param=n_param, n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head, d_k=args.d_k,
                  d_ffn=args.d_ffn))


if __name__ == '__main__':
    main()

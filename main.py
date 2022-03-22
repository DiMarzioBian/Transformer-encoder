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
    parser.add_argument('--n_unit', type=int, default=25,
                        help='number of transformer unit')
    parser.add_argument('--dim_emb_word', type=int, default=100,
                        help='word embedding dimension')
    parser.add_argument('--dim_out_char', type=int, default=25,
                        help='character encoder output dimension')
    parser.add_argument('--dim_out_word', type=int, default=25,
                        help='word encoder output dimension')
    parser.add_argument('--window_kernel', type=int, default=5,
                        help='window width of CNN kernel')

    # training settings
    parser.add_argument('--num_worker', type=int, default=5,
                        help='number of dataloader worker')
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
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for computing')
    parser.add_argument('--path_data', type=str, default='./data/wikitext-2/',
                        help='path of the data corpus')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    print('\n[info] Project starts...')
    print('\n[info] Load dataset and other resources...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # prepare data and model
    train_loader, valid_loader, test_loader = get_dataloader(args)

    model = Transformer(args).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Start modeling
    print('\n[info] | lr: {lr} | dropout: {dropout} |char: {mode_char} | word: {mode_word} | CRF: {crf} '
          '| Param: {n_param} | '
          .format(lr=args.lr, dropout=args.dropout, mode_char=args.mode_char, mode_word=args.mode_word,
                  crf=args.enable_crf, n_param=n_param))
    best_val_loss = 1e5
    best_f1 = 0
    best_epoch = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        t_start = time.time()
        loss_train, f1_train = train(args, model, train_loader, optimizer)
        scheduler.step()
        print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'.format(loss_train, f1_train, time.time() - t_start))
        val_loss, val_f1 = evaluate(args, model, valid_loader)

        # Save the model if the validation loss is the best we've seen so far.
        if val_f1 > best_f1:
            if val_f1 - best_f1 > args.eps_f1:
                es_patience = 0  # reset if beyond threshold
            with open(args.path_model, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_epoch = epoch
            best_f1 = val_f1
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best | Epoch {:d} | Loss {:5.4f} | F1 {:5.4f} |'
                      .format(best_epoch, best_val_loss, best_f1))
                break
        # logging
        print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'.format(val_loss, val_f1, es_patience))

    # Load the best saved model and test
    print('\n[Testing]')
    with open(args.path_model, 'rb') as f:
        model = torch.load(f)
    loss_test, f1_test = evaluate(args, model, test_loader)

    print('  | Test | loss {:5.4f} | F1 {:5.4f} |'.format(loss_test, f1_test))
    print('\n[info] | lr: {lr} | dropout: {dropout} |char: {mode_char} | word: {mode_word} | CRF: {crf} '
          '| Param: {n_param} | '
          .format(lr=args.lr, dropout=args.dropout, mode_char=args.mode_char, mode_word=args.mode_word,
                  crf=args.enable_crf, n_param=n_param))


if __name__ == '__main__':
    main()

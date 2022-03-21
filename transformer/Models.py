import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from utils import init_embedding, init_lstm, init_linear


class Transformer(nn.Module):
    def __init__(self, args):
        """
        Input parameters from args:
                args = Dictionary that maps NER tags to indices
                word2idx = Dimension of word embeddings (int)
                tag2idx = hidden state dimension
                char2idx = Dictionary that maps characters to indices
                glove_word = Numpy array which provides mapping from word embeddings to word indices
        """
        super(Transformer, self).__init__()
        self.device = args.device
        self.enable_crf = args.enable_crf
        self.idx_pad_tag = args.idx_pad_tag

        self.max_len_word = args.max_len_word

        self.mode_char = args.mode_char
        self.mode_word = args.mode_word

        # embedding layer
        self.embedding_char = nn.Embedding(self.n_char+1, self.dim_emb_char, padding_idx=self.idx_pad_char)
        init_embedding(self.embedding_char)

        if args.enable_pretrained:
            self.embedding_word = nn.Embedding.from_pretrained(torch.FloatTensor(glove_word), freeze=args.freeze_glove,
                                                               padding_idx=self.idx_pad_word)
        else:
            self.embedding_word = nn.Embedding(self.n_word+1, self.dim_emb_word)
            init_embedding(self.embedding_word)

        # character encoder
        if self.mode_char == 'lstm':
            self.lstm_char = nn.LSTM(self.dim_emb_char, self.dim_out_char, num_layers=1, batch_first=True,
                                     bidirectional=True)
            init_lstm(self.lstm_char)
        elif self.mode_char == 'cnn':
            self.conv_char = nn.Conv2d(in_channels=1, out_channels=self.dim_out_char * 2,
                                       kernel_size=(3, self.dim_emb_char), padding=(2, 0))
            init_linear(self.conv_char)
            self.mp_char = nn.MaxPool2d((self.max_len_word + 2, 1))  # padding x 2 - kernel_size + 1
        else:
            raise Exception('Character encoder mode unknown...')
        self.dropout1 = nn.Dropout(args.dropout)


    def forward(self, words_batch, chars_batch, lens_word):
        len_batch, len_sent = words_batch.shape

        # character-level modelling
        emb_chars = self.embedding_char(chars_batch)
        if self.mode_char == 'lstm':
            # covered padded characters that have 0 length to 1
            lens_char = (chars_batch != self.idx_pad_char).sum(dim=2)
            lens_char_covered = torch.where(lens_char == 0, 1, lens_char)
            packed_char = pack_padded_sequence(emb_chars.view(-1, self.max_len_word, self.dim_emb_char),
                                               lens_char_covered.view(-1).cpu(), batch_first=True, enforce_sorted=False)
            out_lstm_char, _ = self.lstm_char(packed_char)

            # return to (len_batch x len_sent x len_char x dim_emb_char)
            output_char, _ = pad_packed_sequence(out_lstm_char, batch_first=True, total_length=self.max_len_word)
            output_char = output_char * lens_char.view(-1, 1, 1).bool()
            output_char = output_char.reshape(len_batch, len_sent, self.max_len_word, self.dim_emb_char*2)

            output_char = torch.cat(
                (torch.stack(
                    [sample[torch.arange(len_sent).long(), lens-1, :self.dim_out_char]
                     for sample, lens in zip(output_char, lens_char)]),
                 torch.stack(
                     [sample[torch.arange(len_sent).long(), lens*0, self.dim_out_char:]
                      for sample, lens in zip(output_char, lens_char)]))
                , dim=-1)

        elif self.mode_char == 'cnn':
            enc_char = self.conv_char(emb_chars.unsqueeze(2).view(-1, 1, self.max_len_word, self.dim_emb_char))
            output_char = self.mp_char(enc_char).view(len_batch, len_sent, self.dim_out_char * 2)
        else:
            raise Exception('Unknown character encoder: '+self.mode_char+'...')

        # load word embeddings
        emb_words = self.embedding_word(words_batch)
        emb_words_chars = torch.cat((emb_words, output_char), dim=-1)
        emb_words_chars = self.dropout1(emb_words_chars)

        # word lstm
        if self.mode_word == 'lstm':
            packed_word = pack_padded_sequence(emb_words_chars, lens_word.cpu(), batch_first=True)
            out_lstm_word, _ = self.lstm_word(packed_word)
            enc_word, _ = pad_packed_sequence(out_lstm_word, batch_first=True)

        elif self.mode_word == 'cnn1':
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        elif self.mode_word == 'cnn2':
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv2(out_cnn_word)
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        elif self.mode_word in ['cnn3', 'cnn_d']:
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv2(out_cnn_word)
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv3(out_cnn_word)
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        else:
            raise Exception('Unknown word encoder: '+self.mode_word+'...')

        outputs = self.hidden2tag(enc_word)
        return outputs


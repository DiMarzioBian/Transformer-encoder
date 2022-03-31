# Transformer

This is a toy PyTorch implementation for the NIPS'17 paper [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

This repo works on next-word prediction task, which is only for learning purposes.


## Training model

Run `python main.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --seed                    random seed
  --device                  device for computing
  --path_data               path of the data corpus
  --path_processed          path to save the filtered processed data
  --path_filtered           optimizer type: Adam, AdamW, RMSprop, Adagrad, SGD
  --path_pretrained         path of the data corpus
  --path_model              path of the trained model
  --num_worker              number of dataloader worker
  --batch_size              batch size
  --epochs                  upper epoch limit
  --es_patience_max         max early stopped patience

  --dim_emb_char            character embedding dimension
  --dim_emb_word            word embedding dimension
  --dim_out_char            character encoder output dimension
  --dim_out_word            word encoder output dimension
  --window_kernel           window width of CNN kernel

  --enable_pretrained       use pretrained glove dimension
  --freeze_glove            free pretrained glove embedding
  --dropout                 dropout rate applied to layers (0 = no dropout)
  --lr                      initial learning rate
  --lr_step                 number of epoch for each lr downgrade
  --lr_gamma                strength of lr downgrade
  --eps_f1                  minimum f1 score difference threshold

  --mode_char               character encoder: lstm or cnn
  --mode_word               word encoder: lstm or cnn1, cnn2, cnn3, cnn_d
  --enable_crf              employ CRF
  --filter_word             filter meaningless words
```
No arguments will run the model in the settings that achieved best result.

## File structure
```bash
----Transformer\
    |----data\
    |    |----wikitext-2\
    |    |    |----data.pkl
    |    |    |----README
    |    |    |----text.txt
    |    |    |----train.txt
    |    |    |----valid.txt
    |    |----data_bundle.pkl
    |    |----data_filtered_bundle.pkl
    |----result\
    |    |----models\
    |    |    |----model.pt
    |----transformer\
    |    |----Layers.py\
    |    |----Models.py\
    |    |----SubLayers.py\
    |----dataloader.py
    |----epoch.py
    |----main.py
    |----plot_attention.py
    |----README.md
    |----utils.py
```

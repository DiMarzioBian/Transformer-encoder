# Transformer-encoder

This is an unofficialtoy PyTorch implementation for the NIPS'17 paper [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

This repo works on next-word prediction task, which is only for learning purposes.

This repo implements decoder layer, but it remains unused in model.

## Training model

Run `python main.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --n_layer                 number of transformer encoder layer
  --d_model                 model feature dimension
  --n_head                  number of attention heads
  --d_inner                 hidden representation size of the feed-forward layer
  --scaled_attn             scale attention in multi-head attention layer
  
  --pad_number              pad all numbers to a same <num>
  --lower_char              lower cases of characters
  --weight_sharing          sharing weights of predictor and embedding:
                              0 -> weight not sharing
                              1 -> weight sharing with learnable bias
                              2 -> weight sharing with no bias
                              others -> embedding inner-product
  
  --n_gram                  max input sequence length
  --num_worker              number of dataloader worker
  --batch_size              batch size
  --epochs                  upper epoch limit
  --dropout                 dropout rate applied to layers (0 = no dropout)
  --lr                      initial learning rate
  --lr_step                 number of epoch for each lr downgrade
  --lr_gamma                strength of lr downgrade
  --eps_loss                  minimum loss difference threshold
  
  --seed                    random seed
  --device                  device for computing
  --path_data               path of the data corpus
  --path_processed          path to save the filtered processed data
  --path_model              path of the trained model
  --num_worker              number of dataloader worker
  --batch_size              batch size
  --epochs                  upper epoch limit
  --es_patience_max         max early stopped patience
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
    |----result\
    |    |----images\
    |    |    |----attn.jpg
    |    |    |----pos_enc.jpg
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

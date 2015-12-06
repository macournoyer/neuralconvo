# Sequence-to-Sequence model for Torch

This is my attempt at implementing [Sequence to Sequence Learning with Neural Networks (seq2seq)](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

And reproduce the results in [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869.pdf).

## Sample conversations

TODO

## Installation

1. [Install Torch](http://torch.ch/docs/getting-started.html)
2. Install the following additional Lua libs:

```sh
luarocks install nn
luarocks install rnn
luarocks install penlight
```

Download the [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html) and extract all the files into data/cornell_movie-dialogs.

## Training

```sh
th train.lua [-h / options]
```

Use the `--dataset NUMBER` option to control the size of the dataset. Training on the full dataset can take about a day for a single epoch.

The model will be saved to `data/model.t7` after an epoch if the model improved(errors decreased).

## Evaluating

```sh
th -i eval.lua
# ...
th> say "Hello there!"
```

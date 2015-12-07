# Sequence-to-Sequence model for Torch

This is my attempt at implementing [Sequence to Sequence Learning with Neural Networks (seq2seq)](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

And reproduce the results in [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869.pdf).

## Sample conversations

Here's a sample conversations after training for 10 epoch with only 5000 examples using the following command:

```sh
$ th train.lua --cuda --dataset 5000 --hiddenSize 1000
```

_(Note: All words are down-cased before training)_

> me: hi
> bot: hey .
> 
> me: what's your name?
> bot: pris .
> 
> me: how old are you?
> bot: thirty five eleven .
> 
> me: what's 2 + 2?
> bot: nothing .
> 
> me: That's funny.
> bot: no .
> 
> me: Where are you from?
> bot: helsinki , there !
> 
> me: That's a nice place.
> bot: yes .
> 
> me: How long have you been living in Helsinki?
> bot: thirty years .
> 
> me: Talk to you later.
> bot: what ?
> 
> me: I'm leaving.
> bot: leaving what ?
> 
> me: Leaving this conversation.
> bot: yes .

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

Use the `--dataset NUMBER` option to control the size of the dataset. Training on the full dataset takes about 5h for a single epoch.

The model will be saved to `data/model.t7` after each epoch if it has improved (error decreased).

## Testing

To load the model and have a conversation:

```sh
th -i eval.lua
# ...
th> say "Hello."
```

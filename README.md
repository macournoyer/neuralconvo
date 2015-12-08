# Sequence-to-Sequence model for Torch

This is an attempt at implementing [Sequence to Sequence Learning with Neural Networks (seq2seq)](http://arxiv.org/abs/1409.3215) and reproducing the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot).

The Google chatbot paper [became famous](http://www.sciencealert.com/google-s-ai-bot-thinks-the-purpose-of-life-is-to-live-forever) after answering "the purpose of life":

> **Human:** What is the purpose of living?
> **Machine:** To live forever.

## Sample conversation

Here's a sample conversations after training for 10 epoch with only 5000 examples, using the following command:

```sh
$ th train.lua --cuda --dataset 5000 --hiddenSize 1000
```

_(Note: All words are down-cased before training)_

> **me:** Hi
> **bot:** Hey.
> 
> **me:** What's your name?
> **bot:** Pris.
> 
> **me:** How old are you?
> **bot:** Thirty five eleven.
> 
> **me:** What's 2 + 2?
> **bot:** Nothing.
> 
> **me:** That's funny.
> **bot:** No.
> 
> **me:** Where are you from?
> **bot:** Helsinki, there!
> 
> **me:** That's a nice place.
> **bot:** Yes.
> 
> **me:** How long have you been living in Helsinki?
> **bot:** Thirty years.
> 
> **me:** Talk to you later.
> **bot:** What?
> 
> **me:** I'm leaving.
> **bot:** Leaving what?
> 
> **me:** Leaving this conversation.
> **bot:** Yes.

The results are far less impressive than in the paper. But this is probably because of the extremely small dataset and small network I used.

Sadly, my graphic card doesn't have enough memory to train larger networks.

If you manage to run it on a larger network do let me know!

## Installing

1. [Install Torch](http://torch.ch/docs/getting-started.html).
2. Install the following additional Lua libs:

   ```sh
   luarocks install nn
   luarocks install rnn
   luarocks install penlight
   ```
   
   To train with CUDA install the latest CUDA drivers, toolkit and run:

   ```sh
   luarocks install cutorch
   luarocks install cunn
   ```

3. Download the [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html) and extract all the files into data/cornell_movie-dialogs.

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

## Credits

Copyright Marc-Andre Cournoyer <macournoyer@gmail.com>.

Thanks to [rnn](https://github.com/Element-Research/rnn), Torch, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [TensorFlow seq2seq tutorial](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html).

# Neural Conversational Model in Torch

This is an attempt at implementing [Sequence to Sequence Learning with Neural Networks (seq2seq)](http://arxiv.org/abs/1409.3215) and reproducing the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot).

The Google chatbot paper [became famous](http://www.sciencealert.com/google-s-ai-bot-thinks-the-purpose-of-life-is-to-live-forever) after cleverly answering a few philosophical questions, such as:

> **Human:** What is the purpose of living?  
> **Machine:** To live forever.

## How it works

The model is based on two [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) layers. One for encoding the input sentence into a "thought vector", and another for decoding that vector into a response. This model is called Sequence-to-sequence or seq2seq.

![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)  
_Source: http://googleresearch.blogspot.ca/2015/11/computer-respond-to-this-email.html_

In this experiment, we train the seq2seq model with movie dialogs from the [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html). The lines are shortened to the first sentence.

## Sample conversation

Here's a sample conversation after training for 10 epoch with only 5000 examples, using the following command:

```sh
th train.lua --cuda --dataset 5000 --hiddenSize 1000
```

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

_(Disclaimer: nonsensical responses have been removed.)_

The results are fun but far less impressive than in the paper. This is probably because of the extremely small dataset and small network I used.

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

3. Download the [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html) and extract all the files into data/cornell_movie_dialogs.

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

## License

MIT License

Copyright (c) 2016 Marc-Andre Cournoyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
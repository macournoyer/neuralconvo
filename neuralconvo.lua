require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

neuralconvo = {}

torch.include('neuralconvo', 'cornell_movie_dialogs.lua')
torch.include('neuralconvo', 'dataset.lua')
torch.include('neuralconvo', 'seq2seq.lua')

return neuralconvo

require 'torch'
require 'nn'
require 'rnn'

neuralconvo = {}

torch.include('neuralconvo', 'cornell_movie_dialogs.lua')
torch.include('neuralconvo', 'dataset.lua')
torch.include('neuralconvo', 'movie_script_parser.lua')
torch.include('neuralconvo', 'seq2seq.lua')

return neuralconvo

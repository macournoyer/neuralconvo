require 'torch'
require 'nn'
require 'rnn'

e = {}

torch.include('e', 'cornell_movie_dialogs.lua')
torch.include('e', 'dataset.lua')
torch.include('e', 'movie_script_parser.lua')
torch.include('e', 'word2vec.lua')

return e
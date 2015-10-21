require 'torch'
require 'rnn'
require 'bot'

-- Load data
local batchSize = 32 -- number of examples per batch
local rho = 5 -- back-propagate through time (BPTT) for rho time-steps



-- Prep model

local model = nn.Sequential()

local inputSize = 300
local hiddenSize = inputSize

model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.Linear(inputSize, data:vocabularySize())))
model:add(nn.Sequencer(nn.LogSoftMax()))

-- will recurse a single continuous sequence
-- model:remember('both')


print(model)

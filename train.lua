require 'torch'
require 'rnn'
require 'e'

-- Load data
local word2vec = e.Word2Vec("data/GoogleNews-vectors-negative300.bin")
local dataset = e.CornellMovieDialogs("data/cornell_movie_dialogs")
local EOS = word2vec:get("</s>")

-- Prep model
local model = nn.Sequential()
local inputSize = 300
local hiddenSize = 500

model:add(nn.FastLSTM(inputSize, hiddenSize))
model:add(nn.FastLSTM(hiddenSize, inputSize))
-- model:add(nn.Linear(inputSize, data:vocabularySize()))
model:add(nn.LogSoftMax())

-- will recurse a single continuous sequence
model:remember('both')

-- Loss function
local criterion = nn.MSECriterion()

-- Train
for i,convertation in ipairs(dataset.convertations) do
  for j,line in ipairs(convertation) do

    -- Inputs
    for t,word in e.tokenize(line.text) do
      local vec = word2vec:get(word)
      if vec ~= nil then
        model:forward(vec)
      else
        print("Vec missing for " .. word)
      end
    end
    model:forward(EOS)

    -- Outputs
    -- ...

  end
end
require 'torch'
require 'rnn'
require 'e'

-- Load data
local batchSize = 32 -- number of examples per batch
local rho = 5 -- back-propagate through time (BPTT) for rho time-steps

-- Prep model
local model = nn.Sequential()

local inputSize = 300
local hiddenSize = 500

model:add(nn.FastLSTM(inputSize, hiddenSize))
model:add(nn.FastLSTM(hiddenSize, inputSize))
-- model:add(nn.Linear(inputSize, data:vocabularySize()))
model:add(nn.LogSoftMax())

-- will recurse a single continuous sequence
-- model:remember('both')

-- print(model)

local word2vec = e.Word2Vec("/Users/ma/Downloads/GoogleNews-vectors-negative300.bin")
local dialogs = e.MovieScriptParser():parse("data/Seinfeld-Good-News,-Bad-News.html")
dialogs = e.PreProcessor():process(dialogs)

local eos = word2vec:get("</s>")

for i,dialog in ipairs(dialogs) do
  for j,speech in ipairs(dialog) do
    for t,word in e.tokenize(speech.text) do
      local vec = word2vec:get(word)
      if vec ~= nil then
        model:forward(vec)
      else
        print("Vec missing for " .. word)
      end
    end
    model:forward(eos)
  end
end
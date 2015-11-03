require 'nn'
require 'rnn'
require 'xlua'
require 'e'

-- Data
-- local dataset = e.DataSet("data/cornell_movie_dialogs.t7",
--                           e.CornellMovieDialogs("data/cornell_movie_dialogs"))
dataset = e.DataSet("data/cornell_movie_dialogs_tiny.t7",
                          e.CornellMovieDialogs("data/cornell_movie_dialogs"), 1000)

EOS = dataset.word2id["</s>"]


-- Model
model = nn.Sequential()
local inputSize = 300
local hiddenSize = inputSize
local dropout = 0.5

model:add(nn.LookupTable(dataset.wordsCount, inputSize))
model:add(nn.SplitTable(1,2))
model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
-- model:add(nn.Sequencer(nn.Dropout(dropout)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, dataset.wordsCount)))
model:add(nn.JoinTable(1,2))
model:add(nn.LogSoftMax())

model:remember('both')

-- print(model)

-- Training
local criterion = nn.ClassNLLCriterion()
local learningRate = 0.05
local momentum = 0.9
local epochCount = 1

for epoch = 1, epochCount do
  print("-- Epoch " .. epoch)

  for i, example in ipairs(dataset.examples) do
    local inputs = example[1]
    local targets = example[2]

    -- seq2seq paper recommends passing input in reverse order
    for i = #inputs, 1, -1 do
      local input = inputs[i]
      model:forward(torch.Tensor{input})
    end

    local input = EOS
    for i = 1, #targets + 1 do
      local target = targets[i] or EOS

      local output = model:forward(torch.Tensor{input})
      local err = criterion:forward(output, torch.Tensor{target})

      local gradOutput = criterion:backward(output, torch.Tensor{target})
      model:backward(torch.Tensor{input}, gradOutput)

      input = target
    end

    model:updateGradParameters(momentum)
    model:updateParameters(learningRate)
    model:zeroGradParameters()

    model:forget()
    xlua.progress(i, #dataset.examples)
  end

  print("-- Epoch done. Saving model")
  torch.save("data/model.t7", model)

end


-- Testing
function output2wordId(t)
  local max = t:max()
  for i = 1, t:size(1) do
    if t[i] == max then
      return i
    end
  end
end

local tokenizer = require "tokenizer"
function say(text)
  local inputs = {}
  for t, word in tokenizer.tokenize(text) do
    local t = dataset.word2id[word:lower()]
    table.insert(inputs, t)
  end

  model:forget()

  for i = #inputs, 1, -1 do
    local input = inputs[i]
    model:forward(torch.Tensor{input})
  end

  local input = EOS
  repeat
    local output = model:forward(torch.Tensor{input})
    io.write(dataset.id2word[output2wordId(output)] .. " ")
    input = output2wordId(output)
  until input == EOS

  print("")
end

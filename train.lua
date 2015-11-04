require 'e'
require 'xlua'

-- Data
-- dataset = e.DataSet("data/cornell_movie_dialogs.t7",
--                     e.CornellMovieDialogs("data/cornell_movie_dialogs"))
dataset = e.DataSet("data/cornell_movie_dialogs_tiny.t7",
                    e.CornellMovieDialogs("data/cornell_movie_dialogs"), 1000)

EOS = torch.IntTensor{dataset.word2id["</s>"]}


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
local epochCount = 5

for epoch = 1, epochCount do
  print("-- Epoch " .. epoch)

  for i, example in ipairs(dataset.examples) do
    local inputs = example[1]
    local targets = example[2]

    -- seq2seq paper recommends passing input in reverse order
    for i = inputs:size(1), 1, -1 do
      local input = inputs[i]
      model:forward(input)
    end

    local input = EOS
    for i = 1, targets:size(1) + 1 do
      local target
      if i > targets:size(1) then
        target = EOS
      else
        target = targets[i]
      end

      local output = model:forward(input)
      local err = criterion:forward(output, target)

      local gradOutput = criterion:backward(output, target)
      model:backward(input, gradOutput)

      input = target
    end

    model:updateGradParameters(momentum)
    model:updateParameters(learningRate)
    model:zeroGradParameters()

    model:forget()
    xlua.progress(i, #dataset.examples)

    -- TODO remove this when training is faster
    if i % 1000 == 0 then
      torch.save("data/model.t7", model)
    end
  end

  print("-- Saving model")
  torch.save("data/model.t7", model)
end

-- Load testing script
require "eval"
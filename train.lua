require 'nn'
require 'rnn'
require 'xlua'
require 'e'

-- Data
local dataset = e.DataSet("data/cornell_movie_dialogs.t7",
                          e.CornellMovieDialogs("data/cornell_movie_dialogs"))

-- Model
local model = nn.Sequential()
local inputSize = 300
local hiddenSize = inputSize
local dropout = 0.5

model:add(nn.LookupTable(dataset.vocabSize, inputSize))
model:add(nn.SplitTable(1,2))
model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
-- model:add(nn.Sequencer(nn.Dropout(dropout)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, #vocab)))
model:add(nn.JoinTable(1,2))
model:add(nn.LogSoftMax())

model:remember('both')

print(model)

-- Training
local criterion = nn.ClassNLLCriterion()
local learningRate = 0.05
local momentum = 0.9
local epochCount = 10
local stepsCount = epochCount * #dataset.examples

for epoch = 1, epochCount do
  for i, example in ipairs(dataset.examples) do
    local inputs = example[1]
    local targets = example[2]
    local input
    local output

    for i, input in ipairs(inputs) do
      output = model:forward(input)
    end
    output = model:forward(input)

    for i, target in ipairs(targets) do
      -- TODO review this whole block to implement seq2seq
      local err = criterion:forward(output, target)

      local gradOutput = criterion:backward(output, target)
      model:backward(input, gradOutput)

      model:updateGradParameters(momentum)
      model:updateParameters(learningRate)
      model:zeroGradParameters()

      output = model:forward(input)
    end

    xlua.progress((epoch - 1) * #dataset.examples + i, stepsCount)
    model:forget()
  end

end
xlua.progress(stepsCount, stepsCount)


-- Testing
-- model:forward(w2t("do"))
-- model:forward(w2t("re"))
-- local output = model:forward(w2t("mi"))
-- io.write("do re mi --> ")
-- for i=1,6 do
--   io.write(t2w(output) .. " ")
--   output = model:forward(w2t(t2w(output)))
-- end
-- print("")

require 'nn'
require 'rnn'
require 'xlua'

-- Data
local dataset = {}

for i,word in ipairs(vocab) do
  local t = torch.Tensor{i}
  table.insert(dataset, t)
end

function t2w(t)
  local max = t:max()
  for i = 1, t:size(1) do
    if t[i] == max then
      return vocab[i]
    end
  end
end

function w2t(word)
  for i,w in ipairs(vocab) do
    if word == w then
      return torch.Tensor{i}
    end
  end
end


-- Model
local model = nn.Sequential()
local inputSize = 100
local hiddenSize = inputSize
local dropout = 0.5

model:add(nn.LookupTable(#vocab, inputSize))
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
local stepsCount = epochCount * #dataset

for epoch = 1, epochCount do
  for i,input in ipairs(dataset) do
    local target = dataset[i+1]
    if target == nil then
      break
    end

    local output = model:forward(input)
    local err = criterion:forward(output, target)

    local gradOutput = criterion:backward(output, target)
    model:backward(input, gradOutput)

    model:updateGradParameters(momentum)
    model:updateParameters(learningRate)
    model:zeroGradParameters()

    xlua.progress((epoch - 1) * #dataset + i, stepsCount)
  end

  model:forget()
end
xlua.progress(stepsCount, stepsCount)


-- Testing
model:forward(w2t("do"))
model:forward(w2t("re"))
local output = model:forward(w2t("mi"))
io.write("do re mi --> ")
for i=1,6 do
  io.write(t2w(output) .. " ")
  output = model:forward(w2t(t2w(output)))
end
print("")

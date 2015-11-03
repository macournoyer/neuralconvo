require 'e'

-- dataset = e.DataSet("data/cornell_movie_dialogs.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"))
-- dataset = e.DataSet("data/cornell_movie_dialogs_small.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"), 10000)
-- dataset = e.DataSet("data/cornell_movie_dialogs_tiny.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"), 1000)

require 'nn'
require 'rnn'
require 'xlua'

-- Data
local vocab = { "</s>", "do", "re", "mi", "fa", "sol", "la", "si", "do" }

function output2w(t)
  local max = t:max()
  for i = 1, t:size(1) do
    if t[i] == max then
      return vocab[i]
    end
  end
end

function w2id(word)
  for i,w in ipairs(vocab) do
    if word == w then
      return i
    end
  end
end

function id2w(id)
  return vocab[id]
end

function w2t(word)
  return torch.Tensor{w2id(word)}
end

function t2w(t)
  return id2w(t[1])
end

local examples = {
  { {w2t("do"), w2t("re"), w2t("mi"), w2t("fa")},
    {w2t("sol"), w2t("la"), w2t("si"), w2t("do")} },

  { {w2t("do"), w2t("mi"), w2t("sol")},
    {w2t("re"), w2t("mi"), w2t("fa")} }
}

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

-- print(model)

-- Training
local criterion = nn.ClassNLLCriterion()
local learningRate = 0.05
local momentum = 0.9
local epochCount = 100

function log(str)
  -- print(str)
end

for epoch = 1, epochCount do
  for i,example in ipairs(examples) do
    local inputs = example[1]
    local targets = example[2]

    for i = #inputs, 1, -1 do
      local input = inputs[i]
      log(t2w(input) .. "  -->  ")
      model:forward(input)
    end

    local input = w2t("</s>")
    for i = 1, #targets + 1 do
      local target = targets[i] or w2t("</s>")
      log(t2w(input) .. "  -->  " .. t2w(target))

      local output = model:forward(input)
      local err = criterion:forward(output, target)

      local gradOutput = criterion:backward(output, target)
      model:backward(input, gradOutput)

      input = target
    end

    log("---")

    model:updateGradParameters(momentum)
    model:updateParameters(learningRate)
    model:zeroGradParameters()

    model:forget()
  end
end


-- Testing
model:forward(w2t("fa"))
model:forward(w2t("mi"))
model:forward(w2t("re"))
model:forward(w2t("do"))
local output = model:forward(w2t("</s>"))
io.write("do re mi fa </s> --> ")
for i=1,5 do
  io.write(output2w(output) .. " ")
  output = model:forward(w2t(output2w(output)))
end
print("")

model:forget()

model:forward(w2t("sol"))
model:forward(w2t("mi"))
model:forward(w2t("do"))
local output = model:forward(w2t("</s>"))
io.write("do mi sol </s> --> ")
for i=1,4 do
  io.write(output2w(output) .. " ")
  output = model:forward(w2t(output2w(output)))
end
print("")

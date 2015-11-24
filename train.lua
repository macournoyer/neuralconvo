require 'e'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'size of dataset to use (0 = all)')
cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Data
dataset = e.DataSet("data/cornell_movie_dialogs_" .. (options.dataset or "full") .. ".t7",
                    e.CornellMovieDialogs("data/cornell_movie_dialogs"), options.dataset)

-- Model
local hiddenSize = 300
model = e.Seq2Seq(dataset.wordsCount, hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = 0.5
model.momentum = 0.9
local epochCount = 3
local minErr = 0.1
local err = 0

for epoch = 1, epochCount do
  print("-- Epoch " .. epoch .. " / " .. epochCount)

  for i,example in ipairs(dataset.examples) do
    local e = model:train(unpack(example))
    err = math.max(err, e)
    xlua.progress(i, #dataset.examples)
  end

  if err < minErr then
    print("Breaking at error " .. err)
    break
  end

  print("Max error: " .. err)

  print("-- Saving model")
  torch.save("data/model.t7", model)
end

-- Load testing script
require "eval"
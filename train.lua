require 'e'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'size of dataset to use (0 = all)')
cmd:option('--cuda', false, 'Use CUDA')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')

cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Data
dataset = e.DataSet("data/cornell_movie_dialogs_" .. (options.dataset or "full") .. ".t7",
                    e.CornellMovieDialogs("data/cornell_movie_dialogs"), options.dataset)

-- Model
model = e.Seq2Seq(dataset.wordsCount, options.hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch

-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  dataset:cuda()
  model:cuda()
end

-- Run the experiment

for epoch = 1, options.maxEpoch do
  print("-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("Learning rate: " .. model.learningRate)

  local errors = torch.Tensor(#dataset.examples):zero()

  for i,example in ipairs(dataset.examples) do
    local err = model:train(unpack(example))
    errors[i] = err
    xlua.progress(i, #dataset.examples)
  end

  print("Error:      min=" .. errors:min() .. " max=" .. errors:max() ..
                   " median=" .. errors:median()[1] .. " mean=" .. errors:mean())
  print("Perplexity: " .. torch.exp(errors:mean()))

  print("Saving model ...")
  torch.save("data/model.t7", model)

  model.learningRate = model.learningRate + decayFactor
  model.learningRate = math.max(options.minLR, model.learningRate)

  collectgarbage()
end

-- Load testing script
require "eval"
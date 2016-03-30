require 'neuralconvo'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--minWordFreq', 1, 'minimum frequency of words kept in vocab')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--opencl', false, 'use opencl')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 1000, 'number of examples to load at once')

cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Data
print("-- Loading dataset")
dataset = neuralconvo.DataSet(neuralconvo.CornellMovieDialogs("data/cornell_movie_dialogs"),
                    {
                      loadFirst = options.dataset,
                      minWordFreq = options.minWordFreq
                    })

print("\nDataset stats:")
print("  Vocabulary size: " .. dataset.wordsCount)
print("         Examples: " .. dataset.examplesCount)

-- Model
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  model:cuda()
elseif options.opencl then
  require 'cltorch'
  require 'clnn'
  model:cl()
end


-- Run the experiment

for epoch = 1, options.maxEpoch do
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")

  local errors = torch.Tensor(dataset.examplesCount):fill(0)
  local timer = torch.Timer()

  local i = 1
  for examples in dataset:batches(options.batchSize) do
    collectgarbage()

    for _, example in ipairs(examples) do
      local input, target = unpack(example)

      if options.cuda then
        input = input:cuda()
        target = target:cuda()
      elseif options.opencl then
        input = input:cl()
        target = target:cl()
      end

      local err = model:train(input, target)

      -- Check if error is NaN. If so, it's probably a bug.
      if err ~= err then
        error("Invalid error! Exiting.")
      end

      errors[i] = err
      xlua.progress(i, dataset.examplesCount)
      i = i + 1
    end
  end

  timer:stop()

  print("\nFinished in " .. xlua.formatTime(timer:time().real) .. " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. model.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())

  -- Save the model if it improved.
  if minMeanError == nil or errors:mean() < minMeanError then
    print("\n(Saving model ...)")
    torch.save("data/model.t7", model)
    minMeanError = errors:mean()
  end

  model.learningRate = model.learningRate + decayFactor
  model.learningRate = math.max(options.minLR, model.learningRate)
end

-- Load testing script
require "eval"

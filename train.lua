require 'neuralconvo'
require 'xlua'
require 'optim'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--minWordFreq', 1, 'minimum frequency of words kept in vocab')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--opencl', false, 'use opencl')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.001, 'learning rate at t=0')
cmd:option('--gradientClipping', 5, 'clip gradients at this value')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 10, 'number of examples to load at once')

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
if options.batchSize > 1 then
  model.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
else
  model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
end
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

-- Define optimizer

local nextBatch = dataset:batches(options.batchSize)

local params, gradParams = model:getParameters()
    
local optimState = {learningRate=model.learningRate}
  
local function feval(x)
  if x ~= params then
    params:copy(x)
  end
  
  gradParams:zero()
  
  local encoderInputs, decoderInputs, decoderTargets = nextBatch()
  
  if options.cuda then
    encoderInputs = encoderInputs:cuda()
    decoderInputs = decoderInputs:cuda()
    decoderTargets = decoderTargets:cuda()
  elseif options.opencl then
    encoderInputs = encoderInputs:cl()
    decoderInputs = decoderInputs:cl()
    decoderTargets = decoderTargets:cl()
  end

  -- Forward pass
  local encoderOutput = model.encoder:forward(encoderInputs)
  model:forwardConnect(encoderInputs:size(1))
  local decoderOutput = model.decoder:forward(decoderInputs)
  local loss = model.criterion:forward(decoderOutput, decoderTargets)
  
  avgSeqLen = torch.sum(torch.sign(decoderInputs)) / decoderInputs:size(2)
  loss = loss / avgSeqLen
  
  -- Backward pass
  local dloss_doutput = model.criterion:backward(decoderOutput, decoderTargets)
  model.decoder:backward(decoderInputs, dloss_doutput)
  model:backwardConnect()
  model.encoder:backward(encoderInputs, encoderOutput:zero())
  
  gradParams:clamp(-options.gradientClipping, options.gradientClipping)
  
  return loss,gradParams
end



-- Run the experiment

for epoch = 1, options.maxEpoch do
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")

  local errors = {}
  local timer = torch.Timer()

  for i=1, dataset.examplesCount/options.batchSize do
    collectgarbage()
    
    
    local _,tloss = optim.adam(feval, params, optimState)
    err = tloss[1] -- optim returns a list

  
    model.decoder:forget()
    model.encoder:forget()

    table.insert(errors,err)
    xlua.progress(i * options.batchSize, dataset.examplesCount)
  end
  nextBatch = dataset:batches(options.batchSize)

  timer:stop()
  
  errors = torch.Tensor(errors)
  print("\nFinished in " .. xlua.formatTime(timer:time().real) .. " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. model.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())
  print("          ppl= " .. torch.exp(errors:mean()))

  -- Save the model if it improved.
  if minMeanError == nil or errors:mean() < minMeanError then
    print("\n(Saving model ...)")
    model:float()
    torch.save("data/model.t7", model) -- model is saved by default as cpu
    if options.cuda then
      model:cuda()
    elseif options.opencl then
      model:cl()
    end
    minMeanError = errors:mean()
  end

  --model.learningRate = model.learningRate + decayFactor
  --model.learningRate = math.max(options.minLR, model.learningRate)
end

-- Load testing script
require "eval"

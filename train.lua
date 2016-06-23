--require("mobdebug").start()
require 'neuralconvo'
require 'xlua'
require 'optim'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--valSetSize', 0.05, 'percentage of validation data')
cmd:option('--vocabSize', -1, 'size of the vocabulary')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--opencl', false, 'use opencl')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--numLayers', 1, 'number of LSTM layers')
cmd:option('--learningRate', 0.001, 'learning rate at t=0')
cmd:option('--gradientClipping', 5, 'clip gradients at this value')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 10, 'number of examples to load at once')
cmd:option('--weightDecay', 0.001, 'weightDecay')

cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
  options.dataset = nil
end

-- Data
print("-- Loading dataset")
if not path.exists("data/cornell_movie_dialogs/contextResponse.csv") then
  neuralconvo.CornellMovieDialogs("data/cornell_movie_dialogs"):load()
end
dataset = neuralconvo.DataSet("data/cornell_movie_dialogs/contextResponse.csv",options)
dataset:load()

print("\nDataset stats:")
print("  Vocabulary size: " .. dataset.wordsCount)
print("         Examples: " .. dataset.examplesCount)

-- Model
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize, options.numLayers)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
if options.batchSize > 1 then
  model.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
else
  model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
end


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


-- validation function
function eval_val(vmodel,val_data)
  print "\n-- Eval on validation.. "
  local nextBatch = dataset:batches(val_data,options.batchSize)
  local batches_loss = {}
  for i=1, (#val_data)/options.batchSize+1 do
    local encoderInputs, decoderInputs, decoderTargets = nextBatch()
    if encoderInputs == nil then break end
    
    if options.cuda then
      encoderInputs = encoderInputs:cuda()
      decoderInputs = decoderInputs:cuda()
      decoderTargets = decoderTargets:cuda()
    elseif options.opencl then
      encoderInputs = encoderInputs:cl()
      decoderInputs = decoderInputs:cl()
      decoderTargets = decoderTargets:cl()
    end
    
    local lloss = vmodel:evalLoss(encoderInputs, decoderInputs, decoderTargets)
    table.insert(batches_loss,lloss)
    xlua.progress(i*options.batchSize,#val_data)
  end
  return torch.Tensor(batches_loss):mean()
end

-- Run the experiment

for epoch = 1, options.maxEpoch do

-- Define optimizer
  collectgarbage()

  dataset:shuffleExamples()
  local nextBatch = dataset:batches(dataset.examples,options.batchSize)

  local params, gradParams = model:getParameters()
      
  local optimState = {learningRate=options.learningRate,momentum=options.momentum}
    
  local function feval(x)
    
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
    
    local avgSeqLen = nil
    if #decoderInputs:size() == 1 then
      avgSeqLen = decoderInputs:size(1)
    else
      avgSeqLen = torch.sum(torch.sign(decoderInputs)) / decoderInputs:size(2)
    end
    loss = loss / avgSeqLen
    
    -- Backward pass
    local dloss_doutput = model.criterion:backward(decoderOutput, decoderTargets)
    model.decoder:backward(decoderInputs, dloss_doutput)
    model:backwardConnect()
    model.encoder:backward(encoderInputs, encoderOutput:zero())
    
    gradParams:clamp(-options.gradientClipping, options.gradientClipping)
    
    return loss,gradParams
  end

  -- run epoch
  
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")

  local errors = {}
  local timer = torch.Timer()

  for i=1, dataset.examplesCount/options.batchSize do
    collectgarbage()
    
    --local diff,dC,dC_est = optim.checkgrad(feval,params)
    
    local _,tloss = optim.adam(feval, params, optimState)
    --cutorch.synchronize()
    err = tloss[1] -- optim returns a list

  
    model.decoder:forget()
    model.encoder:forget()

    table.insert(errors,err)
    xlua.progress(i * options.batchSize, dataset.examplesCount)
  end
  cutorch.synchronize()

  timer:stop()
  
  local val_loss = eval_val(model,dataset.devExamples)
  
  errors = torch.Tensor(errors)
  print("\nFinished in " .. xlua.formatTime(timer:time().real) .. " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. optimState.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())
  print("          ppl= " .. torch.exp(errors:mean()))
  print("     val loss= " .. val_loss)
  print("      val ppl= " .. torch.exp(val_loss))



  -- Save the model if it improved.
  if minMeanError == nil or val_loss < minMeanError then
    print("\n(Saving model ...)")
    params, gradParams = nil,nil
    collectgarbage()
    model:float()
    collectgarbage()
    torch.save("data/model.t7", model) -- model is saved by default as cpu
    collectgarbage()
    if options.cuda then
      model:cuda()
    elseif options.opencl then
      model:cl()
    end
    minMeanError = val_loss
  end

  -- optimState.learningRate = optimState.learningRate + decayFactor
  -- optimState.learningRate = math.max(options.minLR, optimState.learningRate)
end

-- Load testing script
require "eval"

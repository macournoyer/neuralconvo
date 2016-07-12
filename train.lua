require 'neuralconvo'
require 'xlua'
require 'optim'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--valSetSize', 0.05, 'percentage of validation data')
cmd:option('--earlyStopOnTrain', false, 'early stop based training loss (default=val loss)')
cmd:option('--vocabSize', -1, 'size of the vocabulary')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--opencl', false, 'use opencl')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--seqLstm', false, 'Use SeqLSTM? (allows more hidden units)')
cmd:option('--numLayers', 1, 'number of LSTM layers')
cmd:option('--learningRate', 0.001, 'learning rate at t=0')
cmd:option('--gradientClipping', 5, 'clip gradients at this value')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 10, 'mini-batch size')
cmd:option('--weightDecay', 0.001, 'Weight decay aka L2 regularization')
cmd:option('--dropout', 0.2, 'dropout regularization (0=none)')

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
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize, options.numLayers,options)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil,false),1))


local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  model:cuda()
  model.criterion:cuda()
elseif options.opencl then
  require 'cltorch'
  require 'clnn'
  model:cl()
  model.criterion:cl()
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
  collectgarbage()

  dataset:shuffleExamples()

  local nextBatch = dataset:batches(dataset.examples,options.batchSize)

  local params, gradParams = model:getParameters()
      
  local optimState = {learningRate=options.learningRate,momentum=options.momentum}

  model:training() -- set flag for dropout
  -- Define closure for optimizer  
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
    
    loss = loss / torch.sign(decoderInputs):sum()
    
    -- Backward pass
    local dloss_doutput = model.criterion:backward(decoderOutput, decoderTargets)
    model.decoder:backward(decoderInputs, dloss_doutput)
    model:backwardConnect()
    model.encoder:backward(encoderInputs, encoderOutput:zero())
    
    gradParams:clamp(-options.gradientClipping, options.gradientClipping)
    
    return loss,gradParams
  end

  -- run epoch
  
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch ..
    "  (LR= " .. optimState.learningRate .. ")")
  print("")

  local errors,gradNorms = {},{}
  local timer = torch.Timer()

  for i=1, dataset.examplesCount/options.batchSize do
    collectgarbage()
    
    local _,tloss = optim.adam(feval, params, optimState)
    --cutorch.synchronize()
    err = tloss[1] -- optim returns a list
  
    model.decoder:forget()
    model.encoder:forget()

    table.insert(errors,err)
    table.insert(gradNorms,gradParams:norm())
    xlua.progress(i * options.batchSize, dataset.examplesCount)
  end
  cutorch.synchronize()

  xlua.progress(dataset.examplesCount, dataset.examplesCount)
  timer:stop()
  
  local val_loss = eval_val(model,dataset.devExamples)
  errors = torch.Tensor(errors)
  
  print("\n\nFinished in " .. xlua.formatTime(timer:time().real) ..
    " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
  local train_loss = errors:mean()
  
  print("\nEpoch stats:")
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. train_loss)
  print("          std= " .. errors:std())
  print("          ppl= " .. torch.exp(train_loss))
  print("     val loss= " .. val_loss)
  print("      val ppl= " .. torch.exp(val_loss))
  --print(" gradNorm avg= " .. torch.Tensor(gradNorms):mean())
  
  local earlyStopLoss = val_loss
  if options.earlyStopOnTrain then
    earlyStopLoss = train_loss
  end
  
  -- Save the model if it improved.
  if minMeanError == nil or earlyStopLoss < minMeanError then
    print("\n(Saving model ...)")
    params, gradParams, optimState, feval = nil,nil,nil,nil
    collectgarbage()
    -- Model is saved as CPU
    model:float()
    model.criterion:float()
    collectgarbage()
    torch.save("data/model.t7", model) -- model is saved by default as cpu
    collectgarbage()
    if options.cuda then
      model:cuda()
      model.criterion:cuda()
    elseif options.opencl then
      model:cl()
      model.criterion:cl()
    end
    minMeanError = earlyStopLoss
  end

  -- # adam optimizer take cares of learning rate decay
  -- optimState.learningRate = optimState.learningRate + decayFactor 
  -- optimState.learningRate = math.max(options.minLR, optimState.learningRate)
end

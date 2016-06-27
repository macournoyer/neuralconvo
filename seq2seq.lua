-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize, numLayers, options)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")
  local optOptions = options or {}
  self.numLayers = numLayers or 1
  self.dropout = optOptions.dropout or 0
  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  
  self.encLstmLayers = {}
  for i=1,self.numLayers do
    self.encLstmLayers[i] = nn.FastLSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
    self.encoder:add(nn.Sequencer(self.encLstmLayers[i]))
    self.encoder:add(nn.Dropout(self.dropout))
  end
  
  self.encoder:add(nn.Select(1,-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  
  self.decLstmLayers = {}
  for i=1,self.numLayers do
    self.decLstmLayers[i] = nn.FastLSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
    self.decoder:add(nn.Sequencer(self.decLstmLayers[i]))
    self.decoder:add(nn.Dropout(self.dropout))
  end
  
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(self.hiddenSize, self.vocabSize),1)))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  for i=1,self.numLayers do
    self.decLstmLayers[i].userPrevOutput =
      nn.rnn.recursiveCopy(self.decLstmLayers[i].userPrevOutput, self.encLstmLayers[i].outputs[inputSeqLen])
    self.decLstmLayers[i].userPrevCell =
      nn.rnn.recursiveCopy(self.decLstmLayers[i].userPrevCell, self.encLstmLayers[i].cells[inputSeqLen])
  end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  for i=1,self.numLayers do
    self.encLstmLayers[i].userNextGradCell =
      nn.rnn.recursiveCopy(self.encLstmLayers[i].userNextGradCell, self.decLstmLayers[i].userGradPrevCell)
    self.encLstmLayers[i].gradPrevOutput =
      nn.rnn.recursiveCopy(self.encLstmLayers[i].gradPrevOutput, self.decLstmLayers[i].userGradPrevOutput)
  end
end

function Seq2Seq:training()
  self.encoder:training()
  self.decoder:training()
end

function Seq2Seq:evaluate()
  self.encoder:evaluate()
  self.decoder:evaluate()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end
end

function Seq2Seq:float()
  self.encoder:float()
  self.decoder:float()

  if self.criterion then
    self.criterion:float()
  end
end

function Seq2Seq:cl()
  self.encoder:cl()
  self.decoder:cl()

  if self.criterion then
    self.criterion:cl()
  end
end

function Seq2Seq:getParameters()
  return nn.Container():add(self.encoder):add(self.decoder):getParameters()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLSTM.outputs[inputSeqLen])
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLSTM.cells[inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
  self:evaluate()
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = {self.goToken}
  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.decoder:forward(torch.Tensor(output))[#output]
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    local prob, wordIds = prediction:topk(5, 1, true, true)

    -- First one is the most likely.
    next_output = wordIds[1]
    table.insert(output, next_output)

    -- Terminate on EOS token
    if next_output == self.eosToken then
      break
    end

    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return predictions, probabilities
end

function Seq2Seq:evalLoss(encoderInputs, decoderInputs, decoderTargets)
  self:evaluate()
  -- Forward pass
  local encoderOutput = self.encoder:forward(encoderInputs)
  self:forwardConnect(encoderInputs:size(1))
  local decoderOutput = self.decoder:forward(decoderInputs)
  local loss = self.criterion:forward(decoderOutput, decoderTargets)
  
  loss = loss / torch.sign(decoderInputs):sum()
  
  self.decoder:forget()
  self.encoder:forget()
  return loss
end

-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.encoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
  self.encoder:add(nn.Sequencer(self.encoderLSTM))
  self.encoder:add(nn.Select(1,-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(self.hiddenSize, self.vocabSize),1)))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end
end

function Seq2Seq:cl()
  self.encoder:cl()
  self.decoder:cl()

  if self.criterion then
    self.criterion:cl()
  end
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

function Seq2Seq:train(encoderInputs, decoderInputs, decoderTargets)

  -- Forward pass
  local encoderOutput = self.encoder:forward(encoderInputs)
  self:forwardConnect(encoderInputs:size(1))
  local decoderOutput = self.decoder:forward(decoderInputs)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTargets)

  if Edecoder ~= Edecoder then -- Exist early on bad error
    return Edecoder
  end

  -- Backward pass
  local gEdec = self.criterion:backward(decoderOutput, decoderTargets)
  self.decoder:backward(decoderInputs, gEdec)
  self:backwardConnect()
  self.encoder:backward(encoderInputs, encoderOutput:zero())

  self.encoder:updateGradParameters(self.momentum)
  self.decoder:updateGradParameters(self.momentum)
  self.decoder:updateParameters(self.learningRate)
  self.encoder:updateParameters(self.learningRate)
  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.decoder:forget()
  self.encoder:forget()

  return Edecoder
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
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

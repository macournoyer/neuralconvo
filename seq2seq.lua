-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("e.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.encoder:add(nn.SplitTable(1, 2))
  self.encoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.encoder:add(nn.Sequencer(self.encoderLSTM))
  self.encoder:add(nn.SelectTable(-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.decoder:add(nn.SplitTable(1, 2))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.decoder:add(nn.Sequencer(nn.LogSoftMax()))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.zeroTensor = torch.Tensor(2):zero()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end

  self.zeroTensor = self.zeroTensor:cuda()
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

function Seq2Seq:train(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- TODO pad & batch data
  -- local encoderInput = torch.cat(input, input, 2):transpose(1, 2)
  -- local decoderInput = torch.cat(target:sub(1, -2), target:sub(1, -2), 2):transpose(1, 2)
  -- local decoderTarget = torch.cat(target:sub(2, -1), target:sub(2, -1), 2):transpose(1, 2)

  -- Split for batching
  decoderTarget = nn.SplitTable(1, 1):forward(decoderTarget)

  -- Forward pass
  self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

  -- Backward pass
  local gEdec = self.criterion:backward(decoderOutput, decoderTarget)
  self.decoder:backward(decoderInput, gEdec)
  self:backwardConnect()
  -- TODO this seems off! Why a zero tensor?
  self.encoder:backward(encoderInput, self.zeroTensor)

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

local function argmax(t)
  local max = t:max()
  for i = 1, t:size(1) do
    if t[i] == max then
      return i
    end
  end
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = self.goToken
  local outputs = {}
  for i = 1, MAX_OUTPUT_SIZE do
    local predictions = self.decoder:forward(torch.Tensor{output})
    output = argmax(predictions[1])
    if output == self.eosToken then
      break
    end
    table.insert(outputs, output)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return outputs
end

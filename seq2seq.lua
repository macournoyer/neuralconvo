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
  self.encoderLSTM = nn.SeqLSTM(self.hiddenSize, self.hiddenSize)
  self.encoderLSTM:maskZero()
  self.encoder:add(self.encoderLSTM)
  self.encoder:add(nn.Select(1,-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.decoderLSTM = nn.SeqLSTM(self.hiddenSize, self.hiddenSize)
  self.decoderLSTM:maskZero()
  self.decoder:add(self.decoderLSTM)
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
  self.decoderLSTM.userPrevOutput = self.encoderLSTM.output[inputSeqLen]
  self.decoderLSTM.userPrevCell = self.encoderLSTM.cell[inputSeqLen]
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell = self.decoderLSTM.userGradPrevCell
  self.encoderLSTM.gradPrevOutput = self.decoderLSTM.userGradPrevOutput
end

local MAX_OUTPUT_SIZE = 25

function Seq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")
  assert(input:size(1) <= MAX_OUTPUT_SIZE, "Input too big")

  local input_len = input:size(1)

  -- Resize and pad input to MAX_OUTPUT_SIZE
  input:resize(MAX_OUTPUT_SIZE, 1)
  input:sub(input_len + 1, MAX_OUTPUT_SIZE):zero()
  input = input:t()

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = {self.goToken}
  for i = 1, MAX_OUTPUT_SIZE do
    -- Resize and pad output to MAX_OUTPUT_SIZE
    local t = torch.Tensor(output):resize(MAX_OUTPUT_SIZE, 1)
    t:sub(#output + 1, MAX_OUTPUT_SIZE):zero()
    t = t:t()

    local prediction = self.decoder:forward(t)[1][#output]
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

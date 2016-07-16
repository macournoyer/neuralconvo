--[[
Format movie dialog data as a table of line 1:

  { {word_ids of character1}, {word_ids of character2} }

Then flips it around and get the dialog from the other character's perspective:

  { {word_ids of character2}, {word_ids of character1} }

Also builds the vocabulary.
]]--

local DataSet = torch.class("neuralconvo.DataSet")
local xlua = require "xlua"
local tokenizer = require "tokenizer"
local list = require "pl.List"
local utils = require "pl.utils"
local function_arg = utils.function_arg

function DataSet:__init(loader, options)
  options = options or {}

  self.examplesFilename = "data/examples.t7"

  self.createNewVocabAndExamples = options.createNewVocabAndExamples

  -- Reject words once vocab size reaches this threshold
  self.maxVocabSize = options.maxVocabSize or 0

  -- Maximum number of words in an example sentence
  self.maxExampleLen = options.maxExampleLen or 25

  -- Load only first fews examples (approximately)
  self.loadFirst = options.loadFirst

  self.examples = {}
  self.word2id = {}
  self.id2word = {}
  self.wordsCount = 0

  self:load(loader)
end

function DataSet:buildVocab(conversations)

  print("-- Building vocab")

  -- Add magic tokens
  self.goToken = self:makeWordId("<go>") -- Start of sequence
  self.eosToken = self:makeWordId("<eos>") -- End of sequence
  self.unknownToken = self:makeWordId("<unknown>") -- Word dropped from vocabulary

  self.wordFreqs = {}

  -- number of conversations to be traversed
  local total = self.loadFirst or #conversations

  -- traverse all the conversations to count the frequency of words
  for i, conversation in ipairs(conversations) do
    if i > total then break end
    for j = 1, #conversation do
      local conversationLine = conversation[j]
      -- accumulate the word frequency
      self:countWords(conversationLine.text)
    end
    if i % 1000 == 0 then
      xlua.progress(i,total)
    end
  end

  -- sort the words on their frequencies
  local sortedCounts = f_sortv(self.wordFreqs,function(x,y) return x>y end)

  for word,freq in sortedCounts do
      nWordId = self:addWordToVocab(word)
      if self.maxVocabSize > 0 and nWordId >= self.maxVocabSize then
        break
      end
  end

  print("-- Vocab built")

end

function DataSet:load(loader)
  local filename = "data/vocab.t7"

  if not self.createNewVocabAndExamples and path.exists(filename) then
    print("Loading vocabulary from " .. filename .. " ...")
    local data = torch.load(filename)
    self.word2id = data.word2id
    self.id2word = data.id2word
    self.wordsCount = data.wordsCount
    self.goToken = data.goToken
    self.eosToken = data.eosToken
    self.unknownToken = data.unknownToken
    self.examplesCount = data.examplesCount
  else
    print("" .. filename .. " not found")
    local conversations = loader:load()
    self:buildVocab(conversations)
    self:visit(conversations)
    print("Writing " .. filename .. " ...")
    torch.save(filename, {
      word2id = self.word2id,
      id2word = self.id2word,
      wordsCount = self.wordsCount,
      goToken = self.goToken,
      eosToken = self.eosToken,
      unknownToken = self.unknownToken,
      examplesCount = self.examplesCount
    })
  end
end

function DataSet:visit(conversations)
  self.examples = {}

  print("-- Pre-processing data")

  local total = self.loadFirst or #conversations * 2

  for i, conversation in ipairs(conversations) do
    if i > total then break end
    self:visitConversation(conversation)
    xlua.progress(i, total)
  end

  -- Revisit from the perspective of 2nd character
  for i, conversation in ipairs(conversations) do
    if #conversations + i > total then break end
    self:visitConversation(conversation, 2)
    xlua.progress(#conversations + i, total)
  end

  print("-- Shuffling ")
  newIdxs = torch.randperm(#self.examples)
  local sExamples = {}
  for i, sample in ipairs(self.examples) do
    sExamples[i] = self.examples[newIdxs[i]]
  end
  self.examples = sExamples

  self.examplesCount = #self.examples
  self:writeExamplesToFile()
  self.examples = nil

  collectgarbage()
end

function DataSet:writeExamplesToFile()
  print("Writing " .. self.examplesFilename .. " ...")
  local file = torch.DiskFile(self.examplesFilename, "w")

  for i, example in ipairs(self.examples) do
    file:writeObject(example)
    xlua.progress(i, #self.examples)
  end

  file:close()
end

function DataSet:batches(size)
  local file = torch.DiskFile(self.examplesFilename, "r")
  file:quiet()
  local done = false

  return function()
    if done then
      return
    end

    local inputSeqs,targetSeqs = {},{}
    local maxInputSeqLen,maxTargetOutputSeqLen = 0,0

    for i = 1, size do
      local example = file:readObject()
      if example == nil then
        done = true
        file:close()
        return examples
      end
      inputSeq,targetSeq = unpack(example)
      if inputSeq:size(1) > maxInputSeqLen then
        maxInputSeqLen = inputSeq:size(1)
      end
      if targetSeq:size(1) > maxTargetOutputSeqLen then
        maxTargetOutputSeqLen = targetSeq:size(1)
      end
      table.insert(inputSeqs, inputSeq)
      table.insert(targetSeqs, targetSeq)
    end

    local encoderInputs,decoderInputs,decoderTargets = nil,nil,nil
    if size == 1 then
      encoderInputs = torch.IntTensor(maxInputSeqLen):fill(0)
      decoderInputs = torch.IntTensor(maxTargetOutputSeqLen-1):fill(0)
      decoderTargets = torch.IntTensor(maxTargetOutputSeqLen-1):fill(0)
    else
      encoderInputs = torch.IntTensor(maxInputSeqLen,size):fill(0)
      decoderInputs = torch.IntTensor(maxTargetOutputSeqLen-1,size):fill(0)
      decoderTargets = torch.IntTensor(maxTargetOutputSeqLen-1,size):fill(0)
    end

    for samplenb = 1, #inputSeqs do
      for word = 1,inputSeqs[samplenb]:size(1) do
        eosOffset = maxInputSeqLen - inputSeqs[samplenb]:size(1) -- for left padding
        if size == 1 then
          encoderInputs[word] = inputSeqs[samplenb][word]
        else
          encoderInputs[word+eosOffset][samplenb] = inputSeqs[samplenb][word]
        end
      end
    end

    for samplenb = 1, #targetSeqs do
      trimmedEosToken = targetSeqs[samplenb]:sub(1,-2)
      for word = 1, trimmedEosToken:size(1) do
        if size == 1 then
          decoderInputs[word] = trimmedEosToken[word]
        else
          decoderInputs[word][samplenb] = trimmedEosToken[word]
        end
      end
    end

    for samplenb = 1, #targetSeqs do
      trimmedGoToken = targetSeqs[samplenb]:sub(2,-1)
      for word = 1, trimmedGoToken:size(1) do
        if size == 1 then
          decoderTargets[word] = trimmedGoToken[word]
        else
          decoderTargets[word][samplenb] = trimmedGoToken[word]
        end
      end
    end

    return encoderInputs,decoderInputs,decoderTargets
  end
end

function DataSet:visitConversation(lines, start)
  start = start or 1

  for i = start, #lines, 2 do
    local input = lines[i]
    local target = lines[i+1]

    if target then
      local inputIds = self:visitText(input.text)
      local targetIds = self:visitText(target.text, 2)

      if inputIds and targetIds then
        -- Revert inputs
        inputIds = list.reverse(inputIds)

        table.insert(targetIds, 1, self.goToken)
        table.insert(targetIds, self.eosToken)

        table.insert(self.examples, { torch.IntTensor(inputIds), torch.IntTensor(targetIds) })
      end
    end
  end
end

function DataSet:visitText(text, additionalTokens)
  local words = {}
  additionalTokens = additionalTokens or 0

  if text == "" then
    return
  end

  for t, word in tokenizer.tokenize(text) do
    local cWord = self.word2id[word:lower()]
    if not cWord then
      cWord = self.unknownToken
    end
    table.insert(words, cWord)
    -- Only keep the first sentence
    if t == "endpunct" or #words >= self.maxExampleLen - additionalTokens then
      break
    end
  end

  if #words == 0 then
    return
  end

  return words
end

function DataSet:countWords(sentence)
  --if text == "" then
  --  return
  --end
  for t, word in tokenizer.tokenize(sentence) do
    local lword = word:lower()
    if self.wordFreqs[lword] == nil then
      self.wordFreqs[lword] = 0
    end
    self.wordFreqs[lword] = self.wordFreqs[lword] + 1
  end
end

function DataSet:makeWordId(word)
  if self.maxVocabSize > 0 and self.wordsCount >= self.maxVocabSize then
    -- We've reached the maximum size for the vocab. Replace w/ unknown token
    return self.unknownToken
  end

  word = word:lower()

  local id = self.word2id[word]

  if not id then
    self.wordsCount = self.wordsCount + 1
    id = self.wordsCount
    self.id2word[id] = word
    self.word2id[word] = id
  end

  return id
end

function DataSet:addWordToVocab(word)
  word = word:lower()
  self.wordsCount = self.wordsCount + 1
  self.word2id[word] = self.wordsCount
  self.id2word[self.wordsCount] = word
  return self.wordsCount
end

-- penlight from luarocks is outdated.. below fixed version for sortv
--- return an iterator to a table sorted by its values
-- @within Iterating
-- @tab t the table
-- @func f an optional comparison function (f(x,y) is true if x < y)
-- @usage for k,v in tablex.sortv(t) do print(k,v) end
-- @return an iterator to traverse elements sorted by the values
function f_sortv(t,f)
    f = function_arg(2, f or '<')
    local keys = {}
    for k in pairs(t) do keys[#keys + 1] = k end
    table.sort(keys,function(x, y) return f(t[x], t[y]) end)
    local i = 0
    return function()
        i = i + 1
        return keys[i], t[keys[i]]
    end
end

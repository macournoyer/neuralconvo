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

function DataSet:__init(loader, options)
  options = options or {}

  self.examplesFilename = "data/examples.t7"

  -- Discard words with lower frequency then this
  self.minWordFreq = options.minWordFreq or 1

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

function DataSet:load(loader)
  local filename = "data/vocab.t7"

  if path.exists(filename) then
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
    self:visit(loader:load())
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
  -- Table for keeping track of word frequency
  self.wordFreq = {}
  self.examples = {}

  -- Add magic tokens
  self.goToken = self:makeWordId("<go>") -- Start of sequence
  self.eosToken = self:makeWordId("<eos>") -- End of sequence
  self.unknownToken = self:makeWordId("<unknown>") -- Word dropped from vocabulary

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

  print("-- Removing low frequency words")

  for i, datum in ipairs(self.examples) do
    self:removeLowFreqWords(datum[1])
    self:removeLowFreqWords(datum[2])
    xlua.progress(i, #self.examples)
  end

  self.wordFreq = nil

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

    local examples = {}

    for i = 1, size do
      local example = file:readObject()
      if example == nil then
        done = true
        file:close()
        return examples
      end
      table.insert(examples, example)
    end

    return examples
  end
end

function DataSet:removeLowFreqWords(input)
  for i = 1, input:size(1) do
    local id = input[i]
    local word = self.id2word[id]

    if word == nil then
      -- Already removed
      input[i] = self.unknownToken

    elseif self.wordFreq[word] < self.minWordFreq then
      input[i] = self.unknownToken
      
      self.word2id[word] = nil
      self.id2word[id] = nil
      self.wordsCount = self.wordsCount - 1
    end
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
    table.insert(words, self:makeWordId(word))
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

function DataSet:makeWordId(word)
  word = word:lower()

  local id = self.word2id[word]

  if id then
    self.wordFreq[word] = self.wordFreq[word] + 1
  else
    self.wordsCount = self.wordsCount + 1
    id = self.wordsCount
    self.id2word[id] = word
    self.word2id[word] = id
    self.wordFreq[word] = 1
  end

  return id
end

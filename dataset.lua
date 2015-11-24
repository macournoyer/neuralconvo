--[[
Format movie dialog data as a table of line 1:

  { {word_ids of character1}, {word_ids of character2} }

Then flips it around and get the dialog from the other character's perspective:

  { {word_ids of character2}, {word_ids of character1} }

Also build the vocabulary.
]]-- 

local DataSet = torch.class("e.DataSet")
local xlua = require "xlua"
local tokenizer = require "tokenizer"
local list = require "pl.list"

function DataSet:__init(filename, loader, loadFirst)
  -- Discard words with lower frequency then this
  self.minWordFreq = 1

  -- Make length of one text
  self.maxTextLen = 10

  -- Load only first fews examples
  self.loadFirst = loadFirst

  self.examples = {}
  self.word2id = {}
  self.id2word = {}
  self.wordsCount = 0

  self:load(filename, loader)
end

function DataSet:load(filename, loader)
  if path.exists(filename) then
    print("-- Loading from " .. filename .. " ...")
    local data = torch.load(filename)
    self.examples = data.examples
    self.word2id = data.word2id
    self.id2word = data.id2word
    self.wordsCount = data.wordsCount
    self.goToken = data.goToken
    self.eosToken = data.eosToken
    self.unknownToken = data.unknownToken
  else
    print("-- " .. filename .. " not found")
    self:visit(loader:load())
    print("-- Writing " .. filename .. " ...")
    torch.save(filename, {
      examples = self.examples,
      word2id = self.word2id,
      id2word = self.id2word,
      wordsCount = self.wordsCount,
      goToken = self.goToken,
      eosToken = self.eosToken,
      unknownToken = self.unknownToken
    })
  end
  print("-- Done")
end

function DataSet:visit(conversations)
  -- Table for keeping track of word frequency
  self.wordFreq = {}

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

  self.wordFreq = {}
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
      local targetIds = self:visitText(target.text)

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

function DataSet:visitText(text)
  local words = {}

  if text == "" then
    return
  end

  local i = 0

  for t, word in tokenizer.tokenize(text) do
    table.insert(words, self:makeWordId(word))
    i = i + 1
    if i > self.maxTextLen then
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

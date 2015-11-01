--[[
Format movie dialog data as a table of line 1:

  { {word_ids of character1}, {word_ids of character2} }

Then flips it around and get the dialog from the other character's perspective:

  { {word_ids of character2}, {word_ids of character1} }

Also build the vocabulary.
]]-- 

local PreProcessor = torch.class("e.PreProcessor")
local xlua = require "xlua"
local tokenizer = require "tokenizer"

function PreProcessor:__init(minWordFreq)
  -- Discard words with lower frequency then this
  self.minWordFreq = minWordFreq or 10

  self.examples = {}
  
  self.word2id = {}
  self.id2word = {}
  self.wordFreq = {}
  
  self.vocab = self.word2id
  self.vocabSize = 0
end

function PreProcessor:visit(conversations)
  -- Add magic tokens
  self:getWordId("</s>") -- End of sequence
  self:getWordId("<unknown>") -- Word dropped from vocabulary

  print("-- Pre-processing data")

  for i, conversation in ipairs(conversations) do
    self:visitConversation(conversation)
    xlua.progress(i, #conversations * 2)
  end

  -- Revisit from the perspective of 2nd character
  for i, conversation in ipairs(conversations) do
    self:visitConversation(conversation, 2)
    xlua.progress(#conversations + i, #conversations * 2)
  end

  print("-- Removing low frequency words")

  for i, datum in ipairs(self.examples) do
    self:removeLowFreqWords(datum[1])
    self:removeLowFreqWords(datum[2])
    xlua.progress(i, #self.examples)
  end

  return {
    examples = self.examples,
    vocab = self.vocab,
    vocabSize = self.vocabSize
  }
end

function PreProcessor:removeLowFreqWords(input)
  local unknown = self:getWordId("<unknown>")

  for i, id in ipairs(input) do
    local word = self.id2word[id]

    if word == nil then
      -- Already removed
      input[i] = unknown

    elseif self.wordFreq[word] < self.minWordFreq then
      input[i] = unknown
      
      self.word2id[word] = nil
      self.id2word[id] = nil
      self.vocabSize = self.vocabSize - 1
    end
  end
end

function PreProcessor:visitConversation(lines, start)
  start = start or 1

  for i = start, #lines, 2 do
    local input = lines[i]
    local target = lines[i+1]

    if target then
      local inputIds = self:visitText(input.text)
      local targetIds = self:visitText(target.text)

      if inputIds and targetIds then
        table.insert(self.examples, { inputIds, targetIds })
      end
    end
  end
end

function PreProcessor:visitText(text)
  local words = {}

  if text == "" then
    return
  end

  for t, word in tokenizer.tokenize(text) do
    table.insert(words, self:getWordId(word))
  end

  return words
end

function PreProcessor:getWordId(word)
  word = word:lower()

  local id = self.word2id[word]

  if id then
    self.wordFreq[word] = self.wordFreq[word] + 1
  else
    self.vocabSize = self.vocabSize + 1
    id = self.vocabSize
    self.id2word[id] = word
    self.word2id[word] = id
    self.wordFreq[word] = 1
  end

  return id
end

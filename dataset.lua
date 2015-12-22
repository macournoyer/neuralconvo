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

  -- Maximum number of words in an example sentence
  self.maxExampleLen = options.maxExampleLen or 25

  -- Max ratio of unknown words / known words in a sentence.
  -- Or else sentence is discarded.
  self.maxUnknownWordsRatio = 0

  -- Load only first fews examples (approximately)
  self.loadFirst = options.loadFirst

  self.vocab = neuralconvo.Word2Vec("data/GoogleNews-vectors-negative300.bin", 100000)

  -- Add magic tokens
  self.goToken = torch.FloatTensor(self.vocab.vecSize):fill(1) -- Start of sequence
  self.eosToken = self.vocab:get("</s>") -- End of sequence

  if not path.exists(self.examplesFilename) then
    self:visit(loader:load())
  end
end

function DataSet:visit(conversations)
  self.examples = {}

  print("Pre-processing data")

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

  print("Writing " .. self.examplesFilename)
  local file = torch.DiskFile(self.examplesFilename, "w")

  self.examplesCount = #self.examples
  file:writeInt(self.examplesCount)

  for i, example in ipairs(self.examples) do
    file:writeObject(example)
    xlua.progress(i, #self.examples)
  end

  file:close()
  self.examples = nil

  collectgarbage()
end

function DataSet:size()
  if self.examplesCount == nil then
    local file = torch.DiskFile(self.examplesFilename, "r")
    self.examplesCount = file:readInt()
    file:close()
  end

  return self.examplesCount
end

function DataSet:batches(size)
  local file = torch.DiskFile(self.examplesFilename, "r")
  file:readInt() -- examplesCount
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

local function table2tensor(tbl)
  assert(#tbl > 0)
  local t = torch.Tensor(#tbl, tbl[1]:size(1))

  for i, v in ipairs(tbl) do
    t[i] = v
  end

  return t
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

        table.insert(self.examples, { table2tensor(inputIds), table2tensor(targetIds) })
      end
    end
  end
end

function DataSet:visitText(text, additionalTokens)
  additionalTokens = additionalTokens or 0

  local words = {}
  local unknownWords = 0

  if text == "" then
    return
  end

  for t, word in tokenizer.tokenize(text) do
    -- Ignore punctuations
    if t ~= "punct" and t ~= "endpunct" then
      local vec = self.vocab:get(word) or self.vocab:get(word:lower())
      if vec then
        table.insert(words, vec)
      else
        unknownWords = unknownWords + 1
      end
    end

    -- Only keep the first sentence
    if t == "endpunct" or #words >= self.maxExampleLen - additionalTokens then
      break
    end
  end

  if #words == 0 then
    return
  end

  -- Ignore sentence if too much unknown (not in vocabulary) words.
  -- TODO log ignored words and their frequency
  if unknownWords / #words > self.maxUnknownWordsRatio then
    return
  end

  return words
end

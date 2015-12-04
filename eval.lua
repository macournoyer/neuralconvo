require 'e'
local tokenizer = require "tokenizer"
local list = require "pl.list"

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = e.DataSet()

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

-- Word IDs tensor to sentence
function t2s(t, reverse)
  local words = {}

  for i = 1, t:size(1) do
    table.insert(words, dataset.id2word[t[i]])
  end

  if reverse then
    words = list.reverse(words)
  end

  return table.concat(words, " ")
end

-- for i,example in ipairs(dataset.examples) do
--   print("-- " .. t2s(example[1], true))
--   print(">> " .. t2s(example[2]))
-- end

function say(text)
  local wordIds = {}

  for t, word in tokenizer.tokenize(text) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(wordIds, id)
  end

  local input = torch.Tensor(list.reverse(wordIds))
  print("-- " .. t2s(input, true))

  local output = model:eval(input)

  print(">> " .. t2s(torch.Tensor(output)))
end

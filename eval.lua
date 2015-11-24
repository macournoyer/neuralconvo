require 'e'
local tokenizer = require "tokenizer"

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--datafile', "data/cornell_movie_dialogs.t7", 'data file to load')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = e.DataSet(options.datafile)
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

function say(text)
  local inputs = {}

  for t, word in tokenizer.tokenize(text) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(inputs, id)
  end

  local outputs = model:eval(torch.Tensor(inputs))
  local words = {}

  for i,id in ipairs(outputs) do
    table.insert(words, dataset.id2word[id])
  end

  return table.concat(words, " ")
end

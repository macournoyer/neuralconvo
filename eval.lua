require 'e'
local tokenizer = require "tokenizer"

if dataset == nil then
  -- dataset = e.DataSet("data/cornell_movie_dialogs.t7",
  --                     e.CornellMovieDialogs("data/cornell_movie_dialogs"))
  dataset = e.DataSet("data/cornell_movie_dialogs_tiny.t7",
                      e.CornellMovieDialogs("data/cornell_movie_dialogs"), 1000)
end

EOS = torch.IntTensor{dataset.word2id["</s>"]}

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

function output2wordId(t)
  local max = t:max()
  for i = 1, t:size(1) do
    if t[i] == max then
      return i
    end
  end
end

function say(text)
  local inputs = {}
  for t, word in tokenizer.tokenize(text) do
    local t = dataset.word2id[word:lower()]
    table.insert(inputs, torch.IntTensor{t})
  end

  model:forget()

  for i = #inputs, 1, -1 do
    local input = inputs[i]
    model:forward(input)
  end

  local input = EOS
  repeat
    local output = model:forward(input)
    local outputWordId = output2wordId(output)
    io.write(dataset.id2word[outputWordId] .. " ")
    input = torch.IntTensor{outputWordId}
  until input[1] == EOS[1]

  print("")
end

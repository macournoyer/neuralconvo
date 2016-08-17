require 'neuralconvo'
local tokenizer = require "tokenizer"
local list = require "pl.List"
local options = {}

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--debug', false, 'show debug info')
cmd:text()
options = cmd:parse(arg)

-- Data
dataset = neuralconvo.DataSet()

print("-- Loading model")
model = torch.load("data/model.t7")

-- Word IDs to sentence
function pred2sent(wordIds, i)
  local words = {}
  i = i or 1

  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId[i]]
    table.insert(words, word)
  end

  return tokenizer.join(words)
end

function printProbabilityTable(wordIds, probabilities, num)
  print(string.rep("-", num * 22))

  for p, wordId in ipairs(wordIds) do
    local line = "| "
    for i = 1, num do
      local word = dataset.id2word[wordId[i]]
      line = line .. string.format("%-10s(%4d%%)", word, torch.exp(probabilities[p][i]) * 100) .. "  |  "
    end
    print(line)
  end

  print(string.rep("-", num * 22))
end

function say(text)
  local wordIds = {}

  for t, word in tokenizer.tokenize(text) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(wordIds, id)
  end

  local input = torch.Tensor(list.reverse(wordIds))
  local wordIds, probabilities = model:eval(input)

  print("neuralconvo> " .. pred2sent(wordIds))

  if options.debug then
    printProbabilityTable(wordIds, probabilities, 4)
  end
end

print("\nType a sentence and hit enter to submit.")
print("CTRL+C then enter to quit.\n")
while true do
  io.write("you> ")
  io.flush()
  io.write(say(io.read()))
end

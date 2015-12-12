local Word2Vec = torch.class("neuralconvo.Word2Vec")
require 'xlua'

local function unpackFloat(x)
  local sign = 1
  local mantissa = string.byte(x, 3) % 128
  for i = 2, 1, -1 do mantissa = mantissa * 256 + string.byte(x, i) end
  if string.byte(x, 4) > 127 then sign = -1 end
  local exponent = (string.byte(x, 4) % 128) * 2 +
                   math.floor(string.byte(x, 3) / 128)
  if exponent == 0 then return 0 end
  mantissa = (math.ldexp(mantissa, -23) + 1) * sign
  return math.ldexp(mantissa, exponent - 127)
end

function Word2Vec:__init(dataFile, first)
  local data = {}
  local f = assert(io.open(dataFile, 'rb'))

  local words = assert(f:read("*number"))
  local vecSize = assert(f:read("*number"))
  f:read("*line")

  print("Loading word2vec file " .. dataFile .. " (" .. words .. " words)")

  if first then
    words = first
  end

  for i = 1, words do
    local word = "", c
    while true do
      c = f:read(1)
      if c == ' ' then
        break
      end
      word = word .. c
    end

    local vectors = {}, vector
    local len = 0

    for v = 1, vecSize do
      vector = unpackFloat(assert(f:read(4)))
      len = len + vector ^ 2
      table.insert(vectors, vector)
    end
    len = math.sqrt(len)
    for v = 1, vecSize do
      vectors[v] = vectors[v] / len
    end

    data[word] = torch.Tensor(vectors)

    xlua.progress(i, words)
  end

  f:close()

  self._data = data
end

function Word2Vec:toVec(word)
  return self._data[word]
end

-- Based on https://github.com/rotmanmi/word2vec.torch
function Word2Vec:distance(vec, k)
  local k = k or 1  
  --self.zeros = self.zeros or torch.zeros(self.M:size(1));
  local norm = vec:norm(2)
  vec:div(norm)
  local distances = torch.mv(self.M ,vec)
  distances , oldindex = torch.sort(distances,1,true)
  local returnwords = {}
  local returndistances = {}
  for i = 1,k do
    table.insert(returnwords, w2vutils.v2wvocab[oldindex[i]])
    table.insert(returndistances, distances[i])
  end
  return {returndistances, returnwords}
end
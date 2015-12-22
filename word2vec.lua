-- Based on https://github.com/rotmanmi/word2vec.torch
local Word2Vec = torch.class("neuralconvo.Word2Vec")
require 'xlua'

local MAX_WORD_LEN = 50

local function readString(file)  
  local str = {}
  for i = 1, MAX_WORD_LEN do
    local char = file:readChar()
    
    if char == 32 or char == 10 or char == 0 then
      break
    else
      str[#str+1] = char
    end
  end
  str = torch.CharStorage(str)
  return str:string()
end

function Word2Vec:__init(binFile, first)
  print("Loading word2vec file " .. binFile)

  local file = torch.DiskFile(binFile, 'r')
  --Reading Header
  file:ascii()
  self.words = file:readInt()
  self.vecSize = file:readInt()

  print(self.words .. " words in file")
  if first then
    print("(Limiting to first " .. first .. " words)")
    self.words = first
  end

  self.w2vvocab = {}
  self.v2wvocab = {}
  self.M = torch.FloatTensor(self.words, self.vecSize)

  file:binary()
  for i = 1, self.words do
    local str = readString(file)
    local vecrep = file:readFloat(300)
    vecrep = torch.FloatTensor(vecrep)
    local norm = torch.norm(vecrep,2)
    if norm ~= 0 then vecrep:div(norm) end
    self.w2vvocab[str] = i
    self.v2wvocab[i] = str
    self.M[{{i},{}}] = vecrep
    xlua.progress(i, self.words)
  end

  file:close()
end

function Word2Vec:get(word)
   return self:lookup(word) or self:lookup(word:lower())
end

function Word2Vec:lookup(word)
   local ind = self.w2vvocab[word]
   return self.M[ind]
end

function Word2Vec:distance(vec, k)
  local k = k or 1  
  local norm = vec:norm(2)
  vec:div(norm)
  local distances = torch.mv(self.M ,vec)
  distances , oldindex = torch.sort(distances,1,true)
  local returnwords = {}
  local returndistances = {}
  for i = 1,k do
    table.insert(returnwords, self.v2wvocab[oldindex[i]])
    table.insert(returndistances, distances[i])
  end
  return returnwords, returndistances
end

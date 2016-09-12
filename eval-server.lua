require 'neuralconvo'
local tokenizer = require "tokenizer"
local list = require "pl.List"
local options = {}

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:option('--opencl', false, 'use OpenCL. Training must be done on OpenCL')
  cmd:option('--debug', false, 'show debug info')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = neuralconvo.DataSet()

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  elseif options.opencl then
    require 'cltorch'
    require 'clnn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/model.t7")
end

-- Word IDs to sentence
function pred2sent(wordIds, i)
  local words = {}
  i = i or 1

  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId[i]]
    --print(wordId[i]..word)
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
      line = line .. string.format("%-10s(%4d%%)", word, probabilities[p][i] * 100) .. "  |  "
    end
    print(line)
  end

  print(string.rep("-", num * 22))
end

function say(text)
  local wordIds = {}


  
  --print(text)
  local values = {}
  for w in text:gmatch("[\33-\127\192-\255]+[\128-\191]*") do
     table.insert(values, w)
  end

  for i, word in ipairs(values) do
    local id = dataset.word2id[word] or dataset.unknownToken
    --print(i.." "..word.." "..id)

    table.insert(wordIds, id)

  end

--[[
  for t, word in tokenizer.tokenize(text) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(wordIds, id)
  end
]]--

  local input = torch.Tensor(list.reverse(wordIds))
  local wordIds, probabilities = model:eval(input)

  local ret = pred2sent(wordIds)
  print(">> " .. ret)

  if options.debug then
    printProbabilityTable(wordIds, probabilities, 4)
  end

  return ret

end


--[[ http server using ASyNC]]--

 function unescape (s)
      s = string.gsub(s, "+", " ")
      s = string.gsub(s, "%%(%x%x)", function (h)
            return string.char(tonumber(h, 16))
          end)
      return s
    end


local async = require 'async'
require('pl.text').format_operator()

async.http.listen('http://0.0.0.0:8082/', function(req,res)
   print('request:',req)
   local resp

   if req.url.path == '/' and  req.url.query ~= nil and  #req.url.query > 0 then

    local text_in = unescape(req.url.query)
    print(text_in)
    local ret = say(text_in)
    resp = [[${data}]] % {data = ret}

   else
    resp = 'Oops~  This is a wrong place, please goto <a href="/?你好啊"> here!</a>' 

   end

  --  if req.url.path == '/test' then
  --     resp  = [[
  --     <p>You requested route /test</p>
  --     ]]
  --  else
  --     -- Produce a random story:
  --     resp = [[
  --     <h1>From my server</h1>
  --     <p>It's working!<p>
  --     <p>Randomly generated number: ${number}</p>
  --     <p>A variable in the global scope: ${ret}</p>
  --     ]] % {
  --        number = math.random(),
  --        ret = ret
  --     }
  --  end

   res(resp, {['Content-Type']='text/html; charset=UTF-8'})
end)

print('server listening to port 8082')

async.go()
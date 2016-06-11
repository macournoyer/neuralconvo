local Large = torch.class("neuralconvo.Large")
local stringx = require "pl.stringx"
local xlua = require "xlua"

local TOTAL_LINES = 100000000 
local FILE_NAME = "100000000.txt"

local function parsedLines(file)
  local f = assert(io.open(file, 'r'))

  return function()

    local line = f:read("*line")

    if line == nil then
      f:close()
      return
    end

    local t = {}
    t["text"] = line

    return t
  end
end

function Large:__init(dir)
  self.dir = dir
end

local function progress(c)
  if c % 100000 == 0 then
    xlua.progress(c, TOTAL_LINES)
  end
end

function Large:load()
  local lines = {}
  local conversations = {}
  local conversation = {}
  local count = 0
  --local lineID = 1
    
  for line in parsedLines(self.dir .. "/"..FILE_NAME) do
  
    table.insert(conversation, line)

    if count % 100 == 0 then
      table.insert(conversations, conversation)
      conversation = {}
    end 

    count = count + 1
    progress(count)
  end
  
  xlua.progress(TOTAL_LINES, TOTAL_LINES)

  print("-- Finished Parsing Open Subtitle data set ...")
  return conversations
end

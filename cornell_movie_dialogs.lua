local CornellMovieDialogs = torch.class("neuralconvo.CornellMovieDialogs")
local stringx = require "pl.stringx"
local xlua = require "xlua"

function CornellMovieDialogs:__init(dir)
  self.dir = dir
end


function CornellMovieDialogs:load()
  local lines = {}
  local conversations = {}
  local count = 1

  print("-- Parsing Cornell movie dialogs data set ...")


  local f = assert(io.open('../xiaohuangji50w_fenciA.conv', 'r'))

  while true do
      local line = f:read("*line")
      if line == nil then
        f:close()
        break
      end
      
      lines[count] = line
      count = count + 1
  end

  print("Total lines = "..count)
  local tmpconv = nil

  local TOTAL = #lines
  local count = 0

  for i, line in ipairs(lines) do
    --print(i..'  '..line)
    if  string.sub(line, 0, 1) == "E" then 

      if tmpconv ~= nil then
        --print('new conv'..#tmpconv)
        table.insert(conversations, tmpconv)
      end 
      --print('e make the tmpconv')
      tmpconv = {}
      
    end 

    if string.sub(line, 0, 1) == "M" then
      --print('insert into conv')
      local tmpl = string.sub(line, 3, #line)
      --print(tmpl)
      table.insert(tmpconv, tmpl)
    end

    count = count + 1
    if count%1000 == 0 then
      xlua.progress(count, TOTAL)
    end
  end

  return conversations
end

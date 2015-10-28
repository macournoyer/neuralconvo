local CornellMovieDialogs = torch.class("e.CornellMovieDialogs")
local stringx = require "pl.stringx"

local function parsedLines(file, fields)
  local f = assert(io.open(file, 'r'))

  return function()
    local line = f:read("*line")

    if line == nil then
      f:close()
      return
    end

    local values = stringx.split(line, " +++$+++ ")
    local t = {}

    for i,field in ipairs(fields) do
      t[field] = values[i]
    end

    return t
  end
end

function CornellMovieDialogs:__init(dir)
  local lines = {}
  self.convertations = {}
  self.lines_count = 0
  
  for line in parsedLines(dir .. "/movie_lines.txt", {"lineID","characterID","movieID","character","text"}) do
    lines[line.lineID] = line
    line.lineID = nil
    -- Remove unused fields
    line.characterID = nil
    line.movieID = nil
    self.lines_count = self.lines_count + 1
  end

  for conv in parsedLines(dir .. "/movie_conversations.txt", {"character1ID","character2ID","movieID","utteranceIDs"}) do
    local conversation = {}
    local lineIDs = stringx.split(conv.utteranceIDs:sub(3, -3), "', '")
    for i,lineID in ipairs(lineIDs) do
      table.insert(conversation, lines[lineID])
    end
    table.insert(self.convertations, conversation)
  end
end

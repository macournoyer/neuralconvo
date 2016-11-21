local CornellMovieDialogs = torch.class("CornellMovieDialogs")
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
  self.dir = dir
end

local MOVIE_LINES_FIELDS = {"lineID","characterID","movieID","character","text"}
local MOVIE_CONVERSATIONS_FIELDS = {"character1ID","character2ID","movieID","utteranceIDs"}
local TOTAL_LINES = 387810

function CornellMovieDialogs:load(callback, progress)
  local lines = {}
  local count = 0

  for line in parsedLines(self.dir .. "/movie_lines.txt", MOVIE_LINES_FIELDS) do
    lines[line.lineID] = line
    line.lineID = nil
    -- Remove unused fields
    line.characterID = nil
    line.movieID = nil
    count = count + 1
    progress(count, TOTAL_LINES)
  end

  for conv in parsedLines(self.dir .. "/movie_conversations.txt", MOVIE_CONVERSATIONS_FIELDS) do
    local conversation = {}
    local lineIDs = stringx.split(conv.utteranceIDs:sub(3, -3), "', '")
    for i,lineID in ipairs(lineIDs) do
      table.insert(conversation, lines[lineID])
    end
    callback(conversation)
    count = count + 1
    progress(count, TOTAL_LINES)
  end

  progress(TOTAL_LINES, TOTAL_LINES)

  return conversations
end

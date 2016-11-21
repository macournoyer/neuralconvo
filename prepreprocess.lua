require 'torch'
require 'cornell_movie_dialogs'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--srcfile', 'data/src.txt', 'Path to source training data')
cmd:option('--targetfile', 'data/target.txt', 'Path to target training data')
cmd:text()
local options = cmd:parse(arg)

local srcFile = assert(io.open(options.srcfile, "w"))
local targetFile = assert(io.open(options.targetfile, "w"))

local loader = CornellMovieDialogs("data/cornell_movie_dialogs")
local conversations = {}

print("Loading Cornell Movie Dialogs data set ...")
loader:load(
  -- Loaded callback
  function(conversation)
    table.insert(conversations, conversation)
  end,
  -- Progress callback
  function(i, total)
    if i % 10000 == 0 or i == total then
      xlua.progress(i, total)
    end
  end
)

print("Writing files ...")

local total = #conversations * 2

local function visitConversation(lines, start)
  start = start or 1

  for i = start, #lines, 2 do
    local input = lines[i]
    local target = lines[i+1]

    if target then
      local src = input.text
      local target = target.text

      if src and target then
        srcFile:write(src .. "\n")
        targetFile:write(target .. "\n")
      end
    end
  end
end

for i, conversation in ipairs(conversations) do
  visitConversation(conversation)
  xlua.progress(i, total)
end
-- Revisit from the perspective of 2nd character
for i, conversation in ipairs(conversations) do
  visitConversation(conversation, 2)
  xlua.progress(#conversations + i, total)
end

srcFile:close()
targetFile:close()

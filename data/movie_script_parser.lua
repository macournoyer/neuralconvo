local htmlparser = require 'htmlparser'

local MovieScriptParser = torch.class("bot.MovieScriptParser")

function MovieScriptParser:__init(options)
  assert(options)
  self.actor_indent = assert(options.actor_indent)
  self.dialog_indent = assert(options.dialog_indent)
end

function MovieScriptParser:parse(file)
  local f = assert(io.open(file, 'r'))
  self.input = f:read("*all")
  f:close()

  self.dialogs = {}
  self.pos = 0
  self.match = nil

  while self:acceptDialog() or
        self:acceptLine() do end

  return self.dialogs
end

function MovieScriptParser:accept(regexp)
  local match = string.match(self.input, "^" .. regexp, self.pos)
  if match then
    self.pos = self.pos + #match
    self.match = match
    return true
  end
end

function MovieScriptParser:acceptLine()
  return self:accept(".-\n") or self:accept(".+$")
end

function MovieScriptParser:acceptDialog()
  local name

  self:accept("</b>")

  if self:accept("<b>" .. string.rep(" ", self.actor_indent)) and
     self:accept("[^\n]+") then
    name = self.match
    self:accept("\n")
  else
    return
  end

  if not self:accept("</b>") then
    return
  end

  -- Dialog is intend by 25 spaces
  local lines = {}
  while self:accept(string.rep(" ", self.dialog_indent)) do
    self:accept(" *") -- Extra leading spaces

    -- Remove (...), usually who talking to or indication
    if self:accept("%(.-%)") then
      table.insert(lines, " ")
    end

    -- Sometimes there's a leading -
    self:accept("–")
    
    -- The actual line of dialog
    if self:accept("[^\n]+") then
      table.insert(lines, self.match)
    end
    self:accept("\n")
  end

  if #lines > 0 then
    table.insert(self.dialogs, {
      actor=name,
      text=table.concat(lines)
    })
    return true
  end
end

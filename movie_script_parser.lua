local MovieScriptParser = torch.class("e.MovieScriptParser")

function MovieScriptParser:parse(file)
  local f = assert(io.open(file, 'r'))
  self.input = f:read("*all")
  f:close()

  self.script = {}
  self.pos = 0
  self.match = nil

  while self:acceptDialog() or
        self:acceptLine() do end

  return self.script
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
  self:accept("<b>")

  -- Get the actor name (all caps)
  if self:accept(" +") and self:accept("[A-Z%- %.]+") then
    name = self.match
    self:accept("\n")
  else
    return
  end

  if not self:accept("</b>") then
    return
  end

  -- Get the indentation of the dialog that follows
  if not self:accept(" +") then
    return
  end
  local dialog_indent = #self.match

  -- Get the dialog lines
  local lines = {}
  repeat
    self:accept("%.+")

    -- Remove (...), usually who talking to or indication
    if self:accept("%(.-%)") and #lines > 0 then
      table.insert(lines, " ")
    end

    -- Sometimes there's a leading -
    self:accept("–")

    self:accept(" +")
    
    -- The actual line of dialog
    if self:accept("[^\n]+") then
      table.insert(lines, self.match)
    end
    self:accept("\n")
  until not self:accept(string.rep(" ", dialog_indent))

  if #lines > 0 then
    table.insert(self.script, {
      type  = 'dialog',
      actor = name,
      text  = table.concat(lines)
    })
    return true
  end
end

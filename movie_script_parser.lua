local Parser = torch.class("neuralconvo.MovieScriptParser")

function Parser:parse(file)
  local f = assert(io.open(file, 'r'))
  self.input = f:read("*all")
  f:close()

  self.pos = 0
  self.match = nil

  -- Find start of script
  repeat self:acceptLine() until self:accept("<pre>")

  local dialogs = {}

  -- Apply rules until end of script
  while not self:accept("</pre>") and self:acceptLine() do
    local dialog = self:parseDialog()
    if dialog then
      table.insert(dialogs, dialog)
    end
  end

  return dialogs
end

-- Returns true if regexp matches and advance position
function Parser:accept(regexp)
  local match = string.match(self.input, "^" .. regexp, self.pos)
  if match then
    self.pos = self.pos + #match
    self.match = match
    return true
  end
end

-- Accept anything up to the end of line
function Parser:acceptLine()
  return self:accept(".-\n")
end

function Parser:acceptSep()
  while self:accept("</?b>") or self:accept(" +") do end
  return self:accept("\n")
end

function Parser:parseDialog()
  local dialogs = {}

  repeat
    local dialog = self:parseSpeech()
    if dialog then
      table.insert(dialogs, dialog)
    end
  until not self:acceptSep()

  if #dialogs > 0 then
    return dialogs
  end
end

-- Matches:
--
--        NAME
--    some nice text
--    more text.
--
-- or
--
--    NAME; text
function Parser:parseSpeech()
  local name

  self:accept("</b>")
  self:accept("<b>")

  -- Get the character name (all caps)
  -- TODO remove parenthesis from name
  if self:accept(" +") and self:accept("[A-Z][A-Z%- %.%(%)]+") then
    name = self.match
  else
    return
  end

  -- Handle inline dialog: `NAME; text`
  if self:accept(";") and self:accept("[^\n]+") then
    return {
      character = name,
      text  = self.match
    }
  end

  self:accept("\n")

  if not self:accept("</b>") then
    return
  end

  -- Get the dialog lines
  -- TODO remove parenthesis from text
  local lines = {}
  while self:accept(" +") do
    -- The actual line of dialog
    if self:accept("[^\n]+") then
      table.insert(lines, self.match)
    end
    self:accept("\n")
  end

  if #lines > 0 then
    return {
      character = name,
      text  = table.concat(lines)
    }
  end
end

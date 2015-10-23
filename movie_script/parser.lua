local Parser = torch.class("e.MovieScript.Parser")

function Parser:parse(file)
  local f = assert(io.open(file, 'r'))
  self.input = f:read("*all")
  f:close()

  self.script = {}
  self.pos = 0
  self.match = nil

  -- Find start of script
  repeat self:acceptLine() until self:accept("<pre>")

  -- Apply rules until end of script
  while not self:accept("</pre>") and
        (
          -- Rules
          self:acceptDialog() or
          self:acceptScene() or
          self:acceptLine()
        ) do end

  return self.script
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

-- Matches:
--
--        NAME
--    Dialog text
--    more text.
--
-- or
--
--    NAME; dialog text
function Parser:acceptDialog()
  local name

  self:accept("</b>")
  self:accept("<b>")

  -- Get the actor name (all caps)
  if self:accept(" +") and self:accept("[A-Z%- %.]+") then
    name = self.match
  else
    return
  end

  -- Handle inline dialog: `NAME; text`
  if self:accept(";") and self:accept("[^\n]+") then
    table.insert(self.script, {
      type  = 'dialog',
      actor = name,
      text  = self.match
    })
    return true
  end

  self:accept("\n")

  if not self:accept("</b>") then
    return
  end

  -- Get the dialog lines
  local lines = {}
  while self:accept(" +") do
    self:accept("%.+")

    -- Remove (...), usually who talking to or indication
    if self:accept("%(.-%)") and #lines > 0 then
      table.insert(lines, " ")
    end

    -- Sometimes there's a leading -
    self:accept("–")

    -- Ignore leading spaces
    self:accept(" +")
    
    -- The actual line of dialog
    if self:accept("[^\n]+") then
      table.insert(lines, self.match)
    end
    self:accept("\n")
  end

  if #lines > 0 then
    table.insert(self.script, {
      type  = 'dialog',
      actor = name,
      text  = table.concat(lines)
    })
    return true
  end
end

-- Try to parse the end of a scene. Any block of text that is not dialog ends the scene.
function Parser:acceptScene()
  if not self:accept("[^\n]*%w+") then
    return
  end

  self:accept("\n")

  local last = self.script[#self.script]

  if last and last.type ~= 'scene' then
    table.insert(self.script, {
      type = 'scene',
      count = lines
    })
  end

  return true
end

local lexer = require "pl.lexer"
local yield = coroutine.yield
local M = {}

local function word(token)
  return yield("word", token)
end

local function quote(token)
  return yield("quote", token)
end

local function space(token)
  return yield("space", token)
end

local function tag(token)
  return yield("tag", token)
end

local function punct(token)
  return yield("punct", token)
end

local function endpunct(token)
  return yield("endpunct", token)
end

local function unknown(token)
  return yield("unknown", token)
end

function M.tokenize(text)
  return lexer.scan(text, {
      { "^%s+", space },
      { "^['\"]", quote },
      { "^%w+", word },
      { "^%-+", space },
      { "^[,:;%-]", punct },
      { "^%.+", endpunct },
      { "^[%.%?!]", endpunct },
      { "^</?.->", tag },
      { "^.", unknown },
    }, { [space]=true, [tag]=true })
end

function M.join(words)
  local s = table.concat(words, " ")
  s = s:gsub("^%l", string.upper)
  s = s:gsub(" (') ", "%1")
  s = s:gsub(" ([,:;%-%.%?!])", "%1")
  return s
end

return M
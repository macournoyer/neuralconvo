local lexer = require "pl.lexer"
local yield = coroutine.yield

local function word(token)
  return yield("word", token)
end

local function space(token)
  return yield("space", token)
end

local function punct(token)
  return yield("punct", token)
end

local function unknown(token)
  return yield("unknown", token)
end

function e.tokenize(text)
  return lexer.scan(text, {
      { "^%s+", space },
      { "^[%w%-']+", word },
      { "^[,:;%.%?!]+", punct },
      { "^.+", unknown },
    }, { [space]=true })
end

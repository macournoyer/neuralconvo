local PreProcessor = torch.class("e.PreProcessor")
local utf8 = require "lua-utf8"
local stringx = require "pl.stringx"
local string = require 'lua-utf8'

function PreProcessor:process(dialogs)
  for i = #dialogs, 1, -1 do
    if not self:processDialog(dialogs[i]) then
      table.remove(dialogs, i)
    end
  end

  return dialogs
end

function PreProcessor:processDialog(dialog)
  for i = #dialog, 1, -1 do
    if not self:processSpeech(dialog[i]) then
      table.remove(dialogs, i)
    end
  end

  -- Discard monologues
  if #dialog <= 1 then
    return
  end

  return dialog
end

local function removeParentheses(text)
  local t = string.gsub(text, " *%b() *", " ")
  return stringx.strip(t)
end

local EOS_TOKEN = "</s>"

function PreProcessor:processSpeech(speech)
  speech.actor = removeParentheses(speech.actor)
  speech.text = removeParentheses(speech.text)

  speech.text = string.gsub(speech.text, '"', "")

  -- Discard weird separators
  -- FIXME not really working ...
  -- speech.text = utf8.gsub(speech.text, "–", "")

  -- Remove punctuations
  -- speech.text = string.gsub(speech.text, "%.%.%.", " ")
  -- speech.text = string.gsub(speech.text, "[,:;%.%?!]+", " ")

  -- Mark end of sentences
  -- speech.text = string.gsub(speech.text, "[%.%?!]+", " " .. EOS_TOKEN .. " ") .. " " .. EOS_TOKEN

  -- Squeeze spaces
  -- speech.text = string.gsub(speech.text, "  +", " ")

  -- Remove redundant EOS tokens
  -- speech.text = string.gsub(speech.text, EOS_TOKEN .. " " .. EOS_TOKEN, EOS_TOKEN)

  return speech
end
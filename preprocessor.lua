local PreProcessor = torch.class("e.PreProcessor")

function PreProcessor:process(dialogs)
  for i = #dialogs, 1, -1 do
    local dialog = dialogs[i]

    dialog = self:processDialog(dialog)

    if dialog == nil or #dialog == 0 then
      table.remove(dialogs, i)
    end
  end

  return dialogs
end

function PreProcessor:processDialog(dialog)
  -- Discard monologues
  if #dialog == 1 then
    return
  end

  for i = #dialog, 1, -1 do
    local speech = dialog[i]

    speech = self:processSpeech(speech)

    if speech == nil then
      table.remove(dialogs, i)
    end
  end

  return dialog
end

function PreProcessor:processSpeech(speech)
  -- TODO
end
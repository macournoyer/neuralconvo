local DataSet = torch.class("e.DataSet")
local path = require "pl.path"

function DataSet:__init(filename, loader)
  self.filename = filename
  self.loader = loader

  if path.exists(filename) then
    local data = torch.load(filename)
    self.examples = data.examples
    self.vocab = data.vocab
    self.vocabSize = data.vocabSize
  else
    print("-- " .. filename .. " not found")
    self:load()
    print("-- Writing " .. filename .. " ...")
    torch.save(filename, {
      examples = self.examples,
      vocab = self.vocab,
      vocabSize = self.vocabSize,
    })
    print("-- Done")
  end
end

function DataSet:load()
  local data = e.PreProcessor():visit(self.loader:load())

  self.examples = data.examples
  self.vocab = data.vocab
  self.vocabSize = data.vocabSize
end

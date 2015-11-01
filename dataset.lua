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
    self.loader.load()
    torch.save(filename, {
      examples = self.examples,
      vocab = self.vocab,
      vocabSize = self.vocabSize,
    })
  end
end

function DataSet:load()
  local rawData = self.loader.load()
  local processor = e.PreProcessor()
  local data = processor:visit(rawDataset)

  self.examples = data.examples
  self.vocab = data.vocab
  self.vocabSize = data.vocabSize
end

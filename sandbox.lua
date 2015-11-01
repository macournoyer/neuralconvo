require 'e'

local loader = e.CornellMovieDialogs("data/cornell_movie_dialogs")
local rawData = loader:load()
-- local dataset = e.DataSet("data/cornell_movie_dialogs.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"))

local processor = e.PreProcessor()
processor:visit(rawData)
local dataset = processor

print(#dataset.examples .. " examples")
print(dataset.vocabSize .. " words")

print(dataset.examples[1])
print(dataset.vocab["</s>"])
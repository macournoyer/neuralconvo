require 'e'

-- local loader = e.CornellMovieDialogs("data/cornell_movie_dialogs")
-- local rawDataset = loader:load()
local dataset = e.DataSet("data/cornell_movie_dialogs.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"))

-- local processor = e.PreProcessor()
-- processor:visit(rawDataset)
-- local dataset = processor.data

print(#dataset.examples .. " examples")
print(dataset.vocabSize .. " words")

require 'e'

-- local loader = e.CornellMovieDialogs("data/cornell_movie_dialogs")
-- local rawData = loader:load()
-- dataset = e.DataSet("data/cornell_movie_dialogs.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"))
-- dataset = e.DataSet("data/cornell_movie_dialogs_small.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"), 10000)
dataset = e.DataSet("data/cornell_movie_dialogs_tiny.t7", e.CornellMovieDialogs("data/cornell_movie_dialogs"), 1000)

-- local processor = e.PreProcessor()
-- processor:visit(rawData)
-- local dataset = processor

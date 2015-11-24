require 'e'

local size = 1000
dataset = e.DataSet("data/cornell_movie_dialogs_" .. size .. ".t7",
                    e.CornellMovieDialogs("data/cornell_movie_dialogs"), size)

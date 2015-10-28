require 'e'

local dataset = e.CornellMovieDialogs("data/cornell_movie_dialogs")

print(#dataset.convertations)
print(dataset.lines_count)
print(dataset.loaded_lines_count)
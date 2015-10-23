require 'e'

-- local word2vec = e.Word2Vec("/Users/ma/Downloads/GoogleNews-vectors-negative300.bin")
-- print(word2vec:get("for"))

-- print(e.MovieScriptParser():parse("data/pulp_fiction.html"))
-- print(e.MovieScriptParser():parse("data/Seinfeld-Good-News,-Bad-News.html"))

local script = e.MovieScript.Parser():parse("data/Futurama-Space-Pilot-3000.html")
print(script)
-- local dialog = e.MovieScript.Processor():toDialog(script)

-- TODO ...
-- local movie = e.MovieScript("data/Futurama-Space-Pilot-3000.bin")

-- for i, dialog in ipairs(movie.dialog()) do
--   local q, a = dialog
-- end
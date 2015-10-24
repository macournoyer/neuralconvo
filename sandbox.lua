require 'e'

-- local word2vec = e.Word2Vec("/Users/ma/Downloads/GoogleNews-vectors-negative300.bin")
-- print(word2vec:get("for"))

print(e.MovieScriptParser():parse("data/pulp_fiction.html"))
-- print(e.MovieScriptParser():parse("data/Seinfeld-Good-News,-Bad-News.html"))
-- print(e.MovieScriptParser():parse("data/Futurama-Space-Pilot-3000.html"))

-- local dialogs = e.MovieScriptParser():parse("data/Futurama-Space-Pilot-3000.html")
-- print(dialogs)
-- dialogs = e.PreProcessor():process(dialogs)

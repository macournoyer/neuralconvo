require 'bot'

-- local word2vec = bot.Word2Vec("/Users/ma/Downloads/GoogleNews-vectors-negative300.bin")
-- print(word2vec:get("for"))

-- print(bot.MovieScriptParser({actor_indent=37, dialog_indent=25}):parse("data/pulp_fiction.html"))
print(bot.MovieScriptParser({actor_indent=29, dialog_indent=15}):parse("data/Seinfeld-Good-News,-Bad-News.html"))
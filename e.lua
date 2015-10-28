e = {}

torch.include('e', 'cornell_movie_dialogs.lua')
torch.include('e', 'movie_script_parser.lua')
torch.include('e', 'preprocessor.lua')
torch.include('e', 'tokenizer.lua')
torch.include('e', 'word2vec.lua')

return e
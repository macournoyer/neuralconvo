bot = {}

torch.include('bot', 'word2vec.lua')
torch.include('bot', 'data/movie_script_parser.lua')

return bot
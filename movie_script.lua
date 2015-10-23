local MovieScript = torch.class("e.MovieScript")

torch.include('e', 'movie_script/parser.lua')
torch.include('e', 'movie_script/processor.lua')

function MovieScript:__init()
end
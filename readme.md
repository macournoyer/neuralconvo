# Neural Conversational Model in Torch

Forked from https://github.com/chenb67/neuralconvo

@dgkae dgkae.cbf#gmail.com

##How
Use https://github.com/dgkae/dgk_lost_conv as training corpus. The chinese sentenses should be splited by semantic words, using '/'. We modify cornell_movie_dialog.lua to support it. Lua save all string(e.g. chinese) all in multibyte, so in chinese the formal pl.lexer is not working. We use outsider word-splitting tool and using '/' as the tag.

##Result

![result](a.png)
![result2](b.png)


## License

MIT 


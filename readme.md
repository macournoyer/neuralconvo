# Neural Conversational Model in Torch

Forked from https://github.com/chenb67/neuralconvo

如果训练中遇到问题请先翻原始fork的issue，这里只是改成中文而已！很多人找到这里，总问我怎么训练呀等等，我也没时间回答，所以我给大家建了一个QQ群：



![](qqun.png)

进群请先自我介绍下，例如您的学校或单位，从事什么研究等。谢啦！

##How
Use https://github.com/dgkae/dgk_lost_conv as training corpus. The chinese sentenses should be splited by semantic words, using '/'. We modify cornell_movie_dialog.lua to support it. Lua save all string(e.g. chinese) all in multibyte, so in chinese the formal pl.lexer is not working. We use outsider word-splitting tool and using '/' as the tag.

##Result

![result](a.png)
![result2](b.png)



## License

MIT 


# Neural Conversational Model in Torch

Forked from https://github.com/chenb67/neuralconvo

如果训练中遇到问题请先翻原始fork的issue，这里只是改成中文而已！

本repo不会更新，交流请加QQ群：

1号群（满）

![](data/qqun.png)

2号群：

![](data/qq2.jpeg)

微信拉群：

![](data/fate2.jpeg)



[网易云课堂视频课程1-聊天机器人](https://study.163.com/course/introduction/1005049028.htm?utm_source=400000000173015&utm_medium=share&utm_campaign=commission&hideAppEntrance=1)   [网易云课堂视频课程2-知识图谱](https://study.163.com/course/introduction/1004964005.htm?utm_source=400000000173015&utm_medium=share&utm_campaign=commission&hideAppEntrance=1)

##How
Use https://github.com/dgkae/dgk_lost_conv as training corpus. The chinese sentenses should be splited by semantic words, using '/'. We modify cornell_movie_dialog.lua to support it. Lua save all string(e.g. chinese) all in multibyte, so in chinese the formal pl.lexer is not working. We use outsider word-splitting tool and using '/' as the tag.

##Result

![result](a.png)
![result2](b.png)



## License

MIT 


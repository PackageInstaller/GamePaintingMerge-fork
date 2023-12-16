# 立绘合成(明日方舟&amp;少女前线为例)

# 原理

这两个游戏的立绘都是将PNG的RGBA（红绿蓝和透明通道）分开存放

本程序的作用就是将RGB文件与A文件组合，合成完整的立绘文件。

明日方舟的A文件是人物白色，背景黑色，直接将A文件按灰度图读取，然后将其作为RGBA文件的A通道。

少女前线的A文件是人物白色，背景透明，需要将A文件按四通道原始文件读取，然后提取出A通道作为RGBA文件的A通道。

最后RGB+A=RGBA输出即可。

# 说明
## 原仓库[地址](https://github.com/Vistyxio/GamePaintingMerge)
## 拿大佬的过来改成了自己用的，只需要指定输入的文件夹以及对应Alpha文件的名称，就能批量合成

# 依赖

opencv

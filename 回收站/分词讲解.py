import jieba
import re

s="""湖南农民运动考察报告[插图]（一九二七年三月）农民问题的严重性我这回到湖南[插图]，实地考察了湘潭、湘乡、衡山、醴陵、长沙五县的情况。从一月四日起至二月五日止，共三十二天，在乡下，在县城，召集有经验的农民和农运工作同志开调查会，仔细听他们的报告，所得材料不少。"""
s = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', s)
print("过滤后：",s)

words = jieba.cut(s)
words = list(words)
result=" ".join(words)
print(result)
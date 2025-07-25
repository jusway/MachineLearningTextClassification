import json
import re
import jieba


def load_jsonl_data(file_path: str) :
    """
    加载JSON Lines格式的数据
    Args:
        file_path: JSON Lines文件路径
    Returns:
        texts: 文本内容列表
        labels: 标签列表
    """
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_dict = json.loads(line)
            texts.append(data_dict['内容'])
            labels.append(data_dict['类别'])

    return texts, labels


def cut_text(text):
    """
    正则过滤只留汉语英文数字，分词后空格隔开
    """
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    words = jieba.cut(text)
    words_list=[]
    for word in words:
        if len( word.strip())>=2:
            words_list.append(word.strip())
    res=" ".join(words_list)
    return res



if __name__ == '__main__':
    s = """湖南农民运动考察报告[插图]（一九二七年三月）农民问题的严重性我这回到湖南[插图]，实地考察了湘潭、湘乡、衡山、醴陵、长沙五县的情况。从一月四日起至二月五日止，共三十二天，在乡下，在县城，召集有经验的农民和农运工作同志开调查会，仔细听他们的报告，所得材料不少。"""
    res=cut_text(s)
    print(res)
















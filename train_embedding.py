# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:43:22 2020

@author: Michaeltose

用于训练字嵌入模型或者字频数据
"""
#%%
from gensim.models import word2vec
import pandas as pd
import numpy as np


def train_chinese_wiki(loc:str, model_loc:str, clst_loc:str):
    words = []
    char_list = []
    with open(loc, encoding = 'utf-8') as file:
        for line in file:
            if len(line) >=10:
                words.append(list(line))
                char_list.extend(list(line))
    model = word2vec.Word2Vec(words, min_count = 3, window = 5, size = 300)
    model.wv.save_word2vec_format(model_loc, binary = True)
    np.save(clst_loc, char_list)
    return model.wv
     
def get_words(list_of_passage: str):
    words = []
    for sent in list_of_passage:
        words.append(list(sent))
    return words

def train_embedding_model(data: str, model_loc: str):
    '''
    :param data: 文本列表
    '''
    words = get_words(data)
    model = word2vec.Word2Vec(words, min_count = 1, window = 5, size = 300)
    model.wv.save(model_loc)
    return

def train_char_list(data_loc: str, char_list_loc: str) -> list:
    char_list = []
    data = pd.read_excel(data_loc)['AB']
    for text in data:
        char_list.extend(list(text))
    np.save(char_list_loc, char_list, allow_pickle=True)
    return char_list
    
#%%
if __name__ == '__main__':
    #训练预训练字向量并保存
    clst_loc = './data/char_list_wiki.npy'
    model_loc= './data/model/w2v_wiki.model'
    loc = './data/chinese_wiki.txt'
    model = train_chinese_wiki(loc, model_loc, clst_loc)
#%%


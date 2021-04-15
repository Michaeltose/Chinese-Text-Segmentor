import gc
import os
import re
import hmm
from pathlib import Path
import _pickle as pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense,
                                     Embedding, Input, TimeDistributed)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from representation import TextRepresent


class Segmentor():
    def __init__(self, mode = 'bilstm_emb', max_len = 150):
        '''
        mode取值：
            bilstm_emb：用预训练字向量矩阵训练的模型分词
            bilstm：用普通的bilstm分词
            hmm_emb：字嵌入的hmm
            hmm：普通hmm
        max_len：最大的句子长度，默认150
        '''
        self.mode = mode
        self.model_path = {'bilstm_emb':r'./data/model/bilstm_wiki_emb.h5',
                            'bilstm':r'./data/model/bilstm_wiki.h5',
                            'hmm_emb':r'./data/model/hmm_emb.model',
                            'hmm':r'./data/model/hmm.model'}
        self.max_len = 150
        self.trans = self._load_obj('./data/model/trans.pkl')
        self.label_index = {'B':0, 'M':1, 'E':2, 'S':3, 'X':4}
        self.index_label = {0:'B', 1:'M', 2:'E', 3:'S', 4:'X'}

        #加载神经网络
        
        if mode == 'bilstm_emb':
            self.tr = TextRepresent(max_len=max_len)
            self.model = self.load_model(self.model_path[mode])
        elif mode == 'bilstm':
            self.tr = TextRepresent(embedding=False, max_len=max_len)
            self.model = self.load_model(self.model_path[mode])
        elif mode == 'hmm' or mode == 'hmm_emb':
            self.model = hmm.HMM(states=['B','M','E','S','X'],
                    time_num=max_len)
            self.model.load(loc = self.model_path[mode])

    def _load_obj(self, loc):
        with open(loc, 'rb') as file:
            return pickle.load(file)

    def load_model(self, path):
        self.model = load_model(path)
        return self.model
    
    def get_nodes(self, result):
        '''将预测的概率值转换成图中的节点'''
        r = np.log(result) #取对数后要记得概率之间的乘法要变加法
        return [dict(zip(['B', 'M', 'E', 'S'], i[:4])) for i in r]
    
    def viterbi(self, nodes):
        paths = {'B':nodes[0]['B'], 'S':nodes[0]['S']}#初始路径，只有两条单节点路径
        for layer in range(1, len(nodes)):
            pre_paths = paths.copy() #保存上一层的路径
            paths = {}
            for node in nodes[layer].keys():#对本层的节点node
                sub_paths = {}#字典，保存候选 pre_path -> node 的概率
                for pre_path in pre_paths.keys():#对上一层的所有候选路径pre_path
                    if pre_path[-1] + node in self.trans.keys():
                        sub_paths[pre_path + node] = pre_paths[pre_path] +\
                            self.trans[pre_path[-1] + node] + nodes[layer][node] #取过对数，用加法
                #对当前层的节点node，从sub_path中找出到node的概率最大路径
                path_2_node = pd.Series(sub_paths).sort_values()
                paths[path_2_node.index[-1]] = path_2_node[-1]
        return pd.Series(paths).sort_values().index[-1]

    def get_label(self, result):
        label = []
        for w in result:
            label.append(self.index_label[list(w).index(max(w))])
        return label
    
    def get_words(self, text, label):
        segments = []
        word = ''
        for i, w in enumerate(text):
            if label[i] == 'B':
                word = w
            elif label[i] == 'E':
                word += w
                segments.append(word)
            elif label[i] == 'S' or label[i] == 'X':
                word = w
                segments.append(word)
            elif label[i] == 'M':
                word += w
        return segments

    def cut_sent(self, sent, viterbi = True):
        '''
        viterbi = True：使用viterbi算法优化结果
        如果是hmm则必须使用viterbi
        返回分词结果与预测的字标签
        '''
        if self.mode == 'bilstm_emb' or self.mode == 'bilstm':
            indexs = np.array(self.tr.sent_2_vec(sent)).reshape(-1, self.max_len)
            result = self.model.predict(indexs)[0]
            if viterbi:
                nodes = self.get_nodes(result)
                label = self.viterbi(nodes)
            else:
                label = self.get_label(result)
        else:
            result = self.model.predict(sent)
            nodes = self.get_nodes(result)
            label = self.viterbi(nodes)
        return (self.get_words(sent, label), label)

    def cut_para(self, para):
        '''
        将段落切分成句子
        '''
        para = re.sub('([。！；;？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        #para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        #para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        return para.split("\n")

    def segment(self, text, viterbi = True):
        '''
        text为输入的文本
        先将text拆分为句子，再对句子分词，最后对句子的分词结果进行拼接
        '''
        segments = []
        labels = []
        if len(text) > self.max_len:
            for sent in self.cut_para(text):
                seg, label = self.cut_sent(sent[0:self.max_len], viterbi) #只对句子前max_len进行分词，多余的舍去
                labels += label
                segments += seg
        else:
            segments, labels = self.cut_sent(text, viterbi)
        return (segments, labels)

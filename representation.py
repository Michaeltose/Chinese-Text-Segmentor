#%%
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

class TextRepresent:
    def __init__(self, max_len = 150,
                loc1 = r'./data/model/w2v_wiki.model',
                embedding = True):
        '''

        '''
        self.loc1 = loc1
        self.max_len = max_len
        self.model = self.get_wv_model()
        self.embedding = embedding

    def get_wv_model(self):
        '''
        加载字嵌入模型，loc为模型位置
        '''
        try:
            self.model = KeyedVectors.load_word2vec_format(self.loc1, binary=True)
        except UnicodeDecodeError:
            self.model = KeyedVectors.load(self.loc1)
        return self.model

    def get_sub_word(self, text, target, window = 4):
        '''
        使用目标字上下文的字嵌入向量替换目标字
        '''
        words = []
        if (target+1) - window < 0:
            left_side = 0
        else:
            left_side = (target+1) - window
        for move in range(0, window + 1):
            left_side += move
            right_side = left_side + window
            vec = 0
            num = 0
            for word in text[left_side:right_side]:
                try:
                    vec += self.model[word]
                    num += 1 #记录窗口中合法词的个数
                except KeyError:
                    pass
            if num ==0:#没有合法词
                words.append(
                    self.model.similar_by_vector(
                        self.model[self.model.index2word[0]]
                    )[0]
                )
            else:
                vec = vec / num
                words.append(self.model.similar_by_vector(vec)[0])
        return sorted(words, key = lambda d:d[1], reverse = True)[0][0]

    def sent_2_vec(self, sent):
        '''
        将句子sent转换成向量，方法是将句中的字替换为model中对应词的索引
        '''
        result = list(sent)
        lenth = len(result)
        if self.embedding:
            if lenth < self.max_len:
                for i, w in enumerate(result):
                    try:
                        result[i] = self.model.vocab[w].index
                    except KeyError:
                        w = self.get_sub_word(sent, i)
                        result[i] = self.model.vocab[w].index
                return result + [0] * (self.max_len - lenth)
            else:
                result = result[0:self.max_len]
                for i, w in enumerate(result):
                    try:
                        result[i] = self.model.vocab[w].index
                    except KeyError:
                        w = self.get_sub_word(sent, i)
                        result[i] = self.model.vocab[w].index
                return result
        else:
            if lenth < self.max_len:
                for i, w in enumerate(result):
                    try:
                        result[i] = self.model.vocab[w].index
                    except KeyError:
                        result[i] = 0
                return result + [0] * (self.max_len - lenth)
            else:
                result = result[0:self.max_len]
                for i, w in enumerate(result):
                    try:
                        result[i] = self.model.vocab[w].index
                    except KeyError:
                        result[i] = 0
                return result


class TrainTransform:
    def __init__(self, 
            loc1 = r'./data/model/w2v_wiki.model',
            max_len = 150,
            label_dic = {'B':0, 'M':1, 'E':2, 'S':3, 'X':4},
            embedding = True):
        self.label_dic = label_dic
        self.tr = TextRepresent(loc1 = loc1, max_len = max_len, embedding = embedding)
        
    def to_onehot(self, label):
        '''
        将标签数据转换成独热编码
        '''
        new_labels = []
        for l in label:
            r = np.array([0] * len(self.label_dic))
            r[l] = 1
            new_labels.append(r)
        return np.array(new_labels).reshape(-1,len(self.label_dic))

    def label_transform(self, labels, one_hot = True):
        '''
        labels为所有数据的BEMS状态，将其转换为数字形式
        '''
        results = []
        for i, states in tqdm(enumerate(labels)):
            if len(states) < self.tr.max_len:
                r = [self.label_dic[state] for state in states] + [4] * (self.tr.max_len - len(states))
                if one_hot:
                    r = self.to_onehot(r)
                    results.append(r)
                else:
                    results.append(np.array(r).reshape(-1, len(self.label_dic)))
            else:
                r = [self.label_dic[state] for state in states[0:self.tr.max_len]]
                if one_hot:
                    r = self.to_onehot(r)
                    results.append(r)
                else:
                    results.append(np.array(r).reshape(-1, len(self.label_dic)))
        return results
        
    def text_transform(self, data):
        '''
        将句子转换成训练数据的格式：
            二维列表，形状为：(数据量, max_len)
        data为分句列表
        '''
        result = []
        for i, sent in tqdm(enumerate(data)):
            result.append(self.tr.sent_2_vec(sent))
        return np.array(result).reshape(-1, self.tr.max_len)

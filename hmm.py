import re
import representation
import numpy as np
import pandas as pd
import _pickle as pickle
from tqdm import tqdm
from collections import defaultdict
from representation import TrainTransform

class HMM:
    def __init__(self, states: list, time_num: int, embedding = False):
        '''
        :param states: 状态列表
        :param num_time: 时间数
        '''
        
        self.time_num = time_num
        self._states = list(states)
        self._state2index = dict(zip(self._states, range(0, len(self._states))))
        self.tt = TrainTransform(max_len = time_num, label_dic = self._state2index)
        self.pi = np.zeros(len(self._states))
        self.trans_matrix = np.zeros((len(self._states), len(self._states)))
        self._sorted_lunch_matrix = []
        self.embedding = embedding
        
        
    def forward_algo(self, obs_sequence):
        '''
        已知t时刻的状态，计算1到t时间段输出序列概率
        return: obs_sequence的概率
        '''
        self.alpha = np.zeros((self.time_num, len(self._states)))
        #初始化前向概率列表shape = (self.time_num, len(self._states)
        #列表t行i列的元素为：
        #       1到t观测序列与t时刻状态的联合概率p(o1,..,,ot,st=qi)
        alpha_first = np.zeros(len(self._states))
        for j, qj in enumerate(self._states):
            #初始化alpha_first
            alpha_first[j] = self.pi[j] * self._get_lunch_prob(j, 0, obs_sequence)
        self.alpha[0][:] = alpha_first
        for t, oi in enumerate(obs_sequence[1:]):
            #从第二个观测开始遍历观测序列
            alpha_t = np.zeros(len(self._states))
            for j, qj in enumerate(self._states):
            #遍历所有t+1时刻的可能状态j
                j_prob = 0
                for i, prob in enumerate(self.alpha[t]):
                #计算t+1时刻的状态是j的概率: j_prob
                #prob是t时刻的状态节点st处于状态qi的概率
                    j_prob += prob * self.trans_matrix[i][j]
                    #t时刻是状态i的概率 乘 状态i转移到状态j 的概率
                alpha_t[j] = j_prob * self._get_lunch_prob(j, t+1, obs_sequence)
            self.alpha[t+1][:] = alpha_t
        return sum(self.alpha[-1])
        
    def backward_algo(self, obs_sequence):
        '''
        已知t时刻的状态，计算t+1及以后输出序列的概率
        return: obs_sequence的概率
        '''
        self.beta = np.zeros((self.time_num, len(self._states)))
        #初始化后向概率列表beta
        #列表t行i列元素为：
        #   t时刻状态的st为qi情况下，t+1到T的观测序列的概率
        #   beta[t][i] = p(OT,...,ot+1|st=qi)
        #   beta[t+1][j] = p(OT,...,ot+2|st+1=qj)
        #   a[i][j] * b[j][ot+1] = p(st+1=qj|st=qi) * p(ot+1|st+1 = qj)
        #
        #   beta[t][i] = Σ_j{a[i][j] * b[j][ot+1] * beta[t+1][j]}
        beta_last = np.zeros(len(self._states))
        for j, qj in enumerate(self._states):
            #初始化beta_last
            beta_last[j] = 1
        self.beta[-1][:] = beta_last
        for t in range(self.time_num-2, -1, -1):
        #逆时间顺序遍历观测序列:T-1到1
            beta_t = np.zeros(len(self._states))
            for i, qi in enumerate(self._states):
            #遍历t时刻可能的状态qi
                obs_prob_next = 0
                for j, qj in enumerate(self._states):
                #遍历t+1时刻可能的状态qj
                    #t+1时刻输出状态概率: (aij * bjk) *
                    #t+1以后所有输出序列概率: (beta_t+1) = 
                    #t+1及以后输出序列概率
                    obs_prob_next += (self.trans_matrix[i][j] *
                        self._get_lunch_prob(j, t+1, obs_sequence) *
                        self.beta[t+1][j])
                beta_t[i] = obs_prob_next
            self.beta[t][:] = beta_t
        obs_prob = 0
        for i, qi in enumerate(self._states):
            obs_prob += (self.beta[0][i] *
                self.pi[i] *
                self._get_lunch_prob(i, 0, obs_sequence))
        return obs_prob
        
    def _get_trans_matrix_mle(self, state_sequences: list):
        '''
        用极大似然估计法计算状态转移矩阵和初始状态概率pi
        :param state_sequences:训练数据的状态序列
        '''
        n = len(state_sequences)
        self._state_freq = [0] * len(self._states)
        #用于保存每个状态出现的次数，后续计算可能用到
        all_states = '$' + '$'.join(state_sequences)
        for i, qi in tqdm(enumerate(self._states)):#计算初始概率pi
            #给所有序列加上$开头然后拼接成一个字符串
            qi_begin = len(list(re.finditer('\$' + qi, all_states))) #计算状态qi在状态序列开头出现的次数
            self.pi[i] = qi_begin / n
            from_qi = 0
            self._state_freq[i] = len(list(re.finditer(qi, all_states))) #qi出现的次数
            for j, qj in enumerate(self._states):
                to_qj = len(list(re.finditer(qi+qj, all_states)))
                from_qi += to_qj
                self.trans_matrix[i][j] = to_qj
            self.trans_matrix[i] = self.trans_matrix[i] / from_qi
        return (self.pi, self.trans_matrix)
    
    def _get_lunch_matrix_mle(self, state_sequences: list, obs_sequences: list):
        '''
        极大似然估计法计算发射矩阵
        调用此函数之前必须保证调用过_get_trans_matrix_mle
        '''
        self._em = False
        self._word_freq = defaultdict(int) #计算所有观测值出现频率
        self.lunch_matrix = [0] * len(self._states)
        #列表第i个位置为状态i的发射概率，状态在列表中的顺序与self._states一致
        for i, state in enumerate(self._states):
            self.lunch_matrix[i] = {} #字典键为观测值O，值为状态i生成此字符O的次数
        for i, state_sequence in tqdm(enumerate(state_sequences)):
            #遍历所有状态序列
            for j, q in enumerate(state_sequence):#遍历第i个训练数据的状态序列的所有状态
                o = obs_sequences[i][j]
                qi = self._state2index[q] #获取当前状态的索引
                self._word_freq[o] += 1
                if o not in self.lunch_matrix[qi]:
                    self.lunch_matrix[qi][o] = 1
                else:
                    self.lunch_matrix[qi][o] += 1
        for i, state in enumerate(self._states):
            for k in self.lunch_matrix[i].keys():
                self.lunch_matrix[i][k] = self.lunch_matrix[i][k] / self._state_freq[i]
        return self.lunch_matrix
        
    def _get_lunch_prob(self, state_i, obs_i, obs_sequence):
        '''
        获取状态state_i生成观测obs_i的概率
        :param state_i:状态的索引
        :param obs_i:观测的索引
        :param obs_sequence:完整观测序列
        '''
        oi = obs_sequence[obs_i]
        if self.embedding:#使用字嵌入向量
            if oi in self._word_freq:
            #如果观测oi在训练数据的观测集中存在
                if oi in self.lunch_matrix[state_i]:
                    #如果观测oi可由qj生成
                    return self.lunch_matrix[state_i][oi]
                else:
                    return float(0)
            else:
                #观测oi未在训练数据中出现过，则替换当前观测值
                oi = self.get_sub_word(text = obs_sequence, target = obs_i)
                if oi in self.lunch_matrix[state_i]:
                    #如果观测oi可由qj生成
                    return self.lunch_matrix[state_i][oi]
                else:
                    return float(0)
        else:
            if oi in self._word_freq:
            #如果观测oi在训练数据的观测集中存在
                if oi in self.lunch_matrix[state_i]:
                    #如果观测oi可由qj生成
                    return self.lunch_matrix[state_i][oi]
                else:
                    return float(0)
            else:
                #观测oi未在训练数据中出现过，则替换当前观测值
                oi = self.tt.tr.model.index2word[0]
                if oi in self.lunch_matrix[state_i]:
                    #如果观测oi可由qj生成
                    return self.lunch_matrix[state_i][oi]
                else:
                    return float(0)
    
    def _save_obj(self, loc, obj):
        with open(loc, 'wb') as file:
            pickle.dump(obj, file)
    
    def _load_obj(self, loc):
        with open(loc, 'rb') as file:
            return pickle.load(file)
    
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
                    vec += self.tt.tr.model[word]
                    num += 1 #记录窗口中合法词的个数
                except KeyError:
                    pass
            if num ==0:#没有合法词
                words.append(
                    self.tt.tr.model.similar_by_vector(
                        self.tt.tr.model[self.tt.tr.model.index2word[0]]
                    )[0]
                )
            else:
                vec = vec / num
                words.append(self.tt.tr.model.similar_by_vector(vec)[0])
        return sorted(words, key = lambda d:d[1], reverse = True)[0][0]
    
    def get_simplified_transmatrix(self, loc = './data/model/trans.pkl'):
        '''
        获取简化版的转移矩阵
        '''
        trans = {}
        for i,r in enumerate(self.trans_matrix):
            for j,prob in enumerate(r):
                 if prob >0:
                    trans[self._states[i] + self._states[j]] = prob
        self._save_obj(loc, trans)
        print('转移矩阵保存在：%s'%loc)
        return trans
    
    def train(self, labels, sents):
        '''
        :return (初始概率，转移矩阵，发射矩阵)
        '''
        obs_sequences, state_sequences = self.to_obs_states(sents, labels)
        print('HMM训练数据转换完毕')
        self._get_trans_matrix_mle(state_sequences)
        self._get_lunch_matrix_mle(state_sequences, obs_sequences)
        return (self.pi,
                self.trans_matrix,
                self.lunch_matrix)
    
    def save(self, loc = './data/model/hmm.model'):
        '''
        保存模型
        '''
        model = (self.pi,
                self.trans_matrix,
                self.lunch_matrix,
                self._states,
                self._word_freq,
                self.embedding,
                self.time_num)
        self.get_simplified_transmatrix()
        if self.embedding:
            self._save_obj('./data/model/hmm_emb.model', model)
            print('字嵌入HMM模型保存在%s'%'./data/model/hmm_emb.model')
        else:
            self._save_obj(loc, model)
            print('HMM模型保存在%s'%loc)
        
    def load(self, loc = './data/model/hmm.model'):
        '''
        加载模型
        '''
        (self.pi,
        self.trans_matrix,
        self.lunch_matrix,
        self._states,
        self._word_freq,
        self.embedding,
        self.time_num) = self._load_obj(loc)
        self._state2index = dict(zip(self._states, range(0, len(self._states))))
        self.tt = TrainTransform(max_len = self.time_num, label_dic = self._state2index)
    
    def predict(self, sent):
        obs_sequence = self.padding(sent)
        self.forward_algo(obs_sequence)
        self.backward_algo(obs_sequence)
        result = [0] * self.time_num
        for t, obs in enumerate(obs_sequence):
            total = 0
            state_t = [0] * len(self._states)
            for i, qi in enumerate(self._states):
                qi_prob = self.alpha[t][i] * self.beta[t][i]
                total += qi_prob
                state_t[i] = qi_prob
            if total != 0:
                result[t] = np.array(state_t) / total #概率归一化
            else:
                result[t] = np.array(state_t)
        return result

    def padding(self, sequence, if_obs = True):
        if if_obs:
            #padding观测
            if len(sequence) > self.time_num:
                return sequence[0:self.time_num]
            else:
                w = self.tt.tr.model.index2word[0]
                sequence += w * (self.time_num - len(sequence))
                return sequence
        else:
            #padding状态
            if len(sequence) > self.time_num:
                return sequence[0:self.time_num]
            else:
                sequence += self._states[-1] * (self.time_num - len(sequence))
                return sequence
    
    def to_obs_states(self, sents, labels):
        '''
        用于生成HMM专用训练数据
        '''
        obs_sequences = []
        state_sequences = []
        for i, sent in tqdm(enumerate(sents)):
            obs_sequences.append(self.padding(sent))
            state_sequences.append(self.padding(labels[i], if_obs= False))
        return (obs_sequences, state_sequences)
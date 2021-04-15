#%%
from tqdm import tqdm 
import pandas as pd
import numpy as np
import _pickle as pickle
import segmentor
from segmentor import Segmentor

def to_region(segments: str) -> tuple:
    '''
    将分词结果转换为区间
    '''
    region = []
    start = 0
    for seg in segments:
        end = start + len(seg)
        region.append((start, end))
        start = end
    return region

def evaluate(preds: str, reals: str, texts: str, dictionary: dict) -> tuple:
    '''
    :param preds: 预测结果列表
    :param reals: 真实分词列表
    :param texts: 原始文本列表
    :param dictionary 训练数据词典
    :return (召回率， 精确率， F1， iv， oov)
    '''
    n_real, n_pred, n_overlap = 0, 0, 0
    iv, oov, iv_r, oov_r = 0, 0, 0, 0
    for i, pred in tqdm(enumerate(preds)):
        r_real = set(to_region(reals[i]))
        r_pred = set(to_region(pred))
        overlap = r_real & r_pred #获取重叠词
        #print(overlap)
        n_real += len(r_real)
        n_pred += len(r_pred)
        n_overlap += len(overlap)
        for (start, end) in r_real:
            word = texts[i][start : end]
            if word in dictionary:
                iv += 1
            else:
                oov += 1
        for (start, end) in overlap:
            word = texts[i][start : end]
            if word in dictionary:
                iv_r += 1
            else:
                oov_r += 1
    r = n_overlap * 100/ n_pred
    p = n_overlap * 100/n_real
    f1 = 2 * p * r/(p + r)
    if iv != 0:
        iv_r = iv_r * 100/ iv
    else:
        iv_r = 100
    if oov != 0:
        oov_r = oov_r * 100/ oov
    else:
        oov_r = 100
    return (r, p, f1, iv_r, oov_r)


def get_preds(texts: str, segmentor: Segmentor, viterbi = True) -> list:
    '''
    :param texts: 原始文本列表
    :param segmentor: 分词器实例
    :param viterbi: 是否用viterbi算法优化
    '''
    preds = []
    for i, text in tqdm(enumerate(texts)):
        preds.append(segmentor.segment(text, viterbi)[0])
    return preds

def get_reals(segs: str, sep = '/  ') -> list:
    reals = []
    for seg in tqdm(segs):
        reals.append(seg.split(sep))
    return reals

def get_dictionary(loc = './data/model/dictionary.pkl'):
    with open(loc, 'rb') as file2:
        return pickle.load(file2)

def get_eva_data(data, dataset = 'test'):
    '''
    使用四个分词模型产生六种分词结果并保存
    :param dataset: test代表论文摘要测试集
                    weibo代表微博评论测试集
    '''
    smt1 = Segmentor(mode = 'bilstm')
    smt2 = Segmentor(mode = 'bilstm_emb')
    smt3 = Segmentor(mode = 'hmm')
    smt4 = Segmentor(mode = 'hmm_emb')
    #HMM分词必须使用Viterbi，故不再区分
    if dataset == 'test':
        
        print('\nlstm非嵌入模型，viterbi优化')
        preds_test1 = get_preds(data['sent'], smt1) #lstm非嵌入模型，viterbi优化
        with open('./data/evaluate/test_lstm_v.pkl', 'wb') as file1:
            pickle.dump(preds_test1, file1)

        print('\nlstm嵌入模型，viterbi优化')
        preds_test2 = get_preds(data['sent'], smt2) #lstm嵌入模型，viterbi优化
        with open('./data/evaluate/test_lstm_emb_v.pkl', 'wb') as file2:
            pickle.dump(preds_test2, file2)

        print('\nlstm非嵌入模型，无viterbi优化')
        preds_test3 = get_preds(data['sent'], smt1, viterbi=False) #lstm非嵌入模型，无viterbi优化
        with open('./data/evaluate/test_lstm.pkl', 'wb') as file3:
            pickle.dump(preds_test3, file3)

        print('\nlstm嵌入模型，无viterbi优化')
        preds_test4 = get_preds(data['sent'], smt2, viterbi=False) #lstm嵌入模型，无viterbi优化
        with open('./data/evaluate/test_lstm_emb.pkl', 'wb') as file4:
            pickle.dump(preds_test4, file4)

        print('\nHMM非嵌入模型，viterbi优化')
        preds_test5 = get_preds(data['sent'], smt3, viterbi=True) #hmm非嵌入，viterbi
        with open('./data/evaluate/test_hmm_v.pkl', 'wb') as file:
            pickle.dump(preds_test5, file)
        
        print('\nHMM嵌入模型，viterbi优化')
        preds_test6 = get_preds(data['sent'], smt4, viterbi=True) #hmm嵌入，viterbi
        with open('./data/evaluate/test_hmm_emb_v.pkl', 'wb') as file:
            pickle.dump(preds_test6, file)

    elif dataset == 'weibo':
        print('\nlstm非嵌入模型，viterbi优化')
        preds_test1 = get_preds(data['raw'], smt1) #lstm非嵌入模型，viterbi优化
        with open('./data/evaluate/weibo_lstm_v.pkl', 'wb') as file1:
            pickle.dump(preds_test1, file1)

        print('\nlstm嵌入模型，viterbi优化')
        preds_test2 = get_preds(data['raw'], smt2) #lstm嵌入模型，viterbi优化
        with open('./data/evaluate/weibo_lstm_emb_v.pkl', 'wb') as file2:
            pickle.dump(preds_test2, file2)

        print('\nlstm非嵌入模型，无viterbi优化')
        preds_test3 = get_preds(data['raw'], smt1, viterbi=False) #lstm非嵌入模型，无viterbi优化
        with open('./data/evaluate/weibo_lstm.pkl', 'wb') as file3:
            pickle.dump(preds_test3, file3)

        print('\nlstm嵌入模型，无viterbi优化')
        preds_test4 = get_preds(data['raw'], smt2, viterbi=False) #lstm嵌入模型，无viterbi优化
        with open('./data/evaluate/weibo_lstm_emb.pkl', 'wb') as file4:
            pickle.dump(preds_test4, file4)

        print('\nHMM非嵌入模型，viterbi优化')
        preds_test5 = get_preds(data['raw'], smt3, viterbi=True) #hmm非嵌入，viterbi
        with open('./data/evaluate/weibo_hmm_v.pkl', 'wb') as file:
            pickle.dump(preds_test5, file)
            
        print('\nHMM嵌入模型，viterbi优化')
        preds_test6 = get_preds(data['raw'], smt4, viterbi=True) #hmm嵌入，viterbi
        with open('./data/evaluate/weibo_hmm_emb_v.pkl', 'wb') as file:
            pickle.dump(preds_test6, file)
    else:
        print('Dataset Error')

def print_evaluate_result(data_loc:dict, reals, texts, dataset = 'test'):
    '''
        读取已保存的测试集分词结果并评估
        :param  data_loc:字典，键用于区分不同测试集，值为分词结果保存路径
        :param  reals:真实的分词结果列表
        :param  texts:测试集原始文本
    '''
    dictionary = get_dictionary()
    print('\n' + dataset + '数据集评估结果：')
    for k in data_loc.keys():
        with open(data_loc[k], 'rb') as file1:
            preds = pickle.load(file1)
        r, p, f1, iv, oov = evaluate(preds,
                reals,
                texts,
                dictionary)
        print('\n\n' + k + '：')
        print('\t召回率：{0}；\n'.format(r) +
                '\t精确率：{0};\n'.format(p) +
                '\tF1：{0}；\n'.format(f1) +
                '\t登录词召回率：{0}；\n'.format(iv) +
                '\t非登录词召回率：{0}；\n'.format(oov))

#%%
if __name__ == "__main__":
    #评估测试集分词结果
    test = pd.read_csv('./data/test.csv')
    weibo = pd.read_excel('./data/test_weibo.xlsx')

    reals_test = get_reals(test['seg'])
    reals_weibo = get_reals(weibo['seg'], sep=' ')

    data_loc_test = {'LSTM，嵌入，Viterbi':'./data/evaluate/test_lstm_emb_v.pkl',
        'LSTM，非嵌入，Viterbi':'./data/evaluate/test_lstm_v.pkl',
        'LSTM，嵌入':'./data/evaluate/test_lstm_emb.pkl',
        'LSTM，非嵌入':'./data/evaluate/test_lstm.pkl',
        'HMM，嵌入':'./data/evaluate/test_hmm_emb_v.pkl',
        'HMM，非嵌入':'./data/evaluate/test_hmm_v.pkl'
        }
    
    data_loc_weibo = {'LSTM，嵌入，Viterbi':'./data/evaluate/weibo_lstm_emb_v.pkl',
        'LSTM，非嵌入，Viterbi':'./data/evaluate/weibo_lstm_v.pkl',
        'LSTM，嵌入':'./data/evaluate/weibo_lstm_emb.pkl',
        'LSTM，非嵌入':'./data/evaluate/weibo_lstm.pkl',
        'HMM，嵌入':'./data/evaluate/weibo_hmm_emb_v.pkl',
        'HMM，非嵌入':'./data/evaluate/weibo_hmm_v.pkl'
        }

    print_evaluate_result(data_loc_test, reals_test, test['sent'], dataset = '论文摘要')
    print_evaluate_result(data_loc_weibo, reals_weibo, weibo['raw'], dataset = '微博')

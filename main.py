import pandas as pd
import segmentor
from segmentor import Segmentor
import evaluate
#%%
if __name__ == '__main__':
    '''
    mode取值：
            bilstm_emb：用预训练字向量模型训练的bilstm分词模型
            bilstm：普通的bilstm分词模型
            hmm_emb：字嵌入的hmm
            hmm：普通hmm
    '''
    smt1 = Segmentor(mode = 'bilstm_emb')
    smt2 = Segmentor(mode = 'hmm')
    smt3 = Segmentor(mode = 'bilstm')
    smt4 = Segmentor(mode = 'hmm_emb')
    print(smt1.segment('武汉市长江大桥')[0])
    print(smt2.segment('武汉市长江大桥')[0])
    print(smt3.segment('武汉市长江大桥')[0])
    print(smt4.segment('武汉市长江大桥')[0])

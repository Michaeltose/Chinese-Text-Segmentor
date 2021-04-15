# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:47:22 2020

@author: Michaeltose
"""

import representation
import hmm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from representation import TextRepresent, TrainTransform
from hmm import HMM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')


def get_train_data(texts, labels, trans:TrainTransform):
    '''
    texts为训练文本
    labels为texts中每一个句子对应的bmes状态
    trans为TrainTransform类实例
    '''
    return (trans.text_transform(texts), trans.label_transform(labels))

def get_embedding_weights(trans: TrainTransform):
    '''
    加载训练好的字嵌入向量
    trans为representation.py中定义的TrainTransform类实例
    '''
    model  = trans.tr.model
    embedding_dim = len(model[model.index2word[0]])
    num_words = len(model.index2word)
    embedding_weights = np.zeros((num_words, embedding_dim))
    for i in range(0, num_words):
        embedding_weights[i, :] = model[model.index2word[i]]
    return embedding_weights

def plot_history(history, loc):
    '''
    用于保存训练结果图片,history为keras.model.fit的返回值
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label = 'Training acc')
    plt.plot(x, val_acc, 'r', label = 'Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label = 'Training loss')
    plt.plot(x, val_loss, 'r', label = 'Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(loc, dpi=300)

def train_lstm(nn_model, X_train, y_train, path_checkpoint, epochs = 32, batch_size = 256):
    '''
    nn_model为创建好的keras神经网络模型
    path_chekpoint指定模型checkpoint存储位置
    X_train为训练数据
    y_trian为训练标签数据
    '''
    #权重储存点
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',\
                                      verbose=1, save_weights_only=True,\
                                      save_best_only=True)
    # 定义early stoping如果3个epoch内validation loss没有改善则停止训练
    earlystopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='loss',
                                       factor=0.1, min_lr=1e-8, patience=0,
                                       verbose=1)
    # 定义callback函数
    callbacks = [earlystopping, checkpoint, lr_reduction]
    time_start=time.time()
    history = nn_model.fit(X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks = callbacks)
    time_end=time.time()
    m, s = divmod(time_end-time_start, 60)
    h, m = divmod(m, 60) 
    print('Time cost：%02d:%02d:%02d' % (h, m, s))
    return (history, nn_model)
    
def get_nn_model_embedding(max_len, embedding_weights):
    '''
    返回一个keras审计网络模型
    max_len为一条训练数据的长
    embedding_weights为字嵌入矩阵
    '''
    num_words, embedding_dim = embedding_weights.shape
    sequence = Input(shape=(max_len,), dtype='int32')
    embedded = Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_weights],
                    input_length=max_len,
                    trainable=False)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    lstm = LSTM(units=16, return_sequences=True)(blstm)
    output = TimeDistributed(Dense(5, activation='softmax'))(lstm)
    model = Model(sequence, output)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

def get_nn_model(max_len, word_size, num_words):
    '''
    返回一个keras审计网络模型
    num_words为词频词典的长度
    word_size为嵌入层返回的字向量长
    '''
    sequence = Input(shape=(max_len,), dtype='int32')
    embedded = Embedding(num_words + 1, word_size, input_length=max_len, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    lstm = LSTM(units=16, return_sequences=True)(blstm)
    output = TimeDistributed(Dense(5, activation='softmax'))(lstm)
    model = Model(sequence, output)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

def train(sents, labels, embedding = True, mode = 'lstm', max_len = 150):
    if mode == 'lstm':
        trans = TrainTransform(max_len=max_len, embedding= embedding)
        X, Y = get_train_data(sents, labels, trans)
        Y = np.array(Y).reshape(-1, len(X[0]), len(Y[0][0]))
        print('lstm训练数据转换完毕')
        if embedding:
            path_checkpoint = './data/check_points/segment_checkpoint_bilstm_wiki_emb.keras'
            ew = get_embedding_weights(trans)
            model = get_nn_model_embedding(max_len = max_len,
                    embedding_weights=ew)
            loc = './data/model/bilstm_wiki_emb.h5'
            pic = 'his_bi_wiki_emb.png'
            info = '字嵌入LSTM模型保存在%s'
        else:
            path_checkpoint = './data/check_points/segment_checkpoint_bilstm_wiki.keras'
            model = get_nn_model(max_len = len(X[0]),
                    word_size = 300,
                    num_words = len(trans.tr.model.index2word))
            loc = './data/model/bilstm_wiki.h5'
            pic = 'his_bi_wiki.png'
            info = 'LSTM模型保存在%s'
        history, trained_mdl = train_lstm(model, X, Y, path_checkpoint)
        plot_history(history, pic)
        trained_mdl.save(loc)
        print('训练完毕')
        print(info % loc)
    elif mode == 'hmm':
        states = ['B', 'M', 'E', 'S', 'X']
        hmm = HMM(states, max_len, embedding = embedding)
        hmm.train(labels, sents)
        print('训练完毕')
        hmm.save()
    else:
        print('mode参数只可以是lstm或hmm')

#%%

if __name__ == '__main__':
    #训练四种分词模型
    data = pd.read_csv('./data/train.csv')
    train(data['sent'], data['tag'], embedding=True, mode='lstm')
    train(data['sent'], data['tag'], embedding=True, mode='hmm')
    train(data['sent'], data['tag'], embedding=False, mode='lstm')
    train(data['sent'], data['tag'], embedding=False, mode='hmm')
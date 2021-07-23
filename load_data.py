import logging
import torch
import pandas as pd
import numpy as np
import pickle
import gensim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)



class ModelEmbedding():
    #加载预训练词向量
    def __init__(self,w2v_path):
        self.w2v=gensim.models.word2vec.Word2Vec.load(w2v_path).wv
        self.input_word_list=self.w2v.index2word
        self.label2id=pickle.load(open(''))
        self.id2label=pickle.load(open(''))
        self.n_class=len(self.label2id)
        self.n_voc=len(self.input_word_list)
        self.word2id=pickle.load(open(''))
        self.id2word=pickle.load(open(''))



class TextDataset(Dataset):
    def __init__(self,args,train,label):
        #已经编码成id
        self.train=train
        self.label=label
        self.args=args
        #self.word2id=self.args.word2id
        #self.id2word=self.args.id2word
        #self.label2id=self.args.label2id
        #self.id2label=self.args.id2word



    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        x=self.train[item]
        x=[int(i) for i in x]
        y=self.label[item]
        y=[int(i) for i in y]
        return torch.tensor(x),torch.tensor(y)



def collate_fn(batch):
    '''
    :param batch:  (batch, ([sentence_len, word_embedding], [sentence_len]))
    :return:
    '''
    x_list=[x[0] for x in batch]
    y_list=[y[1] for y in batch]
    lengths=[len(item[0]) for item in batch]
    x_list=pad_sequence(x_list,padding_value=0,batch_first=True)
    y_list=pad_sequence(y_list,padding_value=-1,batch_first=True)
    return x_list,y_list,lengths







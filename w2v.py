from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import logging
import config as config
logger = logging.getLogger(__name__)

def w2v(data,f,L=100):
    print("word2v")
    print(len(train))
    w2v=Word2Vec(data,size=L,window=3,min_count=1,sg=1,iter=10)
    print("save w2v to {}".format(os.path.join('data',f+'{}d'.format(L))))
    return w2v
    #pickle.dump(w2v,open(os.path.join('data',f+'{}d'.format(L)),'wb'))



if __name__=='__main__':
    train=pickle.load(open('data/train_x_id.pkl','rb'))
    dev=pickle.load(open('data/dev_x_id.pkl','rb'))
    test=pickle.load(open('data/test_x_id.pkl','rb'))
    train.extend(dev)
    train.extend(test)
    w2v=w2v(train,'emb')
    vectors=w2v.wv.vectors
    word_list=w2v.wv.index2word  #['3','4','1','2']
    vectors_=[]
    #按照索引进行排序
    for i in range(1,len(word_list)+1):
        vectors_.append(list(vectors[word_list.index(str(i))]))
    embedding=[[0.]*100]
    embedding.extend(vectors_)
    #Embedding直接加载，0号位为pad 0的向量
    pickle.dump(embedding,open('data/w2v_vectors.pkl','wb'))


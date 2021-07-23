import os
import gc
import torch
import logging
import argparse
import pickle
import random
import gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,RandomSampler
from sklearn.model_selection import StratifiedKFold
from model.lstm import Lstm
from torch.optim import Adam
from load_data import TextDataset,collate_fn
data_path='data'
import config as config
from tqdm import tqdm


if __name__=='__main__':
    logger=logging.getLogger(__name__)
    logger.info("2021-7-15")
    parser=argparse.ArgumentParser()
    parser.add_argument('--n_class',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=3)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_len',type=int,default=100)
    parser.add_argument('--output_path',type=str,default='data')
    parser.add_argument('--hidden_size',type=int,default=256)
    parser.add_argument('--input_size',type=int,default=100)
    parser.add_argument('--n_voc',type=int,default=1671)
    parser.add_argument('--w2v_path',type=str,default='data/w2v_vectors.pkl')

    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--seed',type=int,default=2021)


    args=parser.parse_args()

    #设置参数
    args.vocab=pickle.load(open(config.vocab_path,'rb'))
    args.vocab_size=len(args.vocab)
    args.embedding_matrix=pickle.load(open(config.embedding_matrix,'rb'))
    args.embedding_dim=config.embedding_dim
    args.output_dir='data'

    #读数据
    train_x=pickle.load(open(config.train_x,'rb'))
    train_y=pickle.load(open(config.train_y,'rb'))
    dev_x=pickle.load(open(config.dev_x,'rb'))
    dev_y=pickle.load(open(config.dev_y,'rb'))
    test=pickle.load(open(config.test,'rb'))
    train_dataset=TextDataset(args,train_x,train_y)
    dev_dataset=TextDataset(args,dev_x,dev_y)

    #建立模型
    model=Lstm(args)

    #训练模型
    tr_loss, best_single_acc, avg_loss, tr_nb = 0, 0, 0, 0
    for i in range(3):
        print('当前 epoch : {}'.format(i+1))
        train_dataloader=DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4,collate_fn=collate_fn)
        optimizer=Adam(params=model.parameters(),lr=args.lr,eps=1e-8,weight_decay=0.08)
        print('--------开始训练--------')
        model.zero_grad()
        tr_num=0
        train_loss=0
        for step, batch in tqdm(enumerate(train_dataloader)):
            x,labels,lengths=batch
            output=model.forward(x,lengths)
            loss=model.loss(output,labels,lengths)
            loss.backward()
            tr_num+=1
            train_loss+=loss.item()
            tr_loss+=loss.item()
            #输出log
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            if (step+1)%200 ==0:
                print('  epoch {}  step  {}  loss  {}'.format(i,step,avg_loss))
                #logger.info('  epoch {}  step  {}  loss  {}'.format(i,step,avg_loss))
            #梯度更新
            optimizer.step()
            optimizer.zero_grad()

            #验证

            if (step+1)%300==0 :
                with torch.no_grad():
                    print('step 为 {}  验证'.format(step))
                    single_acc=model.evaluate(DataLoader(dataset=dev_dataset,batch_size=4,num_workers=4,collate_fn=collate_fn))
                    print('----单个字符准确率 {}  '.format(single_acc))
                    #logger.info('----单个字符准确率 {}  '.format(single_acc))
                    if single_acc>best_single_acc:
                        best_single_acc=single_acc
                        torch.save(model.state_dict(),config.model_path)


        #完成一个batch
        with torch.no_grad():
            single_acc = model.evaluate(DataLoader(dataset=dev_dataset, batch_size=4, num_workers=4,collate_fn=collate_fn))
            print('----单个字符准确率 {}  '.format(single_acc))
            logger.info('----单个字符准确率 {}'.format(single_acc))
            if single_acc > best_single_acc:
                best_single_acc = single_acc
                torch.save(model.state_dict(), config.model_path)

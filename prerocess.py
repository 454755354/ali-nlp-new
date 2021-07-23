from tqdm import tqdm
import os
from collections import Counter
import pickle

label_1=('B','E','I')  #O单独算
label_2=('prov','city','district','devzone','town','community','village_group','road','roadno','poi',
         'subpoi','houseno','cellno','floorno','roomno','detail','assist','distance','intersection','redundant','others')



def read_data(path):
    '''
    :param path:
    :return: 两个二维列表：文本，标签
    '''
    all=[]
    with open(path, 'r') as f:
        tmp = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line:
                tmp.append(line)
            elif tmp:
                all.append(tmp)
                tmp = []
            else:
                continue
    z, x= [], []
    for i in all:
        if all == []:
            continue
        z_tmp = []
        x_tmp = []
        for j in i:
            j_1, j_2 = j.split(' ')
            z_tmp.append(j_1)
            x_tmp.append(j_2)
        z.append(z_tmp)
        x.append(x_tmp)
    return z,x

def read_test(path):
    import re
    patten=re.compile(r'\d+\x01')
    res=[]
    with open(path) as f:
        for line in f.readlines():
            res.append(list(re.sub(patten,'',line.strip())))
    return res

def l2i(label,label2id):
    res=[]
    for i in label:
        tmp=[]
        for j in i:
            if j in label2id.keys():
                tmp.append(label2id[j])
            else:
                tmp.append(label2id['O'])
        res.append(tmp)
    return res


def is_chinese(char):
    return char >= '\u4e00' and char <= '\u9fff'

if __name__=='__main__':
    #
    train_path=os.path.join('./data','train.txt')
    dev_path=os.path.join('./data','dev.txt')
    test_path=os.path.join('./data','test.txt')
    train_x,train_y=read_data(train_path)
    dev_x,dev_y=read_data(dev_path)
    test_x=read_test(test_path)
    print('读数')
    print('label生成id')
    label2id = {}
    id2label = {}
    label = []
    for i in label_1:
        for j in label_2:
            label.append('-'.join([i, j]))
    label.append('O')
    for i, j in enumerate(label):
        label2id[j] = i
        id2label[i] = j
    print('word生成id')
    word2id={}
    id2word={}
    counter=Counter()
    for i in train_x:
        for j in i:
            if is_chinese(j):
                counter[j]+=1
    for i in dev_x:
        for j in i:
            if is_chinese(j):
                counter[j] += 1
    for i in test_x:
        for j in i:
            if is_chinese(j):
                counter[j] += 1
    word_set=[]
    for k,v in counter.items():
        if v>10:
            word_set.append(k)
    word_set.append('num')
    word_set.append('char')
    word_set.append('spz')
    word_set.append('unknown')
    for i,j in enumerate(word_set,start=1):
        word2id[j] = str(i)
        id2word[str(i)] = j
    print("保存文件")
    train_x_id = []
    dev_x_id = []
    test_x_id = []
    for i in train_x:
        tmp = []
        for j in i:
            if is_chinese(j):
                if j in word_set:
                    tmp.append(word2id[j])
                else:
                    tmp.append(word2id['spz'])

            elif '0'<=str(j)<='9':
                tmp.append(word2id['num'])
            elif 'a'<=j<='z' or 'A'<=j<='Z':
                tmp.append(word2id['char'])
            else:
                tmp.append(word2id['unknown'])
        train_x_id.append(tmp)
    for i in dev_x:
        tmp = []
        for j in i:
            if is_chinese(j):
                if j in word_set:
                    tmp.append(word2id[j])
                else:
                    tmp.append(word2id['spz'])
            elif '0'<=str(j)<='9':
                tmp.append(word2id['num'])
            elif 'a' <= j <= 'z' or 'A' <= j <= 'Z':
                tmp.append(word2id['char'])
            else:
                tmp.append(word2id['unknown'])
        dev_x_id.append(tmp)

    for i in test_x:
        tmp = []
        for j in i:
            if is_chinese(j):
                if j in word_set:
                    tmp.append(word2id[j])
                else:
                    tmp.append(word2id['spz'])
            elif '0'<=str(j)<='9':
                tmp.append(word2id['num'])
            elif 'a' <= j <= 'z' or 'A' <= j <= 'Z':
                tmp.append(word2id['char'])
            else:
                tmp.append(word2id['unknown'])
        test_x_id.append(tmp)
    print(len(train_x_id))
    print(len(dev_x_id))
    print(len(test_x_id))
    print("保存")

    train_y_id=l2i(train_y,label2id)
    dev_y_id=l2i(dev_y,label2id)
    with open('data/train_y_id.pkl','wb') as f:
        pass
    with open('data/dev_y_id.pkl','wb') as f:
        pass
    pickle.dump(train_y_id,open('data/train_y_id.pkl','wb'))
    pickle.dump(dev_y_id,open('data/dev_y_id.pkl','wb'))

    exit()
    with open('data/vocab.pkl','w') as f:
        pass
    pickle.dump(word_set,open('data/vocab.pkl','wb'))
    pickle.dump(label2id,open('data/label2id.pkl','wb'))
    pickle.dump(id2label,open('data/id2label.pkl','wb'))
    pickle.dump(word2id,open('data/word2id.pkl','wb'))
    pickle.dump(id2word,open('data/id2word.pkl','wb'))
    pickle.dump(train_x_id,open('data/train_x_id.pkl','wb'))
    pickle.dump(dev_x_id,open('data/dev_x_id.pkl','wb'))
    pickle.dump(test_x_id,open('data/test_x_id.pkl','wb'))




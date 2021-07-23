import torch
from load_data import collate_fn,ModelEmbedding,TextDataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pickle
from torch.nn import CrossEntropyLoss, MSELoss
import config as config



def get_mask(length):
    #求一个batch中最大的输入长度
    max_len=int(max(length))
    mask=torch.Tensor()
    for len_ in length:
        mask=torch.cat((mask,torch.Tensor([[1]*len_ + [0]*(max_len-len_)])),dim=0)
    return mask

class Lstm(nn.Module):
    def __init__(self,args):
        super(Lstm,self).__init__()
        self.args=args
        self.input_dim=self.args.input_size
        self.hidden_dim=self.args.hidden_size
        self.n_class=self.args.n_class
        self.n_voc=self.args.n_voc
        # w2v训练结果的wv.vectors赋值给nn.embedding
        self.embedding=nn.Embedding(num_embeddings=self.n_voc,
                                    embedding_dim=100,padding_idx=0)
        self.embedding=self.embedding.from_pretrained(torch.tensor(pickle.load(open(config.embedding_matrix,'rb'))), freeze=True)

        self.lstm=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=2,batch_first=True)
        self.linear=nn.Linear(self.hidden_dim,self.n_class)
        #self.transiton_matrix=nn.Parameter(torch.rand(self.n_class,self.n_class))
        #self.reset_parameters()
        self.hidden=self.init_hidden()

    def init_hidden(self):
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (
            torch.zeros(2,4,self.hidden_dim),
            torch.zeros(2,4, self.hidden_dim)
        )

    def forward(self,input_data,length):
        embedding_input=self.embedding(input_data)
        packed_embed_input=pack_padded_sequence(embedding_input,lengths=length,batch_first=True,enforce_sorted=False)
        packed_out,_=self.lstm(packed_embed_input,self.hidden)
        output,_=pad_packed_sequence(packed_out)
        output=self.linear(output)
        output=output.transpose(0,1)
        return output

    def loss(self,x,y,length):
        mask=get_mask(length)

        #生成one_hot label
        #label=torch.zeros(self.n_class)
        #for i in y:
        #    label=torch.cat((label,one_hot(i,self.n_class)),0)
        #label=label[1:,:].view(y.shape[0],y.shape[1],-1)
        #loss=(x-label).sum(dim=2)
        #loss=loss*mask
        # print(x)
        # print('=======')
        # print(y)
        # print('================')
        # print(length)
        loss_func=CrossEntropyLoss()
        loss=torch.tensor(0.)
        for i in range(len(length)):
            loss+=loss_func(F.log_softmax(x[i][:length[i]],dim=1),y[i][:length[i]])
        return loss

    def evaluate(self,dev_dataloader):
        count=0
        right=0
        for step,batch in enumerate(dev_dataloader):
            x,labels,lengths=batch
            output=self.forward(x,lengths)
            output=output.argmax(dim=2)
            for i,j in zip(output.reshape(-1),labels.reshape(-1)):
                if j!=-1 and i==j:
                    right+=1
                count+=1
        print('总测试字数为 ： {}   预测正确字数为 ： {}'.format(count,right))
        return round(right/count,5)









def one_hot(x,class_count):
    return torch.eye(class_count)[x,:]
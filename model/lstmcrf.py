import torch
from load_data import collate_fn,ModelEmbedding,TextDataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn as nn
from torch.nn import init
import pickle

def get_mask(length):
    #求一个batch中最大的输入长度
    max_len=int(max(length))
    mask=torch.Tensor()
    for len_ in length:
        mask=torch.cat((mask,torch.Tensor([[1]*len_ + [0]*(max_len-len_)])),dim=0)
    return mask

class LstmCrf(nn.Module):
    def __init__(self,args):
        super(LstmCrf,self).__init__()
        self.input_dim=args.input_dim
        self.hidden_dim=args.hiiden_dim
        self.n_class=args.n_class
        self.n_voc=args.n_voc
        # w2v训练结果的wv.vectors赋值给nn.embedding
        self.embedding=nn.Embedding(num_embeddings=self.n_voc,
                                    embedding_dim=self.input_dim,padding_idx=0)
        self.embedding=self.embedding.from_pretrained(torch.tensor(pickle.load(open(''))),freeze=True)

        self.lstm=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=2,batch_first=True)
        self.linear=nn.Linear(self.hidden_dim,self.n_class)
        self.transiton_matrix=nn.Parameter(torch.rand(self.n_class,self.n_class))
        self.reset_parameters()
        self.softmax=nn.Softmax(dim=1)

    def reset_parameters(self):
        # 转移概率位于log空间
        init.normal_(self.transiton_matrix)
        #w2v训练结果的wv.vectors赋值给nn.embedding

    def forward(self,x,y):

        pass

    def loss(self,x,y):
        pass




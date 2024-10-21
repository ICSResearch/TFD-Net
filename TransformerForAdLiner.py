# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:17:43 2022

@author: ZhengHejie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from drop import DropPath
from util.positional_encoding import PostionalEncoding


torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, device):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.device = device

    def forward(self, x):
        N = x.shape[0]
        x = torch.reshape(x, [N, -1, self.patch_size]).to(self.device)
        return x
        
class SelfAttention(nn.Module):
    def __init__(self, patch_size, heads):
        super(SelfAttention,self).__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = patch_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, patch_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values) 
        keys = self.keys(keys)  
        queries = self.queries(query)
        # 对向量、矩阵、张量的求和运算
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        
        attention = torch.softmax(energy / (self.patch_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    """
    Transformer块
    :param patch_size: 维度
    :param heads: 多头注意力数
    :param dropout:
    :param forward_expansion: 隐藏层维数
    """
    def __init__(self, patch_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(patch_size, heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(patch_size)
        self.norm2 = nn.LayerNorm(patch_size)       
        
        self.feed_forward = nn.Sequential(
            nn.Linear(patch_size, forward_expansion*patch_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*patch_size, patch_size)
            )
        # self.dropout = nn.Dropout(dropout)
        self.drop = DropPath(0.02)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        
        x = self.norm1(self.drop(attention) + query)
        out = self.norm2(self.drop(self.feed_forward(x)) + x)
        # return x,out
        return out
class Encoder_Res(nn.Module):
    def __init__(self,
                 block_size,
                 num_layers,
                 heads,
                 device, #device="cuda"
                 forward_expansion,
                 dropout,
                 num_block):
        super(Encoder_Res, self).__init__()
        self.device = device

        self.block_size = block_size
        self.patch_embedding = PatchEmbedding(patch_size=self.block_size , device=device)
        # 其为一个简单的存储固定大小的词典的嵌入向量的查找表
        # num_embeddings(python: int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0 - 4999）
        # embedding_dim(python: int) – 嵌入向量的维度，即用多少维来表示一个符号。
        self.position_embedding = nn.Embedding(num_block, embedding_dim=self.block_size )
        self.num_block = num_block

        self.layers = nn.ModuleList(
            [TransformerBlock(self.block_size ,
                              heads, 
                              dropout=dropout, 
                              forward_expansion=forward_expansion
                              )
             for _ in range(num_layers)]
            )
        self.dropout = nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(nn.Flatten(),
                                       nn.Linear(self.block_size *num_block,80),
                                       nn.ReLU(),
                                       nn.Linear(80,80),
                                       nn.ReLU())
        # 添加残差连接
        self.residual = torch.nn.Sequential(
            nn.Flatten(),
            nn.Identity(),
            nn.Linear(self.block_size * num_block, 80),
            nn.ReLU(inplace=True)
        )

        self.result = torch.nn.Sequential(
            nn.Linear(80, 1),
        )

        # 它针对网络中的所有线性层，如果该层是一个nn.Linear层，则对其权重进行初始化。
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        N = x.shape[0]
        positions = torch.arange(0, self.num_block).expand(N, self.num_block).to(self.device)
        # 残差
        out = self.patch_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x_hat,out_en = layer(out, out, out)
        # 将主网络的输出和残差连接的输出相加
        out_1 = self.mlp(out_en)+self.residual(x_hat)
        out_line = self.result(out_1)
        out_re = self.dropout(out_line)
        return torch.sigmoid(out_re)


class Encoder(nn.Module):
    def __init__(self,
                 block_size,
                 num_layers,
                 heads,
                 device,  # device="cuda"
                 forward_expansion,
                 dropout,
                 num_block):
        super(Encoder, self).__init__()

        self.block_size = block_size
        self.device = device
        self.patch_embedding = PatchEmbedding(patch_size=block_size, device=device)
        # 其为一个简单的存储固定大小的词典的嵌入向量的查找表
        # num_embeddings(python: int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0 - 4999）
        # embedding_dim(python: int) – 嵌入向量的维度，即用多少维来表示一个符号。
        self.position_embedding = nn.Embedding(num_block, embedding_dim=block_size)
        # self.position_embedding = PostionalEncoding(num_block, block_size, device=device)
        self.num_block = num_block

        self.layers = nn.ModuleList(
            [TransformerBlock(block_size,
                              heads,
                              dropout=dropout,
                              forward_expansion=forward_expansion
                              )
             for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(nn.Flatten(),
                                       nn.Linear(block_size * num_block, 80),
                                       nn.ReLU(),
                                       nn.Linear(80, 1))

        # 它针对网络中的所有线性层，如果该层是一个nn.Linear层，则对其权重进行初始化。
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        N = x.shape[0]
        positions = torch.arange(0, self.num_block).expand(N, self.num_block).to(self.device)
        # 残差
        out = self.patch_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            out = layer(out, out, out)
        out = self.dropout(self.mlp(out))
        return torch.sigmoid(out)

class ADTransformer(nn.Module):
    def __init__(self,
                 block_size, #20
                 num_layers=8,
                 forward_expansion=4,
                 heads=4,
                 dropout=0.1,
                 device="cuda"):
        super(ADTransformer, self).__init__()
        self.num_block = 64
        self.encoder = Encoder(block_size, num_layers, heads, device,
                               forward_expansion, dropout, self.num_block)


    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src

    def predict(self, x):
        pred = self.forward(x)
        return pred
    def accuracy_predict(self, x):
        pred = self.forward(x)
        ans = []
        for t in pred:
            if t < 0.5:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

    @classmethod
    def from_pretrained(cls,
                        fpath) -> None:
        chkp = torch.load(fpath)
        model = cls(**chkp.pop("config"))
        # 切换模型为预测模型
        model.eval()
        #将预训练的参数权重加载到新的模型之中
        model.load_state_dict(chkp.pop("weights"))
        return model
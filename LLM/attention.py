import torch
from torch import nn
import math
def attention(query,key,value,dropout = None):
    """

    :param query: 每行是一个查询 n*m
    :param key: 每行是一个键的词向量 k*m
    :param value: k * 1
    :param dropout:
    :return:
    """
    d_k = query.size(-1) # 获得key的维度
    # 计算注意力得分
    # 转置后，此处为 n*m  *  m*k  =  n*k
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)


    p_attn = torch.softmax(scores,dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 得到最终的注意力结果 n*k  *  k*1  =  n * 1
    attn = torch.matmul(p_attn,value)
    return attn,p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args:ModelArgs,is_casual = False):
        super(MultiHeadAttention, self).__init__()
        # 为了提供足够的并行性能，需要将隐藏层节点均分给n个注意力头使用
        assert args.dim % args.n_heads == 0


        # 每个头分得的隐藏层维度
        self.head_dim = args.dim // args.n_heads
        # 注意力头的个数
        self.n_heads = args.n_heads

        # 权重矩阵，通过不含偏置项的线性层实现
        # n_embed是词向量长度
        # head_dim * n_heads 实现了矩阵的拼接，先拼接之后再做内积
        self.Wq = nn.Linear(args.n_embed,self.head_dim*self.n_heads,bias=False)
        self.Wk = nn.Linear(args.n_embed,self.head_dim*self.n_heads,bias=False)
        self.Wv = nn.Linear(args.n_embed,self.head_dim*self.n_heads,bias=False)

        # 输出权重的矩阵
        # 大小为dim * dim 其中head_dim * n_heads就是dim
        self.Wo = nn.Linear(self.head_dim*self.n_heads,args.dim,bias=False)

        self.attention_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)

        self.is_casual = is_casual
        # 设置mask矩阵用来遮蔽未来信息
        # 由于使用了多头注意力，mask的形状由(1,max_len,max_len) 添加一个维度变为(1,1,max_len,max_len)
        if is_casual:
            mask = torch.full((1,1,args.max_seq_len,args.max_seq_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            # 将mask注册为模型的缓冲区
            self.register_buffer("mask",mask)

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        # q的形状为[batch_size,T,n_embed]
        bsz,seqlen,_ = q.shape

        # 计算q,k,v
        # 输入的维度是[B,T,n_embed] * [n_embed,dim] = [B,T,dim]
        xq,xk,xv = self.Wq(q),self.Wk(k),self.Wv(v)

        # 将q,k,v拆分为多头
        # 即将[B,T,dim] 拆分为 [B,T,n_heed,dim // n_head]
        # 由此将dim 维度拆分为 n_head * (dim//n_head) = n_head * head_dim 两个维度
        xq = xq.view(bsz,seqlen,self.n_heads,self.head_dim)
        xk = xk.view(bsz,seqlen,self.n_heads,self.head_dim)
        xv = xv.view(bsz,seqlen,self.n_heads,self.head_dim)

        # 需要交换q,k,v的后两个维度，方便先循环头，再在头里循环训练每个seq
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)


        # 计算注意力，此时形状为[B,n_heads,T,head_dim] * [B,n_heads,head_dim,T] = [B,n_heads,T,T]
        scores = torch.matmul(xq,xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if self.is_casual:
            assert hasattr(self,"mask")
            # 注意这里的mask形状
            scores = scores + self.mask[:,:,:seqlen,:seqlen]

        # 进行softmax运算，注意维度在最后一维，转换成xq的类型
        scores = torch.softmax(scores.float(),dim=-1).type_as(xq)

        # 进行dropout
        scores = self.attention_dropout(scores)

        # 计算输出结果，形状为[B,n_heads,T,T] * [B,n_heads,T,head_dim] = [B,n_heads,T,head_dim]
        output = torch.matmul(scores,xv)

        # 完成计算后，恢复维度，交换n_heads和T，
        output = output.transpose(1,2).contiguous().view(bsz,seqlen,-1)

        # 使用隐藏层
        output = self.Wo(output)
        # 残差连接
        output = self.residual_dropout(output)

        return output
    

class MLP(nn.Module):
    def __init__(self, dim:int,hidden_dim:int,dropout:float,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor):
        return self.dropout(self.w2(torch.relu(self.w1(x))))
    


class LayerNorm(nn.Module):
    def __init__(self, features,epsilon = 1e-6,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = epsilon

    def forward(self,X:torch.Tensor):
        mean = X.mean(-1,keepdim = True)
        std = X.std(-1,keepdim=True)
        return self.a_2 * (X - mean)/(std +self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_norm = LayerNorm(args.n_embed)
        self.attention = MultiHeadAttention(args=args,is_casual=False)
        self.fnn_norm = LayerNorm(args.n_embed)
        self.feed_forward = MLP(args.dim,args.dim,args.dropout)

    def forward(self,X):
        norm_x = self.attention_norm(X)
        atten = self.attention(norm_x,norm_x,norm_x)
        h = X + atten
        output =h +  self.feed_forward.forward(self.fnn_norm(h)) 
        return output

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.n_embed)

    def forward(self,X):
        for layer in self.layers:
            X = layer(X)
        return self.norm(X)
    

class DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_norm1 = LayerNorm(args.n_embed)
        self.mask_attention = MultiHeadAttention(args=args,is_casual=True)
        self.attention_norm2 = LayerNorm(args.n_embed)
        self.nomask_attention = MultiHeadAttention(args=args,is_casual=False)

        self.fnn_norm = LayerNorm(args.n_embed)
        self.feed_forward = MLP(args.dim,args.dim,args.dropout)

    def forward(self,X,encoder_out):
        norm_x = self.attention_norm1(X)
        X = X + self.mask_attention.forward(norm_x,norm_x,norm_x)
        norm_x = self.attention_norm2(X)
        h = X + self.nomask_attention.forward(norm_x,encoder_out,encoder_out)
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
    
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.n_embed)

    def forward(self,x,encoder_out):
        for layer in self.Layers:
            x = layer(x,encoder_out)
        return self.norm(x)


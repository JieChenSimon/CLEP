import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from losses import TripletMSELoss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings_with_tsne(embeddings, labels=None, perplexity=30, n_iter=1000):
    """
    对嵌入进行 t-SNE 可视化。

    参数:
    embeddings - 一个嵌入矩阵，形状为 (n_samples, n_features)。
    labels - 可选，嵌入的标签。用于在可视化时给不同类别的点赋予不同的颜色。
    perplexity - t-SNE 的困惑度参数。
    n_iter - t-SNE 的迭代次数。
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    if labels is not None:
        unique_labels = list(set(labels))
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig('/home/chen/workspace/codeproject/CL4acrossVersionSC/visualize/t-SNE.png')


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

        


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
    def forward(self, inputs_ids, position_idx, attn_mask, labels = None): 
        bs, l = inputs_ids.size()
        inputs_ids = inputs_ids.unsqueeze(1).view(bs * 1, l)
        position_idx = position_idx.unsqueeze(1).view(bs * 1, l)  
        attn_mask = attn_mask.unsqueeze(1).view(bs * 1, l, l)
        
        # Embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2) 
        #直接访问了 RoBERTa 模型的词嵌入层，利用 RobertaForSequenceClassification 内部的 RoBERTa 模型部分来处理文本数据。并不是使用整个RobertaForSequenceClassification类
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]    
        
        #是在执行 RoBERTa 模型的前向传播，这包括了多头注意力机制（multi-head attention）和层归一化（layer normalization）等一系列内部操作
        embedding_outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0] 


        # ，通过索引 [0] 来访问了这个元组的第一个元素，即最后一层的隐藏状态。因此，embedding_outputs 就是模型最后一层的隐藏状态，它代表了模型对输入数据的嵌入表示。     
        
        logits=self.classifier(embedding_outputs)
        prob=F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #contrastive_loss = contrastive_loss_function(predicted, labels)  # 使用您定义的对比损失函数
            loss_CrossEntropy = loss_fct(logits, labels)
            return loss_CrossEntropy,prob, embedding_outputs
        else:
            return prob
        
# 下面是encoder和classifier分开        
# class Model(nn.Module):   
#     def __init__(self, encoder, config, tokenizer, args):
#         super(Model, self).__init__()
#         self.encoder = encoder
#         self.config = config
#         self.tokenizer = tokenizer
#         self.args = args
    
#     def forward(self, inputs_ids, position_idx, attn_mask, labels = None): 
#         bs, l = inputs_ids.size()
#         inputs_ids = inputs_ids.unsqueeze(1).view(bs * 1, l)
#         position_idx = position_idx.unsqueeze(1).view(bs * 1, l)  
#         attn_mask = attn_mask.unsqueeze(1).view(bs * 1, l, l)
        
#         # Embedding
#         nodes_mask = position_idx.eq(0)
#         token_mask = position_idx.ge(2) 
#         inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
#         # print('inputs_embeddings in Model encoder ', inputs_embeddings)
#         nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
#         nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
#         avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
#         inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]    
        
#         outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
#         # print('outputs in Model encoder ', outputs)

#         return outputs

#     # New function for classification
#     def classify(encoder_output, classifier, labels=None):
#         logits = classifier(encoder_output)
#         prob = F.softmax(logits, dim=1)
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits, labels)
#             return loss, prob
#         else:
#             return prob

       

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super(CNN_Text,self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len, d)
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d=100, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True, dist_embeds = False, target_var = 0.1):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            sys.stdout.write("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            # if n_d != len(embvecs[0]):
            #     sys.stdout.write("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
            #         n_d, len(embvecs[0]), len(embvecs[0])
            #     ))
            n_d = len(embvecs[0])

        # for w in deep_iter(words):
        #     if w not in word2id:
        #         word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        self.dist_embeds = dist_embeds
        self.target_variance = target_var

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            sys.stdout.write("embedding shape: {}\n".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False
        
        self.embedding_variance = nn.Embedding(self.n_V, n_d)

    
    def sample_embeds(self, embed_mean, embed_var):
        epsilon = torch.unsqueeze(torch.randn((embed_mean.shape[0], embed_mean.shape[1])), dim=2)
        epsilon = epsilon.repeat((1,1,embed_mean.shape[2]))
        
        embed = embed_mean + torch.exp(embed_var)*epsilon.cuda()
        return embed

    def kl_loss(self, embed_mean_1, embed_mean_2, embed_var_1, embed_var_2):
        exponential = embed_var_1 - embed_var_2 - torch.pow(embed_mean_1 - embed_mean_2, 2) / embed_var_2.exp() - torch.exp(embed_var_1 - embed_var_2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, input):
        embed_mean = self.embedding(input)
        embed_variance = self.embedding_variance(input)
        
        if self.dist_embeds:
            kl_loss = self.kl_loss(embed_mean, embed_mean, embed_variance, torch.full(embed_variance.shape, 0.5).cuda())
            return self.sample_embeds(embed_mean, embed_variance), kl_loss
            
        else:
            return embed_mean

import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, heads, kv, dropout):
        super(GroupedQueryAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.kv_heads = kv
        self.kv_dim = kv * self.head_dim
        self.groups = heads // kv

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"
        assert(self.heads % self.kv_heads == 0), "Groups size has to be divisible by number of heads"


        self.Lq = nn.Linear(embed_size, embed_size, bias=True)
        self.Lkv = nn.Linear(embed_size, 2*self.kv_dim, bias=True)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, seq, mask=None, kv_cache=None):
        N, seq_len, _ = seq.shape
        
        Wq = self.Lq(seq)
        Wkv = self.Lkv(seq)

        Q = Wq.reshape(N, seq_len, self.heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)

        KV = Wkv.view(N, seq_len, 2, self.kv_heads, self.head_dim)
        K = KV[:, :, 0, :, :]      
        V = KV[:, :, 1, :, :]    
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        if kv_cache is not None:
            if kv_cache["k"] is not None:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)

            new_cache = {"k": K, "v": V}
        else:
            new_cache = {"k": K, "v": V}

        K = K.repeat_interleave(self.groups, dim=1)  
        V = V.repeat_interleave(self.groups, dim=1) 

        # assert (Q.shape == K.shape == V.shape), "Q,K,V with different shapes"

        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K])

        if mask is not None and Q.shape[2] > 1:
            energy = energy.masked_fill(mask == 0, torch.finfo(energy.dtype).min)

        attention = self.attn_dropout(
            torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)
        )

        out = torch.einsum("nhqk,nhkd->nhqd", [attention, V])
        out = out.permute(0, 2, 1, 3).reshape(N, seq_len, self.embed_size)

        out = self.fc_out(out)
        return out, new_cache
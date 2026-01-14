import torch.nn as nn
import torch
import torch.nn.functional as F

class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, self.d_k)
        self.W_V = nn.Linear(d_model, self.d_k)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape


        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)


        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)


        K = K.view(batch_size, seq_len, 1, self.d_k)
        K = K.transpose(1, 2)

        V = V.view(batch_size, seq_len, 1, self.d_k)
        V = V.transpose(1, 2)


        # Q: (batch, num_heads, seq_len, d_k)
        # K: (batch, 1, seq_len, d_k) -> broadcast до (batch, num_heads, seq_len, d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)

        # V: (batch, 1, seq_len, d_k) -> broadcast
        context = torch.matmul(attn_probs, V)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)

        output = self.W_O(context)
        return output


"""model = MQA(100, 4)
x = torch.rand(32, 10, 100)  # (batch, seq_len, d_model)
out = model(x)
print(out.shape)"""
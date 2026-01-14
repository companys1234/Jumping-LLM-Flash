import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.d_k = d_model // num_heads

        # Query проекция для всех голов
        self.W_Q = nn.Linear(d_model, d_model)
        # Key/Value проекция для уменьшенного числа голов
        self.W_K = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape


        Q = self.W_Q(x)

        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Повторяем Key/Value для группированного внимания
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)


        context = torch.matmul(attn_probs, V)


        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)


        output = self.W_O(context)
        return output



"""model = GQA(d_model=100, num_heads=4, num_kv_heads=2)  # 4 query heads, 2 key/value heads
x = torch.rand(32, 10, 100)  # (batch, seq_len, d_model)
out = model(x)
print(out.shape)  """
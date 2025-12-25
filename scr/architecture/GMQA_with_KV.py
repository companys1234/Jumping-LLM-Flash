import torch
import torch.nn as nn
import torch.nn.functional as F

class GMQA_with_KV(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups):
        super().__init__()
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads должно делиться на num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model // num_heads * num_kv_groups)
        self.W_V = nn.Linear(d_model, d_model // num_heads * num_kv_groups)
        self.W_O = nn.Linear(d_model, d_model)

        self.register_buffer("cached_k", None)
        self.register_buffer("cached_v", None)

    def forward(self, x, use_cache=False, past_kv=None):

        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, D)

        kv_dim = self.d_k * self.num_kv_groups
        K = self.W_K(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)  # (B, G, T, D)
        V = self.W_V(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)  # (B, G, T, D)

        # Если есть кэш — добавляем к прошлым значениям
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # Повторяем K/V до H голов
        kv_repeat = self.num_heads // self.num_kv_groups
        K_rep = K.repeat_interleave(kv_repeat, dim=1)  # (B, H, T, D)
        V_rep = V.repeat_interleave(kv_repeat, dim=1)

        # Attention
        attn_scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V_rep)  # (B, H, T, D)

        # Слияние голов
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        # Обновляем кэш, если нужно
        present_kv = (K, V) if use_cache else None
        return output, attn_weights, present_kv

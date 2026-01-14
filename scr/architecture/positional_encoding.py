import torch
import math
def positional_encodings(seq_len, d_model, device):
    positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)

    # Создаем делители (10000^(2i/d_model)) для каждого измерения
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
        (-math.log(10000.0) / d_model)
    )

    # Инициализируем тензор кодирований
    pe = torch.zeros(seq_len, d_model, device=device)


    pe[:, 0::2] = torch.sin(positions * div_term)


    pe[:, 1::2] = torch.cos(positions * div_term[:d_model // 2])

    return pe




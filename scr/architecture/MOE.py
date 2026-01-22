import torch
import torch.nn as nn
class MoE(nn.Module):
    def __init__(self, expert, input_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts

        # Определяем экспертов
        self.experts = expert

        # Гейт для выбора экспертов
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Применяем гейт к входным данным
        batch_size = x.size(0)
        gate_outputs = self.gate(x.view(batch_size, -1))  # Приводим к вектору
        gate_weights = torch.softmax(gate_outputs, dim=-1)  # Получаем вероятности выбора экспертов

        # Получаем выходы от каждого эксперта
        expert_outputs = [expert(x) for expert in self.experts]

        # Смешиваем выходы экспертов по весам
        output = sum(gate_weights[:, i].view(batch_size, 1) * expert_outputs[i] for i in range(self.num_experts))

        return output

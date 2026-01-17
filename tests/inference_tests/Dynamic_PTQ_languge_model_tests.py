from jumping_LLM_flash.tests.preprocessing_tests.Dynamic_PTQ_language_model import quantize_language_model, quantize_model_layer_weights, quantize_model_simple, quantize_model_linear_layers, get_model_linear_activation, print_info_model, get_activation_model
import torch
import torch.nn as nn
import numpy as np


vocab_size = 32000

# Создаем модель (для примера, вам нужно использовать ваш реальный класс)
class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Тестовая модель для демонстрации
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 128)
        self.out_proj = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.out_proj(x)

# Создаем и тестируем модель
print("Тестирование на простой модели...")
test_model = TestModel()
quantized_test_model = quantize_language_model(test_model, vocab_size=vocab_size)
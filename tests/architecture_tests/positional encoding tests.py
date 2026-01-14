
import math
import numpy as np
import torch
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

x = torch.rand(16,100)
z = positional_encodings(16,100,'cpu')
print('z', z)
print('x', x)
print('x+z', torch.cat((x, z), 0))






def test_positional_encodings_basic():
    """Тест основных свойств позиционных кодирований"""
    seq_len = 10
    d_model = 16
    device = torch.device('cpu')

    pe = positional_encodings(seq_len, d_model, device)

    # Проверка размеров
    assert pe.shape == (seq_len, d_model), f"Expected shape {(seq_len, d_model)}, got {pe.shape}"

    # Проверка типа данных
    assert pe.dtype == torch.float32, f"Expected dtype torch.float32, got {pe.dtype}"

    # Проверка устройства
    assert pe.device == device, f"Expected device {device}, got {pe.device}"

    # Проверка, что значения в пределах [-1, 1] (sin/cos)
    assert torch.all(pe >= -1.0) and torch.all(pe <= 1.0), "Values should be in range [-1, 1]"

    print("✓ test_positional_encodings_basic passed")


"""def test_positional_encodings_odd_d_model():
 
    seq_len = 5
    d_model = 15  # нечетное
    device = torch.device('cpu')

    pe = positional_encodings(seq_len, d_model, device)

    # Проверка размеров с нечетной размерностью
    assert pe.shape == (seq_len, d_model), f"Expected shape {(seq_len, d_model)}, got {pe.shape}"

    # Проверка, что последний столбец для нечетной размерности равен 0
    # (поскольку 1::2 для нечетного d_model не покрывает последний индекс)
    assert torch.all(pe[:, -1] == 0), "Last column should be 0 for odd d_model"

    print("✓ test_positional_encodings_odd_d_model passed")"""


def test_positional_encodings_pattern():
    """Тест паттерна sin/cos чередования"""
    seq_len = 3
    d_model = 8
    device = torch.device('cpu')

    pe = positional_encodings(seq_len, d_model, device)

    # Проверка чередования sin и cos
    for i in range(seq_len):
        for j in range(0, d_model, 2):
            # Четные индексы: sin
            assert torch.isclose(
                pe[i, j],
                torch.sin(torch.tensor(i * math.exp(j * (-math.log(10000.0) / d_model)))),
                rtol=1e-5
            ), f"Position [{i}, {j}] should be sin"

        for j in range(1, d_model, 2):
            # Нечетные индексы: cos
            if j < d_model:  # проверяем границы
                k = (j - 1) // 2
                assert torch.isclose(
                    pe[i, j],
                    torch.cos(torch.tensor(i * math.exp(k * (-math.log(10000.0) / d_model)))),
                    rtol=1e-5
                ), f"Position [{i}, {j}] should be cos"

    print("✓ test_positional_encodings_pattern passed")


def test_positional_encodings_device():
    """Тест работы с разными устройствами"""
    seq_len = 4
    d_model = 8

    # Тестируем на CPU
    pe_cpu = positional_encodings(seq_len, d_model, torch.device('cpu'))
    assert pe_cpu.device == torch.device('cpu'), "Should be on CPU"

    # Тестируем на GPU, если доступно
    if torch.cuda.is_available():
        pe_gpu = positional_encodings(seq_len, d_model, torch.device('cuda'))
        assert pe_gpu.device == torch.device('cuda'), "Should be on CUDA"

        # Проверяем, что значения одинаковые (с точностью до погрешности)
        assert torch.allclose(pe_cpu, pe_gpu.cpu(), rtol=1e-5), "CPU and GPU results should match"

    print("✓ test_positional_encodings_device passed")


def test_positional_encodings_different_lengths():
    """Тест с разными длинами последовательностей"""
    test_cases = [
        (1, 16),  # минимальная длина
        (10, 32),  # средняя
        (100, 64),  # длинная
        (512, 512),  # типичная для трансформеров
    ]

    for seq_len, d_model in test_cases:
        pe = positional_encodings(seq_len, d_model, torch.device('cpu'))

        # Проверка размеров
        assert pe.shape == (seq_len, d_model), \
            f"Failed for seq_len={seq_len}, d_model={d_model}"

        # Проверка уникальности позиций
        # Разные позиции должны иметь разные кодировки
        if seq_len > 1:
            for i in range(seq_len - 1):
                assert not torch.allclose(pe[i], pe[i + 1], rtol=1e-5), \
                    f"Positions {i} and {i + 1} should be different"

    print("✓ test_positional_encodings_different_lengths passed")


def test_positional_encodings_symmetry():
    """Тест симметричных свойств"""
    seq_len = 10
    d_model = 32
    device = torch.device('cpu')

    pe = positional_encodings(seq_len, d_model, device)

    # Проверка, что для каждой частоты есть и sin и cos компоненты
    # (кроме последней частоты при нечетной размерности)
    half_d = d_model // 2

    for i in range(seq_len):
        for freq_idx in range(half_d):
            sin_idx = 2 * freq_idx
            cos_idx = 2 * freq_idx + 1

            if cos_idx < d_model:  # проверка границ
                # sin и cos для одной частоты должны быть разными
                assert not torch.isclose(pe[i, sin_idx], pe[i, cos_idx], rtol=1e-5), \
                    f"Sin and cos at frequency {freq_idx} should be different"

    print("✓ test_positional_encodings_symmetry passed")


def test_positional_encodings_reproducibility():
    """Тест воспроизводимости результатов"""
    seq_len = 8
    d_model = 16
    device = torch.device('cpu')

    # Два вызова должны давать одинаковый результат
    pe1 = positional_encodings(seq_len, d_model, device)
    pe2 = positional_encodings(seq_len, d_model, device)

    assert torch.allclose(pe1, pe2, rtol=1e-7), "Results should be reproducible"

    # Проверка конкретных значений для контрольной точки
    # Проверяем несколько известных значений
    assert torch.isclose(pe1[0, 0], torch.tensor(0.0), rtol=1e-5), "pe[0,0] should be 0"
    assert torch.isclose(pe1[0, 1], torch.tensor(1.0), rtol=1e-5), "pe[0,1] should be 1"

    print("✓ test_positional_encodings_reproducibility passed")


def test_positional_encodings_extreme_cases():
    """Тест крайних случаев"""
    # d_model = 1 (минимальная размерность)
    pe = positional_encodings(3, 1, torch.device('cpu'))
    assert pe.shape == (3, 1)
    assert torch.all(pe[:, 0] == 0), "For d_model=1, all values should be 0"

    # d_model = 2 (минимальная для sin/cos)
    pe = positional_encodings(3, 2, torch.device('cpu'))
    assert pe.shape == (3, 2)

    # seq_len = 0 (пустая последовательность)
    pe = positional_encodings(0, 8, torch.device('cpu'))
    assert pe.shape == (0, 8)
    assert len(pe) == 0

    print("✓ test_positional_encodings_extreme_cases passed")


def run_all_tests():
    """Запуск всех тестов"""
    print("Running positional encodings tests...")
    print("-" * 50)

    test_functions = [
        test_positional_encodings_basic,
        test_positional_encodings_pattern,
        test_positional_encodings_device,
        test_positional_encodings_different_lengths,
        test_positional_encodings_symmetry,
        test_positional_encodings_reproducibility,
        test_positional_encodings_extreme_cases,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            raise

    print("-" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
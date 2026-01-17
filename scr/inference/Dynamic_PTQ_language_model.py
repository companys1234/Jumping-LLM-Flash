import torch
import torch.nn as nn
import numpy as np

# Словарь для хранения активаций
activations = {}
all_quantized_weights = []

def get_activation_model(name):
    """Функция-хук для сохранения активаций"""
    def hook(model, input, output):
        # Обрабатываем кортежи
        if isinstance(output, tuple):
            # Берем первый элемент кортежа (обычно это тензор)
            output_tensor = output[0] if len(output) > 0 else None
        else:
            output_tensor = output

        activations[name] = {
            'activation': output_tensor.detach() if output_tensor is not None else None,
            'weights': model.weight.detach() if hasattr(model, 'weight') else None,
            'bias': model.bias.detach() if hasattr(model, 'bias') else None
        }
    return hook

def print_info_model(model, x):
    """Выводит полную информацию о модели: веса, смещения и активации"""

    # Собираем активации
    activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            # Обрабатываем разные типы выходов
            if isinstance(output, tuple):
                # Если выход - кортеж, берем первый элемент (обычно основной тензор)
                output_tensor = output[0] if len(output) > 0 else None
                output_type = f'tuple(len={len(output)})'
            else:
                output_tensor = output
                output_type = 'tensor'

            # Получаем входной тензор
            input_tensor = None
            if input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    input_tensor = input[0].detach()
                elif isinstance(input[0], tuple):
                    # Если вход тоже кортеж, берем первый элемент
                    input_tensor = input[0][0].detach() if len(input[0]) > 0 else None

            activations[name] = {
                'activation': output_tensor.detach() if output_tensor is not None else None,
                'input': input_tensor,
                'module': module,
                'output_type': output_type
            }
        return hook

    # Регистрируем хуки для ВСЕХ дочерних модулей
    for name, module in model.named_modules():
        # Пропускаем корневой модуль
        if name != '':
            # Не регистрируем для некоторых сложных модулей, которые возвращают кортежи
            # Это нужно, чтобы избежать ошибок с .detach()
            if not any(excluded in name for excluded in ['GMQA', 'generate']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
                print(f"Зарегистрирован хук для: {name} ({module.__class__.__name__})")

    # Прямой проход
    model.eval()
    with torch.no_grad():
        # Проверяем, возвращает ли модель кортеж
        result = model(x)
        if isinstance(result, tuple):
            output, present_kv = result
        else:
            output = result
            present_kv = None

    # Выводим информацию
    print("=" * 80)
    print(f"Входные данные: {x.shape}")
    print("=" * 80)

    # Проходим по всем модулям
    for name, module in model.named_modules():
        # Пропускаем корневой модуль
        if name == '':
            continue

        print(f"\nСлой: {name} ({module.__class__.__name__}):")
        print("-" * 40)

        # Информация о параметрах слоя
        if hasattr(module, 'named_parameters'):
            for param_name, param in module.named_parameters(recurse=False):
                print(f"  {param_name}: {param.shape}")
                print(f"    Значения: mean={param.mean().item():.6f}, "
                      f"std={param.std().item():.6f}, "
                      f"range=[{param.min().item():.6f}, {param.max().item():.6f}]")

        # Информация об активациях
        if name in activations:
            act_data = activations[name]
            if act_data['activation'] is not None:
                act = act_data['activation']
                print(f"  Активация: {act.shape} (тип: {act_data['output_type']})")
                print(f"    Статистика: mean={act.mean().item():.6f}, "
                      f"std={act.std().item():.6f}, "
                      f"range=[{act.min().item():.6f}, {act.max().item():.6f}]")

                # Для линейных слоев показываем пример вычислений
                if isinstance(module, nn.Linear) and act_data['input'] is not None:
                    input_data = act_data['input']
                    if len(input_data.shape) > 1 and len(module.weight.shape) > 1:
                        print(f"    Пример вычисления (1 нейрон):")
                        try:
                            bias_val = module.bias[0].item() if module.bias is not None else 0
                            act_val = act[0][0].item() if len(act.shape) > 1 else act[0].item()
                            print(f"      input * weights[0] + bias[0] = ... ≈ {act_val:.4f}")
                        except:
                            pass
            else:
                print(f"  Активация: None (тип: {act_data['output_type']})")

    print(f"\nФинальный выход: {output.shape}")
    if present_kv is not None:
        print(f"Present KV: {present_kv.shape if hasattr(present_kv, 'shape') else type(present_kv)}")

    # Удаляем хуки
    for hook in hooks:
        hook.remove()

    return output

# Функция-хук ТОЛЬКО для линейных слоев
def get_model_linear_activation(name):
    def hook(model, input, output):
        # Сохраняем данные только если это Linear слой
        if isinstance(model, nn.Linear):
            # Используем глобальный словарь
            if 'linear_activations' not in globals():
                globals()['linear_activations'] = {}

            # Получаем входной тензор
            input_tensor = None
            if input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    input_tensor = input[0].detach()
                elif isinstance(input[0], tuple):
                    input_tensor = input[0][0].detach() if len(input[0]) > 0 else None

            globals()['linear_activations'][name] = {
                'layer': model,
                'activation': output.detach(),  # Linear слои всегда возвращают тензор
                'weights': model.weight.detach(),
                'bias': model.bias.detach() if model.bias is not None else None,
                'input': input_tensor
            }
    return hook


# ФУНКЦИЯ ДЛЯ КВАНТОВАНИЯ ТОЛЬКО LINEAR СЛОЕВ
def quantize_model_linear_layers(model, dtype=torch.float32):
    """
    Квантует только Linear слои в модели

    Args:
        model: nn.Module модель
        dtype: тип данных для де-квантованных весов
    """
    quant_params = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nКвантование Linear слоя: {name}")

            # Получаем веса
            w = module.weight.data

            # Вычисляем параметры квантования
            if w.numel() > 0:
                scale = (w.max() - w.min()) / 255.0
                zero_point = torch.round(-w.min() / scale)

                # Квантуем
                quant_w = torch.round(w / scale + zero_point)
                quant_w = torch.clamp(quant_w, 0, 255).to(torch.uint8)

                # Де-квантуем
                dequant_w = (quant_w.float() - zero_point) * scale

                # Обновляем веса
                with torch.no_grad():
                    module.weight.data = dequant_w.to(dtype)

                # Сохраняем параметры
                quant_params[name] = {
                    'scale': scale,
                    'zero_point': zero_point,
                    'quantized_weights': quant_w.cpu().numpy(),
                    'original_shape': w.shape
                }

                print(f"  Scale: {scale.item():.6f}")
                print(f"  Zero point: {zero_point.item():.2f}")

    return quant_params

# ФУНКЦИЯ ДЛЯ КВАНТОВАНИЯ МОДЕЛИ С ПОЛНЫМ ПРОЦЕССОМ (УПРОЩЕННАЯ)
def quantize_model_simple(model, x, vocab_size=32000):
    """
    Упрощенный процесс квантования модели - использует только quantize_linear_layers

    Args:
        model: модель для квантования
        x: тестовый вход
        vocab_size: размер словаря
    """
    print("=" * 80)
    print("УПРОЩЕННЫЙ ПРОЦЕСС КВАНТОВАНИЯ МОДЕЛИ")
    print("=" * 80)

    # 1. Печатаем информацию о модели
    print("\n1. ИНФОРМАЦИЯ О МОДЕЛИ:")
    with torch.no_grad():
        print_info_model(model, x)

    # 2. Квантуем линейные слои
    print("\n" + "=" * 80)
    print("2. КВАНТОВАНИЕ LINEAR СЛОЕВ")
    print("=" * 80)

    quant_params = quantize_linear_layers(model)

    # 3. Проверяем работу
    print("\n" + "=" * 80)
    print("3. ПРОВЕРКА РАБОТЫ")
    print("=" * 80)

    with torch.no_grad():
        print("Квантованная модель:")
        result = model(x)
        if isinstance(result, tuple):
            output, _ = result
        else:
            output = result
        print(f"Выход shape: {output.shape}")

        # Тестируем генерацию
        print("\nТестируем генерацию...")
        test_input = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(test_input, max_new_tokens=3)
        print(f"Результат генерации: {generated.shape}")

    print("\n" + "=" * 80)
    print("КВАНТОВАНИЕ ЗАВЕРШЕНО!")
    print(f"Квантовано слоев: {len(quant_params)}")
    print("=" * 80)

    return model, quant_params

# ФУНКЦИЯ ДЛЯ КВАНТОВАНИЯ ВЕСОВ ОТДЕЛЬНОГО СЛОЯ
def quantize_model_layer_weights(layer):
    """
    Квантует веса одного слоя и возвращает параметры
    """
    if not isinstance(layer, nn.Linear):
        print(f"Ошибка: слой {type(layer)} не является Linear слоем")
        return None

    w = layer.weight.data.clone()

    if w.numel() == 0:
        print("Ошибка: веса пустые")
        return None

    # Вычисляем параметры квантования
    scale = (w.max() - w.min()) / 255.0
    zero_point = torch.round(-w.min() / scale)

    # Квантуем
    quant_w = torch.round(w / scale + zero_point)
    quant_w = torch.clamp(quant_w, 0, 255).to(torch.uint8)

    # Де-квантуем
    dequant_w = (quant_w.float() - zero_point) * scale

    return {
        'original': w,
        'quantized': quant_w,
        'dequantized': dequant_w,
        'scale': scale,
        'zero_point': zero_point,
        'error': torch.mean(torch.abs(w - dequant_w))
    }

# ОСНОВНАЯ ФУНКЦИЯ ДЛЯ РАБОТЫ С МОДЕЛЬЮ LLAMA2
def quantize_language_model(model, vocab_size=32000, test_seq_len=10):
    """
    Основная функция для квантования модели llama2

    Args:
        model: модель llama2
        vocab_size: размер словаря
        test_seq_len: длина тестовой последовательности
    """

    x = torch.randint(0, vocab_size, (1, test_seq_len))
    print(f"Тестовый вход: {x.shape}")



    print("ВАРИАНТ 1: ПРОСТОЙ СПОСОБ")
    print("=" * 50)

    quantized_model, quant_params = quantize_model_simple(model, x, vocab_size)


    print("\n" + "=" * 50)
    print("ВАРИАНТ 2: РУЧНОЕ КВАНТОВАНИЕ")
    print("=" * 50)


    print("\nКвантование out_proj слоя:")
    if hasattr(model, 'out_proj') and isinstance(model.out_proj, nn.Linear):
        result = quantize_model_layer_weights(model.out_proj)
        if result:
            print(f"  Scale: {result['scale'].item():.6f}")
            print(f"  Zero point: {result['zero_point'].item():.2f}")
            print(f"  Ошибка: {result['error'].item():.6f}")

            # Обновляем веса
            with torch.no_grad():
                model.out_proj.weight.data = result['dequantized']


    print("\nПоиск других Linear слоев:")
    linear_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
            print(f"  Найден Linear слой: {name}")
            print(f"    Форма весов: {module.weight.shape}")

            # Квантуем
            result = quantize_model_layer_weights(module)
            if result:
                print(f"    Scale: {result['scale'].item():.6f}")
                print(f"    Zero point: {result['zero_point'].item():.2f}")

                # Обновляем веса
                with torch.no_grad():
                    module.weight.data = result['dequantized']

    print(f"\nВсего найдено Linear слоев: {linear_count}")

    # Проверяем конечный результат
    print("\n" + "=" * 50)
    print("ФИНАЛЬНАЯ ПРОВЕРКА")
    print("=" * 50)

    with torch.no_grad():
        # Проверяем прямой проход
        logits, present_kv = model(x)
        print(f"Логиты: {logits.shape}")

        # Проверяем генерацию
        test_input = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(test_input, max_new_tokens=5)
        print(f"Генерация: {generated.shape}")
        print(f"Пример сгенерированных токенов: {generated[0, :10].tolist()}")

    return model



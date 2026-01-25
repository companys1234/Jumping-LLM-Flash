

def example_bpe():
    """Пример использования BPE токенизатора"""

    # Подготовка данных для обучения
    training_texts = [
        "Hello world! This is a test.",
        "Привет мир! Это тест.",
        "自然语言处理是人工智能的重要领域。",
        "I love machine learning and deep learning.",
        "SentencePiece is a great tokenization tool."
    ]

    # Создаём и обучаем токенизатор
    sp_bpe = SentencePieceBPE(vocab_size=100)
    sp_bpe.train(training_texts, verbose=True)

    # Тестируем
    test_text = "Hello world! Привет мир! Natural language processing."

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ BPE ТОКЕНИЗАТОРА")
    print("=" * 50)

    # Токенизация
    pieces = sp_bpe.encode_as_pieces(training_texts[2])
    ids = sp_bpe.encode_as_ids(test_text)

    print(f"Исходный текст: {test_text}")
    print(f"Субтокены: {pieces}")
    print(f"ID токенов: {ids}")

    # Декодирование
    decoded = sp_bpe.decode_pieces(pieces)
    print(f"Декодированный текст: {decoded}")

    # Сохранение и загрузка
    sp_bpe.save("sp_bpe_model.json")

    # Загружаем модель
    sp_bpe2 = SentencePieceBPE()
    sp_bpe2.load("sp_bpe_model.json")

    # Проверяем, что работает так же
    pieces2 = sp_bpe2.encode_as_pieces(test_text)
    print(f"\nЗагруженная модель даёт те же субтокены: {pieces == pieces2}")

    return sp_bpe


def example_unigram():
    """Пример использования Unigram токенизатора"""

    training_texts = [
        "Hello world!",
        "Привет мир!",
        "Test sentence."
    ]

    sp_uni = SentencePieceUnigram(vocab_size=50)
    sp_uni.train(training_texts)

    test_text = "Hello world test"
    pieces = sp_uni.encode_as_pieces(test_text)

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ UNIGRAM ТОКЕНИЗАТОРА")
    print("=" * 50)
    print(f"Текст: {test_text}")
    print(f"Субтокены: {pieces}")

    return sp_uni

def compare_with_original():
    """Сравнение с оригинальной библиотекой (если установлена)"""
    try:
        import sentencepiece as sp
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ BIBLIOTEKOЙ")
        print("=" * 50)

        # Создаём тестовые данные
        with open('test_corpus.txt', 'w', encoding='utf-8') as f:
            f.write("Hello world!\n")
            f.write("Привет мир!\n")
            f.write("This is a test.\n")

        # Обучаем оригинальный SentencePiece
        sp.SentencePieceTrainer.train(
            input='test_corpus.txt',
            model_prefix='sp_original',
            vocab_size=50,
            model_type='bpe'
        )

        # Загружаем модель
        sp_processor = sp.SentencePieceProcessor()
        sp_processor.load('sp_original.model')

        # Токенизируем
        test_text = "Hello world test"
        original_pieces = sp_processor.encode_as_pieces(test_text)

        print(f"Оригинальная библиотека: {original_pieces}")

    except ImportError:
        print("Библиотека sentencepiece не установлена. Установите: pip install sentencepiece")





def real_world_example():
    """Пример более реалистичного использования"""

    # Чтение данных из файла (если есть)
    try:
        with open('sample_text.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # Или используем демо данные
        texts = [
            "Машинное обучение — это подраздел искусственного интеллекта.",
            "Глубокое обучение использует нейронные сети с множеством слоёв.",
            "Natural Language Processing (NLP) enables computers to understand human language.",
            "Токенизация — важный этап предобработки текста в NLP.",
            "SentencePiece обеспечивает субсловную токенизацию для многих языков."
        ]

    # Обучаем модель
    print("\n" + "=" * 50)
    print("РЕАЛЬНЫЙ ПРИМЕР С РУССКИМ И АНГЛИЙСКИМ")
    print("=" * 50)

    sp_model = SentencePieceBPE(vocab_size=150)
    sp_model.train(texts, verbose=True)

    # Тестируем на смешанном тексте
    test_cases = [
        "Машинное обучение и Natural Language Processing",
        "Нейронные сети для обработки текста",
        "Deep learning models require large amounts of data."
    ]

    for i, test_text in enumerate(test_cases):
        print(f"\nТест {i + 1}: {test_text}")
        pieces = sp_model.encode_as_pieces(test_text)
        print(f"Токены: {pieces}")
        decoded = sp_model.decode_pieces(pieces)
        print(f"Декодировано: {decoded}")




# Запускаем примеры
example_bpe()
example_unigram()

# Показываем реальный пример
real_world_example()


#compare_with_original()
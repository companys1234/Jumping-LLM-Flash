
import re
from collections import defaultdict, Counter
import pytest
class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.bpe_vocab = {}

    def get_stats(self, corpus):
        """Подсчитать частоты пар символов"""
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        """Слить самую частую пару в новую подстроку"""
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        new_corpus = {}
        for word in corpus:
            new_word = re.sub(pattern, replacement, word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def fit(self, words):
        """Обучение BPE кодов"""
        corpus = Counter([' '.join(list(w)) + ' ' for w in words])  # слово как символы + end token
        self.bpe_vocab = dict(corpus)

        for i in range(self.vocab_size):
            pairs = self.get_stats(self.bpe_vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            self.bpe_vocab = self.merge_vocab(best, self.bpe_vocab)

    def encode_word(self, word):
        """Разбить слово на BPE-токены"""
        word = list(word) + ['']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            if not pairs:
                break

            mergeable = {pair: self.bpe_codes[pair] for pair in pairs if pair in self.bpe_codes}
            if not mergeable:
                break

            best_pair = min(mergeable, key=mergeable.get)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text):
        """Токенизировать весь текст"""
        return [self.encode_word(word) for word in text.split()]

# ПРОВЕРКА ФУНКЦИЙ
corpus = ['i', 'cannot', 'see', 'where', 'the', 'green', 'arrow', 'is', 'pointing', '.', 'a', 'woman', 'is', 'pointing', 'at', 'the', 'camera', 'while', 'lying', 'down', 'a', 'finger', 'pointing', 'at', 'a', 'hotdog', 'with', 'cheese', ',', 'sauerkraut', 'and', 'ketchup', '.', 'a', 'very', 'crowded', 'street', 'sign', 'pointing', 'to', 'attractions', 'in', 'multiple', 'direction']
bpe = BPE(vocab_size = 100)
bpe.fit(words=corpus)
corpus = {
    "l o w </w>": 5,        # слово "low" встречается 5 раз
    "l o w e r </w>": 2,    # слово "lower" встречается 2 раза
    "n e w e s t </w>": 6,  # слово "newest" встречается 6 раз
    "w i d e r </w>": 3,    # слово "wider" встречается 3 раза
}
text = 'cannot see'
print(bpe.get_stats(corpus))
tokenized = bpe.encode(text)
print(tokenized)
for word, tokens in zip(text.split(), tokenized):
    print(f"{word} -> {tokens}")

# ТЕСТЫ

@pytest.fixture
def bpe():
    """Фикстура для создания экземпляра BPE"""
    return BPE(vocab_size=10)


@pytest.fixture
def sample_words():
    """Фикстура с примерными словами для обучения"""
    return ["hello", "world", "hell", "test", "testing", "hello_world"]


@pytest.fixture
def trained_bpe(bpe, sample_words):
    """Фикстура с предобученным BPE"""
    bpe.fit(sample_words)
    return bpe


def test_init(bpe):
    """Тест инициализации класса"""
    assert bpe.vocab_size == 10
    assert bpe.bpe_codes == {}
    assert bpe.bpe_vocab == {}


def test_get_stats():
    """Тест подсчета статистики пар"""
    bpe = BPE(vocab_size=10)
    corpus = {
        'h e l l o </w>': 3,
        'w o r l d </w>': 2
    }
    stats = bpe.get_stats(corpus)

    # Проверяем наличие пар
    assert ('h', 'e') in stats
    assert ('e', 'l') in stats
    assert ('l', 'l') in stats
    assert ('l', 'o') in stats
    assert ('o', '</w>') in stats

    # Проверяем частоты
    assert stats[('h', 'e')] == 3
    assert stats[('w', 'o')] == 2


def test_merge_vocab():
    """Тест слияния пар в словаре"""
    bpe = BPE(vocab_size=10)
    corpus = {
        'h e l l o </w>': 3,
        'h e l p </w>': 2
    }

    # Сливаем пару ('l', 'l')
    new_corpus = bpe.merge_vocab(('l', 'l'), corpus)

    # Проверяем, что пара слилась
    assert 'h e ll o </w>' in new_corpus
    assert 'h e l p </w>' in new_corpus  # эта пара не должна измениться
    assert new_corpus['h e ll o </w>'] == 3


def test_fit(trained_bpe):
    """Тест обучения BPE"""
    assert len(trained_bpe.bpe_codes) <= 10  # не больше vocab_size
    assert len(trained_bpe.bpe_vocab) > 0

    # Проверяем, что коды уникальны
    code_values = list(trained_bpe.bpe_codes.values())
    assert len(set(code_values)) == len(code_values)  # все значения уникальны


def test_fit_with_empty_vocab():
    """Тест обучения с пустым словарем"""
    bpe = BPE(vocab_size=10)
    bpe.fit([])
    assert bpe.bpe_codes == {}
    assert bpe.bpe_vocab == {}


def test_fit_small_vocab_size():
    """Тест обучения с vocab_size=1"""
    bpe = BPE(vocab_size=1)
    words = ["hello", "world"]
    bpe.fit(words)
    assert len(bpe.bpe_codes) <= 1


def test_encode_word_basic(trained_bpe):
    """Тест кодирования одного слова"""
    # Простое слово
    result = trained_bpe.encode_word("hello")
    assert isinstance(result, list)
    assert len(result) > 0

    # Проверяем, что все токены - строки
    assert all(isinstance(token, str) for token in result)


def test_encode_word_with_unknown_chars(trained_bpe):
    """Тест кодирования слова с неизвестными символами"""
    # Слово с символами, которых не было в обучении
    result = trained_bpe.encode_word("xyz")
    assert isinstance(result, list)
    assert len(result) > 0
    # Должен вернуть исходные символы
    assert ''.join(result).replace('</w>', '') == 'xyz'


def test_encode_word_empty(trained_bpe):
    """Тест кодирования пустой строки"""
    result = trained_bpe.encode_word("")
    assert result == ['</w>'] or result == ['']


def test_encode_text(trained_bpe):
    """Тест кодирования текста"""
    text = "hello world test"
    result = trained_bpe.encode(text)

    assert isinstance(result, list)
    assert len(result) == 3  # 3 слова

    for word_tokens in result:
        assert isinstance(word_tokens, list)
        assert len(word_tokens) > 0


def test_encode_empty_text(trained_bpe):
    """Тест кодирования пустого текста"""
    result = trained_bpe.encode("")
    assert result == []


def test_encode_text_with_multiple_spaces(trained_bpe):
    """Тест кодирования текста с несколькими пробелами"""
    text = "hello  world   test"
    result = trained_bpe.encode(text)
    assert len(result) == 3  # все равно 3 слова


def test_consistency_of_encoding(trained_bpe):
    """Тест консистентности кодирования"""
    word = "testing"
    encoding1 = trained_bpe.encode_word(word)
    encoding2 = trained_bpe.encode_word(word)
    assert encoding1 == encoding2  # одинаковые результаты


def test_bpe_codes_ordering():
    """Тест порядка добавления BPE кодов"""
    bpe = BPE(vocab_size=5)
    words = ["aaaa", "aaab", "aabb", "abbb"]
    bpe.fit(words)

    # Проверяем, что коды присваиваются в порядке слияния
    codes = list(bpe.bpe_codes.values())
    assert codes == list(range(len(codes)))  # 0, 1, 2, ...


def test_special_characters():
    """Тест обработки специальных символов"""
    bpe = BPE(vocab_size=10)
    words = ["test-ing", "user@mail", "file.txt"]
    bpe.fit(words)

    # Проверяем кодирование слова со специальными символами
    result = bpe.encode_word("test-ing")
    assert isinstance(result, list)
    assert len(result) > 0


def test_unicode_characters():
    """Тест обработки юникод символов"""
    bpe = BPE(vocab_size=10)
    words = ["привет", "мир", "тест"]
    bpe.fit(words)

    result = bpe.encode_word("приветмир")
    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)


def test_large_vocab_size():
    """Тест с большим vocab_size"""
    bpe = BPE(vocab_size=100)
    words = ["hello"] * 5 + ["world"] * 3 + ["test"] * 2
    bpe.fit(words)

    assert len(bpe.bpe_codes) <= 100
    # При большом vocab_size должно быть много слияний
    assert len(bpe.bpe_codes) > 0


def test_word_frequencies_preserved():
    """Тест сохранения частот слов"""
    bpe = BPE(vocab_size=5)
    words = ["hello", "hello", "world", "hello", "test"]
    bpe.fit(words)

    # После обучения bpe_vocab должен содержать частоты
    total_freq = sum(bpe.bpe_vocab.values())
    assert total_freq == len(words)


# Тесты для граничных случаев
def test_single_char_words():
    """Тест обработки слов из одного символа"""
    bpe = BPE(vocab_size=10)
    words = ["a", "b", "c", "aa", "bb"]
    bpe.fit(words)

    result = bpe.encode_word("a")
    assert result == ['a', '</w>'] or result == ['a', '']


def test_repeated_chars():
    """Тест обработки повторяющихся символов"""
    bpe = BPE(vocab_size=5)
    words = ["aaaa", "aaa", "aa", "a"]
    bpe.fit(words)

    # Проверяем, что происходит слияние повторяющихся символов
    result = bpe.encode_word("aaaa")
    assert len(result) < 5  # должно быть меньше 5 токенов (4 буквы + </w>)


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
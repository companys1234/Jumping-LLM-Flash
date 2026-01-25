import re
import collections
import unicodedata
from typing import List, Dict, Tuple, Set

class SentencePieceBPE:
    def __init__(self, vocab_size: int = 10000, character_coverage: float = 1.0):
        """
        Args:
            vocab_size: Желаемый размер словаря
            character_coverage: Доля символов для покрытия (0.9995 для мультиязычных)
        """
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.vocab = {}  # токен -> id
        self.reverse_vocab = {}  # id -> токен
        self.merges = {}  # пара токенов -> объединённый токен
        self.special_tokens = {}

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста (упрощённая)"""
        # NFKC нормализация Unicode
        text = unicodedata.normalize('NFKC', text)
        # Заменяем пробелы на специальный символ
        text = text.replace(' ', '▁')
        # Добавляем пробел в начало, если его нет
        if not text.startswith('▁'):
            text = '▁' + text
        return text

    def _denormalize_text(self, text: str) -> str:
        """Восстановление оригинального текста"""
        text = text.replace('▁', ' ')
        return text.strip()

    def _initialize_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Инициализация словаря уникальными символами"""
        chars = set()
        for text in texts:
            normalized = self._normalize_text(text)
            chars.update(normalized)

        # Сортируем символы для детерминированности
        vocab = {chr(i): i for i in range(256)}  # Базовые байты

        # Добавляем специальные токены
        special_tokens = ['<unk>', '<s>', '</s>', '<pad>']
        for i, token in enumerate(special_tokens):
            vocab[token] = len(vocab)
            self.special_tokens[token] = len(vocab) - 1

        return vocab

    def _get_stats(self, tokenized_texts: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Подсчёт частот пар токенов"""
        pairs = collections.defaultdict(int)
        for tokens in tokenized_texts:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokenized_texts: List[List[str]], pair: Tuple[str, str], new_token: str):
        """Объединение пары токенов во всех текстах"""
        new_tokenized = []
        for tokens in tokenized_texts:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokenized.append(new_tokens)
        return new_tokenized

    def train(self, texts: List[str], verbose: bool = False):
        """
        Обучение BPE токенизатора

        Args:
            texts: Список текстов для обучения
            verbose: Вывод информации о процессе
        """
        if verbose:
            print(f"Обучение на {len(texts)} текстах...")

        # Нормализуем все тексты
        normalized_texts = [self._normalize_text(text) for text in texts]

        # Инициализируем словарь символами
        self.vocab = self._initialize_vocab(texts)

        # Токенизируем на символы
        tokenized_texts = [list(text) for text in normalized_texts]

        # Основной цикл BPE
        num_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"Будет выполнено {num_merges} слияний...")

        for i in range(num_merges):
            # Получаем статистику пар
            stats = self._get_stats(tokenized_texts)

            if not stats:
                if verbose:
                    print(f"Нет пар для слияния на шаге {i}")
                break

            # Находим наиболее частую пару
            best_pair = max(stats.items(), key=lambda x: x[1])[0]
            best_freq = stats[best_pair]

            # Создаём новый токен
            new_token = best_pair[0] + best_pair[1]
            token_id = len(self.vocab)

            # Добавляем в словарь
            self.vocab[new_token] = token_id
            self.merges[best_pair] = new_token

            if verbose and i % 100 == 0:
                print(f"Шаг {i}: пара {best_pair} -> {new_token} (частота: {best_freq})")

            # Обновляем все тексты
            tokenized_texts = self._merge_pair(tokenized_texts, best_pair, new_token)

        # Создаём обратный словарь
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Обучение завершено. Размер словаря: {len(self.vocab)}")

    def encode_as_pieces(self, text: str) -> List[str]:
        """Токенизация текста на субтокены"""
        # Нормализуем текст
        normalized = self._normalize_text(text)

        # Начинаем с символов
        tokens = list(normalized)

        # Применяем все слияния в порядке их обучения
        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merges:
                        new_tokens.append(self.merges[pair])
                        i += 2
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode_as_ids(self, text: str) -> List[int]:
        """Токенизация текста в идентификаторы"""
        pieces = self.encode_as_pieces(text)
        ids = []
        for piece in pieces:
            if piece in self.vocab:
                ids.append(self.vocab[piece])
            else:
                # Разбиваем на байты для неизвестных токенов
                for byte in piece.encode('utf-8'):
                    byte_char = chr(byte)
                    ids.append(self.vocab.get(byte_char, self.special_tokens.get('<unk>', 0)))
        return ids

    def decode_pieces(self, pieces: List[str]) -> str:
        """Декодирование субтокенов обратно в текст"""
        text = ''.join(pieces)
        return self._denormalize_text(text)

    def decode_ids(self, ids: List[int]) -> str:
        """Декодирование идентификаторов обратно в текст"""
        pieces = []
        for token_id in ids:
            if token_id in self.reverse_vocab:
                pieces.append(self.reverse_vocab[token_id])
            else:
                pieces.append('<unk>')
        return self.decode_pieces(pieces)

    def save(self, filepath: str):
        """Сохранение модели"""
        import json
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Загрузка модели"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab_size = data['vocab_size']
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.merges = {}
        for k, v in data['merges'].items():
            a, b = k.split(',')
            self.merges[(a, b)] = v
        self.special_tokens = data['special_tokens']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}


class SentencePieceUnigram:
    """
    Упрощённая реализация SentencePiece с Unigram Language Model
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.scores = {}  # Логарифмы вероятностей токенов

    def train(self, texts: List[str], num_iterations: int = 10):
        """Обучение Unigram модели"""
        # Инициализация: каждый символ как токен
        all_chars = set()
        for text in texts:
            all_chars.update(text.replace(' ', '▁'))

        # Начальный словарь (все символы)
        self.vocab = {char: i for i, char in enumerate(all_chars)}

        # Добавляем специальные токены
        specials = ['<unk>', '<s>', '</s>', '<pad>']
        for token in specials:
            self.vocab[token] = len(self.vocab)

        # Инициализируем вероятности равномерно
        for token in self.vocab:
            self.scores[token] = 1.0

        # Упрощённый EM-алгоритм
        for iteration in range(num_iterations):
            # E-шаг: собираем статистику токенов
            token_counts = collections.defaultdict(int)
            total_tokens = 0

            for text in texts:
                normalized = text.replace(' ', '▁')
                # Простая сегментация по символам (упрощённо)
                for char in normalized:
                    token_counts[char] += 1
                    total_tokens += 1

            # M-шаг: обновляем вероятности
            for token in self.vocab:
                count = token_counts.get(token, 0)
                self.scores[token] = count / total_tokens if total_tokens > 0 else 1e-10

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode_as_pieces(self, text: str) -> List[str]:
        """Токенизация с помощью Viterbi алгоритма (упрощённо)"""
        normalized = text.replace(' ', '▁')

        # Динамическое программирование для поиска лучшей сегментации
        n = len(normalized)
        dp = [-float('inf')] * (n + 1)
        dp[0] = 0
        prev = [-1] * (n + 1)

        for i in range(n + 1):
            for j in range(i):
                token = normalized[j:i]
                if token in self.scores:
                    score = dp[j] + self.scores[token]
                    if score > dp[i]:
                        dp[i] = score
                        prev[i] = j

        # Восстанавливаем сегментацию
        pieces = []
        i = n
        while i > 0:
            j = prev[i]
            pieces.append(normalized[j:i])
            i = j

        return list(reversed(pieces))


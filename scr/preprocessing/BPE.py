import re
from collections import defaultdict, Counter
from typing import List, Union

class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.bpe_vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        self.space_token = '<SPACE>'  # Токен для пробелов
        self._build_special_tokens()

    def _build_special_tokens(self):
        """Создать специальные токены"""
        self.special_tokens = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
            self.space_token: 4  # Токен для пробела между словами
        }
        
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
        corpus = Counter([' '.join(list(w)) + ' ' for w in words])
        self.bpe_vocab = dict(corpus)
        
        # Инициализируем словари токенов
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Собираем базовые символы
        base_tokens = set()
        for word in corpus:
            base_tokens.update(word.split())
        
        # Добавляем базовые символы в словарь
        next_id = len(self.token_to_id)
        for token in sorted(base_tokens):
            if token not in self.token_to_id and token != '':
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        for i in range(self.vocab_size):
            pairs = self.get_stats(self.bpe_vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            
            # Добавляем объединенный токен в словарь
            merged_token = ''.join(best)
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.id_to_token[next_id] = merged_token
                next_id += 1
                
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
        
        # Фильтруем пустые токены
        return [token for token in word if token != '']

    def encode(self, text, return_ids=False, add_special_tokens=False):
        """
        Токенизировать текст
        
        Args:
            text: текст для токенизации
            return_ids: если True, возвращает числовые ID токенов
            add_special_tokens: если True, добавляет специальные токены
        Returns:
            Список токенов или их ID
        """
        words = text.split()
        all_tokens = []
        
        for i, word in enumerate(words):
            word_tokens = self.encode_word(word)
            all_tokens.extend(word_tokens)
            
            # Добавляем токен пробела между словами (кроме последнего)
            if i < len(words) - 1:
                all_tokens.append(self.space_token)
        
        if add_special_tokens:
            # Добавляем начало и конец последовательности
            all_tokens = [self.bos_token] + all_tokens + [self.eos_token]
            
            # Добавляем специальные токены в словарь, если их там нет
            for token in [self.bos_token, self.eos_token]:
                if token not in self.token_to_id:
                    new_id = len(self.token_to_id)
                    self.token_to_id[token] = new_id
                    self.id_to_token[new_id] = token
        
        if return_ids:
            return self.tokens_to_ids(all_tokens)
        return all_tokens

    def tokens_to_ids(self, tokens):
        """Преобразовать токены в числовые ID"""
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Для неизвестных токенов используем UNK
                ids.append(self.token_to_id[self.unk_token])
        return ids

    def ids_to_tokens(self, ids):
        """Преобразовать числовые ID обратно в токены"""
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
            else:
                tokens.append(self.unk_token)
        return tokens

    def decode(self, tokens_or_ids, from_ids=False):
        """
        Декодировать токены обратно в текст
        
        Args:
            tokens_or_ids: список токенов или их ID
            from_ids: если True, входные данные - числовые ID
        Returns:
            Декодированный текст
        """
        if from_ids:
            tokens = self.ids_to_tokens(tokens_or_ids)
        else:
            tokens = tokens_or_ids
        
        # Инициализируем список для слов
        words = []
        current_word = []
        
        for token in tokens:
            # Пропускаем специальные токены, кроме пробела
            if token in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                continue
            
            # Если встретили токен пробела - сохраняем текущее слово и начинаем новое
            elif token == self.space_token:
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
            
            # Иначе добавляем токен к текущему слову
            else:
                current_word.append(token)
        
        # Добавляем последнее слово, если оно есть
        if current_word:
            words.append(''.join(current_word))
        
        # Объединяем слова с пробелами
        return ' '.join(words)

    def decode_word(self, tokens_or_ids, from_ids=False):
        """
        Декодировать токены одного слова
        """
        if from_ids:
            tokens = self.ids_to_tokens(tokens_or_ids)
        else:
            tokens = tokens_or_ids
        
        # Фильтруем специальные токены и объединяем
        word_tokens = []
        for token in tokens:
            if token not in [self.unk_token, self.pad_token, self.bos_token, 
                           self.eos_token, self.space_token]:
                word_tokens.append(token)
        
        return ''.join(word_tokens)

    def encode_batch(self, texts, return_ids=True, add_special_tokens=False, max_length=None, padding=False):
        """
        Токенизировать пакет текстов
        
        Args:
            texts: список текстов
            return_ids: возвращать числовые ID
            add_special_tokens: добавлять специальные токены
            max_length: максимальная длина последовательности
            padding: дополнить последовательности до max_length
        Returns:
            Список списков токенов/ID
        """
        encoded_batch = []
        
        for text in texts:
            encoded = self.encode(text, return_ids=return_ids, add_special_tokens=add_special_tokens)
            encoded_batch.append(encoded)
        
        if padding and max_length:
            padded_batch = []
            pad_id = self.token_to_id[self.pad_token] if return_ids else self.pad_token
            
            for encoded in encoded_batch:
                if len(encoded) < max_length:
                    # Добавляем паддинг
                    padded = encoded + [pad_id] * (max_length - len(encoded))
                else:
                    # Обрезаем до максимальной длины
                    padded = encoded[:max_length]
                padded_batch.append(padded)
            
            return padded_batch
        
        return encoded_batch

    def get_vocab_size(self):
        """Получить размер словаря"""
        return len(self.token_to_id)



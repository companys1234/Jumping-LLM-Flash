
import re
from collections import defaultdict, Counter

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


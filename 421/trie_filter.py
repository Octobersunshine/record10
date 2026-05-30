from typing import Optional

from pypinyin import lazy_pinyin


LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
}


def _normalize_char(ch: str) -> str:
    code = ord(ch)
    if 0xFF01 <= code <= 0xFF5E:
        ch = chr(code - 0xFEE0)
    elif code == 0x3000:
        ch = " "
    ch = ch.lower()
    if ch in LEET_MAP:
        ch = LEET_MAP[ch]
    return ch


def _normalize(text: str) -> str:
    return "".join(_normalize_char(ch) for ch in text)


def _pinyin_first_letters(word: str) -> str:
    return "".join(p[0] for p in lazy_pinyin(word))


def _pinyin_full(word: str) -> str:
    return "".join(lazy_pinyin(word))


class TrieNode:
    __slots__ = ("children", "is_end", "word", "match_type")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.word: str = ""
        self.match_type: str = ""


class HitPosition:
    __slots__ = ("word", "start", "end", "match_type", "matched_text")

    def __init__(self, word: str, start: int, end: int, match_type: str, matched_text: str):
        self.word = word
        self.start = start
        self.end = end
        self.match_type = match_type
        self.matched_text = matched_text

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "match_type": self.match_type,
            "matched_text": self.matched_text,
        }


class SensitiveWordFilter:
    def __init__(self):
        self.root = TrieNode()
        self._word_count = 0

    @property
    def word_count(self) -> int:
        return self._word_count

    def _insert(self, text: str, original_word: str, match_type: str) -> None:
        if not text:
            return
        node = self.root
        for ch in text:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            node.is_end = True
            node.word = original_word
            node.match_type = match_type

    def add_word(self, word: str) -> None:
        original = word.strip()
        if not original:
            return
        normalized = _normalize(original)
        self._insert(normalized, original, "exact")
        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in original)
        if has_chinese:
            pinyin_first = _normalize(_pinyin_first_letters(original))
            self._insert(pinyin_first, original, "pinyin_first")
            pinyin_full = _normalize(_pinyin_full(original))
            self._insert(pinyin_full, original, "pinyin_full")
        self._word_count += 1

    def load_from_file(self, filepath: str, encoding: str = "utf-8") -> None:
        with open(filepath, "r", encoding=encoding) as f:
            for line in f:
                self.add_word(line)

    def filter_text(
        self, text: str, replace_char: str = "*"
    ) -> tuple[str, list[str], list[HitPosition]]:
        if not text:
            return "", [], []

        chars = list(text)
        normalized = _normalize(text)
        hit_words: list[str] = []
        positions: list[HitPosition] = []
        i = 0
        n = len(chars)

        while i < n:
            node = self.root
            match_end = -1
            match_word = ""
            match_type = ""
            j = i

            while j < n and normalized[j] in node.children:
                node = node.children[normalized[j]]
                j += 1
                if node.is_end:
                    match_end = j
                    match_word = node.word
                    match_type = node.match_type

            if match_end != -1:
                matched_text = "".join(chars[i:match_end])
                for k in range(i, match_end):
                    chars[k] = replace_char
                hit_words.append(match_word)
                positions.append(
                    HitPosition(
                        word=match_word,
                        start=i,
                        end=match_end,
                        match_type=match_type,
                        matched_text=matched_text,
                    )
                )
                i = match_end
            else:
                i += 1

        return "".join(chars), hit_words, positions

    def contains_sensitive(self, text: str) -> bool:
        _, hits, _ = self.filter_text(text)
        return len(hits) > 0

    def find_sensitive_words(self, text: str) -> list[str]:
        _, hits, _ = self.filter_text(text)
        return hits

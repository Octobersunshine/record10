import jieba


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class TrieAutocomplete:
    DEFAULT_LIMIT = 10
    MAX_LIMIT = 100
    MAX_FUZZY_DISTANCE = 3

    def __init__(self):
        self.root = TrieNode()
        self._all_words_cache = {}
        self._max_frequency = 1
        self._initialize_default_dictionary()

    def _initialize_default_dictionary(self):
        from lexicon import POSITIVE_WORDS, NEGATIVE_WORDS, INTENSIFIER_WORDS, NEGATION_WORDS
        from domain_lexicon import DOMAIN_LEXICON

        default_words = {}

        for word, score in POSITIVE_WORDS.items():
            default_words[word] = default_words.get(word, 0) + int(abs(score) * 100)
        for word, score in NEGATIVE_WORDS.items():
            default_words[word] = default_words.get(word, 0) + int(abs(score) * 100)
        for word, score in INTENSIFIER_WORDS.items():
            default_words[word] = default_words.get(word, 0) + int(abs(score) * 50)
        for word in NEGATION_WORDS:
            default_words[word] = default_words.get(word, 0) + 50

        for domain, lexicon in DOMAIN_LEXICON.items():
            for sentiment, words in lexicon.items():
                for word, score in words.items():
                    default_words[word] = default_words.get(word, 0) + int(abs(score) * 100)

        for word, freq in default_words.items():
            self.add_word(word, freq)

    def _update_max_frequency(self, freq):
        if freq > self._max_frequency:
            self._max_frequency = freq

    def add_word(self, word, frequency=1):
        if not word:
            return False

        original_word = word
        key = word.lower()

        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        is_new = not node.is_end_of_word
        node.is_end_of_word = True
        node.frequency += frequency
        if is_new:
            node.word = original_word

        self._all_words_cache[key] = {
            'word': node.word,
            'frequency': node.frequency,
            'key': key
        }
        self._update_max_frequency(node.frequency)
        return True

    def _find_prefix_node(self, prefix):
        if not prefix:
            return None

        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node, results):
        if node.is_end_of_word:
            results.append((node.word, node.frequency))

        for child in node.children.values():
            self._collect_words(child, results)

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_relevance_score(self, frequency, edit_distance, prefix_length, word_length):
        if self._max_frequency == 0:
            freq_score = 0
        else:
            freq_score = frequency / self._max_frequency

        max_possible_distance = max(prefix_length, word_length)
        if max_possible_distance == 0:
            distance_score = 1
        else:
            distance_score = 1 - (edit_distance / max_possible_distance)
            distance_score = max(0, distance_score)

        length_bonus = min(prefix_length / word_length, 1.0) if word_length > 0 else 0

        final_score = (0.5 * freq_score) + (0.4 * distance_score) + (0.1 * length_bonus)
        return round(final_score * 100, 2)

    def _fuzzy_match(self, query, max_distance=1):
        if max_distance < 0:
            max_distance = 0
        if max_distance > self.MAX_FUZZY_DISTANCE:
            max_distance = self.MAX_FUZZY_DISTANCE

        query_lower = query.lower()
        results = []

        for word_key, word_data in self._all_words_cache.items():
            word = word_data['word']
            freq = word_data['frequency']

            if word_key.startswith(query_lower):
                distance = 0
            else:
                distance = self._levenshtein_distance(query_lower, word_key)
                if distance > max_distance:
                    continue

            score = self._calculate_relevance_score(freq, distance, len(query_lower), len(word_key))
            results.append({
                'word': word,
                'frequency': freq,
                'edit_distance': distance,
                'relevance_score': score
            })

        return results

    def _segment_completion(self, text):
        segments = jieba.lcut(text)
        if not segments:
            return []

        last_segment = segments[-1].strip()
        if not last_segment:
            return []

        prefix_results = self.autocomplete(last_segment, include_score=True)

        for result in prefix_results:
            completed_segments = segments[:-1] + [result['word']]
            result['completed_text'] = ''.join(completed_segments)
            result['base_segment'] = last_segment

        return prefix_results

    def autocomplete(self, prefix, sort_by='relevance', limit=None, fuzzy=False, max_distance=1,
                     segment=False, include_score=False):
        if not prefix or not prefix.strip():
            return []

        if segment:
            segment_results = self._segment_completion(prefix)
            if sort_by == 'relevance':
                segment_results.sort(key=lambda x: (-x['relevance_score'], x['word']))
            elif sort_by == 'frequency':
                segment_results.sort(key=lambda x: (-x['frequency'], x['word']))
            elif sort_by == 'alphabetical':
                segment_results.sort(key=lambda x: x['word'])

            effective_limit = self.DEFAULT_LIMIT
            if limit is not None and limit > 0:
                effective_limit = min(limit, self.MAX_LIMIT)
            segment_results = segment_results[:effective_limit]

            if not include_score:
                return [{'word': r['word'], 'frequency': r['frequency']} for r in segment_results]
            return segment_results

        if fuzzy:
            fuzzy_results = self._fuzzy_match(prefix, max_distance)
            if sort_by == 'relevance':
                fuzzy_results.sort(key=lambda x: (-x['relevance_score'], x['word']))
            elif sort_by == 'frequency':
                fuzzy_results.sort(key=lambda x: (-x['frequency'], x['word']))
            elif sort_by == 'alphabetical':
                fuzzy_results.sort(key=lambda x: x['word'])

            effective_limit = self.DEFAULT_LIMIT
            if limit is not None and limit > 0:
                effective_limit = min(limit, self.MAX_LIMIT)
            fuzzy_results = fuzzy_results[:effective_limit]

            if not include_score:
                return [{'word': r['word'], 'frequency': r['frequency']} for r in fuzzy_results]
            return fuzzy_results

        results = []
        prefix_node = self._find_prefix_node(prefix)

        if prefix_node is None:
            return []

        self._collect_words(prefix_node, results)

        scored_results = []
        prefix_lower = prefix.lower()
        for word, freq in results:
            word_lower = word.lower()
            distance = 0 if word_lower.startswith(prefix_lower) else self._levenshtein_distance(prefix_lower, word_lower)
            score = self._calculate_relevance_score(freq, distance, len(prefix_lower), len(word_lower))
            scored_results.append({
                'word': word,
                'frequency': freq,
                'edit_distance': distance,
                'relevance_score': score
            })

        if sort_by == 'relevance':
            scored_results.sort(key=lambda x: (-x['relevance_score'], x['word']))
        elif sort_by == 'frequency':
            scored_results.sort(key=lambda x: (-x['frequency'], x['word']))
        elif sort_by == 'alphabetical':
            scored_results.sort(key=lambda x: x['word'])

        effective_limit = self.DEFAULT_LIMIT
        if limit is not None and limit > 0:
            effective_limit = min(limit, self.MAX_LIMIT)
        effective_limit = min(effective_limit, self.MAX_LIMIT)
        scored_results = scored_results[:effective_limit]

        if not include_score:
            return [{'word': r['word'], 'frequency': r['frequency']} for r in scored_results]
        return scored_results

    def contains(self, word):
        if not word:
            return False

        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end_of_word

    def get_frequency(self, word):
        if not word:
            return 0

        node = self.root
        for char in word.lower():
            if char not in node.children:
                return 0
            node = node.children[char]

        return node.frequency if node.is_end_of_word else 0

    def get_all_words(self):
        results = []
        self._collect_words(self.root, results)
        return [{'word': word, 'frequency': freq} for word, freq in results]

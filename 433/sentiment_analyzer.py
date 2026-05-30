import re
import jieba
from lexicon import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    NEGATION_WORDS,
    INTENSIFIER_WORDS,
    PUNCTUATIONS,
)
from domain_lexicon import (
    ASPECT_WORDS,
    DOMAIN_LEXICON,
    DOMAIN_NAMES,
)


class SentimentAnalyzer:
    def __init__(self, domain="general"):
        self.domain = domain
        self._build_user_dict()
        self.negation_window = 3
        self.intensifier_window = 2
        self.aspect_window = 5

    def set_domain(self, domain):
        if domain in DOMAIN_LEXICON:
            self.domain = domain
            self._build_user_dict()
        return self

    def get_available_domains(self):
        return list(DOMAIN_LEXICON.keys())

    def _build_user_dict(self):
        all_words = set()
        all_words.update(POSITIVE_WORDS.keys())
        all_words.update(NEGATIVE_WORDS.keys())
        all_words.update(NEGATION_WORDS)
        all_words.update(INTENSIFIER_WORDS.keys())

        if self.domain in DOMAIN_LEXICON:
            domain_lex = DOMAIN_LEXICON[self.domain]
            all_words.update(domain_lex["positive"].keys())
            all_words.update(domain_lex["negative"].keys())

        for domain_key in ASPECT_WORDS:
            for aspect_name, words in ASPECT_WORDS[domain_key].items():
                all_words.update(words)

        for word in all_words:
            jieba.add_word(word)

    def _tokenize(self, text):
        text = text.lower()
        text = self._preprocess_compound_words(text)
        tokens = jieba.lcut(text)
        tokens = self._merge_english_tokens(tokens)
        return [t.strip() for t in tokens if t.strip()]

    def _preprocess_compound_words(self, text):
        text = self._normalize_english_negations(text)
        all_special_words = set()
        all_special_words.update(POSITIVE_WORDS.keys())
        all_special_words.update(NEGATIVE_WORDS.keys())
        all_special_words.update(NEGATION_WORDS)
        all_special_words.update(INTENSIFIER_WORDS.keys())

        if self.domain in DOMAIN_LEXICON:
            domain_lex = DOMAIN_LEXICON[self.domain]
            all_special_words.update(domain_lex["positive"].keys())
            all_special_words.update(domain_lex["negative"].keys())

        for domain_key in ASPECT_WORDS:
            for aspect_name, words in ASPECT_WORDS[domain_key].items():
                all_special_words.update(words)

        sorted_words = sorted(list(all_special_words), key=len, reverse=True)

        for word in sorted_words:
            if len(word) >= 2 and word in text:
                text = text.replace(word, f" {word} ")
        return text

    def _normalize_english_negations(self, text):
        text = re.sub(r"don't", " do not ", text, flags=re.IGNORECASE)
        text = re.sub(r"didn't", " did not ", text, flags=re.IGNORECASE)
        text = re.sub(r"doesn't", " does not ", text, flags=re.IGNORECASE)
        text = re.sub(r"isn't", " is not ", text, flags=re.IGNORECASE)
        text = re.sub(r"aren't", " are not ", text, flags=re.IGNORECASE)
        text = re.sub(r"wasn't", " was not ", text, flags=re.IGNORECASE)
        text = re.sub(r"weren't", " were not ", text, flags=re.IGNORECASE)
        text = re.sub(r"haven't", " have not ", text, flags=re.IGNORECASE)
        text = re.sub(r"hasn't", " has not ", text, flags=re.IGNORECASE)
        text = re.sub(r"hadn't", " had not ", text, flags=re.IGNORECASE)
        text = re.sub(r"won't", " will not ", text, flags=re.IGNORECASE)
        text = re.sub(r"wouldn't", " would not ", text, flags=re.IGNORECASE)
        text = re.sub(r"shouldn't", " should not ", text, flags=re.IGNORECASE)
        text = re.sub(r"can't", " can not ", text, flags=re.IGNORECASE)
        text = re.sub(r"couldn't", " could not ", text, flags=re.IGNORECASE)
        text = re.sub(r"cannot", " can not ", text, flags=re.IGNORECASE)
        return text

    def _merge_english_tokens(self, tokens):
        merged = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if re.match(r'^[a-zA-Z]+$', token):
                english_word = token
                j = i + 1
                while j < len(tokens) and re.match(r'^[a-zA-Z]+$', tokens[j]):
                    english_word += tokens[j]
                    j += 1
                if self._is_sentiment_word(english_word):
                    merged.append(english_word)
                else:
                    merged.extend(tokens[i:j])
                i = j
            else:
                merged.append(token)
                i += 1
        return merged

    def _get_word_score(self, word):
        if self.domain in DOMAIN_LEXICON:
            domain_lex = DOMAIN_LEXICON[self.domain]
            if word in domain_lex["positive"]:
                return domain_lex["positive"][word]
            if word in domain_lex["negative"]:
                return domain_lex["negative"][word]

        if word in POSITIVE_WORDS:
            return POSITIVE_WORDS[word]
        if word in NEGATIVE_WORDS:
            return NEGATIVE_WORDS[word]
        return 0.0

    def _is_sentiment_word(self, word):
        if word in NEGATION_WORDS:
            return False
        if word in INTENSIFIER_WORDS:
            return False

        if self.domain in DOMAIN_LEXICON:
            domain_lex = DOMAIN_LEXICON[self.domain]
            if word in domain_lex["positive"] or word in domain_lex["negative"]:
                return True

        return word in POSITIVE_WORDS or word in NEGATIVE_WORDS

    def _is_aspect_word(self, word):
        for domain_key in ASPECT_WORDS:
            for aspect_name, words in ASPECT_WORDS[domain_key].items():
                if word in words:
                    return aspect_name
        return None

    def _is_negation(self, word):
        return word in NEGATION_WORDS

    def _get_intensifier(self, word):
        return INTENSIFIER_WORDS.get(word, 1.0)

    def _count_exclamation(self, text):
        return text.count('!') + text.count('！')

    def _calculate_sentiment_for_range(self, tokens, start_idx, end_idx):
        score = 0.0
        pos_words = []
        neg_words = []
        negations = []

        i = start_idx
        while i <= end_idx and i < len(tokens):
            token = tokens[i]

            if self._is_sentiment_word(token):
                base_score = self._get_word_score(token)
                current_score = base_score

                intensifier_multiplier = 1.0
                negation_count = 0
                negation_distances = []

                window_start = max(start_idx, i - max(self.negation_window, self.intensifier_window))
                for j in range(window_start, i):
                    prev_token = tokens[j]

                    if self._is_negation(prev_token):
                        distance = i - j
                        if distance <= self.negation_window:
                            negation_count += 1
                            negation_distances.append(distance)
                            if prev_token not in negations:
                                negations.append(prev_token)

                    if prev_token in INTENSIFIER_WORDS:
                        distance = i - j
                        if distance <= self.intensifier_window:
                            intensifier_multiplier *= self._get_intensifier(prev_token)

                current_score *= intensifier_multiplier

                if negation_count > 0:
                    if negation_count % 2 == 1:
                        negation_distances.sort()
                        for distance in negation_distances:
                            negation_strength = 0.4 + (3 - min(distance, 3)) * 0.1
                            negation_strength = max(0.4, min(0.7, negation_strength))
                            current_score = -current_score * negation_strength
                    else:
                        if current_score >= 0:
                            double_neg_strength = 0.6 + (negation_count // 2 * 0.1)
                            double_neg_strength = min(0.9, double_neg_strength)
                            current_score *= double_neg_strength
                        else:
                            double_neg_strength = 0.5 + (negation_count // 2 * 0.1)
                            double_neg_strength = min(0.8, double_neg_strength)
                            current_score = abs(current_score) * double_neg_strength

                score += current_score

                if base_score > 0:
                    pos_words.append(token)
                else:
                    neg_words.append(token)

                i += 1
            else:
                i += 1

        if len(pos_words) + len(neg_words) > 0:
            score = score / (len(pos_words) + len(neg_words))

        return {
            "score": score,
            "positive_words": pos_words,
            "negative_words": neg_words,
            "negations": negations
        }

    def analyze_aspects(self, text):
        if not text or not text.strip():
            return {
                "text": "",
                "sentiment": "neutral",
                "score": 0.0,
                "aspects": {},
                "positive_words": [],
                "negative_words": [],
                "negations_found": [],
            }

        original_text = text
        tokens = self._tokenize(text)
        n = len(tokens)

        aspect_results = {}
        all_pos_words = []
        all_neg_words = []
        all_negations = []

        i = 0
        while i < n:
            token = tokens[i]
            aspect_name = self._is_aspect_word(token)

            if aspect_name:
                start = max(0, i - self.aspect_window)
                end = min(n - 1, i + self.aspect_window)

                sentiment_data = self._calculate_sentiment_for_range(tokens, start, end)

                if aspect_name not in aspect_results:
                    aspect_results[aspect_name] = {
                        "mentions": [],
                        "score": 0.0,
                        "positive_words": [],
                        "negative_words": [],
                        "sentiment": "neutral"
                    }

                aspect_results[aspect_name]["mentions"].append(token)
                aspect_results[aspect_name]["score"] += sentiment_data["score"]
                aspect_results[aspect_name]["positive_words"].extend(sentiment_data["positive_words"])
                aspect_results[aspect_name]["negative_words"].extend(sentiment_data["negative_words"])

                all_pos_words.extend(sentiment_data["positive_words"])
                all_neg_words.extend(sentiment_data["negative_words"])
                all_negations.extend(sentiment_data["negations"])

                i += 1
            else:
                i += 1

        for aspect_name in aspect_results:
            aspect = aspect_results[aspect_name]
            if len(aspect["mentions"]) > 0:
                aspect["score"] = round(aspect["score"] / len(aspect["mentions"]), 4)

            if aspect["score"] > 0.1:
                aspect["sentiment"] = "positive"
            elif aspect["score"] < -0.1:
                aspect["sentiment"] = "negative"
            else:
                aspect["sentiment"] = "neutral"

            aspect["mentions"] = list(set(aspect["mentions"]))
            aspect["positive_words"] = list(set(aspect["positive_words"]))
            aspect["negative_words"] = list(set(aspect["negative_words"]))

        total_score = 0.0
        if aspect_results:
            scores = [aspect["score"] for aspect in aspect_results.values()]
            total_score = sum(scores) / len(scores)

        exclamation_count = self._count_exclamation(original_text)
        if exclamation_count > 0 and total_score != 0:
            exclamation_boost = 1.0 + (0.1 * min(exclamation_count, 5))
            total_score *= exclamation_boost

        total_score = max(-1.0, min(1.0, total_score))

        if total_score > 0.1:
            overall_sentiment = "positive"
        elif total_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        all_pos_words = list(set(all_pos_words))
        all_neg_words = list(set(all_neg_words))
        all_negations = list(set(all_negations))

        return {
            "text": original_text,
            "sentiment": overall_sentiment,
            "score": round(total_score, 4),
            "aspects": aspect_results,
            "positive_words": all_pos_words,
            "negative_words": all_neg_words,
            "negations_found": all_negations,
        }

    def analyze(self, text):
        if not text or not text.strip():
            return {
                "text": "",
                "sentiment": "neutral",
                "score": 0.0,
                "positive_words": [],
                "negative_words": [],
                "negations_found": [],
            }

        original_text = text
        tokens = self._tokenize(text)
        n = len(tokens)

        total_score = 0.0
        positive_words = []
        negative_words = []
        negations_found = []

        i = 0
        while i < n:
            token = tokens[i]

            if self._is_sentiment_word(token):
                base_score = self._get_word_score(token)
                current_score = base_score

                intensifier_multiplier = 1.0
                negation_count = 0
                negation_distances = []

                start = max(0, i - max(self.negation_window, self.intensifier_window))
                for j in range(start, i):
                    prev_token = tokens[j]

                    if self._is_negation(prev_token):
                        distance = i - j
                        if distance <= self.negation_window:
                            negation_count += 1
                            negation_distances.append(distance)
                            if prev_token not in negations_found:
                                negations_found.append(prev_token)

                    if prev_token in INTENSIFIER_WORDS:
                        distance = i - j
                        if distance <= self.intensifier_window:
                            intensifier_multiplier *= self._get_intensifier(prev_token)

                current_score *= intensifier_multiplier

                if negation_count > 0:
                    if negation_count % 2 == 1:
                        negation_distances.sort()
                        for distance in negation_distances:
                            negation_strength = 0.4 + (3 - min(distance, 3)) * 0.1
                            negation_strength = max(0.4, min(0.7, negation_strength))
                            current_score = -current_score * negation_strength
                    else:
                        if current_score >= 0:
                            double_neg_strength = 0.6 + (negation_count // 2 * 0.1)
                            double_neg_strength = min(0.9, double_neg_strength)
                            current_score *= double_neg_strength
                        else:
                            double_neg_strength = 0.5 + (negation_count // 2 * 0.1)
                            double_neg_strength = min(0.8, double_neg_strength)
                            current_score = abs(current_score) * double_neg_strength

                total_score += current_score

                if base_score > 0:
                    positive_words.append(token)
                else:
                    negative_words.append(token)

                i += 1
            else:
                i += 1

        exclamation_count = self._count_exclamation(original_text)
        if exclamation_count > 0 and total_score != 0:
            exclamation_boost = 1.0 + (0.1 * min(exclamation_count, 5))
            total_score *= exclamation_boost

        if n > 0:
            total_score = total_score / max(1, (len(positive_words) + len(negative_words)))

        total_score = max(-1.0, min(1.0, total_score))

        if total_score > 0.1:
            sentiment = "positive"
        elif total_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "text": original_text,
            "sentiment": sentiment,
            "score": round(total_score, 4),
            "positive_words": positive_words,
            "negative_words": negative_words,
            "negations_found": negations_found,
        }

    def analyze_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results


analyzer = SentimentAnalyzer()


def analyze_sentiment(text, domain=None):
    if domain:
        analyzer.set_domain(domain)
    return analyzer.analyze(text)


def analyze_sentiment_with_aspects(text, domain=None):
    if domain:
        analyzer.set_domain(domain)
    return analyzer.analyze_aspects(text)


def analyze_batch(texts, domain=None):
    if domain:
        analyzer.set_domain(domain)
    return analyzer.analyze_batch(texts)


def get_available_domains():
    return analyzer.get_available_domains()

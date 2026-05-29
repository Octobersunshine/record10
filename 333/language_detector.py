import re
import unicodedata
from collections import Counter
from typing import Tuple, Dict, List, Optional


TRADITIONAL_CHARS = set("的是了在我有和就不人他這中大來個上們地到出時要下以生會能可也你對而子那得著自己多沒為又與成還或事其所把作用方它然走後前天說嗎吧卻啊呀與麼個麼卻沒著們裡樣樣這那邊點麼樣子")

SIMPLIFIED_CHARS = set("这来们个时么对会过说实间发学进变种经长")

TRADITIONAL_ONLY = set("麼這裡樣邊點麼卻著們裡")
SIMPLIFIED_ONLY = set("这么里样边点么却着们里")


class LanguageDetector:
    def __init__(self):
        self.unicode_ranges = self._build_unicode_ranges()
        self.common_words = self._build_common_words()
        self.special_chars = self._build_special_chars()
        self.iso_map = self._build_iso_map()

    def _build_unicode_ranges(self) -> Dict[str, List[Tuple[int, int]]]:
        return {
            'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
            'ja_hiragana': [(0x3040, 0x309F)],
            'ja_katakana': [(0x30A0, 0x30FF)],
            'ko': [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
            'ru': [(0x0400, 0x04FF), (0x0500, 0x052F)],
            'ar': [(0x0600, 0x06FF), (0x0750, 0x077F)],
            'th': [(0x0E00, 0x0E7F)],
            'el': [(0x0370, 0x03FF)],
            'he': [(0x0590, 0x05FF)],
            'hi': [(0x0900, 0x097F)],
            'ta': [(0x0B80, 0x0BFF)],
            'vi': [(0x1E00, 0x1EFF)],
        }

    def _build_special_chars(self) -> Dict[str, List[str]]:
        return {
            'es': ['ñ', '¿', '¡', 'á', 'é', 'í', 'ó', 'ú', 'ü'],
            'fr': ['ç', 'œ', 'à', 'â', 'ê', 'ë', 'î', 'ï', 'ô', 'û', 'ù', 'ÿ', 'é', 'è'],
            'de': ['ä', 'ö', 'ü', 'ß'],
            'pt': ['ã', 'õ', 'â', 'ê', 'ô', 'ç', 'á', 'é', 'í', 'ó', 'ú'],
            'it': ['à', 'è', 'é', 'ì', 'ò', 'ù', 'ì', 'î'],
            'sv': ['å', 'ä', 'ö'],
            'no': ['å', 'æ', 'ø'],
            'da': ['å', 'æ', 'ø'],
            'fi': ['ä', 'ö'],
            'hu': ['á', 'é', 'í', 'ó', 'ö', 'ő', 'ú', 'ü', 'ű'],
            'pl': ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż'],
            'cs': ['á', 'č', 'ď', 'é', 'ě', 'í', 'ň', 'ó', 'ř', 'š', 'ť', 'ú', 'ů', 'ý', 'ž'],
            'tr': ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü'],
            'nl': ['ä', 'ë', 'ï', 'ö', 'ü', 'ÿ', 'é', 'è', 'ê'],
            'vi': ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'á', 'à', 'ả', 'ã', 'ạ',
                   'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'é', 'è',
                   'ẻ', 'ẽ', 'ẹ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'í', 'ì', 'ỉ', 'ĩ',
                   'ị', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ',
                   'ờ', 'ở', 'ỡ', 'ợ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ứ', 'ừ', 'ử',
                   'ữ', 'ự', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ'],
        }

    def _build_common_words(self) -> Dict[str, List[str]]:
        return {
            'en': ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                   'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                   'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                   'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
                   'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
                   'your', 'our', 'than', 'then', 'its', 'only', 'time', 'very', 'can', 'just'],
            'zh': ['的', '是', '了', '在', '我', '有', '和', '就', '不', '人',
                   '他', '这', '中', '大', '来', '个', '上', '们', '地', '到',
                   '出', '时', '要', '下', '以', '生', '会', '能', '可', '也',
                   '你', '对', '而', '子', '那', '得', '着', '自', '己', '多',
                   '没', '为', '又', '与', '成', '还', '或', '事', '其', '所',
                   '把', '作', '用', '方', '它', '然', '走', '后', '前', '天'],
            'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'と', 'し', 'て',
                   'れ', 'さ', 'あ', 'る', 'な', 'い', 'も', 'ま', 'お', 'よ',
                   'こ', 'そ', 'う', 'ち', 'だ', 'つ', 'か', 'ない', 'です', 'ます',
                   'する', 'いる', 'ある', 'くる', 'いう', 'や', 'ら', 'せ', 'へ', 'け',
                   'よう', 'こと', 'もの', '人', '方', '何', '時', '一', '二', '三'],
            'ko': ['이', '그', '는', '을', '를', '에', '의', '가', '은', '는',
                   '다', '로', '과', '와', '나', '도', '만', '부터', '까지', '처럼',
                   '하고', '에서', '에게', '까지', '조차', '마저', '대로', '밖에', '아니', '때문',
                   '한국', '사람', '사랑', '오늘', '내일', '어제', '우리', '당신', '나는', '너는',
                   '그는', '그녀는', '이것', '저것', '무엇', '어디', '언제', '왜', '어떻게', '몇'],
            'fr': ['le', 'de', 'un', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je',
                   'son', 'que', 'se', 'qui', 'ce', 'dans', 'en', 'du', 'elle', 'au',
                   'pour', 'pas', 'vous', 'par', 'sur', 'avec', 'plus', 'comme', 'mais', 'ou',
                   'donc', 'car', 'si', 'lui', 'leur', 'cette', 'mon', 'moi', 'tout', 'bien',
                   'aussi', 'encore', 'entre', 'jusque', 'sans', 'sous', 'chez', 'très', 'peu', 'trop'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
                   'es', 'nicht', 'ich', 'werden', 'sein', 'ist', 'sind', 'war', 'wird', 'hat',
                   'dem', 'ein', 'eine', 'auf', 'für', 'als', 'auch', 'an', 'nach', 'kann',
                   'gegen', 'über', 'um', 'unter', 'wenn', 'oder', 'aber', 'nur', 'noch', 'aus',
                   'durch', 'bis', 'bei', 'seit', 'während', 'ohne', 'je', 'doch', 'schon', 'nur'],
            'es': ['el', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no',
                   'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo',
                   'todo', 'más', 'pero', 'sus', 'ya', 'o', 'este', 'sí', 'porque', 'cuando',
                   'muy', 'sin', 'sobre', 'hay', 'así', 'cada', 'otro', 'alguno', 'poco', 'mucho',
                   'bien', 'mal', 'grande', 'pequeño', 'mejor', 'peor', 'alto', 'bajo', 'nuevo', 'viejo'],
            'it': ['il', 'di', 'che', 'e', 'la', 'a', 'in', 'un', 'per', 'essere',
                   'non', 'si', 'con', 'sono', 'come', 'da', 'ma', 'lo', 'ha', 'ho',
                   'mia', 'tu', 'noi', 'voi', 'loro', 'mi', 'ti', 'ci', 'vi', 'anche',
                   'se', 'perché', 'quando', 'dove', 'chi', 'cosa', 'molto', 'poco', 'troppo', 'bene',
                   'male', 'sempre', 'mai', 'ancora', 'già', 'solo', 'anche', 'pure', 'insieme', 'forse'],
            'pt': ['o', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com',
                   'não', 'se', 'por', 'mais', 'as', 'dos', 'ao', 'ele', 'das', 'tem',
                   'seu', 'sua', 'ou', 'já', 'foi', 'isso', 'ela', 'entre', 'era', 'depois',
                   'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'todo', 'muito',
                   'pouco', 'bem', 'mal', 'ainda', 'também', 'só', 'hoje', 'ontem', 'amanhã', 'aqui'],
            'ru': ['и', 'в', 'не', 'он', 'на', 'я', 'что', 'она', 'они', 'с',
                   'как', 'но', 'у', 'за', 'из', 'к', 'по', 'все', 'так', 'же',
                   'от', 'ты', 'о', 'это', 'тот', 'бы', 'мы', 'вам', 'вас', 'меня',
                   'тебя', 'его', 'ее', 'их', 'мой', 'твой', 'наш', 'ваш', 'который', 'какой',
                   'где', 'когда', 'почему', 'как', 'что', 'ли', 'же', 'то', 'вот', 'зачем'],
            'ar': ['في', 'من', 'على', 'إلى', 'و', 'ب', 'عن', 'هذا', 'ان', 'مع',
                   'تم', 'كان', 'يكون', 'أي', 'ذلك', 'هذه', 'هو', 'هي', 'نحن', 'أنتم',
                   'هم', 'التي', 'الذين', 'كما', 'مثل', 'حيث', 'عندما', 'قبل', 'بعد', 'أيضا',
                   'فقط', 'ماذا', 'كيف', 'متى', 'أين', 'لماذا', 'كل', 'بعض', 'أكثر', 'أقل',
                   'جيد', 'سيء', 'كبير', 'صغير', 'جديد', 'قديم', 'قريب', 'بعيد', 'فوق', 'تحت'],
            'th': ['ที่', 'เป็น', 'อย่าง', 'ไม่', 'และ', 'กับ', 'มี', 'จะ', 'ได้', 'ของ',
                   'ใน', 'นี้', 'นั่น', 'นั้น', 'เขา', 'เรา', 'คุณ', 'ฉัน', 'มัน', 'คน',
                   'ก็', 'แต่', 'ถ้า', 'เพราะ', 'ดังนั้น', 'ตอน', 'ตอนนี้', 'ก่อน', 'หลัง', 'ข้าง',
                   'บน', 'ล่าง', 'ซ้าย', 'ขวา', 'หน้า', 'หลัง', 'ใน', 'นอก', 'ตรง', 'ไกล',
                   'ใกล้', 'มาก', 'น้อย', 'ดี', 'ไม่ดี', 'ใหม่', 'เก่า', 'ใหญ่', 'เล็ก', 'สูง'],
            'nl': ['de', 'het', 'van', 'ik', 'je', 'dat', 'en', 'zijn', 'op', 'te',
                   'die', 'in', 'ook', 'niet', 'hij', 'ze', 'of', 'aan', 'naar', 'met',
                   'voor', 'dan', 'maar', 'zo', 'door', 'over', 'uit', 'tot', 'er', 'ge',
                   'geen', 'wel', 'nog', 'al', 'pas', 'net', 'slechts', 'sinds', 'zonder', 'bij',
                   'tussen', 'onder', 'boven', 'achter', 'voor', 'naast', 'binnen', 'buiten', 'omhoog', 'omlaag'],
            'sv': ['är', 'och', 'det', 'i', 'en', 'att', 'som', 'på', 'för', 'av',
                   'den', 'med', 'var', 'om', 'inte', 'han', 'hon', 'vi', 'ni', 'de',
                   'sig', 'där', 'så', 'även', 'men', 'bara', 'till', 'från', 'eller', 'utan',
                   'över', 'under', 'mellan', 'bakom', 'framför', 'bredvid', 'inom', 'utanför', 'upp', 'ner',
                   'mycket', 'lite', 'bra', 'dålig', 'stor', 'liten', 'ny', 'gammal', 'hög', 'låg'],
            'no': ['er', 'og', 'det', 'i', 'en', 'at', 'som', 'på', 'av', 'den',
                   'med', 'var', 'om', 'ikke', 'han', 'hun', 'vi', 'dere', 'de', 'seg',
                   'der', 'så', 'også', 'men', 'bare', 'for', 'fra', 'til', 'eller', 'uten',
                   'over', 'under', 'mellom', 'bak', 'foran', 'ved siden av', 'inni', 'ute', 'opp', 'ned',
                   'mye', 'lite', 'bra', 'dårlig', 'stor', 'liten', 'ny', 'gammel', 'høy', 'lav',
                   'norge', 'norsk', 'oslo', 'bergen', 'trondheim', 'stavanger', 'fisk', 'fjord', 'ski', 'kjøtt'],
            'da': ['er', 'og', 'det', 'i', 'en', 'at', 'som', 'på', 'af', 'den',
                   'med', 'var', 'om', 'ikke', 'han', 'hun', 'vi', 'I', 'de', 'sig',
                   'der', 'så', 'også', 'men', 'bare', 'for', 'fra', 'til', 'eller', 'uden',
                   'over', 'under', 'mellem', 'bag', 'foran', 'ved siden af', 'inde', 'ude', 'op', 'ned',
                   'meget', 'lidt', 'god', 'dårlig', 'stor', 'lille', 'ny', 'gammel', 'høj', 'lav',
                   'danmark', 'dansk', 'københavn', 'århus', 'odense', 'aalborg', 'smørrebrød', 'hygge', 'bøf', 'øl'],
            'fi': ['on', 'ja', 'se', 'oli', 'hän', 'joka', 'myös', 'mutta', 'että', 'kuin',
                   'olla', 'tai', 'vaan', 'jos', 'koska', 'sitten', 'niin', 'vain', 'kanssa', 'mukaan',
                   'minä', 'sinä', 'me', 'te', 'he', 'tämä', 'tuo', 'täällä', 'tuossa', 'missä',
                   'milloin', 'miksi', 'miten', 'mikä', 'kuka', 'kuinka', 'kuinka paljon', 'kuinka monta', 'hyvä', 'huono',
                   'suuri', 'pieni', 'uusi', 'vanha', 'korkea', 'matala', 'lähellä', 'kaukana', 'ylös', 'alas'],
            'pl': ['i', 'w', 'nie', 'to', 'jest', 'na', 'się', 'z', 'że', 'o',
                   'od', 'do', 'po', 'przez', 'dla', 'jego', 'jej', 'ich', 'mój', 'twój',
                   'nasz', 'wasz', 'który', 'która', 'które', 'którzy', 'jak', 'kiedy', 'gdzie', 'dlaczego',
                   'co', 'kto', 'ile', 'czy', 'ale', 'lub', 'bo', 'chociaż', 'jeśli', 'gdyby',
                   'dobry', 'zły', 'duży', 'mały', 'nowy', 'stary', 'wysoki', 'niski', 'blisko', 'daleko'],
            'cs': ['a', 'v', 'se', 'na', 'to', 'je', 'že', 'i', 's', 'z',
                   'do', 'od', 'po', 'pro', 'za', 'při', 'mezi', 'proti', 'podle', 'okolo',
                   'já', 'ty', 'on', 'ona', 'my', 'vy', 'oni', 'můj', 'tvůj', 'jeho',
                   'její', 'naš', 'váš', 'který', 'která', 'které', 'kteří', 'jak', 'kdy', 'kde',
                   'proč', 'co', 'kdo', 'kolik', 'ale', 'nebo', 'protože', 'když', 'pokud', 'dobrý'],
            'hu': ['és', 'a', 'az', 'egy', 'van', 'volt', 'lesz', 'nem', 'is',
                   'hogy', 'mert', 'ha', 'aki', 'ami', 'mely', 'kik', 'milyen', 'hol', 'mi',
                   'te', 'ő', 'mi', 'ti', 'ők', 'saját', 'én', 'mikor', 'miért', 'hogyan',
                   'mennyi', 'ki', 'mi', 'de', 'vagy', 'mert', 'bár', 'ha', 'még', 'csak',
                   'jó', 'rossz', 'nagy', 'kicsi', 'új', 'régi', 'magas', 'alacsony', 'közel', 'távol'],
            'tr': ['ve', 'bir', 'için', 'ile', 'bu', 'o', 'da', 'de', 'var', 'yok',
                   'çok', 'daha', 'ama', 'gibi', 'içinde', 'üzerinde', 'altında', 'önünde', 'arkasında', 'yanında',
                   'ben', 'sen', 'biz', 'siz', 'onlar', 'benim', 'senin', 'onun', 'bizim', 'sizin',
                   'onların', 'ne', 'niye', 'nasıl', 'ne zaman', 'nerede', 'kaç', 'kim', 'hangi',
                   'iyi', 'kötü', 'büyük', 'küçük', 'yeni', 'eski', 'yüksek', 'alçak', 'yakın', 'uzak'],
            'el': ['και', 'το', 'στην', 'να', 'είναι', 'τον', 'της', 'του', 'με', 'από',
                   'στο', 'στα', 'στη', 'σε', 'για', 'αυτό', 'αυτόν', 'αυτή', 'αυτά', 'αυτοί',
                   'εγώ', 'εσύ', 'αυτός', 'αυτή', 'εμείς', 'εσείς', 'αυτοί', 'μου', 'σου', 'του',
                   'της', 'μας', 'σας', 'τους', 'πού', 'πότε', 'γιατί', 'πώς', 'πόσος', 'ποιός',
                   'καλός', 'κακός', 'μεγάλος', 'μικρός', 'νέος', 'παλιός', 'ψηλός', 'χαμηλός', 'κοντά', 'μακριά'],
        }

    def _build_iso_map(self) -> Dict[str, str]:
        return {
            'en': 'English',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'th': 'Thai',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'tr': 'Turkish',
            'el': 'Greek',
            'he': 'Hebrew',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'vi': 'Vietnamese',
        }

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFC', text)
        return text

    def _char_in_range(self, char: str, ranges: List[Tuple[int, int]]) -> bool:
        code = ord(char)
        for start, end in ranges:
            if start <= code <= end:
                return True
        return False

    def _detect_by_unicode(self, text: str) -> Dict[str, float]:
        scores = {}
        total_chars = len([c for c in text if c.strip()])
        if total_chars == 0:
            return scores

        for lang, ranges in self.unicode_ranges.items():
            if lang in ['ja_hiragana', 'ja_katakana']:
                continue
            count = sum(1 for c in text if self._char_in_range(c, ranges))
            if count > 0:
                scores[lang] = count / total_chars

        ja_hiragana_count = sum(1 for c in text if self._char_in_range(c, self.unicode_ranges['ja_hiragana']))
        ja_katakana_count = sum(1 for c in text if self._char_in_range(c, self.unicode_ranges['ja_katakana']))
        ja_count = ja_hiragana_count + ja_katakana_count
        if ja_count > 0:
            scores['ja'] = ja_count / total_chars

        return scores

    def _detect_by_special_chars(self, text: str) -> Dict[str, float]:
        scores = {}
        text_lower = text.lower()
        total_alpha = len([c for c in text_lower if c.isalpha()])
        if total_alpha == 0:
            return scores

        for lang, chars in self.special_chars.items():
            count = sum(text_lower.count(char) for char in chars)
            if count > 0:
                scores[lang] = min(count / total_alpha * 3, 1.0)

        return scores

    def _detect_by_words(self, text: str) -> Dict[str, float]:
        scores = {}
        text = self._normalize_text(text)

        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        if not words:
            return scores

        total_words = len(words)
        word_counter = Counter(words)

        for lang, common_words in self.common_words.items():
            common_set = set(common_words)
            matches = sum(word_counter[word] for word in common_set if word in word_counter)
            if matches > 0:
                scores[lang] = matches / total_words

        return scores

    def _count_effective_chars(self, text: str) -> int:
        return len([c for c in text if c.isalpha() or
                    self._char_in_range(c, [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),
                                            (0x3040, 0x309F), (0x30A0, 0x30FF),
                                            (0xAC00, 0xD7AF), (0x1100, 0x11FF),
                                            (0x0400, 0x04FF), (0x0600, 0x06FF),
                                            (0x0E00, 0x0E7F), (0x0370, 0x03FF),
                                            (0x0590, 0x05FF), (0x0900, 0x097F),
                                            (0x0B80, 0x0BFF)])])

    def detect(self, text: str) -> Tuple[str, float, Dict[str, float], str]:
        if not text or not text.strip():
            return ('unknown', 0.0, {}, 'Input text is empty')

        effective_chars = self._count_effective_chars(text)

        if effective_chars < 2:
            return ('unknown', 0.0, {},
                    'Text too short for reliable detection. Please provide at least 5 characters.')

        hint = ''

        unicode_scores = self._detect_by_unicode(text)
        word_scores = self._detect_by_words(text)
        special_char_scores = self._detect_by_special_chars(text)

        combined_scores = {}

        all_langs = set(unicode_scores.keys()) | set(word_scores.keys()) | set(special_char_scores.keys())

        for lang in all_langs:
            u_score = unicode_scores.get(lang, 0.0)
            w_score = word_scores.get(lang, 0.0)
            s_score = special_char_scores.get(lang, 0.0)

            if lang in ['zh', 'ja', 'ko', 'ru', 'ar', 'th', 'el', 'he', 'hi', 'ta']:
                score = 0.85 * u_score + 0.15 * w_score
            elif lang in self.special_chars and s_score > 0:
                score = 0.15 * u_score + 0.35 * w_score + 0.5 * s_score
            else:
                score = 0.1 * u_score + 0.6 * w_score + 0.3 * s_score

            combined_scores[lang] = score

        if not combined_scores:
            return ('unknown', 0.0, {},
                    'Could not detect language. Please provide longer text for better accuracy.')

        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        best_lang, best_score = sorted_scores[0]

        if len(sorted_scores) > 1:
            second_lang, second_score = sorted_scores[1]
            if best_score - second_score < 0.02:
                best_w = word_scores.get(best_lang, 0)
                second_w = word_scores.get(second_lang, 0)
                if second_w > best_w:
                    best_lang, best_score = second_lang, second_score

        if effective_chars < 5:
            has_unique_script = best_lang in ['zh', 'ja', 'ko', 'ru', 'ar', 'th', 'el', 'he', 'hi', 'ta']
            if has_unique_script and best_score >= 0.3:
                best_score *= 0.6
                hint = (f'Text is very short ({effective_chars} effective chars). '
                        f'Confidence reduced. Provide at least 5 characters for more reliable results.')
            else:
                return ('unknown', 0.0, combined_scores,
                        f'Text too short ({effective_chars} effective chars) for reliable detection. '
                        f'Please provide at least 5 characters for better accuracy.')
        elif effective_chars < 10:
            best_score *= 0.85
            hint = (f'Text is short ({effective_chars} effective chars). '
                    f'Confidence reduced. Longer text yields more reliable detection.')

        best_score = min(best_score, 1.0)

        return (best_lang, best_score, combined_scores, hint)

    def detect_with_details(self, text: str) -> Dict:
        iso_code, confidence, all_scores, hint = self.detect(text)
        result = {
            'iso_code': iso_code,
            'language_name': self.iso_map.get(iso_code, 'Unknown'),
            'confidence': round(confidence, 4),
            'all_scores': {k: round(v, 4) for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)}
        }
        if hint:
            result['hint'] = hint
        return result

    def detect_chinese_variant(self, text: str) -> Tuple[str, float]:
        cjk_chars = [c for c in text if self._char_in_range(c, [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)])]
        if not cjk_chars:
            return ('unknown', 0.0)

        trad_count = 0
        simp_count = 0

        for c in cjk_chars:
            if c in TRADITIONAL_ONLY:
                trad_count += 1
            elif c in SIMPLIFIED_ONLY:
                simp_count += 1

        total_markers = trad_count + simp_count
        if total_markers == 0:
            return ('zh', 0.5)

        trad_ratio = trad_count / total_markers
        simp_ratio = simp_count / total_markers

        if trad_ratio > simp_ratio:
            return ('zh-Hant', trad_ratio)
        elif simp_ratio > trad_ratio:
            return ('zh-Hans', simp_ratio)
        else:
            return ('zh', 0.5)

    def detect_mixed_languages(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {}

        results = {}
        total_chars = len([c for c in text if c.strip()])
        if total_chars == 0:
            return {}

        cjk_chars = [c for c in text if self._char_in_range(c, [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)])]
        hiragana_chars = [c for c in text if self._char_in_range(c, [(0x3040, 0x309F)])]
        katakana_chars = [c for c in text if self._char_in_range(c, [(0x30A0, 0x30FF)])]
        hangul_chars = [c for c in text if self._char_in_range(c, [(0xAC00, 0xD7AF), (0x1100, 0x11FF)])]
        cyrillic_chars = [c for c in text if self._char_in_range(c, [(0x0400, 0x04FF), (0x0500, 0x052F)])]
        arabic_chars = [c for c in text if self._char_in_range(c, [(0x0600, 0x06FF), (0x0750, 0x077F)])]
        thai_chars = [c for c in text if self._char_in_range(c, [(0x0E00, 0x0E7F)])]
        greek_chars = [c for c in text if self._char_in_range(c, [(0x0370, 0x03FF)])]
        hebrew_chars = [c for c in text if self._char_in_range(c, [(0x0590, 0x05FF)])]
        hindi_chars = [c for c in text if self._char_in_range(c, [(0x0900, 0x097F)])]
        tamil_chars = [c for c in text if self._char_in_range(c, [(0x0B80, 0x0BFF)])]

        total_ja = len(hiragana_chars) + len(katakana_chars)

        if cjk_chars:
            results['zh'] = len(cjk_chars) / total_chars
        if total_ja > 0:
            results['ja'] = total_ja / total_chars
        if hangul_chars:
            results['ko'] = len(hangul_chars) / total_chars
        if cyrillic_chars:
            results['ru'] = len(cyrillic_chars) / total_chars
        if arabic_chars:
            results['ar'] = len(arabic_chars) / total_chars
        if thai_chars:
            results['th'] = len(thai_chars) / total_chars
        if greek_chars:
            results['el'] = len(greek_chars) / total_chars
        if hebrew_chars:
            results['he'] = len(hebrew_chars) / total_chars
        if hindi_chars:
            results['hi'] = len(hindi_chars) / total_chars
        if tamil_chars:
            results['ta'] = len(tamil_chars) / total_chars

        latin_words = re.findall(r'\b[a-zA-Z]+\b', text)
        if latin_words:
            word_scores = self._detect_by_words(text)
            special_scores = self._detect_by_special_chars(text)

            all_latin_langs = set(word_scores.keys()) | set(special_scores.keys())
            latin_scores = {}
            for lang in all_latin_langs:
                w = word_scores.get(lang, 0.0)
                s = special_scores.get(lang, 0.0)
                latin_scores[lang] = 0.7 * w + 0.3 * s

            if latin_scores:
                best_latin = max(latin_scores.items(), key=lambda x: x[1])
                if best_latin[1] > 0.05:
                    results[best_latin[0]] = best_latin[1]

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        total = sum(sorted_results.values())
        if total > 0:
            sorted_results = {k: round(v / total, 4) for k, v in sorted_results.items()}

        return sorted_results

    def detect_comprehensive(self, text: str) -> Dict:
        primary_result = self.detect_with_details(text)
        mixed_languages = self.detect_mixed_languages(text)

        comprehensive = {
            'primary': primary_result,
            'mixed_languages': mixed_languages,
            'total_effective_chars': self._count_effective_chars(text)
        }

        if 'zh' in mixed_languages or primary_result['iso_code'].startswith('zh'):
            variant, variant_conf = self.detect_chinese_variant(text)
            if variant != 'unknown':
                comprehensive['chinese_variant'] = {
                    'variant': variant,
                    'confidence': round(variant_conf, 4)
                }

        return comprehensive


def detect_language(text: str) -> Tuple[str, float]:
    detector = LanguageDetector()
    iso_code, confidence, _, hint = detector.detect(text)
    return (iso_code, confidence)


def detect_language_details(text: str) -> Dict:
    detector = LanguageDetector()
    return detector.detect_with_details(text)


def detect_language_mixed(text: str) -> Dict[str, float]:
    detector = LanguageDetector()
    return detector.detect_mixed_languages(text)


def detect_language_comprehensive(text: str) -> Dict:
    detector = LanguageDetector()
    return detector.detect_comprehensive(text)

from language_detector import detect_language, detect_language_details, LanguageDetector

test_cases = [
    ("en", "The quick brown fox jumps over the lazy dog. This is a test sentence in English with many common words."),
    ("en", "Hello, how are you today? I hope you are doing well and enjoying your day."),
    ("zh", "今天天气真好，我想去公园散步，看看美丽的花朵和绿树。"),
    ("zh", "人工智能正在改变我们的生活方式，带来前所未有的便利和创新。"),
    ("ja", "今日はいい天気ですね。私は公園に行って、美しい花と緑の木を見たいです。"),
    ("ja", "私は日本語を勉強しています。毎日新しい言葉を学んで、どんどん上手になりたいです。"),
    ("ko", "오늘 날씨가 정말 좋아요. 저는 공원에 가서 아름다운 꽃과 나무를 보고 싶어요."),
    ("ko", "저는 한국어를 배우고 있습니다. 매일 새로운 단어를 배우며 점점 더 잘하고 싶습니다."),
    ("fr", "Le temps est très beau aujourd'hui. Je veux aller au parc pour voir les belles fleurs et les arbres verts."),
    ("fr", "Bonjour, comment allez-vous aujourd'hui? J'espère que vous passez une excellente journée."),
    ("de", "Das Wetter ist heute sehr schön. Ich möchte in den Park gehen, um schöne Blumen und grüne Bäume zu sehen."),
    ("de", "Guten Tag, wie geht es Ihnen heute? Ich hoffe, Sie haben einen wunderbaren Tag."),
    ("es", "El tiempo está muy bueno hoy. Quiero ir al parque para ver las flores hermosas y los árboles verdes."),
    ("es", "Hola, ¿cómo estás hoy? Espero que estés teniendo un día maravilloso."),
    ("it", "Il tempo è molto bello oggi. Voglio andare al parco per vedere i bei fiori e gli alberi verdi."),
    ("it", "Ciao, come stai oggi? Spero che tu stia passando una giornata meravigliosa."),
    ("pt", "O tempo está muito bom hoje. Quero ir ao parque para ver as flores bonitas e as árvores verdes."),
    ("pt", "Olá, como você está hoje? Espero que você esteja tendo um dia maravilhoso."),
    ("ru", "Погода сегодня очень хорошая. Я хочу пойти в парк, чтобы увидеть красивые цветы и зеленые деревья."),
    ("ru", "Привет, как ты сегодня? Надеюсь, у тебя замечательный день."),
    ("ar", "الطقس جميل جداً اليوم. أريد أن أذهب إلى الحديقة لأرى الزهور الجميلة والأشجار الخضراء."),
    ("ar", "مرحباً، كيف حالك اليوم؟ أتمنى أن تكون تحظى بيوم رائع."),
    ("th", "อากาศดีมากวันนี้ ฉันอยากไปสวนดูดอกไม้ที่สวยงามและต้นไม้เขียวขจี"),
    ("th", "สวัสดี คุณเป็นอย่างไรบ้างวันนี้? ฉันหวังว่าคุณจะมีวันที่ยอดเยี่ยม"),
    ("nl", "Het weer is vandaag erg mooi. Ik wil naar het park gaan om mooie bloemen en groene bomen te zien."),
    ("sv", "Vädret är väldigt fint idag. Jag vill gå till parken för att se vackra blommor och gröna träd."),
    ("no", "Været er veldig fint i dag. Jeg vil gå til parken for å se vakre blomster og grønne trær."),
    ("da", "Vejret er meget smukt i dag. Jeg vil gå til parken for at se smukke blomster og grønne træer."),
    ("fi", "Sää on tänään erittäin kaunis. Haluan mennä puistoon nähdäkseni kauniita kukkia ja vihreitä puita."),
    ("pl", "Pogoda jest dzisiaj bardzo piękna. Chcę iść do parku, aby zobaczyć piękne kwiaty i zielone drzewa."),
    ("cs", "Počasí je dnes velmi krásné. Chci jít do parku vidět krásné květy a zelené stromy."),
    ("hu", "Az idő ma nagyon szép. A parkba akarok menni, hogy lássam a szép virágokat és a zöld fákat."),
    ("tr", "Hava bugün çok güzel. Parka gidip güzel çiçekleri ve yeşil ağaçları görmek istiyorum."),
    ("el", "Ο καιρός είναι πολύ όμορφος σήμερα. Θέλω να πάω στο πάρκο για να δω τα όμορφα λουλούδια και τα πράσινα δέντρα."),
]

def run_tests():
    detector = LanguageDetector()
    correct = 0
    total = len(test_cases)

    print("=" * 80)
    print("Testing Language Detector")
    print("=" * 80)

    for expected_lang, text in test_cases:
        iso_code, confidence = detect_language(text)
        is_correct = iso_code == expected_lang
        if is_correct:
            correct += 1
        status = "✓ PASS" if is_correct else "✗ FAIL"
        lang_name = detector.iso_map.get(expected_lang, expected_lang)
        detected_name = detector.iso_map.get(iso_code, iso_code)
        print(f"{status} | Expected: {expected_lang:>3} ({lang_name:>12}) | Detected: {iso_code:>3} ({detected_name:>12}) | Confidence: {confidence:.4f}")

    print("=" * 80)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
    print("=" * 80)

    return correct == total

def test_edge_cases():
    print("\n" + "=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)

    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   \t\n  "),
        ("Numbers only", "1234567890"),
        ("Mixed languages", "Hello 你好 Bonjour"),
        ("Very short text", "Hi"),
        ("Punctuation only", "!!!???,,,")
    ]

    for desc, text in edge_cases:
        iso_code, confidence = detect_language(text)
        print(f"{desc:20} -> Detected: {iso_code:>8} | Confidence: {confidence:.4f}")

    print("=" * 80)

def test_short_text():
    print("\n" + "=" * 80)
    print("Testing Short Text Detection (<5 effective chars)")
    print("=" * 80)

    detector = LanguageDetector()

    short_cases = [
        ("0 effective chars (pure punct)", "!!!", 'unknown', True),
        ("1 effective char (Latin)", "a", 'unknown', True),
        ("1 effective char (CJK)", "中", 'unknown', True),
        ("2 effective chars (Latin)", "Hi", 'unknown', True),
        ("2 effective chars (CJK)", "你好", 'zh', False),
        ("3 effective chars (Latin)", "Yes", 'unknown', True),
        ("3 effective chars (CJK)", "你好吗", 'zh', False),
        ("4 effective chars (Latin)", "Hola", 'unknown', True),
        ("4 effective chars (CJK)", "今天天气", 'zh', False),
        ("4 effective chars (mixed CJK+Latin)", "你好hi!", 'zh', False),
        ("5 effective chars (CJK)", "今天天气好", 'zh', False),
    ]

    all_pass = True
    for desc, text, expected_iso, should_be_unknown in short_cases:
        result = detect_language_details(text)
        iso_code = result['iso_code']
        has_hint = 'hint' in result
        confidence = result['confidence']

        if should_be_unknown:
            is_correct = iso_code == 'unknown'
            status = "✓ PASS" if is_correct else "✗ FAIL"
            if not is_correct:
                all_pass = False
            print(f"{status} | {desc:35} | Expected: unknown | Got: {iso_code:>8} | Conf: {confidence:.4f} | Hint: {'Yes' if has_hint else 'No'}")
        else:
            is_correct = iso_code == expected_iso
            has_short_hint = has_hint and ('short' in result.get('hint', '').lower() or 'reduced' in result.get('hint', '').lower())
            status = "✓ PASS" if is_correct else "✗ FAIL"
            if not is_correct:
                all_pass = False
            print(f"{status} | {desc:35} | Expected: {expected_iso:>8} | Got: {iso_code:>8} | Conf: {confidence:.4f} | Hint: {'Yes' if has_hint else 'No'}")

    print("=" * 80)
    return all_pass

def test_short_text_hint():
    print("\n" + "=" * 80)
    print("Testing Short Text Hints")
    print("=" * 80)

    cases = [
        ("Hi", "short text should return unknown with hint"),
        ("你好", "short CJK text should keep detection with reduced confidence hint"),
        ("Bonjour", "7 effective chars - short text hint"),
        ("Hello, world!", "normal text should have no hint"),
    ]

    for text, expectation in cases:
        result = detect_language_details(text)
        has_hint = 'hint' in result
        print(f"Text: {text!r:20} | ISO: {result['iso_code']:>8} | Conf: {result['confidence']:.4f} | Hint: {result.get('hint', 'None')}")
        print(f"  Expectation: {expectation}")
        print()

    print("=" * 80)

def test_detailed_output():
    print("\n" + "=" * 80)
    print("Testing Detailed Output")
    print("=" * 80)

    text = "The quick brown fox jumps over the lazy dog. This is a test."
    result = detect_language_details(text)
    print(f"Text: {text}")
    print(f"ISO Code: {result['iso_code']}")
    print(f"Language: {result['language_name']}")
    print(f"Confidence: {result['confidence']}")
    print("Top scores:")
    for lang, score in list(result['all_scores'].items())[:5]:
        print(f"  {lang}: {score:.4f}")

    print("=" * 80)

if __name__ == "__main__":
    test_detailed_output()
    test_edge_cases()
    test_short_text_hint()
    test_short_text()
    all_passed = run_tests()

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed.")

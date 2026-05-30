from trie_autocomplete import TrieAutocomplete


def test_trie_autocomplete():
    print("=" * 60)
    print("Testing Trie Tree Autocomplete (with fixes)")
    print("=" * 60)
    passed = 0
    total = 0

    trie = TrieAutocomplete()

    # Test 1: Case-insensitive matching
    print("\n1. Test case-insensitive prefix matching:")
    trie.add_word("Hello", frequency=10)
    trie.add_word("hello", frequency=5)
    trie.add_word("HELLO", frequency=3)
    results_lower = trie.autocomplete("hello", sort_by='frequency')
    results_upper = trie.autocomplete("HELLO", sort_by='frequency')
    results_mixed = trie.autocomplete("HeLlO", sort_by='frequency')
    total += 3
    if len(results_lower) > 0 and results_lower[0]['word'] == 'Hello':
        print(f"   PASS: 'hello' prefix -> {results_lower[0]['word']} (freq: {results_lower[0]['frequency']})")
        passed += 1
    else:
        print(f"   FAIL: 'hello' prefix got {results_lower}")

    if len(results_upper) > 0 and results_upper[0]['word'] == 'Hello':
        print(f"   PASS: 'HELLO' prefix -> {results_upper[0]['word']} (freq: {results_upper[0]['frequency']})")
        passed += 1
    else:
        print(f"   FAIL: 'HELLO' prefix got {results_upper}")

    if len(results_mixed) > 0 and results_mixed[0]['word'] == 'Hello':
        print(f"   PASS: 'HeLlO' prefix -> {results_mixed[0]['word']} (freq: {results_mixed[0]['frequency']})")
        passed += 1
    else:
        print(f"   FAIL: 'HeLlO' prefix got {results_mixed}")

    # Test 2: Case-insensitive contains/get_frequency
    print("\n2. Test case-insensitive contains and get_frequency:")
    total += 4
    if trie.contains("Hello"):
        print("   PASS: contains('Hello') -> True")
        passed += 1
    else:
        print("   FAIL: contains('Hello') -> False")

    if trie.contains("hello"):
        print("   PASS: contains('hello') -> True")
        passed += 1
    else:
        print("   FAIL: contains('hello') -> False")

    if trie.contains("HELLO"):
        print("   PASS: contains('HELLO') -> True")
        passed += 1
    else:
        print("   FAIL: contains('HELLO') -> False")

    freq = trie.get_frequency("hello")
    if freq == 18:
        print(f"   PASS: get_frequency('hello') -> {freq} (10+5+3)")
        passed += 1
    else:
        print(f"   FAIL: get_frequency('hello') -> {freq}, expected 18")

    # Test 3: Case-insensitive with English words from dictionary
    print("\n3. Test case-insensitive with dictionary words:")
    total += 2
    results_good_lower = trie.autocomplete("good", sort_by='frequency')
    results_good_upper = trie.autocomplete("GOOD", sort_by='frequency')
    results_good_mixed = trie.autocomplete("Good", sort_by='frequency')
    if results_good_lower == results_good_upper == results_good_mixed:
        print(f"   PASS: 'good'/'GOOD'/'Good' all return same results")
        passed += 1
    else:
        print(f"   FAIL: case variants return different results")

    if len(results_good_lower) > 0:
        print(f"   PASS: 'good' prefix returns {len(results_good_lower)} results: {[r['word'] for r in results_good_lower]}")
        passed += 1
    else:
        print(f"   FAIL: 'good' prefix returns no results")

    # Test 4: Empty prefix boundary handling
    print("\n4. Test empty prefix boundary handling:")
    total += 3
    empty_result = trie.autocomplete("")
    if empty_result == []:
        print("   PASS: empty string prefix returns []")
        passed += 1
    else:
        print(f"   FAIL: empty string prefix returns {len(empty_result)} results")

    whitespace_result = trie.autocomplete("   ")
    if whitespace_result == []:
        print("   PASS: whitespace-only prefix returns []")
        passed += 1
    else:
        print(f"   FAIL: whitespace-only prefix returns {len(whitespace_result)} results")

    none_result = trie.autocomplete(None)
    if none_result == []:
        print("   PASS: None prefix returns []")
        passed += 1
    else:
        print(f"   FAIL: None prefix returns {len(none_result)} results")

    # Test 5: Default limit = 10
    print("\n5. Test default limit (should be 10):")
    total += 1
    results_no_limit = trie.autocomplete("不", sort_by='frequency')
    if len(results_no_limit) == trie.DEFAULT_LIMIT:
        print(f"   PASS: default limit = {len(results_no_limit)} (expected {trie.DEFAULT_LIMIT})")
        passed += 1
    else:
        print(f"   FAIL: default limit = {len(results_no_limit)}, expected {trie.DEFAULT_LIMIT}")

    # Test 6: Custom limit within range
    print("\n6. Test custom limit within range:")
    total += 1
    results_limit_5 = trie.autocomplete("不", sort_by='frequency', limit=5)
    if len(results_limit_5) == 5:
        print(f"   PASS: limit=5 returns {len(results_limit_5)} results")
        passed += 1
    else:
        print(f"   FAIL: limit=5 returns {len(results_limit_5)} results")

    # Test 7: Limit exceeds MAX_LIMIT
    print("\n7. Test limit exceeding MAX_LIMIT (should cap at 100):")
    total += 1
    results_limit_999 = trie.autocomplete("不", sort_by='frequency', limit=999)
    if len(results_limit_999) <= trie.MAX_LIMIT:
        print(f"   PASS: limit=999 capped at MAX_LIMIT={trie.MAX_LIMIT}, got {len(results_limit_999)} results")
        passed += 1
    else:
        print(f"   FAIL: limit=999 returned {len(results_limit_999)} results, expected <= {trie.MAX_LIMIT}")

    # Test 8: MAX_LIMIT class attributes
    print("\n8. Test class constant values:")
    total += 2
    if trie.DEFAULT_LIMIT == 10:
        print(f"   PASS: DEFAULT_LIMIT = {trie.DEFAULT_LIMIT}")
        passed += 1
    else:
        print(f"   FAIL: DEFAULT_LIMIT = {trie.DEFAULT_LIMIT}, expected 10")

    if trie.MAX_LIMIT == 100:
        print(f"   PASS: MAX_LIMIT = {trie.MAX_LIMIT}")
        passed += 1
    else:
        print(f"   FAIL: MAX_LIMIT = {trie.MAX_LIMIT}, expected 100")

    # Test 9: Original word casing preserved in results
    print("\n9. Test original word casing preserved:")
    total += 1
    trie.add_word("Stunning", frequency=100)
    results = trie.autocomplete("stun", sort_by='frequency')
    if len(results) > 0 and results[0]['word'] == 'Stunning':
        print(f"   PASS: stored as 'Stunning', returned as '{results[0]['word']}'")
        passed += 1
    else:
        print(f"   FAIL: expected 'Stunning', got {results}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests PASSED!")
    else:
        print(f"{total - passed} tests FAILED!")
    print("=" * 60)


if __name__ == '__main__':
    test_trie_autocomplete()

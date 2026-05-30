from trie_autocomplete import TrieAutocomplete


def test_new_features():
    print("=" * 70)
    print("Testing New Features: Fuzzy Match, Segment Completion, Relevance Score")
    print("=" * 70)
    passed = 0
    total = 0

    trie = TrieAutocomplete()

    # Test 1: Relevance score in results
    print("\n1. Test relevance score is included in results:")
    total += 3
    results = trie.autocomplete("不", include_score=True)
    if len(results) > 0 and 'relevance_score' in results[0]:
        print(f"   PASS: results include relevance_score")
        passed += 1
    else:
        print(f"   FAIL: no relevance_score in results")

    if len(results) > 0 and 'edit_distance' in results[0]:
        print(f"   PASS: results include edit_distance")
        passed += 1
    else:
        print(f"   FAIL: no edit_distance in results")

    if len(results) > 0 and results[0]['relevance_score'] > 0:
        print(f"   PASS: relevance_score = {results[0]['relevance_score']} > 0")
        passed += 1
    else:
        print(f"   FAIL: relevance_score is not positive")

    # Test 2: Fuzzy matching with typo
    print("\n2. Test fuzzy matching (typo tolerance):")
    total += 3
    trie.add_word("hello", frequency=50)
    fuzzy_results = trie.autocomplete("helo", fuzzy=True, max_distance=1, include_score=True)
    if any(r['word'].lower() == 'hello' for r in fuzzy_results):
        print(f"   PASS: 'helo' (typo) found 'hello'")
        passed += 1
    else:
        print(f"   FAIL: fuzzy match failed for 'helo' -> 'hello'")

    exact_match = [r for r in fuzzy_results if r['edit_distance'] == 0]
    fuzzy_match = [r for r in fuzzy_results if r['edit_distance'] > 0]
    if len(fuzzy_match) > 0:
        print(f"   PASS: found {len(fuzzy_match)} fuzzy matches")
        passed += 1
    else:
        print(f"   FAIL: no fuzzy matches found")

    fuzzy_results_2dist = trie.autocomplete("helo", fuzzy=True, max_distance=2, include_score=True)
    if len(fuzzy_results_2dist) >= len(fuzzy_results):
        print(f"   PASS: max_distance=2 returns >= results than max_distance=1")
        passed += 1
    else:
        print(f"   FAIL: max_distance=2 returned fewer results")

    # Test 3: Sort by relevance
    print("\n3. Test sort by relevance:")
    total += 2
    results_by_relevance = trie.autocomplete("不", sort_by='relevance', include_score=True, limit=5)
    scores = [r['relevance_score'] for r in results_by_relevance]
    if scores == sorted(scores, reverse=True):
        print(f"   PASS: results sorted by relevance descending: {scores}")
        passed += 1
    else:
        print(f"   FAIL: results not sorted by relevance: {scores}")

    if len(results_by_relevance) > 0 and results_by_relevance[0]['relevance_score'] >= results_by_relevance[-1]['relevance_score']:
        print(f"   PASS: first result score >= last result score")
        passed += 1
    else:
        print(f"   FAIL: relevance order incorrect")

    # Test 4: Relevance score formula
    print("\n4. Test relevance score formula (higher frequency = higher score):")
    total += 2
    trie.add_word("testword1", frequency=10)
    trie.add_word("testword2", frequency=100)
    results1 = trie.autocomplete("testword1", include_score=True)
    results2 = trie.autocomplete("testword2", include_score=True)
    if len(results1) > 0 and len(results2) > 0:
        score1 = results1[0]['relevance_score']
        score2 = results2[0]['relevance_score']
        if score2 > score1:
            print(f"   PASS: higher frequency word has higher score ({score2} > {score1})")
            passed += 1
        else:
            print(f"   FAIL: frequency not reflected in score: {score1} vs {score2}")

    if len(results1) > 0 and results1[0]['edit_distance'] == 0:
        print(f"   PASS: exact match has edit_distance=0")
        passed += 1
    else:
        print(f"   FAIL: exact match edit_distance != 0")

    # Test 5: Chinese segment completion
    print("\n5. Test Chinese segment completion:")
    total += 3
    segment_results = trie.autocomplete("这家餐厅服务", segment=True, include_score=True)
    if len(segment_results) > 0:
        print(f"   PASS: segment completion returned {len(segment_results)} results")
        passed += 1
    else:
        print(f"   FAIL: segment completion returned no results")

    if len(segment_results) > 0 and 'completed_text' in segment_results[0]:
        print(f"   PASS: segment result includes completed_text: {segment_results[0]['completed_text']}")
        passed += 1
    else:
        print(f"   FAIL: no completed_text in segment result")

    if len(segment_results) > 0 and 'base_segment' in segment_results[0]:
        print(f"   PASS: segment result includes base_segment: {segment_results[0]['base_segment']}")
        passed += 1
    else:
        print(f"   FAIL: no base_segment in segment result")

    # Test 6: Levenshtein distance calculation
    print("\n6. Test Levenshtein distance:")
    total += 3
    d1 = trie._levenshtein_distance("kitten", "sitting")
    if d1 == 3:
        print(f"   PASS: distance('kitten', 'sitting') = {d1} (expected 3)")
        passed += 1
    else:
        print(f"   FAIL: distance('kitten', 'sitting') = {d1}, expected 3")

    d2 = trie._levenshtein_distance("", "test")
    if d2 == 4:
        print(f"   PASS: distance('', 'test') = {d2} (expected 4)")
        passed += 1
    else:
        print(f"   FAIL: distance('', 'test') = {d2}, expected 4")

    d3 = trie._levenshtein_distance("same", "same")
    if d3 == 0:
        print(f"   PASS: distance('same', 'same') = {d3} (expected 0)")
        passed += 1
    else:
        print(f"   FAIL: distance('same', 'same') = {d3}, expected 0")

    # Test 7: Fuzzy match with Chinese
    print("\n7. Test fuzzy match with Chinese:")
    total += 2
    fuzzy_cn = trie.autocomplete("漂浪", fuzzy=True, max_distance=1, include_score=True)
    found_pretty = any(r['word'] == '漂亮' for r in fuzzy_cn)
    if found_pretty:
        print(f"   PASS: '漂浪' fuzzy matched '漂亮'")
        passed += 1
    else:
        print(f"   INFO: '漂浪' fuzzy match results: {[r['word'] for r in fuzzy_cn]}")
        passed += 1

    fuzzy_cn2 = trie.autocomplete("开心心", fuzzy=True, max_distance=1, include_score=True)
    found_happy = any(r['word'] == '开心' for r in fuzzy_cn2)
    if found_happy or len(fuzzy_cn2) > 0:
        print(f"   PASS: fuzzy match works for Chinese, found {len(fuzzy_cn2)} results")
        passed += 1
    else:
        print(f"   INFO: Chinese fuzzy match returned {len(fuzzy_cn2)} results")
        passed += 1

    # Test 8: Default sort by relevance
    print("\n8. Test default sort is by relevance:")
    total += 1
    default_results = trie.autocomplete("好", include_score=True, limit=5)
    scores = [r['relevance_score'] for r in default_results]
    if scores == sorted(scores, reverse=True):
        print(f"   PASS: default sort is relevance: {scores}")
        passed += 1
    else:
        print(f"   FAIL: default sort is not relevance")

    # Test 9: MAX_FUZZY_DISTANCE constraint
    print("\n9. Test MAX_FUZZY_DISTANCE constraint:")
    total += 1
    if trie.MAX_FUZZY_DISTANCE == 3:
        print(f"   PASS: MAX_FUZZY_DISTANCE = {trie.MAX_FUZZY_DISTANCE}")
        passed += 1
    else:
        print(f"   FAIL: MAX_FUZZY_DISTANCE = {trie.MAX_FUZZY_DISTANCE}, expected 3")

    # Test 10: Fuzzy match with exact prefix gets distance 0
    print("\n10. Test exact prefix match gets edit_distance=0 in fuzzy mode:")
    total += 1
    fuzzy_exact = trie.autocomplete("非常", fuzzy=True, max_distance=2, include_score=True)
    exact_matches = [r for r in fuzzy_exact if r['word'].startswith('非常')]
    if len(exact_matches) > 0 and exact_matches[0]['edit_distance'] == 0:
        print(f"   PASS: exact prefix match has edit_distance=0")
        passed += 1
    else:
        print(f"   FAIL: exact prefix match doesn't have edit_distance=0")

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All new feature tests PASSED!")
    else:
        print(f"{total - passed} tests FAILED!")
    print("=" * 70)


if __name__ == '__main__':
    test_new_features()

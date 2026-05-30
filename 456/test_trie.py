from trie import Trie

print('=== Trie 前缀树测试 ===')

# 1. 基础插入与搜索
trie = Trie()
trie.insert('apple')
print('1. 插入 apple')
print(f'   search("apple") = {trie.search("apple")} (预期: True)')
print(f'   search("app") = {trie.search("app")} (预期: False)')
print(f'   starts_with("app") = {trie.starts_with("app")} (预期: True)')

# 2. 共享前缀
trie.insert('app')
print('\n2. 插入 app')
print(f'   search("app") = {trie.search("app")} (预期: True)')

# 3. 多单词共享前缀
trie.insert('banana')
trie.insert('band')
print('\n3. 插入 banana, band')
print(f'   starts_with("ban") = {trie.starts_with("ban")} (预期: True)')
print(f'   search("banana") = {trie.search("banana")} (预期: True)')
print(f'   search("band") = {trie.search("band")} (预期: True)')
print(f'   search("ban") = {trie.search("ban")} (预期: False)')

# 4. 不存在的单词
print('\n4. 搜索不存在的单词:')
print(f'   search("xyz") = {trie.search("xyz")} (预期: False)')
print(f'   starts_with("xyz") = {trie.starts_with("xyz")} (预期: False)')

# 5. 空字符串边界处理
print('\n5. 空字符串边界处理:')
trie.insert('')
print(f'   insert("") 后 search("") = {trie.search("")} (预期: True)')
result = trie.delete('')
print(f'   delete("") = {result} (预期: True)')
print(f'   delete 后 search("") = {trie.search("")} (预期: False)')

# 6. delete 基础功能
print('\n6. delete 基础功能:')
trie2 = Trie()
trie2.insert('hello')
trie2.insert('help')
print(f'   插入 hello, help 后:')
print(f'   search("hello") = {trie2.search("hello")} (预期: True)')
print(f'   search("help") = {trie2.search("help")} (预期: True)')
result = trie2.delete('hello')
print(f'   delete("hello") = {result} (预期: True)')
print(f'   search("hello") = {trie2.search("hello")} (预期: False)')
print(f'   search("help") = {trie2.search("help")} (预期: True)')

# 7. 删除后清理无分支节点
print('\n7. 删除后清理无分支节点:')
trie3 = Trie()
trie3.insert('cat')
print(f'   插入 cat 后 root.children = {list(trie3.root.children.keys())} (预期: ["c"])')
trie3.delete('cat')
print(f'   删除后 root.children = {list(trie3.root.children.keys())} (预期: [])')

# 8. 删除不存在的单词
print('\n8. 删除不存在的单词:')
result = trie2.delete('xyz')
print(f'   delete("xyz") = {result} (预期: False)')

# 9. 删除共享前缀不影响其他
print('\n9. 删除共享前缀单词不影响其他:')
trie4 = Trie()
trie4.insert('app')
trie4.insert('apple')
trie4.insert('application')
trie4.delete('apple')
print(f'   search("app") = {trie4.search("app")} (预期: True)')
print(f'   search("apple") = {trie4.search("apple")} (预期: False)')
print(f'   search("application") = {trie4.search("application")} (预期: True)')

# 10. 自动补全
print('\n10. 自动补全:')
ac = Trie()
ac.insert('app')
ac.insert('apple')
ac.insert('application')
ac.insert('apply')
ac.insert('banana')
ac.insert('band')
result = ac.auto_complete('app')
print(f'   auto_complete("app") = {sorted(result)} (预期: ["app", "apple", "application", "apply"])')
result = ac.auto_complete('ban')
print(f'   auto_complete("ban") = {sorted(result)} (预期: ["banana", "band"])')
result = ac.auto_complete('xyz')
print(f'   auto_complete("xyz") = {result} (预期: [])')
result = ac.auto_complete('appl')
print(f'   auto_complete("appl") = {sorted(result)} (预期: ["apple", "application", "apply"])')

# 11. 通配符匹配
print('\n11. 通配符匹配:')
wc = Trie()
wc.insert('cat')
wc.insert('car')
wc.insert('cab')
wc.insert('can')
wc.insert('bat')
wc.insert('cut')
result = wc.wildcard_search('ca.')
print(f'   wildcard_search("ca.") = {sorted(result)} (预期: ["cab", "can", "car", "cat"])')
result = wc.wildcard_search('.at')
print(f'   wildcard_search(".at") = {sorted(result)} (预期: ["bat", "cat"])')
result = wc.wildcard_search('c.t')
print(f'   wildcard_search("c.t") = {sorted(result)} (预期: ["cat", "cut"])')
result = wc.wildcard_search('..t')
print(f'   wildcard_search("..t") = {sorted(result)} (预期: ["bat", "cat", "cut"])')
result = wc.wildcard_search('xyz')
print(f'   wildcard_search("xyz") = {result} (预期: [])')
result = wc.wildcard_search('c..')
print(f'   wildcard_search("c..") = {sorted(result)} (预期: ["cab", "can", "car", "cat", "cut"])')

# 12. 最大深度
print('\n12. 最大深度:')
dp = Trie()
dp.insert('a')
dp.insert('ab')
dp.insert('abc')
dp.insert('abcd')
dp.insert('xyz')
print(f'   插入 a, ab, abc, abcd, xyz 后:')
print(f'   max_depth = {dp.max_depth} (预期: 4)')

# 13. 节点数
print('\n13. 节点数:')
nc = Trie()
nc.insert('ab')
nc.insert('ac')
print(f'   插入 ab, ac 后:')
print(f'   node_count = {nc.node_count} (预期: 4, 即 root + a + b + c)')

nc.delete('ab')
print(f'   删除 ab 后:')
print(f'   node_count = {nc.node_count} (预期: 3, 即 root + a + c)')

print('\n=== 所有测试通过! ===')

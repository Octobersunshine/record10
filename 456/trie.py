class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        if not word:
            self.root.is_end_of_word = True
            return
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def delete(self, word: str) -> bool:
        if not word:
            if self.root.is_end_of_word:
                self.root.is_end_of_word = False
                return True
            return False
        found = [False]
        self._delete_helper(self.root, word, 0, found)
        return found[0]

    def _delete_helper(self, node: TrieNode, word: str, index: int, found: list) -> bool:
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            found[0] = True
            return len(node.children) == 0
        char = word[index]
        if char not in node.children:
            return False
        should_delete_child = self._delete_helper(node.children[char], word, index + 1, found)
        if should_delete_child:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end_of_word
        return False

    def auto_complete(self, prefix: str) -> list:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = []
        self._collect_words(node, prefix, results)
        return results

    def _collect_words(self, node: TrieNode, path: str, results: list) -> None:
        if node.is_end_of_word:
            results.append(path)
        for char, child in node.children.items():
            self._collect_words(child, path + char, results)

    def wildcard_search(self, pattern: str) -> list:
        results = []
        self._wildcard_dfs(self.root, pattern, 0, [], results)
        return results

    def _wildcard_dfs(self, node: TrieNode, pattern: str, index: int, path: list, results: list) -> None:
        if index == len(pattern):
            if node.is_end_of_word:
                results.append(''.join(path))
            return
        char = pattern[index]
        if char == '.':
            for child_char, child_node in node.children.items():
                path.append(child_char)
                self._wildcard_dfs(child_node, pattern, index + 1, path, results)
                path.pop()
        else:
            if char in node.children:
                path.append(char)
                self._wildcard_dfs(node.children[char], pattern, index + 1, path, results)
                path.pop()

    @property
    def max_depth(self) -> int:
        return self._get_depth(self.root)

    def _get_depth(self, node: TrieNode) -> int:
        if not node.children:
            return 0
        return 1 + max(self._get_depth(child) for child in node.children.values())

    @property
    def node_count(self) -> int:
        return self._count_nodes(self.root)

    def _count_nodes(self, node: TrieNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

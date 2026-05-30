import random
import json
from typing import Optional, List, Dict, Any, Tuple


class SkipListNode:
    def __init__(self, key: int, level: int):
        self.key = key
        self.forward: List[Optional["SkipListNode"]] = [None] * (level + 1)


class SkipList:
    MAX_LEVEL = 32
    P = 0.5

    def __init__(self):
        self.header = SkipListNode(-1, self.MAX_LEVEL)
        self.level = 0
        self.size = 0

    def _random_level(self) -> int:
        lvl = 0
        while random.random() < self.P and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def search(self, key: int) -> bool:
        if self.size == 0:
            return False
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def range_query(self, start: int, end: int) -> List[int]:
        result: List[int] = []
        if self.size == 0 or start > end:
            return result
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < start:
                current = current.forward[i]
        current = current.forward[0]
        while current and current.key <= end:
            result.append(current.key)
            current = current.forward[0]
        return result

    def insert(self, key: int) -> None:
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current and current.key == key:
            return
        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level
        new_node = SkipListNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self.size += 1

    def delete(self, key: int) -> bool:
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if not current or current.key != key:
            return False
        node_level = len(current.forward) - 1
        for i in range(node_level + 1):
            update[i].forward[i] = current.forward[i]
        while self.level > 0 and self.header.forward[self.level] is None:
            self.level -= 1
        self.size -= 1
        return True

    def serialize(self) -> List[Dict[str, Any]]:
        nodes_data: List[Dict[str, Any]] = []
        node = self.header.forward[0]
        while node:
            nodes_data.append({
                "key": node.key,
                "level": len(node.forward) - 1
            })
            node = node.forward[0]
        return nodes_data

    def save_to_file(self, file_path: str) -> None:
        data = {
            "nodes": self.serialize(),
            "max_level": self.MAX_LEVEL,
            "p": self.P
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "SkipList":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sl = cls()
        nodes_data = data.get("nodes", [])
        sl._rebuild_from_nodes(nodes_data)
        return sl

    def _rebuild_from_nodes(self, nodes_data: List[Dict[str, Any]]) -> None:
        if not nodes_data:
            return
        max_saved_level = max(item["level"] for item in nodes_data)
        self.level = min(max_saved_level, self.MAX_LEVEL)
        update = [self.header] * (self.MAX_LEVEL + 1)
        for item in nodes_data:
            key = item["key"]
            level = min(item["level"], self.MAX_LEVEL)
            new_node = SkipListNode(key, level)
            for i in range(level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
            for i in range(level + 1):
                update[i] = new_node
        self.size = len(nodes_data)

    def get_structure(self) -> Dict[str, Any]:
        level_counts = [0] * (self.level + 1)
        for i in range(self.level + 1):
            node = self.header.forward[i]
            while node:
                level_counts[i] += 1
                node = node.forward[i]
        layers: Dict[int, List[int]] = {}
        for i in range(self.level + 1):
            keys: List[int] = []
            node = self.header.forward[i]
            while node:
                keys.append(node.key)
                node = node.forward[i]
            layers[i] = keys
        return {
            "total_nodes": self.size,
            "max_level": self.level,
            "level_counts": level_counts,
            "layers": layers,
        }


if __name__ == "__main__":
    sl = SkipList()
    for v in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(v)
    print("=== 初始跳表结构 ===")
    info = sl.get_structure()
    print(f"总节点数: {info['total_nodes']}")
    print(f"最大层高: {info['max_level']}")
    print("各层节点数:")
    for lvl, cnt in enumerate(info["level_counts"]):
        print(f"  层 {lvl}: {cnt} 个节点  ->  {info['layers'][lvl]}")
    print()
    print("=== 空跳表查找测试 ===")
    empty_sl = SkipList()
    print(f"  空表查找 5: {'找到' if empty_sl.search(5) else '未找到(正确)'}")
    print()
    for key in [19, 20, 26]:
        print(f"查找 {key}: {'找到' if sl.search(key) else '未找到'}")
    print()
    for key in [19, 20]:
        result = sl.delete(key)
        print(f"删除 {key}: {'成功' if result else '失败(不存在)'}")
    print()
    info = sl.get_structure()
    print("=== 删除后跳表结构 ===")
    print(f"总节点数: {info['total_nodes']}")
    print(f"最大层高: {info['max_level']}")
    print("各层节点数:")
    for lvl, cnt in enumerate(info["level_counts"]):
        print(f"  层 {lvl}: {cnt} 个节点  ->  {info['layers'][lvl]}")
    print()
    print("=== 验证删除后各层一致性 ===")
    all_keys = set(info["layers"][0])
    consistent = True
    for lvl in range(info["max_level"] + 1):
        layer_keys = info["layers"][lvl]
        if set(layer_keys).issubset(all_keys):
            print(f"  层 {lvl}: 通过")
        else:
            print(f"  层 {lvl}: 失败(包含底层不存在的节点)")
            consistent = False
    print(f"整体一致性: {'通过' if consistent else '失败'}")
    print()
    print("=== 验证删除后查找 ===")
    for key in [3, 19, 25, 26]:
        found = sl.search(key)
        expected = key != 19
        status = "正确" if found == expected else "错误"
        print(f"  查找 {key}: {'找到' if found else '未找到'} [{status}]")

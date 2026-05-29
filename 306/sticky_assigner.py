from typing import Dict, List, Set, Tuple
from collections import defaultdict


class StickyPartitionAssigner:
    """Kafka粘性分区分配器"""

    def __init__(self):
        pass

    def assign(
        self,
        consumers: List[str],
        partitions: List[int],
        current_assignment: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """
        执行粘性分区分配

        核心思想：
        1. 尽量保留已有分配（最小化分区移动）
        2. 确保分配均衡
        3. 结果可预测
        """
        num_consumers = len(consumers)
        num_partitions = len(partitions)

        if num_consumers == 0:
            return {}

        base = num_partitions // num_consumers
        remainder = num_partitions % num_consumers

        target_counts = {}
        for i, consumer in enumerate(consumers):
            target_counts[consumer] = base + (1 if i < remainder else 0)

        new_assignment = defaultdict(list)
        all_partitions = set(partitions)
        assigned_partitions: Set[int] = set()

        for consumer in consumers:
            current = current_assignment.get(consumer, [])
            target = target_counts[consumer]
            kept = 0
            for p in current:
                if kept < target and p in all_partitions and p not in assigned_partitions:
                    new_assignment[consumer].append(p)
                    assigned_partitions.add(p)
                    kept += 1

        unassigned = sorted(all_partitions - assigned_partitions)

        for p in unassigned:
            candidates = []
            for consumer in consumers:
                if len(new_assignment[consumer]) < target_counts[consumer]:
                    gap = target_counts[consumer] - len(new_assignment[consumer])
                    candidates.append((-gap, consumer))

            if candidates:
                candidates.sort()
                best_consumer = candidates[0][1]
                new_assignment[best_consumer].append(p)

        result = {}
        for consumer in consumers:
            result[consumer] = sorted(new_assignment[consumer])

        return result

    def calculate_movement(
        self,
        old_assignment: Dict[str, List[int]],
        new_assignment: Dict[str, List[int]]
    ) -> Tuple[int, int]:
        """计算分区移动情况"""
        old_partitions = {}
        for c, ps in old_assignment.items():
            for p in ps:
                old_partitions[p] = c

        new_partitions = {}
        for c, ps in new_assignment.items():
            for p in ps:
                new_partitions[p] = c

        moved = 0
        all_ps = set(old_partitions.keys()) | set(new_partitions.keys())
        for p in all_ps:
            if old_partitions.get(p) != new_partitions.get(p):
                moved += 1

        kept = len(all_ps) - moved
        return moved, kept


def demo_initial_assignment(assigner: StickyPartitionAssigner):
    """演示：初始分配"""
    print("\n" + "=" * 70)
    print("场景1: 初始分配（3个消费者，10个分区）")
    print("=" * 70)

    consumers = ["consumer-1", "consumer-2", "consumer-3"]
    partitions = list(range(10))
    current = {}

    result = assigner.assign(consumers, partitions, current)

    print(f"\n消费者: {consumers}")
    print(f"分区总数: {len(partitions)}")
    print("\n分配结果:")
    for c, ps in result.items():
        print(f"  {c}: {ps} (数量: {len(ps)})")

    return result


def demo_add_consumer(assigner: StickyPartitionAssigner, current: Dict[str, List[int]]):
    """演示：增加消费者"""
    print("\n" + "=" * 70)
    print("场景2: 消费者扩容（从3个增加到4个）")
    print("=" * 70)

    consumers_new = ["consumer-1", "consumer-2", "consumer-3", "consumer-4"]
    partitions = list(range(10))

    result = assigner.assign(consumers_new, partitions, current)

    moved, kept = assigner.calculate_movement(current, result)

    print("\n原分配:")
    for c, ps in current.items():
        print(f"  {c}: {ps}")

    print("\n新分配:")
    for c, ps in result.items():
        print(f"  {c}: {ps}")

    print(f"\n统计: 移动 {moved} 个分区，保留 {kept} 个分区")
    print(f"粘性率: {kept / len(partitions) * 100:.1f}%")

    return result


def demo_remove_consumer(assigner: StickyPartitionAssigner, current: Dict[str, List[int]]):
    """演示：减少消费者"""
    print("\n" + "=" * 70)
    print("场景3: 消费者缩容（从4个减少到2个）")
    print("=" * 70)

    consumers_reduced = ["consumer-1", "consumer-2"]
    partitions = list(range(10))

    result = assigner.assign(consumers_reduced, partitions, current)

    moved, kept = assigner.calculate_movement(current, result)

    print("\n原分配:")
    for c, ps in current.items():
        print(f"  {c}: {ps}")

    print("\n新分配:")
    for c, ps in result.items():
        print(f"  {c}: {ps}")

    print(f"\n统计: 移动 {moved} 个分区，保留 {kept} 个分区")

    return result


def demo_compare_range_assigner():
    """对比：Range分配器 vs 粘性分配器"""
    print("\n" + "=" * 70)
    print("场景4: 对比 - Range分配器 vs 粘性分配器")
    print("=" * 70)

    def range_assign(consumers, partitions):
        """简单的Range分配"""
        result = {}
        n = len(consumers)
        for i, c in enumerate(consumers):
            start = i * len(partitions) // n
            end = (i + 1) * len(partitions) // n
            result[c] = partitions[start:end]
        return result

    consumers_old = ["c1", "c2", "c3"]
    consumers_new = ["c1", "c2", "c3", "c4"]
    partitions = list(range(12))

    range_old = range_assign(consumers_old, partitions)
    range_new = range_assign(consumers_new, partitions)

    assigner = StickyPartitionAssigner()
    sticky_old = assigner.assign(consumers_old, partitions, {})
    sticky_new = assigner.assign(consumers_new, partitions, sticky_old)

    def count_moved(old, new):
        old_p = {p: c for c, ps in old.items() for p in ps}
        new_p = {p: c for c, ps in new.items() for p in ps}
        moved = sum(1 for p in old_p if old_p[p] != new_p.get(p))
        return moved

    range_moved = count_moved(range_old, range_new)
    sticky_moved = count_moved(sticky_old, sticky_new)

    print("\nRange分配器:")
    print(f"  原分配: {range_old}")
    print(f"  新分配: {range_new}")
    print(f"  移动分区: {range_moved}")

    print("\n粘性分配器:")
    print(f"  原分配: {sticky_old}")
    print(f"  新分配: {sticky_new}")
    print(f"  移动分区: {sticky_moved}")

    print(f"\n粘性分配器减少了 {range_moved - sticky_moved} 次分区移动!")


def main():
    print("Kafka粘性分区分配策略演示")
    print("=" * 70)

    assigner = StickyPartitionAssigner()

    result1 = demo_initial_assignment(assigner)
    result2 = demo_add_consumer(assigner, result1)
    demo_remove_consumer(assigner, result2)
    demo_compare_range_assigner()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

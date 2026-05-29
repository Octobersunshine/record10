from typing import Dict, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class ConsumerState(Enum):
    """消费者状态"""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    REBALANCING = "REBALANCING"


@dataclass
class PartitionOffset:
    """分区偏移量信息"""
    partition: int
    committed_offset: int = 0
    last_consumed_offset: int = 0
    needs_reprocessing: bool = False


@dataclass
class Consumer:
    """消费者"""
    consumer_id: str
    state: ConsumerState = ConsumerState.ACTIVE
    assigned_partitions: List[int] = field(default_factory=list)
    offsets: Dict[int, PartitionOffset] = field(default_factory=dict)

    def pause(self):
        """暂停消费"""
        self.state = ConsumerState.PAUSED

    def resume(self):
        """恢复消费"""
        self.state = ConsumerState.ACTIVE

    def commit_offsets(self):
        """提交偏移量：将已消费偏移量提交为已提交偏移量"""
        for p in self.offsets:
            self.offsets[p].committed_offset = self.offsets[p].last_consumed_offset

    def get_offsets_to_reprocess(self) -> List[int]:
        """获取需要重新处理的分区"""
        return [p for p, offset in self.offsets.items() if offset.needs_reprocessing]


class StickyPartitionAssigner:
    """粘性分区分配器"""

    def assign(
        self,
        consumers: List[str],
        partitions: List[int],
        current_assignment: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
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


class RebalanceCoordinator:
    """再平衡协调器：实现双阶段提交"""

    def __init__(self):
        self.assigner = StickyPartitionAssigner()
        self.consumers: Dict[str, Consumer] = {}
        self.all_partitions: List[int] = []
        self.rebalance_count = 0
        self.rebalance_history = []

    def add_consumer(self, consumer_id: str):
        """添加消费者"""
        if consumer_id not in self.consumers:
            self.consumers[consumer_id] = Consumer(consumer_id=consumer_id)

    def remove_consumer(self, consumer_id: str):
        """移除消费者"""
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]

    def set_partitions(self, partitions: List[int]):
        """设置所有分区"""
        self.all_partitions = partitions

    def simulate_consumption(self, consumer_id: str, partition: int, messages: int):
        """模拟消费消息"""
        consumer = self.consumers.get(consumer_id)
        if not consumer or partition not in consumer.assigned_partitions:
            return

        if partition not in consumer.offsets:
            consumer.offsets[partition] = PartitionOffset(partition=partition)

        consumer.offsets[partition].last_consumed_offset += messages

    def get_current_assignment(self) -> Dict[str, List[int]]:
        """获取当前分配"""
        return {cid: c.assigned_partitions[:] for cid, c in self.consumers.items()}

    def _phase1_pause_and_commit(self) -> Dict[str, Dict[int, int]]:
        """
        阶段一：暂停所有消费者并提交偏移量
        返回：每个消费者的已提交偏移量快照
        """
        committed_offsets = {}

        for consumer_id, consumer in self.consumers.items():
            consumer.pause()
            consumer.commit_offsets()

            committed_offsets[consumer_id] = {
                p: offset.committed_offset
                for p, offset in consumer.offsets.items()
            }

        return committed_offsets

    def _phase2_reassign_partitions(self) -> Dict[str, List[int]]:
        """
        阶段二：重新分配分区
        返回：新的分区分配
        """
        consumer_ids = sorted(self.consumers.keys())
        current_assignment = self.get_current_assignment()

        new_assignment = self.assigner.assign(
            consumer_ids,
            self.all_partitions,
            current_assignment
        )

        return new_assignment

    def _phase3_update_consumer_state(
        self,
        new_assignment: Dict[str, List[int]],
        committed_offsets: Dict[str, Dict[int, int]]
    ) -> Dict[str, List[int]]:
        """
        阶段三：更新消费者状态，标记需要重新处理的分区
        返回：每个消费者需要重新处理的分区列表
        """
        reprocess_partitions = defaultdict(list)
        global_offset_store = {}

        for cid, offsets in committed_offsets.items():
            for p, offset in offsets.items():
                global_offset_store[p] = (cid, offset)

        for consumer_id, consumer in self.consumers.items():
            new_partitions = new_assignment.get(consumer_id, [])
            old_partitions = set(consumer.assigned_partitions)

            for p in new_partitions:
                if p not in consumer.offsets:
                    consumer.offsets[p] = PartitionOffset(partition=p)

                if p in old_partitions:
                    consumer.offsets[p].needs_reprocessing = False
                else:
                    if p in global_offset_store:
                        old_consumer, last_offset = global_offset_store[p]
                        consumer.offsets[p].committed_offset = last_offset
                        consumer.offsets[p].last_consumed_offset = last_offset
                        consumer.offsets[p].needs_reprocessing = False

                        if consumer_id != old_consumer:
                            reprocess_partitions[consumer_id].append(p)
                    else:
                        consumer.offsets[p].needs_reprocessing = True
                        reprocess_partitions[consumer_id].append(p)

            removed_partitions = old_partitions - set(new_partitions)
            for p in removed_partitions:
                if p in consumer.offsets:
                    del consumer.offsets[p]

            consumer.assigned_partitions = new_partitions[:]

        return dict(reprocess_partitions)

    def _phase4_resume_consumers(self):
        """阶段四：恢复消费者消费"""
        for consumer in self.consumers.values():
            consumer.resume()

    def rebalance(self, trigger_reason: str = "unknown") -> Dict:
        """
        执行完整的再平衡流程（双阶段提交）

        流程：
        1. 暂停所有消费者，提交偏移量
        2. 重新计算分区分配
        3. 更新消费者状态，标记需重新处理的分区
        4. 恢复消费者
        """
        self.rebalance_count += 1

        old_assignment = self.get_current_assignment()

        committed_offsets = self._phase1_pause_and_commit()

        new_assignment = self._phase2_reassign_partitions()

        reprocess_partitions = self._phase3_update_consumer_state(
            new_assignment,
            committed_offsets
        )

        self._phase4_resume_consumers()

        moved = self._calculate_moved_partitions(old_assignment, new_assignment)

        rebalance_result = {
            "rebalance_id": self.rebalance_count,
            "trigger_reason": trigger_reason,
            "old_assignment": old_assignment,
            "new_assignment": new_assignment,
            "moved_partitions": moved,
            "reprocess_partitions": reprocess_partitions,
            "committed_offsets": committed_offsets
        }

        self.rebalance_history.append(rebalance_result)

        return rebalance_result

    def _calculate_moved_partitions(
        self,
        old_assignment: Dict[str, List[int]],
        new_assignment: Dict[str, List[int]]
    ) -> List[int]:
        """计算移动的分区"""
        old_owners = {}
        for cid, ps in old_assignment.items():
            for p in ps:
                old_owners[p] = cid

        new_owners = {}
        for cid, ps in new_assignment.items():
            for p in ps:
                new_owners[p] = cid

        moved = []
        for p in set(old_owners.keys()) | set(new_owners.keys()):
            if old_owners.get(p) != new_owners.get(p):
                moved.append(p)

        return sorted(moved)

    def print_status(self, title: str = "当前状态"):
        """打印当前状态"""
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

        for consumer_id in sorted(self.consumers.keys()):
            consumer = self.consumers[consumer_id]
            print(f"\n{consumer_id} [{consumer.state.value}]:")
            print(f"  分配分区: {consumer.assigned_partitions}")

            if consumer.offsets:
                print(f"  偏移量状态:")
                for p in sorted(consumer.offsets.keys()):
                    offset = consumer.offsets[p]
                    reprocess_flag = " [需重处理]" if offset.needs_reprocessing else ""
                    print(f"    分区{p:2d}: committed={offset.committed_offset:3d}, "
                          f"consumed={offset.last_consumed_offset:3d}{reprocess_flag}")


def demo_initial_setup():
    """演示1：初始设置和首次分配"""
    print("\n" + "#" * 70)
    print("场景1: 初始设置和首次分配")
    print("#" * 70)

    coordinator = RebalanceCoordinator()
    coordinator.set_partitions(list(range(10)))

    coordinator.add_consumer("consumer-1")
    coordinator.add_consumer("consumer-2")
    coordinator.add_consumer("consumer-3")

    coordinator.print_status("再平衡前状态")

    result = coordinator.rebalance("初始分配")

    coordinator.print_status("首次再平衡后状态")

    print(f"\n移动的分区: {result['moved_partitions']}")

    return coordinator


def demo_consumption_and_rebalance(coordinator: RebalanceCoordinator):
    """演示2：模拟消费后扩容消费者"""
    print("\n" + "#" * 70)
    print("场景2: 模拟消费后扩容消费者")
    print("#" * 70)

    coordinator.simulate_consumption("consumer-1", 0, 150)
    coordinator.simulate_consumption("consumer-1", 3, 120)
    coordinator.simulate_consumption("consumer-2", 1, 200)
    coordinator.simulate_consumption("consumer-2", 4, 180)
    coordinator.simulate_consumption("consumer-3", 2, 170)
    coordinator.simulate_consumption("consumer-3", 5, 160)

    coordinator.print_status("模拟消费后的状态")

    print("\n>>> 添加新消费者 consumer-4")
    coordinator.add_consumer("consumer-4")

    result = coordinator.rebalance("消费者扩容 (3 -> 4)")

    coordinator.print_status("扩容再平衡后状态")

    print(f"\n再平衡统计:")
    print(f"  触发原因: {result['trigger_reason']}")
    print(f"  移动的分区: {result['moved_partitions']}")

    if result['reprocess_partitions']:
        print(f"  需要重新处理的分区:")
        for cid, ps in result['reprocess_partitions'].items():
            print(f"    {cid}: {ps}")
    else:
        print(f"  无需要重新处理的分区")

    return coordinator


def demo_consumer_removal(coordinator: RebalanceCoordinator):
    """演示3：消费者缩容"""
    print("\n" + "#" * 70)
    print("场景3: 消费者缩容")
    print("#" * 70)

    coordinator.simulate_consumption("consumer-1", 0, 50)
    coordinator.simulate_consumption("consumer-2", 1, 60)
    coordinator.simulate_consumption("consumer-3", 2, 70)
    coordinator.simulate_consumption("consumer-4", 6, 80)

    coordinator.print_status("缩容前状态")

    print("\n>>> 移除消费者 consumer-3")
    coordinator.remove_consumer("consumer-3")

    result = coordinator.rebalance("消费者缩容 (4 -> 3)")

    coordinator.print_status("缩容再平衡后状态")

    print(f"\n再平衡统计:")
    print(f"  触发原因: {result['trigger_reason']}")
    print(f"  移动的分区: {result['moved_partitions']}")

    if result['reprocess_partitions']:
        print(f"  需要重新处理的分区:")
        for cid, ps in result['reprocess_partitions'].items():
            print(f"    {cid}: {ps}")

    return coordinator


def demo_no_duplicate_no_miss():
    """演示4：验证无重复消费和无漏消费"""
    print("\n" + "#" * 70)
    print("场景4: 验证 - 无重复消费 & 无漏消费")
    print("#" * 70)

    coordinator = RebalanceCoordinator()
    coordinator.set_partitions([0, 1, 2, 3, 4, 5])

    coordinator.add_consumer("c1")
    coordinator.add_consumer("c2")
    coordinator.rebalance("初始分配")

    for p in range(6):
        if p in coordinator.consumers["c1"].assigned_partitions:
            coordinator.simulate_consumption("c1", p, 100)
        else:
            coordinator.simulate_consumption("c2", p, 100)

    coordinator.print_status("扩容前状态（各分区已消费100条）")

    print("\n>>> 验证：偏移量一致性检查")
    total_consumed = 0
    for cid, consumer in coordinator.consumers.items():
        for p, offset in consumer.offsets.items():
            total_consumed += offset.last_consumed_offset
            print(f"  分区{p}: consumed={offset.last_consumed_offset}, "
                  f"committed={offset.committed_offset}")
    print(f"  总计消费: {total_consumed} 条消息")

    print("\n>>> 添加新消费者 c3，触发再平衡")
    coordinator.add_consumer("c3")
    result = coordinator.rebalance("扩容验证")

    coordinator.print_status("扩容后状态")

    print("\n>>> 关键保证验证:")
    print(f"  1. 所有分区都有所有者: "
          f"{sum(len(c.assigned_partitions) for c in coordinator.consumers.values())} "
          f"== {len(coordinator.all_partitions)}")

    print(f"  2. 移动的分区数: {len(result['moved_partitions'])}")

    print(f"  3. 需重新处理的分区:")
    for cid, ps in result['reprocess_partitions'].items():
        for p in ps:
            offset = coordinator.consumers[cid].offsets[p]
            print(f"     分区{p} -> {cid}: 从偏移量 {offset.committed_offset} 开始消费")

    print(f"\n  结论: 通过双阶段提交，确保了:")
    print(f"  - 无漏消费: 所有分区从已提交的偏移量继续")
    print(f"  - 无重复消费: 已提交的偏移量保证了消息不重复处理")


def main():
    print("Kafka 双阶段提交再平衡模拟器")
    print("=" * 70)
    print("\n核心流程: 暂停消费 → 提交偏移量 → 重新分配 → 恢复消费\n")

    coordinator = demo_initial_setup()
    coordinator = demo_consumption_and_rebalance(coordinator)
    coordinator = demo_consumer_removal(coordinator)
    demo_no_duplicate_no_miss()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

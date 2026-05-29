from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import time


class ConsumerState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    REBALANCING = "REBALANCING"


class RebalanceType(Enum):
    STOP_THE_WORLD = "STOP_THE_WORLD"
    INCREMENTAL = "INCREMENTAL"


@dataclass
class PartitionOffset:
    partition: int
    committed_offset: int = 0
    last_consumed_offset: int = 0
    needs_reprocessing: bool = False


@dataclass
class Consumer:
    consumer_id: str
    state: ConsumerState = ConsumerState.ACTIVE
    assigned_partitions: List[int] = field(default_factory=list)
    offsets: Dict[int, PartitionOffset] = field(default_factory=dict)

    def pause(self):
        self.state = ConsumerState.PAUSED

    def resume(self):
        self.state = ConsumerState.ACTIVE

    def commit_offsets(self):
        for p in self.offsets:
            self.offsets[p].committed_offset = self.offsets[p].last_consumed_offset


@dataclass
class RebalanceImpact:
    """再平衡影响评估"""
    rebalance_type: RebalanceType
    total_partitions: int
    affected_partitions: List[int]
    moved_partitions: List[int]
    paused_consumers: List[str]
    estimated_pause_time_ms: float
    rounds_completed: int = 1
    impact_score: float = 0.0

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"  再平衡影响评估报告")
        print(f"{'='*70}")
        print(f"  类型: {self.rebalance_type.value}")
        print(f"  总分区数: {self.total_partitions}")
        print(f"  受影响分区: {len(self.affected_partitions)} ({self.affected_partitions})")
        print(f"  移动分区数: {len(self.moved_partitions)} ({self.moved_partitions})")
        print(f"  暂停消费者数: {len(self.paused_consumers)}")
        print(f"  预计停顿时间: ~{self.estimated_pause_time_ms:.1f} ms")
        print(f"  再平衡轮次: {self.rounds_completed}")
        print(f"  影响评分: {self.impact_score:.2f} (越低越好)")
        print(f"{'='*70}")


class StickyPartitionAssigner:
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


class IncrementalRebalanceCoordinator:
    """增量再平衡协调器：支持停服式和协作式两种模式"""

    def __init__(self, max_move_per_round: int = 2):
        self.assigner = StickyPartitionAssigner()
        self.consumers: Dict[str, Consumer] = {}
        self.all_partitions: List[int] = []
        self.rebalance_history = []
        self.max_move_per_round = max_move_per_round

    def add_consumer(self, consumer_id: str):
        if consumer_id not in self.consumers:
            self.consumers[consumer_id] = Consumer(consumer_id=consumer_id)

    def remove_consumer(self, consumer_id: str):
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]

    def set_partitions(self, partitions: List[int]):
        self.all_partitions = partitions

    def simulate_consumption(self, consumer_id: str, partition: int, messages: int):
        consumer = self.consumers.get(consumer_id)
        if not consumer or partition not in consumer.assigned_partitions:
            return
        if partition not in consumer.offsets:
            consumer.offsets[partition] = PartitionOffset(partition=partition)
        consumer.offsets[partition].last_consumed_offset += messages

    def get_current_assignment(self) -> Dict[str, List[int]]:
        return {cid: c.assigned_partitions[:] for cid, c in self.consumers.items()}

    def _get_ideal_assignment(self) -> Dict[str, List[int]]:
        """获取理想的最终分配状态"""
        consumer_ids = sorted(self.consumers.keys())
        return self.assigner.assign(
            consumer_ids,
            self.all_partitions,
            self.get_current_assignment()
        )

    def _calculate_moves_needed(
        self,
        current: Dict[str, List[int]],
        ideal: Dict[str, List[int]]
    ) -> List[Tuple[int, str, str]]:
        """计算需要移动的分区列表: [(partition, from_consumer, to_consumer)]"""
        current_owners = {}
        for cid, ps in current.items():
            for p in ps:
                current_owners[p] = cid

        ideal_owners = {}
        for cid, ps in ideal.items():
            for p in ps:
                ideal_owners[p] = cid

        moves = []
        for p in self.all_partitions:
            if current_owners.get(p) != ideal_owners.get(p):
                moves.append((p, current_owners.get(p, ""), ideal_owners.get(p, "")))

        return moves

    def _execute_single_move(
        self,
        partition: int,
        from_consumer: str,
        to_consumer: str
    ) -> float:
        """执行单个分区移动，返回暂停时间（毫秒）"""
        start_time = time.time()

        if from_consumer and from_consumer in self.consumers:
            self.consumers[from_consumer].pause()
            self.consumers[from_consumer].commit_offsets()

        if to_consumer and to_consumer in self.consumers:
            self.consumers[to_consumer].pause()

        offset = None
        if from_consumer and from_consumer in self.consumers:
            if partition in self.consumers[from_consumer].assigned_partitions:
                self.consumers[from_consumer].assigned_partitions.remove(partition)
            if partition in self.consumers[from_consumer].offsets:
                offset = self.consumers[from_consumer].offsets.pop(partition)

        if to_consumer and to_consumer in self.consumers:
            if partition not in self.consumers[to_consumer].assigned_partitions:
                self.consumers[to_consumer].assigned_partitions.append(partition)
                self.consumers[to_consumer].assigned_partitions.sort()
            if offset:
                self.consumers[to_consumer].offsets[partition] = PartitionOffset(
                    partition=partition,
                    committed_offset=offset.committed_offset,
                    last_consumed_offset=offset.committed_offset,
                    needs_reprocessing=True
                )
            else:
                self.consumers[to_consumer].offsets[partition] = PartitionOffset(
                    partition=partition,
                    needs_reprocessing=True
                )

        if from_consumer and from_consumer in self.consumers:
            self.consumers[from_consumer].resume()
        if to_consumer and to_consumer in self.consumers:
            self.consumers[to_consumer].resume()

        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms

    def rebalance_stop_the_world(self, trigger_reason: str = "unknown") -> RebalanceImpact:
        """
        停服式再平衡（Stop-the-World）
        一次性暂停所有消费者，完成所有分区移动后恢复
        """
        start_time = time.time()

        old_assignment = self.get_current_assignment()
        ideal_assignment = self._get_ideal_assignment()
        moves = self._calculate_moves_needed(old_assignment, ideal_assignment)

        for consumer in self.consumers.values():
            consumer.pause()
            consumer.commit_offsets()

        for cid in self.consumers:
            self.consumers[cid].assigned_partitions = ideal_assignment.get(cid, [])
            self.consumers[cid].offsets = {}
            for p in self.consumers[cid].assigned_partitions:
                self.consumers[cid].offsets[p] = PartitionOffset(
                    partition=p,
                    needs_reprocessing=True
                )

        for consumer in self.consumers.values():
            consumer.resume()

        total_time_ms = (time.time() - start_time) * 1000

        moved_partitions = [p for p, _, _ in moves]
        impact = RebalanceImpact(
            rebalance_type=RebalanceType.STOP_THE_WORLD,
            total_partitions=len(self.all_partitions),
            affected_partitions=moved_partitions,
            moved_partitions=moved_partitions,
            paused_consumers=list(self.consumers.keys()),
            estimated_pause_time_ms=total_time_ms,
            rounds_completed=1,
            impact_score=len(moved_partitions) * len(self.consumers) * total_time_ms / 1000
        )

        self.rebalance_history.append({
            "type": "STOP_THE_WORLD",
            "trigger": trigger_reason,
            "impact": impact
        })

        return impact

    def rebalance_incremental(self, trigger_reason: str = "unknown") -> RebalanceImpact:
        """
        协作式增量再平衡（Incremental）
        多轮执行，每轮只移动少量分区，减少单次停顿时间
        """
        total_start_time = time.time()
        total_pause_time_ms = 0.0
        all_moved_partitions = []
        rounds = 0

        ideal_assignment = self._get_ideal_assignment()
        max_rounds = 100

        while rounds < max_rounds:
            rounds += 1
            current_assignment = self.get_current_assignment()
            moves = self._calculate_moves_needed(current_assignment, ideal_assignment)

            if not moves:
                break

            moves_this_round = moves[:self.max_move_per_round]

            for partition, from_c, to_c in moves_this_round:
                pause_time = self._execute_single_move(partition, from_c, to_c)
                total_pause_time_ms += pause_time
                all_moved_partitions.append(partition)

        total_time_ms = (time.time() - total_start_time) * 1000

        unique_moved = sorted(list(set(all_moved_partitions)))
        affected_consumers = set()
        for p in unique_moved:
            for cid, consumer in self.consumers.items():
                if p in consumer.assigned_partitions or p in consumer.offsets:
                    affected_consumers.add(cid)

        impact = RebalanceImpact(
            rebalance_type=RebalanceType.INCREMENTAL,
            total_partitions=len(self.all_partitions),
            affected_partitions=unique_moved,
            moved_partitions=unique_moved,
            paused_consumers=sorted(list(affected_consumers)),
            estimated_pause_time_ms=total_pause_time_ms,
            rounds_completed=rounds,
            impact_score=len(unique_moved) * len(affected_consumers) * total_pause_time_ms / 1000 / rounds
        )

        self.rebalance_history.append({
            "type": "INCREMENTAL",
            "trigger": trigger_reason,
            "impact": impact
        })

        return impact

    def print_status(self, title: str = "当前状态"):
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


def create_coordinator_with_state() -> IncrementalRebalanceCoordinator:
    """创建一个有初始状态的协调器"""
    coordinator = IncrementalRebalanceCoordinator(max_move_per_round=2)
    coordinator.set_partitions(list(range(20)))

    for i in range(1, 5):
        coordinator.add_consumer(f"consumer-{i}")

    coordinator.rebalance_stop_the_world("初始分配")

    for cid, consumer in coordinator.consumers.items():
        for p in consumer.assigned_partitions:
            coordinator.simulate_consumption(cid, p, 100 + p * 10)

    return coordinator


def demo_stop_the_world_vs_incremental():
    """演示：停服式 vs 增量式再平衡对比"""
    print("\n" + "#" * 70)
    print("对比演示: 停服式(Stop-the-World) vs 增量式(Incremental)再平衡")
    print("#" * 70)

    print("\n>>> 初始化协调器，4个消费者，20个分区")
    coordinator_stw = create_coordinator_with_state()
    coordinator_stw.print_status("初始状态")

    print("\n>>> 添加2个新消费者，触发再平衡")
    coordinator_stw.add_consumer("consumer-5")
    coordinator_stw.add_consumer("consumer-6")

    print("\n" + "=" * 70)
    print("方案A: 停服式再平衡")
    print("=" * 70)
    impact_stw = coordinator_stw.rebalance_stop_the_world("消费者扩容 4->6")
    impact_stw.print_report()

    print("\n" + "=" * 70)
    print("方案B: 增量式再平衡")
    print("=" * 70)

    coordinator_inc = create_coordinator_with_state()
    coordinator_inc.add_consumer("consumer-5")
    coordinator_inc.add_consumer("consumer-6")

    impact_inc = coordinator_inc.rebalance_incremental("消费者扩容 4->6")
    impact_inc.print_report()

    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"  移动分区数: 相同 ({len(impact_stw.moved_partitions)} 个)")
    print(f"  停顿方式:")
    print(f"    - 停服式: 一次性长时间停顿 ({impact_stw.estimated_pause_time_ms:.1f}ms)")
    print(f"    - 增量式: {impact_inc.rounds_completed} 次短停顿 (总计 {impact_inc.estimated_pause_time_ms:.1f}ms)")
    print(f"  影响评分:")
    print(f"    - 停服式: {impact_stw.impact_score:.2f}")
    print(f"    - 增量式: {impact_inc.impact_score:.2f}")
    reduction = ((impact_stw.impact_score - impact_inc.impact_score) / impact_stw.impact_score * 100)
    print(f"  优势: 增量式影响评分降低 {reduction:.1f}%")

    return coordinator_stw, coordinator_inc


def demo_incremental_round_details():
    """演示：增量再平衡的每轮细节"""
    print("\n\n" + "#" * 70)
    print("增量再平衡轮次详情演示")
    print("#" * 70)

    coordinator = IncrementalRebalanceCoordinator(max_move_per_round=2)
    coordinator.set_partitions(list(range(10)))

    for i in range(1, 3):
        coordinator.add_consumer(f"c{i}")
    coordinator.rebalance_stop_the_world("初始分配 2个消费者")

    coordinator.print_status("初始状态 (2个消费者)")

    print("\n>>> 添加3个新消费者，观察增量再平衡过程")
    for i in range(3, 6):
        coordinator.add_consumer(f"c{i}")

    ideal = coordinator._get_ideal_assignment()
    print(f"\n目标分配:")
    for cid, ps in ideal.items():
        print(f"  {cid}: {ps}")

    print(f"\n开始增量再平衡 (每轮最多移动 {coordinator.max_move_per_round} 个分区)...")

    round_num = 0
    while True:
        round_num += 1
        current = coordinator.get_current_assignment()
        moves = coordinator._calculate_moves_needed(current, ideal)

        if not moves:
            print(f"\n第 {round_num} 轮: 无需移动，再平衡完成!")
            break

        print(f"\n第 {round_num} 轮: 需要移动 {len(moves)} 个分区")
        for p, from_c, to_c in moves[:coordinator.max_move_per_round]:
            print(f"  移动分区 {p}: {from_c or '无'} -> {to_c or '无'}")

        for p, from_c, to_c in moves[:coordinator.max_move_per_round]:
            coordinator._execute_single_move(p, from_c, to_c)

        coordinator.print_status(f"第 {round_num} 轮后状态")

    coordinator.print_status("最终状态")


def demo_impact_analysis():
    """演示：再平衡影响分析"""
    print("\n\n" + "#" * 70)
    print("再平衡影响分析")
    print("#" * 70)

    scenarios = [
        ("小扩容", 4, 5, list(range(16))),
        ("大扩容", 2, 6, list(range(24))),
        ("小缩容", 5, 3, list(range(15))),
        ("大缩容", 8, 2, list(range(24))),
    ]

    print(f"\n{'场景':<10} {'变化':<10} {'类型':<15} {'移动分区':<10} {'停顿时间(ms)':<15} {'影响评分':<10}")
    print("-" * 70)

    for name, old_count, new_count, partitions in scenarios:
        for rebalance_type in ["STW", "INC"]:
            coordinator = IncrementalRebalanceCoordinator(max_move_per_round=2)
            coordinator.set_partitions(partitions)

            for i in range(1, old_count + 1):
                coordinator.add_consumer(f"c{i}")
            coordinator.rebalance_stop_the_world("初始")

            if new_count > old_count:
                for i in range(old_count + 1, new_count + 1):
                    coordinator.add_consumer(f"c{i}")
            else:
                for i in range(new_count + 1, old_count + 1):
                    coordinator.remove_consumer(f"c{i}")

            if rebalance_type == "STW":
                impact = coordinator.rebalance_stop_the_world(f"{name}")
            else:
                impact = coordinator.rebalance_incremental(f"{name}")

            type_label = "停服式" if rebalance_type == "STW" else "增量式"
            change_label = f"{old_count}->{new_count}"
            print(f"{name:<10} {change_label:<10} {type_label:<15} "
                  f"{len(impact.moved_partitions):<10} "
                  f"{impact.estimated_pause_time_ms:<15.1f} "
                  f"{impact.impact_score:<10.2f}")


def main():
    print("Kafka 协作式增量再平衡模拟器")
    print("=" * 70)
    print("\n核心特性:")
    print("  1. 停服式再平衡（Stop-the-World）")
    print("  2. 协作式增量再平衡（多轮少量移动）")
    print("  3. 再平衡影响评估报告")
    print("  4. 两种模式对比分析\n")

    demo_stop_the_world_vs_incremental()
    demo_incremental_round_details()
    demo_impact_analysis()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

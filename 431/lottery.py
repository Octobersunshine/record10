import random
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable


class Lottery:
    def __init__(
        self,
        prizes: List[Dict],
        default_equal_weight: bool = True,
        precision: int = 6,
        pity_enabled: bool = False,
        pity_threshold: int = 10,
        grand_prize_ids: Optional[List] = None,
        with_replacement: bool = True,
        weight_modifier: Optional[Callable[[Dict], float]] = None
    ):
        self.original_prizes = [p.copy() for p in prizes]
        self.prizes = [p.copy() for p in prizes]
        self.default_equal_weight = default_equal_weight
        self.precision = precision
        self.pity_enabled = pity_enabled
        self.pity_threshold = pity_threshold
        self.grand_prize_ids = grand_prize_ids or []
        self.with_replacement = with_replacement
        self.weight_modifier = weight_modifier
        self.consecutive_without_grand = 0
        self.total_draws = 0
        self._validate_prizes()
        self._ensure_prize_counts()
        self._apply_weight_modifier()
        self._convert_to_integer_weights()
        self._calculate_cumulative_weights()

    def _validate_prizes(self):
        if not self.prizes:
            raise ValueError("奖品列表不能为空")
        
        required_fields = ['id', 'name', 'weight']
        for prize in self.prizes:
            for field in required_fields:
                if field not in prize:
                    raise ValueError(f"奖品缺少必要字段: {field}")
            if not isinstance(prize['weight'], (int, float)) or prize['weight'] < 0:
                raise ValueError(f"奖品权重必须为非负数: {prize}")
        
        total_weight = sum(p['weight'] for p in self.prizes)
        if total_weight <= 0:
            if self.default_equal_weight:
                equal_weight = 1.0 / len(self.prizes)
                for prize in self.prizes:
                    prize['weight'] = equal_weight
            else:
                raise ValueError("所有奖品的权重之和必须大于0")

    def _ensure_prize_counts(self):
        for prize in self.prizes:
            if 'count' not in prize:
                prize['count'] = 1

    def _apply_weight_modifier(self):
        if self.weight_modifier:
            for prize in self.prizes:
                original_weight = prize['weight']
                modified_weight = self.weight_modifier(prize)
                if modified_weight < 0:
                    modified_weight = 0
                prize['weight'] = modified_weight

    def _convert_to_integer_weights(self):
        self.integer_weights = []
        multiplier = 10 ** self.precision
        for prize in self.prizes:
            weight = prize['weight'] if prize.get('count', 1) > 0 else 0
            int_weight = int(round(weight * multiplier))
            self.integer_weights.append(int_weight)

    def _calculate_cumulative_weights(self):
        self.cumulative_weights = []
        cumulative = 0
        for int_weight in self.integer_weights:
            cumulative += int_weight
            self.cumulative_weights.append(cumulative)
        self.total_weight = cumulative
        if self.total_weight <= 0:
            if self.default_equal_weight:
                available_count = sum(1 for p in self.prizes if p.get('count', 1) > 0)
                if available_count > 0:
                    equal_int = 10 ** self.precision
                    self.integer_weights = []
                    for prize in self.prizes:
                        self.integer_weights.append(equal_int if prize.get('count', 1) > 0 else 0)
                    self.cumulative_weights = []
                    cumulative = 0
                    for int_weight in self.integer_weights:
                        cumulative += int_weight
                        self.cumulative_weights.append(cumulative)
                    self.total_weight = cumulative
            else:
                raise ValueError("所有奖品的整数权重之和必须大于0")

    def _is_grand_prize(self, prize: Dict) -> bool:
        if not self.grand_prize_ids:
            return False
        return prize['id'] in self.grand_prize_ids

    def _draw_pity_prize(self) -> Dict:
        grand_prizes = [
            p for p in self.prizes 
            if p['id'] in self.grand_prize_ids and p.get('count', 1) > 0
        ]
        if grand_prizes:
            prize = random.choice(grand_prizes)
            return {'id': prize['id'], 'name': prize['name']}
        
        available_prizes = [p for p in self.prizes if p.get('count', 1) > 0]
        if available_prizes:
            prize = max(available_prizes, key=lambda p: p['weight'])
            return {'id': prize['id'], 'name': prize['name']}
        
        return {'id': self.prizes[0]['id'], 'name': self.prizes[0]['name']}

    def draw_one(self) -> Dict:
        if self.total_weight <= 0:
            available_prizes = [p for p in self.prizes if p.get('count', 1) > 0]
            if not available_prizes:
                raise ValueError("没有可抽取的奖品")
            prize = random.choice(available_prizes)
            result = {'id': prize['id'], 'name': prize['name']}
        else:
            if self.pity_enabled and self.consecutive_without_grand >= self.pity_threshold:
                result = self._draw_pity_prize()
            else:
                rand = random.randint(0, self.total_weight - 1)
                selected_index = len(self.prizes) - 1
                for i, cum_weight in enumerate(self.cumulative_weights):
                    if rand < cum_weight:
                        selected_index = i
                        break
                prize = self.prizes[selected_index]
                result = {'id': prize['id'], 'name': prize['name']}
        
        self.total_draws += 1
        
        prize_in_list = next((p for p in self.prizes if p['id'] == result['id']), None)
        is_grand = self._is_grand_prize(prize_in_list) if prize_in_list else False
        
        if is_grand:
            self.consecutive_without_grand = 0
        else:
            self.consecutive_without_grand += 1
        
        if not self.with_replacement and prize_in_list:
            prize_in_list['count'] = prize_in_list.get('count', 1) - 1
            self._convert_to_integer_weights()
            self._calculate_cumulative_weights()
        
        return result

    def draw_many(self, count: int, unique: bool = False) -> List[Dict]:
        if count <= 0:
            raise ValueError("抽取数量必须大于0")
        
        results = []
        
        if unique or not self.with_replacement:
            original_with_replacement = self.with_replacement
            self.with_replacement = False
            
            try:
                for _ in range(count):
                    available = sum(1 for p in self.prizes if p.get('count', 1) > 0)
                    if available <= 0:
                        raise ValueError("没有足够的奖品可抽取")
                    results.append(self.draw_one())
            finally:
                self.with_replacement = original_with_replacement
                if original_with_replacement:
                    self.prizes = [p.copy() for p in self.original_prizes]
                    self._validate_prizes()
                    self._ensure_prize_counts()
                    self._apply_weight_modifier()
                    self._convert_to_integer_weights()
                    self._calculate_cumulative_weights()
        else:
            for _ in range(count):
                results.append(self.draw_one())
        
        return results

    def reset_pity(self):
        self.consecutive_without_grand = 0

    def get_state(self) -> Dict:
        return {
            'total_draws': self.total_draws,
            'consecutive_without_grand': self.consecutive_without_grand,
            'pity_enabled': self.pity_enabled,
            'pity_threshold': self.pity_threshold,
            'with_replacement': self.with_replacement,
            'prize_counts': {p['id']: p.get('count', 1) for p in self.prizes}
        }

    def update_weight_modifier(self, weight_modifier: Optional[Callable[[Dict], float]]):
        self.weight_modifier = weight_modifier
        self.prizes = [p.copy() for p in self.original_prizes]
        self._validate_prizes()
        self._ensure_prize_counts()
        self._apply_weight_modifier()
        self._convert_to_integer_weights()
        self._calculate_cumulative_weights()


def draw_prize(
    prizes: List[Dict],
    count: int = 1,
    unique: bool = False,
    default_equal_weight: bool = True,
    precision: int = 6,
    pity_enabled: bool = False,
    pity_threshold: int = 10,
    grand_prize_ids: Optional[List] = None,
    with_replacement: bool = True,
    weight_modifier: Optional[Callable[[Dict], float]] = None
) -> List[Dict]:
    lottery = Lottery(
        prizes,
        default_equal_weight=default_equal_weight,
        precision=precision,
        pity_enabled=pity_enabled,
        pity_threshold=pity_threshold,
        grand_prize_ids=grand_prize_ids,
        with_replacement=with_replacement,
        weight_modifier=weight_modifier
    )
    if count == 1:
        return [lottery.draw_one()]
    return lottery.draw_many(count, unique)


if __name__ == '__main__':
    prizes = [
        {'id': 1, 'name': 'iPhone 15', 'weight': 1, 'count': 3},
        {'id': 2, 'name': 'iPad Air', 'weight': 3, 'count': 5},
        {'id': 3, 'name': 'AirPods Pro', 'weight': 6, 'count': 10},
        {'id': 4, 'name': '100元优惠券', 'weight': 20, 'count': 100},
        {'id': 5, 'name': '10元优惠券', 'weight': 70, 'count': 1000},
    ]

    print("=== 测试保底机制 ===")
    lottery_pity = Lottery(
        prizes,
        pity_enabled=True,
        pity_threshold=5,
        grand_prize_ids=[1, 2]
    )
    print(f"大奖ID: {lottery_pity.grand_prize_ids}")
    print(f"保底阈值: {lottery_pity.pity_threshold}次")
    grand_count = 0
    for i in range(20):
        result = lottery_pity.draw_one()
        is_grand = result['id'] in lottery_pity.grand_prize_ids
        if is_grand:
            grand_count += 1
        state = lottery_pity.get_state()
        print(f"第{i+1:2d}次: {result['name']:<15} 连续未中大奖: {state['consecutive_without_grand']} {'★大奖!' if is_grand else ''}")
    print(f"20次抽中大奖 {grand_count} 次\n")

    print("=== 测试不放回抽奖 ===")
    lottery_no_replace = Lottery(
        [{'id': i, 'name': f'奖品{i}', 'weight': 1, 'count': 2} for i in range(1, 4)],
        with_replacement=False
    )
    print("奖品初始状态:", {p['name']: p['count'] for p in lottery_no_replace.prizes})
    for i in range(6):
        result = lottery_no_replace.draw_one()
        state = lottery_no_replace.prizes
        print(f"第{i+1}次抽中: {result['name']}, 剩余: { {p['name']: p['count'] for p in state} }")
    try:
        lottery_no_replace.draw_one()
    except ValueError as e:
        print(f"第7次: {e}\n")

    print("=== 测试概率动态变化（活动期间大奖概率翻倍） ===")
    normal_prizes = [
        {'id': 1, 'name': '★5星角色', 'weight': 1, 'rarity': 5},
        {'id': 2, 'name': '☆4星角色', 'weight': 10, 'rarity': 4},
        {'id': 3, 'name': '3星角色', 'weight': 89, 'rarity': 3},
    ]
    
    def normal_weight(prize: Dict) -> float:
        return prize['weight']
    
    def event_weight(prize: Dict) -> float:
        if prize.get('rarity') == 5:
            return prize['weight'] * 2
        return prize['weight']
    
    lottery_dynamic = Lottery(normal_prizes, weight_modifier=normal_weight)
    
    print("正常概率:")
    stats_normal = {p['id']: {'name': p['name'], 'count': 0, 'weight': p['weight']} for p in normal_prizes}
    total = 10000
    for _ in range(total):
        r = lottery_dynamic.draw_one()
        stats_normal[r['id']]['count'] += 1
    
    total_weight = sum(p['weight'] for p in normal_prizes)
    print(f"{'奖品':<15} {'权重占比':>10} {'实际占比':>10} {'差距':>10}")
    print("-" * 50)
    for prize_id, data in stats_normal.items():
        expected = data['weight'] / total_weight * 100
        actual = data['count'] / total * 100
        diff = actual - expected
        print(f"{data['name']:<15} {expected:>9.2f}% {actual:>9.2f}% {diff:>+9.2f}%")
    
    print("\n活动期间（5星概率翻倍）:")
    lottery_dynamic.update_weight_modifier(event_weight)
    stats_event = {p['id']: {'name': p['name'], 'count': 0} for p in normal_prizes}
    for _ in range(total):
        r = lottery_dynamic.draw_one()
        stats_event[r['id']]['count'] += 1
    
    event_total_weight = sum(event_weight(p) for p in normal_prizes)
    print(f"{'奖品':<15} {'权重占比':>10} {'实际占比':>10} {'差距':>10}")
    print("-" * 50)
    for prize_id, data in stats_event.items():
        prize = next(p for p in normal_prizes if p['id'] == prize_id)
        expected = event_weight(prize) / event_total_weight * 100
        actual = data['count'] / total * 100
        diff = actual - expected
        print(f"{data['name']:<15} {expected:>9.2f}% {actual:>9.2f}% {diff:>+9.2f}%")
    print()

    print("=== 测试保底+不放回+动态概率组合使用 ===")
    gacha_prizes = [
        {'id': 1, 'name': '★5星限定角色', 'weight': 0.6, 'rarity': 5, 'count': 1},
        {'id': 2, 'name': '★5星常驻角色', 'weight': 0.6, 'rarity': 5, 'count': 5},
        {'id': 3, 'name': '☆4星武器', 'weight': 5.1, 'rarity': 4, 'count': 20},
        {'id': 4, 'name': '☆4星角色', 'weight': 5.1, 'rarity': 4, 'count': 30},
        {'id': 5, 'name': '3星材料', 'weight': 88.6, 'rarity': 3, 'count': 1000},
    ]
    
    def event_gacha_weight(prize: Dict) -> float:
        if prize['name'] == '★5星限定角色':
            return prize['weight'] * 3
        return prize['weight']
    
    gacha = Lottery(
        gacha_prizes,
        pity_enabled=True,
        pity_threshold=90,
        grand_prize_ids=[1, 2],
        with_replacement=False,
        weight_modifier=event_gacha_weight
    )
    
    print("模拟抽卡（目标：5星限定角色）")
    pulls = 0
    got_limited = False
    for i in range(200):
        result = gacha.draw_one()
        pulls = i + 1
        if result['id'] == 1:
            got_limited = True
            print(f"第{pulls:3d}抽: {result['name']} ✓ 抽到限定！")
            break
        if result['id'] in [1, 2]:
            print(f"第{pulls:3d}抽: {result['name']} ★5星")
        elif gacha.get_state()['consecutive_without_grand'] % 10 == 0:
            print(f"第{pulls:3d}抽: {result['name']} (连续{int(gacha.get_state()['consecutive_without_grand'])}抽未出5星)")
    
    if not got_limited:
        print("200抽仍未抽到限定角色...")
    print(f"总抽取次数: {pulls}")
    print(f"当前状态: {gacha.get_state()['prize_counts']}\n")

    print("=== 所有功能测试完成 ===")

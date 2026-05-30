def josephus_variant(n, step_sequence=None, start_index=0):
    people = list(range(1, n + 1))
    removed_order = []
    stats = []
    idx = start_index
    round_num = 0

    while len(people) > 1:
        remaining = len(people)

        if step_sequence is not None:
            step = step_sequence[round_num % len(step_sequence)]
        else:
            step = round_num + 1

        effective_step = step % remaining
        if effective_step == 0:
            effective_step = remaining

        idx = (idx + effective_step - 1) % remaining
        removed_person = people.pop(idx)
        removed_order.append(removed_person)

        stats.append({
            'round': round_num + 1,
            'step_value': step,
            'removed_person': removed_person,
            'removed_position': idx + 1,
            'remaining_count': len(people)
        })

        round_num += 1

    return removed_order, people[0], stats


if __name__ == "__main__":
    n = int(input("请输入人数: "))

    step_input = input("请输入步长序列（逗号分隔，回车使用默认递增步长）: ").strip()
    if step_input:
        step_sequence = [int(s.strip()) for s in step_input.split(',')]
        start_index = int(input("请输入起始索引（从0开始，默认0）: ") or "0")
        removed, survivor, stats = josephus_variant(n, step_sequence, start_index)
    else:
        removed, survivor, stats = josephus_variant(n)

    print("\n" + "=" * 60)
    print("详细统计:")
    print("-" * 60)
    print(f"{'轮次':<6}{'步长值':<8}{'被移除者':<10}{'移除位置':<10}{'剩余人数':<10}")
    print("-" * 60)
    for stat in stats:
        print(f"{stat['round']:<6}{stat['step_value']:<8}{stat['removed_person']:<10}"
              f"{stat['removed_position']:<10}{stat['remaining_count']:<10}")
    print("=" * 60)
    print(f"移除顺序: {removed}")
    print(f"最后剩下的人: {survivor}")

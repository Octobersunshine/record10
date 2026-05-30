import time


def hanoi(n, source='A', target='C', auxiliary='B'):
    if n <= 0:
        return [], 0

    steps = []

    def move(k, src, tgt, aux):
        if k == 1:
            steps.append(f"{src}→{tgt}")
            return
        move(k - 1, src, aux, tgt)
        steps.append(f"{src}→{tgt}")
        move(k - 1, aux, tgt, src)

    move(n, source, target, auxiliary)
    return steps, len(steps)


def hanoi_iterative(n, source='A', target='C', auxiliary='B'):
    if n <= 0:
        return [], 0

    pegs = {
        source: list(range(n, 0, -1)),
        auxiliary: [],
        target: []
    }
    steps = []
    names = [source, auxiliary, target] if n % 2 == 0 else [source, target, auxiliary]

    def move_between(a, b):
        if not pegs[a]:
            pegs[a].append(pegs[b].pop())
            steps.append(f"{b}→{a}")
        elif not pegs[b]:
            pegs[b].append(pegs[a].pop())
            steps.append(f"{a}→{b}")
        elif pegs[a][-1] > pegs[b][-1]:
            pegs[a].append(pegs[b].pop())
            steps.append(f"{b}→{a}")
        else:
            pegs[b].append(pegs[a].pop())
            steps.append(f"{a}→{b}")

    total_moves = 2 ** n - 1
    for i in range(total_moves):
        if i % 3 == 0:
            move_between(names[0], names[1])
        elif i % 3 == 1:
            move_between(names[0], names[2])
        else:
            move_between(names[1], names[2])

    return steps, len(steps)


def validate_steps(steps, n, source='A', target='C', auxiliary='B'):
    pegs = {source: list(range(n, 0, -1)), target: [], auxiliary: []}

    for i, step in enumerate(steps, 1):
        src, tgt = step.split('→')
        if not pegs[src]:
            raise ValueError(f"第 {i} 步: 柱子 {src} 为空，无法移动")
        disk = pegs[src][-1]
        if pegs[tgt] and pegs[tgt][-1] < disk:
            raise ValueError(
                f"第 {i} 步: 大盘 {disk} 不能放在小盘 {pegs[tgt][-1]} 上"
            )
        pegs[src].pop()
        pegs[tgt].append(disk)

    if len(pegs[target]) != n:
        raise ValueError(f"目标柱未包含所有盘子: {pegs[target]}")
    return True


def validate_steps_4pegs(steps, n, source='A', target='D', aux1='B', aux2='C'):
    pegs = {source: list(range(n, 0, -1)), target: [], aux1: [], aux2: []}

    for i, step in enumerate(steps, 1):
        src, tgt = step.split('→')
        if not pegs[src]:
            raise ValueError(f"第 {i} 步: 柱子 {src} 为空，无法移动")
        disk = pegs[src][-1]
        if pegs[tgt] and pegs[tgt][-1] < disk:
            raise ValueError(
                f"第 {i} 步: 大盘 {disk} 不能放在小盘 {pegs[tgt][-1]} 上"
            )
        pegs[src].pop()
        pegs[tgt].append(disk)

    if len(pegs[target]) != n:
        raise ValueError(f"目标柱未包含所有盘子: {pegs[target]}")
    return True


def frame_stewart(n, source='A', target='D', aux1='B', aux2='C'):
    if n <= 0:
        return [], 0
    if n == 1:
        return [f"{source}→{target}"], 1

    steps = []

    def move(k, src, tgt, a1, a2):
        if k == 0:
            return
        if k == 1:
            steps.append(f"{src}→{tgt}")
            return

        m = k - 1 if k <= 2 else max(1, int(k - (2 * k) ** 0.5) + 1)

        move(m, src, a1, tgt, a2)

        _move_3pegs(k - m, src, tgt, a2)

        move(m, a1, tgt, src, a2)

    def _move_3pegs(k, src, tgt, aux):
        if k == 1:
            steps.append(f"{src}→{tgt}")
            return
        _move_3pegs(k - 1, src, aux, tgt)
        steps.append(f"{src}→{tgt}")
        _move_3pegs(k - 1, aux, tgt, src)

    move(n, source, target, aux1, aux2)
    return steps, len(steps)


def benchmark():
    print("=" * 60)
    print("性能对比测试")
    print("=" * 60)

    test_cases = [10, 15, 20]

    for n in test_cases:
        print(f"\n--- n = {n} ---")

        t0 = time.time()
        steps1, count1 = hanoi(n)
        t1 = time.time()
        print(f"递归三柱: {count1:6d} 步, 耗时 {(t1 - t0) * 1000:.2f} ms")

        t0 = time.time()
        steps2, count2 = hanoi_iterative(n)
        t1 = time.time()
        print(f"迭代三柱: {count2:6d} 步, 耗时 {(t1 - t0) * 1000:.2f} ms")

        t0 = time.time()
        steps3, count3 = frame_stewart(n)
        t1 = time.time()
        print(f"Frame-Stewart四柱: {count3:6d} 步, 耗时 {(t1 - t0) * 1000:.2f} ms")

        if n <= 15:
            validate_steps(steps1, n)
            validate_steps(steps2, n)
            validate_steps_4pegs(steps3, n)
            print("✓ 所有算法结果验证通过")


if __name__ == "__main__":
    for n in [0, 3, 4]:
        steps, total = hanoi(n)
        print(f"递归汉诺塔 {n} 层的移动步骤：")
        for i, step in enumerate(steps, 1):
            print(f"第 {i} 步: {step}")
        print(f"总步数: {total} (2^{n} - 1 = {2**n - 1})")
        if steps:
            validate_steps(steps, n)
            print("✓ 验证通过\n")

    print("=" * 60)
    n = 4
    steps, total = frame_stewart(n)
    print(f"四柱汉诺塔 (Frame-Stewart) {n} 层:")
    for i, step in enumerate(steps, 1):
        print(f"第 {i} 步: {step}")
    print(f"总步数: {total} (三柱需 {2**n - 1} 步)")
    validate_steps_4pegs(steps, n)
    print("✓ 四柱验证通过\n")

    benchmark()

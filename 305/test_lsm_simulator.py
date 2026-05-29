import unittest
import random
from lsm_simulator import (
    SSTable, Level, Metrics, BloomFilter, BloomFilterStats,
    LeveledCompaction, SizeTieredCompaction, LSMTreeSimulator,
    run_simulation, recommend_strategy, WorkloadStats,
    AdaptiveCompaction, ComparisonReport
)


class TestBloomFilter(unittest.TestCase):
    def test_bloom_filter_basic(self):
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)
        bf.add(42)
        bf.add(100)
        bf.add(200)

        self.assertTrue(bf.contains(42))
        self.assertTrue(bf.contains(100))
        self.assertTrue(bf.contains(200))

        self.assertFalse(bf.contains(999))
        self.assertFalse(bf.contains(1000))

    def test_bloom_filter_add_all(self):
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)
        keys = {1, 2, 3, 4, 5}
        bf.add_all(keys)

        for k in keys:
            self.assertTrue(bf.contains(k))

        self.assertFalse(bf.contains(999))

    def test_bloom_filter_contains_operator(self):
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)
        bf.add(42)

        self.assertTrue(42 in bf)
        self.assertFalse(100 in bf)

    def test_bloom_filter_size_calculation(self):
        bf = BloomFilter(capacity=1000, false_positive_rate=0.01)
        self.assertGreater(bf.bit_size, 0)
        self.assertGreater(bf.num_hash_functions, 0)

    def test_bloom_filter_low_fpr(self):
        bf = BloomFilter(capacity=1000, false_positive_rate=0.001)
        self.assertGreater(bf.bit_size, 0)

    def test_bloom_filter_statistics(self):
        random.seed(42)
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)

        inserted = set()
        for i in range(50):
            key = random.randint(0, 200)
            bf.add(key)
            inserted.add(key)

        fp_count = 0
        tn_count = 0
        total_negative = 0

        for i in range(201, 500):
            total_negative += 1
            if bf.contains(i):
                fp_count += 1
            else:
                tn_count += 1

        actual_fpr = fp_count / total_negative if total_negative > 0 else 0
        self.assertLess(actual_fpr, 0.1, "FPR should be reasonably low")


class TestBloomFilterStats(unittest.TestCase):
    def test_stats_calculation(self):
        stats = BloomFilterStats()
        stats.total_checks = 100
        stats.true_positive = 30
        stats.true_negative = 60
        stats.false_positive = 10
        stats.saved_lookups = 60

        self.assertAlmostEqual(stats.hit_rate, 0.4)
        self.assertAlmostEqual(stats.true_negative_rate, 0.6)
        self.assertAlmostEqual(stats.false_positive_rate_actual, 10 / 70)
        self.assertAlmostEqual(stats.lookup_savings_rate, 0.6)

    def test_zero_stats(self):
        stats = BloomFilterStats()
        self.assertEqual(stats.hit_rate, 0.0)
        self.assertEqual(stats.true_negative_rate, 0.0)
        self.assertEqual(stats.false_positive_rate_actual, 0.0)
        self.assertEqual(stats.lookup_savings_rate, 0.0)


class TestSSTable(unittest.TestCase):
    def test_sstable_with_bloom_filter(self):
        keys = {1, 2, 3, 4, 5}
        bf = BloomFilter(capacity=10, false_positive_rate=0.01)
        bf.add_all(keys)

        sst = SSTable(
            sstable_id=1, size=500, min_key=1, max_key=5,
            level=0, key_count=5, keys=keys, bloom_filter=bf
        )

        self.assertIsNotNone(sst.bloom_filter)
        self.assertTrue(sst.bloom_filter.contains(3))
        self.assertFalse(sst.bloom_filter.contains(10))


class TestMetrics(unittest.TestCase):
    def test_amplification_with_bloom_filter(self):
        metrics = Metrics()
        metrics.total_writes = 100
        metrics.bytes_written = 500 * 100

        metrics.read_amplification_samples = [1, 2, 3, 4, 5]
        metrics.read_amplification_with_bf_samples = [1, 1, 2, 2, 3]
        metrics.use_bloom_filter = True

        self.assertEqual(metrics.write_amplification, 5.0)
        self.assertEqual(metrics.read_amplification, 3.0)
        self.assertAlmostEqual(metrics.read_amplification_with_bf, 1.8)
        self.assertAlmostEqual(metrics.read_amplification_improvement, 40.0)

    def test_zero_amplification(self):
        metrics = Metrics()
        self.assertEqual(metrics.write_amplification, 0.0)
        self.assertEqual(metrics.read_amplification, 0.0)
        self.assertEqual(metrics.read_amplification_with_bf, 0.0)


class TestLeveledCompaction(unittest.TestCase):
    def setUp(self):
        self.strategy = LeveledCompaction(max_levels=5, sstable_size=2000)

    def test_create_sstable_with_bloom_filter(self):
        keys = {1, 2, 3, 4, 5}
        sst = self.strategy.create_sstable(keys, 0)

        self.assertIsNotNone(sst.bloom_filter)
        self.assertEqual(sst.min_key, 1)
        self.assertEqual(sst.max_key, 5)
        self.assertEqual(sst.key_count, 5)
        self.assertEqual(sst.level, 0)

        self.assertTrue(sst.bloom_filter.contains(3))
        self.assertFalse(sst.bloom_filter.contains(10))

    def test_create_sstable_without_bloom_filter(self):
        strategy = LeveledCompaction(max_levels=5, sstable_size=2000, use_bloom_filter=False)
        keys = {1, 2, 3, 4, 5}
        sst = strategy.create_sstable(keys, 0)

        self.assertIsNone(sst.bloom_filter)

    def test_compaction_with_bloom_filter(self):
        random.seed(42)
        strategy = LeveledCompaction(max_levels=5, sstable_size=2000,
                                      max_bytes_for_level_base=8000,
                                      max_bytes_for_level_multiplier=5)
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        for i in range(1000):
            simulator.write(i)

        self.assertGreater(simulator.metrics.compaction_count, 0)

        for level in simulator.levels:
            for sst in level.sstables:
                self.assertIsNotNone(sst.bloom_filter)

        for i in range(200):
            simulator.read(random.randint(0, 2000))

        self.assertGreater(simulator.metrics.bloom_filter_stats.total_checks, 0)
        self.assertGreater(simulator.metrics.read_amplification_with_bf, 0)
        self.assertLessEqual(
            simulator.metrics.read_amplification_with_bf,
            simulator.metrics.read_amplification
        )

    def test_compaction_without_bloom_filter(self):
        random.seed(42)
        strategy = LeveledCompaction(max_levels=5, sstable_size=2000,
                                      max_bytes_for_level_base=8000,
                                      max_bytes_for_level_multiplier=5,
                                      use_bloom_filter=False)
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        for i in range(1000):
            simulator.write(i)

        for level in simulator.levels:
            for sst in level.sstables:
                self.assertIsNone(sst.bloom_filter)

        self.assertEqual(simulator.metrics.bloom_filter_stats.total_checks, 0)


class TestSizeTieredCompaction(unittest.TestCase):
    def setUp(self):
        self.strategy = SizeTieredCompaction(max_levels=5, sstable_size=2000, min_threshold=2)

    def test_create_sstable_with_bloom_filter(self):
        keys = {1, 2, 3, 4, 5}
        sst = self.strategy.create_sstable(keys, 0)

        self.assertIsNotNone(sst.bloom_filter)
        self.assertTrue(sst.bloom_filter.contains(3))
        self.assertFalse(sst.bloom_filter.contains(10))

    def test_compaction_with_bloom_filter(self):
        random.seed(42)
        strategy = SizeTieredCompaction(max_levels=5, sstable_size=2000, min_threshold=2)
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        for i in range(1000):
            simulator.write(i)

        self.assertGreater(simulator.metrics.compaction_count, 0)

        for level in simulator.levels:
            for sst in level.sstables:
                self.assertIsNotNone(sst.bloom_filter)


class TestLSMTreeSimulator(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        strategy = LeveledCompaction(max_levels=5, sstable_size=2000,
                                      max_bytes_for_level_base=8000,
                                      max_bytes_for_level_multiplier=5)
        self.simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

    def test_check_bloom_filter(self):
        keys = {1, 2, 3, 4, 5}
        sst = self.simulator.strategy.create_sstable(keys, 0)

        bf_pass, is_fp = self.simulator._check_bloom_filter(sst, 3)
        self.assertTrue(bf_pass)
        self.assertFalse(is_fp)

        bf_pass, is_fp = self.simulator._check_bloom_filter(sst, 100)
        self.assertFalse(bf_pass)
        self.assertFalse(is_fp)

        self.assertEqual(self.simulator.metrics.bloom_filter_stats.true_positive, 1)
        self.assertEqual(self.simulator.metrics.bloom_filter_stats.true_negative, 1)
        self.assertEqual(self.simulator.metrics.bloom_filter_stats.saved_lookups, 1)

    def test_read_with_bloom_filter(self):
        for i in range(200):
            self.simulator.write(i)

        found, sst_read = self.simulator.read(50)
        self.assertTrue(found)
        self.assertGreater(sst_read, 0)

        found, sst_read = self.simulator.read(99999)
        self.assertFalse(found)

        self.assertGreater(self.simulator.metrics.bloom_filter_stats.total_checks, 0)
        self.assertGreater(len(self.simulator.metrics.read_amplification_with_bf_samples), 0)

    def test_read_amplification_improvement(self):
        for i in range(500):
            self.simulator.write(i)

        for i in range(200):
            self.simulator.read(random.randint(0, 1000))

        ra_no_bf = self.simulator.metrics.read_amplification
        ra_with_bf = self.simulator.metrics.read_amplification_with_bf

        self.assertGreater(ra_no_bf, 0)
        self.assertGreater(ra_with_bf, 0)
        self.assertLessEqual(ra_with_bf, ra_no_bf)
        self.assertGreaterEqual(self.simulator.metrics.read_amplification_improvement, 0)

    def test_simulate_with_bloom_filter(self):
        metrics = self.simulator.simulate(
            write_rate=1000.0,
            read_ratio=0.3,
            total_operations=1000
        )

        self.assertEqual(metrics.total_writes, 700)
        self.assertEqual(metrics.total_reads, 300)
        self.assertGreater(metrics.write_amplification, 0)
        self.assertGreater(metrics.read_amplification, 0)
        self.assertGreater(metrics.read_amplification_with_bf, 0)
        self.assertGreater(metrics.bloom_filter_stats.total_checks, 0)

    def test_simulate_without_bloom_filter(self):
        random.seed(42)
        strategy = LeveledCompaction(max_levels=5, sstable_size=2000,
                                      max_bytes_for_level_base=8000,
                                      max_bytes_for_level_multiplier=5,
                                      use_bloom_filter=False)
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        metrics = simulator.simulate(
            write_rate=1000.0,
            read_ratio=0.3,
            total_operations=1000
        )

        self.assertEqual(metrics.bloom_filter_stats.total_checks, 0)
        self.assertEqual(len(metrics.read_amplification_with_bf_samples), 0)


class TestSimulationFunctions(unittest.TestCase):
    def test_run_simulation_with_bloom_filter(self):
        random.seed(123)
        metrics, stats = run_simulation(
            'leveled',
            write_rate=1000.0,
            read_ratio=0.5,
            total_operations=2000,
            sstable_size=2000,
            max_bytes_for_level_base=8000,
            memtable_size=100,
            use_bloom_filter=True,
            bloom_filter_fpr=0.01
        )

        self.assertGreater(metrics.total_writes, 0)
        self.assertGreater(metrics.total_reads, 0)
        self.assertIsInstance(stats, dict)
        self.assertGreater(metrics.compaction_count, 0)
        self.assertGreater(metrics.read_amplification_with_bf, 0)
        self.assertGreater(metrics.bloom_filter_stats.total_checks, 0)

    def test_run_simulation_without_bloom_filter(self):
        random.seed(123)
        metrics, stats = run_simulation(
            'size_tiered',
            write_rate=1000.0,
            read_ratio=0.5,
            total_operations=2000,
            sstable_size=2000,
            min_threshold=2,
            memtable_size=100,
            use_bloom_filter=False
        )

        self.assertGreater(metrics.total_writes, 0)
        self.assertEqual(metrics.bloom_filter_stats.total_checks, 0)

    def test_recommend_strategy_with_bloom_filter(self):
        random.seed(456)
        result = recommend_strategy(
            write_rate=1000.0,
            read_ratio=0.3,
            total_operations=1000,
            use_bloom_filter=True,
            bloom_filter_fpr=0.01
        )

        self.assertIn('best_strategy', result)
        self.assertIn('read_amplification', result)
        self.assertIn('read_amplification_no_bf', result)
        self.assertIn('bloom_filter_stats', result)
        self.assertGreater(len(result['all_results']), 0)
        self.assertTrue(result['use_bloom_filter'])

    def test_recommend_strategy_without_bloom_filter(self):
        random.seed(456)
        result = recommend_strategy(
            write_rate=1000.0,
            read_ratio=0.3,
            total_operations=1000,
            use_bloom_filter=False
        )

        self.assertIn('best_strategy', result)
        self.assertFalse(result['use_bloom_filter'])


class TestWorkloadStats(unittest.TestCase):
    def test_workload_stats_basic(self):
        stats = WorkloadStats(window_size=10)

        for i in range(20):
            stats.record_write(i)

        self.assertEqual(len(stats.recent_writes), 10)
        self.assertGreater(stats.get_write_rate(20), 0)

    def test_workload_stats_read_ratio(self):
        stats = WorkloadStats(window_size=100)

        for i in range(30):
            stats.record_write(i)
        for i in range(70):
            stats.record_read_latency(2, i + 30)

        read_ratio = stats.get_read_ratio(100)
        self.assertAlmostEqual(read_ratio, 0.7, delta=0.1)

    def test_workload_stats_avg_latency(self):
        stats = WorkloadStats(window_size=100)

        for i in range(10):
            stats.record_read_latency(i + 1, i)

        self.assertEqual(stats.get_avg_read_latency(), 5.5)

    def test_strategy_switch_recording(self):
        stats = WorkloadStats()
        stats.record_switch(100, 'leveled', 'size_tiered')
        stats.record_switch(200, 'size_tiered', 'leveled')

        self.assertEqual(stats.total_switches, 2)
        self.assertEqual(len(stats.strategy_switches), 2)


class TestAdaptiveCompaction(unittest.TestCase):
    def test_adaptive_initialization(self):
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6
        )

        self.assertEqual(strategy.current_strategy, 'leveled')
        self.assertIsNotNone(strategy.leveled)
        self.assertIsNotNone(strategy.size_tiered)
        self.assertEqual(strategy.switch_threshold, 0.6)

    def test_adaptive_on_write(self):
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=10
        )

        for i in range(100):
            strategy.on_write()

        self.assertGreater(strategy.operation_counter, 0)
        self.assertGreater(len(strategy.workload_stats.recent_writes), 0)

    def test_adaptive_on_read(self):
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=10
        )

        for i in range(50):
            strategy.on_write()
        for i in range(50):
            strategy.on_read(3)

        self.assertGreater(strategy.workload_stats.get_avg_read_latency(), 0)

    def test_adaptive_switch_to_size_tiered(self):
        random.seed(42)
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=20
        )

        for i in range(100):
            strategy.on_write()

        self.assertEqual(strategy.current_strategy, 'size_tiered')
        self.assertGreater(strategy.workload_stats.total_switches, 0)

    def test_adaptive_switch_to_leveled(self):
        random.seed(42)
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=20
        )

        strategy.current_strategy = 'size_tiered'
        strategy.active_strategy = strategy.size_tiered
        strategy.ops_since_switch = 0

        for i in range(100):
            strategy.on_read(2)

        self.assertEqual(strategy.current_strategy, 'leveled')

    def test_adaptive_create_sstable(self):
        strategy = AdaptiveCompaction(max_levels=5, sstable_size=2000)
        keys = {1, 2, 3, 4, 5}

        sst = strategy.create_sstable(keys, 0)

        self.assertEqual(sst.min_key, 1)
        self.assertEqual(sst.max_key, 5)
        self.assertIsNotNone(sst.bloom_filter)

    def test_adaptive_should_compact(self):
        strategy = AdaptiveCompaction(max_levels=5, sstable_size=2000)
        levels = [Level(level_num=0, target_size=8000)]

        self.assertIsNone(strategy.should_compact(levels))

        sst = SSTable(sstable_id=1, size=9000, min_key=1, max_key=100, level=0, key_count=100)
        levels[0].sstables.append(sst)

        self.assertIsNotNone(strategy.should_compact(levels))

    def test_adaptive_get_level_target_size(self):
        strategy = AdaptiveCompaction(max_levels=5, sstable_size=2000)

        self.assertEqual(strategy.get_level_target_size(0), 8000)


class TestLSMTreeWithAdaptive(unittest.TestCase):
    def test_simulator_with_adaptive(self):
        random.seed(42)
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=50
        )
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        for i in range(500):
            simulator.write(i)

        self.assertGreater(simulator.metrics.compaction_count, 0)
        self.assertGreater(strategy.operation_counter, 0)

    def test_adaptive_with_reads(self):
        random.seed(42)
        strategy = AdaptiveCompaction(
            max_levels=5,
            sstable_size=2000,
            switch_threshold=0.6,
            min_ops_before_switch=50
        )
        simulator = LSMTreeSimulator(strategy=strategy, memtable_size=100)

        for i in range(200):
            simulator.write(i)
        for i in range(200):
            simulator.read(random.randint(0, 500))

        self.assertGreater(strategy.workload_stats.get_avg_read_latency(), 0)


class TestComparisonReport(unittest.TestCase):
    def test_report_creation(self):
        report = ComparisonReport()
        self.assertEqual(len(report.strategies), 0)

    def test_report_add_result(self):
        report = ComparisonReport()
        metrics = Metrics()
        stats = {0: {'sstables': 1}}

        report.add_result('Test', metrics, stats)

        self.assertEqual(len(report.strategies), 1)
        self.assertIn('Test', report.metrics)

    def test_report_set_adaptive_stats(self):
        report = ComparisonReport()
        adaptive_data = {'switches': 5, 'final_strategy': 'leveled'}

        report.set_adaptive_stats(adaptive_data)

        self.assertEqual(report.adaptive_stats['switches'], 5)

    def test_report_generation(self):
        report = ComparisonReport()

        metrics1 = Metrics()
        metrics1._write_amplification = 5.0
        metrics1.read_amplification_samples = [3.0] * 100
        metrics1.read_amplification_with_bf_samples = [0.5] * 100
        metrics1.use_bloom_filter = True
        metrics1.compaction_count = 10
        metrics1.bytes_written = 100000
        metrics1.total_writes = 1000
        report.add_result('Test1', metrics1, {})

        metrics2 = Metrics()
        metrics2._write_amplification = 3.0
        metrics2.read_amplification_samples = [5.0] * 100
        metrics2.read_amplification_with_bf_samples = [0.8] * 100
        metrics2.use_bloom_filter = True
        metrics2.compaction_count = 5
        metrics2.bytes_written = 60000
        metrics2.total_writes = 1000
        report.add_result('Test2', metrics2, {})

        report_text = report.generate_report()

        self.assertIn('Test1', report_text)
        self.assertIn('Test2', report_text)
        self.assertIn('最优写放大', report_text)
        self.assertIn('最优读放大', report_text)


class TestAdaptiveSimulation(unittest.TestCase):
    def test_run_adaptive_simulation(self):
        random.seed(42)
        from lsm_simulator import run_adaptive_simulation

        metrics, level_stats, adaptive_stats = run_adaptive_simulation(
            write_rate=1000.0,
            read_ratio=0.5,
            total_operations=1000,
            sstable_size=2000,
            memtable_size=100,
            max_levels=5
        )

        self.assertGreater(metrics.total_writes, 0)
        self.assertGreater(metrics.total_reads, 0)
        self.assertIn('switches', adaptive_stats)
        self.assertIn('final_strategy', adaptive_stats)

    def test_compare_all_strategies(self):
        random.seed(42)
        from lsm_simulator import compare_all_strategies

        report = compare_all_strategies(
            write_rate=1000.0,
            read_ratio=0.5,
            total_operations=500,
            sstable_size=2000,
            memtable_size=100,
            max_levels=5,
            use_bloom_filter=True
        )

        self.assertIn('Leveled', report.strategies)
        self.assertIn('Size-tiered', report.strategies)
        self.assertIn('Adaptive', report.strategies)
        self.assertIsNotNone(report.adaptive_stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)

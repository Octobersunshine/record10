import unittest
from index_recommender import (
    SQLIndexRecommender, ColumnStats, CostEstimate, ExplainPlan
)


class TestSQLIndexRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = SQLIndexRecommender()
        self.schema_input = """
-- TABLE STATS: 10000 rows
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    gender VARCHAR(10),
    age INT,
    city VARCHAR(50),
    status VARCHAR(20),
    created_at DATE
);
-- STATS id: 10000 rows, 10000 distinct
-- STATS username: 10000 rows, 9800 distinct
-- STATS email: 10000 rows, 9900 distinct
-- STATS gender: 10000 rows, 2 distinct
-- STATS age: 10000 rows, 80 distinct
-- STATS city: 10000 rows, 500 distinct
-- STATS status: 10000 rows, 5 distinct
-- STATS created_at: 10000 rows, 3650 distinct

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    status VARCHAR(20)
);
-- STATS id: 50000 rows, 50000 distinct
-- STATS user_id: 50000 rows, 8000 distinct
-- STATS status: 50000 rows, 6 distinct
"""
        self.schemas = self.recommender.parse_table_schemas(self.schema_input)

    def test_parse_table_schemas(self):
        self.assertIn('users', self.schemas)
        self.assertIn('orders', self.schemas)
        self.assertEqual(len(self.schemas['users'].columns), 8)
        self.assertEqual(len(self.schemas['orders'].columns), 3)

    def test_parse_column_stats(self):
        users = self.schemas['users']
        self.assertIn('gender', users.column_stats)
        self.assertEqual(users.column_stats['gender'].distinct_values, 2)
        self.assertEqual(users.column_stats['gender'].total_rows, 10000)
        self.assertAlmostEqual(users.column_stats['gender'].selectivity, 0.0002)

    def test_parse_table_total_rows(self):
        users = self.schemas['users']
        self.assertEqual(users.total_rows, 10000)

    def test_get_selectivity(self):
        users = self.schemas['users']
        selectivity = users.get_selectivity('gender')
        self.assertIsNotNone(selectivity)
        self.assertAlmostEqual(selectivity, 0.0002)

    def test_column_stats_calculate_selectivity(self):
        stats = ColumnStats('test_col', 1000, 50)
        selectivity = stats.calculate_selectivity()
        self.assertEqual(selectivity, 0.05)
        self.assertEqual(stats.selectivity, 0.05)

    def test_sort_fields_by_selectivity(self):
        fields = ['city', 'id', 'age']
        sorted_fields = self.recommender._sort_fields_by_selectivity(
            fields, 'users', self.schemas
        )
        self.assertEqual(sorted_fields[0], 'id')
        self.assertEqual(sorted_fields[1], 'city')
        self.assertEqual(sorted_fields[2], 'age')

    def test_calculate_composite_selectivity(self):
        columns = ['city', 'id']
        composite = self.recommender._calculate_composite_selectivity(
            columns, 'users', self.schemas
        )
        expected = 0.05 * 1.0
        self.assertAlmostEqual(composite, expected)

    def test_explain_without_index(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        analysis = self.recommender.analyze_query(sql, self.schemas)
        plan = self.recommender._explain_without_index(
            analysis, 'users', self.schemas
        )
        self.assertEqual(plan.access_type, 'ALL')
        self.assertEqual(plan.rows_scanned, 10000)
        self.assertGreater(plan.cost, 0)

    def test_explain_with_index(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        analysis = self.recommender.analyze_query(sql, self.schemas)
        plan = self.recommender._explain_with_index(
            analysis, 'users', ['city'], self.schemas
        )
        self.assertIn(plan.access_type, ['ref', 'range', 'index'])
        self.assertLess(plan.rows_scanned, 10000)

    def test_estimate_cost(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        analysis = self.recommender.analyze_query(sql, self.schemas)
        cost_est = self.recommender.estimate_cost(
            analysis, 'users', ['city'], self.schemas
        )
        self.assertIsInstance(cost_est, CostEstimate)
        self.assertGreater(cost_est.original_cost, cost_est.optimized_cost)
        self.assertGreater(cost_est.speedup_ratio, 1.0)

    def test_selectivity_threshold_skips_low_cardinality(self):
        sql = "SELECT * FROM users WHERE gender = 'male'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertNotIn('gender', rec.columns)
        self.assertEqual(len(rec.skipped_fields), 1)
        self.assertEqual(rec.skipped_fields[0].column_name, 'gender')

    def test_high_cardinality_field_included(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertIn('city', rec.columns)
        self.assertEqual(len(rec.skipped_fields), 0)

    def test_mixed_cardinality_fields(self):
        sql = "SELECT * FROM users WHERE gender = 'male' AND city = 'Beijing'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertIn('city', rec.columns)
        self.assertNotIn('gender', rec.columns)
        self.assertEqual(len(rec.skipped_fields), 1)
        self.assertEqual(rec.skipped_fields[0].column_name, 'gender')

    def test_all_fields_low_cardinality_no_recommendation(self):
        sql = "SELECT * FROM users WHERE gender = 'male' AND status = 'active'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertEqual(len(rec.columns), 0)
        self.assertGreaterEqual(len(rec.skipped_fields), 2)

    def test_custom_selectivity_threshold(self):
        custom_recommender = SQLIndexRecommender(selectivity_threshold=0.0001)
        sql = "SELECT * FROM users WHERE gender = 'male'"
        analysis, recommendations = custom_recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertIn('gender', rec.columns)

    def test_skipped_field_contains_selectivity(self):
        sql = "SELECT * FROM users WHERE gender = 'male'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        rec = recommendations[0]
        self.assertEqual(len(rec.skipped_fields), 1)
        skipped = rec.skipped_fields[0]
        self.assertEqual(skipped.table_name, 'users')
        self.assertEqual(skipped.column_name, 'gender')
        self.assertIsNotNone(skipped.selectivity)
        self.assertIn('区分度过低', skipped.reason)

    def test_recommendation_contains_cost_estimate(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertIsNotNone(rec.cost_estimate)
        self.assertIsInstance(rec.cost_estimate, CostEstimate)
        self.assertGreater(rec.cost_estimate.speedup_ratio, 1.0)

    def test_composite_index_columns_sorted_by_selectivity(self):
        sql = """
        SELECT * FROM users 
        WHERE city = 'Beijing' AND username = 'user1' AND age >= 18
        """
        analysis, recommendations = self.recommender.process(sql, self.schema_input)
        
        self.assertEqual(len(recommendations), 1)
        rec = recommendations[0]
        self.assertIn('username', rec.columns)
        self.assertIn('city', rec.columns)
        username_idx = rec.columns.index('username')
        city_idx = rec.columns.index('city')
        self.assertLess(username_idx, city_idx)

    def test_extract_table_aliases(self):
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        self.recommender.extract_table_aliases(sql)
        self.assertEqual(self.recommender.alias_to_table['u'], 'users')
        self.assertEqual(self.recommender.alias_to_table['o'], 'orders')

    def test_analyze_where_equality(self):
        sql = "SELECT * FROM users WHERE city = 'Beijing'"
        equality, range_fields = self.recommender.analyze_where_clause(sql, self.schemas)
        self.assertEqual(len(equality), 1)
        self.assertEqual(equality[0], ('users', 'city'))

    def test_analyze_where_range(self):
        sql = "SELECT * FROM users WHERE age >= 18"
        equality, range_fields = self.recommender.analyze_where_clause(sql, self.schemas)
        self.assertEqual(len(range_fields), 1)
        self.assertEqual(range_fields[0], ('users', 'age'))

    def test_analyze_join_conditions(self):
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        joins = self.recommender.analyze_join_conditions(sql, self.schemas)
        self.assertEqual(len(joins), 1)
        self.assertEqual(joins[0], ('users', 'id', 'orders', 'user_id'))

    def test_cost_estimate_post_init(self):
        cost = CostEstimate(original_cost=1000, optimized_cost=100, speedup_ratio=0)
        self.assertAlmostEqual(cost.speedup_ratio, 10.0)

    def test_explain_plan_creation(self):
        plan = ExplainPlan(
            table_name='users',
            access_type='ref',
            rows_scanned=500,
            cost=100.5,
            extra_info='Using where'
        )
        self.assertEqual(plan.table_name, 'users')
        self.assertEqual(plan.access_type, 'ref')
        self.assertEqual(plan.rows_scanned, 500)
        self.assertEqual(plan.cost, 100.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

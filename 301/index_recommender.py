import re
import math
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ColumnStats:
    column_name: str
    total_rows: int = 0
    distinct_values: int = 0
    selectivity: float = 0.0

    def calculate_selectivity(self) -> float:
        if self.total_rows > 0:
            self.selectivity = self.distinct_values / self.total_rows
        return self.selectivity


@dataclass
class TableSchema:
    table_name: str
    columns: List[str]
    primary_keys: List[str] = field(default_factory=list)
    existing_indexes: List[str] = field(default_factory=list)
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    total_rows: int = 0

    def get_selectivity(self, column: str) -> Optional[float]:
        if column in self.column_stats:
            return self.column_stats[column].selectivity
        return None


@dataclass
class QueryAnalysis:
    where_equality_fields: List[Tuple[str, str]] = field(default_factory=list)
    where_range_fields: List[Tuple[str, str]] = field(default_factory=list)
    join_fields: List[Tuple[str, str, str, str]] = field(default_factory=list)
    order_by_fields: List[Tuple[str, str]] = field(default_factory=list)
    group_by_fields: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class SkippedField:
    table_name: str
    column_name: str
    reason: str
    selectivity: Optional[float]


@dataclass
class CostEstimate:
    original_cost: float
    optimized_cost: float
    speedup_ratio: float
    details: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.optimized_cost > 0:
            self.speedup_ratio = self.original_cost / self.optimized_cost


@dataclass
class IndexRecommendation:
    table_name: str
    columns: List[str]
    index_type: str
    reason: str
    create_statement: str
    skipped_fields: List[SkippedField] = field(default_factory=list)
    cost_estimate: Optional[CostEstimate] = None


@dataclass
class ExplainPlan:
    table_name: str
    access_type: str
    rows_scanned: float
    cost: float
    extra_info: str = ""


class SQLIndexRecommender:
    SELECTIVITY_THRESHOLD = 0.05
    IO_COST_PER_ROW = 1.0
    INDEX_LOOKUP_COST = 0.5
    SORT_COST_PER_ROW = 0.1

    def __init__(self, selectivity_threshold: float = SELECTIVITY_THRESHOLD):
        self.table_aliases: Dict[str, str] = {}
        self.alias_to_table: Dict[str, str] = {}
        self.selectivity_threshold = selectivity_threshold

    def parse_table_schemas(self, schema_input: str) -> Dict[str, TableSchema]:
        schemas = {}
        current_table = None
        columns = []
        column_stats = {}
        total_rows = 0
        last_table_name = None
        pending_table_rows = None
        
        lines = schema_input.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            total_rows_match = re.match(
                r'--\s*TABLE\s+STATS\s*:\s*(\d+)\s+rows?',
                line,
                re.IGNORECASE
            )
            if total_rows_match:
                rows = int(total_rows_match.group(1))
                if current_table:
                    total_rows = rows
                elif last_table_name and last_table_name in schemas:
                    schemas[last_table_name].total_rows = rows
                else:
                    pending_table_rows = rows
                continue
            
            create_match = re.match(r'CREATE\s+TABLE\s+(\w+)', line, re.IGNORECASE)
            if create_match:
                if current_table:
                    schemas[current_table] = TableSchema(
                        current_table, columns, 
                        column_stats=column_stats, 
                        total_rows=total_rows
                    )
                    last_table_name = current_table
                current_table = create_match.group(1)
                columns = []
                column_stats = {}
                if pending_table_rows is not None:
                    total_rows = pending_table_rows
                    pending_table_rows = None
                else:
                    total_rows = 0
                continue
            
            stats_match = re.match(
                r'--\s*STATS\s+(\w+)\s*:\s*(\d+)\s+rows?\s*,\s*(\d+)\s+distinct',
                line,
                re.IGNORECASE
            )
            if stats_match:
                col_name = stats_match.group(1)
                rows = int(stats_match.group(2))
                distinct = int(stats_match.group(3))
                stats = ColumnStats(col_name, rows, distinct)
                stats.calculate_selectivity()
                
                if current_table:
                    column_stats[col_name] = stats
                elif last_table_name and last_table_name in schemas:
                    schemas[last_table_name].column_stats[col_name] = stats
                continue
            
            if current_table and line.startswith(')'):
                schemas[current_table] = TableSchema(
                    current_table, columns,
                    column_stats=column_stats,
                    total_rows=total_rows
                )
                last_table_name = current_table
                current_table = None
                continue
            
            if current_table:
                col_match = re.match(r'(\w+)\s+', line)
                if col_match:
                    col_name = col_match.group(1).upper()
                    if col_name not in ['PRIMARY', 'FOREIGN', 'KEY', 'UNIQUE', 'INDEX', 'CONSTRAINT']:
                        actual_col = col_match.group(1)
                        columns.append(actual_col)
                        if actual_col not in column_stats:
                            column_stats[actual_col] = ColumnStats(actual_col, 0, 0, 1.0)
        
        return schemas

    def extract_table_aliases(self, sql: str) -> None:
        self.table_aliases = {}
        self.alias_to_table = {}
        
        from_match = re.search(r'FROM\s+(.+?)(?:WHERE|GROUP|ORDER|JOIN|$)', sql, re.IGNORECASE | re.DOTALL)
        if from_match:
            from_clause = from_match.group(1)
            self._parse_table_list(from_clause)
        
        join_matches = re.finditer(r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql, re.IGNORECASE)
        for match in join_matches:
            table = match.group(1)
            alias = match.group(2) or table
            self.table_aliases[table] = alias
            self.alias_to_table[alias] = table

    def _parse_table_list(self, clause: str) -> None:
        tables = [t.strip() for t in clause.split(',')]
        for table_str in tables:
            parts = table_str.strip().split()
            if len(parts) >= 2 and parts[1].upper() != 'AS':
                table = parts[0]
                alias = parts[1]
                self.table_aliases[table] = alias
                self.alias_to_table[alias] = table
            elif len(parts) >= 3 and parts[1].upper() == 'AS':
                table = parts[0]
                alias = parts[2]
                self.table_aliases[table] = alias
                self.alias_to_table[alias] = table
            elif len(parts) >= 1:
                table = parts[0]
                self.table_aliases[table] = table
                self.alias_to_table[table] = table

    def _resolve_column(self, column_ref: str, schemas: Dict[str, TableSchema]) -> Tuple[str, str]:
        if '.' in column_ref:
            parts = column_ref.split('.')
            alias_or_table = parts[0]
            column = parts[1]
            table = self.alias_to_table.get(alias_or_table, alias_or_table)
            return table, column
        else:
            for table, schema in schemas.items():
                if column_ref in schema.columns:
                    return table, column_ref
            return list(schemas.keys())[0] if schemas else '', column_ref

    def analyze_where_clause(self, sql: str, schemas: Dict[str, TableSchema]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        equality_fields = []
        range_fields = []
        
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return equality_fields, range_fields
        
        where_clause = where_match.group(1)
        
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
        
        for cond in conditions:
            cond = cond.strip()
            
            eq_match = re.match(r'([\w.]+)\s*=\s*', cond)
            if eq_match:
                col = eq_match.group(1)
                table, column = self._resolve_column(col, schemas)
                equality_fields.append((table, column))
                continue
            
            range_match = re.match(r'([\w.]+)\s*(>=|<=|>|<|LIKE|BETWEEN|IN)\s*', cond, re.IGNORECASE)
            if range_match:
                col = range_match.group(1)
                table, column = self._resolve_column(col, schemas)
                range_fields.append((table, column))
        
        return equality_fields, range_fields

    def analyze_join_conditions(self, sql: str, schemas: Dict[str, TableSchema]) -> List[Tuple[str, str, str, str]]:
        join_fields = []
        
        join_pattern = r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?\s+ON\s+([\w.]+)\s*=\s*([\w.]+)'
        matches = re.finditer(join_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            table2 = match.group(1)
            col1 = match.group(2)
            col2 = match.group(3)
            
            t1, c1 = self._resolve_column(col1, schemas)
            t2, c2 = self._resolve_column(col2, schemas)
            
            join_fields.append((t1, c1, t2, c2))
        
        return join_fields

    def analyze_order_by(self, sql: str, schemas: Dict[str, TableSchema]) -> List[Tuple[str, str]]:
        order_fields = []
        
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if not order_match:
            return order_fields
        
        order_clause = order_match.group(1)
        fields = [f.strip().split()[0] for f in order_clause.split(',')]
        
        for field in fields:
            field_name = field.strip()
            table, column = self._resolve_column(field_name, schemas)
            for schema in schemas.values():
                if column in schema.columns:
                    order_fields.append((table, column))
        
        return order_fields

    def analyze_group_by(self, sql: str, schemas: Dict[str, TableSchema]) -> List[Tuple[str, str]]:
        group_fields = []
        
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if not group_match:
            return group_fields
        
        group_clause = group_match.group(1)
        fields = [f.strip() for f in group_clause.split(',')]
        
        for field in fields:
            table, column = self._resolve_column(field.strip(), schemas)
            group_fields.append((table, column))
        
        return group_fields

    def analyze_query(self, sql: str, schemas: Dict[str, TableSchema]) -> QueryAnalysis:
        self.extract_table_aliases(sql)
        
        analysis = QueryAnalysis()
        analysis.where_equality_fields, analysis.where_range_fields = self.analyze_where_clause(sql, schemas)
        analysis.join_fields = self.analyze_join_conditions(sql, schemas)
        analysis.order_by_fields = self.analyze_order_by(sql, schemas)
        analysis.group_by_fields = self.analyze_group_by(sql, schemas)
        
        return analysis

    def _check_field_selectivity(self, table: str, column: str, schemas: Dict[str, TableSchema], field_type: str) -> Tuple[bool, Optional[SkippedField]]:
        if table not in schemas:
            return True, None
        
        schema = schemas[table]
        selectivity = schema.get_selectivity(column)
        
        if selectivity is None:
            return True, None
        
        if selectivity < self.selectivity_threshold:
            reason = "区分度过低(%.2f%%) < 阈值(%.2f%%)，不建议建索引" % (
                selectivity * 100, self.selectivity_threshold * 100)
            skipped = SkippedField(
                    table_name=table,
                    column_name=column,
                    reason=reason,
                    selectivity=selectivity
                )
            return False, skipped
        
        return True, None

    def _sort_fields_by_selectivity(self, fields: List[str], table: str, schemas: Dict[str, TableSchema]) -> List[str]:
        def get_sel(col):
            sel = schemas[table].get_selectivity(col)
            return sel if sel is not None else 1.0
        
        return sorted(fields, key=get_sel, reverse=True)

    def _calculate_composite_selectivity(self, columns: List[str], table: str, schemas: Dict[str, TableSchema]) -> float:
        composite_sel = 1.0
        for col in columns:
            sel = schemas[table].get_selectivity(col)
            if sel is not None:
                composite_sel *= sel
        return composite_sel

    def _explain_without_index(self, analysis: QueryAnalysis, table: str, schemas: Dict[str, TableSchema]) -> ExplainPlan:
        schema = schemas[table]
        total_rows = max(schema.total_rows, 1)
        
        cost = total_rows * self.IO_COST_PER_ROW
        
        where_count = len([f for f in analysis.where_equality_fields if f[0] == table]) + \
                     len([f for f in analysis.where_range_fields if f[0] == table])
        
        if where_count > 0:
            cost += total_rows * where_count * 0.1
        
        has_order = any(f[0] == table for f in analysis.order_by_fields)
        has_group = any(f[0] == table for f in analysis.group_by_fields)
        
        if has_order:
            cost += total_rows * math.log2(max(total_rows, 2)) * self.SORT_COST_PER_ROW
        
        if has_group:
            cost += total_rows * 0.5
        
        return ExplainPlan(
            table_name=table,
            access_type='ALL',
            rows_scanned=total_rows,
            cost=cost,
            extra_info='Using where; Using filesort' if has_order else 'Using where'
        )

    def _explain_with_index(self, analysis: QueryAnalysis, table: str, index_columns: List[str], schemas: Dict[str, TableSchema]) -> ExplainPlan:
        schema = schemas[table]
        total_rows = max(schema.total_rows, 1)
        
        equality_cols = [f[1] for f in analysis.where_equality_fields if f[0] == table and f[1] in index_columns]
        range_cols = [f[1] for f in analysis.where_range_fields if f[0] == table and f[1] in index_columns]
        join_cols = []
        for t1, c1, t2, c2 in analysis.join_fields:
            if t1 == table and c1 in index_columns:
                join_cols.append(c1)
            if t2 == table and c2 in index_columns:
                join_cols.append(c2)
        
        all_used_cols = equality_cols + range_cols + join_cols
        composite_sel = self._calculate_composite_selectivity(all_used_cols, table, schemas)
        
        estimated_rows = max(total_rows * composite_sel, 1)
        
        if equality_cols or join_cols:
            access_type = 'ref'
            index_lookup_cost = self.INDEX_LOOKUP_COST * math.log2(max(total_rows, 2))
            scan_cost = estimated_rows * self.IO_COST_PER_ROW * 0.1
            cost = index_lookup_cost + scan_cost
        elif range_cols:
            access_type = 'range'
            index_lookup_cost = self.INDEX_LOOKUP_COST * math.log2(max(total_rows, 2)) * 2
            scan_cost = estimated_rows * self.IO_COST_PER_ROW * 0.3
            cost = index_lookup_cost + scan_cost
        else:
            access_type = 'index'
            cost = total_rows * self.IO_COST_PER_ROW * 0.5
        
        order_cols = [f[1] for f in analysis.order_by_fields if f[0] == table]
        group_cols = [f[1] for f in analysis.group_by_fields if f[0] == table]
        
        index_prefix = index_columns[:len(order_cols)]
        order_covered = order_cols and index_prefix == order_cols
        
        group_prefix = index_columns[:len(group_cols)]
        group_covered = group_cols and group_prefix == group_cols
        
        has_order = any(f[0] == table for f in analysis.order_by_fields)
        has_group = any(f[0] == table for f in analysis.group_by_fields)
        
        extra_parts = ['Using where']
        if has_order and not order_covered:
            cost += estimated_rows * math.log2(max(estimated_rows, 2)) * self.SORT_COST_PER_ROW * 0.5
            extra_parts.append('Using filesort')
        elif has_order and order_covered:
            extra_parts.append('Using index for order by')
        
        if has_group and not group_covered:
            cost += estimated_rows * 0.2
            extra_parts.append('Using temporary')
        elif has_group and group_covered:
            extra_parts.append('Using index for group by')
        
        return ExplainPlan(
            table_name=table,
            access_type=access_type,
            rows_scanned=estimated_rows,
            cost=cost,
            extra_info='; '.join(extra_parts)
        )

    def estimate_cost(self, analysis: QueryAnalysis, table: str, index_columns: List[str], schemas: Dict[str, TableSchema]) -> CostEstimate:
        original_plan = self._explain_without_index(analysis, table, schemas)
        optimized_plan = self._explain_with_index(analysis, table, index_columns, schemas)
        
        speedup = original_plan.cost / max(optimized_plan.cost, 0.0001)
        
        details = {
            'original_rows': original_plan.rows_scanned,
            'original_cost': original_plan.cost,
            'original_access_type': original_plan.access_type,
            'optimized_rows': optimized_plan.rows_scanned,
            'optimized_cost': optimized_plan.cost,
            'optimized_access_type': optimized_plan.access_type,
            'composite_selectivity': self._calculate_composite_selectivity(index_columns, table, schemas)
        }
        
        return CostEstimate(
            original_cost=original_plan.cost,
            optimized_cost=optimized_plan.cost,
            speedup_ratio=speedup,
            details=details
        )

    def recommend_indexes(self, analysis: QueryAnalysis, schemas: Dict[str, TableSchema]) -> List[IndexRecommendation]:
        recommendations = []
        seen_indexes = set()
        
        table_fields: Dict[str, Dict[str, List[str]]] = {}
        table_skipped: Dict[str, List[SkippedField]] = {}
        for table in schemas.keys():
            table_fields[table] = {
                'equality': [],
                'range': [],
                'order': [],
                'group': [],
                'join': []
            }
            table_skipped[table] = []
        
        for table, column in analysis.where_equality_fields:
            if table in table_fields:
                include, skipped = self._check_field_selectivity(table, column, schemas, '等值查询')
                if include and column not in table_fields[table]['equality']:
                    table_fields[table]['equality'].append(column)
                elif skipped:
                    table_skipped[table].append(skipped)
        
        for table, column in analysis.where_range_fields:
            if table in table_fields:
                include, skipped = self._check_field_selectivity(table, column, schemas, '范围查询')
                if include and column not in table_fields[table]['range']:
                    table_fields[table]['range'].append(column)
                elif skipped:
                    table_skipped[table].append(skipped)
        
        for t1, c1, t2, c2 in analysis.join_fields:
            if t1 in table_fields:
                include, skipped = self._check_field_selectivity(t1, c1, schemas, 'JOIN')
                if include and c1 not in table_fields[t1]['join']:
                    table_fields[t1]['join'].append(c1)
                elif skipped:
                    table_skipped[t1].append(skipped)
            
            if t2 in table_fields:
                include, skipped = self._check_field_selectivity(t2, c2, schemas, 'JOIN')
                if include and c2 not in table_fields[t2]['join']:
                    table_fields[t2]['join'].append(c2)
                elif skipped:
                    table_skipped[t2].append(skipped)
        
        for table, column in analysis.order_by_fields:
            if table in table_fields:
                include, skipped = self._check_field_selectivity(table, column, schemas, 'ORDER BY')
                if include and column not in table_fields[table]['order']:
                    table_fields[table]['order'].append(column)
                elif skipped:
                    table_skipped[table].append(skipped)
        
        for table, column in analysis.group_by_fields:
            if table in table_fields:
                include, skipped = self._check_field_selectivity(table, column, schemas, 'GROUP BY')
                if include and column not in table_fields[table]['group']:
                    table_fields[table]['group'].append(column)
                elif skipped:
                    table_skipped[table].append(skipped)
        
        for table, fields in table_fields.items():
            index_columns = []
            reasons = []
            
            sorted_equality = self._sort_fields_by_selectivity(fields['equality'], table, schemas)
            for col in sorted_equality:
                if col not in index_columns:
                    sel = schemas[table].get_selectivity(col)
                    sel_str = " (%.1f%%)" % (sel * 100) if sel else ""
                    index_columns.append(col)
                    reasons.append("等值查询字段: %s%s" % (col, sel_str))
            
            sorted_join = self._sort_fields_by_selectivity(fields['join'], table, schemas)
            for col in sorted_join:
                if col not in index_columns:
                    sel = schemas[table].get_selectivity(col)
                    sel_str = " (%.1f%%)" % (sel * 100) if sel else ""
                    index_columns.append(col)
                    reasons.append("JOIN关联字段: %s%s" % (col, sel_str))
            
            sorted_range = self._sort_fields_by_selectivity(fields['range'], table, schemas)
            for col in sorted_range:
                if col not in index_columns:
                    sel = schemas[table].get_selectivity(col)
                    sel_str = " (%.1f%%)" % (sel * 100) if sel else ""
                    index_columns.append(col)
                    reasons.append("范围查询字段: %s%s" % (col, sel_str))
            
            sorted_group = self._sort_fields_by_selectivity(fields['group'], table, schemas)
            for col in sorted_group:
                if col not in index_columns:
                    sel = schemas[table].get_selectivity(col)
                    sel_str = " (%.1f%%)" % (sel * 100) if sel else ""
                    index_columns.append(col)
                    reasons.append("GROUP BY字段: %s%s" % (col, sel_str))
            
            for col in fields['order']:
                if col not in index_columns:
                    sel = schemas[table].get_selectivity(col)
                    sel_str = " (%.1f%%)" % (sel * 100) if sel else ""
                    index_columns.append(col)
                    reasons.append("ORDER BY字段: %s%s" % (col, sel_str))
            
            if index_columns:
                index_key = "%s:%s" % (table, ','.join(index_columns))
                if index_key not in seen_indexes:
                    seen_indexes.add(index_key)
                    index_name = "idx_%s_%s" % (table.lower(), '_'.join([c.lower() for c in index_columns[:3]]))
                    if len(index_columns) > 3:
                        index_name += "_etc"
                    
                    create_stmt = "CREATE INDEX %s ON %s (%s);" % (index_name, table, ', '.join(index_columns))
                    
                    cost_estimate = self.estimate_cost(analysis, table, index_columns, schemas)
                    
                    recommendations.append(IndexRecommendation(
                        table_name=table,
                        columns=index_columns,
                        index_type='B-TREE',
                        reason='; '.join(reasons),
                        create_statement=create_stmt,
                        skipped_fields=table_skipped.get(table, []),
                        cost_estimate=cost_estimate
                    ))
            elif table_skipped.get(table, []):
                recommendations.append(IndexRecommendation(
                    table_name=table,
                    columns=[],
                    index_type='B-TREE',
                    reason='所有候选字段区分度均过低',
                    create_statement='',
                    skipped_fields=table_skipped[table],
                    cost_estimate=None
                ))
        
        return recommendations

    def process(self, sql: str, schema_input: str) -> Tuple[QueryAnalysis, List[IndexRecommendation]]:
        schemas = self.parse_table_schemas(schema_input)
        analysis = self.analyze_query(sql, schemas)
        recommendations = self.recommend_indexes(analysis, schemas)
        return analysis, recommendations


def main():
    print("=" * 70)
    print("SQL 索引推荐工具（含区分度分析、成本估算、加速比计算）")
    print("=" * 70)
    
    schema_input = """
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

-- TABLE STATS: 50000 rows
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    product_id INT,
    amount DECIMAL(10,2),
    order_date DATE,
    status VARCHAR(20)
);
-- STATS id: 50000 rows, 50000 distinct
-- STATS user_id: 50000 rows, 8000 distinct
-- STATS product_id: 50000 rows, 2000 distinct
-- STATS amount: 50000 rows, 1000 distinct
-- STATS order_date: 50000 rows, 1825 distinct
-- STATS status: 50000 rows, 6 distinct
"""
    
    print("\n表结构及统计信息:")
    print("-" * 70)
    print(schema_input)
    
    sql_queries = [
        """
        SELECT * FROM users
        WHERE gender = 'male' AND city = 'Beijing' AND age >= 18
        ORDER BY created_at DESC
        """,
        
        """
        SELECT u.username, o.amount, o.order_date
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.city = 'Shanghai' AND o.status = 'paid'
        ORDER BY o.order_date DESC
        """,
        
        """
        SELECT city, COUNT(*) as cnt
        FROM users
        WHERE age BETWEEN 20 AND 40
        GROUP BY city
        ORDER BY cnt DESC
        """
    ]
    
    recommender = SQLIndexRecommender()
    
    for i, sql in enumerate(sql_queries, 1):
        print("\n" + "=" * 70)
        print("查询 %d:" % i)
        print("=" * 70)
        print(sql.strip())
        
        analysis, recommendations = recommender.process(sql, schema_input)
        
        print("\n" + "=" * 70)
        print("查询分析结果:")
        print("=" * 70)
        
        if analysis.where_equality_fields:
            print("\nWHERE 等值查询字段:")
            for table, col in analysis.where_equality_fields:
                print("  - %s.%s" % (table, col))
        
        if analysis.where_range_fields:
            print("\nWHERE 范围查询字段:")
            for table, col in analysis.where_range_fields:
                print("  - %s.%s" % (table, col))
        
        if analysis.join_fields:
            print("\nJOIN 关联字段:")
            for t1, c1, t2, c2 in analysis.join_fields:
                print("  - %s.%s = %s.%s" % (t1, c1, t2, c2))
        
        if analysis.order_by_fields:
            print("\nORDER BY 字段:")
            for table, col in analysis.order_by_fields:
                print("  - %s.%s" % (table, col))
        
        if analysis.group_by_fields:
            print("\nGROUP BY 字段:")
            for table, col in analysis.group_by_fields:
                print("  - %s.%s" % (table, col))
        
        print("\n" + "=" * 70)
        print("推荐索引及成本估算:")
        print("=" * 70)
        
        for j, rec in enumerate(recommendations, 1):
            print("\n推荐 %d:" % j)
            print("  表名: %s" % rec.table_name)
            if rec.columns:
                print("  列: %s" % ', '.join(rec.columns))
                print("  类型: %s" % rec.index_type)
                print("  原因: %s" % rec.reason)
                print("  创建语句: %s" % rec.create_statement)
                
                if rec.cost_estimate:
                    print("\n  成本估算 (模拟 EXPLAIN):")
                    print("    原始成本: %.2f (全表扫描 %d 行)" % (
                        rec.cost_estimate.original_cost,
                        rec.cost_estimate.details.get('original_rows', 0)
                    ))
                    print("    优化成本: %.2f (索引扫描 %.0f 行)" % (
                        rec.cost_estimate.optimized_cost,
                        rec.cost_estimate.details.get('optimized_rows', 0)
                    ))
                    print("    访问类型: %s -> %s" % (
                        rec.cost_estimate.details.get('original_access_type', 'ALL'),
                        rec.cost_estimate.details.get('optimized_access_type', 'index')
                    ))
                    print("    预期加速比: %.2fx" % rec.cost_estimate.speedup_ratio)
            else:
                print("  状态: 无推荐索引")
            
            if rec.skipped_fields:
                print("\n  跳过低区分度字段:")
                for skipped in rec.skipped_fields:
                    print("    - %s.%s: %s" % (skipped.table_name, skipped.column_name, skipped.reason))


if __name__ == "__main__":
    main()

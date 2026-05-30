from excel_processor import ExcelProcessor


def test_merged_cells():
    print("=" * 60)
    print("测试合并单元格提取功能")
    print("=" * 60)
    
    processor = ExcelProcessor("merged_cells_test.xlsx")
    
    print("\n1. 提取全部数据（包含各种合并单元格）:")
    print("-" * 60)
    
    data = processor.extract_range_data(
        sheet_name="合并单元格测试",
        start_row=1,
        end_row=11,
        start_col=1,
        end_col=4,
        has_header=False
    )
    
    print(f"数据行数: {data['row_count']}")
    print(f"列数: {data['column_count']}")
    print(f"表头: {data['headers']}")
    print("\n所有数据:")
    for i, row in enumerate(data['data'], 1):
        print(f"  第{i}行: {row}")
    
    print("\n" + "=" * 60)
    print("\n2. 测试横向合并单元格 (A1:C1):")
    print("-" * 60)
    
    print("  A1单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 1, 1))
    print("  B1单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 1, 2))
    print("  C1单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 1, 3))
    print("  D1单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 1, 4))
    
    print("\n" + "=" * 60)
    print("\n3. 测试纵向合并单元格 (A4:A6):")
    print("-" * 60)
    
    print("  A4单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 4, 1))
    print("  A5单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 5, 1))
    print("  A6单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 6, 1))
    
    print("\n" + "=" * 60)
    print("\n4. 测试复杂合并 (A2:A3 和 B2:D2):")
    print("-" * 60)
    
    print("  A2单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 2, 1))
    print("  A3单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 3, 1))
    print("  B2单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 2, 2))
    print("  C2单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 2, 3))
    print("  D2单元格值:", processor._get_merged_cell_value(processor.workbook['合并单元格测试'], 2, 4))
    
    print("\n" + "=" * 60)
    print("\n5. 测试JSON输出:")
    print("-" * 60)
    
    json_output = processor.to_json(data)
    print(f"JSON输出长度: {len(json_output)}")
    print(f"JSON前800字符:\n{json_output[:800]}...")
    
    print("\n" + "=" * 60)
    print("\n6. 测试指定范围提取（只提取Q1和Q2数据，行4-9）:")
    print("-" * 60)
    
    range_data = processor.extract_range_data(
        sheet_name="合并单元格测试",
        start_row=3,
        end_row=9,
        start_col=1,
        end_col=4,
        has_header=True
    )
    
    print(f"表头: {range_data['headers']}")
    print("数据:")
    for row in range_data['data']:
        print(f"  {row}")
    
    print("\n" + "=" * 60)
    print("合并单元格测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_merged_cells()

from excel_processor import ExcelProcessor
import os


def test_new_features():
    print("=" * 70)
    print("测试新增功能：多工作表提取、HTML表格、分页读取")
    print("=" * 70)
    
    processor = ExcelProcessor("sample_data.xlsx")
    
    print("\n" + "=" * 70)
    print("1. 测试多工作表批量提取功能")
    print("=" * 70)
    
    all_sheets_data = processor.extract_multiple_sheets()
    print(f"文件: {all_sheets_data['file_path']}")
    print(f"工作表数量: {all_sheets_data['sheet_count']}")
    
    for sheet_name, sheet_data in all_sheets_data['sheets'].items():
        print(f"\n  工作表: {sheet_name}")
        if 'error' in sheet_data:
            print(f"    错误: {sheet_data['error']}")
        else:
            print(f"    行数: {sheet_data['row_count']}")
            print(f"    列数: {sheet_data['column_count']}")
            print(f"    表头: {sheet_data['headers']}")
            print(f"    前2条数据:")
            for row in sheet_data['data'][:2]:
                print(f"      {row}")
    
    print("\n" + "-" * 70)
    print("  批量提取指定工作表（销售数据）:")
    specific_sheets = processor.extract_multiple_sheets(
        sheet_names=["销售数据"],
        start_row=1,
        end_row=6
    )
    print(f"  提取的工作表: {list(specific_sheets['sheets'].keys())}")
    sales_data = specific_sheets['sheets']['销售数据']
    print(f"  数据行数: {sales_data['row_count']}")
    
    print("\n" + "=" * 70)
    print("2. 测试Excel转HTML表格功能")
    print("=" * 70)
    
    sales_data = processor.extract_range_data(
        sheet_name="销售数据",
        start_row=1,
        end_row=7
    )
    
    html_table = processor.to_html_table(sales_data)
    print(f"HTML表格长度: {len(html_table)} 字符")
    print("\nHTML表格预览 (前800字符):")
    print("-" * 70)
    print(html_table[:800])
    print("...")
    
    html_file = "sales_report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>销售报表</title>
</head>
<body>
    <h1>2024年上半年销售报表</h1>
    {html_table}
    <p style="margin-top: 20px; color: #666; font-style: italic;">
        数据来源: sample_data.xlsx / 销售数据
    </p>
</body>
</html>""")
    print(f"\n完整HTML文件已保存: {html_file}")
    print(f"文件大小: {os.path.getsize(html_file)} 字节")
    
    print("\n" + "-" * 70)
    print("  不带样式的纯HTML表格:")
    plain_html = processor.to_html_table(sales_data, include_style=False)
    print(f"  纯HTML长度: {len(plain_html)} 字符")
    print(f"  内容预览:")
    print(plain_html[:200])
    
    print("\n" + "=" * 70)
    print("3. 测试大数据量分页读取功能")
    print("=" * 70)
    
    print("\n  使用小的page_size(5行)演示分页效果:")
    print("-" * 70)
    
    total_rows = 0
    for page in processor.read_paginated(
        sheet_name="销售数据",
        page_size=5,
        start_row=1,
        has_header=True
    ):
        print(f"\n  第 {page['page_number']} 页 / 共 {page['total_pages']} 页")
        print(f"    行范围: {page['start_row']} - {page['end_row']}")
        print(f"    当前页行数: {page['row_count']}")
        print(f"    页大小: {page['page_size']}")
        print(f"    表头: {page['headers']}")
        print(f"    数据预览:")
        for row in page['data'][:2]:
            print(f"      {row}")
        total_rows += page['row_count']
    
    print(f"\n  分页读取完成，总行数: {total_rows}")
    
    print("\n" + "-" * 70)
    print("  使用默认page_size(10000行)读取员工信息:")
    emp_pages = list(processor.read_paginated(
        sheet_name="员工信息",
        page_size=10000
    ))
    print(f"  总页数: {len(emp_pages)}")
    if emp_pages:
        print(f"  第一页行数: {emp_pages[0]['row_count']}")
        print(f"  所有员工数据:")
        for row in emp_pages[0]['data']:
            print(f"    {row}")
    
    print("\n" + "=" * 70)
    print("4. 综合测试：多工作表 + HTML导出")
    print("=" * 70)
    
    all_html_parts = []
    all_data = processor.extract_multiple_sheets()
    
    for sheet_name, sheet_data in all_data['sheets'].items():
        if 'error' not in sheet_data:
            html_part = processor.to_html_table(sheet_data, include_style=False)
            all_html_parts.append(f"<h2>{sheet_name}</h2>\n{html_part}")
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Excel完整数据报表</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #4472C4; }}
        h2 {{ color: #2E75B6; margin-top: 30px; }}
        .excel-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
        }}
        .excel-table th {{
            background-color: #4472C4;
            color: white;
            padding: 10px;
            border: 1px solid #305496;
        }}
        .excel-table td {{
            padding: 8px 10px;
            border: 1px solid #D4D4D4;
        }}
        .excel-table tr:nth-child(even) {{
            background-color: #F2F2F2;
        }}
    </style>
</head>
<body>
    <h1>Excel数据完整报表</h1>
    <p>文件: sample_data.xlsx</p>
    {''.join(all_html_parts)}
</body>
</html>"""
    
    full_html_file = "full_report.html"
    with open(full_html_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"完整多工作表HTML报表已保存: {full_html_file}")
    print(f"文件大小: {os.path.getsize(full_html_file)} 字节")
    
    print("\n" + "=" * 70)
    print("所有新功能测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_new_features()

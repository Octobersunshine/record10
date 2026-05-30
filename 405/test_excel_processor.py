import json
import base64
from excel_processor import ExcelProcessor, process_excel


def test_excel_processor():
    print("=" * 60)
    print("测试 ExcelProcessor 功能")
    print("=" * 60)
    
    processor = ExcelProcessor("sample_data.xlsx")
    
    print("\n1. 获取所有工作表名称:")
    sheets = processor.get_sheet_names()
    print(f"   工作表列表: {sheets}")
    
    print("\n2. 提取 '销售数据' 工作表的全部数据:")
    sales_data = processor.extract_range_data(
        sheet_name="销售数据",
        has_header=True
    )
    print(f"   数据行数: {sales_data['row_count']}")
    print(f"   列数: {sales_data['column_count']}")
    print(f"   表头: {sales_data['headers']}")
    print(f"   前3条数据:")
    for row in sales_data['data'][:3]:
        print(f"      {row}")
    
    print("\n3. 转换为JSON格式:")
    json_output = processor.to_json(sales_data)
    print(f"   JSON输出 (前500字符):")
    print(f"   {json_output[:500]}...")
    
    print("\n4. 提取指定范围数据 (行2-7, 列1-4):")
    range_data = processor.extract_range_data(
        sheet_name="销售数据",
        start_row=1,
        end_row=7,
        start_col=1,
        end_col=4,
        has_header=True
    )
    print(f"   数据行数: {range_data['row_count']}")
    print(f"   列数: {range_data['column_count']}")
    for row in range_data['data']:
        print(f"      {row}")
    
    print("\n5. 提取 '员工信息' 工作表数据:")
    emp_data = processor.extract_range_data("员工信息")
    print(f"   数据行数: {emp_data['row_count']}")
    for row in emp_data['data']:
        print(f"      {row}")
    
    print("\n6. 生成柱状图 (Base64):")
    try:
        bar_chart = processor.generate_bar_chart(
            sales_data,
            x_column="月份",
            y_columns=["产品A销量", "产品B销量", "产品C销量"],
            title="2024年各产品销量对比"
        )
        print(f"   柱状图Base64长度: {len(bar_chart)}")
        print(f"   前100字符: {bar_chart[:100]}...")
        
        with open("bar_chart.png", "wb") as f:
            f.write(base64.b64decode(bar_chart))
        print("   柱状图已保存为: bar_chart.png")
    except Exception as e:
        print(f"   生成柱状图失败: {e}")
    
    print("\n7. 生成折线图 (Base64):")
    try:
        line_chart = processor.generate_line_chart(
            sales_data,
            x_column="月份",
            y_columns=["产品A销量", "产品B销量"],
            title="产品销量趋势图"
        )
        print(f"   折线图Base64长度: {len(line_chart)}")
        print(f"   前100字符: {line_chart[:100]}...")
        
        with open("line_chart.png", "wb") as f:
            f.write(base64.b64decode(line_chart))
        print("   折线图已保存为: line_chart.png")
    except Exception as e:
        print(f"   生成折线图失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试 process_excel 便捷函数")
    print("=" * 60)
    
    result = process_excel(
        file_path="sample_data.xlsx",
        sheet_name="销售数据",
        start_row=1,
        end_row=6,
        start_col=1,
        end_col=4,
        has_header=True,
        chart_type="bar",
        x_column="月份",
        y_columns=["产品A销量", "产品B销量"],
        chart_title="上半年销量对比"
    )
    
    print(f"\n返回数据包含: {list(result.keys())}")
    print(f"数据行数: {result['data']['row_count']}")
    print(f"图表类型: {result.get('chart_type')}")
    print(f"图表Base64长度: {len(result.get('chart_base64', ''))}")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_excel_processor()

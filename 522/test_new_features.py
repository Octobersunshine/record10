import os
import sys

instance_dir = os.path.join(os.path.dirname(__file__), 'instance')
db_path = os.path.join(instance_dir, 'notes.db')
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"已删除旧数据库: {db_path}")

import requests
import json
import threading
import time
from app import app

def run_server():
    app.run(port=5001, debug=False)

server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()
time.sleep(2)

BASE_URL = 'http://localhost:5001'

def test_create_note_with_tags():
    print("=== 测试创建带标签的笔记 ===")
    data = {
        'title': 'Python学习笔记',
        'content': 'Flask是一个轻量级的Web框架，非常适合快速开发API。',
        'tags': ['Python', 'Flask', 'Web开发']
    }
    response = requests.post(f'{BASE_URL}/notes', json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result.get('id'), result.get('version_count')

def test_create_note2():
    print("\n=== 创建第二条笔记 ===")
    data = {
        'title': '数据库设计教程',
        'content': '关系型数据库设计的三范式原则，以及索引优化技巧。',
        'tags': ['数据库', 'MySQL', '后端']
    }
    response = requests.post(f'{BASE_URL}/notes', json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.json().get('id')

def test_get_all_tags():
    print("\n=== 获取所有标签 ===")
    response = requests.get(f'{BASE_URL}/tags')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return len(response.json())

def test_filter_by_tag():
    print("\n=== 按标签筛选笔记 (标签: Python) ===")
    response = requests.get(f'{BASE_URL}/notes?tag=Python')
    print(f"状态码: {response.status_code}")
    notes = response.json()
    print(f"找到 {len(notes)} 条笔记")
    print(f"响应: {json.dumps(notes, ensure_ascii=False, indent=2)}")
    return len(notes) == 1

def test_filter_by_tag2():
    print("\n=== 按标签筛选笔记 (标签: 后端) ===")
    response = requests.get(f'{BASE_URL}/notes?tag=后端')
    print(f"状态码: {response.status_code}")
    notes = response.json()
    print(f"找到 {len(notes)} 条笔记")
    print(f"响应: {json.dumps(notes, ensure_ascii=False, indent=2)}")
    return len(notes) == 1

def test_update_note_with_tags(note_id):
    print(f"\n=== 更新笔记及其标签 (ID: {note_id}) ===")
    data = {
        'title': 'Python学习笔记 - 第二版',
        'content': 'Flask是一个轻量级的Web框架，非常适合快速开发API。更新：添加了更多内容。',
        'tags': ['Python', 'Flask', 'API', '教程']
    }
    response = requests.put(f'{BASE_URL}/notes/{note_id}', json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result.get('version_count')

def test_get_version_history(note_id):
    print(f"\n=== 获取笔记版本历史 (ID: {note_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}/versions')
    print(f"状态码: {response.status_code}")
    versions = response.json()
    print(f"共有 {len(versions)} 个版本")
    print(f"响应: {json.dumps(versions, ensure_ascii=False, indent=2)}")
    return versions

def test_get_version_detail(note_id, version_id):
    print(f"\n=== 获取版本详情 (笔记ID: {note_id}, 版本ID: {version_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}/versions/{version_id}')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.json()

def test_restore_version(note_id, version_id):
    print(f"\n=== 恢复到历史版本 (笔记ID: {note_id}, 版本ID: {version_id}) ===")
    response = requests.post(f'{BASE_URL}/notes/{note_id}/versions/{version_id}/restore')
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result.get('note', {}).get('version_count')

def test_export_markdown(note_id):
    print(f"\n=== 导出Markdown (ID: {note_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}/export/markdown')
    print(f"状态码: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Disposition: {response.headers.get('Content-Disposition')}")
    print("内容预览:")
    print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
    
    md_path = os.path.join(os.path.dirname(__file__), 'export_test.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"\nMarkdown文件已保存到: {md_path}")
    return response.status_code == 200

def test_export_pdf(note_id):
    print(f"\n=== 导出PDF (ID: {note_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}/export/pdf')
    print(f"状态码: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Disposition: {response.headers.get('Content-Disposition')}")
    print(f"文件大小: {len(response.content)} bytes")
    
    pdf_path = os.path.join(os.path.dirname(__file__), 'export_test.pdf')
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"\nPDF文件已保存到: {pdf_path}")
    return response.status_code == 200

if __name__ == '__main__':
    try:
        all_passed = True
        
        print("\n" + "="*60)
        print("测试标签功能")
        print("="*60)
        
        note_id1, initial_version = test_create_note_with_tags()
        note_id2 = test_create_note2()
        
        tag_count = test_get_all_tags()
        print(f"\n标签总数: {tag_count} (期望: 6)")
        if tag_count != 6:
            all_passed = False
        
        if not test_filter_by_tag():
            all_passed = False
        if not test_filter_by_tag2():
            all_passed = False
        
        print("\n" + "="*60)
        print("测试版本历史功能")
        print("="*60)
        
        print(f"\n初始版本号: {initial_version} (期望: 1)")
        if initial_version != 1:
            all_passed = False
        
        new_version = test_update_note_with_tags(note_id1)
        print(f"\n更新后版本号: {new_version} (期望: 2)")
        if new_version != 2:
            all_passed = False
        
        versions = test_get_version_history(note_id1)
        if len(versions) != 2:
            print(f"版本数量错误: {len(versions)} (期望: 2)")
            all_passed = False
        
        if versions:
            first_version = versions[-1]
            version_detail = test_get_version_detail(note_id1, first_version['id'])
            
            restored_version = test_restore_version(note_id1, first_version['id'])
            print(f"\n恢复后新版本号: {restored_version} (期望: 3)")
            if restored_version != 3:
                all_passed = False
        
        print("\n" + "="*60)
        print("测试导出功能")
        print("="*60)
        
        if not test_export_markdown(note_id2):
            all_passed = False
        
        try:
            if not test_export_pdf(note_id2):
                all_passed = False
        except Exception as e:
            print(f"PDF导出跳过 (可能缺少依赖): {e}")
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ 所有新功能测试通过！")
        else:
            print("✗ 部分测试失败！")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()

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
    app.run(port=5000, debug=False)

server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()
time.sleep(2)

BASE_URL = 'http://localhost:5000'

def test_create_note():
    print("=== 测试创建笔记 ===")
    data = {
        'title': 'Python学习笔记',
        'content': 'Flask是一个轻量级的Web框架，非常适合快速开发API。'
    }
    response = requests.post(f'{BASE_URL}/notes', json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.json().get('id')

def test_create_note2():
    print("\n=== 创建第二条笔记 ===")
    data = {
        'title': 'SQLAlchemy教程',
        'content': 'SQLAlchemy是Python中最流行的ORM框架，可以简化数据库操作。'
    }
    response = requests.post(f'{BASE_URL}/notes', json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.json().get('id')

def test_get_all_notes():
    print("\n=== 获取所有笔记 ===")
    response = requests.get(f'{BASE_URL}/notes')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return len(response.json())

def test_update_nonexistent():
    print("\n=== 测试更新不存在的笔记 (ID: 9999) ===")
    data = {'title': '测试'}
    response = requests.put(f'{BASE_URL}/notes/9999', json=data)
    print(f"状态码: {response.status_code} (期望: 404)")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 404

def test_get_nonexistent():
    print("\n=== 测试获取不存在的笔记 (ID: 9999) ===")
    response = requests.get(f'{BASE_URL}/notes/9999')
    print(f"状态码: {response.status_code} (期望: 404)")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 404

def test_delete_nonexistent():
    print("\n=== 测试删除不存在的笔记 (ID: 9999) ===")
    response = requests.delete(f'{BASE_URL}/notes/9999')
    print(f"状态码: {response.status_code} (期望: 404)")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 404

def test_get_note_detail(note_id):
    print(f"\n=== 获取笔记详情 (ID: {note_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_update_note(note_id):
    print(f"\n=== 更新笔记 (ID: {note_id}) ===")
    data = {
        'title': 'Python学习笔记 - 更新版',
        'content': 'Flask是一个轻量级的Web框架，非常适合快速开发API。更新：添加了更多关于路由的内容。'
    }
    response = requests.put(f'{BASE_URL}/notes/{note_id}', json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_delete_note(note_id):
    print(f"\n=== 删除笔记 (ID: {note_id}) ===")
    response = requests.delete(f'{BASE_URL}/notes/{note_id}')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_get_deleted_note(note_id):
    print(f"\n=== 测试获取已删除的笔记 (ID: {note_id}) ===")
    response = requests.get(f'{BASE_URL}/notes/{note_id}')
    print(f"状态码: {response.status_code} (期望: 404 - 软删除后不可见)")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 404

def test_update_deleted_note(note_id):
    print(f"\n=== 测试更新已删除的笔记 (ID: {note_id}) ===")
    data = {'title': '测试'}
    response = requests.put(f'{BASE_URL}/notes/{note_id}', json=data)
    print(f"状态码: {response.status_code} (期望: 404 - 软删除后不可更新)")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.status_code == 404

def test_id_not_reused():
    print("\n=== 测试ID不复用（创建新笔记检查ID） ===")
    data = {
        'title': '第三条笔记',
        'content': '测试ID是否从3开始，证明ID没有复用'
    }
    response = requests.post(f'{BASE_URL}/notes', json=data)
    print(f"状态码: {response.status_code}")
    new_id = response.json().get('id')
    print(f"新笔记ID: {new_id} (期望 > 2，证明ID没有复用)")
    return new_id > 2

if __name__ == '__main__':
    try:
        all_passed = True
        
        note_id1 = test_create_note()
        note_id2 = test_create_note2()
        
        count_before = test_get_all_notes()
        
        print("\n" + "="*50)
        print("测试404错误处理")
        print("="*50)
        
        if not test_update_nonexistent():
            all_passed = False
        if not test_get_nonexistent():
            all_passed = False
        if not test_delete_nonexistent():
            all_passed = False
        
        print("\n" + "="*50)
        print("测试正常CRUD操作")
        print("="*50)
        
        test_get_note_detail(note_id1)
        test_update_note(note_id1)
        test_delete_note(note_id1)
        
        print("\n" + "="*50)
        print("测试软删除功能")
        print("="*50)
        
        if not test_get_deleted_note(note_id1):
            all_passed = False
        if not test_update_deleted_note(note_id1):
            all_passed = False
        
        count_after = test_get_all_notes()
        print(f"\n删除前笔记数: {count_before}, 删除后笔记数: {count_after}")
        print(f"软删除验证: {'通过' if count_after == count_before - 1 else '失败'}")
        
        print("\n" + "="*50)
        print("测试ID不复用")
        print("="*50)
        
        if not test_id_not_reused():
            all_passed = False
        
        print("\n" + "="*50)
        if all_passed:
            print("✓ 所有测试通过！")
        else:
            print("✗ 部分测试失败！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()

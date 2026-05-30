import requests
import json


def test_mask_api():
    url = 'http://localhost:5000/api/mask'

    payload = {
        'data': {
            'name': '张三',
            'phone': '13880001234',
            'id_card': '110101199001011234',
            'email': 'zhangsan@example.com',
            'user_info': {
                'mobile': '13999998888',
                'contact_email': 'lisi@test.org'
            },
            'orders': [
                {'id': 1, 'phone': '13666665555'},
                {'id': 2, 'phone': '13777774444'}
            ],
            'groups': [
                {
                    'name': 'A组',
                    'members': [
                        {'name': '李四', 'phone': '13111112222', 'email': 'lisi@test.com'}
                    ]
                }
            ]
        },
        'field_mappings': {
            'phone': 'phone',
            'id_card': 'id_card',
            'email': 'email',
            'mobile': 'phone',
            'contact_email': 'email',
            'orders.*.phone': 'phone',
            'groups.*.members.*.phone': 'phone',
            'groups.*.members.*.email': 'email'
        }
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print('Status Code:', response.status_code)
        print('Response:')
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except requests.exceptions.ConnectionError:
        print('无法连接到API服务，请先运行: python app.py')
        print('\n示例请求格式:')
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    test_mask_api()

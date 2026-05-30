import unittest
import re
from mask_utils import (
    mask_phone, mask_id_card, mask_email,
    mask_regex, mask_encrypt, mask_decrypt, mask_random,
    mask_value, mask_json_data, unmask_json_data,
    _match_path, _classify_char, _build_sub_table, _build_rev_table,
    _random_phone, _random_id_card, _random_email, _random_name, _random_default
)


class TestMaskUtils(unittest.TestCase):

    def test_mask_phone(self):
        self.assertEqual(mask_phone('13812345678'), '138****5678')
        self.assertEqual(mask_phone('13880001234'), '138****1234')
        self.assertEqual(mask_phone('123'), '123')
        self.assertEqual(mask_phone(''), '')
        self.assertEqual(mask_phone(None), None)

    def test_mask_id_card(self):
        self.assertEqual(mask_id_card('110101199001011234'), '1101***1234')
        self.assertEqual(mask_id_card('11011234'), '1101***1234')
        self.assertEqual(mask_id_card('12345'), '12345')
        self.assertEqual(mask_id_card(''), '')

    def test_mask_email(self):
        self.assertEqual(mask_email('ab@example.com'), 'ab@example.com')
        self.assertEqual(mask_email('abc@example.com'), 'ab***@example.com')
        self.assertEqual(mask_email('a@example.com'), 'a@example.com')
        self.assertEqual(mask_email('test.user@example.com'), 'te***@example.com')
        self.assertEqual(mask_email('zhangsan@example.com'), 'zh***@example.com')
        self.assertEqual(mask_email('invalid'), 'invalid')
        self.assertEqual(mask_email(''), '')


class TestMaskRegex(unittest.TestCase):

    def test_phone_regex(self):
        result = mask_regex('13812345678', r'(\d{3})\d{4}(\d{4})', r'\1****\2')
        self.assertEqual(result, '138****5678')

    def test_id_card_regex(self):
        result = mask_regex('110101199001011234', r'(\d{4})\d{10}(\d{4})', r'\1**********\2')
        self.assertEqual(result, '1101**********1234')

    def test_email_regex(self):
        result = mask_regex('zhangsan@example.com', r'(\w{2})\w+(@)', r'\1***\2')
        self.assertEqual(result, 'zh***@example.com')

    def test_custom_regex_multiple_groups(self):
        result = mask_regex('ABC-1234-XYZ', r'([A-Z]{3})-(\d{4})-([A-Z]{3})', r'\1-****-\3')
        self.assertEqual(result, 'ABC-****-XYZ')

    def test_empty_value(self):
        self.assertEqual(mask_regex('', r'\d', r'*'), '')
        self.assertEqual(mask_regex(None, r'\d', r'*'), None)

    def test_no_match(self):
        result = mask_regex('hello', r'\d+', r'***')
        self.assertEqual(result, 'hello')


class TestMaskEncrypt(unittest.TestCase):

    def test_encrypt_decrypt_roundtrip(self):
        key = 'my-secret-key'
        original = '13812345678'
        encrypted = mask_encrypt(original, key)
        self.assertNotEqual(encrypted, original)
        decrypted = mask_decrypt(encrypted, key)
        self.assertEqual(decrypted, original)

    def test_encrypt_preserves_format(self):
        key = 'test-key'
        original = '13812345678'
        encrypted = mask_encrypt(original, key)
        self.assertTrue(encrypted.isdigit())
        self.assertEqual(len(encrypted), len(original))

    def test_encrypt_email_preserves_format(self):
        key = 'test-key'
        original = 'test@example.com'
        encrypted = mask_encrypt(original, key)
        decrypted = mask_decrypt(encrypted, key)
        self.assertEqual(decrypted, original)
        self.assertIn('@', encrypted)
        self.assertIn('.', encrypted)
        at_idx_orig = original.index('@')
        at_idx_enc = encrypted.index('@')
        self.assertEqual(at_idx_orig, at_idx_enc)

    def test_encrypt_id_card(self):
        key = 'secret'
        original = '110101199001011234'
        encrypted = mask_encrypt(original, key)
        decrypted = mask_decrypt(encrypted, key)
        self.assertEqual(decrypted, original)
        self.assertTrue(encrypted.isdigit())

    def test_different_keys_produce_different_output(self):
        original = '13812345678'
        enc1 = mask_encrypt(original, 'key1')
        enc2 = mask_encrypt(original, 'key2')
        self.assertNotEqual(enc1, enc2)

    def test_same_key_same_result(self):
        original = '13812345678'
        key = 'consistent-key'
        enc1 = mask_encrypt(original, key)
        enc2 = mask_encrypt(original, key)
        self.assertEqual(enc1, enc2)

    def test_decrypt_with_wrong_key_fails(self):
        original = '13812345678'
        encrypted = mask_encrypt(original, 'correct-key')
        decrypted = mask_decrypt(encrypted, 'wrong-key')
        self.assertNotEqual(decrypted, original)

    def test_empty_value(self):
        self.assertEqual(mask_encrypt('', 'key'), '')
        self.assertEqual(mask_encrypt(None, 'key'), None)

    def test_empty_key(self):
        self.assertEqual(mask_encrypt('13812345678', ''), '13812345678')

    def test_mixed_charset(self):
        key = 'test'
        original = 'Abc123!@#'
        encrypted = mask_encrypt(original, key)
        decrypted = mask_decrypt(encrypted, key)
        self.assertEqual(decrypted, original)
        self.assertTrue(encrypted[3].isdigit())
        self.assertTrue(encrypted[0].isupper())
        self.assertTrue(encrypted[1].islower())
        self.assertEqual(encrypted[6], '!')
        self.assertEqual(encrypted[7], '@')
        self.assertEqual(encrypted[8], '#')


class TestMaskRandom(unittest.TestCase):

    def test_random_phone(self):
        result = mask_random('13812345678', 'phone')
        self.assertTrue(result.startswith('1'))
        self.assertEqual(len(result), 11)
        self.assertTrue(result.isdigit())

    def test_random_id_card(self):
        result = mask_random('110101199001011234', 'id_card')
        self.assertEqual(len(result), 18)
        self.assertTrue(result.isdigit())

    def test_random_email(self):
        result = mask_random('test@example.com', 'email')
        self.assertIn('@', result)
        parts = result.split('@')
        self.assertTrue(len(parts[0]) >= 4)
        self.assertTrue(len(parts[1]) > 0)

    def test_random_name(self):
        result = mask_random('张三', 'name')
        self.assertTrue(len(result) >= 2)

    def test_random_default_preserves_format(self):
        original = 'ABC-1234'
        result = mask_random(original, 'default')
        self.assertEqual(len(result), len(original))
        self.assertTrue(result[:3].isupper())
        self.assertEqual(result[3], '-')
        self.assertTrue(result[4:].isdigit())

    def test_random_default_preserves_special_chars(self):
        original = 'test@example.com'
        result = mask_random(original, 'default')
        self.assertEqual(len(result), len(original))
        self.assertEqual(result[4], '@')
        self.assertEqual(result[12], '.')

    def test_random_irreversible(self):
        original = '13812345678'
        result = mask_random(original, 'phone')
        self.assertNotEqual(result, original)

    def test_empty_value(self):
        self.assertEqual(mask_random('', 'phone'), '')
        self.assertEqual(mask_random(None, 'phone'), None)


class TestMaskValue(unittest.TestCase):

    def test_string_config_builtin(self):
        self.assertEqual(mask_value('13812345678', 'phone'), '138****5678')

    def test_dict_config_regex(self):
        config = {'type': 'regex', 'pattern': r'(\d{3})\d{4}(\d{4})', 'replace': r'\1****\2'}
        self.assertEqual(mask_value('13812345678', config), '138****5678')

    def test_dict_config_encrypt(self):
        config = {'type': 'encrypt', 'key': 'test-key'}
        encrypted = mask_value('13812345678', config)
        self.assertNotEqual(encrypted, '13812345678')
        decrypted = mask_decrypt(encrypted, 'test-key')
        self.assertEqual(decrypted, '13812345678')

    def test_dict_config_random(self):
        config = {'type': 'random', 'category': 'phone'}
        result = mask_value('13812345678', config)
        self.assertEqual(len(result), 11)
        self.assertTrue(result.isdigit())

    def test_dict_config_builtin_type(self):
        config = {'type': 'phone'}
        self.assertEqual(mask_value('13812345678', config), '138****5678')

    def test_unknown_string_type(self):
        self.assertEqual(mask_value('13812345678', 'unknown'), '13812345678')

    def test_unknown_dict_type(self):
        config = {'type': 'nonexistent'}
        self.assertEqual(mask_value('13812345678', config), '13812345678')


class TestMatchPath(unittest.TestCase):

    def test_exact_match(self):
        self.assertTrue(_match_path('phone', 'phone'))
        self.assertTrue(_match_path('user.phone', 'user.phone'))

    def test_wildcard_match(self):
        self.assertTrue(_match_path('user.*.phone', 'user.contact.phone'))
        self.assertTrue(_match_path('*.phone', 'contact.phone'))
        self.assertTrue(_match_path('*.*.phone', 'user.contact.phone'))

    def test_no_match(self):
        self.assertFalse(_match_path('phone', 'email'))
        self.assertFalse(_match_path('user.phone', 'admin.phone'))
        self.assertFalse(_match_path('user.*.phone', 'user.phone'))
        self.assertFalse(_match_path('a.b', 'a.b.c'))

    def test_multiple_wildcards(self):
        self.assertTrue(_match_path('*.*.phone', 'a.b.phone'))
        self.assertTrue(_match_path('*.*.*.phone', 'a.b.c.phone'))
        self.assertFalse(_match_path('*.*.phone', 'a.phone'))


class TestMaskJsonData(unittest.TestCase):

    def test_simple_flat_data(self):
        data = {
            'name': '张三',
            'phone': '13812345678',
            'id_card': '110101199001011234',
            'email': 'zhangsan@example.com',
            'address': '北京市朝阳区'
        }
        field_mappings = {
            'phone': 'phone',
            'id_card': 'id_card',
            'email': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], '138****5678')
        self.assertEqual(result['id_card'], '1101***1234')
        self.assertEqual(result['email'], 'zh***@example.com')
        self.assertEqual(result['name'], '张三')
        self.assertEqual(result['address'], '北京市朝阳区')

    def test_nested_dict_with_simple_mapping(self):
        data = {
            'user': {
                'name': '李四',
                'contact': {
                    'phone': '13987654321',
                    'email': 'lisi@test.com'
                }
            },
            'id_card': '310101198505055678'
        }
        field_mappings = {
            'phone': 'phone',
            'id_card': 'id_card',
            'email': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['user']['contact']['phone'], '139****4321')
        self.assertEqual(result['user']['contact']['email'], 'li***@test.com')
        self.assertEqual(result['id_card'], '3101***5678')

    def test_list_data_with_simple_mapping(self):
        data = [
            {'name': '用户1', 'phone': '13800001111'},
            {'name': '用户2', 'phone': '13900002222'}
        ]
        field_mappings = {'phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result[0]['phone'], '138****1111')
        self.assertEqual(result[1]['phone'], '139****2222')

    def test_wildcard_path_single_level(self):
        data = {
            'admin': {'phone': '13800001111'},
            'guest': {'phone': '13900002222'}
        }
        field_mappings = {'*.phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['admin']['phone'], '138****1111')
        self.assertEqual(result['guest']['phone'], '139****2222')

    def test_wildcard_path_in_array(self):
        data = {
            'users': [
                {'name': '用户1', 'phone': '13800001111'},
                {'name': '用户2', 'phone': '13900002222'}
            ]
        }
        field_mappings = {'users.*.phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['users'][0]['phone'], '138****1111')
        self.assertEqual(result['users'][1]['phone'], '139****2222')
        self.assertEqual(result['users'][0]['name'], '用户1')

    def test_wildcard_path_deeply_nested(self):
        data = {
            'groups': [
                {
                    'name': 'A组',
                    'members': [
                        {'name': '张三', 'phone': '13800001111'},
                        {'name': '李四', 'phone': '13900002222'}
                    ]
                },
                {
                    'name': 'B组',
                    'members': [
                        {'name': '王五', 'phone': '13700003333'}
                    ]
                }
            ]
        }
        field_mappings = {'groups.*.members.*.phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['groups'][0]['members'][0]['phone'], '138****1111')
        self.assertEqual(result['groups'][0]['members'][1]['phone'], '139****2222')
        self.assertEqual(result['groups'][1]['members'][0]['phone'], '137****3333')
        self.assertEqual(result['groups'][0]['name'], 'A组')

    def test_path_mapping_precedence_over_simple(self):
        data = {
            'info': '13800001111',
            'contact': {
                'info': 'testuser@example.com'
            }
        }
        field_mappings = {
            'info': 'phone',
            'contact.info': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['info'], '138****1111')
        self.assertEqual(result['contact']['info'], 'te***@example.com')

    def test_wildcard_path_dict_key(self):
        data = {
            'contacts': {
                'alice': {'phone': '13800001111', 'email': 'alice@test.com'},
                'bob': {'phone': '13900002222', 'email': 'bob@test.com'}
            }
        }
        field_mappings = {
            'contacts.*.phone': 'phone',
            'contacts.*.email': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['contacts']['alice']['phone'], '138****1111')
        self.assertEqual(result['contacts']['bob']['phone'], '139****2222')
        self.assertEqual(result['contacts']['alice']['email'], 'al***@test.com')
        self.assertEqual(result['contacts']['bob']['email'], 'bo***@test.com')

    def test_mixed_simple_and_path_mappings(self):
        data = {
            'phone': '13800001111',
            'users': [
                {'phone': '13900002222', 'email': 'user1@test.com'}
            ],
            'admin': {
                'phone': '13700003333'
            }
        }
        field_mappings = {
            'phone': 'phone',
            'users.*.email': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], '138****1111')
        self.assertEqual(result['users'][0]['phone'], '139****2222')
        self.assertEqual(result['users'][0]['email'], 'us***@test.com')
        self.assertEqual(result['admin']['phone'], '137****3333')

    def test_nested_list_inside_list(self):
        data = {
            'matrix': [
                [{'phone': '13800001111'}, {'phone': '13900002222'}],
                [{'phone': '13700003333'}]
            ]
        }
        field_mappings = {'matrix.*.*.phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['matrix'][0][0]['phone'], '138****1111')
        self.assertEqual(result['matrix'][0][1]['phone'], '139****2222')
        self.assertEqual(result['matrix'][1][0]['phone'], '137****3333')

    def test_non_string_value_with_matching_key_is_preserved(self):
        data = {
            'phone': 13800001111,
            'email': None,
            'name': '张三'
        }
        field_mappings = {'phone': 'phone', 'email': 'email'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], 13800001111)
        self.assertIsNone(result['email'])
        self.assertEqual(result['name'], '张三')

    def test_empty_mappings_returns_original(self):
        data = {'phone': '13800001111', 'name': '张三'}
        result = mask_json_data(data, {})
        self.assertEqual(result['phone'], '13800001111')
        self.assertEqual(result['name'], '张三')

    def test_root_level_list_with_path_mapping(self):
        data = [
            {'phone': '13800001111'},
            {'phone': '13900002222'}
        ]
        field_mappings = {'*.phone': 'phone'}
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result[0]['phone'], '138****1111')
        self.assertEqual(result[1]['phone'], '139****2222')

    def test_all_types_together(self):
        data = {
            'name': '系统',
            'phone': '13800001111',
            'id_card': '110101199001011234',
            'email': 'admin@example.com',
            'users': [
                {
                    'name': '用户1',
                    'phone': '13900002222',
                    'id_card': '310101198505055678',
                    'email': 'user1@test.com',
                    'contacts': [
                        {'phone': '13700003333', 'email': 'contact1@org.com'}
                    ]
                }
            ],
            'metadata': {
                'admin': {
                    'phone': '13600004444'
                }
            }
        }
        field_mappings = {
            'phone': 'phone',
            'id_card': 'id_card',
            'email': 'email',
            'users.*.contacts.*.phone': 'phone',
            'users.*.contacts.*.email': 'email'
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], '138****1111')
        self.assertEqual(result['id_card'], '1101***1234')
        self.assertEqual(result['email'], 'ad***@example.com')
        self.assertEqual(result['users'][0]['phone'], '139****2222')
        self.assertEqual(result['users'][0]['id_card'], '3101***5678')
        self.assertEqual(result['users'][0]['email'], 'us***@test.com')
        self.assertEqual(result['users'][0]['contacts'][0]['phone'], '137****3333')
        self.assertEqual(result['users'][0]['contacts'][0]['email'], 'co***@org.com')
        self.assertEqual(result['metadata']['admin']['phone'], '136****4444')

    def test_dict_config_regex_in_json(self):
        data = {
            'phone': '13812345678',
            'custom_field': 'ABC-1234-XYZ'
        }
        field_mappings = {
            'phone': {'type': 'regex', 'pattern': r'(\d{3})\d{4}(\d{4})', 'replace': r'\1****\2'},
            'custom_field': {'type': 'regex', 'pattern': r'([A-Z]{3})-(\d{4})-([A-Z]{3})', 'replace': r'\1-****-\3'}
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], '138****5678')
        self.assertEqual(result['custom_field'], 'ABC-****-XYZ')

    def test_dict_config_encrypt_in_json(self):
        data = {
            'phone': '13812345678',
            'email': 'test@example.com'
        }
        field_mappings = {
            'phone': {'type': 'encrypt', 'key': 'mykey'},
            'email': {'type': 'encrypt', 'key': 'mykey'}
        }
        result = mask_json_data(data, field_mappings)
        self.assertNotEqual(result['phone'], '13812345678')
        self.assertNotEqual(result['email'], 'test@example.com')
        self.assertTrue(result['phone'].isdigit())
        self.assertIn('@', result['email'])

    def test_dict_config_random_in_json(self):
        data = {
            'phone': '13812345678',
            'name': '张三'
        }
        field_mappings = {
            'phone': {'type': 'random', 'category': 'phone'},
            'name': {'type': 'random', 'category': 'name'}
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(len(result['phone']), 11)
        self.assertTrue(result['phone'].isdigit())
        self.assertTrue(len(result['name']) >= 2)

    def test_mixed_str_and_dict_configs(self):
        data = {
            'phone': '13812345678',
            'id_card': '110101199001011234',
            'email': 'test@example.com',
            'custom': 'ABC-1234'
        }
        field_mappings = {
            'phone': 'phone',
            'id_card': {'type': 'encrypt', 'key': 'secret'},
            'email': {'type': 'random', 'category': 'email'},
            'custom': {'type': 'regex', 'pattern': r'([A-Z]{3})-(\d{4})', 'replace': r'\1-****'}
        }
        result = mask_json_data(data, field_mappings)
        self.assertEqual(result['phone'], '138****5678')
        self.assertNotEqual(result['id_card'], '110101199001011234')
        decrypted = mask_decrypt(result['id_card'], 'secret')
        self.assertEqual(decrypted, '110101199001011234')
        self.assertIn('@', result['email'])
        self.assertEqual(result['custom'], 'ABC-****')


class TestUnmaskJsonData(unittest.TestCase):

    def test_decrypt_roundtrip(self):
        data = {
            'phone': '13812345678',
            'email': 'test@example.com',
            'name': '张三'
        }
        field_mappings = {
            'phone': {'type': 'encrypt', 'key': 'mykey'},
            'email': {'type': 'encrypt', 'key': 'mykey'}
        }
        masked = mask_json_data(data, field_mappings)
        self.assertNotEqual(masked['phone'], data['phone'])
        self.assertNotEqual(masked['email'], data['email'])
        self.assertEqual(masked['name'], '张三')

        unmasked = unmask_json_data(masked, field_mappings)
        self.assertEqual(unmasked['phone'], data['phone'])
        self.assertEqual(unmasked['email'], data['email'])
        self.assertEqual(unmasked['name'], '张三')

    def test_decrypt_nested(self):
        data = {
            'user': {
                'phone': '13812345678',
                'email': 'test@example.com'
            }
        }
        field_mappings = {
            'phone': {'type': 'encrypt', 'key': 'key1'},
            'email': {'type': 'encrypt', 'key': 'key2'}
        }
        masked = mask_json_data(data, field_mappings)
        unmasked = unmask_json_data(masked, field_mappings)
        self.assertEqual(unmasked['user']['phone'], '13812345678')
        self.assertEqual(unmasked['user']['email'], 'test@example.com')

    def test_decrypt_with_array(self):
        data = {
            'users': [
                {'phone': '13800001111'},
                {'phone': '13900002222'}
            ]
        }
        field_mappings = {
            'phone': {'type': 'encrypt', 'key': 'key'}
        }
        masked = mask_json_data(data, field_mappings)
        unmasked = unmask_json_data(masked, field_mappings)
        self.assertEqual(unmasked['users'][0]['phone'], '13800001111')
        self.assertEqual(unmasked['users'][1]['phone'], '13900002222')

    def test_decrypt_only_affects_encrypt_type(self):
        data = {
            'phone': '13812345678',
            'name': '张三'
        }
        field_mappings = {
            'phone': {'type': 'encrypt', 'key': 'mykey'},
            'name': 'phone'
        }
        masked = mask_json_data(data, field_mappings)
        unmasked = unmask_json_data(masked, field_mappings)
        self.assertEqual(unmasked['phone'], '13812345678')

    def test_decrypt_with_wildcard_path(self):
        data = {
            'contacts': {
                'alice': {'phone': '13800001111'},
                'bob': {'phone': '13900002222'}
            }
        }
        field_mappings = {
            'contacts.*.phone': {'type': 'encrypt', 'key': 'mykey'}
        }
        masked = mask_json_data(data, field_mappings)
        unmasked = unmask_json_data(masked, field_mappings)
        self.assertEqual(unmasked['contacts']['alice']['phone'], '13800001111')
        self.assertEqual(unmasked['contacts']['bob']['phone'], '13900002222')


class TestClassifyChar(unittest.TestCase):

    def test_digits(self):
        for ch in '0123456789':
            self.assertEqual(_classify_char(ch), '0123456789')

    def test_lowercase(self):
        for ch in 'abcxyz':
            self.assertEqual(_classify_char(ch), 'abcdefghijklmnopqrstuvwxyz')

    def test_uppercase(self):
        for ch in 'ABCXYZ':
            self.assertEqual(_classify_char(ch), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def test_special_chars(self):
        self.assertIsNone(_classify_char('@'))
        self.assertIsNone(_classify_char('.'))
        self.assertIsNone(_classify_char('-'))
        self.assertIsNone(_classify_char('中'))


class TestSubTables(unittest.TestCase):

    def test_sub_table_is_bijection(self):
        import string
        for charset in [string.digits, string.ascii_lowercase, string.ascii_uppercase]:
            table = _build_sub_table('test-key', charset)
            values = list(table.values())
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(set(values), set(charset))

    def test_rev_table_reverses_sub_table(self):
        import string
        for charset in [string.digits, string.ascii_lowercase, string.ascii_uppercase]:
            fwd = _build_sub_table('test-key', charset)
            rev = _build_rev_table('test-key', charset)
            for k, v in fwd.items():
                self.assertEqual(rev[v], k)


if __name__ == '__main__':
    unittest.main()

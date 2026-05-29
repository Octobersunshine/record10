import regex as re
from flask import Flask, request, jsonify
from flask_cors import CORS
from itertools import product

app = Flask(__name__)
CORS(app)

REGEX_TIMEOUT = 2.0


class RegexGenerator:
    COMMON_PATTERNS = [
        (r'\d+', '数字'),
        (r'[a-zA-Z]+', '英文字母'),
        (r'[a-zA-Z0-9]+', '字母数字'),
        (r'[\u4e00-\u9fa5]+', '中文字符'),
        (r'[\w.-]+@[\w.-]+\.\w+', '邮箱地址'),
        (r'1[3-9]\d{9}', '手机号码'),
        (r'\d{4}-\d{2}-\d{2}', '日期格式(YYYY-MM-DD)'),
        (r'\d{4}/\d{2}/\d{2}', '日期格式(YYYY/MM/DD)'),
        (r'\d{2}:\d{2}(:\d{2})?', '时间格式'),
        (r'https?://[\w.-]+(?:/[\w./?%&=-]*)?', 'URL地址'),
    ]

    @staticmethod
    def char_to_pattern(c):
        if c.isdigit():
            return r'\d'
        elif c.isalpha():
            if '\u4e00' <= c <= '\u9fa5':
                return r'[\u4e00-\u9fa5]'
            else:
                return r'[a-zA-Z]'
        elif c.isspace():
            return r'\s'
        elif c in '.+*?^$()[]{}|\\':
            return '\\' + c
        else:
            return re.escape(c)

    @classmethod
    def analyze_string(cls, s):
        tokens = []
        if not s:
            return tokens
        
        current_type = None
        current_chars = []
        
        for c in s:
            if c.isdigit():
                char_type = 'digit'
            elif '\u4e00' <= c <= '\u9fa5':
                char_type = 'chinese'
            elif c.isalpha():
                char_type = 'alpha'
            elif c.isspace():
                char_type = 'space'
            else:
                char_type = 'special'
            
            if char_type == current_type:
                current_chars.append(c)
            else:
                if current_type is not None:
                    tokens.append((current_type, ''.join(current_chars)))
                current_type = char_type
                current_chars = [c]
        
        if current_type is not None:
            tokens.append((current_type, ''.join(current_chars)))
        
        return tokens

    @classmethod
    def token_to_patterns(cls, token_type, token_value):
        patterns = []
        
        if token_type == 'digit':
            if len(token_value) == 1:
                patterns.append((r'\d', 1.0))
            else:
                patterns.append((r'\d+', 0.9))
                patterns.append((r'\d{' + str(len(token_value)) + r'}', 0.8))
        elif token_type == 'chinese':
            patterns.append((r'[\u4e00-\u9fa5]+', 0.9))
            patterns.append((r'[\u4e00-\u9fa5]{' + str(len(token_value)) + r'}', 0.7))
        elif token_type == 'alpha':
            patterns.append((r'[a-zA-Z]+', 0.85))
            if token_value.islower():
                patterns.append((r'[a-z]+', 0.75))
            elif token_value.isupper():
                patterns.append((r'[A-Z]+', 0.75))
            patterns.append((re.escape(token_value), 0.6))
        elif token_type == 'space':
            if len(token_value) == 1:
                patterns.append((r'\s', 0.9))
            else:
                patterns.append((r'\s+', 0.85))
                patterns.append((r'\s{' + str(len(token_value)) + r'}', 0.7))
        else:
            patterns.append((re.escape(token_value), 1.0))
        
        return patterns

    @classmethod
    def generate_candidates(cls, positive_examples):
        if not positive_examples:
            return []
        
        all_tokens = [cls.analyze_string(s) for s in positive_examples]
        
        max_len = max(len(tokens) for tokens in all_tokens)
        aligned_tokens = []
        for tokens in all_tokens:
            if len(tokens) < max_len:
                continue
            aligned_tokens.append(tokens)
        
        if not aligned_tokens:
            aligned_tokens = all_tokens
        
        reference_tokens = aligned_tokens[0]
        
        token_pattern_options = []
        for i, (token_type, token_value) in enumerate(reference_tokens):
            all_same = True
            for tokens in aligned_tokens[1:]:
                if i >= len(tokens) or tokens[i][0] != token_type or tokens[i][1] != token_value:
                    all_same = False
                    break
            
            if all_same:
                token_pattern_options.append([(re.escape(token_value), 1.0)])
            else:
                all_same_type = True
                for tokens in aligned_tokens[1:]:
                    if i >= len(tokens) or tokens[i][0] != token_type:
                        all_same_type = False
                        break
                
                if all_same_type:
                    patterns = cls.token_to_patterns(token_type, token_value)
                    token_pattern_options.append(patterns)
                else:
                    token_pattern_options.append([(r'.+?', 0.3), (r'.*', 0.2)])
        
        candidates = []
        for combo in product(*[opt for opt in token_pattern_options]):
            pattern = '^' + ''.join(p for p, _ in combo) + '$'
            score = sum(s for _, s in combo) / len(combo)
            
            matches_all = True
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
                
            for example in positive_examples:
                try:
                    if not compiled.match(example, timeout=REGEX_TIMEOUT):
                        matches_all = False
                        break
                except (TimeoutError, re.error):
                    matches_all = False
                    break
            
            if matches_all:
                candidates.append({
                    'pattern': pattern,
                    'score': round(score, 3),
                    'description': cls.describe_pattern(pattern)
                })
        
        for pattern, desc in cls.COMMON_PATTERNS:
            matches_all = True
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
                
            for example in positive_examples:
                try:
                    if not compiled.fullmatch(example, timeout=REGEX_TIMEOUT):
                        matches_all = False
                        break
                except (TimeoutError, re.error):
                    matches_all = False
                    break
            
            if matches_all:
                candidates.append({
                    'pattern': pattern,
                    'score': 0.95,
                    'description': desc
                })
        
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand['pattern'] not in seen:
                seen.add(cand['pattern'])
                unique_candidates.append(cand)
        
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return unique_candidates

    @staticmethod
    def describe_pattern(pattern):
        descriptions = []
        if pattern.startswith('^'):
            pattern = pattern[1:]
        if pattern.endswith('$'):
            pattern = pattern[:-1]
        
        if r'\d+' in pattern:
            descriptions.append('包含数字')
        if r'[a-zA-Z]+' in pattern:
            descriptions.append('包含字母')
        if r'[\u4e00-\u9fa5]' in pattern:
            descriptions.append('包含中文')
        if r'\s' in pattern:
            descriptions.append('包含空格')
        
        if not descriptions:
            descriptions.append('精确匹配')
        
        return '，'.join(descriptions)

    @classmethod
    def filter_with_negative(cls, candidates, negative_examples):
        if not negative_examples:
            return candidates
        
        filtered = []
        for cand in candidates:
            try:
                compiled = re.compile(cand['pattern'])
            except re.error:
                continue
            
            matches_negative = False
            for neg in negative_examples:
                try:
                    if compiled.fullmatch(neg, timeout=REGEX_TIMEOUT):
                        matches_negative = True
                        break
                except TimeoutError:
                    matches_negative = True
                    break
            
            if not matches_negative:
                cand['filtered_by_negative'] = True
                filtered.append(cand)
        
        return filtered


@app.route('/api/regex/generate', methods=['POST'])
def generate_regex():
    data = request.get_json()

    if not data:
        return jsonify({
            'success': False,
            'error': '请求体不能为空'
        }), 400

    positive = data.get('positive', [])
    negative = data.get('negative', [])
    max_candidates = data.get('max_candidates', 10)

    if not positive:
        return jsonify({
            'success': False,
            'error': '请至少提供一个正例样本'
        }), 400

    if not isinstance(positive, list) or not isinstance(negative, list):
        return jsonify({
            'success': False,
            'error': '正例和反例必须是数组格式'
        }), 400

    for item in positive + negative:
        if not isinstance(item, str):
            return jsonify({
                'success': False,
                'error': '样本必须是字符串格式'
            }), 400

    try:
        candidates = RegexGenerator.generate_candidates(positive)
        
        if negative:
            filtered = RegexGenerator.filter_with_negative(candidates, negative)
            candidates = filtered if filtered else candidates[:max_candidates]
        else:
            candidates = candidates[:max_candidates]

        for cand in candidates:
            try:
                compiled = re.compile(cand['pattern'])
                cand['matches'] = {
                    'positive': [],
                    'negative': []
                }
                for ex in positive:
                    try:
                        m = compiled.fullmatch(ex, timeout=REGEX_TIMEOUT)
                        cand['matches']['positive'].append({
                            'example': ex,
                            'matches': bool(m)
                        })
                    except TimeoutError:
                        cand['matches']['positive'].append({
                            'example': ex,
                            'matches': False,
                            'error': '匹配超时'
                        })
                for ex in negative:
                    try:
                        m = compiled.fullmatch(ex, timeout=REGEX_TIMEOUT)
                        cand['matches']['negative'].append({
                            'example': ex,
                            'matches': bool(m)
                        })
                    except TimeoutError:
                        cand['matches']['negative'].append({
                            'example': ex,
                            'matches': True,
                            'error': '匹配超时'
                        })
            except (re.error, TimeoutError):
                cand['matches'] = {'error': '正则编译失败或超时'}

        response = {
            'success': True,
            'positive_count': len(positive),
            'negative_count': len(negative),
            'candidate_count': len(candidates),
            'candidates': candidates
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'生成失败: {str(e)}'
        }), 500


@app.route('/api/regex/test', methods=['POST'])
def test_regex():
    data = request.get_json()

    if not data:
        return jsonify({
            'success': False,
            'error': '请求体不能为空'
        }), 400

    pattern = data.get('pattern', '')
    test_string = data.get('test_string', '')
    flags = data.get('flags', [])

    if not pattern:
        return jsonify({
            'success': False,
            'error': '正则表达式不能为空'
        }), 400

    flag_value = 0
    flag_descriptions = []
    for flag in flags:
        flag = flag.upper()
        if hasattr(re, flag):
            flag_value |= getattr(re, flag)
            flag_descriptions.append(flag)
        else:
            return jsonify({
                'success': False,
                'error': f'无效的正则标志: {flag}'
            }), 400

    try:
        compiled_pattern = re.compile(pattern, flag_value)
    except re.error as e:
        return jsonify({
            'success': False,
            'error': f'正则表达式语法错误: {str(e)}',
            'error_position': e.pos if hasattr(e, 'pos') else None
        }), 400

    try:
        matches = []
        for match in compiled_pattern.finditer(test_string, timeout=REGEX_TIMEOUT):
            match_data = {
                'match': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'span': match.span(),
                'groups': [],
                'named_groups': {}
            }

            for i, group in enumerate(match.groups(), 1):
                match_data['groups'].append({
                    'index': i,
                    'value': group,
                    'start': match.start(i),
                    'end': match.end(i)
                })

            for name, value in match.groupdict().items():
                match_data['named_groups'][name] = {
                    'value': value,
                    'start': match.start(name),
                    'end': match.end(name)
                }

            matches.append(match_data)

        response = {
            'success': True,
            'pattern': pattern,
            'test_string': test_string,
            'flags': flag_descriptions,
            'match_count': len(matches),
            'matches': matches,
            'match_positions': [(m['start'], m['end']) for m in matches]
        }

        return jsonify(response)

    except TimeoutError:
        return jsonify({
            'success': False,
            'error': '正则过于复杂，匹配超时',
            'timeout': REGEX_TIMEOUT,
            'warning': '检测到灾难性回溯风险，请优化正则表达式'
        }), 408


@app.route('/api/regex/validate', methods=['POST'])
def validate_regex():
    data = request.get_json()

    if not data:
        return jsonify({
            'valid': False,
            'error': '请求体不能为空'
        }), 400

    pattern = data.get('pattern', '')
    flags = data.get('flags', [])

    if not pattern:
        return jsonify({
            'valid': False,
            'error': '正则表达式不能为空'
        }), 400

    flag_value = 0
    for flag in flags:
        flag = flag.upper()
        if hasattr(re, flag):
            flag_value |= getattr(re, flag)
        else:
            return jsonify({
                'valid': False,
                'error': f'无效的正则标志: {flag}'
            }), 400

    try:
        compiled_pattern = re.compile(pattern, flag_value)
    except re.error as e:
        return jsonify({
            'valid': False,
            'error': f'正则表达式语法错误: {str(e)}',
            'error_position': e.pos if hasattr(e, 'pos') else None
        })

    try:
        test_string = 'a' * 200
        list(compiled_pattern.finditer(test_string, timeout=REGEX_TIMEOUT))
        return jsonify({
            'valid': True,
            'pattern': pattern
        })
    except TimeoutError:
        return jsonify({
            'valid': False,
            'error': '正则过于复杂，存在灾难性回溯风险',
            'timeout': REGEX_TIMEOUT,
            'warning': '检测到灾难性回溯风险，请优化正则表达式'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

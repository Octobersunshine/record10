from pypinyin import pinyin, Style
from typing import List, Union, Dict, Any, Set, Tuple
import re


class PolyphoneDictionary:
    """
    多音字词库 - 包含常用多音字的不同读音和对应的词组
    """

    POLYPHONE_WORDS = {
        '行': {
            'xing': ['行走', '出行', '行动', '行为', '旅行', '运行', '进行', '自行', '不行', '可行', '步行', '行车', '行销', '行礼'],
            'hang': ['银行', '行业', '行列', '同行', '商行', '行长', '行家', '行会', '行规', '行话']
        },
        '重': {
            'zhong': ['重要', '重量', '重大', '严重', '沉重', '重视', '重点', '重伤', '重任', '重力', '体重', '尊重', '郑重'],
            'chong': ['重庆', '重复', '重新', '重来', '重逢', '重播', '重申', '重写', '重建', '重组', '重阳', '重叠']
        },
        '长': {
            'zhang': ['长大', '成长', '长辈', '校长', '班长', '增长', '长高', '部长', '院长', '厂长', '首长'],
            'chang': ['长短', '长江', '长城', '长度', '长远', '长久', '长期', '擅长', '特长', '延长', '伸长']
        },
        '乐': {
            'le': ['快乐', '欢乐', '乐观', '乐园', '乐意', '乐趣', '乐事', '乐子', '喜乐', '玩乐'],
            'yue': ['音乐', '乐器', '乐曲', '乐谱', '乐队', '乐章', '乐理', '乐坛', '声乐', '民乐']
        },
        '着': {
            'zhe': ['看着', '听着', '走着', '坐着', '穿着', '拿着', '带着'],
            'zhao': ['着急', '着火', '着凉', '着迷', '睡着', '着慌', '着火'],
            'zhuo': ['着手', '着重', '着眼', '着陆', '着想', '着力', '着色'],
            'zhao1': ['着数', '高着']
        },
        '了': {
            'le': ['好了', '来了', '走了', '吃了', '看了', '听了', '完了'],
            'liao': ['了解', '了结', '了不起', '了如指掌', '了事', '了断', '明了']
        },
        '都': {
            'dou': ['都是', '都有', '都好', '都行', '都来', '全都'],
            'du': ['首都', '都市', '都城', '成都', '古都', '国都']
        },
        '还': {
            'hai': ['还有', '还是', '还好', '还在', '还要', '还能'],
            'huan': ['还书', '还钱', '归还', '还债', '还手', '还原', '还礼']
        },
        '种': {
            'zhong1': ['种子', '种类', '品种', '种族', '播种', '物种', '种种'],
            'zhong4': ['种地', '种田', '种植', '种花', '种树', '耕种']
        },
        '得': {
            'de': ['得到', '获得', '得意', '得分', '得救', '得体'],
            'de0': ['跑得快', '飞得高', '做得好', '说得对', '长得帅'],
            'dei': ['得亏', '总得', '得去', '得有']
        },
        '地': {
            'di': ['土地', '地方', '地球', '地区', '地址', '地面', '地理', '天地'],
            'de0': ['慢慢地', '轻轻地', '悄悄地', '快乐地', '认真地']
        },
        '发': {
            'fa': ['发现', '发生', '发送', '发展', '发布', '出发', '发表'],
            'fa4': ['头发', '毛发', '理发', '发型', '白发', '黑发']
        },
        '干': {
            'gan': ['干净', '干燥', '干杯', '干涉', '干枯', '干旱', '干洗'],
            'gan4': ['干活', '干事', '干部', '干练', '干线', '干劲', '才干']
        }
    }

    @classmethod
    def get_polyphone_chars(cls) -> Set[str]:
        """获取所有多音字"""
        return set(cls.POLYPHONE_WORDS.keys())

    @classmethod
    def get_pronunciations(cls, char: str) -> List[str]:
        """获取某个多音字的所有读音"""
        if char in cls.POLYPHONE_WORDS:
            return list(cls.POLYPHONE_WORDS[char].keys())
        return []

    @classmethod
    def get_words_for_pronunciation(cls, char: str, pronunciation: str) -> List[str]:
        """获取某个多音字特定读音的所有词组"""
        if char in cls.POLYPHONE_WORDS and pronunciation in cls.POLYPHONE_WORDS[char]:
            return cls.POLYPHONE_WORDS[char][pronunciation]
        return []

    @classmethod
    def find_correct_pronunciation(cls, char: str, context: str) -> str:
        """
        根据上下文查找多音字的正确读音
        
        Args:
            char: 多音字
            context: 上下文文本
            
        Returns:
            正确的读音，如果无法判断返回None
        """
        if char not in cls.POLYPHONE_WORDS:
            return None

        best_pinyin = None
        max_matches = 0

        for pinyin_pron, words in cls.POLYPHONE_WORDS[char].items():
            matches = sum(1 for word in words if word in context)
            if matches > max_matches:
                max_matches = matches
                best_pinyin = pinyin_pron

        return best_pinyin


class PinyinReverseIndex:
    """
    拼音反向索引 - 用于拼音模糊匹配搜索
    """

    COMMON_CHARS = {
        'a': ['啊', '阿', '呵'],
        'ai': ['爱', '艾', '哎', '唉', '矮', '癌', '碍', '哀', '挨', '蔼'],
        'an': ['安', '按', '暗', '案', '岸', '氨', '庵', '鞍', '俺', '谙'],
        'ang': ['昂', '盎'],
        'ao': ['奥', '熬', '傲', '凹', '澳', '拗', '翱', '獒'],
        'ba': ['吧', '八', '把', '爸', '巴', '霸', '罢', '扒', '拔', '跋'],
        'bai': ['白', '百', '拜', '败', '摆', '佰', '柏'],
        'ban': ['班', '半', '办', '版', '般', '板', '搬', '伴', '斑', '扮'],
        'bang': ['帮', '棒', '榜', '绑', '磅', '蚌', '梆', '谤'],
        'bao': ['包', '报', '保', '抱', '宝', '暴', '薄', '爆', '堡', '胞'],
        'bei': ['北', '被', '背', '杯', '悲', '贝', '备', '辈', '碑', '卑'],
        'ben': ['本', '笨', '奔', '苯', '畚'],
        'beng': ['蹦', '崩', '绷', '甭', '迸'],
        'bi': ['比', '必', '笔', '闭', '壁', '臂', '逼', '鼻', '币', '蔽'],
        'bian': ['边', '变', '便', '遍', '编', '辩', '辨', '辫', '扁', '贬'],
        'biao': ['表', '标', '彪', '膘', '镖', '飙', '裱'],
        'bie': ['别', '憋', '鳖', '瘪', '蹩'],
        'bin': ['宾', '滨', '彬', '斌', '濒', '殡', '鬓'],
        'bing': ['并', '病', '冰', '兵', '饼', '丙', '柄', '禀', '摒'],
        'bo': ['波', '博', '播', '伯', '驳', '薄', '勃', '脖', '膊', '泊'],
        'bu': ['不', '步', '部', '布', '补', '捕', '卜', '哺', '埠', '簿'],
        'ca': ['擦', '嚓'],
        'cai': ['才', '菜', '财', '猜', '材', '彩', '裁', '采', '蔡', '睬'],
        'can': ['参', '餐', '残', '惨', '蚕', '灿', '璨', '孱'],
        'cang': ['藏', '仓', '苍', '舱', '沧', '伧'],
        'cao': ['草', '操', '曹', '槽', '糙', '嘈', '漕'],
        'ce': ['测', '册', '策', '侧', '厕', '恻', '岑'],
        'ceng': ['层', '曾', '蹭'],
        'cha': ['查', '茶', '差', '插', '察', '叉', '茬', '碴', '诧', '刹'],
        'chai': ['拆', '柴', '差', '钗'],
        'chan': ['产', '缠', '馋', '掺', '蝉', '馋', '颤', '潺', '蟾'],
        'chang': ['长', '常', '场', '唱', '厂', '尝', '肠', '畅', '昌', '娼'],
        'chao': ['超', '朝', '潮', '炒', '吵', '抄', '钞', '巢', '嘲', '绰'],
        'che': ['车', '扯', '彻', '撤', '掣', '坼'],
        'chen': ['陈', '沉', '晨', '称', '趁', '衬', '辰', '尘', '臣', '郴'],
        'cheng': ['成', '城', '程', '称', '诚', '承', '乘', '盛', '橙', '澄'],
        'chi': ['吃', '尺', '迟', '持', '池', '翅', '赤', '斥', '齿', '耻'],
        'chong': ['冲', '虫', '充', '重', '崇', '宠', '忡', '憧'],
        'chou': ['抽', '丑', '臭', '仇', '愁', '筹', '畴', '稠', '酬', '瞅'],
        'chu': ['出', '处', '初', '除', '楚', '触', '础', '储', '厨', '锄'],
        'chuan': ['穿', '川', '传', '船', '串', '喘', '椽', '舛'],
        'chuang': ['窗', '床', '创', '闯', '疮', '幢', '怆'],
        'chui': ['吹', '垂', '锤', '炊', '捶', '槌'],
        'chun': ['春', '纯', '唇', '蠢', '醇', '淳', '莼'],
        'chuo': ['戳', '绰', '辍', '龊'],
        'ci': ['次', '此', '词', '刺', '磁', '瓷', '慈', '雌', '辞', '祠'],
        'cong': ['从', '丛', '聪', '葱', '囱', '淙'],
        'cou': ['凑', '辏'],
        'cu': ['粗', '促', '醋', '簇', '蹴'],
        'cuan': ['窜', '篡', '蹿', '攒', '汆'],
        'cui': ['催', '脆', '翠', '崔', '摧', '粹', '悴', '萃', '瘁', '璀'],
        'cun': ['村', '存', '寸', '忖'],
        'cuo': ['错', '措', '挫', '撮', '搓', '蹉', '嵯'],
        'da': ['大', '打', '达', '答', '搭', '瘩', '耷', '鞑'],
        'dai': ['带', '代', '待', '大', '戴', '袋', '贷', '歹', '傣', '殆'],
        'dan': ['但', '单', '蛋', '担', '淡', '弹', '丹', '胆', '郸', '殚'],
        'dang': ['当', '党', '挡', '档', '荡', '铛', '裆', '凼'],
        'dao': ['到', '道', '刀', '倒', '导', '岛', '盗', '捣', '悼', '蹈'],
        'de': ['的', '得', '德', '地'],
        'deng': ['等', '灯', '登', '凳', '邓', '瞪', '蹬', '噔'],
        'di': ['的', '地', '低', '底', '第', '弟', '递', '敌', "的", '滴', '笛'],
        'dian': ['点', '电', '店', '典', '颠', '垫', '淀', '碘', '惦', '巅'],
        'diao': ['掉', '调', '吊', '钓', '叼', '雕', '碉', '凋', '吊'],
        'die': ['跌', '爹', '碟', '蝶', '叠', '谍', '迭', '牒', '垤'],
        'ding': ['定', '顶', '订', '丁', '钉', '盯', '叮', '锭', '仃', '酊'],
        'diu': ['丢'],
        'dong': ['东', '懂', '冬', '动', '洞', '冻', '栋', '咚', '峒', '胨'],
        'dou': ['都', '斗', '豆', '抖', '逗', '陡', '蚪', '窦'],
        'du': ['度', '读', '独', '毒', '堵', '杜', '肚', '督', '镀', '睹'],
        'duan': ['段', '短', '断', '端', '缎', '锻', '椴', '簖'],
        'dui': ['对', '队', '堆', '兑', '怼', '憝'],
        'dun': ['吨', '顿', '蹲', '盾', '敦', '墩', '炖', '盹', '沌', '遁'],
        'duo': ['多', '朵', '躲', '夺', '堕', '舵', '惰', '跺', '踱', '铎'],
        'e': ['饿', '恶', '额', '俄', '鹅', '蛾', '讹', '鄂', '遏', '噩'],
        'en': ['恩', '嗯'],
        'er': ['二', '而', '儿', '尔', '耳', '饵', '洱', '贰', '迩'],
        'fa': ['发', '法', '罚', '伐', '乏', '阀', '筏', '砝'],
        'fan': ['饭', '反', '番', '犯', '烦', '翻', '凡', '泛', '范', '蕃'],
        'fang': ['方', '放', '房', '防', '访', '仿', '妨', '坊', '纺', '舫'],
        'fei': ['非', '飞', '费', '肥', '废', '肺', '沸', '菲', '匪', '诽'],
        'fen': ['分', '份', '纷', '芬', '粉', '奋', '愤', '坟', '焚', '汾'],
        'feng': ['风', '封', '丰', '疯', '蜂', '锋', '峰', '逢', '凤', '奉'],
        'fo': ['佛'],
        'fou': ['否'],
        'fu': ['服', '父', '夫', '付', '福', '府', '副', '复', '富', '附'],
        'gai': ['该', '改', '盖', '概', '钙', '芥', '垓'],
        'gan': ['干', '感', '敢', '赶', '甘', '杆', '肝', '竿', '柑', '尴'],
        'gang': ['刚', '钢', '港', '岗', '纲', '肛', '缸', '杠', '罡', '戆'],
        'gao': ['高', '搞', '告', '稿', '糕', '皋', '篙', '膏', '镐', '诰'],
        'ge': ['个', '歌', '各', '格', '哥', '隔', '革', '葛', '阁', '胳'],
        'gei': ['给'],
        'gen': ['跟', '根', '亘'],
        'geng': ['更', '耕', '庚', '羹', '埂', '耿', '梗'],
        'gong': ['工', '公', '共', '功', '攻', '供', '公', '宫', '弓', '恭'],
        'gou': ['够', '狗', '沟', '购', '构', '钩', '勾', '苟', '垢', '篝'],
        'gu': ['古', '故', '顾', '骨', '谷', '鼓', '股', '孤', '姑', '估'],
        'gua': ['瓜', '挂', '刮', '寡', '卦', '呱', '剐', '诖'],
        'guai': ['怪', '乖', '拐'],
        'guan': ['关', '管', '观', '官', '馆', '惯', '冠', '灌', '罐', '棺'],
        'guang': ['光', '广', '逛', '胱'],
        'gui': ['贵', '归', '鬼', '规', '柜', '桂', '跪', '瑰', '圭', '诡'],
        'gun': ['滚', '棍', '辊', '磙'],
        'guo': ['国', '过', '果', '锅', '裹', '郭', '涡', '蝈', '帼'],
        'ha': ['哈', '蛤'],
        'hai': ['还', '海', '害', '孩', '嗨', '骸', '氦'],
        'han': ['汉', '含', '寒', '喊', '汗', '旱', '函', '韩', '罕', '翰'],
        'hang': ['行', '航', '杭', '巷', '夯'],
        'hao': ['好', '号', '浩', '耗', '豪', '毫', '郝', '嚎', '濠', '皓'],
        'he': ['和', '喝', '河', '合', '何', '贺', '赫', '鹤', '荷', '核'],
        'hei': ['黑', '嘿', '嗨'],
        'hen': ['很', '恨', '狠', '痕'],
        'heng': ['横', '恒', '衡', '亨', '哼', '衡', '珩'],
        'hong': ['红', '洪', '宏', '轰', '虹', '鸿', '弘', '烘', '泓', '薨'],
        'hou': ['后', '厚', '候', '侯', '猴', '喉', '吼', '逅', '侯'],
        'hu': ['户', '湖', '胡', '虎', '护', '乎', '忽', '壶', '葫', '糊'],
        'hua': ['话', '花', '画', '华', '化', '划', '滑', '哗', '骅', '桦'],
        'huai': ['坏', '怀', '淮', '槐', '徊'],
        'huan': ['还', '换', '欢', '环', '缓', '幻', '患', '唤', '焕', '涣'],
        'huang': ['黄', '皇', '荒', '慌', '晃', '谎', '磺', '簧', '璜', '徨'],
        'hui': ['会', '回', '灰', '辉', '汇', '惠', '慧', '毁', '悔', '贿'],
        'hun': ['混', '婚', '魂', '浑', '昏', '荤', '馄', '珲'],
        'huo': ['活', '火', '或', '货', '获', '祸', '豁', '霍', '藿', '镬'],
        'ji': ['几', '机', '集', '记', '己', '及', '急', '即', '既', '级'],
        'jia': ['家', '加', '假', '价', '架', '甲', '夹', '佳', '嘉', '伽'],
        'jian': ['见', '间', '建', '件', '简', '检', '剑', '监', '坚', '尖'],
        'jiang': ['将', '讲', '江', '降', '姜', '匠', '浆', '僵', '疆', '酱'],
        'jiao': ['叫', '教', '交', '脚', '角', '觉', '较', '浇', '娇', '嚼'],
        'jie': ['结', '接', '节', '街', '解', '界', '借', '介', '届', '皆'],
        'jin': ['进', '今', '金', '近', '紧', '尽', '劲', '禁', '斤', '津'],
        'jing': ['经', '京', '精', '静', '境', '景', '警', '竞', '净', '敬'],
        'jiu': ['就', '九', '久', '酒', '旧', '救', '纠', '究', '揪', '韭'],
        'ju': ['句', '举', '局', '具', '剧', '据', '距', '聚', '拒', '菊'],
        'juan': ['卷', '娟', '倦', '眷', '捐', '鹃', '镌', '绢'],
        'jue': ['觉', '绝', '决', '角', '掘', '嚼', '爵', '厥', '蹶', '攫'],
        'jun': ['军', '君', '均', '俊', '菌', '钧', '骏', '竣', '浚', '皲'],
        'ka': ['卡', '咖', '喀', '咔'],
        'kai': ['开', '凯', '慨', '楷', '锴'],
        'kan': ['看', '砍', '刊', '堪', '勘', '瞰', '龛', '戡'],
        'kang': ['抗', '扛', '康', '糠', '炕', '亢', '钪'],
        'kao': ['考', '靠', '烤', '拷', '犒'],
        'ke': ['可', '课', '克', '客', '刻', '科', '颗', '壳', '渴', '棵'],
        'ken': ['肯', '啃', '垦', '恳', '裉'],
        'keng': ['坑', '铿'],
        'kong': ['空', '孔', '控', '恐'],
        'kou': ['口', '扣', '寇', '叩', '抠'],
        'ku': ['苦', '哭', '库', '酷', '裤', '枯', '窟', '骷'],
        'kua': ['夸', '跨', '垮', '挎'],
        'kuai': ['快', '块', '筷', '会', '脍'],
        'kuan': ['宽', '款'],
        'kuang': ['况', '狂', '框', '矿', '筐', '旷', '眶', '诳'],
        'kui': ['亏', '愧', '溃', '葵', '魁', '馈', '聩', '睽', '盔', '魁'],
        'kun': ['困', '昆', '捆', '坤', '琨', '鲲'],
        'kuo': ['扩', '阔', '括', '廓'],
        'la': ['拉', '啦', '辣', '腊', '垃', '喇', '辣', '邋', '旯'],
        'lai': ['来', '赖', '莱', '睐', '赉'],
        'lan': ['兰', '蓝', '烂', '懒', '栏', '拦', '篮', '澜', '谰', '揽'],
        'lang': ['浪', '狼', '郎', '朗', '廊', '琅', '榔', '螂', '锒'],
        'lao': ['老', '劳', '牢', '捞', '姥', '酪', '烙', '涝', '唠', '崂'],
        'le': ['了', '乐', '勒', '雷', '镭'],
        'lei': ['类', '累', '雷', '泪', '垒', '擂', '镭', '蕾', '磊', '儡'],
        'leng': ['冷', '愣', '棱'],
        'li': ['里', '理', '李', '力', '立', '利', '例', '礼', '丽', '历'],
        'lian': ['连', '脸', '练', '联', '恋', '怜', '廉', '莲', '帘', '涟'],
        'liang': ['两', '亮', '量', '良', '凉', '梁', '粮', '辆', '谅', '粱'],
        'liao': ['了', '料', '聊', '辽', '疗', '燎', '寥', '僚', '寥', '镣'],
        'lie': ['列', '烈', '裂', '猎', '劣', '冽', '趔', '鬣'],
        'lin': ['林', '临', '邻', '淋', '琳', '鳞', '凛', '赁', '吝', '躏'],
        'ling': ['领', '零', '灵', '令', '另', '凌', '陵', '菱', '羚', '翎'],
        'liu': ['六', '留', '流', '刘', '柳', '溜', '瘤', '硫', '琉', '馏'],
        'long': ['龙', '隆', '笼', '聋', '拢', '垄', '咙', '胧', '窿', '泷'],
        'lou': ['楼', '漏', '陋', '搂', '篓', '镂', '蝼', '髅'],
        'lu': ['路', '录', '鹿', '卢', '露', '鲁', '陆', '炉', '庐', '颅'],
        'lv': ['绿', '律', '旅', '虑', '率', '铝', '屡', '缕', '履', '侣'],
        'luan': ['乱', '卵', '滦', '峦', '孪', '栾', '銮', '脔'],
        'lun': ['论', '轮', '伦', '沦', '纶', '囵'],
        'luo': ['落', '罗', '洛', '络', '骆', '螺', '逻', '锣', '箩', '骡'],
        'ma': ['吗', '妈', '马', '麻', '骂', '嘛', '抹', '蚂', '玛'],
        'mai': ['买', '卖', '麦', '埋', '迈', '霾', '劢'],
        'man': ['满', '慢', '漫', '忙', '芒', '盲', '茫', '莽', '猫'],
        'mang': ['忙', '芒', '盲', '茫', '莽'],
        'mao': ['毛', '猫', '冒', '帽', '茂', '贸', '矛', '茅', '铆', '髦'],
        'me': ['么', '嘛'],
        'mei': ['没', '美', '每', '妹', '眉', '煤', '梅', '酶', '霉', '玫'],
        'men': ['们', '门', '闷', '扪', '焖'],
        'meng': ['梦', '猛', '蒙', '盟', '孟', '锰', '萌', '朦', '檬', '勐'],
        'mi': ['米', '密', '迷', '蜜', '眯', '谜', '弥', '觅', '泌', '幂'],
        'mian': ['面', '免', '绵', '棉', '眠', '缅', '腼', '冕'],
        'miao': ['秒', '苗', '庙', '妙', '描', '瞄', '渺', '缈', '藐'],
        'min': ['民', '敏', '闽', '悯', '皿', '抿', '泯', '珉', '鳘'],
        'ming': ['明', '名', '命', '鸣', '冥', '铭', '螟', '瞑'],
        'miu': ['谬'],
        'mo': ['摸', '末', '墨', '默', '莫', '魔', '模', '磨', '摩', '膜'],
        'mou': ['某', '谋', '眸', '牟', '鍪'],
        'mu': ['母', '木', '目', '牧', '幕', '墓', '慕', '穆', '沐', '募'],
        'na': ['那', '拿', '哪', '呐', '纳', '娜', '钠', '捺'],
        'nai': ['奶', '耐', '乃', '奈', '萘', '鼐'],
        'nan': ['南', '男', '难', '楠', '囊', '馕', '囔'],
        'nang': ['囊', '馕', '囔'],
        'nao': ['脑', '闹', '恼', '挠', '瑙', '硇'],
        'ne': ['呢'],
        'nei': ['内', '馁'],
        'nen': ['嫩'],
        'neng': ['能'],
        'ni': ['你', '呢', '逆', '泥', '拟', '尼', '匿', '溺', '妮', '霓'],
        'nian': ['年', '念', '粘', '碾', '撵', '拈', '黏', '鲇'],
        'niang': ['娘', '酿'],
        'niao': ['鸟', '尿'],
        'nie': ['捏', '涅', '聂', '镍', '孽', '蘖', '蹑', '嗫'],
        'nin': ['您'],
        'ning': ['宁', '凝', '拧', '柠', '咛', '狞'],
        'niu': ['牛', '扭', '妞', '钮', '拗', '狃'],
        'nong': ['农', '浓', '弄', '脓'],
        'nu': ['女', '努', '怒', '奴', '弩', '胬'],
        'nuan': ['暖'],
        'nuo': ['挪', '诺', '糯', '懦', '傩', '喏'],
        'o': ['哦', '噢'],
        'ou': ['欧', '偶', '呕', '藕', '殴', '鸥', '耦', '讴'],
        'pa': ['怕', '爬', '帕', '趴', '啪', '琶', '葩', '耙'],
        'pai': ['排', '派', '拍', '牌', '徘', '湃', '俳'],
        'pan': ['盘', '判', '盼', '攀', '潘', '磐', '蹒', '蟠', '泮'],
        'pang': ['旁', '胖', '庞', '彷', '磅', '螃'],
        'pao': ['跑', '炮', '泡', '抛', '袍', '刨', '咆', '疱', '庖'],
        'pei': ['配', '陪', '培', '佩', '赔', '沛', '胚', '佩', '辔'],
        'pen': ['盆', '喷'],
        'peng': ['朋', '碰', '彭', '捧', '蓬', '棚', '硼', '篷', '澎', '嘭'],
        'pi': ['皮', '批', '屁', '脾', '匹', '劈', '坯', '僻', '譬', '疲'],
        'pian': ['片', '篇', '偏', '骗', '便', '翩', '骈', '胼'],
        'piao': ['票', '飘', '漂', '瓢', '嫖', '瞟', '剽', '缥'],
        'pie': ['撇', '瞥'],
        'pin': ['品', '拼', '贫', '频', '聘', '嫔', '颦'],
        'ping': ['平', '评', '瓶', '苹', '凭', '屏', '乒', '萍', '坪', '娉'],
        'po': ['破', '坡', '泼', '婆', '迫', '魄', '泊', '粕', '叵'],
        'pou': ['剖', '掊'],
        'pu': ['普', '仆', '扑', '铺', '朴', '葡', '蒲', '瀑', '曝', '圃'],
        'qi': ['起', '其', '气', '期', '七', '奇', '齐', '骑', '旗', '企'],
        'qia': ['恰', '洽', '掐', '卡'],
        'qian': ['前', '钱', '千', '浅', '签', '铅', '迁', '牵', '谦', '乾'],
        'qiang': ['强', '抢', '枪', '墙', '腔', '羌', '抢', '锵', '蔷', '樯'],
        'qiao': ['桥', '敲', '巧', '瞧', '翘', '壳', '锹', '侨', '荞', '樵'],
        'qie': ['切', '且', '窃', '茄', '怯', '惬', '箧', '锲'],
        'qin': ['亲', '琴', '秦', '侵', '勤', '禽', '寝', '沁', '嗪', '噙'],
        'qing': ['请', '清', '情', '青', '轻', '氢', '倾', '卿', '晴', '氰'],
        'qiong': ['穷', '琼', '穹', '蛩', '茕'],
        'qiu': ['球', '求', '秋', '丘', '囚', '酋', '泅', '俅', '巯'],
        'qu': ['去', '取', '区', '曲', '趣', '驱', '躯', '屈', '蛆', '渠'],
        'quan': ['全', '权', '圈', '泉', '拳', '犬', '劝', '券', '诠', '痊'],
        'que': ['却', '确', '缺', '雀', '瘸', '却', '鹊', '榷', '阕'],
        'qun': ['群', '裙', '逡'],
        'ran': ['然', '燃', '染', '冉', '髯'],
        'rang': ['让', '嚷', '壤', '攘', '禳'],
        'rao': ['绕', '扰', '饶', '娆'],
        're': ['热', '惹'],
        'ren': ['人', '认', '仁', '忍', '任', '刃', '韧', '妊', '饪', '仞'],
        'reng': ['仍', '扔'],
        'ri': ['日'],
        'rong': ['容', '荣', '融', '熔', '溶', '戎', '茸', '蓉', '榕', '冗'],
        'rou': ['肉', '柔', '揉', '蹂', '糅'],
        'ru': ['如', '入', '乳', '儒', '辱', '茹', '蠕', '汝', '孺', '褥'],
        'ruan': ['软', '阮'],
        'rui': ['瑞', '锐', '蕊', '睿', '芮'],
        'run': ['润', '闰'],
        'ruo': ['若', '弱', '偌'],
        'sa': ['撒', '洒', '萨', '飒'],
        'sai': ['赛', '塞', '腮', '鳃'],
        'san': ['三', '散', '伞', '叁', '毵'],
        'sang': ['桑', '丧', '嗓', '搡', '磉'],
        'sao': ['扫', '骚', '嫂', '臊', '瘙'],
        'se': ['色', '涩', '瑟', '塞', '啬', '铯'],
        'sen': ['森'],
        'seng': ['僧'],
        'sha': ['杀', '沙', '傻', '啥', '纱', '刹', '莎', '砂', '煞', '鲨'],
        'shai': ['晒', '筛', '色'],
        'shan': ['山', '善', '闪', '衫', '扇', '陕', '杉', '珊', '删', '煽'],
        'shang': ['上', '商', '尚', '伤', '赏', '响', '晌', '裳', '觞'],
        'shao': ['少', '烧', '稍', '勺', '邵', '韶', '梢', '捎', '芍', '鞘'],
        'she': ['社', '设', '射', '蛇', '舌', '舍', '摄', '涉', '赦', '麝'],
        'shei': ['谁'],
        'shen': ['什', '深', '身', '神', '甚', '肾', '慎', '渗', '申', '伸'],
        'sheng': ['生', '声', '省', '圣', '胜', '盛', '剩', '牲', '升', '笙'],
        'shi': ['是', '时', '事', '市', '十', '世', '史', '石', '式', '师'],
        'shou': ['手', '收', '首', '受', '瘦', '售', '守', '熟', '寿', '授'],
        'shu': ['书', '数', '树', '熟', '输', '叔', '束', '术', '述', '舒'],
        'shua': ['刷', '耍', '唰'],
        'shuai': ['帅', '摔', '衰', '甩', '率'],
        'shuan': ['栓', '拴', '涮'],
        'shuang': ['双', '爽', '霜', '孀'],
        'shui': ['水', '睡', '谁', '税', '说'],
        'shun': ['顺', '瞬', '舜'],
        'shuo': ['说', '硕', '朔', '烁', '铄', '妁'],
        'si': ['四', '思', '死', '私', '司', '丝', '撕', '斯', '嘶', '寺'],
        'song': ['送', '松', '宋', '颂', '诵', '耸', '竦', '淞', '崧'],
        'sou': ['搜', '艘', '嗖', '叟', '嗖', '馊'],
        'su': ['苏', '速', '素', '诉', '俗', '肃', '粟', '塑', '溯', '夙'],
        'suan': ['算', '酸', '蒜', '狻'],
        'sui': ['虽', '随', '岁', '碎', '隋', '遂', '隧', '髓', '绥', '祟'],
        'sun': ['孙', '损', '笋', '荪', '榫'],
        'suo': ['所', '锁', '索', '缩', '梭', '唆', '蓑', '嗦', '唢'],
        'ta': ['他', '她', '它', '塔', '踏', '塌', '獭', '挞', '蹋', '趿'],
        'tai': ['太', '台', '态', '泰', '抬', '胎', '苔', '酞', '钛'],
        'tan': ['谈', '弹', '探', '叹', '碳', '谭', '潭', '贪', '滩', '瘫'],
        'tang': ['堂', '糖', '唐', '汤', '躺', '趟', '烫', '倘', '塘', '搪'],
        'tao': ['套', '逃', '淘', '桃', '涛', '掏', '滔', '韬', '饕', '洮'],
        'te': ['特'],
        'teng': ['疼', '腾', '藤', '滕', '誊'],
        'ti': ['提', '体', '题', '替', '踢', '蹄', '啼', '屉', '涕', '惕'],
        'tian': ['天', '田', '甜', '填', '添', '腆', '掭', '钿'],
        'tiao': ['条', '跳', '调', '挑', '眺', '窕', '笤', '粜'],
        'tie': ['铁', '贴', '帖'],
        'ting': ['听', '停', '亭', '廷', '挺', '艇', '庭', '蜓', '汀', '廷'],
        'tong': ['同', '通', '痛', '统', '童', '铜', '桶', '筒', '彤', '桐'],
        'tou': ['头', '投', '透', '偷', '骰'],
        'tu': ['图', '土', '涂', '途', '兔', '吐', '秃', '突', '徒', '屠'],
        'tuan': ['团', '湍', '疃'],
        'tui': ['推', '退', '腿', '颓', '蜕', '褪', '煺'],
        'tun': ['吞', '屯', '臀', '囤', '豚', '氽'],
        'tuo': ['脱', '拖', '托', '驮', '妥', '拓', '唾', '鸵', '陀', '驼'],
        'wa': ['挖', '哇', '蛙', '瓦', '袜', '凹', '娲'],
        'wai': ['外', '歪'],
        'wan': ['完', '万', '晚', '玩', '碗', '弯', '湾', '丸', '顽', '婉'],
        'wang': ['王', '网', '往', '望', '忘', '旺', '妄', '亡', '枉'],
        'wei': ['为', '位', '围', '微', '味', '喂', '胃', '魏', '唯', '威'],
        'wen': ['问', '文', '闻', '温', '稳', '蚊', '纹', '吻', '瘟', '紊'],
        'weng': ['翁', '嗡', '瓮', '蓊'],
        'wo': ['我', '握', '窝', '蜗', '卧', '涡', '挝', '龌'],
        'wu': ['五', '无', '物', '武', '午', '吴', '务', '悟', '误', '屋'],
        'xi': ['西', '习', '喜', '洗', '系', '细', '吸', '戏', '稀', '溪'],
        'xia': ['下', '夏', '吓', '虾', '瞎', '峡', '侠', '狭', '霞', '匣'],
        'xian': ['先', '现', '线', '县', '鲜', '纤', '弦', '贤', '咸', '闲'],
        'xiang': ['想', '向', '像', '香', '响', '乡', '相', '箱', '祥', '详'],
        'xiao': ['小', '笑', '校', '萧', '消', '销', '宵', '晓', '孝', '肖'],
        'xie': ['写', '谢', '些', '鞋', '斜', '血', '歇', '协', '胁', '谐'],
        'xin': ['新', '心', '信', '欣', '辛', '薪', '馨', '芯', '锌', '昕'],
        'xing': ['行', '星', '兴', '型', '姓', '性', '醒', '刑', '杏', '幸'],
        'xiong': ['雄', '熊', '凶', '兄', '胸', '汹', '雄'],
        'xiu': ['修', '休', '秀', '绣', '袖', '羞', '宿', '锈', '嗅', '溴'],
        'xu': ['需', '许', '续', '须', '虚', '序', '叙', '绪', '畜', '蓄'],
        'xuan': ['选', '宣', '悬', '旋', '玄', '选', '炫', '绚', '喧', '轩'],
        'xue': ['学', '雪', '血', '穴', '靴', '薛', '谑', '鳕'],
        'xun': ['寻', '训', '讯', '迅', '巡', '旬', '询', '循', '逊', '殉'],
        'ya': ['呀', '压', '牙', '鸦', '雅', '亚', '哑', '芽', '蚜', '涯'],
        'yan': ['眼', '言', '严', '烟', '沿', '盐', '颜', '燕', '厌', '宴'],
        'yang': ['样', '阳', '养', '央', '羊', '洋', '仰', '痒', '杨', '扬'],
        'yao': ['要', '药', '摇', '咬', '腰', '邀', '耀', '窑', '谣', '遥'],
        'ye': ['也', '夜', '叶', '业', '野', '爷', '液', '耶', '咽', '掖'],
        'yi': ['一', '以', '已', '意', '义', '议', '医', '衣', '依', '易'],
        'yin': ['因', '音', '银', '引', '印', '阴', '饮', '隐', '吟', '淫'],
        'ying': ['应', '英', '影', '营', '迎', '赢', '硬', '映', '鹰', '颖'],
        'yo': ['哟'],
        'yong': ['用', '永', '拥', '勇', '涌', '庸', '咏', '泳', '蛹', '恿'],
        'you': ['有', '又', '右', '友', '优', '游', '油', '由', '尤', '犹'],
        'yu': ['与', '于', '语', '雨', '鱼', '遇', '玉', '育', '域', '欲'],
        'yuan': ['原', '远', '园', '元', '员', '圆', '源', '缘', '院', '愿'],
        'yue': ['月', '越', '约', '乐', '跃', '阅', '岳', '悦', '钥', '粤'],
        'yun': ['云', '运', '允', '韵', '孕', '蕴', '酝', '耘', '匀', '陨'],
        'za': ['杂', '咋', '砸', '扎'],
        'zai': ['在', '再', '载', '灾', '栽', '宰', '崽'],
        'zan': ['咱', '暂', '赞', '攒', '簪'],
        'zang': ['脏', '葬', '藏', '臧', '奘'],
        'zao': ['早', '造', '找', '遭', '糟', '凿', '枣', '澡', '藻', '灶'],
        'ze': ['则', '责', '择', '泽', '啧', '仄', '赜'],
        'zei': ['贼'],
        'zen': ['怎', '谮'],
        'zeng': ['增', '曾', '憎', '赠', '缯'],
        'zha': ['扎', '炸', '渣', '眨', '榨', '咋', '札', '轧', '铡', '痄'],
        'zhai': ['摘', '窄', '宅', '债', '斋', '寨', '瘵'],
        'zhan': ['站', '战', '占', '展', '沾', '粘', '盏', '斩', '绽', '湛'],
        'zhang': ['张', '长', '章', '掌', '涨', '帐', '仗', '胀', '瘴', '彰'],
        'zhao': ['找', '照', '招', '召', '赵', '罩', '爪', '昭', '兆', '肇'],
        'zhe': ['这', '着', '者', '哲', '折', '遮', '蛰', '辙', '蔗', '赭'],
        'zhei': ['这'],
        'zhen': ['真', '针', '镇', '阵', '振', '震', '枕', '珍', '贞', '斟'],
        'zheng': ['正', '整', '政', '证', '征', '争', '症', '郑', '蒸', '挣'],
        'zhi': ['只', '知', '之', '直', '至', '制', '志', '治', '指', '纸'],
        'zhong': ['中', '重', '众', '钟', '终', '种', '忠', '肿', '仲', '衷'],
        'zhou': ['周', '州', '洲', '舟', '粥', '轴', '肘', '咒', '皱', '昼'],
        'zhu': ['主', '住', '注', '猪', '竹', '助', '著', '祝', '筑', '逐'],
        'zhuan': ['转', '专', '赚', '砖', '传', '撰', '篆', '啭'],
        'zhuang': ['装', '状', '壮', '撞', '庄', '妆', '幢', '奘'],
        'zhui': ['追', '坠', '锥', '缀', '椎', '赘'],
        'zhun': ['准', '谆', '肫'],
        'zhuo': ['着', '桌', '捉', '灼', '卓', '浊', '琢', '茁', '酌', '擢'],
        'zi': ['子', '自', '字', '紫', '资', '姿', '滋', '孜', '籽', '姊'],
        'zong': ['总', '宗', '综', '踪', '棕', '鬃', '腙'],
        'zou': ['走', '邹', '奏', '揍', '诹'],
        'zu': ['足', '组', '族', '阻', '祖', '卒', '租', '诅', '俎'],
        'zuan': ['钻', '躜', '纂'],
        'zui': ['最', '嘴', '醉', '罪', '最'],
        'zun': ['尊', '遵', '樽', '鳟'],
        'zuo': ['做', '坐', '左', '作', '座', '昨', '佐', '撮', '祚']
    }

    @classmethod
    def search_by_pinyin(
        cls,
        pinyin_input: str,
        fuzzy_match: bool = True,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        根据拼音搜索汉字
        
        Args:
            pinyin_input: 拼音输入（可以是完整拼音或首字母）
            fuzzy_match: 是否启用模糊匹配
            limit: 返回结果数量限制
            
        Returns:
            匹配结果列表，每项包含拼音和对应的汉字列表
        """
        pinyin_clean = pinyin_input.lower().strip()
        results = []

        for pinyin_str, chars in cls.COMMON_CHARS.items():
            match = False
            match_type = ""

            if pinyin_str == pinyin_clean:
                match = True
                match_type = "exact"
            elif fuzzy_match:
                if pinyin_str.startswith(pinyin_clean):
                    match = True
                    match_type = "prefix"
                elif pinyin_clean in pinyin_str:
                    match = True
                    match_type = "contains"
                elif len(pinyin_clean) == 1 and pinyin_str[0] == pinyin_clean:
                    match = True
                    match_type = "first_letter"

            if match:
                results.append({
                    "pinyin": pinyin_str,
                    "chars": chars[:limit],
                    "match_type": match_type,
                    "char_count": len(chars)
                })

        results.sort(key=lambda x: {
            "exact": 0,
            "prefix": 1,
            "first_letter": 2,
            "contains": 3
        }.get(x["match_type"], 4))

        return results[:limit]

    @classmethod
    def get_chars_by_pinyin(cls, pinyin_str: str) -> List[str]:
        """
        获取指定拼音对应的所有汉字
        """
        return cls.COMMON_CHARS.get(pinyin_str.lower(), [])

    @classmethod
    def get_pinyin_by_char(cls, char: str) -> List[str]:
        """
        获取指定汉字的拼音（反向查找）
        """
        result = []
        for pinyin_str, chars in cls.COMMON_CHARS.items():
            if char in chars:
                result.append(pinyin_str)
        return result


class PinyinResult:
    """
    拼音转换结果类，包含详细的转换信息
    """

    def __init__(
        self,
        original_text: str,
        pinyin_text: str,
        pinyin_list: List[str],
        converted_chars: List[Dict[str, str]],
        preserved_chars: List[Dict[str, Any]],
        style: str,
        polyphone_corrections: List[Dict[str, Any]] = None
    ):
        self.original_text = original_text
        self.pinyin_text = pinyin_text
        self.pinyin_list = pinyin_list
        self.converted_chars = converted_chars
        self.preserved_chars = preserved_chars
        self.style = style
        self.polyphone_corrections = polyphone_corrections or []

    def to_dict(self) -> Dict[str, Any]:
        """
        将结果转换为字典格式
        """
        return {
            "original_text": self.original_text,
            "pinyin_text": self.pinyin_text,
            "pinyin_list": self.pinyin_list,
            "converted_count": len(self.converted_chars),
            "preserved_count": len(self.preserved_chars),
            "converted_chars": self.converted_chars,
            "preserved_chars": self.preserved_chars,
            "polyphone_corrections": self.polyphone_corrections,
            "style": self.style
        }

    def __str__(self) -> str:
        return self.pinyin_text

    def get_detail_report(self) -> str:
        """
        获取详细的转换报告
        """
        lines = [
            "=== 拼音转换详细报告 ===",
            f"原文: {self.original_text}",
            f"转换结果: {self.pinyin_text}",
            f"输出格式: {self.style}",
            f"",
            f"转换统计:",
            f"  - 中文字符数: {len(self.converted_chars)}",
            f"  - 保留字符数: {len(self.preserved_chars)}",
            f"  - 多音字纠正数: {len(self.polyphone_corrections)}",
            f"",
            f"转换详情:"
        ]

        for item in self.converted_chars:
            lines.append(f"  {item['char']} → {item['pinyin']}")

        if self.polyphone_corrections:
            lines.append(f"")
            lines.append(f"多音字纠正:")
            for corr in self.polyphone_corrections:
                lines.append(
                    f"  {corr['char']}: 修正为 '{corr['corrected']}' "
                    f"(原默认读音: '{corr['original']}') - 匹配词组: {corr['matched_word']}"
                )

        if self.preserved_chars:
            lines.append(f"")
            lines.append(f"保留字符:")
            for item in self.preserved_chars:
                char_type = item['type']
                char = item['char']
                lines.append(f"  [{char_type}] {repr(char)}")

        return "\n".join(lines)


class PinyinAPI:
    """
    中文转拼音API，支持多种输出格式，保留非中文字符，
    支持多音字智能纠正和拼音模糊匹配搜索
    """

    @staticmethod
    def _is_chinese_char(char: str) -> bool:
        """判断是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    @staticmethod
    def _get_char_type(char: str) -> str:
        """获取字符类型"""
        if char.isdigit():
            return "数字"
        elif char.isalpha():
            return "英文"
        elif char.isspace():
            return "空格"
        else:
            return "符号"

    @staticmethod
    def _remove_tone(pinyin_str: str) -> str:
        """去除拼音声调"""
        tone_map = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',
            'ü': 'v'
        }
        result = []
        for char in pinyin_str:
            result.append(tone_map.get(char, char))
        return ''.join(result)

    @staticmethod
    def to_pinyin(
        text: str,
        style: str = "normal",
        heteronym: bool = False,
        separator: str = " ",
        return_detail: bool = False,
        smart_polyphone: bool = True
    ) -> Union[str, PinyinResult]:
        """
        将中文转换为拼音，非中文字符原样保留
        
        Args:
            text: 输入字符串
            style: 输出格式: 'tone', 'normal', 'first_letter'
            heteronym: 是否启用多音字模式
            separator: 拼音之间的分隔符
            return_detail: 是否返回详细结果对象
            smart_polyphone: 是否启用多音字智能纠正
            
        Returns:
            转换后的拼音字符串或PinyinResult对象
        """
        if not text:
            if return_detail:
                return PinyinResult("", "", [], [], [], style)
            return ""

        style_map = {
            "tone": Style.TONE,
            "normal": Style.NORMAL,
            "first_letter": Style.FIRST_LETTER
        }

        if style not in style_map:
            raise ValueError(
                f"无效的style参数: {style}。可选值: 'tone', 'normal', 'first_letter'"
            )

        pinyin_style = style_map[style]

        result_parts = []
        pinyin_list = []
        converted_chars = []
        preserved_chars = []
        polyphone_corrections = []

        polyphone_chars = PolyphoneDictionary.get_polyphone_chars()

        for idx, char in enumerate(text):
            if PinyinAPI._is_chinese_char(char):
                char_pinyin = pinyin(char, style=pinyin_style, heteronym=heteronym)[0][0]
                original_pinyin = char_pinyin

                if smart_polyphone and char in polyphone_chars:
                    context_window = max(0, idx - 2), min(len(text), idx + 3)
                    context = text[context_window[0]:context_window[1]]
                    correct_pinyin = PolyphoneDictionary.find_correct_pronunciation(char, context)

                    if correct_pinyin:
                        if style == "tone":
                            tone_pinyin = pinyin(
                                char, style=Style.TONE, heteronym=True
                            )
                            for variants in tone_pinyin:
                                for variant in variants:
                                    if PinyinAPI._remove_tone(variant) == correct_pinyin:
                                        char_pinyin = variant
                                        break
                        elif style == "normal":
                            char_pinyin = correct_pinyin
                        elif style == "first_letter":
                            char_pinyin = correct_pinyin[0].upper()

                        if char_pinyin != original_pinyin:
                            matched_words = [
                                word for word in
                                PolyphoneDictionary.get_words_for_pronunciation(char, correct_pinyin)
                                if word in context
                            ]
                            polyphone_corrections.append({
                                "char": char,
                                "index": idx,
                                "original": original_pinyin,
                                "corrected": char_pinyin,
                                "matched_word": matched_words[0] if matched_words else ""
                            })

                result_parts.append(char_pinyin)
                pinyin_list.append(char_pinyin)
                converted_chars.append({
                    "index": idx,
                    "char": char,
                    "pinyin": char_pinyin
                })
            else:
                result_parts.append(char)
                if style == "first_letter":
                    pinyin_list.append(char.upper())
                else:
                    pinyin_list.append(char)
                preserved_chars.append({
                    "index": idx,
                    "char": char,
                    "type": PinyinAPI._get_char_type(char)
                })

        if style == "first_letter":
            pinyin_text = "".join(pinyin_list).upper()
        else:
            pinyin_text = ""
            for i, part in enumerate(result_parts):
                if i > 0:
                    prev_char = text[i - 1]
                    curr_char = text[i]
                    prev_is_chinese = PinyinAPI._is_chinese_char(prev_char)
                    curr_is_chinese = PinyinAPI._is_chinese_char(curr_char)
                    prev_is_alnum = prev_char.isalnum() and not prev_is_chinese
                    curr_is_alnum = curr_char.isalnum() and not curr_is_chinese

                    if prev_is_chinese and curr_is_chinese:
                        pinyin_text += separator
                    elif (prev_is_chinese or curr_is_chinese) and (prev_is_alnum or curr_is_alnum):
                        pinyin_text += separator
                pinyin_text += part

        result = PinyinResult(
            original_text=text,
            pinyin_text=pinyin_text,
            pinyin_list=pinyin_list,
            converted_chars=converted_chars,
            preserved_chars=preserved_chars,
            style=style,
            polyphone_corrections=polyphone_corrections
        )

        if return_detail:
            return result
        return pinyin_text

    @staticmethod
    def to_pinyin_with_tone(
        text: str,
        heteronym: bool = False,
        return_detail: bool = False,
        smart_polyphone: bool = True
    ) -> Union[str, PinyinResult]:
        """转换为带声调的拼音"""
        return PinyinAPI.to_pinyin(
            text, style="tone", heteronym=heteronym,
            return_detail=return_detail, smart_polyphone=smart_polyphone
        )

    @staticmethod
    def to_pinyin_without_tone(
        text: str,
        heteronym: bool = False,
        return_detail: bool = False,
        smart_polyphone: bool = True
    ) -> Union[str, PinyinResult]:
        """转换为无声调的拼音"""
        return PinyinAPI.to_pinyin(
            text, style="normal", heteronym=heteronym,
            return_detail=return_detail, smart_polyphone=smart_polyphone
        )

    @staticmethod
    def to_pinyin_first_letter(
        text: str,
        heteronym: bool = False,
        return_detail: bool = False,
        smart_polyphone: bool = True
    ) -> Union[str, PinyinResult]:
        """转换为首字母大写形式"""
        return PinyinAPI.to_pinyin(
            text, style="first_letter", heteronym=heteronym,
            return_detail=return_detail, smart_polyphone=smart_polyphone
        )

    @staticmethod
    def to_pinyin_list(
        text: str,
        style: str = "normal",
        heteronym: bool = False,
        smart_polyphone: bool = True
    ) -> List[str]:
        """将中文转换为拼音列表"""
        result = PinyinAPI.to_pinyin(
            text, style=style, heteronym=heteronym,
            return_detail=True, smart_polyphone=smart_polyphone
        )
        return result.pinyin_list

    @staticmethod
    def search_by_pinyin(
        pinyin_input: str,
        fuzzy_match: bool = True,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        根据拼音搜索汉字（用于搜索建议）
        
        Args:
            pinyin_input: 拼音输入（可以是完整拼音、首字母或部分拼音）
            fuzzy_match: 是否启用模糊匹配
            limit: 返回结果数量限制
            
        Returns:
            匹配结果列表
        """
        return PinyinReverseIndex.search_by_pinyin(pinyin_input, fuzzy_match, limit)

    @staticmethod
    def get_suggestions(
        input_str: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        智能搜索建议：支持拼音、汉字混合输入
        
        Args:
            input_str: 输入字符串（可以是拼音、汉字或混合）
            limit: 返回结果数量限制
            
        Returns:
            搜索建议结果
        """
        suggestions = {
            "input": input_str,
            "pinyin_matches": [],
            "combined_suggestions": []
        }

        has_chinese = any(PinyinAPI._is_chinese_char(c) for c in input_str)
        has_pinyin = any(c.isalpha() for c in input_str)

        if has_pinyin or not has_chinese:
            pinyin_results = PinyinAPI.search_by_pinyin(
                ''.join([c for c in input_str if c.isalpha()]),
                fuzzy_match=True,
                limit=limit
            )
            suggestions["pinyin_matches"] = pinyin_results

            for result in pinyin_results:
                for char in result["chars"][:5]:
                    suggestions["combined_suggestions"].append({
                        "text": char,
                        "pinyin": result["pinyin"],
                        "type": "single_char"
                    })

        return suggestions


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("=== 多音字智能纠正测试 ===")
    print("=" * 60)

    polyphone_tests = [
        "银行",
        "行走",
        "重庆火锅很重要",
        "快乐听音乐",
        "小明长大要当校长",
        "长江很长",
        "我了解好了",
        "首都都是美丽的"
    ]

    for test_text in polyphone_tests:
        print(f"\n原文: {test_text}")
        
        result_with_smart = PinyinAPI.to_pinyin_with_tone(
            test_text, smart_polyphone=True, return_detail=True
        )
        print(f"智能纠正: {result_with_smart.pinyin_text}")
        
        result_without_smart = PinyinAPI.to_pinyin_with_tone(
            test_text, smart_polyphone=False, return_detail=True
        )
        print(f"默认读音: {result_without_smart.pinyin_text}")
        
        if result_with_smart.polyphone_corrections:
            print("纠正详情:")
            for corr in result_with_smart.polyphone_corrections:
                print(f"  - {corr['char']}: {corr['original']} → {corr['corrected']} "
                      f"(匹配: {corr['matched_word']})")

    print("\n" + "=" * 60)
    print("=== 拼音模糊匹配搜索测试 ===")
    print("=" * 60)

    search_tests = ["li", "z", "zhang", "hua", "sh"]

    for test_pinyin in search_tests:
        print(f"\n搜索拼音: '{test_pinyin}'")
        results = PinyinAPI.search_by_pinyin(test_pinyin, fuzzy_match=True, limit=5)
        
        for result in results:
            match_type_label = {
                "exact": "精确匹配",
                "prefix": "前缀匹配",
                "first_letter": "首字母匹配",
                "contains": "包含匹配"
            }.get(result["match_type"], result["match_type"])
            
            print(f"  [{match_type_label}] {result['pinyin']}: "
                  f"{''.join(result['chars'][:10])}... (共{result['char_count']}字)")

    print("\n" + "=" * 60)
    print("=== 智能搜索建议测试 ===")
    print("=" * 60)

    suggestion_tests = ["li", "z", "wang"]

    for test_input in suggestion_tests:
        print(f"\n输入: '{test_input}'")
        suggestions = PinyinAPI.get_suggestions(test_input, limit=3)
        
        for match in suggestions["pinyin_matches"][:3]:
            print(f"  {match['pinyin']}: {''.join(match['chars'][:5])}")

    print("\n" + "=" * 60)
    print("=== 详细报告示例 ===")
    print("=" * 60)

    result = PinyinAPI.to_pinyin_with_tone(
        "银行行长很重要", smart_polyphone=True, return_detail=True
    )
    print(result.get_detail_report())

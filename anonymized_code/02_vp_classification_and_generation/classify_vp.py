#!/usr/bin/env python3
"""
classify_vp.py
对SVO语料中的VP进行13类分类

用法：
    python classify_vp.py --input svo_all.csv --output svo_classified.csv

分类标签（不互斥，多标签用+连接）：
    V_de_resultative     VP含得字补语（搞得很乱）
    V_yi_V               尝试体（试一试）
    V_object             述谓动词（当朋友/做老师/变成液体）
    V_directional        趋向补语（拿出来/推下去）
    V_resultative_simple 结果补语（打碎/关上）
    V_quantified         动量时量短语（唱三遍/打一下）
    V_dat_PP             与格介词后置（寄给他/推向他）
    V_loc_PP             处所介词后置（放在桌上/推到墙上）
    V_aspect_zhe         着字体标记
    V_aspect_le          了字体标记
    V_aspect_guo         过字体标记
    V_adv                前置副词修饰（依存标签ADV/advmod）
    UNKNOWN              以上均未检测到
"""

import re
import argparse
import sys
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    sys.exit("请安装 pandas：pip install pandas")


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 趋向补语词
DIRECTIONAL = {'来', '去', '出', '进', '上', '下', '回', '起',
               '出来', '出去', '进来', '进去', '上来', '上去',
               '下来', '下去', '回来', '回去', '起来'}

# 结果补语（单字为主）
RESULTATIVE = {'碎', '开', '好', '完', '掉', '死', '走', '跑',
               '断', '破', '烂', '散', '倒', '住', '到', '成',
               '赢', '输', '哭', '醒', '牢', '爆', '崩', '伤',
               '坏', '通', '净', '光', '满', '空', '干', '湿'}

# 与格介词（动词后置才算补语）
DAT_PREPS = {'给', '向', '朝', '于'}

# 处所介词（动词后置才算补语）
LOC_PREPS = {'在', '到', '于', '往', '至'}

# 系词性动词（V_object）
COPULA_VERBS = {'当', '做', '成', '为', '变', '改', '称', '叫',
                '算', '作', '视', '当作', '看作', '变成', '改成',
                '称为', '视为'}

# 名词性词性标签
NOUN_POS = {'NN', 'NR', 'NT'}

# 副词性依存标签
ADV_LABELS = {'ADV', 'advmod', 'ADVMOD'}


# ══════════════════════════════════════════════════════════════
# 依存解析辅助
# ══════════════════════════════════════════════════════════════

def parse_deps_raw(deps_raw_str, n_tokens):
    """
    解析deps_raw字符串，返回0-based的依存列表。
    deps_raw格式：'head1,label1 head2,label2 ...'（1-based head）
    返回：[(head_0based_or_None, label), ...]
    """
    deps = []
    for item in deps_raw_str.strip().split():
        parts = item.split(',', 1)
        if len(parts) != 2:
            deps.append((None, ''))
            continue
        try:
            head_1based = int(parts[0])
            head = head_1based - 1 if head_1based > 0 else None
        except ValueError:
            head = None
        deps.append((head, parts[1]))
    # 补齐长度
    while len(deps) < n_tokens:
        deps.append((None, ''))
    return deps


def build_children(deps_0):
    """构建parent→[(child, label)]索引"""
    children = defaultdict(list)
    for child, (head, label) in enumerate(deps_0):
        if head is not None:
            children[head].append((child, label))
    return children


# ══════════════════════════════════════════════════════════════
# 分类函数
# ══════════════════════════════════════════════════════════════

def classify_vp(row):
    """
    对单行SVO数据进行VP分类，返回标签列表。
    """
    tokens_str  = str(row.get('tokens', ''))
    pos_str     = str(row.get('pos_tags', ''))
    deps_str    = str(row.get('deps_raw', ''))
    verb_idx    = int(row.get('verb_index', 0))

    tokens  = tokens_str.split()
    postags = pos_str.split()
    n       = len(tokens)

    if verb_idx >= n:
        return ['UNKNOWN']

    deps_0   = parse_deps_raw(deps_str, n)
    children = build_children(deps_0)

    # 动词后面的tokens
    after_tokens = tokens[verb_idx + 1:]
    after_pos    = postags[verb_idx + 1:] if len(postags) > verb_idx + 1 else []

    labels = []

    # ── 1. V_aspect_le / zhe / guo ────────────────────────────
    # 体标记紧跟动词后，或出现在VP范围内
    if '了' in after_tokens:
        labels.append('V_aspect_le')
    if '着' in after_tokens:
        labels.append('V_aspect_zhe')
    if '过' in after_tokens:
        labels.append('V_aspect_guo')

    # ── 2. V_de_resultative ────────────────────────────────────
    # 动词后有"得"，且"得"后接形容词/副词（VA/AD）或补语
    if '得' in after_tokens:
        de_idx = after_tokens.index('得')
        # 得后面有词
        if de_idx + 1 < len(after_tokens):
            labels.append('V_de_resultative')

    # ── 3. V_directional ──────────────────────────────────────
    # 动词后出现趋向词
    for t in after_tokens:
        if t in DIRECTIONAL:
            labels.append('V_directional')
            break
    # 也检测两字趋向词（出来/进去等）
    after_joined = ''.join(after_tokens)
    for d in DIRECTIONAL:
        if len(d) == 2 and d in after_joined:
            if 'V_directional' not in labels:
                labels.append('V_directional')
            break

    # ── 4. V_resultative_simple ───────────────────────────────
    # 动词后紧跟单字结果补语（排除了/着/过/得等体标记）
    if after_tokens:
        first_after = after_tokens[0]
        if first_after in RESULTATIVE:
            labels.append('V_resultative_simple')
        # 也检查依存子节点中是否有CMP（补语）标签
        for child_idx, label in children[verb_idx]:
            if label in ('CMP', 'cmp', 'COMP', 'comp') and child_idx > verb_idx:
                if 'V_resultative_simple' not in labels:
                    labels.append('V_resultative_simple')

    # ── 5. V_dat_PP ────────────────────────────────────────────
    # 动词后出现与格介词
    for t in after_tokens:
        if t in DAT_PREPS:
            labels.append('V_dat_PP')
            break

    # ── 6. V_loc_PP ────────────────────────────────────────────
    # 动词后出现处所介词
    for t in after_tokens:
        if t in LOC_PREPS:
            labels.append('V_loc_PP')
            break

    # ── 7. V_quantified ───────────────────────────────────────
    # 动词后出现数量短语：数词+量词，或"一下/一次/一遍"等
    quant_re = re.compile(r'^[一两三四五六七八九十百千\d]+[下次遍回趟年月天小时分钟]$')
    for t in after_tokens:
        if quant_re.match(t) or t in {'一下', '一次', '一遍', '一回', '一趟'}:
            labels.append('V_quantified')
            break
    # 也检查数词+量词的组合（分开分词的情况）
    for i, t in enumerate(after_tokens):
        if re.match(r'^[一两三四五六七八九十百千\d]+$', t):
            if i + 1 < len(after_tokens):
                next_t = after_tokens[i + 1]
                if re.match(r'^[下次遍回趟年月天时分钟个]$', next_t):
                    if 'V_quantified' not in labels:
                        labels.append('V_quantified')
                    break

    # ── 8. V_yi_V（尝试体）────────────────────────────────────
    # 动词后出现"一"，且"一"后有相同动词
    verb_token = tokens[verb_idx]
    for i, t in enumerate(after_tokens):
        if t == '一' and i + 1 < len(after_tokens):
            if after_tokens[i + 1] == verb_token:
                labels.append('V_yi_V')
                break
        # 也检测重叠式：AABB（动词直接重叠）
        if t == verb_token and i == 0:
            labels.append('V_yi_V')
            break

    # ── 9. V_object（述谓动词）────────────────────────────────
    # 动词本身是系词性动词，或动词后有成/为+名词
    if verb_token in COPULA_VERBS:
        labels.append('V_object')
    else:
        # 动词后有"成/为"且后接名词
        for i, t in enumerate(after_tokens):
            if t in ('成', '为'):
                if i + 1 < len(after_tokens):
                    next_pos = after_pos[i + 1] if i + 1 < len(after_pos) else ''
                    if next_pos in NOUN_POS:
                        labels.append('V_object')
                        break

    # ── 10. UNKNOWN ────────────────────────────────────────────
    if not labels:
        labels.append('UNKNOWN')

    return labels


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='输入CSV（svo_all.csv）')
    parser.add_argument('--output', default='svo_classified.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"读入 {len(df)} 行")

    # 分类
    df['vp_type'] = df.apply(lambda row: '+'.join(classify_vp(row)), axis=1)

    # 生成verbal_modifier_type
    ADVERBIAL_TYPES = {
        'V_resultative_simple', 'V_directional', 'V_loc_PP',
        'V_de_resultative', 'V_quantified', 'V_yi_V'
    }
    NOMINAL_TYPES = {'V_object', 'V_dat_PP'}

    def get_modifier_type(vp_type_str):
        tags = set(vp_type_str.split('+'))
        if tags & NOMINAL_TYPES:
            return 'nominal'
        if tags & ADVERBIAL_TYPES:
            return 'adverbial'
        return 'without_verbal_modifier'

    df['verbal_modifier_type'] = df['vp_type'].apply(get_modifier_type)

    # 统计
    print("\nVP类型分布（单标签，不互斥）：")
    all_labels = []
    for tags in df['vp_type']:
        all_labels.extend(tags.split('+'))
    from collections import Counter
    for label, count in Counter(all_labels).most_common():
        print(f"  {label}: {count}")

    print("\nverbal_modifier_type分布：")
    print(df['verbal_modifier_type'].value_counts().to_string())

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

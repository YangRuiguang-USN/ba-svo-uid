#!/usr/bin/env python3
"""
post_process.py
对已有SVO+把字句数据做三件事：
1. 加入vp_type_primary列（按优先级保留单一主要标签）
2. 处理模糊量词（些/点/点儿/个），更新object和vp_type
3. 只重新生成quant_fix不为空的行的把字句

用法：
    python post_process.py --input svo_filtered.csv --output svo_final.csv
"""

import argparse
import sys
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    sys.exit("请安装 pandas：pip install pandas")


# ══════════════════════════════════════════════════════════════
# VP优先级
# ══════════════════════════════════════════════════════════════

VP_PRIORITY = [
    'V_de_resultative',
    'V_resultative_simple',
    'V_directional',
    'V_loc_PP',
    'V_dat_PP',
    'V_quantified',
    'V_yi_V',
    'V_object',
    'V_aspect_le',
    'V_aspect_zhe',
    'V_aspect_guo',
    'UNKNOWN',
]

def get_primary(vp_type_str):
    tags = set(str(vp_type_str).split('+'))
    for vp in VP_PRIORITY:
        if vp in tags:
            return vp
    return 'UNKNOWN'


# ══════════════════════════════════════════════════════════════
# 模糊量词处理
# ══════════════════════════════════════════════════════════════

SAFE_FUZZY_QUANT = ['点儿', '点', '些']  # 长的先匹配

GE_BLACKLIST = {
    '个人', '个体', '个别', '个性', '个头', '个子',
    '个案', '个位', '个股'
}

def is_fuzzy_quant_object(obj_str):
    """检测object是否以模糊量词开头，返回(quantifier, noun)或(None, None)"""
    if not obj_str or obj_str == 'nan':
        return None, None
    for q in SAFE_FUZZY_QUANT:
        if obj_str.startswith(q) and len(obj_str) > len(q):
            return q, obj_str[len(q):]
    if obj_str.startswith('个') and len(obj_str) > 1:
        if obj_str not in GE_BLACKLIST and obj_str[:2] not in {w[:2] for w in GE_BLACKLIST}:
            return '个', obj_str[1:]
    return None, None


# ══════════════════════════════════════════════════════════════
# 依存解析辅助（用于重新生成把字句）
# ══════════════════════════════════════════════════════════════

SENTENCE_FINAL = {'吗', '呢', '啊', '嘛', '哦', '哈', '呀', '吧', '嗯', '哎', '喂'}

def parse_deps_raw(deps_raw_str, n_tokens):
    deps = []
    for item in str(deps_raw_str).strip().split():
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
    while len(deps) < n_tokens:
        deps.append((None, ''))
    return deps

def build_children(deps_0):
    children = defaultdict(list)
    for child, (head, label) in enumerate(deps_0):
        if head is not None:
            children[head].append((child, label))
    return children

def get_np_left_span(node_idx, children):
    def collect_left(idx, visited):
        if idx in visited:
            return
        visited.add(idx)
        for child_idx, label in children[idx]:
            if child_idx < idx:
                collect_left(child_idx, visited)
    visited = set()
    collect_left(node_idx, visited)
    if not visited:
        return node_idx, node_idx
    return min(visited), node_idx

def find_np2_span(object_str, verb_idx, tokens, deps_0, children, skip_prefix=None):
    """
    找NP2的完整span。
    skip_prefix: 如果不为None，在tokens里搜索时跳过这个前缀量词
    """
    if not object_str:
        return None

    # 如果有量词前缀，先尝试搜索"量词+名词"整体
    search_str = (skip_prefix + object_str) if skip_prefix else object_str

    n = len(tokens)
    obj_len = len(search_str)

    for start in range(verb_idx + 1, n):
        accumulated = ''
        for end in range(start, n):
            accumulated += tokens[end]
            if accumulated == search_str:
                # 找到了，NP2从量词后面开始（跳过量词）
                if skip_prefix:
                    # 找量词结束的位置
                    q_len = len(skip_prefix)
                    q_accumulated = ''
                    q_end = start
                    for q_end in range(start, end + 1):
                        q_accumulated += tokens[q_end]
                        if q_accumulated == skip_prefix:
                            return q_end + 1, end
                return start, end
            if len(accumulated) > obj_len:
                break

    # 退化：直接搜索object_str
    obj_len2 = len(object_str)
    for start in range(verb_idx + 1, n):
        accumulated = ''
        for end in range(start, n):
            accumulated += tokens[end]
            if accumulated == object_str:
                return start, end
            if len(accumulated) > obj_len2:
                break

    # 最后退化：单token匹配
    for i in range(n - 1, verb_idx, -1):
        if tokens[i] in object_str:
            np2_start, np2_end = get_np_left_span(i, children)
            if np2_end + 1 < n and tokens[np2_end + 1] in ('们', '等'):
                np2_end += 1
            return np2_start, np2_end

    return None

def extract_vp(tokens, verb_idx, np2_start, np2_end, quant_idx=None):
    """
    提取VP前半和后半。
    quant_idx: 量词在tokens里的起始位置，VP前半不包含量词
    """
    verb_token = tokens[verb_idx]
    # VP前半结束位置：如果有量词，到量词之前；否则到NP2之前
    vp_end = quant_idx if quant_idx is not None else np2_start
    vp_pre = tokens[verb_idx:vp_end]
    vp_post_raw = tokens[np2_end + 1:]
    while vp_post_raw and vp_post_raw[-1] in SENTENCE_FINAL:
        vp_post_raw = vp_post_raw[:-1]
    if vp_post_raw and vp_post_raw[0] == verb_token:
        vp_post_raw = vp_post_raw[1:]
    return vp_pre, vp_post_raw

def find_subject_idx(subject, verb_idx, tokens):
    if not subject:
        return None
    for i in range(verb_idx - 1, -1, -1):
        if tokens[i] == subject:
            return i
    return None

def generate_ba(row, quantifier=None):
    """
    重新生成把字句。
    quantifier: 如果不为None，在VP后面加上量词
    """
    tokens_str = str(row.get('tokens', ''))
    deps_str   = str(row.get('deps_raw', ''))
    verb_idx   = int(row.get('verb_index', 0))
    subject    = str(row.get('subject', '')) if pd.notna(row.get('subject')) else ''
    object_tok = str(row.get('object', ''))  if pd.notna(row.get('object'))  else ''

    tokens = tokens_str.split()
    n      = len(tokens)

    if verb_idx >= n or not object_tok:
        return '', 'SKIP_no_verb_or_object'

    deps_0   = parse_deps_raw(deps_str, n)
    children = build_children(deps_0)

    np2_span = find_np2_span(object_tok, verb_idx, tokens, deps_0, children,
                             skip_prefix=quantifier)
    if np2_span is None:
        return '', 'SKIP_np2_not_found'

    np2_start, np2_end = np2_span
    np2_tokens = tokens[np2_start:np2_end + 1]

    # 如果有量词，量词在np2_start之前，VP前半应该到量词之前
    # 找量词在tokens里的位置
    quant_idx = None
    if quantifier:
        # 量词紧挨在np2_start之前
        q_end = np2_start - 1
        q_accumulated = ''
        for qi in range(q_end, verb_idx, -1):
            g_accumulated = ''.join(tokens[qi:np2_start])
            if g_accumulated == quantifier:
                quant_idx = qi
                break

    vp_pre, vp_post = extract_vp(tokens, verb_idx, np2_start, np2_end, quant_idx=quant_idx)

    subj_idx = find_subject_idx(subject, verb_idx, tokens)
    if subj_idx is not None:
        pre_subj = tokens[:subj_idx]
        np1      = [tokens[subj_idx]]
        middle   = tokens[subj_idx + 1:verb_idx]
    else:
        pre_subj = tokens[:verb_idx]
        np1      = []
        middle   = []

    if len(vp_pre) == 1 and not vp_post and not quantifier:
        note = 'REVIEW_bare_verb'
    else:
        note = 'OK'

    parts = []
    parts.extend(pre_subj)
    parts.extend(np1)
    parts.extend(middle)
    parts.append('把')
    parts.extend(np2_tokens)
    parts.extend(vp_pre)
    parts.extend(vp_post)
    if quantifier:
        parts.append(quantifier)

    return ''.join(parts), note


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', default='svo_final.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"读入 {len(df)} 行")

    # ── 1. vp_type_primary ────────────────────────────────────
    # 兼容vp_type和VP_type两种列名
    if 'vp_type' not in df.columns and 'VP_type' in df.columns:
        df = df.rename(columns={'VP_type': 'vp_type'})

    df['vp_type_primary'] = df['vp_type'].apply(get_primary)
    print("\nvp_type_primary分布：")
    print(df['vp_type_primary'].value_counts().to_string())

    # ── 2. 模糊量词处理 ────────────────────────────────────────
    if 'quant_fix' not in df.columns:
        df['quant_fix'] = ''

    quant_counts = {}
    quant_fixed  = 0

    for idx, row in df.iterrows():
        obj_str = str(row.get('object', '')) if pd.notna(row.get('object')) else ''
        quantifier, noun = is_fuzzy_quant_object(obj_str)
        if quantifier is None:
            continue

        # 更新object（去掉量词）
        df.at[idx, 'object'] = noun

        # 更新vp_type
        old_vp = str(row.get('vp_type', ''))
        if 'V_quantified' not in old_vp:
            new_vp = 'V_quantified' if old_vp == 'UNKNOWN' else old_vp + '+V_quantified'
            df.at[idx, 'vp_type'] = new_vp
            df.at[idx, 'vp_type_primary'] = get_primary(new_vp)

        df.at[idx, 'quant_fix'] = quantifier
        quant_counts[quantifier] = quant_counts.get(quantifier, 0) + 1
        quant_fixed += 1

    print(f"\n模糊量词处理：")
    for q, c in sorted(quant_counts.items()):
        print(f"  {q}: {c}")
    print(f"总计：{quant_fixed} 行")

    # ── 3. 重新生成有量词修改的把字句 ─────────────────────────
    regen = 0
    for idx, row in df[df['quant_fix'] != ''].iterrows():
        quantifier = str(row.get('quant_fix', ''))
        ba, note = generate_ba(row, quantifier=quantifier)
        if ba:
            df.at[idx, 'ba_sentence'] = ba
            df.at[idx, 'ba_note']     = note
            regen += 1

    print(f"\n重新生成把字句：{regen} 行")

    # 抽样输出
    print("\n量词修改样本（前10条）：")
    sample = df[df['quant_fix'] != ''][['raw_clause', 'object', 'ba_sentence', 'vp_type_primary']].head(10)
    print(sample.to_string())

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

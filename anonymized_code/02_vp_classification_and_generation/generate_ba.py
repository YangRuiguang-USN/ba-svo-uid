#!/usr/bin/env python3
"""
generate_ba.py
根据SVO分类结果生成对应的把字句

用法：
    python generate_ba.py --input svo_classified.csv --output svo_with_ba.csv

生成结构：NP1 + 把 + NP2 + VP
- NP1：subject列（可为空，零主语）
- NP2：从deps_raw里找完整NP边界
- VP：从verb_index开始，跳过NP2，取后置成分到句末（去除语气词）
"""

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

# 句末语气词
SENTENCE_FINAL = {'吗', '呢', '啊', '嘛', '哦', '哈', '呀', '吧', '嗯', '哎', '喂'}

# 名词性词性标签
NOUN_POS = {'NN', 'NR', 'NT', 'PN', 'QP', 'CD', 'M'}


# ══════════════════════════════════════════════════════════════
# 依存解析辅助
# ══════════════════════════════════════════════════════════════

def parse_deps_raw(deps_raw_str, n_tokens):
    """解析deps_raw字符串，返回0-based依存列表"""
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
    """构建parent→[(child, label)]索引"""
    children = defaultdict(list)
    for child, (head, label) in enumerate(deps_0):
        if head is not None:
            children[head].append((child, label))
    return children


def get_np_left_span(node_idx, children):
    """
    找名词短语的左边界。
    递归收集所有依附于node_idx的左侧子节点。
    返回(min_idx, node_idx)
    """
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


def find_np2_span(object_str, verb_idx, tokens, deps_0, children):
    """
    找NP2的完整span。
    在tokens里verb_idx之后找object_str对应的连续子序列，
    返回(np2_start, np2_end)，如果找不到返回None
    """
    if not object_str:
        return None

    # 把object_str按字符拆开，在tokens里找连续匹配的子序列
    # tokens是分词结果，object_str是无空格拼接的字符串
    # 策略：找verb_idx之后连续tokens拼接等于object_str的起始位置
    n = len(tokens)
    obj_len = len(object_str)

    for start in range(verb_idx + 1, n):
        # 从start开始累积token字符串，直到长度超过object_str
        accumulated = ''
        for end in range(start, n):
            accumulated += tokens[end]
            if accumulated == object_str:
                return start, end
            if len(accumulated) > obj_len:
                break

    # 如果完整匹配失败，退化到单token匹配（找最后一个包含的token）
    for i in range(n - 1, verb_idx, -1):
        if tokens[i] in object_str:
            np2_start, np2_end = get_np_left_span(i, children)
            if np2_end + 1 < n and tokens[np2_end + 1] in ('们', '等'):
                np2_end += 1
            return np2_start, np2_end

    return None


# ══════════════════════════════════════════════════════════════
# VP提取
# ══════════════════════════════════════════════════════════════

def extract_vp(tokens, verb_idx, np2_start, np2_end):
    """
    提取VP：
    - VP前半：verb_idx到np2_start之间（不含NP2）
    - VP后半：np2_end+1到句末（去除语气词）
    - 去除句末语气词
    - 跳过VP后半开头与动词相同的token（口语重复动词）
    """
    verb_token = tokens[verb_idx]

    # VP前半：动词到NP2左边界之间
    vp_pre = tokens[verb_idx:np2_start]

    # VP后半：NP2右边界之后到句末
    vp_post_raw = tokens[np2_end + 1:]

    # 去除句末语气词
    while vp_post_raw and vp_post_raw[-1] in SENTENCE_FINAL:
        vp_post_raw = vp_post_raw[:-1]

    # 跳过开头与动词相同的token（口语重复）
    if vp_post_raw and vp_post_raw[0] == verb_token:
        vp_post_raw = vp_post_raw[1:]

    vp_post = vp_post_raw

    return vp_pre, vp_post


# ══════════════════════════════════════════════════════════════
# 生成把字句
# ══════════════════════════════════════════════════════════════

def find_subject_idx(subject, verb_idx, tokens):
    """找主语在tokens里verb_idx之前最近的位置"""
    if not subject:
        return None
    for i in range(verb_idx - 1, -1, -1):
        if tokens[i] == subject:
            return i
    return None


def generate_ba(row):
    """
    对单行生成把字句，返回(ba_sentence, note)。

    拼接顺序：
    [句首成分] + NP1 + [中间成分] + 把 + NP2 + VP前半 + VP后半
    """
    tokens_str  = str(row.get('tokens', ''))
    deps_str    = str(row.get('deps_raw', ''))
    verb_idx    = int(row.get('verb_index', 0))
    subject     = str(row.get('subject', '')) if pd.notna(row.get('subject')) else ''
    object_tok  = str(row.get('object', ''))  if pd.notna(row.get('object'))  else ''

    tokens  = tokens_str.split()
    n       = len(tokens)

    if verb_idx >= n or not object_tok:
        return '', 'SKIP_no_verb_or_object'

    deps_0   = parse_deps_raw(deps_str, n)
    children = build_children(deps_0)

    # 找NP2完整span
    np2_span = find_np2_span(object_tok, verb_idx, tokens, deps_0, children)
    if np2_span is None:
        return '', 'SKIP_np2_not_found'

    np2_start, np2_end = np2_span
    np2_tokens = tokens[np2_start:np2_end + 1]

    # 提取VP
    vp_pre, vp_post = extract_vp(tokens, verb_idx, np2_start, np2_end)

    # 找主语位置
    subj_idx = find_subject_idx(subject, verb_idx, tokens)

    if subj_idx is not None:
        # 句首成分：主语之前的所有tokens
        pre_subj = tokens[:subj_idx]
        # NP1：主语token
        np1 = [tokens[subj_idx]]
        # 中间成分：主语之后到动词之前的所有tokens
        middle = tokens[subj_idx + 1:verb_idx]
    else:
        # 零主语：没有主语
        # 句首成分：动词之前的所有tokens
        pre_subj = tokens[:verb_idx]
        np1 = []
        middle = []

    # 检查VP是否为光杆动词
    if len(vp_pre) == 1 and not vp_post:
        note = 'REVIEW_bare_verb'
    else:
        note = 'OK'

    # 拼接：[句首成分] + NP1 + [中间成分] + 把 + NP2 + VP前半 + VP后半
    parts = []
    parts.extend(pre_subj)
    parts.extend(np1)
    parts.extend(middle)
    parts.append('把')
    parts.extend(np2_tokens)
    parts.extend(vp_pre)
    parts.extend(vp_post)

    ba_sentence = ''.join(parts)
    return ba_sentence, note


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='输入CSV（svo_classified.csv）')
    parser.add_argument('--output', default='svo_with_ba.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"读入 {len(df)} 行")

    results = df.apply(generate_ba, axis=1)
    df['ba_sentence'] = results.apply(lambda x: x[0])
    df['ba_note']     = results.apply(lambda x: x[1])

    # 统计
    print("\nba_note分布：")
    print(df['ba_note'].value_counts().to_string())

    print("\n生成样本（前20条OK）：")
    ok = df[df['ba_note'] == 'OK'][['svo_text', 'ba_sentence', 'vp_type']].head(20)
    print(ok.to_string())

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

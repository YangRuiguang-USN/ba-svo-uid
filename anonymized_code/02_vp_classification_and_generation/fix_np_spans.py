#!/usr/bin/env python3
"""
fix_np_spans.py
重新从依存树提取完整的NP1（subject）和NP2（object）span，
替换原来只有核心词的subject/object列。

新增列：
  subject_full  — 完整NP1字符串
  object_full   — 完整NP2字符串

用法：
    python fix_np_spans.py --input svo_classified.csv --output svo_fixed_np.csv
"""

import argparse
import sys
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    sys.exit("请安装 pandas：pip install pandas")


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
    找名词短语的完整左侧span。
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


def get_np_right_span(node_idx, tokens, children):
    """
    扩展NP右边界：如果核心词后紧跟'们'/'等'，纳入右边界。
    """
    right = node_idx
    if right + 1 < len(tokens) and tokens[right + 1] in ('们', '等'):
        right += 1
    return right


# ══════════════════════════════════════════════════════════════
# NP提取
# ══════════════════════════════════════════════════════════════

SUBJ_LABELS = {'nsubj', 'nsubjpass', 'SBV', 'top', 'TOP'}
OBJ_LABELS  = {'dobj', 'obj', 'VOB', 'OBJ'}


def extract_full_nps(row):
    """
    从依存树重新提取完整NP1和NP2。
    返回(subject_full, object_full)
    """
    tokens_str = str(row.get('tokens', ''))
    deps_str   = str(row.get('deps_raw', ''))
    verb_idx   = int(row.get('verb_index', 0))

    tokens = tokens_str.split()
    n      = len(tokens)

    if verb_idx >= n:
        return '', ''

    deps_0   = parse_deps_raw(deps_str, n)
    children = build_children(deps_0)

    subject_full = ''
    object_full  = ''

    for child_idx, label in children[verb_idx]:
        # ── NP1（主语）────────────────────────────────────────
        if label in SUBJ_LABELS and not subject_full:
            left, right = get_np_left_span(child_idx, children)
            right = get_np_right_span(right, tokens, children)
            subject_full = ''.join(tokens[left:right + 1])

        # ── NP2（宾语）────────────────────────────────────────
        if label in OBJ_LABELS and not object_full:
            left, right = get_np_left_span(child_idx, children)
            right = get_np_right_span(right, tokens, children)
            # NP2必须在动词后面
            if child_idx > verb_idx:
                object_full = ''.join(tokens[left:right + 1])

    return subject_full, object_full


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', default='svo_fixed_np.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"读入 {len(df)} 行")

    results = df.apply(extract_full_nps, axis=1)
    df['subject_full'] = results.apply(lambda x: x[0])
    df['object_full']  = results.apply(lambda x: x[1])

    # 统计对比
    subj_changed = (df['subject_full'] != df['subject'].fillna('')).sum()
    obj_changed  = (df['object_full']  != df['object'].fillna('')).sum()
    print(f"\nsubject有变化的行：{subj_changed}")
    print(f"object有变化的行：{obj_changed}")

    # 删除原始列，重命名full列为subject/object
    df = df.drop(columns=['subject', 'object'])
    df = df.rename(columns={'subject_full': 'subject', 'object_full': 'object'})

    # 抽样对比
    print("\nobject变化样本（前10条）：")
    changed = df[df['object_full'] != df['object'].fillna('')]
    print(changed[['svo_text', 'object', 'object_full']].head(10).to_string())

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

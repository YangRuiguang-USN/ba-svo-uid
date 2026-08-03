#!/usr/bin/env python3
"""
auto_filter.py
根据四条规则自动标记qualite=0

规则：
1. raw_clause中语气词出现在句中（非句末）→ qualite=0, reason=ORAL_FILLER
2. raw_clause中有重复词语（任意位置）→ qualite=0, reason=REPEAT
3. decision列不是keep → qualite=0, reason=SUBORD_OR_REVIEW
4. ba_note=REVIEW_bare_verb → qualite=0, reason=BARE_VERB

只处理qualite为空的行，已标注的不覆盖。

用法：
    python auto_filter.py --input svo_with_ba.csv --output svo_filtered.csv
"""

import argparse
import sys
import re

try:
    import pandas as pd
except ImportError:
    sys.exit("请安装 pandas：pip install pandas")


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 句中语气词（出现在非句末位置时标0）
FILLER_WORDS = {'呃', '啊', '嗯', '哦', '哈', '呀', '嘛', '吧', '哎', '喂', '嘿'}

# 重叠式动词（不算重复）
REDUPLICATION_OK = set()
# 动态生成：两字相同的叠词（看看/想想/说说等）
# 在检测时单独处理


def has_filler_in_middle(text):
    """
    检测raw_clause中是否有语气词出现在句中（非句末）。
    """
    if not isinstance(text, str):
        return False
    # 去掉句末最后一个字符再检测
    text_no_end = text[:-1] if text else text
    for fw in FILLER_WORDS:
        if fw in text_no_end:
            return True
    return False


def has_repeat_word(tokens_str):
    """
    检测tokens里是否有重复出现的词（任意位置）。
    排除重叠式动词（AA型，如看看/想想）。
    排除高频虚词（了/的/个/一/也/都/就/是/有/在等）。
    """
    if not isinstance(tokens_str, str):
        return False

    # 不计入重复检测的高频虚词
    STOPWORDS = {
        '了', '的', '个', '一', '也', '都', '就', '是', '有', '在',
        '和', '与', '或', '把', '被', '让', '给', '对', '从', '到',
        '不', '没', '很', '太', '真', '还', '又', '再', '更', '最',
        '啊', '吧', '呢', '吗', '嘛', '哦', '哈', '呀', '嗯', '哎',
        '这', '那', '什么', '哪', '谁', '怎么', '为什么',
        '我', '你', '他', '她', '它', '们', '自己',
    }

    tokens = tokens_str.split()
    seen = {}
    for i, t in enumerate(tokens):
        # 跳过虚词
        if t in STOPWORDS:
            continue
        # 排除单字重叠式（AA型）：连续两个相同单字token
        if len(t) == 1 and i > 0 and tokens[i-1] == t:
            continue
        # 排除双字重叠式（AABB型）：token本身是两个相同字
        if len(t) == 2 and t[0] == t[1]:
            continue
        if t in seen:
            return True
        seen[t] = i
    return False


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', default='svo_filtered.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"读入 {len(df)} 行")

    # 如果quality列不存在则创建
    if 'quality' not in df.columns:
        df['quality'] = ''

    # 只处理quality为空的行
    empty_mask = df['quality'].isna() | (df['quality'] == '')

    # 初始化auto_filter列（如果不存在）
    if 'auto_filter' not in df.columns:
        df['auto_filter'] = ''

    counts = {'ORAL_FILLER': 0, 'REPEAT': 0, 'SUBORD': 0, 'BARE_VERB': 0}

    for idx in df[empty_mask].index:
        row = df.loc[idx]

        # 规则3：decision=exclude（明确从句/嵌入结构）
        if str(row.get('decision', '')) == 'exclude':
            df.at[idx, 'quality'] = 0
            df.at[idx, 'auto_filter'] = 'SUBORD'
            counts['SUBORD'] += 1
            continue

        # 规则4：光杆动词
        if str(row.get('ba_note', '')) == 'REVIEW_bare_verb':
            df.at[idx, 'quality'] = 0
            df.at[idx, 'auto_filter'] = 'BARE_VERB'
            counts['BARE_VERB'] += 1
            continue

        # 规则1：句中语气词
        if has_filler_in_middle(row.get('raw_clause', '')):
            df.at[idx, 'quality'] = 0
            df.at[idx, 'auto_filter'] = 'ORAL_FILLER'
            counts['ORAL_FILLER'] += 1
            continue

        # 规则2：重复词语
        if has_repeat_word(row.get('tokens', '')):
            df.at[idx, 'quality'] = 0
            df.at[idx, 'auto_filter'] = 'REPEAT'
            counts['REPEAT'] += 1
            continue

    print("\n自动标记统计：")
    for reason, count in counts.items():
        print(f"  {reason}: {count}")

    total_auto = sum(counts.values())
    still_empty = (df['quality'].isna() | (df['quality'] == '')).sum()
    print(f"\n自动标记总数：{total_auto}")
    print(f"仍需人工处理：{still_empty}")

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
fix_svo_asp.py

根据正确模板重新生成以下两类的svo字段：

V_aspect_le：
    svo_core = V_core + trailing_asp + internal_3 + internal_2 + NP2

V_resultative_simple：
    svo_core = vp_no_asp + trailing_asp + internal_3 + internal_2 + NP2

完整SVO = prefix + NP1 + internal_1 + svo_core + suffix

跳过以下行（不修改）：
    - svo_invalid=1
    - svo字段为空
    - VP_type_primary不是这两类

用法：
    python fix_svo_asp.py --input your_csv.csv --output fixed.csv
"""

import csv
import argparse
import os


def strip_aspect(s: str) -> tuple[str, str]:
    """去掉末尾一个了/过，返回(stripped, asp)"""
    for asp in ('了', '过'):
        if s.endswith(asp):
            return s[:-1], asp
    return s, ''


def get_boundaries(sentence: str, np1: str, np2: str, vp: str) -> dict:
    """从sentence里重新计算prefix/suffix/internal_1"""
    result = {'prefix': '', 'suffix': '', 'internal_1': ''}

    np2_start = sentence.find(np2)
    if np2_start == -1:
        return result

    vp_start = sentence.find(vp, np2_start + len(np2))
    if vp_start == -1:
        vp_start = sentence.find(vp)
    if vp_start == -1:
        return result
    vp_end = vp_start + len(vp)

    ba_pos = sentence[:np2_start].rfind('把')

    if np1:
        np1_start = sentence.find(np1)
        if np1_start != -1:
            np1_end = np1_start + len(np1)
            result['prefix'] = sentence[:np1_start]
            if ba_pos != -1:
                result['internal_1'] = sentence[np1_end:ba_pos].strip()
        else:
            result['prefix'] = sentence[:ba_pos] if ba_pos != -1 else ''
    else:
        result['prefix'] = sentence[:ba_pos] if ba_pos != -1 else ''

    result['suffix'] = sentence[vp_end:]
    return result


def fix_row(row: dict) -> dict:
    primary = row.get('VP_type_primary', '').strip()
    if primary not in ('V_resultative_simple', 'V_aspect_le'):
        return row

    if row.get('svo_invalid', '0') == '1':
        return row

    if not row.get('svo', '').strip():
        return row

    np1        = row.get('NP1', '').strip()
    np2        = row.get('NP2', '').strip()
    vp         = row.get('VP', '').strip()
    v_core     = row.get('V_core', '').strip()
    internal_2 = row.get('internal_2', '').strip()
    internal_3 = row.get('internal_3', '').strip()

    # sentence字段可能重复，取第一个非空的
    sentence = ''
    for key in row:
        if key == 'sentence' and row[key].strip():
            sentence = row[key].strip()
            break

    if not np2 or not vp or not sentence:
        return row

    # 动态计算prefix/suffix/internal_1
    bounds = get_boundaries(sentence, np1, np2, vp)
    prefix     = bounds['prefix']
    suffix     = bounds['suffix']
    internal_1 = bounds['internal_1']

    vp_no_asp, trailing_asp = strip_aspect(vp)

    # 两类统一模板：vp + internal_3 + internal_2 + NP2
    svo_core = vp + internal_3 + internal_2 + np2

    new_svo = prefix + np1 + internal_1 + svo_core + suffix

    row = dict(row)
    row['svo'] = new_svo
    return row


def process(input_path: str, output_path: str):
    with open(input_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        print("输入文件为空。")
        return

    modified = 0
    output_rows = []
    for row in rows:
        new_row = fix_row(row)
        if new_row.get('svo') != row.get('svo'):
            modified += 1
        output_rows.append(new_row)

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n处理完成：{len(rows)} 条")
    print(f"修正行数：{modified} 条")
    print(f"输出文件：{output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='修正V_aspect_le和V_resultative_simple的svo生成')
    parser.add_argument('--input',  required=True, help='输入CSV路径')
    parser.add_argument('--output', required=True, help='输出CSV路径')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在 → {args.input}")
        return

    process(args.input, args.output)


if __name__ == '__main__':
    main()

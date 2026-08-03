# -*- coding: utf-8 -*-
"""
generate_svo_v3.1.py
把字句 → SVO 语序转换脚本

输入字段（必须）: sentence, NP1, NP2, VP
输入字段（辅助）: verbal_modifier, verbal_modifier_position, verbal_modifier_type

输出新增字段:
    V_core              从VP提取的动词核心
    internal_1          NP1和把之间的残余成分
    internal_2          把和NP2之间的残余成分
    internal_3          NP2和VP之间的残余成分
    VP_type             多值，| 分隔
    VP_type_primary     单值，驱动生成
    svo                 生成的SVO句，invalid时为空
    svo_invalid         1=排除出对照集
    svo_review_flag     1=需人工确认
    svo_review_reason   触发原因，| 分隔

用法:
    python generate_svo_v3.6.py --input annotated.csv --output svo_output.csv
"""

import csv
import re
import argparse
import os

# ══════════════════════════════════════════════════════════════════
# 词表
# ══════════════════════════════════════════════════════════════════

# 结果补语词表（状态变化义）
RESULTATIVE_SIMPLE = {
    '好', '完', '碎', '坏', '掉', '走', '开', '错', '断', '烂',
    '裂', '死', '净', '光', '空', '满', '透', '通', '懂', '住',
    '定', '稳', '准', '清', '醒', '响', '亮', '干', '湿', '软',
    '硬', '平', '直', '弯', '歪', '正', '紧', '松', '冷', '热',
    # 扩充：口语语料常见
    '够', '瞎', '哭', '牢', '跑', '倒', '晴', '火',
}

# 双字趋向补语词表
DIRECTIONAL_BI = {
    '出来', '进来', '上来', '下来', '回来', '过来', '起来',
    '出去', '进去', '上去', '下去', '回去', '过去',
}

# 单字趋向补语
DIRECTIONAL_MONO = {'来', '去'}

# 述谓动词词表（V_object）
OBJECT_PRED_VERBS = {
    '当', '叫', '称', '视为', '看作', '看成', '当成', '当作',
    '封', '立', '选', '推', '拜',
}

# verbal_modifier以成/为开头时也是V_object信号
OBJECT_VM_PREFIXES = {'成', '为'}

# 与格介词
DATIVE_PREPS = {'给', '向', '对'}

# 处所介词
LOCATIVE_PREPS = {'在', '到', '于'}

# 动量/时量词（V_quantified专用，排除名量词）
VERBAL_MEASURE_WORDS = {
    '次', '遍', '回', '下', '趟', '场',
    '天', '年', '月', '日', '周',
    '小时', '分钟', '秒',
}

# ══════════════════════════════════════════════════════════════════
# 正则
# ══════════════════════════════════════════════════════════════════

# 动量/时量短语：数词+动量词，或好几+动量词
_mw = '|'.join(VERBAL_MEASURE_WORDS)
RE_VERBAL_QUANT = re.compile(
    r'([零一二三四五六七八九十百千万两几多好][一二三四五六七八九十百千万]?'
    r'(?:' + _mw + r')'
    r'|好几(?:' + _mw + r')'
    r'|[0-9]+(?:' + _mw + r'))'
)

# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def strip_aspect(s: str) -> tuple[str, str]:
    """
    去掉字符串末尾的一个体标记（了/着/过）。
    返回 (stripped, trailing_asp)
    """
    for asp in ('了', '着', '过'):
        if s.endswith(asp):
            return s[:-1], asp
    return s, ''


def is_yi_v(vp: str) -> bool:
    """
    判断是否为尝试体结构：
    - VV叠词：看看 / 想想 / 考虑考虑
    - V一V：看一看 / 试一试
    """
    vp = vp.strip()
    n = len(vp)

    # 单字VV：看看
    if n == 2 and vp[0] == vp[1]:
        return True
    # 双字VV：考虑考虑
    if n == 4 and vp[:2] == vp[2:]:
        return True
    # 单字V一V：看一看
    if n == 3 and vp[1] == '一' and vp[0] == vp[2]:
        return True
    # 双字V一V：考虑一考虑
    if n == 5 and vp[2] == '一' and vp[:2] == vp[3:]:
        return True
    return False


def starts_with_pred_verb(vp: str) -> bool:
    """判断VP是否以述谓动词开头"""
    for v in sorted(OBJECT_PRED_VERBS, key=len, reverse=True):
        if vp.startswith(v):
            return True
    return False


def contains_dative(vm: str) -> bool:
    return any(p in vm for p in DATIVE_PREPS)


def contains_locative(vm: str) -> bool:
    return any(p in vm for p in LOCATIVE_PREPS)


# ══════════════════════════════════════════════════════════════════
# Step 1：句子结构分析
# ══════════════════════════════════════════════════════════════════

def analyze_sentence(sentence: str, np1: str, np2: str,
                     vp: str) -> dict:
    """
    在sentence里定位各成分边界。
    返回包含 prefix / suffix / internal_1/2/3 及错误信息的字典。
    """
    result = {
        'prefix': '',
        'suffix': '',
        'internal_1': '',
        'internal_2': '',
        'internal_3': '',
        'invalid': 0,
        'review_flags': [],
    }

    # NP2 / VP 缺失直接排除
    if not np2:
        result['invalid'] = 1
        result['review_flags'].append('NP2_MISSING')
        return result
    if not vp:
        result['invalid'] = 1
        result['review_flags'].append('VP_MISSING')
        return result

    # 定位把字：在NP2之前找最近的把（rfind避免多个把时定位错误）
    # 需要先定位NP2才能用rfind，所以把字定位放在NP2定位之后

    # 定位NP2（先于把字定位）
    np2_start = sentence.find(np2)
    if np2_start == -1:
        result['invalid'] = 1
        result['review_flags'].append('BOUNDARY_FAILED')
        return result
    np2_end = np2_start + len(np2)

    # 在NP2之前找最近的把
    ba_pos = sentence[:np2_start].rfind('把')
    if ba_pos == -1:
        result['review_flags'].append('BA_NOT_FOUND')

    # 定位VP（从NP2结束位置之后开始找，避免和NP2重叠）
    vp_start = sentence.find(vp, np2_end)
    if vp_start == -1:
        # fallback：从头找
        vp_start = sentence.find(vp)
    if vp_start == -1:
        result['invalid'] = 1
        result['review_flags'].append('BOUNDARY_FAILED')
        return result
    vp_end = vp_start + len(vp)

    # 定位NP1
    if np1:
        np1_start = sentence.find(np1)
        if np1_start == -1:
            result['review_flags'].append('BOUNDARY_FAILED')
            np1_start = ba_pos if ba_pos != -1 else 0
        np1_end = np1_start + len(np1)
        left_bound = np1_start
    else:
        np1_start = -1
        np1_end   = ba_pos if ba_pos != -1 else 0
        left_bound = np1_end

    # prefix / suffix
    result['prefix'] = sentence[:left_bound]
    result['suffix'] = sentence[vp_end:]

    # internal_1：NP1末尾到把字之间
    if np1 and np1_start != -1 and ba_pos != -1:
        result['internal_1'] = sentence[np1_end:ba_pos].strip()

    # internal_2：把字之后到NP2开头之间
    if ba_pos != -1:
        result['internal_2'] = sentence[ba_pos + 1:np2_start].strip()

    # internal_3：NP2末尾到VP开头之间
    result['internal_3'] = sentence[np2_end:vp_start].strip()

    # review flags for non-empty internals / suffix
    if result['suffix']:
        result['review_flags'].append(
            f"SUFFIX_NONEMPTY:{result['suffix']}"
        )
    if result['internal_1']:
        result['review_flags'].append(
            f"INTERNAL_1:{result['internal_1']}"
        )
    if result['internal_2']:
        result['review_flags'].append(
            f"INTERNAL_2:{result['internal_2']}"
        )
    if result['internal_3']:
        result['review_flags'].append(
            f"INTERNAL_3:{result['internal_3']}"
        )

    return result


# ══════════════════════════════════════════════════════════════════
# Step 2：VP_type检测
# ══════════════════════════════════════════════════════════════════

PRIORITY = [
    'V_de_resultative',
    'V_yi_V',
    'V_object',
    'V_directional',
    'V_resultative_simple',
    'V_quantified',
    'V_dat_PP',
    'V_loc_PP',
    'V_aspect_zhe',
    'V_aspect_le',
    'V_aspect_guo',
    'V_adv',
    'UNKNOWN',
]


def detect_vp_type(vp: str, vm: str,
                   vmp: str, vmt: str) -> tuple[list, str]:
    """
    检测VP结构类型。
    返回 (types: list, primary: str)
    """
    vp  = vp.strip()
    vm  = (vm or '').strip()
    vmp = (vmp or '').strip().lower()
    vmt = (vmt or '').strip().lower()

    detected = []

    # 1. V_de_resultative
    if '得' in vp:
        idx = vp.index('得')
        if idx < len(vp) - 1:  # 得后非空
            detected.append('V_de_resultative')

    # 2. V_object：VP以述谓动词开头，或VM以成/为开头
    if starts_with_pred_verb(vp):
        detected.append('V_object')
    elif vm and any(vm.lstrip().startswith(p) for p in OBJECT_VM_PREFIXES):
        detected.append('V_object')

    # 3. V_directional（双字优先，再检测单字）
    vp_no_asp, _ = strip_aspect(vp)
    found_directional = False
    for dc in sorted(DIRECTIONAL_BI, key=len, reverse=True):
        if vp_no_asp.endswith(dc):
            detected.append('V_directional')
            found_directional = True
            break
    if not found_directional:
        for dc in DIRECTIONAL_MONO:
            if vp_no_asp.endswith(dc):
                detected.append('V_directional')
                break

    # 4. V_resultative_simple（去掉末尾了/着/过后末字在词表里）
    if vp_no_asp and vp_no_asp[-1] in RESULTATIVE_SIMPLE:
        detected.append('V_resultative_simple')

    # 5. V_quantified（verbal_modifier优先，否则从VP里找）
    vm_check = vm if vm else vp
    if RE_VERBAL_QUANT.search(vm_check):
        detected.append('V_quantified')

    # 6. V_yi_V
    if is_yi_v(vp):
        detected.append('V_yi_V')

    # 7. V_dat_PP
    if vm and contains_dative(vm) and 'post' in vmp:
        detected.append('V_dat_PP')

    # 8. V_loc_PP
    if vm and contains_locative(vm) and 'post' in vmp:
        detected.append('V_loc_PP')

    # 9-11. 体标记
    if vp.endswith('着'):
        detected.append('V_aspect_zhe')
    if vp.endswith('了'):
        detected.append('V_aspect_le')
    if vp.endswith('过'):
        detected.append('V_aspect_guo')

    # 12. V_adv
    if vmt == 'adverbial' and 'pre' in vmp:
        detected.append('V_adv')

    # 13. UNKNOWN
    if not detected:
        detected.append('UNKNOWN')

    # primary：按优先级取第一个
    primary = 'UNKNOWN'
    for p in PRIORITY:
        if p in detected:
            primary = p
            break

    return detected, primary


# ══════════════════════════════════════════════════════════════════
# Step 3：V_core提取
# ══════════════════════════════════════════════════════════════════

def extract_vcore(vp: str, vm: str, vmp: str) -> tuple[str, str, list]:
    """
    从VP中提取V_core和trailing_asp。
    返回 (v_core, trailing_asp, review_flags)
    """
    vp  = vp.strip()
    vm  = (vm or '').strip()
    vmp = (vmp or '').strip().lower()
    flags = []

    # both → invalid（在Step 4处理，这里返回空）
    if 'both' in vmp:
        return '', '', flags

    # none
    if 'none' in vmp or not vm:
        v_core, trailing_asp = strip_aspect(vp)
        return v_core, trailing_asp, flags

    # pre-verbal
    if 'pre' in vmp and 'post' not in vmp:
        if vp.startswith(vm):
            rest = vp[len(vm):]
            v_core, trailing_asp = strip_aspect(rest)
            return v_core, trailing_asp, flags
        else:
            flags.append('VM_NOT_FOUND')
            v_core, trailing_asp = strip_aspect(vp)
            return v_core, trailing_asp, flags

    # post-verbal
    if 'post' in vmp and 'pre' not in vmp:
        vm_start = vp.find(vm)
        if vm_start == -1:
            flags.append('VM_NOT_FOUND')
            v_core, trailing_asp = strip_aspect(vp)
            return v_core, trailing_asp, flags

        before_vm = vp[:vm_start]
        after_vm  = vp[vm_start + len(vm):]

        # VM等于整个VP（before_vm为空）：V_core退回strip_aspect(VP)
        if not before_vm:
            flags.append('VM_EQUALS_VP')
            v_core, trailing_asp = strip_aspect(vp)
            return v_core, trailing_asp, flags

        if after_vm in ('了', '着', '过'):
            trailing_asp = after_vm
            v_core = before_vm
        else:
            trailing_asp = ''
            v_core = before_vm

        return v_core, trailing_asp, flags

    # fallback
    v_core, trailing_asp = strip_aspect(vp)
    return v_core, trailing_asp, flags


# ══════════════════════════════════════════════════════════════════
# Step 4 & 5：生成策略 + SVO生成
# ══════════════════════════════════════════════════════════════════

# A类：直接拼接（verbal_modifier为空或none时）
CLASS_A = {
    'V_yi_V',
    'V_aspect_le',
    'V_aspect_guo',
    'V_aspect_zhe',
}

# C类：直接排除
CLASS_C = {'UNKNOWN'}


def generate_svo(np1: str, np2: str, vp: str,
                 vm: str, vmp: str,
                 vp_primary: str,
                 v_core: str, trailing_asp: str,
                 prefix: str, suffix: str,
                 internal_1: str, internal_2: str,
                 internal_3: str) -> tuple[str, int, int, list]:
    """
    生成SVO句。
    返回 (svo, svo_invalid, svo_review_flag, review_reasons)
    """
    np1 = np1.strip()
    np2 = np2.strip()
    vm  = (vm or '').strip()
    vmp = (vmp or '').strip().lower()

    invalid = 0
    review  = 0
    reasons = []

    # both → invalid
    if 'both' in vmp:
        return '', 1, 0, ['BOTH_POSITION']

    # C类 → invalid
    if vp_primary in CLASS_C:
        return '', 1, 0, ['UNKNOWN_TYPE']

    # ── SVO核心生成 ───────────────────────────────────────────────

    svo_core = ''

    # A类：V_core + internal_3 + internal_2 + NP2 + trailing_asp
    if vp_primary in CLASS_A:
        svo_core = v_core + internal_3 + internal_2 + np2 + trailing_asp

    # V_resultative_simple：VP整体 + internal_3 + internal_2 + NP2 + trailing_asp
    elif vp_primary == 'V_resultative_simple':
        vp_no_asp, t_asp = strip_aspect(vp)
        svo_core = vp_no_asp + internal_3 + internal_2 + np2 + t_asp

    # without verbal modifier或vmp=none：退化A类
    elif not vm or 'none' in vmp:
        svo_core = v_core + internal_3 + internal_2 + np2 + trailing_asp

    # B类
    else:
        if 'post' in vmp and 'pre' not in vmp:

            # V_de_resultative
            if vp_primary == 'V_de_resultative':
                if vm.startswith('得'):
                    vm_after_de = vm[1:]
                    svo_core = v_core + '得' + internal_3 + internal_2 + np2 + vm_after_de + trailing_asp
                else:
                    svo_core = v_core + '得' + internal_3 + internal_2 + np2 + vm + trailing_asp
                review = 1
                reasons.append('DE_REORDER')

            # 其余B类(post)：V_core + internal_3 + internal_2 + NP2 + VM + trailing_asp
            else:
                svo_core = v_core + internal_3 + internal_2 + np2 + vm + trailing_asp
                if vp_primary == 'V_directional':
                    review = 1
                    reasons.append('DIRECTIONAL_REVIEW')
                elif vp_primary == 'V_dat_PP':
                    review = 1
                    reasons.append('DAT_PP_REVIEW')
                elif vp_primary == 'V_loc_PP':
                    review = 1
                    reasons.append('LOC_PP_REVIEW')

        # B类(pre)：VM + internal_3 + V_core + internal_2 + NP2 + trailing_asp
        elif 'pre' in vmp and 'post' not in vmp:
            svo_core = vm + internal_3 + v_core + internal_2 + np2 + trailing_asp

        # B类(none)：退化为A类
        else:
            svo_core = v_core + internal_3 + internal_2 + np2 + trailing_asp

    # ── 完整拼接 ──────────────────────────────────────────────────
    # internal_2和internal_3已在各类svo_core里统一处理
    svo = prefix + np1 + internal_1 + svo_core + suffix

    # ── 通用追加检查 ──────────────────────────────────────────────
    # internal_3非空：生成候选SVO但强制review，排除出主分析集
    if internal_3:
        review = 1
        # reason已在analyze_sentence里记录为INTERNAL_3:内容，这里不重复添加

    if len(np2) > 6:
        review = 1
        reasons.append('NP2_LONG')
    if len(vp) > 10:
        review = 1
        reasons.append('VP_LONG')
    if '把' in vp or '被' in vp:
        review = 1
        reasons.append('EMBEDDED_BA_BEI')

    return svo, invalid, review, reasons


# ══════════════════════════════════════════════════════════════════
# 主处理
# ══════════════════════════════════════════════════════════════════

def process(input_path: str, output_path: str):
    with open(input_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        print("输入文件为空。")
        return

    required = {'sentence', 'NP1', 'NP2', 'VP'}
    missing = required - set(fieldnames)
    if missing:
        print(f"错误：输入文件缺少必要字段 {missing}")
        return

    # 过滤掉输入里的旧字段（SVO_sentence/main_verb），统一用新字段名输出
    OLD_FIELDS = {'SVO_sentence'}
    fieldnames = [f for f in fieldnames if f not in OLD_FIELDS]

    new_fields = [
        'V_core',
        'internal_1', 'internal_2', 'internal_3',
        'VP_type', 'VP_type_primary',
        'svo_invalid', 'svo_review_flag', 'svo_review_reason',
    ]
    # svo和main_verb：已存在则覆盖，不存在则新增
    for f in ['svo', 'main_verb']:
        if f not in fieldnames:
            new_fields.append(f)
    out_fields = fieldnames + new_fields

    stats = {
        'total': 0, 'invalid': 0, 'review': 0, 'clean': 0,
    }
    for t in PRIORITY:
        stats[t] = 0

    output_rows = []

    for r in rows:
        sentence = r.get('sentence', '')
        np1 = r.get('NP1', '')
        np2 = r.get('NP2', '')
        vp  = r.get('VP', '')
        vm  = r.get('verbal_modifier', '')
        vmp = r.get('verbal_modifier_position', '')
        vmt = r.get('verbal_modifier_type', '')

        # Step 1
        struct = analyze_sentence(sentence, np1, np2, vp)

        # Step 2
        types, primary = detect_vp_type(vp, vm, vmp, vmt)

        # Step 3
        v_core, trailing_asp, vcore_flags = extract_vcore(vp, vm, vmp)

        # Step 4 & 5
        all_review_reasons = struct['review_flags'] + vcore_flags

        if struct['invalid']:
            svo      = ''
            invalid  = 1
            review   = 0
            gen_reasons = struct['review_flags']
        else:
            svo, invalid, review, gen_reasons = generate_svo(
                np1, np2, vp, vm, vmp, primary,
                v_core, trailing_asp,
                struct['prefix'], struct['suffix'],
                struct['internal_1'],
                struct['internal_2'],
                struct['internal_3'],
            )
            all_review_reasons += gen_reasons

        # 去重保持顺序
        seen = set()
        deduped = []
        for r_ in all_review_reasons:
            if r_ not in seen:
                seen.add(r_)
                deduped.append(r_)

        final_review = 1 if (review or vcore_flags or
                             struct['review_flags']) and not invalid else 0

        # 统计
        stats['total'] += 1
        stats[primary] = stats.get(primary, 0) + 1
        if invalid:
            stats['invalid'] += 1
        elif final_review:
            stats['review'] += 1
        else:
            stats['clean'] += 1

        # 过滤掉旧字段名，避免DictWriter报错
        row_clean = {k: v for k, v in r.items() if k not in OLD_FIELDS}
        output_rows.append({
            **row_clean,
            'V_core':            v_core,
            'internal_1':        struct['internal_1'],
            'internal_2':        struct['internal_2'],
            'internal_3':        struct['internal_3'],
            'VP_type':           '|'.join(types),
            'VP_type_primary':   primary,
            'svo':               svo,
            'main_verb':         v_core,
            'svo_invalid':       invalid,
            'svo_review_flag':   final_review,
            'svo_review_reason': '|'.join(deduped),
        })

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    # ── 报告 ──────────────────────────────────────────────────────
    n = stats['total']
    print(f"\n{'─'*55}")
    print(f"处理完成：{n} 条")
    print(f"{'─'*55}")
    print(f"VP_type_primary 分布：")
    for t in PRIORITY:
        cnt = stats.get(t, 0)
        if cnt > 0:
            print(f"  {t:<25} {cnt:4d}  ({cnt/n*100:5.1f}%)")
    print(f"\nSVO生成结果：")
    print(f"  直接可用 (clean)   : {stats['clean']:4d}  ({stats['clean']/n*100:5.1f}%)")
    print(f"  需人工确认 (review): {stats['review']:4d}  ({stats['review']/n*100:5.1f}%)")
    print(f"  排除 (invalid)     : {stats['invalid']:4d}  ({stats['invalid']/n*100:5.1f}%)")
    print(f"\n输出文件：{output_path}")
    print(f"{'─'*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description='把字句 → SVO 语序转换（v3.1）'
    )
    parser.add_argument('--input',  required=True, help='输入CSV路径')
    parser.add_argument('--output', required=True, help='输出CSV路径')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在 → {args.input}")
        return

    process(args.input, args.output)


if __name__ == '__main__':
    main()

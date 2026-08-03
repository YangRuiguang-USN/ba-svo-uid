#!/usr/bin/env python3
"""
extract_svo.py
从CTS语料中提取含有指定动词的SVO句子

用法：
    python extract_svo.py --ba_csv path/to/ba_corpus.csv --cts_dir path/to/cts/ --output svo_results.csv

    可选参数：
    --keep_only     只输出 decision=keep 的行（默认同时输出 review）
    --batch_size    HanLP批处理大小（默认64）
    --max_len       句子最大字符数（默认40）
    --min_len       句子最小字符数（默认5）
"""

import re
import argparse
import os
import sys
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    sys.exit("请安装 pandas：pip install pandas")

try:
    import hanlp
except ImportError:
    sys.exit("请安装 hanlp：pip install hanlp")


# ══════════════════════════════════════════════════════════════
# CTS 文本处理
# ══════════════════════════════════════════════════════════════

NOISE_SPEAKERS = {"G00000000"}
CTS_NOISE_RE   = re.compile(r'\[\*\]|\[/\]|\[//\]|\(\(.*?\)\)|<[^>]+>')
CTS_OVERLAP_RE = re.compile(r'\[[\+\-]\]')
SPLIT_RE       = re.compile(r'[，。！？；]')
STRIP_RE       = re.compile(r'\[.*?\]|[""\'\'【】《》（）…—～\s]')


def parse_cts_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 4:
        return None
    speaker_id = parts[1].strip()
    if speaker_id in NOISE_SPEAKERS:
        return None
    return speaker_id, parts[3].strip()


def clean_clause(text):
    text = CTS_NOISE_RE.sub('', text)
    text = CTS_OVERLAP_RE.sub('', text)
    text = STRIP_RE.sub('', text)
    return text.strip()


def load_cts_corpus(path, min_len, max_len):
    """
    读取CTS文件。
    返回 (sentences, cts_lines_by_file)：
      - sentences: 小句列表，含 sentence_idx（在 cts_lines 中的行索引）
      - cts_lines_by_file: {filename: [(speaker_id, raw_text), ...]}
        用于背景句查找（行级别 = 说话人轮次）
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(r, f)
                 for r, _, fs in os.walk(path)
                 for f in fs if f.endswith(('.txt', '.cts'))]
    print(f"找到 {len(files)} 个CTS文件")

    sentences = []
    cts_lines_by_file = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        file_lines = []
        with open(fpath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                result = parse_cts_line(line)
                if result is None:
                    continue
                speaker_id, raw = result
                line_idx = len(file_lines)
                file_lines.append((speaker_id, raw))

                for clause in SPLIT_RE.split(raw):
                    clause = clause.strip()
                    cleaned = clean_clause(clause)
                    if min_len <= len(cleaned) <= max_len:
                        sentences.append({
                            'source_file':  fname,
                            'speaker_id':   speaker_id,
                            'raw_clause':   clause,
                            'text':         cleaned,
                            'line_idx':     line_idx,  # 在 file_lines 中的位置
                        })
        cts_lines_by_file[fname] = file_lines

    print(f"共提取 {len(sentences)} 个小句")
    return sentences, cts_lines_by_file


def get_context(source_file, line_idx, cts_lines_by_file, before=2, after=1):
    """
    返回目标行前后的CTS行文本，分为三个字段：
      - context_before: 前2行，用 | 分隔
      - sentence_original: 目标行原文
      - context_after: 后1行
    """
    lines = cts_lines_by_file.get(source_file, [])

    before_parts = []
    for i in range(max(0, line_idx - before), line_idx):
        _, raw = lines[i]
        before_parts.append(raw)

    sentence_original = lines[line_idx][1] if line_idx < len(lines) else ''

    after_parts = []
    for i in range(line_idx + 1, min(len(lines), line_idx + after + 1)):
        _, raw = lines[i]
        after_parts.append(raw)

    return {
        'context_before':    ' | '.join(before_parts),
        'sentence_original': sentence_original,
        'context_after':     ' | '.join(after_parts),
    }


# ══════════════════════════════════════════════════════════════
# 动词加载
# ══════════════════════════════════════════════════════════════

def load_target_verbs(ba_csv_path):
    df = pd.read_csv(ba_csv_path)
    if 'V_core' not in df.columns:
        sys.exit(f"CSV中找不到V_core列，现有列：{list(df.columns)}")
    verbs = {v.strip() for v in df['V_core'].dropna()
             if isinstance(v, str) and 1 <= len(v.strip()) <= 4}
    print(f"共加载 {len(verbs)} 个唯一动词")
    return verbs


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 句首出现时整句排除（真正的从属连词）
SENT_SUBORD_TRIGGERS = {
    '如果', '假如', '要是', '若', '倘若', '假设', '一旦',
    '虽然', '尽管', '即使', '就算', '哪怕',
    '因为', '由于', '为了', '为使',
}

# 句首出现时降为 review（话语连接词，不排除）
DISCOURSE_MARKERS = {'但是', '然而', '不过'}

# 使役/感知/言说动词：其宾语位置的动词排除
EMBEDDING_VERBS = {
    '让', '叫', '使', '令', '请', '派', '命令', '要求',
    '看见', '听见', '发现', '感到', '觉得', '知道', '听说',
    '说', '表示', '认为', '感觉', '证明', '相信', '希望', '担心', '怀疑',
}

LIGHT_OR_SPECIAL_VERBS = {
    '让', '叫', '使', '令', '请',
    '给', '去', '来', '过', '说',
}

SUBJ_LABELS         = {'nsubj', 'SBV'}          # 主动主语
PASSIVE_SUBJ_LABELS = {'nsubjpass'}              # 被动主语：单独处理
OBJ_LABELS          = {'dobj', 'obj', 'VOB', 'OBJ'}
ROOT_LABELS         = {'root', 'HED'}
COORD_LABELS        = {'conj', 'COO'}


# ══════════════════════════════════════════════════════════════
# 依存分析辅助
# ══════════════════════════════════════════════════════════════

def normalize_deps(deps):
    """HanLP 1-based → 0-based，root head 设为 None"""
    return [(h - 1 if h > 0 else None, l) for h, l in deps]


def build_children(deps_0):
    children = defaultdict(list)
    for child, (head, label) in enumerate(deps_0):
        if head is not None:
            children[head].append((child, label))
    return children


# ══════════════════════════════════════════════════════════════
# 核心：对单个动词分类
# ══════════════════════════════════════════════════════════════

def classify_verb(verb_idx, tokens, deps_0, children, postags):
    """
    返回 (decision, reason, subject, object, dep_label, head_token)
    decision: 'keep' | 'review' | 'exclude'

    排除策略（按优先级）：
      1. 词性不是动词
      2. 句首从属连词（字符串前缀匹配）
      3. dep标签明确是从句（rcmod/ccomp/xcomp）
      4. head 是嵌入动词（使役/感知/言说）
      5. 被动主语（nsubjpass）
      6. 无宾语 → exclude；无主语 → 保留但标记 zero_subject
      7. dep=root/HED → keep
      8. dep=conj/COO 且 head 是 root → keep
      9. 句首话语连接词（但是/然而/不过）→ review
      10. 其他模糊情况 → review
    """
    head, dep_label = deps_0[verb_idx]
    pos = postags[verb_idx] if verb_idx < len(postags) else ''
    head_token = 'ROOT' if head is None else tokens[head]

    def ret(decision, reason, subj='', obj=''):
        return decision, reason, subj, obj, dep_label, head_token

    # 1. 词性
    if pos and not pos.startswith('V'):
        return ret('exclude', 'pos_not_verb')

    # 2. 句首连词：基于字符串前缀，不依赖分词粒度
    sent_text = ''.join(tokens)
    if any(sent_text.startswith(t) for t in SENT_SUBORD_TRIGGERS):
        return ret('exclude', 'subord_trigger')

    # 3. 明确从句 dep 标签
    if dep_label in ('rcmod', 'relcl', 'RC'):
        return ret('exclude', 'rcmod')
    if dep_label in ('ccomp', 'xcomp', 'COMP'):
        return ret('exclude', 'ccomp')

    # 4. head 是嵌入动词
    if head is not None and head < len(tokens) and tokens[head] in EMBEDDING_VERBS:
        return ret('exclude', 'embedded_under_verb')

    # 5. 提取主语/宾语
    subj_token       = ''
    obj_token        = ''
    has_topic        = False
    has_passive_subj = False

    for child_idx, label in children[verb_idx]:
        if label in SUBJ_LABELS:
            subj_token = tokens[child_idx]
        elif label in PASSIVE_SUBJ_LABELS:
            has_passive_subj = True
        elif label in OBJ_LABELS:
            obj_token = tokens[child_idx]
        elif label in ('top', 'TOP'):
            has_topic = True

    # 6. 被动主语排除
    if has_passive_subj:
        return ret('exclude', 'passive_subject', subj_token, obj_token)

    if not obj_token:
        return ret('exclude', 'no_object', subj_token)
    if not subj_token:
        # 零主语：与把字句保持对称，保留但标记供人工核查
        # 后续主句判断正常进行，reason里会附加+zero_subject
        pass

    # 7-9. 主句判断
    if dep_label in ROOT_LABELS:
        decision = 'keep'
        reason   = 'root_svo'
    elif dep_label in COORD_LABELS:
        if head is not None and deps_0[head][1] in ROOT_LABELS:
            decision = 'keep'
            reason   = 'coord_root_svo'
        else:
            decision = 'review'
            reason   = 'coord_nonroot'
    else:
        decision = 'review'
        reason   = f'nonroot_dep={dep_label}'

    # 零主语：降为 review 并标记
    if not subj_token:
        decision = 'review'
        reason  += '+zero_subject'

    # 轻动词/多功能动词：降为 review 避免进入高置信度 keep
    if tokens[verb_idx] in LIGHT_OR_SPECIAL_VERBS:
        decision = 'review'
        reason  += '+light_or_special_verb'

    # 10. 话语连接词（但是/然而/不过）降为 review
    if any(sent_text.startswith(m) for m in DISCOURSE_MARKERS):
        decision = 'review'
        reason  += '+discourse_marker'

    if has_topic:
        decision = 'review'
        reason  += '+topic'

    return ret(decision, reason, subj_token, obj_token)


# ══════════════════════════════════════════════════════════════
# 句级分析
# ══════════════════════════════════════════════════════════════

def analyze_sentence(dep_result, target_verbs):
    """
    返回该句中所有命中目标动词的分类结果（keep/review）。
    按位置遍历，不做动词文本去重（同一句同一动词出现两次各自分析）。
    """
    tokens   = dep_result['tok/fine']
    raw_deps = dep_result['dep']
    postags  = dep_result.get('pos/ctb', [''] * len(tokens))

    # 含独立"把"/"被" token：排除把字句和被动句
    # token级别精确匹配，不误杀"把握""被子"等多字词
    if '把' in tokens or '被' in tokens:
        return []

    deps_0   = normalize_deps(raw_deps)
    children = build_children(deps_0)

    tokens_str = ' '.join(tokens)
    pos_str    = ' '.join(postags)
    deps_str   = ' '.join(f'{h},{l}' for h, l in raw_deps)

    rows = []
    for i, token in enumerate(tokens):
        if token not in target_verbs:
            continue

        decision, reason, subj, obj, dep_lbl, head_tok = classify_verb(
            i, tokens, deps_0, children, postags)

        if decision == 'exclude':
            continue

        rows.append({
            'decision':     decision,
            'reason':       reason,
            'matched_verb': token,
            'verb_index':   i,
            'dep_label':    dep_lbl,
            'head_token':   head_tok,
            'subject':      subj,
            'object':       obj,
            'tokens':       tokens_str,
            'pos_tags':     pos_str,
            'deps_raw':     deps_str,
        })
    return rows


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ba_csv',     required=True)
    parser.add_argument('--cts_dir',    required=True)
    parser.add_argument('--output',     default='svo_results.csv')
    parser.add_argument('--keep_only',  action='store_true',
                        help='只输出 decision=keep 的行')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len',    type=int, default=40)
    parser.add_argument('--min_len',    type=int, default=5)
    args = parser.parse_args()

    target_verbs = load_target_verbs(args.ba_csv)
    sentences, cts_lines_by_file = load_cts_corpus(
        args.cts_dir, args.min_len, args.max_len)

    print("预筛选含目标动词的句子...")
    candidates = [s for s in sentences
                  if any(v in s['text'] for v in target_verbs)]
    print(f"候选句：{len(candidates)} 条，开始依存分析...")

    print("加载HanLP模型...")
    HanLP = hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

    all_rows = []
    texts = [s['text'] for s in candidates]

    for batch_start in range(0, len(texts), args.batch_size):
        if batch_start % 1000 == 0:
            print(f"  进度：{batch_start}/{len(texts)}")

        batch_texts = texts[batch_start:batch_start + args.batch_size]
        batch_meta  = candidates[batch_start:batch_start + args.batch_size]

        try:
            doc = HanLP(batch_texts, tasks=['tok/fine', 'pos/ctb', 'dep'])
        except Exception as e:
            print(f"  HanLP出错（batch {batch_start}）：{e}，跳过")
            continue

        # Document是字典：{'tok/fine': [[句1tokens], [句2tokens], ...], ...}
        # 第一个batch：打印Document结构，确认key名称
        if batch_start == 0:
            print(f"  HanLP Document keys: {list(doc.keys())}")

        toks_list = doc['tok/fine']
        pos_list  = doc.get('pos/ctb')
        if pos_list is None:
            pos_list = [[''] * len(toks) for toks in toks_list]
        deps_list = doc['dep']

        for i in range(len(toks_list)):
            dep_result = {
                'tok/fine': toks_list[i],
                'pos/ctb':  pos_list[i],
                'dep':      deps_list[i],
            }
            for m in analyze_sentence(dep_result, target_verbs):
                meta = batch_meta[i]
                ctx  = get_context(
                    meta['source_file'],
                    meta['line_idx'],
                    cts_lines_by_file,
                )
                all_rows.append({
                    'source_file':       meta['source_file'],
                    'speaker_id':        meta['speaker_id'],
                    'line_idx':          meta['line_idx'],
                    'context_before':    ctx['context_before'],
                    'sentence_original': ctx['sentence_original'],
                    'context_after':     ctx['context_after'],
                    'raw_clause':        meta['raw_clause'],
                    'svo_text':          meta['text'],
                    **m,
                })

    if not all_rows:
        print("未找到任何匹配，请检查语料路径和动词列表")
        return

    df = pd.DataFrame(all_rows)

    keep_n   = (df['decision'] == 'keep').sum()
    review_n = (df['decision'] == 'review').sum()
    print(f"\n结果统计：keep={keep_n}  review={review_n}")
    print("\n各动词 keep 数量（前20）：")
    print(df[df['decision'] == 'keep']['matched_verb']
            .value_counts().head(20).to_string())

    if args.keep_only:
        df = df[df['decision'] == 'keep']

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n已保存至：{args.output}  ({len(df)} 行)")


if __name__ == '__main__':
    main()

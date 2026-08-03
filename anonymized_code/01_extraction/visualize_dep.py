# -*- coding: utf-8 -*-
"""
visualize_dep.py
────────────────
Visualise l'analyse en dépendances HanLP pour une phrase donnée.
Affiche un tableau lisible + l'arbre ASCII dans le terminal.

Usage :
  python visualize_dep.py
  → saisir la phrase quand demandé

Ou directement :
  python visualize_dep.py "他把那个苹果吃了"
"""

import sys
import re


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT HANLP
# ══════════════════════════════════════════════════════════════════════════════

def load_hanlp():
    try:
        import transformers
        if not hasattr(transformers.BertTokenizer, 'encode_plus'):
            transformers.BertTokenizer.encode_plus = (
                transformers.BertTokenizer._encode_plus
            )
        import hanlp
        print("Chargement HanLP...")
        pipeline = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )
        print("Modèle chargé.\n")
        return pipeline
    except Exception as e:
        print(f"Erreur HanLP : {e}")
        sys.exit(1)


def clean(sentence):
    s = re.sub(r'\[.*?\]', '', sentence)
    s = re.sub(r'\(.*?\)', '', s)
    return re.sub(r'\s+', '', s).strip()


# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

# Traduction des étiquettes dep les plus courantes vers le chinois
DEP_LABELS_ZH = {
    'root':     'ROOT 根节点',
    'nsubj':    'nsubj 主语',
    'nsubjpass':'nsubjpass 被动主语',
    'dobj':     'dobj 直接宾语',
    'iobj':     'iobj 间接宾语',
    'ba':       'ba 把字标记 ★',
    'range':    'range 把字宾语 ★',
    'prep':     'prep 介词',
    'pobj':     'pobj 介词宾语',
    'attr':     'attr 谓语属性',
    'csubj':    'csubj 从句主语',
    'ccomp':    'ccomp 补语从句',
    'xcomp':    'xcomp 开放补语',
    'rcomp':    'rcomp 结果补语 ★',
    'asp':      'asp 体标记 ★',
    'mmod':     'mmod 情态修饰',
    'tmod':     'tmod 时间修饰',
    'loc':      'loc 处所修饰',
    'rcmod':    'rcmod 关系从句',
    'assmod':   'assmod 关联修饰',
    'advmod':   'advmod 状语修饰',
    'neg':      'neg 否定词',
    'amod':     'amod 形容词修饰',
    'nummod':   'nummod 数量修饰',
    'clf':      'clf 量词',
    'det':      'det 限定词',
    'dep':      'dep 未分类依存',
    'punct':    'punct 标点',
    'conj':     'conj 并列',
    'cc':       'cc 连词',
    'mark':     'mark 标记词',
    'cop':      'cop 系动词',
    'aux':      'aux 助动词',
    'prt':      'prt 语气词',
    'top':      'top 话题',
    'etc':      'etc 等等',
    'eff':      'eff 效果补语',
    'comod':    'comod 并列动词',
    'vmod':     'vmod 动词修饰',
    'appos':    'appos 同位语',
    'nn':       'nn 名词修饰',
    'pass':     'pass 被动标记',
}


def print_dep_table(tokens, deps, pos=None):
    """
    Affiche un tableau des dépendances :
    索引 | 词语 | 词性 | 中心词 | 中心词语 | 依存关系
    """
    # En-têtes
    header = f"{'索引':>4}  {'词语':<8}  {'词性':<6}  {'→中心词':>6}  {'中心词语':<8}  依存关系"
    print(header)
    print("─" * 70)

    for i, (tok, (head_1based, label)) in enumerate(zip(tokens, deps)):
        head_0 = head_1based - 1  # convertir en 0-based
        head_tok = tokens[head_0] if head_0 >= 0 else 'ROOT'
        pos_tag  = pos[i] if pos else ''
        label_zh = DEP_LABELS_ZH.get(label, label)

        # Mettre en valeur les étiquettes liées à 把
        marker = ' ◀' if label in ('ba', 'range', 'rcomp', 'asp') else ''

        print(
            f"  {i:>2}  {tok:<8}  {pos_tag:<6}  "
            f"  {head_0:>3} {head_tok:<8}  "
            f"{label_zh}{marker}"
        )
    print()


def print_ascii_tree(tokens, deps):
    """
    Affiche un arbre ASCII simplifié centré sur chaque token.
    Format :
      ROOT
        └── 动词 (root)
              ├── 主语 (nsubj)
              ├── 把 (ba)
              │     └── 宾语 (range)
              └── 补语 (rcomp)
    """
    n = len(tokens)
    heads = [h - 1 for h, _ in deps]   # 0-based, -1 = root
    labels = [l for _, l in deps]

    # Trouver la racine
    roots = [i for i in range(n) if heads[i] == -1]
    if not roots:
        print("(arbre non disponible)\n")
        return

    # Construire les enfants
    children = {i: [] for i in range(n)}
    for i in range(n):
        if heads[i] >= 0:
            children[heads[i]].append(i)

    def render(node, prefix='', is_last=True):
        connector = '└── ' if is_last else '├── '
        label = labels[node]
        label_zh = DEP_LABELS_ZH.get(label, label)
        marker = ' ★' if label in ('ba', 'range', 'rcomp', 'asp') else ''
        print(f"{prefix}{connector}{tokens[node]}  [{label_zh}]{marker}")
        child_list = children[node]
        for j, child in enumerate(child_list):
            extension = '    ' if is_last else '│   '
            render(child, prefix + extension, j == len(child_list) - 1)

    print("ROOT")
    for k, root in enumerate(roots):
        render(root, '', k == len(roots) - 1)
    print()


def print_srl(srl, tokens):
    """Affiche le SRL de façon lisible."""
    if not srl:
        print("  (aucun résultat SRL)\n")
        return
    for idx, predicate_roles in enumerate(srl):
        print(f"  谓词框架 {idx + 1} :")
        for span_text, label, start, end in predicate_roles:
            label_map = {
                'PRED': 'PRED  谓词',
                'ARG0': 'ARG0  施事/主语',
                'ARG1': 'ARG1  受事/宾语 ★',
                'ARG2': 'ARG2  受益者',
                'ARGM-ADV': 'ARGM-ADV  状语',
                'ARGM-TMP': 'ARGM-TMP  时间',
                'ARGM-LOC': 'ARGM-LOC  处所',
                'ARGM-MNR': 'ARGM-MNR  方式',
                'ARGM-NEG': 'ARGM-NEG  否定',
            }
            lbl = label_map.get(label, label)
            print(f"    [{lbl}]  「{span_text}」  (位置 {start}–{end})")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def analyze(pipeline, sentence):
    cleaned = clean(sentence)
    if not cleaned:
        print("Phrase vide après nettoyage.")
        return

    print(f"\n{'═' * 70}")
    print(f"  原句  : {sentence}")
    if cleaned != sentence:
        print(f"  清理后 : {cleaned}")
    print(f"{'═' * 70}\n")

    result = pipeline(cleaned)

    tokens = result.get('tok/fine', [])
    deps   = result.get('dep', [])
    pos    = result.get('pos/ctb', result.get('pos/pku', None))
    srl    = result.get('srl', [])

    # ── 1. Tableau de dépendances ─────────────────────────────────────────────
    print("【依存关系表】")
    print_dep_table(tokens, deps, pos)

    # ── 2. Arbre ASCII ────────────────────────────────────────────────────────
    print("【依存树】")
    print_ascii_tree(tokens, deps)

    # ── 3. SRL ───────────────────────────────────────────────────────────────
    print("【语义角色标注 SRL】")
    print_srl(srl, tokens)

    # ── 4. Résumé pour 把字句 ─────────────────────────────────────────────────
    ba_found = False
    for i, (tok, (head_1based, label)) in enumerate(zip(tokens, deps)):
        if tok == '把' and label == 'ba':
            ba_found = True
            vp_head = head_1based - 1
            print("【把字句成分识别】")
            print(f"  ✓ 「把」确认为把字标记 (索引 {i})")
            print(f"  ✓ 核心动词 : 「{tokens[vp_head]}」 (索引 {vp_head})")

            # NP2 : dépendant direct de 把 à droite
            np2_candidates = [
                j for j in range(i + 1, len(tokens))
                if deps[j][0] - 1 == i
            ]
            if np2_candidates:
                print(f"  ✓ NP2候选词根 : 「{tokens[np2_candidates[0]]}」 (索引 {np2_candidates[0]}，依存标签: {deps[np2_candidates[0]][1]})")

            # NP1 : nsubj du vp_head
            nsubj = [
                j for j in range(len(tokens))
                if deps[j][0] - 1 == vp_head and deps[j][1] in ('nsubj', 'nsubjpass', 'top')
            ]
            if nsubj:
                print(f"  ✓ NP1候选词 : 「{tokens[nsubj[0]]}」 (索引 {nsubj[0]}，依存标签: {deps[nsubj[0]][1]})")

            # Compléments verbaux
            vcomplements = [
                j for j in range(vp_head + 1, len(tokens))
                if deps[j][0] - 1 == vp_head and deps[j][1] in ('rcomp', 'asp', 'ccomp', 'eff', 'range')
            ]
            if vcomplements:
                vc_str = '  '.join(f"「{tokens[j]}」({deps[j][1]})" for j in vcomplements)
                print(f"  ✓ 动词补语 : {vc_str}")
            print()
            break

    if not ba_found:
        print("【把字句成分识别】")
        print("  ✗ HanLP 未识别到 dep='ba' 标签")
        # Chercher quand même 把 dans les tokens
        ba_positions = [i for i, t in enumerate(tokens) if t == '把']
        if ba_positions:
            for bp in ba_positions:
                print(f"  ℹ  「把」存在于索引 {bp}，依存标签为 : {deps[bp][1]}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pipeline = load_hanlp()

    if len(sys.argv) > 1:
        # Phrase passée en argument
        sentence = ' '.join(sys.argv[1:])
        analyze(pipeline, sentence)
    else:
        # Mode interactif
        print("输入把字句进行依存分析（输入 q 退出）\n")
        while True:
            try:
                sentence = input("请输入句子 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break
            if not sentence:
                continue
            if sentence.lower() in ('q', 'quit', 'exit', '退出'):
                print("退出。")
                break
            analyze(pipeline, sentence)


if __name__ == '__main__':
    main()

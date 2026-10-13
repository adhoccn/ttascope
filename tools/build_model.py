"""用实验语料训练并导出随产品发布的模型文件。

训练协议与论文一致：ChatGPT-only，前 800 对（人类侧取自 HC3-Chinese 的
同一问题的人类回答，AI 侧取自采集语料），5 折之外不引入任何目标 LLM 文本。

    python tools/build_model.py --data ../exp/data --out models

产出 models/ttascope_15d.joblib 与 models/ttascope_char_2_4.joblib。
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.model import TTADetector          # noqa: E402

N_TRAIN = 800


def load_pairs(data_dir, model='chatgpt'):
    """返回 (human_texts, ai_texts)，按问题把 HC3 人类回答与目标 LLM 回答配对。"""
    hc3 = {}
    with io.open(data_dir / 'all.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            q = (r.get('question') or '').strip()
            hs = r.get('human_answers') or []
            if q and hs:
                hc3[q] = hs[0]

    fp = data_dir / 'collected' / ('%s_seed42_n1000.jsonl' % model)
    hs, ais = [], []
    with io.open(fp, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            q = (r.get('question') or '').strip()
            if q in hc3:
                hs.append(hc3[q])
                ais.append(r['answer'])
    return hs, ais


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT.parent / 'exp' / 'data'))
    ap.add_argument('--out', default=str(ROOT / 'models'))
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--n-train', type=int, default=N_TRAIN)
    args = ap.parse_args()

    data_dir = Path(args.data)
    hs, ais = load_pairs(data_dir)
    if len(hs) < args.n_train:
        raise SystemExit('配对不足：%d < %d' % (len(hs), args.n_train))
    texts = list(hs[:args.n_train]) + list(ais[:args.n_train])
    y = np.array([0] * args.n_train + [1] * args.n_train)
    print('训练对 %d（人类 %d / AI %d）' % (args.n_train, args.n_train, args.n_train))

    os.makedirs(args.out, exist_ok=True)
    for name in args.feats.split(','):
        det = TTADetector(feat=name).fit(texts, y)
        p = os.path.join(args.out, 'ttascope_%s.joblib' % name)
        det.save(p)
        print('  %-12s dim=%-4d train_acc=%.4f -> %s (%.1f MB)'
              % (name, det.dim, det.train_acc, p,
                 os.path.getsize(p) / 1e6))


if __name__ == '__main__':
    main()

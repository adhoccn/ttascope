"""在目标 LLM 语料上复核产品实现与论文表格是否一致。

    python tools/validate.py --targets deepseek,glm --feats 15D,char_2_4

口径与论文表 11 一致：训练 ChatGPT-only 前 800 对（随产品发布的模型），
测试目标 LLM 整批（1000 对），报告 macro-F1。auto 一列为无标签 GMM 准则
自动选参，对应论文的 UBC；oracle 一列为用真值标签在候选族里选最优，
是 auto 的上界参考。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.model import TTADetector             # noqa: E402
from tools.build_model import load_pairs           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT.parent / 'exp' / 'data'))
    ap.add_argument('--models', default=str(ROOT / 'models'))
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--feats', default='15D,char_2_4')
    args = ap.parse_args()

    data = Path(args.data)
    print('%-12s %-9s %7s %7s %7s %7s %7s' %
          ('特征集', '目标', 'none', 'bn', 'em', 'auto', 'oracle'))
    for feat in args.feats.split(','):
        path = os.path.join(args.models, 'ttascope_%s.joblib' % feat)
        if not os.path.exists(path):
            print('  (跳过 %s：无模型文件)' % feat)
            continue
        det = TTADetector.load(path)
        for tgt in args.targets.split(','):
            hs, ais = load_pairs(data, tgt)
            texts = list(hs) + list(ais)
            y = np.array([0] * len(hs) + [1] * len(ais))
            res = det.detect(texts, with_items=False)
            B = det.transform(texts)
            preds = {}
            for name in res['candidates']:
                Bc = det._candidates(B)[name]
                preds[name] = (det.clf.predict(Bc)).astype(int)
            best_oracle = max(preds, key=lambda k: f1_score(y, preds[k], average='macro'))
            print('%-12s %-9s %7.4f %7.4f %7.4f %7.4f %7.4f' % (
                feat, tgt,
                f1_score(y, preds['none'], average='macro'),
                f1_score(y, preds['bn'], average='macro'),
                f1_score(y, preds['em_l1.0'], average='macro'),
                f1_score(y, preds[res['chosen']], average='macro'),
                f1_score(y, preds[best_oracle], average='macro')))


if __name__ == '__main__':
    main()

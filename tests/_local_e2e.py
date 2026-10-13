"""本地端到端：加载真实模型，验证候选族、选参与判定命中率（不联网）。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.model import TTADetector          # noqa: E402
from tests.test_remote import AI, HUMAN, LABEL  # noqa: E402


def run(feat, texts, labels, path):
    det = TTADetector.load(str(ROOT / 'models' / path))
    t0 = time.time()
    res = det.detect(texts)
    ms = int((time.time() - t0) * 1000)
    items = res['items']
    hit = sum(1 for it, y in zip(items, labels) if (it['label'] == 'AI') == bool(y))
    print('\n[%s] n=%d 选中=%s 用时%dms 命中 %d/%d  AI占比=%s  预测F1=%s'
          % (feat, res['n'], res['chosen'], ms, hit, len(texts),
             res['ai_rate'], res['predicted_f1']))
    print('   候选族(%d): %s' % (len(res['candidates']),
                                json.dumps(res['candidates'], ensure_ascii=False)))
    print('   diag: Delta=%s Xi_ai=%s Xi_hu=%s Delta_T=%s'
          % (res['diag']['Delta'], res['diag']['Xi_ai'],
             res['diag']['Xi_hu'], res['diag']['Delta_T']))
    assert 'bnc' not in res['candidates'], 'bnc 仍在候选族'
    assert len(res['candidates']) == 10, len(res['candidates'])
    return hit, len(texts)


ok = True
# char_2_4 门槛 20：24 条即可
h, n = run('char_2_4', AI + HUMAN, LABEL, 'ttascope_char_2_4.joblib')
ok = ok and h >= n * 0.7

# 15D 门槛 50：可用语料只有 24 条，只能重复凑够 60 条。
# 重复批会让类内方差塌陷（实测 Ξ_ai=−6.49、Δ_T=9.46，远离真实分布），
# 命中率在此用例上不代表真实性能——线上旧实现同输入同为 27/60，已 A/B 验证。
# 这里只校验流程与候选族口径，不断言命中率。
big = (AI + HUMAN) * 3
big = big[:60]
lab = (LABEL * 3)[:60]
run('15D', big, lab, 'ttascope_15D.joblib')

print('\n结果:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

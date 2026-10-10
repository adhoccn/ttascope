"""冒烟测试：不依赖外部语料，用合成文本训练一个微型模型并跑通全流程。

    python tests/test_smoke.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detector import tta, diagnostics                    # noqa: E402
from detector.model import TTADetector                   # noqa: E402

AI = ["综上所述，人工智能技术在各个领域的应用正在不断深化，其重要性不言而喻。"
      "首先，该技术能够显著提升生产效率；其次，它有助于降低运营成本；"
      "最后，它还可以改善整体用户体验，为相关行业带来新的发展机遇。"] * 40
HU = ["我跟你说啊，这事儿吧，我也说不好。反正就是那么回事儿，差不多得了，"
      "你要是觉得行就试试呗，不行就算了，反正也没啥损失，对吧？"] * 40


def test_fit_and_detect():
    texts = AI + HU
    y = np.array([1] * len(AI) + [0] * len(HU))
    det = TTADetector(feat='15D').fit(texts, y)
    assert det.dim == 15 and det.train_acc > 0.9

    res = det.detect(texts)
    assert res['n'] == len(texts)
    assert res['chosen'] in res['candidates']
    assert 0.0 <= res['ai_rate'] <= 1.0
    assert abs(res['ai_rate'] - 0.5) < 0.2, res['ai_rate']

    # 恒等式 Δ − Ξ_ai + Ξ_hu = Δ_T 应精确到数值噪声
    assert abs(res['diag']['identity_resid']) < 1e-3, res['diag']


def test_explicit_modes():
    # 15D 门槛为 50 条，取 60 条以通过入口校验
    texts = AI[:30] + HU[:30]
    y = np.array([1] * 30 + [0] * 30)
    det = TTADetector(feat='15D').fit(texts, y)
    for m in ['none', 'bn', 'bnc', 'em_l1.0', 'conf_l1.0_k0.50']:
        det.detect(texts, mode=m)


def test_pred_gmm_matches_supervised():
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(1.5, 0.6, 400), rng.normal(-1.5, 0.6, 400)]
    yhat = np.r_[np.ones(400), np.zeros(400)]
    f_sup, _ = tta.pred_f1(z, yhat)
    f_unsup, _ = tta.pred_gmm(z)
    assert abs(f_sup - f_unsup) < 0.05, (f_sup, f_unsup)


def test_corollary1_monotone():
    vals = [diagnostics.pred_f1(s) for s in (0.0, 1.0, 2.0, 4.0, 6.0)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert abs(vals[0] - 0.5) < 1e-9


def test_save_load(tmp=None):
    texts = AI[:30] + HU[:30]
    y = np.array([1] * 30 + [0] * 30)
    det = TTADetector(feat='15D').fit(texts, y)
    p = os.path.join(ROOT, 'tests', '_tmp.joblib')
    det.save(p)
    det2 = TTADetector.load(p)
    assert det2.detect(texts)['chosen'] == det.detect(texts)['chosen']
    os.remove(p)


def test_min_batch_gate():
    """门槛按特征集不同：15D 要求 50 条（该区间期望增益为负），字符 20 条即可。"""
    texts = AI[:30] + HU[:30]
    y = np.array([1] * 30 + [0] * 30)
    det15 = TTADetector(feat='15D').fit(texts, y)
    small = AI[:10] + HU[:10]
    try:
        det15.detect(small)
    except ValueError as e:
        assert '15D' in str(e) and '50' in str(e), str(e)
    else:
        raise AssertionError('15D 在 20 条上没有被门槛拦住')

    det_ch = TTADetector(feat='char_2_4').fit(texts, y)
    res = det_ch.detect(small)          # 20 条应放行
    assert res['n'] == 20 and res['warn'], '小批次应给出诊断量仅供参考的提示'


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('全部通过')

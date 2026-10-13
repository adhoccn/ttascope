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

    # 恒等式 Δ − Ξ_ai + Ξ_hu = Δ_T 应精确到数值噪声。
    # 本用例的两类文本是同一句话重复 40 遍，类内方差≈0 使 Δ 放大到 1e17，
    # 此时只能比相对误差（实测残差/Δ ≈ 3e-16），绝对阈值必然误报。
    d = res['diag']
    scale = max(abs(d['Delta']), abs(d['Delta_T']), 1.0)
    assert abs(d['identity_resid']) / scale < 1e-9, d


def test_explicit_modes():
    # 15D 门槛为 50 条，取 60 条以通过入口校验
    texts = AI[:30] + HU[:30]
    y = np.array([1] * 30 + [0] * 30)
    det = TTADetector(feat='15D').fit(texts, y)
    names = sorted(det._candidates(det.transform(texts)))
    # 候选族必须是论文的 10 个统一变体，且不含整批对齐 bnc
    assert len(names) == 10, names
    assert 'bnc' not in names, names
    for m in ['none', 'bn', 'em_l1.0', 'conf_l1.0_k0.50']:
        det.detect(texts, mode=m)
    # bnc 不再是合法变体
    try:
        det.detect(texts, mode='bnc')
    except KeyError:
        pass
    else:
        raise AssertionError('bnc 应已从候选族移除')


def test_pred_gmm_matches_supervised():
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(1.5, 0.6, 400), rng.normal(-1.5, 0.6, 400)]
    yhat = np.r_[np.ones(400), np.zeros(400)]
    f_sup, _ = tta.pred_f1(z, yhat)
    f_unsup, _ = tta.pred_gmm(z)
    assert abs(f_sup - f_unsup) < 0.05, (f_sup, f_unsup)


def test_prior_reject_rule():
    """前置拒绝规则：r_AI（未校正批次判为 AI 的比例）超过 0.64 即拒绝校正。

    判据只依赖未校正后验、不使用目标域标签；「是否真的退回 none」由调用方
    决定并写入 applied，显式指定变体时应保留用户意图。
    """
    from detector.diagnostics import REJECT_R_AI, prior_reject
    assert REJECT_R_AI == 0.64

    # 整批判为 AI：r_AI = 1.0，应触发
    g = prior_reject(np.r_[np.ones(80) * 0.9])
    assert g['r_ai'] == 1.0 and g['rejected'] and not g['applied']

    # 整批判为人类：r_AI = 0.0，不应触发
    g = prior_reject(np.r_[np.ones(80) * 0.1])
    assert g['r_ai'] == 0.0 and not g['rejected']

    # 恰好 60% 判为 AI：低于阈值，仍放行
    g = prior_reject(np.r_[np.ones(60) * 0.8, np.ones(40) * 0.2])
    assert g['r_ai'] == 0.6 and not g['rejected']

    # 70% 判为 AI：超过阈值，应拒绝
    g = prior_reject(np.r_[np.ones(70) * 0.8, np.ones(30) * 0.2])
    assert g['r_ai'] == 0.7 and g['rejected']


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

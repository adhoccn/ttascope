"""检测器：源域训练 + 目标批无标签自适应 + 漂移诊断。

用法：

    det = TTADetector(feat='15D').fit(train_texts, train_labels)
    res = det.detect(batch_texts)          # 一批未知来源的文本
    res['items']                           # 每条文本的 P(AI) 与判定
    res['diag']                            # Ξ_ai / Ξ_hu / Δ_T / 预测 F1 / 建议
    det.save('models/ttascope_15d.joblib')

设计要点（与论文一致）：
  * 规范管线：SimpleImputer(median) + StandardScaler 只在训练集拟合，
    然后用于测试批。缺了这一步，15 维特征因量纲差异会让 Δ 与 Ξ 失真。
  * 单步优先：类条件对齐迭代多次会退化，候选族里同时放了单步与三步，
    由无标签准则挑选。
  * 校正顺序：先 bnc 锁住批中心，再做类条件位移；每步扣掉位移的批均值。
"""
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from . import tta, diagnostics
from .features import build

try:
    import joblib
except ImportError:                                     # pragma: no cover
    joblib = None


# 批次规模门槛，按特征集区分。依据：小批次重采样实验（n=10/20/30/40）显示
# 15 维手工特征的类条件对齐在该区间增益转负（−0.12 ~ −0.03），且 28%~68% 的
# 折出现"校正反而更差"；字符 2-4gram 同时期仍稳定为正（+0.12 ~ +0.19）。
# 原因是两类方法的类间距 Δ 不同：15 维 Δ≈2.3，n≤40 时均值估计噪声已与
# 类间距同量级；字符 n-gram Δ≈5.0，同样噪声相对无害。
MIN_N = {'15D': 50, 'char_2_4': 20, 'roberta_frozen': 50}
REC_N = {'15D': 100, 'char_2_4': 50, 'roberta_frozen': 100}
DEFAULT_MIN_N = 50
DEFAULT_REC_N = 100


def min_n(feat):
    """该特征集允许检测的最小批规模，低于此值校正在期望意义上是有害的。"""
    return int(MIN_N.get(feat, DEFAULT_MIN_N))


def rec_n(feat):
    """诊断量（Ξ、Δ_T）稳定所需的批规模。"""
    return int(REC_N.get(feat, DEFAULT_REC_N))


class TTADetector:
    def __init__(self, feat='15D', iters=3, lams=(0.5, 1.0), keeps=(0.3, 0.5, 0.7),
                 seed=42, max_iter=2000):
        self.feat_name = feat
        self.iters = iters
        self.lams = tuple(lams)
        self.keeps = tuple(keeps)
        self.seed = seed
        self.max_iter = max_iter

    # -- 训练 ---------------------------------------------------------------
    def fit(self, texts, labels):
        texts = list(texts)
        y = np.asarray(labels, dtype=int)
        self.feat = build(self.feat_name).fit(texts)
        A0 = np.asarray(self.feat.transform(texts), dtype=np.float64)
        A0 = np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0)

        self.imputer = SimpleImputer(strategy='median').fit(A0)
        self.scaler = StandardScaler().fit(self.imputer.transform(A0))
        A = self.scaler.transform(self.imputer.transform(A0))

        self.clf = LogisticRegression(max_iter=self.max_iter, C=1.0,
                                      random_state=self.seed).fit(A, y)
        self.X_src = A
        self.y_src = y
        self.mu_tr = A.mean(axis=0)
        self.sd_tr = A.std(axis=0)
        self.mu0_tr = A[y == 0].mean(axis=0)
        self.mu1_tr = A[y == 1].mean(axis=0)
        self.dim = int(A.shape[1])
        self.train_acc = float(self.clf.score(A, y))
        return self

    # -- 特征 ---------------------------------------------------------------
    def transform(self, texts):
        B0 = np.asarray(self.feat.transform(list(texts)), dtype=np.float64)
        B0 = np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(self.imputer.transform(B0))

    # -- 校正候选 -----------------------------------------------------------
    def _candidates(self, B):
        """产出 {变体名: 校正后的特征矩阵}。

        候选族与论文表 10 的无标签选参实验一致：批对齐（bn / bnc）+
        类条件对齐（em，λ∈{0.5,1}）+ 高置信选通（conf，λ∈{0.5,1} ×
        keep∈{0.3,0.5,0.7}），共 11 个变体。选参准则在这些变体上取
        预测 macro-F1 最高者。
        """
        out = {'none': B,
               'bn': tta.bn(B, self.mu_tr, self.sd_tr),
               'bnc': tta.bnc(B, self.mu_tr)}
        for lam in self.lams:
            out['em_l%.1f' % lam] = tta.align(
                B, self.clf, self.mu0_tr, self.mu1_tr,
                iters=self.iters, lam=lam, mode='em')
            for k in self.keeps:
                out['conf_l%.1f_k%.2f' % (lam, k)] = tta.align(
                    B, self.clf, self.mu0_tr, self.mu1_tr,
                    iters=self.iters, lam=lam, mode='conf', keep=k)
        return out

    def _scores(self, Bc):
        return self.clf.decision_function(Bc)

    # -- 检测 ---------------------------------------------------------------
    def detect(self, texts, mode='auto', with_items=True, item_limit=200):
        """mode: auto = 无标签准则选最优变体；也可直接指定 none/bn/em/em1/conf_0.50..."""
        texts = [t for t in texts if str(t).strip()]
        need_min = min_n(self.feat_name)
        if len(texts) < 2:
            raise ValueError('批校正需要成批文本，至少 2 条（建议 ≥ 20 条）')
        if len(texts) < need_min:
            raise ValueError(
                '%s 特征要求不少于 %d 条：小批次实验显示该区间类条件对齐的期望增益为负，'
                '继续检测会得到比不校正更差的结果。请补充文本，或改用字符 2-4gram '
                '（门槛 20 条）。' % (self.feat_name, need_min))

        B = self.transform(texts)
        cands = self._candidates(B)
        scored = {}
        for name, Bc in cands.items():
            z = self._scores(Bc)
            pf, _ = tta.pred_gmm(z)
            scored[name] = (Bc, z, pf)

        if mode == 'auto':
            pick = max(scored, key=lambda k: (-np.inf
                                              if np.isnan(scored[k][2])
                                              else scored[k][2]))
        else:
            if mode not in scored:
                raise KeyError('未知变体 %s，可选：%s'
                               % (mode, ', '.join(sorted(scored))))
            pick = mode

        Bc, z, pf = scored[pick]
        prob = self.clf.predict_proba(Bc)[:, 1]
        gamma0 = self.clf.predict_proba(B)[:, 1]
        diag = diagnostics.estimate(self.X_src, self.y_src, B, gamma0)

        warn = None
        need = rec_n(self.feat_name)
        if len(texts) < need:
            warn = ('批次仅 %d 条：类条件均值由软标签估计，%s 特征少于 %d 条时 '
                    'Ξ 与 Δ_T 波动较大，判定结果仍可用，诊断量仅供参考。'
                    % (len(texts), self.feat_name, need))

        res = {
            'n': len(texts),
            'warn': warn,
            'feat': self.feat_name,
            'dim': self.dim,
            'chosen': pick,
            'candidates': {k: (None if np.isnan(v[2]) else round(float(v[2]), 4))
                           for k, v in scored.items()},
            'predicted_f1': None if np.isnan(pf) else round(float(pf), 4),
            'ai_rate': round(float((prob > 0.5).mean()), 4),
            'mean_prob': round(float(prob.mean()), 4),
            'diag': diag,
        }
        if with_items:
            order = np.argsort(-prob)
            if item_limit and len(order) > item_limit:
                order = order[:item_limit]
            res['items'] = [{'index': int(i), 'p_ai': round(float(prob[i]), 4),
                             'label': 'AI' if prob[i] > 0.5 else '人类',
                             'text': str(texts[i])[:200]} for i in order]
        return res

    # -- 存取 ---------------------------------------------------------------
    def save(self, path, compress=3):
        if joblib is None:
            raise RuntimeError('需要 joblib：pip install joblib')
        import os
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        joblib.dump(self, path, compress=compress)
        return path

    @staticmethod
    def load(path):
        if joblib is None:
            raise RuntimeError('需要 joblib：pip install joblib')
        return joblib.load(path)

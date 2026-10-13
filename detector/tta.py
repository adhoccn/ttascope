"""测试时自适应：从整批对齐到类条件对齐的校正族，以及无标签的选参准则。

统一族（与论文最终实现一致，见 exp/X5_unified_impl.py）。设批特征矩阵 B、
源域类条件均值 mu0_tr / mu1_tr、批中心 centre_tr = ½(mu0_tr + mu1_tr)：

    bn     (B - mean(B)) / sd(B) · sd_tr + mu_tr     批 z-score 后仿射回源域
    em     软后验 γ，位移 a·(γ-π) 施加于全样本       类条件对齐
    conf   均值用全样本软后验估计，位移只作用于 top-keep 置信样本

其中 a = dmu_tr - dmu_te，dmu 是两类均值之差；**每一轮迭代后**都把批混合
中心钉回 centre_tr（Bc += centre_tr - mean(Bc)），而不是像早期实现那样只在
开头做一次 bnc 预步。

整批对齐 bnc 已移出候选族：论文表 S5 显示它会损害排序质量（AUC），无标签
选参准则在若干配置上会选中它并导致结果为负（§5.3）。

选参（E30）：候选变体各自产出一批分数，用一维两分量 GMM 在**无标签**的
测试分数上解析预测 macro-F1，取预测值最高的变体。排序本身不被校正改变，
因此收益全部来自阈值，该准则与"用真值标签选"高度一致。
"""
import numpy as np
from scipy.stats import norm

try:
    from sklearn.mixture import GaussianMixture      # 顶层导入：首次请求才导入会多付 ~1.5 s
except ImportError:                                  # pragma: no cover
    GaussianMixture = None

PI = 0.5


def bnc(B, mu_tr):
    """批中心对齐：把测试批的均值搬回训练批均值。"""
    return B + (mu_tr - B.mean(axis=0))


def bn(B, mu_tr, sd_tr):
    """批 z-score 后仿射回训练坐标系，保留分类器的尺度语义。"""
    mu = B.mean(axis=0)
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (B - mu) / sd * sd_tr + mu_tr


def tta_final(B, clf, mu0_tr, mu1_tr, iters=3, lam=1.0, keep=1.0, mode='em'):
    """统一实现的类条件对齐（= exp/X5_unified_impl.tta_final，论文口径）。

    与旧 align 的三点差异：
      1. 不做开头的 bnc 预步；
      2. 每轮迭代后把批混合中心钉回 ½(mu0_tr + mu1_tr)；
      3. conf 下均值仍用全样本软后验估计，只有位移被 mask 选通。

    mode: em / conf（trim 未进入候选族，保留分支仅为兼容旧调用）
    keep: conf 下参与位移的高置信样本比例
    """
    dmu_tr = mu1_tr - mu0_tr
    centre_tr = 0.5 * (mu0_tr + mu1_tr)
    Bc = np.asarray(B, dtype=np.float64).copy()
    for _ in range(iters):
        g = clf.predict_proba(Bc)[:, 1]
        if keep < 1.0:
            conf = np.abs(g - PI)
            k = max(int(round(keep * len(g))), 20)
            idx = np.argsort(-conf)[:k]
            mask = np.zeros(len(g), dtype=bool)
            mask[idx] = True
        else:
            mask = np.ones(len(g), dtype=bool)
        if mode == 'trim':
            gw = np.where(mask, g, 0.0)
            gw0 = np.where(mask, 1.0 - g, 0.0)
            s1, s0 = gw.sum(), gw0.sum()
        else:
            s1, s0 = g.sum(), (1.0 - g).sum()
            gw, gw0 = g, 1.0 - g
        if s1 < 1e-6 or s0 < 1e-6:
            break
        mu1 = (Bc * gw[:, None]).sum(axis=0) / s1
        mu0 = (Bc * gw0[:, None]).sum(axis=0) / s0
        a = dmu_tr - (mu1 - mu0)
        step = lam * a * (g - PI)[:, None]
        if mode == 'conf':
            step = step * mask[:, None]
        Bc = Bc + step
        Bc = Bc + (centre_tr - Bc.mean(axis=0))
    return Bc


def align(B, clf, mu0_tr, mu1_tr, iters=3, lam=1.0, mode='em', keep=0.5,
          oracle_g=None):
    """早期实现（开头一次 bnc 预步 + 每步扣位移批均值）。

    仅保留供复现旧实验与 oracle 上界标定，**不进入在线候选族**。
    在线口径一律走 tta_final。
    """
    Bc = bnc(B, 0.5 * (mu0_tr + mu1_tr))
    dmu_tr = mu1_tr - mu0_tr
    for _ in range(iters):
        g = (oracle_g.astype(np.float64) if oracle_g is not None
             else clf.predict_proba(Bc)[:, 1])
        if mode in ('trim', 'conf'):
            k = max(int(round(keep * len(g))), min(20, len(g)))
            score = np.ones(len(g)) if oracle_g is not None else np.abs(g - PI)
            mask = np.zeros(len(g), dtype=bool)
            mask[np.argsort(-score)[:k]] = True
        else:
            mask = np.ones(len(g), dtype=bool)

        w1 = np.where(mask, g, 0.0)
        w0 = np.where(mask, 1.0 - g, 0.0)
        s1, s0 = w1.sum(), w0.sum()
        if s1 < 1e-6 or s0 < 1e-6:
            break
        mu1 = (Bc * w1[:, None]).sum(axis=0) / s1
        mu0 = (Bc * w0[:, None]).sum(axis=0) / s0
        a = dmu_tr - (mu1 - mu0)
        step = lam * a * (g - PI)[:, None]
        if mode == 'conf':
            step = step * mask[:, None]
        Bc = Bc + step - step.mean(axis=0)
        if oracle_g is not None:
            break
    return Bc


def pred_f1(z, yhat):
    """由 score 分布解析预测 macro-F1（高斯近似）。

    每类正确率 ≈ Φ(|mean_c| / sd_c)，边界在 0；由此得 TP/FP/FN/TN 与
    macro-F1。yhat 用于分组估计 (mean, sd)：真值标签用于机制验证，伪标签
    用于无监督选参。
    """
    z = np.asarray(z, dtype=np.float64)
    yhat = np.asarray(yhat)
    m1, s1 = _ms(z[yhat == 1])
    m0, s0 = _ms(z[yhat == 0])
    sgn = 1.0 if m1 >= m0 else -1.0
    e1 = float(norm.cdf(-sgn * m1 / s1))          # AI 被判为人类
    e0 = float(norm.cdf(sgn * m0 / s0))           # 人类被判为 AI
    return _macro_f1(e0, e1, 0.5, 0.5), dict(m1=round(m1, 3), s1=round(s1, 3),
                                             m0=round(m0, 3), s0=round(s0, 3))


def pred_gmm(z):
    """无标签版本：在测试 score 上拟合一维两分量 GMM，再解析预测 macro-F1。"""
    try:
        if GaussianMixture is None:
            return float('nan'), {}
        zz = np.asarray(z, dtype=np.float64).reshape(-1, 1)
        # max_iter 上限取 100：批次过小或分数退化时 GMM 不收敛，若放到
        # 默认的 300 会把单次拟合拖到数百毫秒（11 个候选变体即数秒）。
        # tol=1e-3 在正常批次下远早于 100 轮收敛，实测选型结果不变。
        gm = GaussianMixture(n_components=2, random_state=0, max_iter=100,
                             tol=1e-3,
                             means_init=np.array([[zz.min()], [zz.max()]])).fit(zz)
        order = np.argsort(gm.means_.ravel())
        mu0, mu1 = gm.means_.ravel()[order]
        s0 = np.sqrt(max(gm.covariances_.ravel()[order[0]], 1e-9))
        s1 = np.sqrt(max(gm.covariances_.ravel()[order[1]], 1e-9))
        w0, w1 = gm.weights_.ravel()[order]
        e1 = float(norm.cdf(-mu1 / s1))
        e0 = float(norm.cdf(mu0 / s0))
        f1 = _macro_f1(e0, e1, w0, w1)
        return f1, dict(mu1=round(float(mu1), 3), mu0=round(float(mu0), 3),
                        s1=round(float(s1), 3), s0=round(float(s0), 3),
                        pi1=round(float(w1), 3))
    except Exception:
        return float('nan'), {}


def _ms(v):
    if len(v) == 0:
        return 0.0, 1e-9
    return float(v.mean()), max(float(v.std()), 1e-9)


def _macro_f1(e0, e1, w0, w1):
    tp = w1 * (1 - e1)
    fp = w0 * e0
    fn = w1 * e1
    tn = w0 * (1 - e0)
    p1 = tp / max(tp + fp, 1e-9)
    r1 = tp / max(tp + fn, 1e-9)
    p0 = tn / max(tn + fn, 1e-9)
    r0 = tn / max(tn + fp, 1e-9)
    f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-9)
    f1_0 = 2 * p0 * r0 / max(p0 + r0, 1e-9)
    return 0.5 * (f1_0 + f1_1)

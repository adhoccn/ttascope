"""测试时自适应：从整批对齐到类条件对齐的校正族，以及无标签的选参准则。

统一族（E28/E29）。设批特征矩阵 B、源域类条件均值 mu0_tr / mu1_tr：

    bnc    B + (mu_tr - mean(B))                     批中心对齐，理论残留 1/2
    em     软后验 γ，位移 a·(γ-π) 施加于全样本      类条件对齐
    trim   均值只由 top-keep 置信样本估计，位移仍施加全样本
    conf   均值与位移都只作用于 top-keep 置信样本

其中 a = dmu_tr - dmu_te，dmu 是两类均值之差；每步都减去位移的批均值以锁住
批中心。λ 在 (0,1) 内给出批对齐与类条件对齐的连续插值。

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


def align(B, clf, mu0_tr, mu1_tr, iters=3, lam=1.0, mode='em', keep=0.5,
          oracle_g=None):
    """类条件对齐的通用实现。

    mode: em / trim / conf
    keep: trim 与 conf 下参与估计（conf 下也参与位移）的高置信样本比例
    oracle_g: 给定则 E 步用真值后验（仅用于标定上界）
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

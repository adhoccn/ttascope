"""判别漂移诊断。

Ξ_eff 是漂移在判别方向上的有符号投影——同一个 δ 若与判别方向正交，
总量再大也不损害分类：

    u   = Σ_W^{-1}(μ_AI - μ_HU) / Δ         单位马氏判别方向
    Δ   = sqrt(dmuᵀ Σ_W^{-1} dmu)           源域马氏分离度
    Ξ_ai = -uᵀ(μ_AI^te - μ_AI^src)
    Ξ_hu = -uᵀ(μ_HU^te - μ_HU^src)
    Δ_T  = Δ - Ξ_ai + Ξ_hu                  目标域实测类间间隔（恒等式）

推论 1 给出目标域精度的解析预测：macro-F1 ≈ 1 - Φ(-Δ'/2)，其中
Δ' = Δ - Ξ_eff。实测 MAE 0.042（论文 §4.4）。

目标批次没有标签，故类条件均值用分类器自身的后验 γ 加权估计
（软分配），与 E 步同源；这正是"无标签"能成立的原因。
"""
import numpy as np
from scipy.stats import norm

# 论文表 12：按初始 Ξ_eff 分档的实测平均校正增益（30 组配置）
GAIN_BANDS = [(0.3, 8, 0.001), (0.6, 6, 0.002), (1.2, 5, 0.031),
              (float('inf'), 11, 0.045)]

# 类先验的前置拒绝规则（§5.4 / 表 S11）。
#
# 方法隐含 π≈0.5；当批次整批同源（π→1）时加权矩估计失效，校正不再带来收益
# 反而造成 −0.024~−0.060 的损失。但类平衡告警本身不足以捕捉这一点：
# 批次观测量满足仿射律 r_AI = FPR + π(TPR−FPR)，斜率 κ=TPR−FPR 恒小于 1，
# 故 |r_AI−0.5| 系统性**低估**真实失衡（π=0.7 时 15 维/deepseek 的偏离仅
# 0.008，告警完全静默）。因此把 r_AI 本身作为显式的前置判据：
# 10 组配置、每格 5 次抽样下，可用区（π≤0.6）误触发率 0.00，
# 危险区（π≥0.9）平均触发率 0.71。
REJECT_R_AI = 0.64


def estimate(X_src, y_src, X_tgt, gamma, ridge=1e-2):
    """估计目标批的判别漂移。

    X_src/y_src : 源域特征与标签（用于 Σ_W 与判别方向）
    X_tgt       : 目标批特征
    gamma       : 目标批每条文本属于 AI 的后验（软标签）

    返回 dict：Delta / Xi_ai / Xi_hu / Delta_T / SNR_T / pred_f1 /
               pred_f1_prime / band / expected_gain
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    y_src = np.asarray(y_src).astype(int)
    gamma = np.clip(np.asarray(gamma, dtype=np.float64), 1e-6, 1 - 1e-6)
    d = X_src.shape[1]

    mu = {}
    SW = np.zeros((d, d))
    for c in (0, 1):
        Xc = X_src[y_src == c]
        if len(Xc) < 2:
            continue
        mu[c] = Xc.mean(axis=0)
        Xc0 = Xc - mu[c]
        SW += Xc0.T @ Xc0
    if len(mu) < 2:
        return {}
    SW /= max(len(X_src) - 2, 1)
    SW = SW + ridge * (np.trace(SW) / max(d, 1)) * np.eye(d)
    SWi = np.linalg.pinv(SW)

    dmu = mu[1] - mu[0]
    Delta = float(np.sqrt(max(float(dmu @ SWi @ dmu), 0.0)))
    u = (SWi @ dmu) / Delta if Delta > 1e-9 else np.zeros_like(dmu)

    w1, w0 = gamma, 1.0 - gamma
    mu_ai_te = (X_tgt * w1[:, None]).sum(axis=0) / w1.sum()
    mu_hu_te = (X_tgt * w0[:, None]).sum(axis=0) / w0.sum()

    xi_ai = float(-u @ (mu_ai_te - mu[1]))
    xi_hu = float(-u @ (mu_hu_te - mu[0]))
    delta_T = float(u @ (mu_ai_te - mu_hu_te))

    # 目标域类内尺度：两类的加权协方差在判别方向上的标准差
    resid = np.vstack([X_tgt - mu_ai_te, X_tgt - mu_hu_te])
    sw_t = resid.T @ resid / max(len(resid), 1)
    var_t = float(u @ sw_t @ u)
    sigma_T = float(np.sqrt(max(var_t, 1e-12)))

    # 类条件均值由软标签加权估计，样本太少时 Ξ 与 Δ_T 不可靠
    reliable = bool(len(X_tgt) >= 50)
    return {
        'n': int(len(X_tgt)),
        'reliable': reliable,
        'Delta': round(Delta, 4),
        'Xi_ai': round(xi_ai, 4),
        'Xi_hu': round(xi_hu, 4),
        'Delta_T': round(delta_T, 4),
        'identity_resid': round(Delta - xi_ai + xi_hu - delta_T, 6),
        'sigma_T': round(sigma_T, 4),
        'SNR_T': round(delta_T / max(sigma_T, 1e-9), 4),
        'pred_f1': round(pred_f1(delta_T), 4),
        'pred_f1_source': round(pred_f1(Delta), 4),
        'band': band_of(xi_ai),
        'expected_gain': expected_gain(xi_ai),
    }


def prior_reject(gamma, cut=REJECT_R_AI):
    """类先验的前置拒绝判据（§5.4 / 表 S11）。

    gamma : 未校正批次上每条文本属于 AI 的后验（不使用目标域标签）
    cut   : 拒绝阈值，超过即认为批次疑似整批同源

    返回 dict：r_ai / cut / rejected / applied，其中 applied 由调用方
    在"确实把选参结果退回 none"后置 True——判据与是否执行分离，便于显式
    指定变体时保留用户意图。
    """
    r_ai = float(np.mean(np.asarray(gamma, dtype=np.float64) > 0.5))
    return {'r_ai': round(r_ai, 4), 'cut': cut,
            'rejected': bool(r_ai > cut), 'applied': False}


def pred_f1(sep):
    """推论 1：等协方差高斯下 macro-F1 ≈ 1 - Φ(-Δ/2)。"""
    return float(1.0 - norm.cdf(-sep / 2.0))


def band_of(xi):
    for hi, n, _ in GAIN_BANDS:
        if abs(xi) <= hi:
            return '≤%.1f（%d 组）' % (hi, n) if hi != float('inf') else '>1.2（11 组）'
    return '>1.2（11 组）'


def expected_gain(xi):
    for hi, _, g in GAIN_BANDS:
        if abs(xi) <= hi:
            return g
    return GAIN_BANDS[-1][2]

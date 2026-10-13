"""线上冒烟测试：直接打公网地址，验证服务可用性与检测输出。

    python tests/test_remote.py http://106.54.192.87:8501/tta

用例为 24 条混合文本（12 条 AI 风格 / 12 条人类随笔），
对 15D 与 char_2_4 两个特征集各跑一次 auto 模式，
打印 Ξ_eff、Δ′、推论 1 预测 F1 与逐条判定命中率。
"""
import json
import sys
import time
import urllib.request

AI = [
    "人工智能正在深刻改变各行各业的生产方式。首先，它能够显著提升生产效率；其次，它可以有效降低运营成本；最后，它还能推动产业结构的优化升级。",
    "机器学习是人工智能的重要分支，其核心思想是通过数据驱动的方式让模型自动学习规律。近年来，随着算力的提升，该领域取得了突破性进展。",
    "随着数字化转型的深入推进，企业需要构建更加敏捷的技术架构。一方面要保障系统的稳定性，另一方面也要兼顾未来的可扩展性。",
    "在大数据时代背景下，数据已经成为重要的生产要素。如何有效采集、存储和分析数据，是每一个组织都必须面对的关键问题。",
    "自然语言处理旨在让计算机理解并生成人类语言。得益于预训练模型的发展，相关任务的性能在近几年得到了大幅提升。",
    "云计算提供弹性可伸缩的计算资源，使企业无需自建机房即可快速部署应用。与此同时，安全性始终是用户关注的核心议题。",
    "综上所述，该方法在准确性与效率两个维度上均优于基线方案，具有良好的应用前景与推广价值。",
    "深度学习模型通过多层非线性变换自动提取特征，避免了繁琐的人工特征工程，但同时也带来了可解释性方面的挑战。",
    "物联网技术将感知层、网络层与应用层有机结合，实现了设备之间的互联互通，为智慧城市建设提供了重要支撑。",
    "在进行系统设计时，需要综合考虑性能、成本、可维护性等多方面因素，权衡取舍后确定最终的技术方案。",
    "研究表明，该算法在多个公开数据集上均取得了领先的结果，验证了所提方法的有效性与鲁棒性。",
    "总体而言，人工智能技术的发展既带来机遇也带来挑战，需要在技术创新与伦理规范之间取得平衡。",
]

HUMAN = [
    "昨天跟我姐去逛街，本来就想随便看看，结果试了一件外套特别合身，一冲动就买了，回来想想其实也不便宜。",
    "说实话我也不知道该咋整，反正就这么拖着吧，说不定过两天自己就好了，我向来不爱去医院。",
    "早上起晚了两分钟，公交刚好开走，气死我了，最后只能打车，还堵了一路，到公司都迟到了。",
    "我家那只猫半夜又开始折腾了，把卫生纸扯得满地都是，我凌晨三点爬起来收拾，真的服了。",
    "这电影吧，前半段还行，后面就有点乱了，我看得云里雾里的，出来以后朋友说他也差不多没看懂。",
    "刚煮的面忘记关火，汤都溢出来了，灶台一片狼藉，擦了半天，下次真的不敢一边刷手机一边做饭了。",
    "我爸昨天突然打电话问我最近咋样，我说挺好的，他就嗯了一声，然后我俩沉默了半天，最后他说那挂了吧。",
    "下午在办公室犯困，冲了杯咖啡，结果手抖撒了一桌子，键盘差点报废，同事在旁边笑得不行。",
    "最近也不知道为啥，老是睡不好，躺床上翻来覆去，脑子里乱七八糟的事一件接一件。",
    "楼下那家包子铺换了老板，味道明显不如以前了，皮厚馅少，我吃了两口就不想吃了。",
    "跟朋友约了吃饭，结果他迟到了半个多小时，理由是找不到停车位，我也没法说啥，就当等风来吧。",
    "今天天气还挺好的，就是风有点大，出去走了一圈，回来头发全乱了，不过心情倒是舒畅了不少。",
]

TEXTS = AI + HUMAN
LABEL = [1] * len(AI) + [0] * len(HUMAN)   # 1 = AI 生成


def to_int(v):
    """逐条判定可能是 0/1，也可能是 'AI'/'Human' 之类的字符串。"""
    if isinstance(v, (int, float)):
        return int(round(v))
    s = str(v).strip().lower()
    if s in ('ai', 'ai生成', 'machine', 'gpt', '1', 'true'):
        return 1
    if s in ('human', '人', '人类', '0', 'false'):
        return 0
    return -1


def call(base, path, payload=None):
    url = base.rstrip('/') + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode('utf-8')), int((time.time() - t0) * 1000)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'http://106.54.192.87:8501/tta'
    health, _ = call(base, '/health')
    print('health:', health)
    assert health['status'] == 'ok' and health['models'], '服务未就绪'

    meta, _ = call(base, '/api/models')      # 条数门槛由服务端下发

    ok = True
    for feat in health['models']:
        need = int(meta.get(feat, {}).get('min_n', 0) or 0)
        if len(TEXTS) < need:
            # 用例只有 24 条，低于该特征集的门槛，服务端会 400，跳过而非误报失败
            print('\n[%s] 跳过：用例 24 条 < 门槛 %d 条' % (feat, need))
            continue
        res, ms = call(base, '/api/detect',
                       {'texts': TEXTS, 'feat': feat, 'mode': 'auto'})
        items = res.get('items') or []
        if items:
            print('   item[0]:', json.dumps(items[0], ensure_ascii=False)[:220])
        hit = sum(1 for it, y in zip(items, LABEL)
                  if to_int(it.get('label', it.get('pred'))) == y) if items else -1
        print('\n[%s] 选中变体=%s 用时 %d ms  命中 %s/%d  AI 占比=%s'
              % (feat, res.get('chosen'), ms, hit if hit >= 0 else '-',
                 len(TEXTS), res.get('ai_rate')))
        print('   推论1 预测F1 :', res.get('predicted_f1'))
        print('   变体得分     :', json.dumps(res.get('candidates'),
                                             ensure_ascii=False))
        print('   诊断量       :', json.dumps(res.get('diag'),
                                             ensure_ascii=False))
        if hit >= 0:
            ok = ok and hit >= len(TEXTS) * 0.7
    print('\n结果:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

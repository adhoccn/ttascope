"""长文端到端测试：整篇文本按句切块后检测，验证线上能否给出整篇结论。

    python tests/test_long.py http://localhost:8501/tta

切分规则与前端 `splitTexts(..., 'sent', size)` 等价：按句切分后累积到
size 字成一块，残余 ≥ 20 字单独成块，否则并入上一块。
"""
import json
import re
import sys
import urllib.request

AI_DOC = (
    "人工智能正在深刻改变各行各业的生产方式，其重要性不言而喻。首先，它能够显著"
    "提升生产效率，把大量重复性劳动交给自动化系统完成；其次，它可以有效降低运营"
    "成本，减少人为失误带来的损失；最后，它还能够改善用户体验，通过个性化推荐与"
    "智能客服提升服务质量。"
    "从技术层面看，机器学习是人工智能的重要分支，其核心思想是通过数据驱动的方式"
    "让模型自动学习规律，而不是依靠人工编写规则。近年来，得益于算力的快速增长与"
    "大规模数据集的积累，深度学习在图像识别、语音识别、自然语言处理等领域取得了"
    "突破性进展，部分任务的表现已经接近甚至超过人类水平。"
    "然而，技术进步也带来了新的挑战。一方面，深度模型的可解释性仍然不足，在医疗、"
    "司法等高风险场景中难以直接部署；另一方面，训练数据中的偏见会被模型放大，造"
    "成不公平的判定结果。因此，研究者提出了包括对抗训练、公平性约束在内的多种缓"
    "解手段，但从工程角度看，这些方法的成本依然偏高。"
    "展望未来，人工智能的发展将呈现三个趋势：其一是多模态融合，文本、图像、语音"
    "的统一建模将成为主流；其二是模型小型化，使能力可以在端侧设备上运行；其三是"
    "与行业知识的深度结合，通用模型配合领域微调将是落地的主要形态。综上所述，人"
    "工智能技术的应用前景十分广阔，但也需要配套治理机制以保障其安全可控。"
)

HUMAN_DOC = (
    "昨天下午我本来打算去图书馆还书，结果出门的时候发现天阴沉沉的，想着可能要下"
    "雨，又折回去拿了把伞。走到一半果然下起来了，还好雨不大，伞也没白拿。"
    "到了图书馆才发现周一闭馆，我完全忘了这回事，站在门口愣了一下，只好又原路返回的。"
    "路上顺手买了杯奶茶，店员问我要不要加珍珠，我说随便吧，结果端上来甜得有点齁，"
    "喝了半杯就放包里了。"
    "回家以后我妈打电话问我这周末回不回去，我说看情况吧，她嗯了一声，又叮嘱我别老"
    "点外卖，我说知道了知道了，其实我也懒得做饭。"
    "晚上刷手机看到之前收藏的一家面馆，想着明天中午去试试吧，希望别又踩雷，上次那"
    "家真的咸得我喝了两大杯水。"
)


def split_sent(text, size=200):
    flat = re.sub(r'\s+', ' ', text).strip()
    sents = re.findall(r'[^。！？；!?;]+[。！？；!?;]*', flat) or [flat]
    out, buf = [], ''
    for s in sents:
        buf += s
        if len(buf) >= size:
            out.append(buf.strip())
            buf = ''
    if len(buf.strip()) >= 20:
        out.append(buf.strip())
    elif buf.strip() and out:
        out[-1] += buf.strip()
    elif buf.strip():
        out.append(buf.strip())
    return out


def post(base, payload):
    req = urllib.request.Request(
        base.rstrip('/') + '/api/detect',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode('utf-8'))


def summarize(tag, doc, base, feat='char_2_4'):
    chunks = split_sent(doc)
    res = post(base, {'texts': chunks, 'feat': feat, 'mode': 'auto',
                      'item_limit': 500})
    items = res.get('items') or []
    ai = sum(1 for it in items if it.get('label') == 'AI')
    print('[%s] %d 块（%d 字）→ AI 块 %d/%d，AI 占比 %.1f%%，平均 P(AI) %.3f，'
          '选中 %s，预测 F1 %s'
          % (tag, len(chunks), len(doc), ai, len(items),
             100.0 * ai / max(1, len(items)),
             sum(it['p_ai'] for it in items) / max(1, len(items)),
             res.get('chosen'), res.get('predicted_f1')))
    return ai > len(items) / 2


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8501/tta'
    summarize('AI 长文', AI_DOC, base)
    summarize('人类长文', HUMAN_DOC, base)
    print('\n注：单篇长文切出的块数通常不足 20，批校正与整篇结论此时波动很大，'
          '本脚本只验证链路可用与输出结构，不对判定正确性下断言。')
    return 0


if __name__ == '__main__':
    sys.exit(main())

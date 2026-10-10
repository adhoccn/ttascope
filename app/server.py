"""TTAScope 在线检测服务。

    uvicorn app.server:app --host 127.0.0.1 --port 8511

对外接口：
    GET  /                 检测页面
    GET  /health           健康检查
    GET  /api/models       已装载的模型清单
    POST /api/detect       批量检测（正文 texts / feat / mode）

模型文件放在 models/ 下，启动时全部装载；一个特征集一个 .joblib。
"""
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS_DIR = Path(os.environ.get('TTASCOPE_MODELS', ROOT / 'models'))
WEB_DIR = HERE / 'web'

import sys
sys.path.insert(0, str(ROOT))
from detector.model import TTADetector, min_n, rec_n   # noqa: E402

app = FastAPI(title='TTAScope', version='1.0.0',
              description='面向未知大语言模型的中文 AI 生成文本检测（判别漂移 + 无标签批校正）')

_MODELS = {}
_START = time.time()


def load_models():
    _MODELS.clear()
    if not MODELS_DIR.is_dir():
        return
    for p in sorted(MODELS_DIR.glob('*.joblib')):
        try:
            det = TTADetector.load(str(p))
            _MODELS[det.feat_name] = {'path': p.name, 'det': det,
                                      'dim': getattr(det, 'dim', None),
                                      'train_acc': round(getattr(det, 'train_acc', 0.0), 4)}
        except Exception as e:
            print('[warn] 模型装载失败 %s: %s' % (p.name, e), flush=True)


_WARMUP = ["人工智能正在深刻改变各行各业的生产方式，其重要性不言而喻。首先，它能够显著提升生产效率；其次，它可以降低运营成本。",
           "机器学习是人工智能的重要分支，其核心思想是通过数据驱动的方式让模型自动学习规律，近年来取得了突破性进展。",
           "随着数字化转型的深入推进，企业需要构建更加敏捷的技术架构，一方面保障稳定性，另一方面兼顾可扩展性。",
           "在大数据时代背景下，数据已经成为重要的生产要素，如何有效采集、存储和分析数据是关键问题。",
           "自然语言处理旨在让计算机理解并生成人类语言，得益于预训练模型的发展，相关任务性能大幅提升。",
           "云计算提供弹性可伸缩的计算资源，使企业无需自建机房即可快速部署应用，安全性始终是核心议题。",
           "我觉得这事儿吧，得看具体情况，反正我上次试了一下，感觉还行，不过也说不好是不是运气好。",
           "说实话我也不是很懂这个，就是随便看看，你要是问我为什么，我大概也说不太清楚，差不多得了。",
           "昨天跟朋友出去吃饭，点了好多菜，结果根本吃不完，最后打包带回来了，那家店味道还挺不错的。",
           "我跟你说啊，这个东西真的挺有意思的，一开始我也没当回事，后来发现还挺好玩的，就是有点费时间。"]


def warmup():
    """启动时跑一遍，避免首个请求承担惰性导入与首次分配的开销（实测约 3 s）。"""
    for name, m in _MODELS.items():
        try:
            m['det'].detect(_WARMUP, with_items=False)
        except Exception as e:
            print('[warn] 预热失败 %s: %s' % (name, e), flush=True)


load_models()
warmup()


class DetectRequest(BaseModel):
    texts: list = Field(..., description='待检测文本列表，一行一条')
    feat: str = '15D'
    mode: str = 'auto'
    item_limit: int = 200


@app.get('/health')
def health():
    return {'status': 'ok', 'models': sorted(_MODELS),
            'uptime_sec': round(time.time() - _START, 1)}


@app.get('/api/models')
def models():
    # 门槛随模型返回，前端不再各写一份，避免两边漂移
    return {k: dict({kk: vv for kk, vv in v.items() if kk != 'det'},
                    min_n=min_n(k), rec_n=rec_n(k))
            for k, v in _MODELS.items()}


@app.post('/api/detect')
def detect(req: DetectRequest):
    texts = [str(t).strip() for t in req.texts if str(t).strip()]
    if len(texts) < 2:
        raise HTTPException(400, '批校正需要成批文本，至少 2 条（建议 ≥ 20 条）')
    if req.feat not in _MODELS:
        raise HTTPException(400, '模型 %s 未装载，可用：%s'
                            % (req.feat, ', '.join(sorted(_MODELS)) or '无'))
    need = min_n(req.feat)
    if len(texts) < need:
        raise HTTPException(
            400, '%s 特征要求不少于 %d 条，当前 %d 条：小批次实验显示该区间类条件'
                 '对齐的期望增益为负，强行检测会得到比不校正更差的结果。'
                 '请补充文本或改用门槛更低的字符 2-4gram。'
                 % (req.feat, need, len(texts)))
    t0 = time.time()
    try:
        res = _MODELS[req.feat]['det'].detect(texts, mode=req.mode,
                                              item_limit=req.item_limit)
    except Exception as e:
        raise HTTPException(500, '检测失败：%s' % e)
    res['elapsed_ms'] = int((time.time() - t0) * 1000)
    res['mode_requested'] = req.mode
    return JSONResponse(res)


app.mount('/static', StaticFiles(directory=str(WEB_DIR)), name='static')


@app.get('/')
def index():
    # 前端改动频繁，禁缓存，避免浏览器留着旧页面
    return FileResponse(str(WEB_DIR / 'index.html'),
                        headers={'Cache-Control': 'no-store'})


# 路径拼错时给个提示页，而不是一屏 Not Found。
# 反向代理会剥掉 /tta/ 前缀，跳转必须用相对路径。
_FALLBACK = """<!doctype html><meta charset="utf-8">
<title>TTAScope · 路径不存在</title>
<style>
 body{font:15px/1.7 -apple-system,"Microsoft YaHei",sans-serif;background:#0f1115;
      color:#e6e8ec;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
 .box{max-width:520px;padding:32px 36px;background:#171a21;border:1px solid #272b34;border-radius:12px}
 h1{margin:0 0 12px;font-size:18px} code{background:#22262f;padding:2px 6px;border-radius:4px}
 a{color:#61a8ff}
</style><div class="box">
<h1>页面路径不存在</h1>
<p>请访问 <a href="./">./</a> 打开检测页。</p>
<p>接口入口为 <code>&lt;本地址&gt;/health</code> 与 <code>&lt;本地址&gt;/api/detect</code>。</p>
</div>"""


@app.exception_handler(404)
async def fallback_404(request: Request, exc):
    if request.url.path.startswith('/api'):
        return JSONResponse({'detail': 'Not Found'}, status_code=404)
    return HTMLResponse(_FALLBACK, status_code=404)

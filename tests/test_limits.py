"""服务侧资源护栏测试（#4 限流的应用层部分）。

    python tests/test_limits.py

nginx 层的 limit_req / limit_conn 必须有真实 nginx 才能验证，不在这里覆盖
（线上验证方式见 deploy/DEPLOY_LOG.md）；本用例校验应用层的最后一道护栏：
单批条数、总字符数上限、单条超长截断，以及 /health 对外暴露的门限值。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient           # noqa: E402

import app.server as srv                            # noqa: E402

client = TestClient(srv.app)
AI = ("综上所述，人工智能技术在各个领域的应用正在不断深化，其重要性不言而喻。"
      "首先，该技术能够显著提升生产效率；其次，它有助于降低运营成本。")
HU = ("我跟你说啊，这事儿吧，我也说不好。反正就是那么回事儿，差不多得了，"
      "你要是觉得行就试试呗，不行就算了。")


def _feat():
    m = client.get('/api/models').json()
    if not m:
        raise AssertionError('未装载任何模型，无法测试接口护栏')
    return 'char_2_4' if 'char_2_4' in m else sorted(m)[0]


def test_health_limits():
    h = client.get('/health').json()
    lim = h.get('limits') or {}
    assert lim, 'health 未暴露 limits'
    for k in ('max_items', 'max_chars', 'max_item_chars'):
        assert k in lim and lim[k] > 0, (k, lim)
    assert lim['max_items'] == srv.MAX_ITEMS


def test_batch_too_many():
    texts = [AI if i % 2 else HU for i in range(srv.MAX_ITEMS + 200)]
    r = client.post('/api/detect', json={'texts': texts, 'feat': _feat()})
    assert r.status_code == 400, r.status_code
    assert '单批最多' in r.json()['detail'], r.json()


def test_total_chars_too_large():
    long_ = '人' * 3000
    texts = [long_] * 1000                     # 3e6 字 > MAX_CHARS
    r = client.post('/api/detect', json={'texts': texts, 'feat': _feat()})
    assert r.status_code == 400, r.status_code
    assert '分批提交' in r.json()['detail'], r.json()


def test_truncate_long_item():
    feat = _feat()
    need = client.get('/api/models').json()[feat]['min_n']
    texts = [AI if i % 2 else HU for i in range(max(need, 24))]
    texts[0] = '机' * (srv.MAX_ITEM_CHARS + 5000)   # 单条超长，应被截断而非拒
    r = client.post('/api/detect', json={'texts': texts, 'feat': feat})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    assert r.json().get('truncated') == 1, r.json().get('truncated')


def test_normal_batch_ok():
    feat = _feat()
    need = client.get('/api/models').json()[feat]['min_n']
    n = max(need, 24)
    texts = [AI if i % 2 else HU for i in range(n)]
    r = client.post('/api/detect', json={'texts': texts, 'feat': feat})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    assert r.json()['n'] == n


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('全部通过')

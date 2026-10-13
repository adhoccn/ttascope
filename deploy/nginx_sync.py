#!/usr/bin/env python3
"""把 TTAScope 的 nginx 片段幂等写入站点配置（含限流）。

    sudo python3 deploy/nginx_sync.py                 # 站点 8501 / 上游 8511
    sudo python3 deploy/nginx_sync.py --port 8510     # 上游端口不同时

做两件事：

1. 安装 http 级限流区到 /etc/nginx/conf.d/ttascope-limits.conf。zone 必须
   定义在 http 上下文，不能塞进 server/location，因此单独成文件。
2. 把渲染后的 location 片段写进站点配置：有哨兵标记就整段替换（配置升级），
   没有就插到 server 块第一个 location 之前（首次安装）。

只改配置文件，不做 nginx -t / reload —— 校验与生效交给调用方，避免在这里
失败时留下半生效状态。所有写入前都会备份为 <文件>.bak.<时间戳>。
"""
import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SENTINEL_BEG = '# >>> ttascope begin'
SENTINEL_END = '# <<< ttascope end'
# 哨兵标记之前的版本：整块以这行注释开头，到 `location = /tta` 结束
LEGACY_BEG = '# ── TTAScope'
LEGACY_END_RE = re.compile(r'^\s*location\s*=\s*/tta\s*\{')


def render(port, auth_off=True):
    """展开模板里的占位符。"""
    src = open(os.path.join(HERE, 'nginx-ttascope.conf'), encoding='utf-8').read()
    src = src.replace('__PORT__', str(port))
    src = src.replace('__AUTH_OFF__', 'auth_basic off;' if auth_off else '')
    return src.rstrip('\n')


def install_limits(limits_conf):
    src = open(os.path.join(HERE, 'nginx-ttascope-limits.conf'), encoding='utf-8').read()
    old = open(limits_conf, encoding='utf-8').read() if os.path.exists(limits_conf) else ''
    if old.strip() == src.strip():
        print('==> 限流区已是最新，跳过：%s' % limits_conf)
        return False
    if old:
        bak = '%s.bak.%d' % (limits_conf, int(time.time()))
        os.rename(limits_conf, bak)
        print('==> 备份旧限流区 %s' % bak)
    open(limits_conf, 'w', encoding='utf-8').write(src)
    print('==> 写入限流区 %s' % limits_conf)
    return True


def install_location(site_conf, frag):
    """幂等写入 location 片段。返回是否发生改动。"""
    s = open(site_conf, encoding='utf-8').read()
    if SENTINEL_BEG in s and SENTINEL_END in s:
        i = s.index(SENTINEL_BEG)
        j = s.index(SENTINEL_END) + len(SENTINEL_END)
        out, act = s[:i] + frag.rstrip('\n') + s[j:], '升级'
    else:
        lines = s.split('\n')
        beg = next((k for k, ln in enumerate(lines) if ln.strip().startswith(LEGACY_BEG)), None)
        if beg is not None:
            # 升级旧版：整块换掉，避免新旧两个 location /tta/ 并存导致 nginx -t 报 duplicate
            end = next((k for k in range(beg, len(lines)) if LEGACY_END_RE.match(lines[k])), None)
            if end is None:
                print('!! 找到旧版 %s 但缺收尾的 location = /tta，未改动' % LEGACY_BEG)
                return False
            lines = lines[:beg] + frag.split('\n') + lines[end + 1:]
            out, act = '\n'.join(lines), '升级（旧版整块替换）'
        else:
            # 首次安装：插到 server 块第一个 location 之前；找不到就追加到末尾
            hit = None
            for k, ln in enumerate(lines):
                if ln.lstrip().startswith('location'):
                    hit = k
                    break
            if hit is None:
                lines.append('')
                lines.append(frag)
            else:
                lines.insert(hit, frag)
            out, act = '\n'.join(lines), '插入'
    if out == s:
        print('==> location /tta/ 已是最新，跳过')
        return False
    bak = '%s.bak.%d' % (site_conf, int(time.time()))
    with open(bak, 'w', encoding='utf-8') as f:
        f.write(s)
    open(site_conf, 'w', encoding='utf-8').write(out)
    print('==> 已%s location /tta/（原配置备份为 %s）' % (act, bak))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='/etc/nginx/conf.d/hybridscope.conf',
                    help='站点配置文件（server 块所在）')
    ap.add_argument('--port', default=8511, help='上游 uvicorn 端口')
    ap.add_argument('--auth', action='store_true',
                    help='保留站点口令（默认对 /tta/ 关闭，便于外部测试）')
    a = ap.parse_args()

    frag = render(a.port, auth_off=not a.auth)
    limits = os.path.join(os.path.dirname(a.site), 'ttascope-limits.conf')

    if not os.path.exists(a.site):
        print('!! 未找到站点配置 %s' % a.site)
        return 1
    install_limits(limits)
    install_location(a.site, frag)
    print('==> 接下来：nginx -t && systemctl reload nginx')
    return 0


if __name__ == '__main__':
    sys.exit(main())

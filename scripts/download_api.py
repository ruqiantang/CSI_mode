#!/usr/bin/env python
import urllib.request, urllib.parse, json, subprocess, os, sys, time

# 仓库根目录（scripts/ 的上一级）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """极简 .env 加载：仅在环境变量尚未设置时写入，不覆盖已有值。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv(os.path.join(REPO_ROOT, ".env"))

# 凭据与本地路径均来自环境变量（见 .env.example），不再硬编码进仓库。
TOK = os.environ.get("ANYSHARE_TOKEN")
if not TOK:
    raise SystemExit(
        "缺少 ANYSHARE_TOKEN：请在仓库根目录创建 .env（参考 .env.example），"
        "或 export ANYSHARE_TOKEN=ory_at_... 后再运行。"
    )

REF = "https://disk.pku.edu.cn/anyshare/en-us/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E"
API = "https://disk.pku.edu.cn/api"
ROOT = "gns://F587FC39F6FA4FEF8E538C0E216361A3/13D975E9B37D426583414C5CABFFC1BD"
OUT = os.environ.get("ANYSHARE_DATA_DIR", os.path.join(REPO_ROOT, "data"))

def api(path, body=None, retries=5):
    for i in range(retries):
        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(API+path, data=data,
                headers={"Authorization":"Bearer "+TOK, "Referer":REF, "Content-Type":"application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=90).read())
        except Exception as e:
            if i == retries-1: raise
            time.sleep(5)

root_enc = urllib.parse.quote(ROOT, safe="")
data = api(f"/efast/v1/folders/{root_enc}/sub_objects?limit=100&sort=name&direction=asc&permission_attributes_required=false")
dirs = data.get("dirs", [])
dirs.sort(key=lambda d: (d["name"] != "D4", d["name"]))
print(f"顺序: {[d['name'] for d in dirs]}", flush=True)

for d in dirs:
    ds = d["name"]; fid = d["id"]
    fid_enc = urllib.parse.quote(fid, safe="")
    try:
        sub = api(f"/efast/v1/folders/{fid_enc}/sub_objects?limit=100&sort=name&direction=asc&permission_attributes_required=false")
    except Exception as e:
        print(f"  [x] {ds} 列表失败: {e}", flush=True); continue
    for f in sub.get("files", []):
        fname = f["name"]
        outdir = os.path.join(OUT, ds); os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, fname)
        if os.path.exists(out) and os.path.getsize(out) == f.get("size"):
            print(f"  [skip] {ds}/{fname}", flush=True); continue
        try:
            resp = api("/efast/v1/file/osdownload", {"docid": f["id"], "authtype": "anonymous", "savename": fname, "usehttps": True, "rev": f.get("rev","")})
        except Exception as e:
            print(f"  [x] {ds}/{fname} 获取下载地址失败: {e}", flush=True); continue
        ar = resp["authrequest"]; url = ar[1]
        cmd = ["curl", "-sS", "--retry", "5", "-C", "-", "-o", out]
        for h in ar[2:]: cmd += ["-H", h]
        cmd += [url]
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            sz = os.path.getsize(out); dt = time.time()-t0
            print(f"  [ok] {ds}/{fname} {sz/1e6:.0f}MB {sz/1e6/dt:.1f}MB/s", flush=True)
        except Exception as e:
            print(f"  [x] {ds}/{fname} 下载失败: {e}", flush=True)
print("=== 完成 ===", flush=True)

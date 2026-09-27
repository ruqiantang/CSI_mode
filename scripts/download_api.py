#!/usr/bin/env python
import urllib.request, urllib.parse, json, subprocess, os, sys, time
TOK = "ory_at_uy-lrQvstadDrWNNmmoXCT3OTIZiLE-FZVeGsQnc2os.6UDoVvhk4-7cQaEpnDg7rpMiUhGq-CQwaZv1doT__mE"
REF = "https://disk.pku.edu.cn/anyshare/en-us/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E"
API = "https://disk.pku.edu.cn/api"
ROOT = "gns://F587FC39F6FA4FEF8E538C0E216361A3/13D975E9B37D426583414C5CABFFC1BD"
OUT = "/root/WiFo/data"

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

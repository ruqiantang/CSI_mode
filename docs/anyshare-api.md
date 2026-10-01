# 北大网盘（AnyShare）数据下载 API 备忘

> 记录从北大网盘（disk.pku.edu.cn）下载 WiFo 数据集 D1–D16 的 API 流程。
> 完整可运行脚本见 `scripts/download_api.py`。

## 关键发现

- **API 地址是 `https://disk.pku.edu.cn/api/`**（注意 `/api/` 前缀，之前踩过坑）。
- 匿名分享链接**无需登录**（type=anonymous, password_required=false）。

## 下载流程（6 步）

1. **获取匿名 token**：用浏览器打开分享链接
   `https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E`，
   浏览器 cookie 中 `link_token:AA003E48...` 的值即 Ory token（`ory_at_...`）。

2. **获取链接信息**：
   `GET /api/shared-link/v1/links/{link_id}`
   → 返回 `{type:"anonymous", title:"dataset4train", item:{type:"folder"}}`。

3. **获取根文件夹 id**：
   `GET /api/efast/v1/entry-item`（Header: `Authorization: Bearer <ory_token>`）
   → 返回根条目，`id` 形如 `gns://F587.../13D9...`。

4. **列出子文件夹/文件**：
   `GET /api/efast/v1/folders/{folder_id}/sub_objects?limit=100&sort=name&direction=asc&permission_attributes_required=false`
   → 返回 `{"dirs":[...], "files":[...]}`，每个条目含 `id`、`rev`、`size`。

5. **获取下载地址**：
   `POST /api/efast/v1/file/osdownload`，body：
   `{"docid": "<file_id>", "authtype": "anonymous", "savename": "<文件名>", "usehttps": true, "rev": "<rev>"}`
   → 返回 `authrequest`，形如 `["GET", "https://diskasu.pku.edu.cn:10002/Bucket/.../rev", "Authorization: AWS ASE:...", "x-amz-date: ..."]`。

6. **下载**：用 curl 请求 `authrequest` 里的 URL，带上其中的认证头即可。

## 注意事项

- Ory token **会过期**，重新用浏览器打开链接即可拿到新 token。
- `scripts/download_api.py` 不再硬编码 token：请把 token 写入仓库根目录的 `.env`（`ANYSHARE_TOKEN=ory_at_...`，参考 `.env.example`），或用环境变量传入；输出目录可用 `ANYSHARE_DATA_DIR` 覆盖。
- 下载 URL 是 S3 直链（`diskasu.pku.edu.cn:10002`），有时效性，需尽快下载。
- 下载速度实测约 **10 MB/s**（服务器直连网盘 S3 存储）。
- 数据集结构：`dataset4train/` 下是 `D1`~`D16` 文件夹，每个含 `X_train.mat`（9000 样本）、`X_val.mat`（2000）、`X_test.mat`（1000），共约 34GB。

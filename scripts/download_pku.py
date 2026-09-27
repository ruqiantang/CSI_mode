"""从北大网盘（AnyShare 匿名分享）自动下载 WiFo D1-D16 训练数据。

已本地验证可行（Selenium 驱动浏览器：打开匿名链接 -> dataset4train -> D{i} -> 右键文件 -> 下载）。

用法：
  本地(Windows, 用 Edge):
    python scripts/download_pku.py --datasets D1 D2 --output data
  服务器(Ubuntu, 用 Chrome/Chromium):
    python scripts/download_pku.py --browser chrome --datasets D1 D2 ... --output data

依赖：pip install selenium （浏览器驱动由 selenium-manager 自动下载）
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

LINK = "https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E"
FILES = ("X_train.mat", "X_val.mat", "X_test.mat")


def make_driver(browser: str, download_dir: str):
    os.makedirs(download_dir, exist_ok=True)
    prefs = {
        "download.default_directory": str(Path(download_dir).resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    if browser == "chrome":
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_argument("--headless=new")
    else:  # edge
        from selenium.webdriver.edge.options import Options
        opts = Options()
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1400,900")
    opts.add_experimental_option("prefs", prefs)
    if browser == "chrome":
        return webdriver.Chrome(options=opts)
    return webdriver.Edge(options=opts)


def wait_new_file(download_dir: str, before: set, timeout: int = 7200):
    """轮询等待新文件下载完成（.crdownload 消失）并返回最终文件路径。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = set(glob.glob(os.path.join(download_dir, "*"))) - before
        for f in new:
            if not f.endswith(".crdownload"):
                return f
        time.sleep(3)
    return None


def click_name(driver, text: str):
    el = driver.find_element(
        By.XPATH,
        f"//div[contains(@class,'item-name') and normalize-space(text())='{text}']",
    )
    ActionChains(driver).move_to_element(el).click().perform()


def download_one(driver, download_dir: str, fname: str) -> bool:
    before = set(glob.glob(os.path.join(download_dir, "*")))
    name = driver.find_element(
        By.XPATH,
        f"//div[contains(@class,'item-name') and normalize-space(text())='{fname}']",
    )
    ActionChains(driver).context_click(name).perform()
    time.sleep(2)
    menus = driver.find_elements(
        By.XPATH,
        "//span[contains(@class,'as-controls-context-menu-label') "
        "and normalize-space(text())='下载']",
    )
    if not menus:
        print(f"  [x] {fname}: 未找到下载菜单项", flush=True)
        return False
    ActionChains(driver).move_to_element(menus[0]).click().perform()
    print(f"  [下载中] {fname} ...", flush=True)
    done = wait_new_file(download_dir, before)
    if done is None:
        print(f"  [x] {fname}: 下载超时", flush=True)
        return False
    print(f"  [完成] {os.path.basename(done)} ({os.path.getsize(done)/1e6:.1f} MB)", flush=True)
    return True


def go_back(driver):
    # 点面包屑里的 dataset4train 返回上级
    crumbs = driver.find_elements(By.XPATH, "//*[contains(@class,'breadcrumb')]//*[contains(text(),'dataset4train')]")
    if crumbs:
        ActionChains(driver).move_to_element(crumbs[0]).click().perform()
    else:
        # 退回：重新点 dataset4train 入口不可行，改为刷新重进
        driver.get(LINK)
        time.sleep(15)
        row = driver.find_elements(By.XPATH, "//div[contains(@class,'item-name')]")
        if row:
            ActionChains(driver).move_to_element(row[0]).click().perform()
    time.sleep(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[f"D{i}" for i in range(1, 17)])
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--browser", choices=("edge", "chrome"), default="edge")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已存在且非空的文件")
    args = ap.parse_args()

    staging = args.output / "_staging"
    os.makedirs(staging, exist_ok=True)
    driver = make_driver(args.browser, str(staging))
    try:
        driver.get(LINK)
        time.sleep(18)
        # 进入 dataset4train
        row = driver.find_elements(By.XPATH, "//div[contains(@class,'item-name')]")
        ActionChains(driver).move_to_element(row[0]).click().perform()
        time.sleep(10)

        for ds in args.datasets:
            outdir = args.output / ds
            print(f"\n=== {ds} ===", flush=True)
            click_name(driver, ds)
            time.sleep(10)
            for fname in FILES:
                target = outdir / fname
                if args.skip_existing and target.exists() and target.stat().st_size > 0:
                    print(f"  [跳过] {target} 已存在", flush=True)
                    continue
                outdir.mkdir(parents=True, exist_ok=True)
                ok = download_one(driver, str(staging), fname)
                if ok:
                    # 找到 staging 里的新文件并改名移动到 outdir
                    news = sorted(glob.glob(str(staging / "*")), key=os.path.getmtime, reverse=True)
                    if news:
                        src = news[0]
                        target = outdir / fname
                        if target.exists():
                            target.unlink()
                        os.rename(src, target)
                        print(f"  [保存] {target} ({target.stat().st_size/1e6:.1f} MB)", flush=True)
            go_back(driver)
    finally:
        driver.quit()
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()

"""从北大网盘(AnyShare)下载 D1-D16 训练数据 — V2（健壮版）。

修复了 V1 的两个 bug：
1. 导航改为「每个数据集重新打开链接」，避免 breadcrumb 返回失败；
2. 用 WebDriverWait 显式等待，替代固定 sleep；下载菜单用更宽松的选择器。
"""
from __future__ import annotations
import argparse, glob, os, time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

LINK = "https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E"
FILES = ("X_train.mat", "X_val.mat", "X_test.mat")


def wait_name(driver, text, timeout=30):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.XPATH, f"//div[contains(@class,'item-name') and normalize-space(text())='{text}']")
        )
    )


def open_and_enter(driver):
    """打开链接并进入 dataset4train，返回 D1-D16 列表页。"""
    driver.get(LINK)
    wait_name(driver, "dataset4train", 40)
    time.sleep(3)
    # 点击 dataset4train 文件夹进入
    row = driver.find_elements(By.XPATH, "//div[contains(@class,'item-name')]")
    ActionChains(driver).move_to_element(row[0]).click().perform()
    wait_name(driver, "D1", 40)
    time.sleep(3)


def download_one(driver, fname, staging, timeout=3600):
    before = set(glob.glob(os.path.join(staging, "*")))
    name = wait_name(driver, fname, 20)
    ActionChains(driver).context_click(name).perform()
    # 等待右键菜单出现，找"下载"
    menu = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located(
            (By.XPATH, "//*[contains(@class,'menu') or contains(@class,'context-menu')]//*[normalize-space(text())='下载']")
        )
    )
    # 更宽松：直接找可见的"下载"文本元素
    ActionChains(driver).move_to_element(menu).click().perform()
    print(f"  [下载中] {fname} ...", flush=True)
    # 等下载完成（staging 里出现完整文件，无 .crdownload）
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = set(glob.glob(os.path.join(staging, "*"))) - before
        for f in new:
            if not f.endswith(".crdownload"):
                return f
        time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[f"D{i}" for i in range(1, 17)])
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    staging = args.output / "_staging"
    os.makedirs(staging, exist_ok=True)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(staging.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })
    driver = webdriver.Chrome(options=opts)
    try:
        for ds in args.datasets:
            outdir = args.output / ds
            print(f"\n=== {ds} ===", flush=True)
            open_and_enter(driver)  # 每次重新进入
            try:
                wait_name(driver, ds, 30)
            except Exception:
                print(f"  [x] 找不到 {ds}，跳过", flush=True)
                continue
            d_el = driver.find_elements(
                By.XPATH, f"//div[contains(@class,'item-name') and normalize-space(text())='{ds}']"
            )
            ActionChains(driver).move_to_element(d_el[0]).click().perform()
            time.sleep(8)
            for fname in FILES:
                target = outdir / fname
                if args.skip_existing and target.exists() and target.stat().st_size > 0:
                    print(f"  [跳过] {target} 已存在", flush=True)
                    continue
                outdir.mkdir(parents=True, exist_ok=True)
                done = download_one(driver, fname, str(staging))
                if done:
                    os.rename(done, target)
                    print(f"  [保存] {target} ({target.stat().st_size/1e6:.1f} MB)", flush=True)
                else:
                    print(f"  [x] {fname} 下载失败", flush=True)
    finally:
        driver.quit()
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()

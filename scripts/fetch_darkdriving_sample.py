# -*- coding: utf-8 -*-
"""
从 DarkDriving 官方的 darkdriving_det.zip（OneDrive，7.56 GB）里只取少量样本，不下载整个包。

原理：zip 的文件目录在包的末尾，用 HTTP Range 请求只读目录和需要的那几个文件。
挑选规则：验证集里白天、夜间都有图且标注都非空的图像对，按文件名排序后等间隔取 N 对。

下载直链是临时的（带 tempauth，约 1 小时过期），需要从 OneDrive 分享页拿到后放进一个文本文件：
    python scripts\\fetch_darkdriving_sample.py --url-file D:\\tmp\\url.txt --num 10
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import io
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "DarkDriving_sample"
LOG_FILE = PROJECT_ROOT / "logs" / "fetch_darkdriving_sample.log"

# 包内路径 -> 本地子目录
PARTS = {
    "day": ("Det_data/day/images/val/", "Det_data/day/labels/val/"),
    "night": ("Det_data/night/images/val/", "Det_data/night/labels/val/"),
}
SNR_IMAGES = "Det_data/snr-aware/night/images/val/"   # 官方给的 SNR-Aware 增强结果，顺带取来做参照


class HttpRangeFile(io.RawIOBase):
    """把一个支持 Range 的 URL 包装成可 seek 的只读文件，带简单的预读缓存。"""

    def __init__(self, url, chunk=4 * 2 ** 20):
        self.url, self.chunk, self.pos = url, chunk, 0
        self.buf_start, self.buf = 0, b""
        self.bytes_fetched = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.bytes_fetched += len(data)
                return data
            except Exception as e:  # 网络抖动时重试
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        lo, hi = self.pos, self.pos + n
        if not (self.buf_start <= lo and hi <= self.buf_start + len(self.buf)):
            end = min(max(hi, lo + self.chunk), self.size) - 1
            self.buf_start, self.buf = lo, self._fetch(lo, end)
        data = self.buf[lo - self.buf_start: hi - self.buf_start]
        self.pos = hi
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-file", required=True, help="存有 darkdriving_det.zip 临时直链的文本文件")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    url = Path(args.url_file).read_text(encoding="utf-8").strip()
    out = Path(args.out)
    LOG_FILE.parent.mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')}  来源：DarkDriving 官方 OneDrive darkdriving_det.zip")
    fh = HttpRangeFile(url)
    log(f"远程包大小 {fh.size / 2 ** 30:.2f} GB")
    zf = zipfile.ZipFile(fh)
    info = {i.filename: i for i in zf.infolist()}

    def stems(img_prefix, lbl_prefix):
        """有图且标注非空的文件名（不含扩展名）。"""
        s = set()
        for name in info:
            if name.startswith(img_prefix) and name.endswith(".jpg"):
                stem = name[len(img_prefix):-4]
                lbl = info.get(f"{lbl_prefix}{stem}.txt")
                if lbl is not None and lbl.file_size > 0:
                    s.add(stem)
        return s

    common = sorted(stems(*PARTS["day"]) & stems(*PARTS["night"]))
    log(f"验证集里白天/夜间都有图且标注非空的图像对：{len(common)}")
    step = (len(common) - 1) / (args.num - 1)
    picks = [common[round(i * step)] for i in range(args.num)]

    for stem in picks:
        for part, (img_prefix, lbl_prefix) in PARTS.items():
            for sub, src in (("images", f"{img_prefix}{stem}.jpg"), ("labels", f"{lbl_prefix}{stem}.txt")):
                dst = out / part / sub / Path(src).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(zf.read(src))
        snr = f"{SNR_IMAGES}{stem}.jpg"
        if snr in info:
            dst = out / "snr_aware" / "images" / f"{stem}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(zf.read(snr))
        log(f"已取 {stem}")

    (out / "picked.txt").write_text("\n".join(picks) + "\n", encoding="utf-8")
    log(f"完成：{len(picks)} 对，实际下载 {fh.bytes_fetched / 2 ** 20:.1f} MB，保存到 {out}")


if __name__ == "__main__":
    main()

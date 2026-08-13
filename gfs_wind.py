#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_wind.py  ---  NOAA GFS(0.25 度)10m 風 -> wind.json 生成スクリプト

EAMS Lab / EAMS-App 用。日本全域(沖縄〜小笠原を含む)の地上10m 風 (U/V) を
NOAA NOMADS の GFS filter API から取得し、アプリが読み込む軽量な wind.json を作る。

- 取得元 : https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
- 変数   : UGRD / VGRD @ 10 m above ground
- 予報   : fh = 0,3,6,...,72 (3時間毎・25コマ)
- 出力   : カレントディレクトリの wind.json (アプリ側の想定フォーマット)

依存: Python 3.12 / numpy, xarray, cfgrib, eccodes / OS 側 libeccodes0
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# 日本全域の切り出し範囲(度)。0.25 グリッドの格子点に合わせた整数値。
#   西端 120E 〜 東端 156E : 沖縄(〜122E)〜 南鳥島(〜154E)を包含
#   南端  20N 〜 北端  48N : 沖ノ鳥島(〜20N)〜 北海道北端(〜46N)を包含
LEFTLON = 120.0
RIGHTLON = 156.0
TOPLAT = 48.0
BOTTOMLAT = 20.0

# 予報時刻(時間)。0〜72 を 3 時間刻み = 25 コマ。
FORECAST_HOURS = list(range(0, 73, 3))

# GFS の格子間隔(度)。0.25 度固定。
DX = 0.25
DY = 0.25

# JST(UTC+9)
JST = timezone(timedelta(hours=9))

# 出力ファイル名(リポジトリ直下)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(REPO_DIR, "wind.json")

# ---------------------------------------------------------------------------
# コマ分割出力(PERF-03 / 2026-08-13)
#
# なぜ分けるか:
#   wind.json は 25 コマを 1 本に束ねているため 3.6 MB(圧縮後 874 KB)ある。
#   アプリが最初に描くのは 1 コマだけで、残りは「スライダーを動かすかもしれない」
#   ために先払いしていた。さらにアプリはこれを localStorage に丸ごと保存するので
#   3.40 M 文字(上限 5 M 文字前後)を占領し、飛行記録の保存を圧迫していた。
#   1 コマだけなら 139 KB = 全体の 4.0%。
#
# 出し方:
#   wind-index.json      … grid + 各コマの (fh, t, file)。1 KB 程度。
#   frames/fNNN.json     … 1 コマ分 {fh, t, u, v}。これ単体で描ける。
#
# ★wind.json も従来どおり出し続ける。旧版のアプリ(eams-v525 まで)がこれを読むため。
#   アプリが分割版へ移り、利用者の更新が行き渡ってから外す。
# ---------------------------------------------------------------------------
FRAMES_DIR = os.path.join(REPO_DIR, "frames")
INDEX_PATH = os.path.join(REPO_DIR, "wind-index.json")
FRAME_NAME = "f%03d.json"          # fh を 3 桁ゼロ詰め(f000, f003, ... f072)
INDEX_VER = 1                      # 形を変えたら上げる(アプリ側が見て判断できるように)

# ダウンロード時の設定
HTTP_TIMEOUT = 180          # 秒
DOWNLOAD_RETRIES = 3        # 1コマあたりの再試行回数(ネットワーク一時障害向け)
RETRY_WAIT = 10             # 秒(再試行間隔)
USER_AGENT = "eams-wind-bot/1.0 (+https://eamslab.com; claude@eamslab.com)"

# GRIB の変数名は環境(eccodes の版)で揺れるため両対応にする。
U_NAMES = ["u10", "u", "10u", "UGRD"]
V_NAMES = ["v10", "v", "10v", "VGRD"]


# ---------------------------------------------------------------------------
# run(初期時刻)の選択
# ---------------------------------------------------------------------------

def candidate_runs(now_utc: datetime) -> list[datetime]:
    """直近の GFS run から古い方向へ 6 時間刻みで候補を並べて返す。

    GFS は 00/06/12/18Z サイクルで、各 run は初期時刻の +4〜5 時間後に公開される。
    そこで「現在時刻の 5 時間前」を起点に、直近で公開済みと期待できる run を推定する。
    (例: cron 10:25Z → 5h 前 = 05:25Z → 直近 run = 当日 00Z。00Z は 04〜05Z に公開済み)

    未公開の可能性に備え、そこから 6 時間ずつ最大 24 時間さかのぼった計 5 候補を
    新しい順に返す(呼び出し側は新しい順に試し、最初に取得できた run を採用する)。
    """
    anchor = now_utc - timedelta(hours=5)
    run_hour = (anchor.hour // 6) * 6
    base_run = anchor.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    # 0, 6, 12, 18, 24 時間前 = 5 候補(24 時間ぶんさかのぼる)
    return [base_run - timedelta(hours=back) for back in range(0, 25, 6)]


# ---------------------------------------------------------------------------
# ダウンロード / GRIB 読み取り
# ---------------------------------------------------------------------------

def build_url(run_dt: datetime, fh: int) -> str:
    """指定 run / 予報時刻の GRIB2(サブリージョン)を取得する URL を組み立てる。"""
    ymd = run_dt.strftime("%Y%m%d")
    hh = run_dt.strftime("%H")
    fff = f"{fh:03d}"
    params = {
        "dir": f"/gfs.{ymd}/{hh}/atmos",
        "file": f"gfs.t{hh}z.pgrb2.0p25.f{fff}",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "lev_10_m_above_ground": "on",
        # サブリージョン切り出し(bottomlat 等に値を入れると有効化される)
        "subregion": "",
        "leftlon": f"{LEFTLON:g}",
        "rightlon": f"{RIGHTLON:g}",
        "toplat": f"{TOPLAT:g}",
        "bottomlat": f"{BOTTOMLAT:g}",
    }
    return BASE_URL + "?" + urllib.parse.urlencode(params)


def download_grib(url: str) -> bytes | None:
    """GRIB2 を取得。成功時は bytes、未公開/失敗時は None を返す。

    NOMADS は未公開ファイルに対して小さな HTML エラーページを返すことがあるため、
    先頭 4 バイトが GRIB マジックであることを確認して真偽を判定する。
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                if resp.status != 200:
                    return None
                data = resp.read()
        except Exception as exc:  # ネットワーク一時障害・404 など
            print(f"    [warn] download attempt {attempt} failed: {exc}", flush=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(RETRY_WAIT)
            continue
        # 内容が本物の GRIB か検証(未公開時は HTML が返る)
        if len(data) < 100 or not data.startswith(b"GRIB"):
            return None
        return data
    return None


def pick_var(ds: "xr.Dataset", names: list[str]) -> "xr.DataArray":
    """候補名の順に見て最初に存在する変数を返す(u10/u, v10/v の揺れに対応)。"""
    for n in names:
        if n in ds.data_vars:
            return ds[n]
    raise KeyError(f"none of {names} found in {list(ds.data_vars)}")


def read_frame(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GRIB2 bytes を読み、(u2d, v2d, lat, lon) を返す。

    緯度は北→南(降順)、経度は西→東(昇順)に整列した 2 次元配列(ny, nx)。
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gfs.grib2")
        with open(path, "wb") as f:
            f.write(data)
        # indexpath="" で .idx の生成を抑止(再実行時の残骸・権限問題を避ける)
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        try:
            u = pick_var(ds, U_NAMES)
            v = pick_var(ds, V_NAMES)
            # 北→南・西→東に整列(行優先で NW 始点にするため)
            u = u.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
            v = v.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
            lat = np.asarray(u["latitude"].values, dtype=float)
            lon = np.asarray(u["longitude"].values, dtype=float)
            u2d = np.asarray(u.values, dtype=float)
            v2d = np.asarray(v.values, dtype=float)
        finally:
            ds.close()
    return u2d, v2d, lat, lon


def flatten_round(arr2d: np.ndarray) -> list[float]:
    """2 次元(ny, nx)を行優先で平坦化し、小数 1 桁に丸めた list を返す。

    NaN は 0.0 に置換(サブリージョン全域で 10m 風は必ず値を持つが保険)。
    -0.0 を +0.0 に正規化して JSON を短くする。
    """
    a = np.nan_to_num(np.asarray(arr2d, dtype=float), nan=0.0)
    a = np.round(a, 1) + 0.0  # +0.0 で -0.0 を正規化
    return a.ravel(order="C").tolist()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def build_frames(run_dt: datetime) -> tuple[list[dict], dict | None]:
    """指定 run について全予報コマを取得。frames と grid を返す。

    f000 が取得できなければ「未公開」とみなし ([], None) を返す(呼び出し側で次の
    (古い)run に切り替える)。f000 が取れれば以降のコマは取れたぶんだけ採用する
    (公開途中で末尾コマが未生成でも、取れたところまでで作る)。
    """
    frames: list[dict] = []
    grid: dict | None = None

    for i, fh in enumerate(FORECAST_HOURS):
        url = build_url(run_dt, fh)
        data = download_grib(url)
        if data is None:
            if i == 0:
                # 初期コマ(f000)が無い = この run は未公開。上位で次候補へ。
                return [], None
            # 途中コマが未公開: ここで打ち切り(取れたぶんで確定)
            print(f"    [info] f{fh:03d} not available yet; stop at {len(frames)} frames", flush=True)
            break

        u2d, v2d, lat, lon = read_frame(data)

        if grid is None:
            ny, nx = u2d.shape
            grid = {
                "nx": int(nx),
                "ny": int(ny),
                "lo1": round(float(lon.min()), 3),  # 西端
                "la1": round(float(lat.max()), 3),  # 北端
                "lo2": round(float(lon.max()), 3),  # 東端
                "la2": round(float(lat.min()), 3),  # 南端
                "dx": DX,
                "dy": DY,
            }

        valid_utc = run_dt + timedelta(hours=fh)
        t_jst = valid_utc.astimezone(JST).strftime("%Y-%m-%dT%H:%M")

        frames.append({
            "fh": int(fh),
            "t": t_jst,
            "u": flatten_round(u2d),
            "v": flatten_round(v2d),
        })
        print(f"    [ok] f{fh:03d} -> {t_jst} JST", flush=True)

    return frames, grid


def write_split(grid: dict, frames: list[dict], run_dt: datetime) -> tuple[int, int]:
    """コマ別ファイルと索引を書く。戻り値は (書いたコマ数, 索引のバイト数)。

    ★前回より コマ数が減ったとき(公開途中で末尾が未生成だった等)、古いコマの
      ファイルを残さない。残すと索引に無いコマが CDN に居座り、次に増えたときに
      古い予報が混ざる。索引に載っていないファイルは消す。
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    keep: set[str] = set()
    entries: list[dict] = []
    for fr in frames:
        name = FRAME_NAME % int(fr["fh"])
        keep.add(name)
        one = {"fh": int(fr["fh"]), "t": fr["t"], "u": fr["u"], "v": fr["v"]}
        text = json.dumps(one, separators=(",", ":"), ensure_ascii=False)
        with open(os.path.join(FRAMES_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)
        entries.append({"fh": int(fr["fh"]), "t": fr["t"], "file": "frames/" + name})

    removed = 0
    for old in sorted(os.listdir(FRAMES_DIR)):
        if old.endswith(".json") and old not in keep:
            os.remove(os.path.join(FRAMES_DIR, old))
            removed += 1
    if removed:
        print(f"    [info] removed {removed} stale frame file(s)", flush=True)

    index = {
        "ver": INDEX_VER,
        "run": run_dt.astimezone(JST).strftime("%Y-%m-%dT%H:%M"),   # 予報の初期時刻(JST)
        "grid": grid,
        "frames": entries,
    }
    itext = json.dumps(index, separators=(",", ":"), ensure_ascii=False)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(itext)
    return len(entries), len(itext.encode("utf-8"))


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    print(f"[gfs_wind] now(UTC)={now_utc:%Y-%m-%dT%H:%M}Z", flush=True)

    for run_dt in candidate_runs(now_utc):
        print(f"[gfs_wind] try run {run_dt:%Y-%m-%d %H}Z ...", flush=True)
        frames, grid = build_frames(run_dt)
        if frames and grid is not None:
            out = {"grid": grid, "frames": frames}
            # 詰めた JSON(区切りに空白を入れない)
            text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write(text)
            size_mb = len(text.encode("utf-8")) / 1024 / 1024
            print(
                f"[gfs_wind] OK: run {run_dt:%Y-%m-%d %H}Z, "
                f"{len(frames)} frames, grid {grid['nx']}x{grid['ny']}, "
                f"{size_mb:.2f} MB -> {OUT_PATH}",
                flush=True,
            )
            # コマ別＋索引も出す(アプリは初回にこの 2 本だけ読む)
            n_split, idx_bytes = write_split(grid, frames, run_dt)
            one_kb = len(json.dumps({"fh": frames[0]["fh"], "t": frames[0]["t"],
                                     "u": frames[0]["u"], "v": frames[0]["v"]},
                                    separators=(",", ":")).encode("utf-8")) / 1024
            print(
                f"[gfs_wind] split: {n_split} files -> frames/, "
                f"index {idx_bytes} B, 1 frame {one_kb:.0f} KB "
                f"({one_kb / (size_mb * 1024) * 100:.1f}% of wind.json)",
                flush=True,
            )
            return 0
        print(f"[gfs_wind] run {run_dt:%Y-%m-%d %H}Z not ready; going older.", flush=True)

    # どの run からも 1 コマも作れなかった → 壊れた wind.json を置かず異常終了。
    # (前回の正常な wind.json を残す方が安全)
    print("[gfs_wind] ERROR: no frames could be built from any candidate run.", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

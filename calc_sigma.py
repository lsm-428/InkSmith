"""打印模拟器参数计算器：喷头数据库 + 实测铺展 -> film 模式 σ。

数据出处：InteractivePrintingSimulator/printheads.json（喷头规格库）
物理链：drop volume pL -> 球形直径 -> 实测铺展直径 -> 高斯等效 σ=直径/4
      -> σ[px] = σ[µm] * dpi / 25400

用法：
  uv run python calc_sigma.py                      # 默认 KM1024iSHE，铺展 50µm
  uv run python calc_sigma.py --spread-um 45 --dpi 1080
"""
import argparse
import json
from pathlib import Path

HEADS_DB = Path(__file__).parent.parent / "InteractivePrintingSimulator" / "printheads.json"


def sphere_diameter_um(volume_pl):
    """pL 液滴的球形直径（µm）。1 pL = 1e-12 L = 1e3 µm³。"""
    return (6 * volume_pl * 1e3 / 3.141592653589793) ** (1 / 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="KM1024iSHE")
    p.add_argument("--spread-um", type=float, default=50.0,
                   help="实测墨滴在基材上的铺展直径（µm）")
    p.add_argument("--dpi", type=int, default=None, help="只看这一个 dpi")
    args = p.parse_args()

    # 文件是 JS 赋值格式：printheadDatabase = [...]
    text = HEADS_DB.read_text()
    text = text[text.index("["):text.rindex("]") + 1]
    heads = json.loads(text)
    head = next((h for h in heads if h.get("Model") == args.head), None)
    if head is None:
        raise SystemExit(f"数据库 {HEADS_DB} 里找不到喷头 {args.head}")

    vol = float(head["drop volume pL"])
    d_sphere = sphere_diameter_um(vol)
    sigma_um = args.spread_um / 4  # 高斯等效：99.7% 能量在 6σ ≈ 1.5×直径 内
    dot_gain = args.spread_um / d_sphere

    print(f"喷头：{args.head}（{head['Manufacturer']}）")
    print(f"  墨滴体积 {vol} pL -> 球形直径 {d_sphere:.1f} µm")
    print(f"  实测铺展直径 {args.spread_um} µm -> dotGain(直径比) ≈ {dot_gain:.2f}")
    print(f"  高斯等效 σ = 铺展直径/4 = {sigma_um:.2f} µm")
    print(f"  基材：非吸收膜，100% 固含 -> 线性体积模型（FilmSimulator，无 exp 饱和）")
    print()
    dpis = [args.dpi] if args.dpi else [508, 720, 1080, 1440, 2880, 5000]
    print(f"{'dpi':>6} {'像素µm':>8} {'σ[px]':>8} {'铺展直径[px]':>12}")
    for dpi in dpis:
        um_px = 25400 / dpi
        print(f"{dpi:>6} {um_px:>8.2f} {sigma_um / um_px:>8.3f} "
              f"{args.spread_um / um_px:>12.2f}")
    print()
    d = args.dpi or 720
    print("训练命令（把 --dpi 换成实际打印分辨率）：")
    print(f"  uv run python train.py --printer film --sigma-um {sigma_um:.2f} "
          f"--dpi {d} --out runs/film")


if __name__ == "__main__":
    main()

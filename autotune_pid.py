"""autotune_pid.py — LU-926U 温控器 PID 自整定 (Modbus 远程触发)

流程: 恢复出厂控制参数 → 设 SV → crL=4 归位 → crL=2 (Aut) 触发
→ 位式振荡 3 周期 (面板显示 At) → 仪表自动转 crL=4 并写入新 P1/P2/rt
→ 读回对比 → heat_off → 参数存档 JSON

安全: PV>=60°C 硬停 / 超时中止 / Ctrl+C / 面板干预检测 / 通信故障检测,
任何退出路径 (含正常完成) 一律 heat_off。

用法 (串口独占, 须先停 uvicorn):
  python autotune_pid.py                 # 恢复默认 → SV=40°C 整定 → 存档
  python autotune_pid.py --sv 36 --twice # 指定温度 + 连续两次
  python autotune_pid.py --dry-run       # 只读参数不写, 验证连通
  python autotune_pid.py --apply measurements\\pid_params_xxx.json  # 从存档恢复
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from app.config import MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE, AGENT_TEMP_LIMIT
from app.controller.modbus_temp import TempController, REGISTERS, FACTORY_DEFAULTS

MEAS_DIR = Path(__file__).resolve().parent / "measurements"
CRL_NAME = {1: "onF位式", 2: "Aut自整定", 3: "MAn手动", 4: "Pid智能", 5: "Pad", 6: "oPi"}


def fmt_params(p: dict) -> str:
    return (f"P1={p.get('P1')} P2={p.get('P2')} rt={p.get('rt')} "
            f"cHy={p.get('cHy')} ctL={p.get('ctL')} "
            f"crL={CRL_NAME.get(p.get('crL'), p.get('crL'))} SV={p.get('SV')}")


def run_one_tune(ctrl: TempController, sv: float, timeout_s: float,
                 csv_writer, csv_file, t0: float) -> dict | None:
    """单轮自整定。返回整定后参数快照; 失败/中止返回 None。"""
    # 触发 (SV 已由调用方设好)
    if not ctrl.start_autotune():
        print("[FAIL] 未能进入自整定态 (crL 未变 2), 中止")
        return None
    print(f"[At] 自整定已启动 (SV={sv}°C), 位式振荡 3 周期, 预计 15~40 分钟...")
    print("     面板下排应显示 At; 按 Ctrl+C 可随时安全中止 (参数不会被更新)")

    last_pv_ok = time.time()
    while True:
        time.sleep(2.0)
        pv = ctrl.get_pv()
        mv = ctrl.get_mv()
        crl = ctrl.read_reg(REGISTERS['crL'])
        now = time.time()

        if pv is not None:
            last_pv_ok = now
            # 安全: 超温硬停
            if pv >= AGENT_TEMP_LIMIT:
                print(f"[SAFETY] PV={pv:.1f}°C >= {AGENT_TEMP_LIMIT}°C, 硬停!")
                return None
        elif now - last_pv_ok > 10:
            print("[FAIL] 温度连续 10s 读不到, 判通信故障, 中止")
            return None

        csv_writer.writerow([f"{now:.1f}", f"{now - t0:.1f}",
                             f"{pv if pv is not None else ''}",
                             f"{mv if mv is not None else ''}", crl])
        csv_file.flush()
        el = (now - t0) / 60
        print(f"  t={el:5.1f}min  PV={pv if pv is not None else '--':>5}°C  "
              f"MV={mv if mv is not None else '--':>5}%  "
              f"crL={CRL_NAME.get(crl, crl)}")

        if crl == 4:
            print("[DONE] crL 已自动转 Pid — 自整定完成")
            return ctrl.read_pid_params()
        if crl is not None and crl != 2:
            print(f"[ABORT] crL={CRL_NAME.get(crl, crl)} (非 At/Pid), "
                  f"疑似面板人工干预, 中止 (参数未更新)")
            return None
        if now - t0 > timeout_s:
            print(f"[TIMEOUT] 超过 {timeout_s/60:.0f} 分钟未完成, 中止")
            return None


def apply_archive(ctrl: TempController, path: str) -> int:
    """从存档 JSON 把控制参数写回设备 (设备复位/换表后恢复)"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    after = data.get("after") or {}
    fails = []
    for n in ('P1', 'P2', 'rt', 'cHy', 'ctL'):
        v = after.get(n)
        if v is None:
            continue
        ok = ctrl.write_reg(REGISTERS[n], int(v))
        time.sleep(0.05)
        rb = ctrl.read_reg(REGISTERS[n])
        status = "OK" if (ok and rb == int(v)) else "FAIL"
        if status == "FAIL":
            fails.append(n)
        print(f"  {n:4s} <- {v:>5}  回读={rb}  [{status}]")
    print("恢复完成" if not fails else f"失败项: {fails}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="LU-926U PID 自整定")
    ap.add_argument("--sv", type=float, default=40.0,
                    help="整定温度°C (默认 40, 实验区间 30~50 常用中点)")
    ap.add_argument("--twice", action="store_true",
                    help="连续整定两次 (第二轮从已到温状态起整, 更准)")
    ap.add_argument("--timeout", type=float, default=45.0, help="单轮超时(分钟)")
    ap.add_argument("--dry-run", action="store_true", help="只读参数不写任何寄存器")
    ap.add_argument("--skip-restore", action="store_true", help="跳过恢复出厂默认步骤")
    ap.add_argument("--apply", type=str, default=None, help="从存档 JSON 恢复参数后退出")
    args = ap.parse_args()

    if not (25.0 <= args.sv <= 60.0):
        print(f"[ERR] --sv {args.sv} 超出合理整定范围 25~60°C")
        return 1

    ctrl = TempController(MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE)
    if not ctrl.open():
        print(f"[FAIL] 串口 {MODBUS_PORT} 打不开 (uvicorn/GUI 在跑? 先停掉)")
        return 1
    wrote = False   # 是否写过寄存器 (dry-run 保持真正只读, finally 不 heat_off)

    try:
        before = ctrl.read_pid_params()
        pv = ctrl.get_pv()
        if before.get('P1') is None or pv is None:
            print("[FAIL] 温控器读不到参数, 检查连接")
            return 1
        print(f"[前] PV={pv}°C  {fmt_params(before)}")

        if args.dry_run:
            print("[dry-run] 不写任何寄存器, 退出")
            return 0
        wrote = True
        if args.apply:
            return apply_archive(ctrl, args.apply)

        # 1. 恢复出厂控制参数 (干净起点; SV 不恢复 — 出厂 80°C 危险)
        if not args.skip_restore:
            fails = ctrl.restore_factory_control_params()
            if fails:
                print(f"[FAIL] 恢复出厂默认失败项: {fails}, 中止")
                return 1
            print(f"[恢复默认] {FACTORY_DEFAULTS} 全部写入并校验通过")

        # 2. 设整定温度
        if not ctrl.set_sv(args.sv):
            print("[FAIL] SV 写入失败, 中止")
            return 1
        print(f"[SV] 整定温度已设为 {args.sv}°C")

        # 3. 整定 (可选两轮), 全程记录 CSV
        MEAS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = MEAS_DIR / f"autotune_{stamp}.csv"
        result = None
        n_rounds = 2 if args.twice else 1
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_epoch", "t_rel", "PV", "MV_pct", "crL"])
            for i in range(n_rounds):
                if n_rounds > 1:
                    print(f"===== 第 {i+1}/{n_rounds} 轮 =====")
                result = run_one_tune(ctrl, args.sv, args.timeout * 60, w, f,
                                      time.time())
                if result is None:
                    break

        if result is None:
            print("[结果] 整定未完成, 仪表参数未更新")
            return 1

        # 4. 对比 + 存档
        print(f"[后] {fmt_params(result)}")
        print(f"[对比] P1: {before['P1']} -> {result['P1']}   "
              f"P2: {before['P2']} -> {result['P2']}   "
              f"rt: {before['rt']} -> {result['rt']}")
        if (result['P1'], result['P2'], result['rt']) == (116, 71, 353):
            print("[警告] 整定后参数仍等于出厂默认 — 可能整定未真正生效, 建议复查")
        archive = {
            "device": "LU-926U", "port": MODBUS_PORT,
            "time": datetime.now().isoformat(),
            "sv_tuned": args.sv, "rounds": n_rounds,
            "before": before, "after": result,
            "monitor_csv": csv_path.name,
        }
        json_path = MEAS_DIR / f"pid_params_{stamp}.json"
        json_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"[存档] {json_path}")
        print(f"[监控] {csv_path}")
        return 0

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 用户中止, 整定未完成, 参数未更新")
        return 1
    finally:
        # 任何写过寄存器的退出路径 (含正常完成) 一律断加热回安全态;
        # 整定出的参数已存仪表 EEPROM, heat_off 不影响
        try:
            if wrote:
                ctrl.heat_off()
                time.sleep(0.3)
                print(f"[安全] 已 heat_off (手动模式/输出0%), MV={ctrl.get_mv()}%")
        except Exception:
            pass
        ctrl.close()


if __name__ == "__main__":
    sys.exit(main())

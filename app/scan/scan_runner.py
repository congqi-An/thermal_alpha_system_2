"""自动扫描 — 三种模式 (温度步进 / N步进 / 即时计数)

temp   温度步进: 首段升温至 T_start, PV>=T_start 稳定 3s 确认进段后才开计数;
       段内 SV=T2 升温, PV>=T2 稳定 5s 立即停计数结算 (防保温吞吐虚计),
       后续段无缝滚入 (上段终点=本段起点, 已确认到温, 不再重新等确认)。
       α_seg = N·λ/2/(L0·dT)。
fringe N步进: 升温至 T_start 确认进段 -> 连续升温到 T_end, 每累计 n_step 条
       条纹结算一段 (dT 取段内实测 PV 区间), 段数由实际温升决定。
free   即时计数: 不控温, 启动即计数, 用户点停止时结算单段 (N/dT/α)。
"""
import time, threading, csv
from datetime import datetime

from .. import state
from ..config import (SCAN_T_START, SCAN_T_END, SCAN_STEP, SCAN_STABILIZE_S,
                      SCAN_STABLE_BAND, SCAN_ENTER_CONFIRM_S, MEASURE_SAVE_DIR,
                      REFERENCE_ALPHA, LASER_WAVELENGTH_M, SAMPLE_LENGTH_M)
from ..measurement.alpha_calc import alpha_from_N, alpha_stats, error_vs_reference

# 理论条纹速率 (条/°C): N/ΔT = 2·α_ref·L0/λ ≈ 9.86 (黄铜 H62, L0=150mm, λ=632.8nm)
# 合理性自检阈值: 偏离理论值超 ±40% 判为异常段 (扫描中断/温控未动/视频片段等),
# 标 valid=False 并从 α 均值剔除 (2026-08-07 审计补充, 防垃圾段静默进报告)
_N_PER_DEGC = 2 * REFERENCE_ALPHA * 1e-6 * SAMPLE_LENGTH_M / LASER_WAVELENGTH_M
_N_RATE_TOL = 0.40


class ScanRunner:
    def __init__(self):
        self.T_start = SCAN_T_START; self.T_end = SCAN_T_END
        self.step = SCAN_STEP; self.stabilize_s = SCAN_STABILIZE_S
        self.stable_band = SCAN_STABLE_BAND
        self.mode = 'temp'       # temp=温度步进 / fringe=N步进 / free=即时计数
        self.n_step = 20         # fringe 模式: 每段条纹数
        self._segs = []
        self.segments = []
        self.current_seg = 0; self.total_seg = 0
        self.current_sv = 0.0
        self.phase = 'idle'      # idle/heating/stabilizing/measuring/done/stopped
        self.message = ''
        self.csv_path = None
        self._running = False
        self._thread = None

    def configure(self, T_start=None, T_end=None, step=None, stabilize_s=None,
                  mode=None, n_step=None):
        if T_start is not None: self.T_start = T_start
        if T_end is not None: self.T_end = T_end
        if step is not None: self.step = step
        if stabilize_s is not None: self.stabilize_s = stabilize_s
        if mode in ('temp', 'fringe', 'free'): self.mode = mode
        if n_step is not None: self.n_step = max(1, int(n_step))
        # 参数校验: step≤0 会死循环 (t 永不超过 T_end); T_end≤T_start 零段假完成
        if self.step <= 0 or self.T_end <= self.T_start:
            self._segs = []
            self.total_seg = 0
            return {'mode': self.mode, 'T_start': self.T_start,
                    'T_end': self.T_end, 'step': self.step,
                    'stabilize_s': self.stabilize_s, 'n_step': self.n_step,
                    'total_segments': 0, 'segments': [],
                    'error': '参数无效: 需 step>0 且 T_end>T_start'}
        segs = []
        t = self.T_start
        while t + 1e-6 < self.T_end:
            segs.append((t, min(t + self.step, self.T_end)))
            t += self.step
        self._segs = segs
        if self.mode == 'fringe':
            # 段数估计: 理论条纹率 N/°C = 2αL0/λ ≈ 9.86 (仅供进度条, 实际段数由温升决定)
            n_per_degc = 2 * REFERENCE_ALPHA * 1e-6 * SAMPLE_LENGTH_M / LASER_WAVELENGTH_M
            self.total_seg = max(1, round((self.T_end - self.T_start) * n_per_degc / self.n_step))
        elif self.mode == 'free':
            self.total_seg = 1
        else:
            self.total_seg = len(segs)
        return {'mode': self.mode, 'T_start': self.T_start, 'T_end': self.T_end,
                'step': self.step, 'stabilize_s': self.stabilize_s,
                'n_step': self.n_step, 'total_segments': self.total_seg,
                # 每段温度区间 (供 LLM 叙事/前端展示, 避免推算错误)
                'segments': [[round(a, 1), round(b, 1)] for a, b in segs]}

    def start(self):
        if self._running:
            return False
        # 防竞态: 上一轮线程尚未退出时不允许重启 (否则双线程并发写 segments/CSV)
        if self._thread is not None and self._thread.is_alive():
            return False
        self.configure()
        self._running = True
        self.segments = []; self.current_seg = 0; self.phase = 'heating'
        self.t_scan_start = time.time()   # 报告图表时间轴原点
        state.n_history = []              # 整轮 N/T 时序 (段内计数器 reset 不清空)
        state.disp_history = []           # 稳定度/振动时序 (诊断振动图)
        state._last_disp_t = 0.0
        self.csv_path = MEASURE_SAVE_DIR / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        target = {'fringe': self._run_fringe, 'free': self._run_free}.get(self.mode, self._run)
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        self.phase = 'stopped'

    def _wait_pv_enter(self, target, confirm_s=SCAN_ENTER_CONFIRM_S, timeout=900):
        """进段等待: 升温至 PV >= target (真正到达及以上) 并保持 confirm_s 秒
        才算进段。进段前不开计数 — 否则首段从当前 PV(可能是室温)就开始
        累计条纹, 计数覆盖区间与标称温度段不一致。
        判据 PV >= target (不带 band 容差): 用户要求"真正到达设定值及以上
        一点"才算, PID 整定后稳态 ±0.2°C 能稳定骑在 target 上方。"""
        if state.temp_ctrl is None or not state.temp_ctrl.online:
            return True   # 演示模式: 无温控, 直接进段
        self.phase = 'heating'; self.current_sv = target
        state.temp_ctrl.regulate_to(target)   # set_sv + heat_pid(crL=4)
        t0 = time.time()
        stable_since = None
        while self._running and time.time() - t0 < timeout:
            # 串口掉线: 立即返回, 不干等 900s (PV 冻结无法判断)
            if not state.temp_ctrl.online:
                return False
            pv = state.current_T
            # 进段判定: 设定温度 ±0.1°C (与段末 _wait_pv_reach 一致)
            # +1e-9 ε: 浮点表示 40.1-40.0=0.1000000000000014 > 0.1, 会误拒边界
            if abs(pv - target) <= 0.1 + 1e-9:
                if stable_since is None:
                    stable_since = time.time()
                if time.time() - stable_since >= confirm_s:
                    return True
            else:
                stable_since = None
            time.sleep(0.5)
        return False

    def _wait_pv_reach(self, target, timeout=900):
        """升温到 PV >= target 并保持 stabilize_s (默认 5s) 即返回。
        期间 grab 线程 measure_active 累计 N。到温确认后立即结算,
        不再长时间保温等待 — 保温期 PID 微波动使条纹吞吐往复,
        振荡计数不辨方向会虚增 N。
        用 PID 模式(crL=4)+SV 驱动, 不用 heat_on 手动 bang-bang —
        说明书: 手动模式(crL=3)忽略 SV, 全功率持续加热必过冲。"""
        # 温控离线时快速返回, 不空转 15 分钟
        if state.temp_ctrl is None or not state.temp_ctrl.online:
            self.phase = 'measuring'
            self.message = '温控离线 — 演示模式, 跳过升温等待'
            time.sleep(1)   # 给 grab 线程一点时间累计帧
            return True
        self.phase = 'stabilizing'; self.current_sv = target
        state.temp_ctrl.regulate_to(target)   # set_sv + heat_pid(crL=4)
        t0 = time.time()
        stable_since = None
        while self._running and time.time() - t0 < timeout:
            # 串口掉线: 立即返回, 不干等 900s (PV 冻结无法判断)
            if not state.temp_ctrl.online:
                return False
            pv = state.current_T
            # 到温判定: 设定温度 ±0.1°C 双向容差 (PID 稳态 ±0.2 可能小幅
            # 回摆, 严格 pv>=target 会让 stable_since 反复清零, 卡住不进下阶段)
            # +1e-9 ε: 浮点表示 40.1-40.0=0.1000000000000014 > 0.1, 会误拒边界
            if abs(pv - target) <= 0.1 + 1e-9:
                if stable_since is None:
                    stable_since = time.time()
                    self.phase = 'measuring'
                if time.time() - stable_since >= self.stabilize_s:
                    return True
            else:
                stable_since = None
                self.phase = 'stabilizing'
            self._sample_n()
            time.sleep(0.5)
        return False

    def _settle_segment(self, w, f, seg_no, T1, T2, dT, T1_pv=None, valid=True):
        """段结算: 取 N -> α -> 记录/写 CSV。
        三种扫描模式共用。dT < 0.05°C 时 α 记 0 (防除零/虚高)。
        valid=False: 段未正常完成 (到温超时/中途异常/合理性自检不过),
        不参与 α 均值统计。
        返回 (N_avg, alpha_seg)。"""
        _, N_per = state.fringe_counter.total_N
        N_avg = state.gated_N_abs          # N 振动门 (惯性覆盖) 后的稳定值
        alpha_seg = alpha_from_N(N_avg, dT) if dT >= 0.05 else 0.0
        # 合理性自检: N/ΔT 应 ≈ 理论速率 (9.86 条/°C), 偏离 >±40% 判异常
        implausible = False
        if dT >= 0.05 and N_avg > 0:
            n_rate = N_avg / dT
            if abs(n_rate - _N_PER_DEGC) / _N_PER_DEGC > _N_RATE_TOL:
                implausible = True
                print(f"  [scan] 警告: 段{seg_no} N/ΔT={n_rate:.2f} 条/°C "
                      f"偏离理论 {_N_PER_DEGC:.2f}±{_N_RATE_TOL * 100:.0f}% "
                      f"(N={N_avg:.1f}, ΔT={dT:.2f}) — 标为异常段, 不计入 α 均值")
        rec = {'seg': seg_no, 'T1': round(T1, 2), 'T2': round(T2, 2), 'N': N_avg,
               'N_per': N_per, 'dT': round(dT, 2), 'alpha': round(alpha_seg, 2),
               'T1_pv': round(T1_pv, 2) if T1_pv is not None else None,
               'valid': bool(valid) and not implausible,
               'flag': 'implausible' if implausible else None,
               't2': round(time.time(), 1)}   # 段结算时刻 (报告图表段边界线)
        self.segments.append(rec)
        w.writerow([datetime.now().isoformat(), round(T2, 2), str(N_per), N_avg,
                    seg_no, round(dT, 2), round(alpha_seg, 3)])
        f.flush()
        return N_avg, alpha_seg

    def _sample_n(self):
        """报告图表: 记录 N(t) 时序 (仅测量期, 上限防内存膨胀)"""
        if state.measure_active and state.fringe_counter is not None:
            state.n_history.append((time.time(), state.gated_N_abs))
            if len(state.n_history) > 20000:
                del state.n_history[:1000]

    def _run(self):
        state.scan_active = True
        f = open(self.csv_path, 'w', newline='', encoding='utf-8')
        w = csv.writer(f)
        w.writerow(['time', 'T_end', 'N_per_r0', 'N_avg', 'seg', 'dT', 'alpha_seg'])
        try:
            aborted = False
            for i, (T1, T2) in enumerate(self._segs):
                if not self._running:
                    break
                self.current_seg = i + 1
                # 仅首段进段确认: PV>=T_start 稳定 3s 才开计数;
                # 后续段无缝滚入 (上段终点=本段起点, 刚确认过 PV>=T1,
                # 再等确认只会白让保温吞吐污染下一段起点)
                if i == 0:
                    self.message = f'段 {i+1}/{self.total_seg}: 升温至起点 {T1}°C (到达后确认3s)'
                    if not self._wait_pv_enter(T1):
                        if not self._running:
                            break
                        self.message = f'段 {i+1}: 升温至起点 {T1}°C 超时, 扫描中止'
                        aborted = True
                        break
                self.message = f'段 {i+1}/{self.total_seg}: {T1}→{T2}°C 升温稳定中'
                # 记录段起点实测 PV (计数覆盖的真实温升区间从这里开始,
                # 首段实际起温可能低于标称 T1, 用标称 dT 会使 α 虚高)
                temp_on = state.temp_ctrl is not None and state.temp_ctrl.online
                T1_pv = state.current_T if temp_on else None
                # 段开始: reset counter + 开 measure (升温过程计数)
                with state.measure_lock:
                    if state.fringe_counter is not None:
                        state.fringe_counter.reset()
                    state.n_gate = None  # N 振动门随之重置
                    state.gated_N = 0.0
                    state.gated_N_abs = 0.0  # 防短段未定标时残留上一段 N
                    state.measure_active = True
                ok = self._wait_pv_reach(T2)
                # 段结束: 停 measure, 取 N
                with state.measure_lock:
                    state.measure_active = False
                if state.fringe_counter is None:
                    continue
                # dT: 用设定温度差 (扫描记录温度=设定温度; 段起点实测 T1_pv 仅作参考字段)
                dT = T2 - T1
                N_avg, alpha_seg = self._settle_segment(
                    w, f, i + 1, T1, T2, dT, T1_pv, valid=ok)
                if not ok:
                    # 到温超时: 硬件未达到设定温度, 后续段只会更热更不可能到 —
                    # 中止扫描(不再逐段空转 900s), 已结算的有效段保留供报告
                    self.message = f'段 {i+1} 到温超时, 扫描中止(保留已结算段)'
                    aborted = True
                    break
                else:
                    self.message = f'段 {i+1} 完成: N={N_avg:.1f} α={alpha_seg:.2f}'
            # 用户手动停止/进段超时中止时不要覆盖为 done
            # (否则前端显示"扫描完成", Agent 也会误判完成并生成报告)
            if self._running and not aborted:
                self.phase = 'done'
                self.message = '扫描完成'
            elif aborted:
                self.phase = 'stopped'   # message 保留超时原因
            else:
                self.phase = 'stopped'
                self.message = '扫描已停止'
        finally:
            self._running = False
            state.measure_active = False
            state.scan_active = False
            # 扫描结束/停止后关闭加热, 避免 SV 停在 T_end 持续加热
            try:
                if state.temp_ctrl is not None and state.temp_ctrl.online:
                    state.temp_ctrl.heat_off()
            except Exception:
                pass
            f.close()

    def _run_fringe(self):
        """N 步进扫描: 连续升温到 T_end, 每累计 n_step 条条纹结算一段。
        段 dT = 段内实测 PV 区间 (N 精确固定, dT 实测 — 与温度步进对偶)。
        每段结算后 reset 计数器重新开始 (段间隙 <0.5s, 漏计风险极小,
        换取各段 series 独立使 MC/误差预算有效)。"""
        state.scan_active = True
        f = open(self.csv_path, 'w', newline='', encoding='utf-8')
        w = csv.writer(f)
        w.writerow(['time', 'T_end', 'N_per_r0', 'N_avg', 'seg', 'dT', 'alpha_seg'])
        try:
            aborted = False
            temp_on = state.temp_ctrl is not None and state.temp_ctrl.online
            # 进段等待: 升温至起点 T_start 保持 3s, 未到前不计数
            self.message = f'N步进: 升温至起点 {self.T_start}°C'
            if not self._wait_pv_enter(self.T_start):
                if self._running:
                    self.message = f'N步进: 升温至起点 {self.T_start}°C 超时, 中止'
                    aborted = True
            if not aborted and self._running:
                if temp_on:
                    # 一次设到终点, PID 连续升温, 期间按 N 分段
                    self.current_sv = self.T_end
                    state.temp_ctrl.regulate_to(self.T_end)
                self.phase = 'measuring'
                seg_no = 0
                reached_end = False
                while self._running and not reached_end:
                    seg_no += 1
                    self.current_seg = seg_no
                    # 段开始: reset + 开计数, 记录段起点 PV
                    with state.measure_lock:
                        if state.fringe_counter is not None:
                            state.fringe_counter.reset()
                        state.n_gate = None  # N 振动门随之重置
                        state.gated_N = 0.0
                        state.gated_N_abs = 0.0
                        state.measure_active = True
                    pv_seg_start = state.current_T
                    # 段内轮询: N 达步 / 到终温 / 用户停止
                    while self._running:
                        N_now = (state.gated_N_abs
                                 if state.fringe_counter is not None else 0.0)
                        pv = state.current_T
                        self.message = (f'N步进 段{seg_no}: {N_now:.1f}/{self.n_step} 条  '
                                        f'PV={pv:.2f}°C')
                        if N_now >= self.n_step:
                            break
                        if temp_on and pv >= self.T_end - self.stable_band:
                            reached_end = True
                            break
                        self._sample_n()
                        time.sleep(0.3)
                    # 段结算 (用户停止/到终温时结算不足段, 数据不浪费)
                    with state.measure_lock:
                        state.measure_active = False
                    if state.fringe_counter is None:
                        break
                    pv_end = state.current_T
                    dT = pv_end - pv_seg_start
                    N_avg, alpha_seg = self._settle_segment(
                        w, f, seg_no, pv_seg_start, pv_end, dT, pv_seg_start)
                    self.message = (f'N步进 段{seg_no} 结算: N={N_avg:.1f} '
                                    f'dT={dT:.2f}°C α={alpha_seg:.2f}')
            if self._running and not aborted:
                self.phase = 'done'
                self.message = f'N步进扫描完成: 共 {len(self.segments)} 段'
            elif aborted:
                self.phase = 'stopped'
            else:
                self.phase = 'stopped'
                self.message = f'N步进已停止 (已记录 {len(self.segments)} 段)'
        finally:
            self._running = False
            state.measure_active = False
            state.scan_active = False
            try:
                if state.temp_ctrl is not None and state.temp_ctrl.online:
                    state.temp_ctrl.heat_off()
            except Exception:
                pass
            f.close()

    def _run_free(self):
        """即时计数: 不控温 (用户自行加热/不加热均可), 启动即计数,
        点停止时结算单段 (N / dT=实测 PV 区间 / α)。
        注意: 本模式不碰温控器, 退出时也不 heat_off (不打断用户手动加热)。"""
        state.scan_active = True
        f = open(self.csv_path, 'w', newline='', encoding='utf-8')
        w = csv.writer(f)
        w.writerow(['time', 'T_end', 'N_per_r0', 'N_avg', 'seg', 'dT', 'alpha_seg'])
        try:
            with state.measure_lock:
                if state.fringe_counter is not None:
                    state.fringe_counter.reset()
                state.n_gate = None  # N 振动门随之重置
                state.gated_N = 0.0
                state.gated_N_abs = 0.0
                state.measure_active = True
            pv_start = state.current_T
            self.phase = 'measuring'
            self.current_seg = 1
            while self._running:
                N_now = (state.gated_N_abs
                         if state.fringe_counter is not None else 0.0)
                dT_now = state.current_T - pv_start
                self.message = (f'即时计数中: N={N_now:.1f}  dT={dT_now:+.2f}°C  '
                                f'(点停止结算)')
                self._sample_n()
                time.sleep(0.5)
            # 用户点停止 → 结算单段
            with state.measure_lock:
                state.measure_active = False
            if state.fringe_counter is not None:
                pv_end = state.current_T
                # 温控在线用实测 PV 区间; 温控离线(视频模拟)用标称温升,
                # 使 α 在无温控硬件时也可由视频 N 结算 (标称 ΔT = 扫描配置 T_end-T_start)
                temp_on = state.temp_ctrl is not None and state.temp_ctrl.online
                if temp_on:
                    dT = pv_end - pv_start
                    t1_label, t2_label = pv_start, pv_end
                else:
                    dT = self.T_end - self.T_start
                    t1_label, t2_label = self.T_start, self.T_end
                N_avg, alpha_seg = self._settle_segment(
                    w, f, 1, t1_label, t2_label, dT, pv_start)
                # free 模式无自然完成, 只能由用户停止结束 → 保持 stop() 的 'stopped',
                # 不覆盖为 'done' (否则前端/Agent 误判"扫描完成"); 结果已在 segments 中
                self.phase = 'stopped'
                if dT >= 0.05:
                    self.message = (f'即时计数结算: N={N_avg:.1f} '
                                    f'dT={dT:.2f}°C α={alpha_seg:.2f}')
                else:
                    self.message = (f'即时计数结算: N={N_avg:.1f} '
                                    f'(dT={dT:.2f}°C 过小, α 无效)')
        finally:
            self._running = False
            state.measure_active = False
            state.scan_active = False
            # 即时模式不 heat_off: 本模式从未开过加热, 不打断用户手动控温
            f.close()

    def _valid_alpha_list(self):
        """有效段的 α 列表 (剔除 valid=False 的超时/无效段, 防污染均值)"""
        return [s['alpha'] for s in self.segments if s.get('valid', True)]

    def status(self):
        segs = list(self.segments)   # 快照: 防并发遍历 append 抛异常
        va = self._valid_alpha_list()
        mean, std = alpha_stats(va) if va else (0, 0)
        err = error_vs_reference(mean) if va else 0
        return {
            'active': self._running, 'phase': self.phase, 'message': self.message,
            'mode': self.mode, 'n_step': self.n_step,
            'current_seg': self.current_seg, 'total_seg': self.total_seg,
            'current_sv': self.current_sv, 'current_T': state.current_T,
            'segments': segs,
            'alpha_avg': round(mean, 2), 'alpha_std': round(std, 2),
            'error_pct': round(err, 2), 'reference_alpha': REFERENCE_ALPHA,
        }

    def result(self):
        va = self._valid_alpha_list()
        mean, std = alpha_stats(va) if va else (0, 0)
        return {
            'alpha_avg': round(mean, 2), 'alpha_std': round(std, 2),
            'error_pct': round(error_vs_reference(mean), 2),
            'reference_alpha': REFERENCE_ALPHA,
            'n_segments': len(self.segments),
            'csv_path': str(self.csv_path) if self.csv_path else None,
            'segments': list(self.segments),
        }

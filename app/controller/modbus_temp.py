"""LU-926U 温控器 Modbus RTU 接口（移植自 web_t_c/app.py，内嵌同进程）"""
import time
import threading
from collections import deque

import serial


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, 'little')


REGISTERS = {
    'SV1': 0x0000, 'SV2': 0x0001, 'SMV': 0x0002,
    'Sn1': 0x0017, 'Poi1': 0x0018, 'oSt1': 0x0019,
    'Addr': 0x0036, 'bps': 0x0037, 'crL': 0x0038,
    'Act': 0x003B, 'cHy': 0x003C, 'P1': 0x003D, 'P2': 0x003E,
    'rt': 0x003F, 'ctL': 0x0040, 'EMV': 0x0042, 'SVH': 0x0043,
}
READONLY = {'SV': 0x0100, 'MV': 0x0101, 'PV1': 0x0102, 'PV2': 0x0103,
             'CJ': 0x0104, 'PV': 0x0102}

# 出厂默认控制参数 (说明书参数表默认列; cHy 寄存器原值 5 = 0.5°C)
# 不含 SV: 出厂 SV=800(80.0°C) 恰为安全上限, 恢复它危险
FACTORY_DEFAULTS = {'P1': 116, 'P2': 71, 'rt': 353, 'cHy': 5, 'ctL': 4, 'Act': 1}


class TempController:
    def __init__(self, port='COM7', baud=9600, slave=1):
        self.port = port; self.baud = baud; self.slave = slave
        self.serial = None
        self._lock = threading.Lock()
        self.pv = 0.0; self.sv = 0.0; self.mv = 0.0; self.online = False
        self.history = deque(maxlen=6000)  # 0.5s×6000=50min, 覆盖完整扫描
        self._mon_running = False
        self._mon_thread = None

    def open(self):
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.3,
                                         bytesize=8, parity='N', stopbits=1)
            return True
        except Exception as e:
            print(f'[TempController] 串口 {self.port} 错误: {e}')
            return False

    def close(self):
        self._mon_running = False
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
        except Exception:
            pass

    def read_reg(self, reg):
        if not self.serial or not self.serial.is_open:
            return None
        with self._lock:
            try:
                frame = bytes([self.slave, 0x03, (reg >> 8) & 0xFF, reg & 0xFF, 0, 1])
                frame += crc16(frame)
                self.serial.rts = True; time.sleep(0.002)
                self.serial.write(frame); self.serial.flush()
                time.sleep(0.006); self.serial.rts = False; time.sleep(0.03)
                buf = bytearray(); dl = time.time() + 0.3
                while time.time() < dl:
                    n = self.serial.in_waiting
                    if n > 0:
                        buf.extend(self.serial.read(n)); dl = time.time() + 0.05
                    else:
                        time.sleep(0.005)
                if len(buf) < 5 or crc16(buf[:-2]) != buf[-2:]:
                    return None
                # 异常响应: 功能码最高位置1, buf[2]为异常码(非法功能码/地址/值等)
                if buf[1] & 0x80:
                    print(f'[TempController] 读 0x{reg:04X} 异常码: {buf[2]}')
                    return None
                # 校验从站地址与功能码, 防串扰/残留帧误命中
                if buf[0] != self.slave or buf[1] != 0x03:
                    return None
                v = (buf[3] << 8) | buf[4]
                if v >= 0x8000:          # 16位有符号补码(负温度/调零/报警值等)
                    v -= 0x10000
                return v
            except Exception as e:
                print(f'[TempController] 读 0x{reg:04X} 错: {e}')
            return None

    def write_reg(self, reg, val):
        if not self.serial or not self.serial.is_open:
            return False
        with self._lock:
            try:
                frame = bytes([self.slave, 0x06, (reg >> 8) & 0xFF, reg & 0xFF,
                               (val >> 8) & 0xFF, val & 0xFF])
                frame += crc16(frame)
                self.serial.rts = True; time.sleep(0.002)
                self.serial.write(frame); self.serial.flush()
                time.sleep(0.008); self.serial.rts = False; time.sleep(0.05)
                # 增量读取 (回显固定 8 字节; 之前 read(15) 每次必等满 0.3s 超时)
                raw = bytearray(); dl = time.time() + 0.3
                while time.time() < dl and len(raw) < 8:
                    n = self.serial.in_waiting
                    if n > 0:
                        raw.extend(self.serial.read(n)); dl = time.time() + 0.05
                    else:
                        time.sleep(0.005)
                if len(raw) < 8 or crc16(raw[:-2]) != raw[-2:]:
                    return False
                if raw[1] & 0x80:
                    print(f'[TempController] 写 0x{reg:04X}={val} 异常码: {raw[2]}')
                    return False
                # 校验回显: 从站/功能码/地址/值须与下发一致
                if (raw[0] != self.slave or raw[1] != 0x06 or
                        (raw[2] << 8) | raw[3] != reg or
                        (raw[4] << 8) | raw[5] != (val & 0xFFFF)):
                    return False
                return True
            except Exception as e:
                print(f'[TempController] 写 0x{reg:04X} 错: {e}')
            return False

    # ── 高层接口 ──
    def get_pv(self):
        v = self.read_reg(READONLY['PV']); return v * 0.1 if v is not None else None
    def get_sv(self):
        v = self.read_reg(REGISTERS['SV1']); return v * 0.1 if v is not None else None
    def get_mv(self):
        # 0x0101, 0~25600 对应 0~100%
        v = self.read_reg(READONLY['MV']); return v / 256.0 if v is not None else None
    def set_sv(self, t):
        return self.write_reg(REGISTERS['SV1'], int(t * 10))
    def heat_on(self):
        # 全速加热: 先切手动模式(crL=3), 再设 SMV=100%
        # (说明书: SMV 仅 crL=3 时有效; 先切模式避免非手动态下 SMV 写入被丢)
        self.write_reg(REGISTERS['crL'], 3)   # 手动
        return self.write_reg(REGISTERS['SMV'], 25600)  # 0~25600 -> 0~100%
    def heat_full(self, percent=100):
        # 任意占空比全速加热: 0~100% -> SMV 0~25600, 手动模式
        self.write_reg(REGISTERS['crL'], 3)   # 手动
        smv = int(max(0, min(percent, 100)) * 256)
        return self.write_reg(REGISTERS['SMV'], smv)
    def heat_pid(self):
        # 切回 PID 智能调节 (保温/精控用) — 由 SV 驱动输出; 需先自整定出合适参数
        return self.write_reg(REGISTERS['crL'], 4)  # PID
    def regulate_to(self, t):
        """PID 智能调节到 t°C: 设 SV + 切 PID 模式(由 SV 驱动, 非手动 bang-bang)
        说明书: crL=4 智能调节, 输出由 SV 驱动; 手动模式(crL=3)忽略 SV, 全功率易过冲。
        首次使用建议先自整定(crL=Aut) 出合适 P1/P2/rt 再用本接口。"""
        self.set_sv(t)
        return self.heat_pid()   # crL=4
    def heat_off(self):
        self.write_reg(REGISTERS['crL'], 3)   # 手动
        return self.write_reg(REGISTERS['SMV'], 0)  # 0%

    # ── PID 自整定 (说明书 §7: crL=2 Aut 位式振荡 3 周期, 完成后自动转 crL=4) ──
    def start_autotune(self) -> bool:
        """触发自整定: crL=4 归位 (面板快捷键要求 PID 态才能切 Aut,
        当前可能在 MAn, 先归位消除未定义行为) → crL=2 触发 → 回读确认。
        返回是否已进入 At 态 (crL==2)。"""
        if not self.write_reg(REGISTERS['crL'], 4):
            return False
        time.sleep(0.1)
        if self.read_reg(REGISTERS['crL']) != 4:
            return False
        if not self.write_reg(REGISTERS['crL'], 2):
            return False
        time.sleep(0.1)
        return self.read_reg(REGISTERS['crL']) == 2

    def read_pid_params(self) -> dict:
        """读控制参数快照 (供整定前后存档与对比)"""
        out = {}
        for n in ('P1', 'P2', 'rt', 'cHy', 'ctL', 'Act', 'crL'):
            out[n] = self.read_reg(REGISTERS[n])
        sv = self.read_reg(REGISTERS['SV1'])
        out['SV'] = sv * 0.1 if sv is not None else None
        return out

    def restore_factory_control_params(self) -> list:
        """控制参数恢复出厂默认 (逐个写+回读校验)。返回失败项列表。
        注意: 不碰 SV/crL/SMV — 调用方自行控制模式与目标温度。"""
        fails = []
        for n, v in FACTORY_DEFAULTS.items():
            ok = self.write_reg(REGISTERS[n], v)
            time.sleep(0.05)
            if not ok or self.read_reg(REGISTERS[n]) != v:
                fails.append(n)
        return fails

    # ── 通用寄存器设置(按名) ──
    def set_by_name(self, name: str, value: float) -> bool:
        if name not in REGISTERS:
            return False
        v = int(value)
        # 0.1度单位量: 写入需 ×10 (与 get_by_name ÷10 对称)
        if name in ('SV1', 'SV2', 'oSt1'):
            v = int(float(value) * 10)
        return self.write_reg(REGISTERS[name], v)

    def get_by_name(self, name: str):
        if name in REGISTERS:
            v = self.read_reg(REGISTERS[name])
        elif name in READONLY:
            v = self.read_reg(READONLY[name])
        else:
            return None
        if v is None:
            return None
        # 0.1度单位量(SV/PV/oSt1)需 ÷10; Sn1/Poi1 是代码不是温度, 原值返回
        if name in ('SV1', 'SV2', 'oSt1', 'PV1', 'PV'):
            return v * 0.1
        if name in ('MV',):       # 0~25600 -> 0~100%
            return v / 256.0
        return v

    def monitoring_loop(self):
        self._mon_running = True
        fail = 0
        while self._mon_running:
            pv = self.read_reg(READONLY['PV'])
            sv = self.read_reg(REGISTERS['SV1'])
            mv = self.read_reg(READONLY['MV'])
            if pv is not None:
                self.pv = pv * 0.1
                if sv is not None: self.sv = sv * 0.1
                if mv is not None: self.mv = mv / 256.0   # 0~25600 -> 0~100%
                self.online = True; fail = 0
                if 0 < self.pv < 300:
                    self.history.append({'t': time.strftime('%H:%M:%S'), 'v': self.pv})
            else:
                fail += 1
                if fail > 5:
                    self.online = False
            time.sleep(0.5)

    def start_monitoring(self):
        if self._mon_thread and self._mon_thread.is_alive():
            return
        self._mon_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self._mon_thread.start()

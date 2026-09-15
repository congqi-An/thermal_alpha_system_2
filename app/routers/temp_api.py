"""温控器 REST API"""
from fastapi import APIRouter
from pydantic import BaseModel

from .. import state

router = APIRouter(prefix="/api/temp", tags=["temp"])


@router.get("/status")
async def status():
    return {"online": state.temp_online, "pv": round(state.current_T, 2),
            "sv": round(state.current_sv, 2), "mv": round(state.current_mv, 2)}


class SvReq(BaseModel):
    sv: float

@router.post("/set_sv")
async def set_sv(req: SvReq):
    if state.temp_ctrl is None or not state.temp_ctrl.online:
        return {"ok": False, "msg": "temp_ctrl offline"}
    ok = state.temp_ctrl.set_sv(req.sv)
    return {"ok": ok, "sv": req.sv}


class HeatReq(BaseModel):
    action: str   # on(PID控温) / full(满功率手动) / off

@router.post("/heat")
async def heat(req: HeatReq):
    if state.temp_ctrl is None or not state.temp_ctrl.online:
        return {"ok": False, "msg": "temp_ctrl offline"}
    if req.action == "on":
        # PID 模式(crL=4), 由当前 SV 驱动 — 已自整定(P1=5612/P2=78/rt=124),
        # 实测过冲 +0.2°C / 稳态 ±0.2°C; 旧 heat_on 满功率手动模式忽略 SV 必过冲
        ok = state.temp_ctrl.heat_pid()
    elif req.action == "full":
        ok = state.temp_ctrl.heat_on()   # 手动满功率 (特殊场景: 快速升温/恒功率采集)
    else:
        ok = state.temp_ctrl.heat_off()
    return {"ok": ok, "action": req.action}


FACTORY_PID = (116, 71, 353)   # 出厂默认 P1/P2/rt, 用于判断是否已整定

@router.get("/pid_params")
async def pid_params():
    """PID 控制参数与整定状态 (tuned=与出厂默认不同即视为已整定)"""
    if state.temp_ctrl is None or not state.temp_ctrl.online:
        return {"ok": False, "msg": "temp_ctrl offline"}
    p = state.temp_ctrl.read_pid_params()
    tuned = (p.get('P1'), p.get('P2'), p.get('rt')) != FACTORY_PID \
        and p.get('P1') is not None
    return {"ok": True, "params": p, "tuned": tuned,
            "factory": {"P1": 116, "P2": 71, "rt": 353}}


class WriteReq(BaseModel):
    reg: str          # 寄存器名: SV1/P1/P2/rt/cHy/ctL/crL/Act/Sn1/...
    value: float


@router.post("/write")
async def write_reg(req: WriteReq):
    if state.temp_ctrl is None or not state.temp_ctrl.online:
        return {"ok": False, "msg": "offline"}
    ok = state.temp_ctrl.set_by_name(req.reg, req.value)
    return {"ok": ok, "reg": req.reg, "value": req.value}


@router.get("/get")
async def get_reg(reg: str):
    if state.temp_ctrl is None:
        return {"reg": reg, "value": None}
    return {"reg": reg, "value": state.temp_ctrl.get_by_name(reg)}


@router.get("/history")
async def history():
    return state.temp_history


@router.post("/reconnect")
async def reconnect():
    """Reconnect temperature controller (close + reopen serial)."""
    from ..config import MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE
    from ..controller.modbus_temp import TempController
    # Close existing
    if state.temp_ctrl is not None:
        try:
            state.temp_ctrl.close()
        except Exception:
            pass
    # Reopen
    try:
        ctrl = TempController(MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE)
        if ctrl.open():
            ctrl.start_monitoring()
            state.temp_ctrl = ctrl
            # 不在此处置 temp_online=True: 串口打开不等于设备在线,
            # 由 monitoring_loop 首次成功读 PV 后确认 (经 _temp_sync_loop 同步)
            return {"ok": True, "msg": f"reconnected {MODBUS_PORT}, 等待首次读数确认在线"}
        else:
            state.temp_online = False
            return {"ok": False, "msg": f"open {MODBUS_PORT} failed"}
    except Exception as e:
        state.temp_online = False
        return {"ok": False, "msg": str(e)}

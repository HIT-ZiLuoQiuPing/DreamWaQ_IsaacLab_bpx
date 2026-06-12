#!/usr/bin/env python3
"""
XWK BPX Joint SDK Policy Lab

Purpose:
- Real BPX JointLevelControl test UI through TCP bridge.
- Safe software-zero, PD stand, PD sit/crouch.
- Safe DWAQ TorchScript policy deployment with UI velocity command.

Control stack:
Python UI -> TCP JSON -> C++ bridge -> bpx_sdk_open JointLevelControl -> BPX.

Author: xwk
"""

from __future__ import annotations

import json
import math
import os
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception as exc:
    raise SystemExit("Missing PyYAML. Install with: pip install pyyaml") from exc

try:
    import torch
except Exception:
    torch = None

try:
    import inputs
    HAS_INPUTS = True
except Exception:
    inputs = None
    HAS_INPUTS = False

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_config_working.yaml"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT / "configs" / "real_config.yaml"
POSES_PATH = ROOT / "configs" / "poses.yaml"
WORKING_CONFIG_PATH = ROOT / "configs" / "real_config_working.yaml"
DEFAULT_POLICY_PATH = ROOT / "policy" / "bpx_dwaq_v2.pt"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Minimal bilingual UI dictionary. Use compact labels so the interface stays clean.
I18N = {
    "zh": {
        "app_title": "XWK BPX 真机策略控制中心",
        "connect": "连接", "disconnect": "断开", "estop": "急停 / 阻尼",
        "safety_gate": "安全门", "hang_confirm": "确认：机器狗已吊起/支撑，周围安全",
        "arm": "ARM 真实关节命令", "stand_confirm": "确认：PD 站姿已稳定",
        "imu_cal": "IMU 起立前校准", "imu_cal_btn": "开始 2 秒 IMU 校准",
        "imu_required": "必须先完成 IMU 静止校准，再起立或启动策略。",
        "scale_contract": "训练 / Sim2Sim Scale 对照",
        "pd_control": "PD 姿态控制", "joystick": "速度摇杆", "policy": "策略控制", "robot_state": "机器人状态",
        "teach": "示教/阻尼", "hold": "保持当前姿态", "stand": "低增益标准站姿", "sit": "坐下/趴下",
        "zero": "从当前趴下姿态记录软件零位", "save": "保存工作配置",
        "load_policy": "加载策略", "start_policy": "启动策略", "stop_policy": "停止策略", "browse": "选择",
        "lang": "English", "physical_joystick": "物理摇杆", "cmd_smooth": "指令平滑", "imu_filter": "IMU滤波",
        "imu_filter_enable": "IMU滤波开关", "cmd_filter_enable": "指令平滑开关",
    },
    "en": {
        "app_title": "XWK BPX Real Policy Control Center",
        "connect": "Connect", "disconnect": "Disconnect", "estop": "E-STOP / Damping",
        "safety_gate": "Safety Gate", "hang_confirm": "Confirm: robot is hanging/supported and area is clear",
        "arm": "ARM real joint command", "stand_confirm": "Confirm: PD stand is stable",
        "imu_cal": "Pre-Stand IMU Calibration", "imu_cal_btn": "Start 2s IMU Calibration",
        "imu_required": "IMU static calibration is required before standing or starting policy.",
        "scale_contract": "Training / Sim2Sim Scale Contract",
        "pd_control": "PD Pose Control", "joystick": "Velocity Joystick", "policy": "Policy Control", "robot_state": "Robot State",
        "teach": "Teach / Damping", "hold": "Hold Current Pose", "stand": "Low-Gain Standard Stand", "sit": "Sit / Crouch",
        "zero": "Record Software Zero from Current Crouch", "save": "Save Working Config",
        "load_policy": "Load Policy", "start_policy": "START POLICY", "stop_policy": "STOP POLICY", "browse": "Browse",
        "lang": "中文", "physical_joystick": "Gamepad", "cmd_smooth": "Cmd smooth", "imu_filter": "IMU filter",
        "imu_filter_enable": "IMU filter ON", "cmd_filter_enable": "Cmd smooth ON",
    },
}

# Cross-platform font selection.
# Linux tkinter uses X11 font server; "helvetica" and "courier" are reliably
# available via Nimbus Sans L / Nimbus Mono L aliases with good CJK coverage.
# On Windows/macOS, use the native system fonts.
import sys as _sys
if _sys.platform.startswith("linux"):
    SANS_FAMILY = "helvetica"
    MONO_FAMILY = "courier"
else:
    SANS_FAMILY = "Segoe UI"
    MONO_FAMILY = "Consolas"

JOINT_ROW_FONT = (MONO_FAMILY, 10)
TITLE_FONT = (SANS_FAMILY, 16, "bold")
SUB_FONT = (SANS_FAMILY, 10)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "--"


def finite_list(xs: List[float]) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except Exception:
        return False


def quat_normalize_xyzw(q: List[float]) -> List[float]:
    if len(q) < 4:
        return [0.0, 0.0, 0.0, 1.0]
    x, y, z, w = [float(v) for v in q[:4]]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [x/n, y/n, z/n, w/n]


def quat_mul_xyzw(a: List[float], b: List[float]) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]


def quat_rotate_inverse_xyzw(q: List[float], v: List[float]) -> List[float]:
    # R(q)^T * v = q^-1 * [v,0] * q
    qn = quat_normalize_xyzw(q)
    qc = [-qn[0], -qn[1], -qn[2], qn[3]]
    vv = [float(v[0]), float(v[1]), float(v[2]), 0.0]
    out = quat_mul_xyzw(quat_mul_xyzw(qc, vv), qn)
    return out[:3]


def quat_inv_xyzw(q: List[float]) -> List[float]:
    qn = quat_normalize_xyzw(q)
    return [-qn[0], -qn[1], -qn[2], qn[3]]


def vec_norm3(v: List[float], fallback: List[float] = None) -> List[float]:
    if fallback is None:
        fallback = [0.0, 0.0, -1.0]
    if len(v) < 3:
        return list(fallback)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-9:
        return list(fallback)
    return [x/n, y/n, z/n]


def avg_quat_xyzw(samples: List[List[float]]) -> List[float]:
    # Simple sign-consistent average, sufficient for short static IMU calibration windows.
    if not samples:
        return [0.0, 0.0, 0.0, 1.0]
    ref = quat_normalize_xyzw(samples[0])
    acc = [0.0, 0.0, 0.0, 0.0]
    for q in samples:
        qn = quat_normalize_xyzw(q)
        dot = sum(qn[i] * ref[i] for i in range(4))
        if dot < 0:
            qn = [-x for x in qn]
        for i in range(4):
            acc[i] += qn[i]
    return quat_normalize_xyzw(acc)


def avg_vec3(samples: List[List[float]]) -> List[float]:
    if not samples:
        return [0.0, 0.0, 0.0]
    n = float(len(samples))
    return [sum(float(s[i]) for s in samples if len(s) >= 3) / n for i in range(3)]


def std_vec3(samples: List[List[float]], mean: List[float]) -> List[float]:
    if not samples:
        return [0.0, 0.0, 0.0]
    n = float(len(samples))
    return [math.sqrt(sum((float(s[i]) - mean[i]) ** 2 for s in samples if len(s) >= 3) / n) for i in range(3)]


class BridgeClient:
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.state_q: queue.Queue[dict] = queue.Queue(maxsize=10)
        self.connected = False
        self.last_error = ""

    def connect(self, host: str, port: int):
        self.close()
        self.stop_evt.clear()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((host, port))
        s.settimeout(0.2)
        self.sock = s
        self.connected = True
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

    def _rx_loop(self):
        buf = b""
        while not self.stop_evt.is_set() and self.sock:
            try:
                data = self.sock.recv(65536)
                if not data:
                    self.last_error = "bridge disconnected"
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        st = json.loads(line.decode("utf-8", errors="replace"))
                        if self.state_q.full():
                            try:
                                self.state_q.get_nowait()
                            except Exception:
                                pass
                        self.state_q.put_nowait(st)
                    except Exception as exc:
                        self.last_error = f"json parse: {exc}"
            except socket.timeout:
                continue
            except Exception as exc:
                self.last_error = str(exc)
                break
        self.connected = False

    def send(self, msg: dict):
        if not self.sock:
            return
        payload = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        self.sock.sendall(payload)

    def close(self):
        self.stop_evt.set()
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.connected = False


class JoystickControl:
    """Read gamepad/joystick axes in background thread, map to vx/vy/wz commands."""

    # Linux evdev axis code → (vel_key, scale)
    AXIS_MAP: Dict[str, tuple] = {
        "ABS_Y":        ("vx", -1.0),   # Left stick Y → forward(+)
        "ABS_X":        ("vy",  1.0),   # Left stick X → strafe right(+)
        "ABS_RX":       ("wz",  1.0),   # Right stick X → yaw right(+)
        "ABS_RY":       ("wz", -1.0),   # Right stick Y → yaw (alt)
        "ABS_Z":        ("wz",  0.5),   # Left trigger → yaw left
        "ABS_RZ":       ("wz", -0.5),   # Right trigger → yaw right
    }
    DEADZONE = 0.12

    def __init__(self, app: "App"):
        self.app = app
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._raw_vals = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        if not HAS_INPUTS:
            return False
        try:
            return len(inputs.devices.gamepads) > 0
        except Exception:
            return False

    def start(self):
        if not HAS_INPUTS:
            return False
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="joystick")
        self.thread.start()
        return True

    def stop(self):
        self.stop_evt.set()
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None

    def read_vel(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._raw_vals)

    def _run(self):
        while not self.stop_evt.is_set():
            try:
                if not inputs.devices.gamepads:
                    self.stop_evt.wait(1.0)
                    continue
                device = inputs.devices.gamepads[0]
                events = device.read()
            except inputs.UnpluggedError:
                self.stop_evt.wait(1.0)
                continue
            except OSError:
                self.stop_evt.wait(1.0)
                continue
            except Exception:
                self.stop_evt.wait(0.5)
                continue

            if not events:
                self.stop_evt.wait(0.01)
                continue

            with self._lock:
                for ev in events:
                    if ev.ev_type == "Absolute":
                        mapping = self.AXIS_MAP.get(ev.code)
                        if mapping is None:
                            continue
                        key, scale = mapping
                        norm = (ev.state / 32767.0) * scale
                        if abs(norm) < self.DEADZONE:
                            norm = 0.0
                        self._raw_vals[key] = norm

                    elif ev.ev_type == "Key" and ev.state == 1:
                        # Button pressed
                        if ev.code == "BTN_SOUTH":   # A button → estop
                            self.app.after_idle(self.app.on_estop)
                        elif ev.code == "BTN_EAST":   # B button → zero vel
                            self.app.after_idle(self.app.zero_vel)
                        elif ev.code == "BTN_NORTH":  # Y button → toggle joystick enabled
                            self.app.after_idle(self._toggle_enabled)

    def _toggle_enabled(self):
        cur = self.app.joystick_enabled_var.get()
        self.app.joystick_enabled_var.set(not cur)

    def apply_to_app(self):
        """Called periodically from main thread to push joystick values into UI vars."""
        if not self.app.joystick_enabled_var.get():
            return
        raw = self.read_vel()
        mx = float(self.app.max_vx_var.get())
        my = float(self.app.max_vy_var.get())
        mw = float(self.app.max_wz_var.get())
        self.app.vx_var.set(clamp(raw["vx"] * mx, -mx, mx))
        self.app.vy_var.set(clamp(raw["vy"] * my, -my, my))
        self.app.wz_var.set(clamp(raw["wz"] * mw, -mw, mw))


class VirtualJoystick(tk.Frame):
    """On-screen virtual joystick pad — drag the thumb to control vx/vy, slider for wz."""

    CENTER_DEADZONE = 0.15  # click within this radius = zero

    def __init__(self, parent, vx_var, vy_var, wz_var,
                 max_vx_var, max_vy_var, max_wz_var,
                 pad_size=180, bg_color="#1f2937"):
        super().__init__(parent, bg=bg_color)
        self.vx_var = vx_var
        self.vy_var = vy_var
        self.wz_var = wz_var
        self.max_vx_var = max_vx_var
        self.max_vy_var = max_vy_var
        self.max_wz_var = max_wz_var

        self.pad_size = pad_size
        self.center = pad_size // 2
        self.outer_r = pad_size // 2 - 12        # boundary radius
        self.thumb_r = 16                         # thumb radius
        self.dragging = False
        self.nx = 0.0   # normalized x [-1, 1]
        self.ny = 0.0   # normalized y [-1, 1]

        # --- Joystick pad ---
        self.canvas = tk.Canvas(self, width=pad_size, height=pad_size,
                                bg=bg_color, highlightthickness=0, relief="flat")
        self.canvas.pack(pady=(4, 2))

        # Draw static elements
        self._draw_static()

        # Mouse bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # --- Yaw slider (snaps back to 0 on release) ---
        slider_frame = tk.Frame(self, bg=bg_color)
        slider_frame.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(slider_frame, text="← yaw →", bg=bg_color, fg="#9ca3af",
                 font=(SANS_FAMILY, 9)).pack(side="left", padx=(0, 4))
        self.wz_scale = tk.Scale(slider_frame, from_=-1.0, to=1.0, resolution=0.01,
                                 orient="horizontal", showvalue=False,
                                 bg="#111827", fg="#e5e7eb", troughcolor="#374151",
                                 highlightthickness=0, sliderlength=20,
                                 command=self._on_wz_slider)
        self.wz_scale.pack(side="left", fill="x", expand=True)
        self.wz_scale.bind("<ButtonRelease-1>", lambda e: self.wz_scale.set(0))
        self.wz_label = tk.Label(slider_frame, text="0.00", bg=bg_color, fg="#e5e7eb",
                                 font=(MONO_FAMILY, 9), width=5)
        self.wz_label.pack(side="left", padx=(4, 0))

        # --- Hint ---
        tk.Label(self, text="拖拽加速 · 松手回中平滑停止",
                 bg=bg_color, fg="#9ca3af", font=(SANS_FAMILY, 8)).pack(pady=(0, 1))
        # --- Vel readout ---
        self.vel_label = tk.Label(self, text="vx: 0.00  vy: 0.00  wz: 0.00",
                                  bg=bg_color, fg="#e5e7eb", font=(MONO_FAMILY, 9))
        self.vel_label.pack(pady=(0, 4))

        # Periodic UI update
        self._update_display()

    def _draw_static(self):
        c = self.canvas
        # Outer boundary
        cx, cy = self.center, self.center
        c.create_oval(cx - self.outer_r, cy - self.outer_r,
                      cx + self.outer_r, cy + self.outer_r,
                      outline="#4b5563", width=2, fill="#111827")
        # Crosshair
        c.create_line(cx - self.outer_r, cy, cx + self.outer_r, cy,
                      fill="#374151", width=1)
        c.create_line(cx, cy - self.outer_r, cx, cy + self.outer_r,
                      fill="#374151", width=1)
        # Center dot
        c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#6b7280", outline="")
        # Direction labels
        c.create_text(cx, cy - self.outer_r - 10, text="前进", fill="#9ca3af",
                      font=(SANS_FAMILY, 9))
        c.create_text(cx, cy + self.outer_r + 12, text="后退", fill="#9ca3af",
                      font=(SANS_FAMILY, 9))
        c.create_text(cx - self.outer_r - 14, cy, text="左", fill="#9ca3af",
                      font=(SANS_FAMILY, 9))
        c.create_text(cx + self.outer_r + 14, cy, text="右", fill="#9ca3af",
                      font=(SANS_FAMILY, 9))
        # Thumb
        self.thumb_id = c.create_oval(cx - self.thumb_r, cy - self.thumb_r,
                                       cx + self.thumb_r, cy + self.thumb_r,
                                       fill="#3b82f6", outline="#60a5fa", width=2)
        # Thumb inner dot
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#ffffff", outline="")

    def _thumb_coords(self, nx, ny):
        cx, cy = self.center, self.center
        dx = nx * self.outer_r
        dy = ny * self.outer_r
        # Clamp within circle
        r = math.sqrt(dx*dx + dy*dy)
        if r > self.outer_r:
            dx = dx / r * self.outer_r
            dy = dy / r * self.outer_r
        return cx + dx - self.thumb_r, cy + dy - self.thumb_r, \
               cx + dx + self.thumb_r, cy + dy + self.thumb_r

    def _on_press(self, event):
        self.dragging = True
        self._update_thumb(event)

    def _on_drag(self, event):
        if self.dragging:
            self._update_thumb(event)

    def _on_release(self, event):
        if self.dragging:
            self.dragging = False
            # Snap thumb back to center (spring-loaded feel)
            self.nx = 0.0
            self.ny = 0.0
            self._move_thumb(0.0, 0.0)
            # Set target velocity to 0 — cmd smoothing ramps it down gently
            self.vx_var.set(0.0)
            self.vy_var.set(0.0)

    def _update_thumb(self, event):
        cx, cy = self.center, self.center
        dx = event.x - cx
        dy = event.y - cy
        r = math.sqrt(dx*dx + dy*dy)
        if r > self.outer_r:
            dx = dx / r * self.outer_r
            dy = dy / r * self.outer_r
        nx = dx / self.outer_r
        ny = dy / self.outer_r
        self.nx = nx
        self.ny = ny
        self._move_thumb(nx, ny)

    def _move_thumb(self, nx, ny):
        x1, y1, x2, y2 = self._thumb_coords(nx, ny)
        self.canvas.coords(self.thumb_id, x1, y1, x2, y2)

    def _on_wz_slider(self, val):
        self.wz_label.config(text=f"{float(val):.2f}")

    def _update_display(self):
        """Periodically push joystick position to velocity vars (20 Hz)."""
        try:
            if self.dragging:
                mx = float(self.max_vx_var.get())
                my = float(self.max_vy_var.get())
                # Invert Y: up = forward = positive vx
                self.vx_var.set(-self.ny * mx)
                self.vy_var.set(self.nx * my)

            # wz from slider (snaps to 0 on release via binding)
            wz = float(self.wz_scale.get()) * float(self.max_wz_var.get())
            self.wz_var.set(wz)

            self.vel_label.config(
                text=f"vx: {self.vx_var.get():+.2f}  vy: {self.vy_var.get():+.2f}  wz: {self.wz_var.get():+.2f}")
        except Exception:
            pass
        self.after(50, self._update_display)


class JointMapper:
    def __init__(self, cfg: dict, poses: dict):
        self.cfg = cfg
        self.poses = poses
        self.joint_order: List[str] = list(cfg.get("rl_joint_order_12", []))
        self.calib: Dict[str, dict] = cfg.get("calibration", {}).get("joints", {})
        self.safety = cfg.get("safety", {})
        self.margin = float(self.safety.get("soft_limit_margin_rad", 0.08))
        self.step_sim = float(self.safety.get("single_step_max_delta_sim_rad", 0.035))
        self._ensure_soft_limits()

    def _raw_to_sim_one(self, name: str, raw: float) -> float:
        c = self.calib[name]
        return float(c.get("scale", 1.0)) * raw + float(c.get("offset", 0.0))

    def _sim_to_raw_one(self, name: str, sim: float) -> float:
        c = self.calib[name]
        scale = float(c.get("scale", 1.0))
        if abs(scale) < 1e-9:
            raise ValueError(f"bad scale for {name}: {scale}")
        return (sim - float(c.get("offset", 0.0))) / scale

    def _measured_sim_limits(self, name: str) -> Tuple[float, float]:
        c = self.calib[name]
        # Strict URDF/sim contract: prefer the URDF joint limits expressed in sim radians.
        # raw_at_sim_lower/raw_at_sim_upper are only calibration samples; they must not
        # silently redefine the simulation joint limits.
        if c.get("sim_lower") is not None and c.get("sim_upper") is not None:
            lo = float(c["sim_lower"])
            hi = float(c["sim_upper"])
            return min(lo, hi), max(lo, hi)
        vals = []
        for key in ("raw_at_sim_lower", "raw_at_sim_upper"):
            if c.get(key) is not None:
                vals.append(self._raw_to_sim_one(name, float(c[key])))
        if len(vals) == 2:
            return min(vals), max(vals)
        return -math.pi, math.pi

    def _ensure_soft_limits(self):
        for name in self.joint_order:
            c = self.calib.get(name, {})
            if not c.get("enabled", False):
                continue
            if c.get("soft_lower") is None or c.get("soft_upper") is None:
                lo, hi = self._measured_sim_limits(name)
                c["soft_lower"] = lo + self.margin
                c["soft_upper"] = hi - self.margin

    def raw_array_to_sim_by_name(self, raw12: List[float]) -> Dict[str, float]:
        out = {}
        for name in self.joint_order:
            c = self.calib[name]
            idx = c.get("sdk_index")
            if idx is None or idx >= len(raw12):
                continue
            out[name] = self._raw_to_sim_one(name, float(raw12[idx]))
        return out

    def raw_vel_array_to_sim_by_name(self, raw_vel12: List[float]) -> Dict[str, float]:
        out = {}
        for name in self.joint_order:
            c = self.calib[name]
            idx = c.get("sdk_index")
            if idx is None or idx >= len(raw_vel12):
                continue
            out[name] = float(c.get("scale", 1.0)) * float(raw_vel12[idx])
        return out

    def sim_targets_to_sdk_raw_array(self, target_by_name: Dict[str, float], current_raw12: List[float]) -> List[float]:
        pos = list(current_raw12[:12]) if len(current_raw12) >= 12 else [0.0] * 12
        for name in self.joint_order:
            c = self.calib[name]
            idx = c.get("sdk_index")
            if idx is None:
                continue
            sim = float(target_by_name[name])
            sim = clamp(sim, float(c["soft_lower"]), float(c["soft_upper"]))
            pos[int(idx)] = self._sim_to_raw_one(name, sim)
        return pos

    def clamp_sim_targets(self, target_by_name: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for name in self.joint_order:
            c = self.calib[name]
            out[name] = clamp(float(target_by_name[name]), float(c["soft_lower"]), float(c["soft_upper"]))
        return out

    def save_working_config(self):
        out = dict(self.cfg)
        out.setdefault("calibration", {})["joints"] = self.calib
        out.setdefault("calibration", {})["last_saved_by"] = "xwk_joint_lab_ui_policy"
        out.setdefault("calibration", {})["last_saved_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(WORKING_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    def record_crouch_zero_from_raw(self, raw12: List[float]):
        crouch = self.poses["poses"]["sit_crouch"]["joints"]
        for name in self.joint_order:
            c = self.calib[name]
            idx = c.get("sdk_index")
            if idx is None:
                continue
            raw = float(raw12[int(idx)])
            scale = float(c.get("scale", 1.0))
            sim_theory = float(crouch[name])
            c["raw_at_crouch_zero"] = raw
            c["sim_at_crouch_zero"] = sim_theory
            c["offset"] = sim_theory - scale * raw
        self._ensure_soft_limits()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # UI / language state — MUST be set before any self.tr() call
        self.lang = "zh"
        self._i18n_widgets: List[Tuple[Any, str, str]] = []
        self.title(self.tr("app_title"))
        self.geometry("1420x940")
        self.minsize(1180, 760)
        self.configure(bg="#111827")

        self.client = BridgeClient()
        self.state: dict = {}
        self.last_raw = [0.0] * 12
        self.last_dq_raw = [0.0] * 12
        self.active = False
        self.policy_active = False
        self.pose_mode = "IDLE"
        self.current_sim_target: Dict[str, float] = {}
        self.final_sim_target: Dict[str, float] = {}
        self.last_action = [0.0] * 12
        self.obs_history: List[List[float]] = []
        self.policy = None
        self.policy_loaded_path = ""
        self.policy_last_ms = 0.0
        self.log_file = LOG_DIR / f"joint_policy_lab_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        # IMU calibration state. This is a software reference only; it does not call the robot's hardware-zero.
        self.imu_calibrated = False
        self.imu_level_quat = [0.0, 0.0, 0.0, 1.0]
        self.imu_quat_correction = [0.0, 0.0, 0.0, 1.0]
        self.imu_omega_bias = [0.0, 0.0, 0.0]
        self.imu_calib_active = False
        self.imu_calib_samples_quat: List[List[float]] = []
        self.imu_calib_samples_omega: List[List[float]] = []
        self.imu_calib_end_time = 0.0
        self.pending_stand_after_calib = False

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(POSES_PATH, "r", encoding="utf-8") as f:
            poses = yaml.safe_load(f)
        imu_cfg = cfg.get("calibration", {}).get("imu", {})
        if isinstance(imu_cfg.get("level_quat_raw_xyzw"), list) and len(imu_cfg.get("level_quat_raw_xyzw")) >= 4:
            self.imu_level_quat = quat_normalize_xyzw(imu_cfg.get("level_quat_raw_xyzw"))
            # Treat non-default saved quaternion as calibrated; default identity still requires a fresh pre-stand calibration.
            if any(abs(self.imu_level_quat[i] - [0,0,0,1][i]) > 1e-6 for i in range(4)):
                self.imu_quat_correction = quat_inv_xyzw(self.imu_level_quat)
                self.imu_calibrated = True
        if isinstance(imu_cfg.get("omega_bias"), list) and len(imu_cfg.get("omega_bias")) >= 3:
            self.imu_omega_bias = [float(x) for x in imu_cfg.get("omega_bias")[:3]]
        self.mapper = JointMapper(cfg, poses)
        policy_contract = cfg.get("isaaclab_policy_contract", {})
        self.obs_dim = int(policy_contract.get("actor_obs_single_frame_dim", 45))
        self.hist_len = int(policy_contract.get("dwaq_history_length", 5))
        self.action_scale = float(policy_contract.get("action_scale", 0.25))
        obs_scales = policy_contract.get("obs_scales", {})
        if not isinstance(obs_scales, dict):
            obs_scales = {}
        command_scale = policy_contract.get("command_scale", {})
        if not isinstance(command_scale, dict):
            command_scale = {}
        self.obs_scale_ang_vel = float(obs_scales.get("ang_vel", 0.25))
        self.obs_scale_projected_gravity = float(obs_scales.get("projected_gravity", 1.0))
        self.obs_scale_joint_pos = float(obs_scales.get("joint_pos", 1.0))
        self.obs_scale_joint_vel = float(obs_scales.get("joint_vel", 0.05))
        self.obs_scale_last_action = float(obs_scales.get("last_action", 1.0))
        self.cmd_vx_scale_default = float(command_scale.get("vx", obs_scales.get("lin_vel_command_x", 2.0)))
        self.cmd_vy_scale_default = float(command_scale.get("vy", obs_scales.get("lin_vel_command_y", 2.0)))
        self.cmd_wz_scale_default = float(command_scale.get("wz", obs_scales.get("ang_vel_command_z", 0.25)))
        # IMPORTANT: sim_clip_actions is the training/sim2sim clip_actions.
        # It is normally 100 and should NOT be confused with the real-robot safety guard below.
        self.sim_clip_actions = float(policy_contract.get("sim_clip_actions", policy_contract.get("clip_actions", 100.0)))
        guard_cfg = policy_contract.get("deploy_action_guard", {}) if isinstance(policy_contract.get("deploy_action_guard", {}), dict) else {}
        self.deploy_guard_default_enabled = bool(guard_cfg.get("enabled", True))
        self.deploy_guard_default_clip = float(guard_cfg.get("clip", cfg.get("safety", {}).get("first_test_action_clip", 0.12)))
        self.deploy_guard_last_action_source = str(guard_cfg.get("last_action_source_when_enabled", "sent_action"))
        self.deploy_rate_limit_default_enabled = bool(guard_cfg.get("target_rate_limit_enabled", True))
        # auto: try both history_only and current_plus_history exported DWAQ signatures.
        self.policy_input_mode_cfg = str(policy_contract.get("policy_input_mode", "auto"))
        self.policy_input_mode = "unknown"
        self.policy_input_dim = 0
        default_joint_pos_cfg = policy_contract.get("default_joint_pos")
        if isinstance(default_joint_pos_cfg, dict):
            self.default_pose = {name: float(default_joint_pos_cfg[name]) for name in self.mapper.joint_order}
        else:
            self.default_pose = poses["poses"]["stand_standard"]["joints"]

        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="8765")
        self.status_var = tk.StringVar(value="Disconnected")
        self.arm_var = tk.BooleanVar(value=False)
        self.hang_confirm_var = tk.BooleanVar(value=False)
        self.stand_confirm_var = tk.BooleanVar(value=False)
        pd_cfg = cfg.get("pd_profiles", {}).get("profiles", {}).get("hang_hold_cautious", {})
        self.kp_var = tk.DoubleVar(value=float(pd_cfg.get("kp", 20.0)))
        self.kd_var = tk.DoubleVar(value=float(pd_cfg.get("kd", 0.8)))
        self.policy_kp_var = tk.DoubleVar(value=6.0)
        self.policy_kd_var = tk.DoubleVar(value=0.35)
        # Real-robot defaults are deliberately softer than sim2sim Kp=30/Kd=1.
        # action_clip_var is an EXTRA real-robot safety guard, not sim2sim clip_actions.
        # sim2sim/training clip_actions stays in self.sim_clip_actions, normally 100.
        self.action_guard_enabled_var = tk.BooleanVar(value=self.deploy_guard_default_enabled)
        self.policy_rate_limit_enabled_var = tk.BooleanVar(value=self.deploy_rate_limit_default_enabled)
        self.action_clip_var = tk.DoubleVar(value=self.deploy_guard_default_clip)
        self.last_policy_action_raw = [0.0] * 12
        self.last_policy_action_sim = [0.0] * 12
        self.last_policy_action_sent = [0.0] * 12
        self.max_vx_var = tk.DoubleVar(value=0.10)
        self.max_vy_var = tk.DoubleVar(value=0.05)
        self.max_wz_var = tk.DoubleVar(value=0.25)
        self.cmd_vx_scale_var = tk.DoubleVar(value=self.cmd_vx_scale_default)
        self.cmd_vy_scale_var = tk.DoubleVar(value=self.cmd_vy_scale_default)
        self.cmd_wz_scale_var = tk.DoubleVar(value=self.cmd_wz_scale_default)
        self.vx_var = tk.DoubleVar(value=0.0)
        self.vy_var = tk.DoubleVar(value=0.0)
        self.wz_var = tk.DoubleVar(value=0.0)
        self.vel_step_var = tk.DoubleVar(value=0.02)
        self.max_roll_var = tk.DoubleVar(value=0.65)
        self.max_pitch_var = tk.DoubleVar(value=0.65)
        self.max_motor_temp_var = tk.DoubleVar(value=75.0)
        # IMU low-pass filter (EMA alpha: 0=heavy filter, 1=no filter)
        self.imu_filter_alpha_var = tk.DoubleVar(value=0.15)
        self.imu_filter_enabled_var = tk.BooleanVar(value=True)
        self._filt_omega = [0.0, 0.0, 0.0]
        self._filt_gravity = [0.0, 0.0, -1.0]
        # Velocity command smoothing (EMA alpha: 0=very smooth/sluggish, 1=instant)
        self.cmd_smooth_alpha_var = tk.DoubleVar(value=0.15)
        self.cmd_filter_enabled_var = tk.BooleanVar(value=True)
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_wz = 0.0
        self.policy_path_var = tk.StringVar(value=os.fspath(DEFAULT_POLICY_PATH))
        self.policy_status_var = tk.StringVar(value="Policy: not loaded")
        self.command_rate_hz = int(cfg.get("safety", {}).get("rate_hz", 50))

        # Joystick vars (init before _build so UI can reference them)
        self.joystick_enabled_var = tk.BooleanVar(value=False)
        self.joystick_status_var = tk.StringVar(value="Joystick: --")
        self.joystick = JoystickControl(self)

        self._style()
        self._build()
        self._bind_keys()
        if self.imu_calibrated:
            self.imu_status_var.set("IMU: calibrated from config; recalibrate before stand if robot moved")
            self.imu_bias_label.configure(text=f"{self.imu_omega_bias[0]:+.4f} {self.imu_omega_bias[1]:+.4f} {self.imu_omega_bias[2]:+.4f}")

        # Joystick init (after _build so self.log() works)
        if JoystickControl.available():
            self.joystick.start()
            self.joystick_status_var.set("Joystick: connected")
            self.log("Joystick detected, ready to use.")
        else:
            self.joystick_status_var.set("Joystick: not detected")
            if HAS_INPUTS:
                self.log("No joystick found. Plug in a gamepad and restart UI.")

        self.after(30, self._poll_state)
        self.after(int(1000 / max(1, self.command_rate_hz)), self._control_loop)
        self.after(100, self._joystick_poll)

    def tr(self, key: str) -> str:
        return I18N.get(self.lang, I18N["zh"]).get(key, key)

    def _register_i18n(self, widget, key: str, option: str = "text"):
        self._i18n_widgets.append((widget, key, option))
        try:
            widget.configure(**{option: self.tr(key)})
        except Exception:
            pass
        return widget

    def _toggle_language(self):
        self.lang = "en" if self.lang == "zh" else "zh"
        self._refresh_i18n()

    def _refresh_i18n(self):
        self.title(self.tr("app_title"))
        for widget, key, option in self._i18n_widgets:
            try:
                widget.configure(**{option: self.tr(key)})
            except Exception:
                pass

    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#111827")
        style.configure("Card.TFrame", background="#1f2937", relief="flat")
        style.configure("TLabel", background="#111827", foreground="#e5e7eb", font=SUB_FONT)
        style.configure("Card.TLabel", background="#1f2937", foreground="#e5e7eb", font=SUB_FONT)
        style.configure("Title.TLabel", background="#111827", foreground="#f9fafb", font=TITLE_FONT)
        style.configure("TButton", font=(SANS_FAMILY, 10, "bold"), padding=6)
        style.configure("Danger.TButton", foreground="#991b1b")
        style.configure("Treeview", background="#0f172a", fieldbackground="#0f172a", foreground="#e5e7eb", rowheight=24, font=JOINT_ROW_FONT)
        style.configure("Treeview.Heading", font=(SANS_FAMILY, 10, "bold"))

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=14, pady=12)
        title_lbl = ttk.Label(top, style="Title.TLabel")
        self._register_i18n(title_lbl, "app_title")
        title_lbl.pack(side="left")
        lang_btn = ttk.Button(top, command=self._toggle_language)
        self._register_i18n(lang_btn, "lang")
        lang_btn.pack(side="right", padx=(8, 0))
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        conn = ttk.Frame(self, style="Card.TFrame")
        conn.pack(fill="x", padx=14, pady=(0, 10))
        for w in [
            ttk.Label(conn, text="Bridge Host", style="Card.TLabel"),
            ttk.Entry(conn, textvariable=self.host_var, width=16),
            ttk.Label(conn, text="Port", style="Card.TLabel"),
            ttk.Entry(conn, textvariable=self.port_var, width=8),
        ]:
            w.pack(side="left", padx=6, pady=8)
        b = ttk.Button(conn, command=self.on_connect); self._register_i18n(b, "connect"); b.pack(side="left", padx=6, pady=8)
        b = ttk.Button(conn, command=self.on_disconnect); self._register_i18n(b, "disconnect"); b.pack(side="left", padx=6, pady=8)
        b = ttk.Button(conn, command=self.on_estop, style="Danger.TButton"); self._register_i18n(b, "estop"); b.pack(side="left", padx=6, pady=8)

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=14, pady=8)

        # Left panel — fixed natural width, does NOT steal space from the right panel
        left = ttk.Frame(main, style="Card.TFrame", width=660)
        left.pack(side="left", fill="both", expand=False, padx=(0, 6))
        left.pack_propagate(False)  # respect width=660

        # Scrollable right panel — gets the remaining width
        right_outer = ttk.Frame(main, style="Card.TFrame")
        right_outer.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self._right_canvas = tk.Canvas(right_outer, bg="#1f2937", highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=self._right_canvas.yview)
        right = ttk.Frame(self._right_canvas, style="Card.TFrame")

        # Keep the inner frame width synced to the canvas
        def _on_right_configure(event):
            self._right_canvas.itemconfigure(self._right_win, width=event.width - 2)
        right.bind("<Configure>", lambda e: self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all")))
        self._right_canvas.bind("<Configure>", _on_right_configure)

        self._right_win = self._right_canvas.create_window((0, 0), window=right, anchor="nw", width=200)
        self._right_canvas.configure(yscrollcommand=right_scrollbar.set)

        self._right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")

        # Cross-platform mouse wheel scrolling
        def _on_mousewheel(event):
            if event.num == 4 or event.delta > 0:
                self._right_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self._right_canvas.yview_scroll(1, "units")
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._right_canvas.bind(ev, _on_mousewheel)
            right.bind(ev, _on_mousewheel)
        # Only scroll when cursor is over the right panel
        def _bind_wheel(e):
            self._right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self._right_canvas.bind_all("<Button-4>", _on_mousewheel)
            self._right_canvas.bind_all("<Button-5>", _on_mousewheel)
        def _unbind_wheel(e):
            self._right_canvas.unbind_all("<MouseWheel>")
            self._right_canvas.unbind_all("<Button-4>")
            self._right_canvas.unbind_all("<Button-5>")
        right_outer.bind("<Enter>", _bind_wheel)
        right_outer.bind("<Leave>", _unbind_wheel)
        self._right_canvas.bind("<Enter>", _bind_wheel)
        self._right_canvas.bind("<Leave>", _unbind_wheel)

        columns = ("name", "sdk", "raw", "sim", "target", "err", "soft")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=24)
        headings = {"name": "Joint", "sdk": "SDK", "raw": "q_raw", "sim": "q_sim", "target": "target_sim", "err": "err", "soft": "soft limit(sim)"}
        widths = {"name": 140, "sdk": 42, "raw": 70, "sim": 70, "target": 75, "err": 65, "soft": 138}
        for c in columns:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor="center", stretch=False)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.rows = {}
        for name in self.mapper.joint_order:
            c = self.mapper.calib[name]
            soft = f"{fmt(c.get('soft_lower'))} ~ {fmt(c.get('soft_upper'))}"
            item = self.tree.insert("", "end", values=(name, c.get("sdk_index"), "--", "--", "--", "--", soft))
            self.rows[name] = item

        self._build_right(right)

        self.log_text = tk.Text(self, height=7, bg="#020617", fg="#d1d5db", relief="flat", font=(MONO_FAMILY, 10))
        self.log_text.pack(fill="x", padx=14, pady=(0, 10))
        self.log("UI ready. Workflow: Connect -> sdk_ok True -> ARM -> PD Stand -> Load Policy -> Start Policy.")
        self._log_contract_summary()

    def _build_right(self, right: ttk.Frame):
        def section(key: str):
            lbl = ttk.Label(right, style="Card.TLabel", font=(SANS_FAMILY, 12, "bold"))
            self._register_i18n(lbl, key)
            lbl.pack(anchor="w", padx=12, pady=(14, 4))
            return lbl

        section("safety_gate")
        chk = ttk.Checkbutton(right, variable=self.hang_confirm_var)
        self._register_i18n(chk, "hang_confirm")
        chk.pack(anchor="w", padx=12, pady=3)
        chk = ttk.Checkbutton(right, variable=self.arm_var, command=self._arm_changed)
        self._register_i18n(chk, "arm")
        chk.pack(anchor="w", padx=12, pady=3)
        chk = ttk.Checkbutton(right, variable=self.stand_confirm_var)
        self._register_i18n(chk, "stand_confirm")
        chk.pack(anchor="w", padx=12, pady=3)

        safety = ttk.Frame(right, style="Card.TFrame")
        safety.pack(fill="x", padx=12, pady=6)
        for i, (label, var) in enumerate([
            ("max roll", self.max_roll_var), ("max pitch", self.max_pitch_var), ("max temp", self.max_motor_temp_var)
        ]):
            ttk.Label(safety, text=label, style="Card.TLabel").grid(row=0, column=i*2, padx=3, pady=3)
            ttk.Entry(safety, textvariable=var, width=6).grid(row=0, column=i*2+1, padx=3)

        section("imu_cal")
        imuf = ttk.Frame(right, style="Card.TFrame")
        imuf.pack(fill="x", padx=12, pady=(0, 6))
        self.imu_status_var = tk.StringVar(value="IMU: not calibrated")
        b = ttk.Button(imuf, command=self.on_imu_calibrate)
        self._register_i18n(b, "imu_cal_btn")
        b.grid(row=0, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        ttk.Label(imuf, textvariable=self.imu_status_var, style="Card.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))
        ttk.Label(imuf, text="omega bias", style="Card.TLabel").grid(row=2, column=0, padx=4, sticky="w")
        self.imu_bias_label = ttk.Label(imuf, text="0.000 0.000 0.000", style="Card.TLabel")
        self.imu_bias_label.grid(row=2, column=1, columnspan=2, padx=4, sticky="w")

        filt = ttk.Frame(right, style="Card.TFrame")
        filt.pack(fill="x", padx=12, pady=(0, 4))
        l = ttk.Label(filt, style="Card.TLabel"); self._register_i18n(l, "imu_filter"); l.pack(side="left", padx=4)
        self.imu_filter_scale = tk.Scale(filt, from_=0.03, to=1.0, resolution=0.01,
                                         orient="horizontal", showvalue=False,
                                         variable=self.imu_filter_alpha_var,
                                         bg="#111827", fg="#e5e7eb", troughcolor="#374151",
                                         highlightthickness=0, sliderlength=16)
        self.imu_filter_scale.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(filt, textvariable=self.imu_filter_alpha_var,
                 bg="#1f2937", fg="#e5e7eb", font=(MONO_FAMILY, 9), width=4).pack(side="left", padx=(0, 4))
        tk.Label(filt, text="smooth ← | → fast", bg="#1f2937", fg="#9ca3af", font=(SANS_FAMILY, 8)).pack(side="left")

        imu_toggle = ttk.Frame(right, style="Card.TFrame")
        imu_toggle.pack(fill="x", padx=12, pady=(0, 4))
        chk = ttk.Checkbutton(imu_toggle, variable=self.imu_filter_enabled_var,
                              command=self._on_filter_toggle)
        self._register_i18n(chk, "imu_filter_enable")
        chk.pack(side="left", padx=4)

        cmd_filt = ttk.Frame(right, style="Card.TFrame")
        cmd_filt.pack(fill="x", padx=12, pady=(0, 4))
        l = ttk.Label(cmd_filt, style="Card.TLabel"); self._register_i18n(l, "cmd_smooth"); l.pack(side="left", padx=4)
        tk.Scale(cmd_filt, from_=0.03, to=1.0, resolution=0.01,
                 orient="horizontal", showvalue=False, variable=self.cmd_smooth_alpha_var,
                 bg="#111827", fg="#e5e7eb", troughcolor="#374151",
                 highlightthickness=0, sliderlength=16).pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(cmd_filt, textvariable=self.cmd_smooth_alpha_var,
                 bg="#1f2937", fg="#e5e7eb", font=(MONO_FAMILY, 9), width=4).pack(side="left", padx=(0, 4))

        cmd_toggle = ttk.Frame(right, style="Card.TFrame")
        cmd_toggle.pack(fill="x", padx=12, pady=(0, 4))
        chk = ttk.Checkbutton(cmd_toggle, variable=self.cmd_filter_enabled_var,
                              command=self._on_filter_toggle)
        self._register_i18n(chk, "cmd_filter_enable")
        chk.pack(side="left", padx=4)

        section("scale_contract")
        sc = ttk.Frame(right, style="Card.TFrame")
        sc.pack(fill="x", padx=12, pady=4)
        scale_lines = [
            ("action_scale", f"{self.action_scale:g}"),
            ("obs: omega", f"×{self.obs_scale_ang_vel:g}"),
            ("obs: gravity", f"×{self.obs_scale_projected_gravity:g}"),
            ("obs: q_sim-default", f"×{self.obs_scale_joint_pos:g}"),
            ("obs: dq_sim", f"×{self.obs_scale_joint_vel:g}"),
            ("policy input", f"{self.hist_len*self.obs_dim}/{self.obs_dim + self.hist_len*self.obs_dim}"),
        ]
        for r, (k, v) in enumerate(scale_lines):
            ttk.Label(sc, text=k, style="Card.TLabel").grid(row=r//2, column=(r%2)*2, sticky="w", padx=4, pady=1)
            ttk.Label(sc, text=v, style="Card.TLabel").grid(row=r//2, column=(r%2)*2+1, sticky="w", padx=4, pady=1)
        cmd_sc = ttk.Frame(right, style="Card.TFrame")
        cmd_sc.pack(fill="x", padx=12, pady=(0, 6))
        for i, (lab, var) in enumerate([("vx scale", self.cmd_vx_scale_var), ("vy", self.cmd_vy_scale_var), ("wz", self.cmd_wz_scale_var)]):
            ttk.Label(cmd_sc, text=lab, style="Card.TLabel").grid(row=0, column=i*2, padx=2, pady=2)
            ttk.Entry(cmd_sc, textvariable=var, width=5).grid(row=0, column=i*2+1, padx=2)

        section("pd_control")
        pd = ttk.Frame(right, style="Card.TFrame")
        pd.pack(fill="x", padx=12, pady=6)
        ttk.Label(pd, text="Pose Kp", style="Card.TLabel").grid(row=0, column=0, padx=4, pady=4)
        ttk.Entry(pd, textvariable=self.kp_var, width=7).grid(row=0, column=1, padx=4)
        ttk.Label(pd, text="Kd", style="Card.TLabel").grid(row=0, column=2, padx=4)
        ttk.Entry(pd, textvariable=self.kd_var, width=7).grid(row=0, column=3, padx=4)
        for key, cmd in [
            ("teach", self.on_estop), ("hold", self.on_hold_current), ("stand", self.on_stand),
            ("sit", self.on_sit), ("zero", self.on_record_zero), ("save", self.on_save_config),
        ]:
            b = ttk.Button(right, command=cmd)
            self._register_i18n(b, key)
            b.pack(fill="x", padx=12, pady=3)

        section("joystick")
        jf = ttk.Frame(right, style="Card.TFrame")
        jf.pack(fill="x", padx=12, pady=(0, 4))
        chk = ttk.Checkbutton(jf, variable=self.joystick_enabled_var)
        self._register_i18n(chk, "physical_joystick")
        chk.pack(side="left", padx=4)
        ttk.Label(jf, textvariable=self.joystick_status_var, style="Card.TLabel").pack(side="left", padx=8)
        ttk.Label(jf, text="A: E-STOP  B: zero", font=(SANS_FAMILY, 8), style="Card.TLabel").pack(side="right", padx=4)
        self.vjoy = VirtualJoystick(right,
            vx_var=self.vx_var, vy_var=self.vy_var, wz_var=self.wz_var,
            max_vx_var=self.max_vx_var, max_vy_var=self.max_vy_var, max_wz_var=self.max_wz_var,
            pad_size=170)
        self.vjoy.pack(padx=12, pady=(0, 4))
        lim = ttk.Frame(right, style="Card.TFrame")
        lim.pack(fill="x", padx=12, pady=(0, 6))
        for i, (lab, var) in enumerate([("vx_lim", self.max_vx_var), ("vy_lim", self.max_vy_var), ("yaw_lim", self.max_wz_var)]):
            ttk.Label(lim, text=lab, style="Card.TLabel").grid(row=0, column=i*2, padx=2, pady=2)
            ttk.Entry(lim, textvariable=var, width=6).grid(row=0, column=i*2+1, padx=2)

        section("policy")
        pol = ttk.Frame(right, style="Card.TFrame")
        pol.pack(fill="x", padx=12, pady=2)
        ttk.Entry(pol, textvariable=self.policy_path_var, width=35).grid(row=0, column=0, columnspan=3, sticky="ew", padx=3, pady=3)
        b = ttk.Button(pol, command=self.browse_policy); self._register_i18n(b, "browse"); b.grid(row=0, column=3, padx=3)
        for col, (key, cmd) in enumerate([("load_policy", self.load_policy), ("start_policy", self.start_policy), ("stop_policy", self.stop_policy)]):
            b = ttk.Button(pol, command=cmd); self._register_i18n(b, key); b.grid(row=1, column=col, padx=3, pady=3, sticky="ew")
        # Two different clips are shown separately:
        #   sim_clip = training/sim2sim clip_actions, normally 100.
        #   safe_guard = extra real-robot limiter before sending target to SDK.
        ttk.Label(pol, text=f"sim_clip={self.sim_clip_actions:g}", style="Card.TLabel").grid(row=2, column=0, columnspan=2, padx=2, pady=3, sticky="w")
        ttk.Checkbutton(pol, text="safe_guard", variable=self.action_guard_enabled_var).grid(row=2, column=2, padx=2, pady=3, sticky="w")
        ttk.Entry(pol, textvariable=self.action_clip_var, width=6).grid(row=2, column=3, padx=2)
        ttk.Checkbutton(pol, text="rate_limit", variable=self.policy_rate_limit_enabled_var).grid(row=2, column=4, padx=2, pady=3, sticky="w")
        ttk.Label(pol, text="Kp", style="Card.TLabel").grid(row=3, column=0, padx=2, pady=3)
        ttk.Entry(pol, textvariable=self.policy_kp_var, width=6).grid(row=3, column=1, padx=2)
        ttk.Label(pol, text="Kd", style="Card.TLabel").grid(row=3, column=2, padx=2, pady=3)
        ttk.Entry(pol, textvariable=self.policy_kd_var, width=6).grid(row=3, column=3, padx=2)
        ttk.Label(right, textvariable=self.policy_status_var, style="Card.TLabel").pack(anchor="w", padx=12, pady=4)

        section("robot_state")
        self.info_text = tk.Text(right, height=8, bg="#0f172a", fg="#d1d5db", insertbackground="#fff", relief="flat", font=(MONO_FAMILY, 8))
        self.info_text.pack(fill="x", padx=12, pady=6)

    def _bind_keys(self):
        self.bind("<space>", lambda e: self.on_estop())
        self.bind("w", lambda e: self.bump_vel("vx", +1))
        self.bind("s", lambda e: self.bump_vel("vx", -1))
        self.bind("a", lambda e: self.bump_vel("vy", +1))
        self.bind("d", lambda e: self.bump_vel("vy", -1))
        self.bind("q", lambda e: self.bump_vel("wz", +1))
        self.bind("e", lambda e: self.bump_vel("wz", -1))
        self.bind("x", lambda e: self.zero_vel())

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")

    def _log_contract_summary(self):
        """Print the exact sim2sim/deploy mapping used by the UI."""
        try:
            self.log("=== SIM2SIM CONTRACT SUMMARY ===")
            self.log(f"action: q_des_sim = default + clip(policy_action, ±{self.sim_clip_actions:g}) * {self.action_scale:g}")
            self.log(f"real safety guard: enabled={self.action_guard_enabled_var.get()} clip={self.action_clip_var.get():g}; rate_limit={self.policy_rate_limit_enabled_var.get()}")
            self.log(
                "obs order: "
                f"omega*{self.obs_scale_ang_vel:g}, "
                f"gravity*{self.obs_scale_projected_gravity:g}, "
                "cmd*[vx,vy,wz scales], "
                f"(q_sim-default)*{self.obs_scale_joint_pos:g}, "
                f"dq_sim*{self.obs_scale_joint_vel:g}, "
                f"last_action*{self.obs_scale_last_action:g}"
            )
            seen = set()
            for i, name in enumerate(self.mapper.joint_order):
                c = self.mapper.calib.get(name, {})
                idx = c.get("sdk_index")
                if idx in seen:
                    self.log(f"[WARN] duplicate sdk_index {idx} for action[{i}] {name}")
                seen.add(idx)
                self.log(f"action[{i:02d}] -> {name:<20s} -> sdk[{idx}] default={float(self.default_pose[name]):+.4f} scale={float(c.get('scale',1.0)):+.4f} offset={float(c.get('offset',0.0)):+.4f}")
        except Exception as exc:
            self.log(f"Contract summary failed: {exc}")

    def on_connect(self):
        try:
            self.client.connect(self.host_var.get().strip(), int(self.port_var.get()))
            self.status_var.set("Connected to bridge")
            self.log("Connected to bridge.")
        except Exception as exc:
            self.status_var.set("Connection failed")
            messagebox.showerror("Connect failed", str(exc))

    def on_disconnect(self):
        self.on_estop()
        self.client.close()
        self.status_var.set("Disconnected")
        self.log("Disconnected.")

    def on_estop(self):
        self.active = False
        self.policy_active = False
        self.pose_mode = "DAMPING"
        self.arm_var.set(False)
        try:
            self.client.send({"cmd": "estop"})
        except Exception:
            pass
        self.policy_status_var.set("Policy: stopped / damping")
        self.log("Emergency damping sent. PD/policy disabled.")

    def _arm_changed(self):
        if self.arm_var.get():
            if not self.hang_confirm_var.get():
                self.arm_var.set(False)
                messagebox.showwarning("Safety gate", "必须先确认机器狗已吊起/支撑，并且周围安全。")
                return
            if not self._sdk_ready(show=False):
                self.arm_var.set(False)
                messagebox.showwarning("SDK not ready", "sdk_ok=False 或没有 high-rate joint position，禁止 ARM。")
                return
            self.log("ARM enabled. Real joint commands are allowed.")
        else:
            self.active = False
            self.policy_active = False
            try:
                self.client.send({"cmd": "zero_joint"})
            except Exception:
                pass
            self.log("ARM disabled -> zero_joint.")

    def _sdk_ready(self, show: bool = True) -> bool:
        ok = bool(self.state.get("sdk_ok", False)) and len(self.last_raw) >= 12 and self.state.get("mode") in ("joint_high_rate", "joint_high_rate_partial", "joint_normal_rate", "mock")
        if not ok and show:
            messagebox.showwarning("SDK not ready", f"当前 SDK 未就绪，不能发关节命令。\nmode={self.state.get('mode')}\nerror={self.state.get('error')}")
        return ok

    def _safety_ok(self, show: bool = False) -> bool:
        if not self._sdk_ready(show=show):
            return False
        rpy = self.state.get("imu_rpy", [0.0, 0.0, 0.0])
        if len(rpy) < 3 or not finite_list(rpy[:3]):
            if show:
                messagebox.showwarning("IMU invalid", "IMU RPY 数据无效。")
            return False
        roll, pitch = abs(float(rpy[0])), abs(float(rpy[1]))
        # Raw RPY is still monitored. If calibrated, also use corrected gravity tilt as a frame-independent guard.
        if self.imu_calibrated:
            q = self._corrected_quat(self.state.get("imu_quat", [0,0,0,1])[:4])
            g = vec_norm3(quat_rotate_inverse_xyzw(q, [0.0, 0.0, -1.0]))
            tilt = math.acos(clamp(-float(g[2]), -1.0, 1.0))
            if tilt > max(float(self.max_roll_var.get()), float(self.max_pitch_var.get())):
                self.log(f"IMU corrected tilt abnormal -> estop: tilt={tilt:.3f}, gravity={g}")
                self.on_estop()
                return False
        if roll > float(self.max_roll_var.get()) or pitch > float(self.max_pitch_var.get()):
            self.log(f"IMU raw RPY abnormal -> estop: roll={roll:.3f}, pitch={pitch:.3f}")
            self.on_estop()
            return False
        temps = self.state.get("motor_temp", [])
        if temps and max(float(x) for x in temps[:12]) > float(self.max_motor_temp_var.get()):
            self.log("Motor temperature too high -> estop")
            self.on_estop()
            return False
        return True

    def _get_current_sim(self) -> Dict[str, float]:
        return self.mapper.raw_array_to_sim_by_name(self.last_raw)

    def _set_pose_target(self, target: Dict[str, float], mode: str):
        if not self.hang_confirm_var.get():
            messagebox.showwarning("Safety gate", "先确认机器狗已吊起/支撑。")
            return
        if not self.arm_var.get():
            messagebox.showwarning("ARM required", "先勾选 ARM real joint command。")
            return
        if not self._safety_ok(show=True):
            return
        current = self._get_current_sim()
        self.current_sim_target = dict(current)
        self.final_sim_target = self.mapper.clamp_sim_targets({name: float(target[name]) for name in self.mapper.joint_order})
        self.active = True
        self.policy_active = False
        self.pose_mode = mode
        self.log(f"Start PD streaming pose: {mode}")

    def on_hold_current(self):
        current = self._get_current_sim()
        self._set_pose_target(current, "HOLD_CURRENT")

    def on_stand(self):
        if not self.imu_calibrated:
            self.log("Stand requested -> running mandatory pre-stand IMU calibration first.")
            self.on_imu_calibrate(auto_stand=True)
            return
        # Strict replay stand pose is the same DEFAULT_JOINT_POS used by sim2sim.
        # This prevents accidental use of an older IsaacLab init pose with hip_roll +/-0.08.
        target = self.default_pose
        self._set_pose_target(target, "STAND_STANDARD_SIM2SIM_DEFAULT")

    def on_sit(self):
        target = self.mapper.poses["poses"]["sit_crouch"]["joints"]
        self._set_pose_target(target, "SIT_CROUCH")

    def on_record_zero(self):
        if not self._sdk_ready(show=True):
            return
        if not messagebox.askyesno("Software zero", "确认当前真机已经摆成趴下标零姿态？\n这不会调用厂家硬件标零，只会更新软件 offset。"):
            return
        self.mapper.record_crouch_zero_from_raw(self.last_raw)
        self.mapper.save_working_config()
        self.log(f"Software crouch zero recorded and saved: {WORKING_CONFIG_PATH}")
        messagebox.showinfo("Saved", f"已保存软件标零到:\n{WORKING_CONFIG_PATH}")

    def on_save_config(self):
        self.mapper.save_working_config()
        self.log(f"Working config saved: {WORKING_CONFIG_PATH}")

    def _corrected_quat(self, raw_quat: List[float]) -> List[float]:
        # q_rel = inv(q_level) * q_raw, so a level robot becomes identity orientation.
        q = quat_normalize_xyzw(raw_quat[:4] if len(raw_quat) >= 4 else [0.0, 0.0, 0.0, 1.0])
        if not self.imu_calibrated:
            return q
        return quat_normalize_xyzw(quat_mul_xyzw(self.imu_quat_correction, q))

    def on_imu_calibrate(self, auto_stand: bool = False):
        if self.imu_calib_active:
            return
        if not self._sdk_ready(show=True):
            return
        self.pending_stand_after_calib = bool(auto_stand)
        self.imu_calib_active = True
        self.imu_calibrated = False
        self.imu_calib_samples_quat = []
        self.imu_calib_samples_omega = []
        self.imu_calib_end_time = time.time() + 2.0
        self.imu_status_var.set("IMU: calibrating 2.0s, keep robot still...")
        self.log("IMU calibration started: keep robot still and level for 2 seconds.")
        self.after(20, self._imu_calib_tick)

    def _imu_calib_tick(self):
        if not self.imu_calib_active:
            return
        q = self.state.get("imu_quat", [0.0, 0.0, 0.0, 1.0])[:4]
        w = self.state.get("imu_omega", [0.0, 0.0, 0.0])[:3]
        if len(q) >= 4 and len(w) >= 3 and finite_list(q[:4]) and finite_list(w[:3]):
            self.imu_calib_samples_quat.append([float(x) for x in q[:4]])
            self.imu_calib_samples_omega.append([float(x) for x in w[:3]])
        remain = self.imu_calib_end_time - time.time()
        if remain > 0:
            self.imu_status_var.set(f"IMU: calibrating {remain:.1f}s, samples={len(self.imu_calib_samples_quat)}")
            self.after(20, self._imu_calib_tick)
            return

        self.imu_calib_active = False
        if len(self.imu_calib_samples_quat) < 20:
            self.imu_status_var.set("IMU: calibration failed, too few samples")
            self.log("IMU calibration failed: too few valid samples.")
            return
        self.imu_level_quat = avg_quat_xyzw(self.imu_calib_samples_quat)
        self.imu_quat_correction = quat_inv_xyzw(self.imu_level_quat)
        self.imu_omega_bias = avg_vec3(self.imu_calib_samples_omega)
        omega_std = std_vec3(self.imu_calib_samples_omega, self.imu_omega_bias)
        self.imu_calibrated = True
        self._filt_omega = [0.0, 0.0, 0.0]
        self._filt_gravity = [0.0, 0.0, -1.0]
        self.imu_bias_label.configure(text=f"{self.imu_omega_bias[0]:+.4f} {self.imu_omega_bias[1]:+.4f} {self.imu_omega_bias[2]:+.4f}")
        self.imu_status_var.set(f"IMU: calibrated | std={omega_std[0]:.3f},{omega_std[1]:.3f},{omega_std[2]:.3f}")
        self.log(f"IMU calibrated. level_quat={self.imu_level_quat}, omega_bias={self.imu_omega_bias}, omega_std={omega_std}")
        # Save into working config so the next session can start from the same reference if desired.
        try:
            self.mapper.cfg.setdefault("calibration", {}).setdefault("imu", {})["level_quat_raw_xyzw"] = list(self.imu_level_quat)
            self.mapper.cfg.setdefault("calibration", {}).setdefault("imu", {})["omega_bias"] = list(self.imu_omega_bias)
            self.mapper.save_working_config()
        except Exception as exc:
            self.log(f"IMU calibration saved in memory, config save failed: {exc}")
        if self.pending_stand_after_calib:
            self.pending_stand_after_calib = False
            self.after(150, self.on_stand)

    def _normalize_policy_output(self, y):
        """Return the first tensor-like policy output as a flat tensor.

        Some exported policies return a tensor directly, while a few wrappers return
        a tuple/list where the first element is the action tensor.  The real robot
        only accepts the final 12-D action vector.
        """
        if isinstance(y, (tuple, list)):
            if not y:
                raise RuntimeError("policy returned an empty tuple/list")
            y = y[0]
        if not hasattr(y, "detach"):
            raise RuntimeError(f"policy returned unsupported type: {type(y)}")
        return y.detach().cpu().reshape(-1)

    def _policy_candidate_inputs(self):
        """Candidate TorchScript input signatures for exported DWAQ policies.

        Two export styles are common in this project.  Their dimensions are derived
        from the active config so IsaacLab policies with different history lengths
        can be loaded without changing UI code.
        """
        hist_dim = self.hist_len * self.obs_dim
        candidates = [
            ("history_only", hist_dim, f"obs_hist only: {self.obs_dim}×{self.hist_len}={hist_dim}"),
            (
                "current_plus_history",
                self.obs_dim + hist_dim,
                f"current_obs + obs_hist: {self.obs_dim}+{hist_dim}={self.obs_dim + hist_dim}",
            ),
        ]
        mode = getattr(self, "policy_input_mode_cfg", "auto")
        if mode == "history_only":
            candidates = [candidates[0]]
        elif mode == "current_plus_history":
            candidates = [candidates[1]]
        return candidates

    def _build_policy_input_vector(self, current_obs: List[float], obs_hist_flat: List[float]) -> List[float]:
        mode = getattr(self, "policy_input_mode", "history_only")
        if mode == "history_only":
            return list(obs_hist_flat)
        if mode == "current_plus_history":
            return list(current_obs) + list(obs_hist_flat)
        raise RuntimeError(f"Unknown policy input mode: {mode}. Re-load policy first.")

    def browse_policy(self):
        p = filedialog.askopenfilename(title="Select TorchScript policy", filetypes=[("Torch policy", "*.pt *.jit"), ("All files", "*")])
        if p:
            self.policy_path_var.set(p)

    def load_policy(self):
        if torch is None:
            messagebox.showerror("Torch missing", "当前 Python 环境没有 torch。请在 isaaclab/torch 环境运行 UI。")
            return
        path = Path(self.policy_path_var.get()).expanduser()
        if not path.exists():
            messagebox.showerror("Policy not found", f"找不到模型文件:\n{path}")
            return
        try:
            torch.set_num_threads(1)
            policy = torch.jit.load(os.fspath(path), map_location="cpu")
            policy.eval()

            errors = []
            detected = None
            detected_output = None
            for mode, dim, desc in self._policy_candidate_inputs():
                x = torch.zeros(1, dim, dtype=torch.float32)
                try:
                    with torch.no_grad():
                        y = policy(x)
                    y = self._normalize_policy_output(y)
                    if y.numel() != 12:
                        raise RuntimeError(f"output dim={y.numel()}, expected 12")
                    detected = (mode, dim, desc)
                    detected_output = y
                    break
                except Exception as exc:
                    msg = str(exc).splitlines()
                    errors.append(f"- {desc} / input={dim}: {msg[-1] if msg else exc}")

            if detected is None:
                detail = "\n".join(errors[-8:])
                raise RuntimeError(
                    "Policy signature mismatch.\n"
                    "当前 UI 已支持两种 DWAQ 导出格式：\n"
                    f"1) history_only: {self.hist_len}×{self.obs_dim}={self.hist_len*self.obs_dim} 维\n"
                    f"2) current_plus_history: {self.obs_dim}+{self.hist_len*self.obs_dim}={self.obs_dim + self.hist_len*self.obs_dim} 维\n"
                    "但这个模型两种都跑不通。请确认不是训练 checkpoint model_xxxx.pt，"
                    "而是导出的 TorchScript policy。\n\n"
                    f"TorchScript dry-run errors:\n{detail}"
                )

            mode, dim, desc = detected
            self.policy = policy
            self.policy_input_mode = mode
            self.policy_input_dim = dim
            self.policy_loaded_path = os.fspath(path)
            self.obs_history = []
            self.last_action = [0.0] * 12
            exact_note = "matches active config contract" if mode == "history_only" else "accepted current+history wrapper format"
            self.policy_status_var.set(f"Policy loaded: {path.name} | {mode} | in={dim} out=12 | {exact_note}")
            self.log(f"Policy loaded: {path}, mode={mode}, input={dim}, output={detected_output.numel()} ({desc}); {exact_note}")
        except Exception as exc:
            self.policy = None
            self.policy_input_mode = "unknown"
            self.policy_input_dim = 0
            self.policy_status_var.set("Policy load failed")
            messagebox.showerror("Load policy failed", str(exc))

    def start_policy(self):
        if self.policy is None:
            messagebox.showwarning("No policy", "先 Load Policy。")
            return
        if not self.hang_confirm_var.get() or not self.arm_var.get():
            messagebox.showwarning("Safety gate", "先确认吊起/支撑，并 ARM。")
            return
        if not self.imu_calibrated:
            messagebox.showwarning("IMU calibration", self.tr("imu_required"))
            return
        if not self.stand_confirm_var.get():
            messagebox.showwarning("Stand first", "必须先通过 PD 标准站姿，并勾选确认已站稳。")
            return
        if not self._safety_ok(show=True):
            return
        # Reset filter state for fresh start
        self._filt_omega = [0.0, 0.0, 0.0]
        self._filt_gravity = [0.0, 0.0, -1.0]
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_wz = 0.0
        obs = self.build_policy_obs()
        self.obs_history = [list(obs) for _ in range(self.hist_len)]
        self.last_action = [0.0] * 12
        current = self._get_current_sim()
        self.current_sim_target = dict(current)
        self.final_sim_target = dict(current)
        self.policy_active = True
        self.active = False
        self.pose_mode = "POLICY_ACTIVE"
        self.policy_status_var.set("Policy: ACTIVE")
        self.log("Policy control started. Velocity command is controlled by UI buttons/keyboard.")

    def stop_policy(self):
        self.policy_active = False
        self.active = False
        self.pose_mode = "POLICY_STOPPED"
        self.zero_vel()
        try:
            self.client.send({"cmd": "zero_joint"})
        except Exception:
            pass
        self.policy_status_var.set("Policy: stopped")
        self.log("Policy stopped -> zero_joint.")

    def bump_vel(self, key: str, sign: int):
        step = float(self.vel_step_var.get())
        if key == "vx":
            self.vx_var.set(clamp(self.vx_var.get() + sign * step, -self.max_vx_var.get(), self.max_vx_var.get()))
        elif key == "vy":
            self.vy_var.set(clamp(self.vy_var.get() + sign * step, -self.max_vy_var.get(), self.max_vy_var.get()))
        elif key == "wz":
            self.wz_var.set(clamp(self.wz_var.get() + sign * step, -self.max_wz_var.get(), self.max_wz_var.get()))
        self.log(f"cmd vel: vx={self.vx_var.get():.3f}, vy={self.vy_var.get():.3f}, wz={self.wz_var.get():.3f}")

    def zero_vel(self):
        self.vx_var.set(0.0)
        self.vy_var.set(0.0)
        self.wz_var.set(0.0)
        self.log("Velocity command zeroed.")

    def _apply_ema(self, prev: List[float], raw: List[float], alpha: float) -> List[float]:
        """Single-step exponential moving average."""
        if alpha >= 1.0:
            return list(raw)
        out = prev[:]
        for i in range(len(out)):
            out[i] = alpha * raw[i] + (1.0 - alpha) * prev[i]
        return out

    def _apply_imu_filter(self, raw_omega, raw_gravity):
        """Low-pass filter for IMU data. Toggle OFF = pass-through."""
        if not self.imu_filter_enabled_var.get():
            self._filt_omega = list(raw_omega)
            self._filt_gravity = list(raw_gravity)
            return self._filt_omega, self._filt_gravity
        alpha = clamp(float(self.imu_filter_alpha_var.get()), 0.01, 1.0)
        self._filt_omega = self._apply_ema(self._filt_omega, raw_omega, alpha)
        self._filt_gravity = self._apply_ema(self._filt_gravity, raw_gravity, alpha)
        return self._filt_omega, self._filt_gravity

    def _smooth_cmd(self):
        """Apply EMA smoothing to velocity commands. Toggle OFF = pass-through."""
        raw = [float(self.vx_var.get()), float(self.vy_var.get()), float(self.wz_var.get())]
        if not self.cmd_filter_enabled_var.get():
            self._smooth_vx, self._smooth_vy, self._smooth_wz = raw
            return
        alpha = clamp(float(self.cmd_smooth_alpha_var.get()), 0.01, 1.0)
        sm = self._apply_ema([self._smooth_vx, self._smooth_vy, self._smooth_wz], raw, alpha)
        self._smooth_vx, self._smooth_vy, self._smooth_wz = sm

    def _on_filter_toggle(self):
        """Log when filter toggles change state."""
        imu_on = self.imu_filter_enabled_var.get()
        cmd_on = self.cmd_filter_enabled_var.get()
        self.log(f"Filter: IMU={'ON' if imu_on else 'OFF'}, Cmd={'ON' if cmd_on else 'OFF'}")

    def build_policy_obs(self) -> List[float]:
        raw = self.last_raw
        dq_raw = self.last_dq_raw
        qsim = self.mapper.raw_array_to_sim_by_name(raw)
        dqsim = self.mapper.raw_vel_array_to_sim_by_name(dq_raw)
        omega = self.state.get("imu_omega", [0.0, 0.0, 0.0])[:3]
        if len(omega) < 3:
            omega = [0.0, 0.0, 0.0]
        # Remove static gyro bias measured during pre-stand calibration.
        omega = [float(omega[i]) - float(self.imu_omega_bias[i]) for i in range(3)]
        quat = self.state.get("imu_quat", [0.0, 0.0, 0.0, 1.0])[:4]
        quat = self._corrected_quat(quat)
        gravity = quat_rotate_inverse_xyzw(quat, [0.0, 0.0, -1.0])
        gravity = vec_norm3(gravity)
        # Apply low-pass filter to IMU readings and re-normalize gravity after filtering.
        omega, gravity = self._apply_imu_filter(omega, gravity)
        gravity = vec_norm3(gravity)
        # Apply low-pass filter to velocity commands
        self._smooth_cmd()
        cmd = [self._smooth_vx, self._smooth_vy, self._smooth_wz]
        obs: List[float] = []
        obs += [float(x) * self.obs_scale_ang_vel for x in omega]
        obs += [float(x) * self.obs_scale_projected_gravity for x in gravity]
        obs += [cmd[0] * float(self.cmd_vx_scale_var.get()),
                cmd[1] * float(self.cmd_vy_scale_var.get()),
                cmd[2] * float(self.cmd_wz_scale_var.get())]
        for name in self.mapper.joint_order:
            obs.append((float(qsim.get(name, self.default_pose[name])) - float(self.default_pose[name])) * self.obs_scale_joint_pos)
        for name in self.mapper.joint_order:
            obs.append(float(dqsim.get(name, 0.0)) * self.obs_scale_joint_vel)
        obs += [float(x) * self.obs_scale_last_action for x in self.last_action]
        if len(obs) != self.obs_dim:
            raise RuntimeError(f"obs dim mismatch: got {len(obs)}, expected {self.obs_dim}")
        return obs

    def infer_policy_target(self) -> Dict[str, float]:
        if self.policy is None or torch is None:
            raise RuntimeError("policy not loaded")
        obs = self.build_policy_obs()
        if not self.obs_history:
            self.obs_history = [list(obs) for _ in range(self.hist_len)]
        self.obs_history.append(list(obs))
        self.obs_history = self.obs_history[-self.hist_len:]
        flat: List[float] = []
        for frame in self.obs_history:
            flat.extend(frame)
        policy_input = self._build_policy_input_vector(obs, flat)
        expected_dim = int(getattr(self, "policy_input_dim", len(policy_input)) or len(policy_input))
        if len(policy_input) != expected_dim:
            raise RuntimeError(f"policy input dim mismatch: got {len(policy_input)}, expected {expected_dim}, mode={getattr(self, 'policy_input_mode', 'unknown')}")
        x = torch.tensor(policy_input, dtype=torch.float32).view(1, -1)
        t0 = time.time()
        with torch.no_grad():
            y = self._normalize_policy_output(self.policy(x))
        self.policy_last_ms = (time.time() - t0) * 1000.0
        if y.numel() != 12:
            raise RuntimeError(f"policy output dim={y.numel()}, expected 12")
        # 1) Exact sim2sim/training action clip. For your current demo this is normally 100,
        #    which almost never limits normal policy output. This is the REAL meaning of clip_actions.
        raw_action = [float(v) for v in y.tolist()]
        sim_clip = abs(float(self.sim_clip_actions))
        sim_action = [clamp(v, -sim_clip, sim_clip) for v in raw_action]

        # 2) Optional real-robot safety guard. This is NOT sim2sim clip_actions.
        #    It is an extra limiter used only to keep the first real tests small.
        sent_action = list(sim_action)
        if self.action_guard_enabled_var.get():
            guard = abs(float(self.action_clip_var.get()))
            sent_action = [clamp(v, -guard, guard) for v in sim_action]

        self.last_policy_action_raw = raw_action
        self.last_policy_action_sim = sim_action
        self.last_policy_action_sent = sent_action

        # For the next obs, last_action should represent the action that actually generated the commanded target.
        # Disable safe_guard for strict sim2sim replay, then sent_action == sim_action.
        self.last_action = list(sent_action)

        target = {}
        for i, name in enumerate(self.mapper.joint_order):
            target[name] = float(self.default_pose[name]) + sent_action[i] * self.action_scale
        target = self.mapper.clamp_sim_targets(target)
        return target

    def _poll_state(self):
        while True:
            try:
                st = self.client.state_q.get_nowait()
            except queue.Empty:
                break
            self.state = st
            raw = st.get("joint_pos", [])
            dq = st.get("joint_vel", [])
            if len(raw) >= 12:
                self.last_raw = [float(x) for x in raw[:12]]
            if len(dq) >= 12:
                self.last_dq_raw = [float(x) for x in dq[:12]]
            self._update_display(st)
            try:
                log_st = dict(st)
                log_st["ui_pose_mode"] = self.pose_mode
                log_st["cmd_vel"] = [self.vx_var.get(), self.vy_var.get(), self.wz_var.get()]
                log_st["last_action"] = self.last_action
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_st) + "\n")
            except Exception:
                pass
        if self.client.connected:
            self.status_var.set(f"Bridge OK | mode={self.state.get('mode','--')} | seq={self.state.get('seq','--')} | control={self.pose_mode}")
        else:
            if self.client.last_error:
                self.status_var.set(f"Disconnected: {self.client.last_error}")
        self.after(30, self._poll_state)

    def _update_display(self, st: dict):
        raw = st.get("joint_pos", [])
        sim_by = self.mapper.raw_array_to_sim_by_name(raw) if len(raw) >= 12 else {}
        for name in self.mapper.joint_order:
            c = self.mapper.calib[name]
            idx = c.get("sdk_index")
            qraw = raw[idx] if idx is not None and idx < len(raw) else None
            qsim = sim_by.get(name)
            target = self.current_sim_target.get(name)
            err = None if qsim is None or target is None else target - qsim
            soft = f"{fmt(c.get('soft_lower'))} ~ {fmt(c.get('soft_upper'))}"
            self.tree.item(self.rows[name], values=(name, idx, fmt(qraw), fmt(qsim), fmt(target), fmt(err), soft))
        rpy = st.get("imu_rpy", [0,0,0])
        temps = st.get("motor_temp", [])
        max_temp = max(temps) if temps else 0
        text = (
            f"source: {st.get('source')}\n"
            f"connected: {st.get('connected')}  sdk_ok: {st.get('sdk_ok')}\n"
            f"mode: {st.get('mode')}\n"
            f"error: {st.get('error')}\n"
            f"seq: {st.get('seq')}\n"
            f"battery: {st.get('battery_level')}%  current: {fmt(st.get('battery_current'),2)} A\n"
            f"rpy(rad): {fmt(rpy[0])}, {fmt(rpy[1])}, {fmt(rpy[2])}\n"
            f"cmd: vx={self.vx_var.get():.2f}→{self._smooth_vx:.2f}, vy={self.vy_var.get():.2f}→{self._smooth_vy:.2f}, wz={self.wz_var.get():.2f}→{self._smooth_wz:.2f}\n"
            f"max motor temp: {fmt(max_temp,1)} C\n"
            f"pose_mode: {self.pose_mode}\n"
            f"policy ms: {self.policy_last_ms:.2f}\n"
            f"policy clip: sim_clip={self.sim_clip_actions:g}, safe_guard={'ON' if self.action_guard_enabled_var.get() else 'OFF'} {self.action_clip_var.get():.3f}, rate_limit={self.policy_rate_limit_enabled_var.get()}\n"
            f"action raw/sent max: {max([abs(x) for x in self.last_policy_action_raw] or [0]):.3f} / {max([abs(x) for x in self.last_policy_action_sent] or [0]):.3f}\n"
            f"joystick: {'ON' if self.joystick_enabled_var.get() else 'OFF'} ({self.joystick_status_var.get()})\n"
            f"log: {self.log_file.name}\n"
        )
        self.info_text.delete("1.0", "end")
        self.info_text.insert("end", text)

    def _stream_target(self, target_by_name: Dict[str, float], kp_value: float, kd_value: float):
        pos_raw = self.mapper.sim_targets_to_sdk_raw_array(target_by_name, self.last_raw)
        kp = [0.0] * 12
        kd = [0.0] * 12
        vel = [0.0] * 12
        tff = [0.0] * 12
        for name in self.mapper.joint_order:
            idx = self.mapper.calib[name].get("sdk_index")
            if idx is not None:
                kp[int(idx)] = float(kp_value)
                kd[int(idx)] = float(kd_value)
        self.client.send({"cmd": "joint", "kp": kp, "pos": pos_raw, "kd": kd, "vel": vel, "tff": tff})

    def _control_loop(self):
        try:
            if (self.active or self.policy_active) and self.client.connected:
                if not self._safety_ok(show=False):
                    self.active = False
                    self.policy_active = False
                elif self.active and self.arm_var.get():
                    done = True
                    current_sim = self._get_current_sim()
                    for name in self.mapper.joint_order:
                        cur = float(self.current_sim_target.get(name, current_sim.get(name, 0.0)))
                        fin = float(self.final_sim_target[name])
                        delta = fin - cur
                        step = clamp(delta, -self.mapper.step_sim, self.mapper.step_sim)
                        nxt = cur + step
                        self.current_sim_target[name] = nxt
                        if abs(fin - nxt) > 1e-3:
                            done = False
                    self._stream_target(self.current_sim_target, self.kp_var.get(), self.kd_var.get())
                elif self.policy_active and self.arm_var.get():
                    raw_target = self.infer_policy_target()
                    if not self.current_sim_target:
                        self.current_sim_target = self._get_current_sim()
                    # Optional extra target rate-limit in sim space. Disable for strict sim2sim replay.
                    if self.policy_rate_limit_enabled_var.get():
                        for name in self.mapper.joint_order:
                            cur = float(self.current_sim_target.get(name, raw_target[name]))
                            delta = float(raw_target[name]) - cur
                            self.current_sim_target[name] = cur + clamp(delta, -self.mapper.step_sim, self.mapper.step_sim)
                    else:
                        self.current_sim_target = dict(raw_target)
                    self._stream_target(self.current_sim_target, self.policy_kp_var.get(), self.policy_kd_var.get())
                    guard_txt = f"guard={self.action_clip_var.get():.2f}" if self.action_guard_enabled_var.get() else "guard=OFF"
                    self.policy_status_var.set(f"Policy ACTIVE | sim_clip={self.sim_clip_actions:g} | {guard_txt} | rate={self.policy_rate_limit_enabled_var.get()} | ms={self.policy_last_ms:.2f}")
        except Exception as exc:
            self.active = False
            self.policy_active = False
            self.arm_var.set(False)
            try:
                self.client.send({"cmd": "estop"})
            except Exception:
                pass
            self.log(f"Control error -> estop: {exc}")
            messagebox.showerror("Control error", str(exc))
        self.after(int(1000 / max(1, self.command_rate_hz)), self._control_loop)

    def _joystick_poll(self):
        """Periodically push physical joystick values into UI velocity vars (10 Hz)."""
        try:
            # Skip physical joystick if virtual joystick thumb is active (being dragged)
            if not (hasattr(self, 'vjoy') and self.vjoy.dragging):
                self.joystick.apply_to_app()
        except Exception:
            pass
        # Update status if joystick state changed.
        if HAS_INPUTS:
            was = self.joystick_status_var.get()
            now_avail = JoystickControl.available()
            if now_avail and "not detected" in was:
                self.joystick.start()
                self.joystick_status_var.set("Joystick: connected")
                self.log("Joystick connected.")
            elif not now_avail and "connected" in was:
                self.joystick.stop()
                self.joystick_status_var.set("Joystick: not detected")
                self.log("Joystick disconnected.")
        self.after(100, self._joystick_poll)

    def destroy(self):
        try:
            self.joystick.stop()
        except Exception:
            pass
        try:
            self.on_estop()
            self.client.close()
        finally:
            super().destroy()


if __name__ == "__main__":
    App().mainloop()

import sys
import time
import csv
import os
import math
import queue
import threading
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple

import serial
import serial.tools.list_ports

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# -----------------------------
# Utils
# -----------------------------
def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default


def parse_kv_pairs(parts, start_idx):
    out = {}
    kv = parts[start_idx:]
    for i in range(0, len(kv) - 1, 2):
        out[kv[i]] = kv[i + 1]
    return out


def parse_line(line: str):
    """
    New firmware examples:
      T,12345,VCC_MV,5002,ADC,611.4,V,2.9876,PH,-1.000,ACT,0,CALMASK,7,SEG,RAWV
      S,VCC_MV,5002,ADC,611.4,V,2.9876,PH,-1.000,CAL,1,CALMASK,7,SEG,RAWV
      CAL,STATUS,MASK,7,V4,3.2000,V7,2.5000,V10,1.7000,A,-4.285714,B,17.714286
      BOOT,OK
      MODE,RAW_VOLTAGE_ONLY
      SAMPLE_HZ,10
      TX_HZ,2
      OK,...
    """
    line = (line or "").strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return {"type": "RAW", "raw": line}

    head = parts[0].upper()

    if head == "T":
        out = {"type": "T", "raw": line}
        out["ms"] = to_int(parts[1], None)
        out.update(parse_kv_pairs(parts, 2))
        return out

    if head == "S":
        out = {"type": "S", "raw": line}
        out.update(parse_kv_pairs(parts, 1))
        return out

    if head == "CAL" and len(parts) >= 2 and parts[1].upper() == "STATUS":
        out = {"type": "CAL_STATUS", "raw": line}
        out.update(parse_kv_pairs(parts, 2))
        return out

    if head in ("OK", "ERR", "BOOT", "HELP", "MODE", "SAMPLE_HZ", "TX_HZ"):
        return {"type": head, "raw": line}

    return {"type": "RAW", "raw": line}


def calmask_to_mode(mask: int):
    if mask is None:
        return "UNKNOWN"
    has4 = (mask & 0x01) != 0
    has7 = (mask & 0x02) != 0
    has10 = (mask & 0x04) != 0
    if has4 and has7 and has10:
        return "3PT (4/7/10)"
    if has4 and has7:
        return "2PT (7/4)"
    if has7 and has10:
        return "2PT (7/10)"
    if has4 and has10:
        return "2PT (4/10)"
    if has7:
        return "1PT (7 only)"
    return "UNCAL"


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def calc_line(v1: float, p1: float, v2: float, p2: float) -> Optional[Tuple[float, float]]:
    if not (is_finite_number(v1) and is_finite_number(v2)):
        return None
    if abs(v2 - v1) < 1e-12:
        return None
    m = (p2 - p1) / (v2 - v1)
    b = p1 - m * v1
    return m, b


def calc_ph_from_voltage(v: Optional[float],
                         mask: Optional[int],
                         v4: Optional[float],
                         v7: Optional[float],
                         v10: Optional[float]) -> Tuple[Optional[float], str]:
    """
    Python-side pH conversion:
      - 3-point: piecewise linear (4-7 / 7-10)
      - 2-point fallback: 7/4, 7/10, 4/10
    """
    if not is_finite_number(v):
        return None, "NOV"

    if mask is None:
        return None, "UNCAL"

    has4 = (mask & 0x01) != 0 and is_finite_number(v4)
    has7 = (mask & 0x02) != 0 and is_finite_number(v7)
    has10 = (mask & 0x04) != 0 and is_finite_number(v10)

    # 3-point piecewise
    if has4 and has7 and has10:
        line_low = calc_line(v4, 4.0, v7, 7.0)
        line_high = calc_line(v7, 7.0, v10, 10.0)

        if line_low is None and line_high is None:
            return None, "UNCAL"

        ph_low = None
        ph_high = None
        if line_low is not None:
            mL, bL = line_low
            ph_low = mL * v + bL
        if line_high is not None:
            mH, bH = line_high
            ph_high = mH * v + bH

        d4 = v4 - v7
        d10 = v10 - v7
        dv = v - v7

        use_low = (abs(d4) > 1e-12) and ((dv * d4) >= 0)
        use_high = (abs(d10) > 1e-12) and ((dv * d10) >= 0)

        if use_low and (not use_high) and ph_low is not None:
            return ph_low, "LOW"
        if use_high and (not use_low) and ph_high is not None:
            return ph_high, "HIGH"

        if ph_low is not None and ph_high is not None:
            dl = abs(ph_low - 7.0)
            dh = abs(ph_high - 7.0)
            if dl <= dh:
                return ph_low, "LOW"
            return ph_high, "HIGH"

        if ph_low is not None:
            return ph_low, "LOW"
        if ph_high is not None:
            return ph_high, "HIGH"

        return None, "UNCAL"

    # 2-point fallback
    if has7 and has4:
        line = calc_line(v7, 7.0, v4, 4.0)
        if line is not None:
            m, b = line
            return m * v + b, "2PT_74"

    if has7 and has10:
        line = calc_line(v7, 7.0, v10, 10.0)
        if line is not None:
            m, b = line
            return m * v + b, "2PT_710"

    if has4 and has10:
        line = calc_line(v4, 4.0, v10, 10.0)
        if line is not None:
            m, b = line
            return m * v + b, "2PT_410"

    return None, "UNCAL"


# -----------------------------
# Configs
# -----------------------------
@dataclass
class DetectConfig:
    window_n: int = 30
    vcc_std_mV: float = 20.0
    ph_std: float = 0.06


@dataclass
class SetupConfig:
    stable_ph_std: float = 0.02


@dataclass
class CaptureSession:
    kind: str               # "MEAS" or "CAL"
    duration_s: float
    t0: float
    target_ph: Optional[float] = None
    ph_samples: List[float] = None
    v_samples: List[float] = None
    vcc_samples: List[float] = None

    def __post_init__(self):
        self.ph_samples = self.ph_samples or []
        self.v_samples = self.v_samples or []
        self.vcc_samples = self.vcc_samples or []


# -----------------------------
# Serial Reader Thread
# -----------------------------
class SerialReader(threading.Thread):
    def __init__(self, port, baud, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.q = out_queue
        self.stop_event = stop_event
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.0)
            self.q.put(("SYS", f"CONNECTED,{self.port},{self.baud}"))
        except Exception as e:
            self.q.put(("SYS", f"ERROR,OPEN,{e}"))
            return

        while not self.stop_event.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self.q.put(("LINE", line))
            except Exception as e:
                self.q.put(("SYS", f"ERROR,READ,{e}"))
                time.sleep(0.2)

        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.q.put(("SYS", "DISCONNECTED"))

    def write_line(self, s: str):
        if self.ser and self.ser.is_open:
            self.ser.write((s.strip() + "\n").encode())


# -----------------------------
# UI Components
# -----------------------------
class LedDot(QtWidgets.QWidget):
    def __init__(self, diameter=10, parent=None):
        super().__init__(parent)
        self._diameter = diameter
        self._color = QtGui.QColor("#666666")
        self.setFixedSize(diameter, diameter)

    def setColor(self, color_hex: str):
        self._color = QtGui.QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(self._color))
        r = QtCore.QRectF(0, 0, self._diameter, self._diameter)
        p.drawEllipse(r)


class BigNumber(QtWidgets.QFrame):
    def __init__(self, title: str, unit: str, big=False, parent=None):
        super().__init__(parent)
        self.setObjectName("BigNumber")
        self._title = QtWidgets.QLabel(title)
        self._title.setObjectName("BigNumberTitle")
        self._value = QtWidgets.QLabel("--")
        self._value.setObjectName("BigNumberValue")
        self._unit = QtWidgets.QLabel(unit)
        self._unit.setObjectName("BigNumberUnit")

        if big:
            self._value.setStyleSheet("font-size: 44px; font-weight: 950; color:#e6e6e6;")
        else:
            self._value.setStyleSheet("font-size: 21px; font-weight: 900; color:#e6e6e6;")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(4)
        lay.addWidget(self._title)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._value, 1)
        row.addWidget(self._unit, 0, QtCore.Qt.AlignBottom)
        lay.addLayout(row)

    def setValue(self, text: str):
        self._value.setText(text)


class CollapsibleSection(QtWidgets.QFrame):
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")

        self.btn = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self.btn.setObjectName("CollapseButton")
        self.btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn.setArrowType(QtCore.Qt.RightArrow)
        self.btn.clicked.connect(self._on_toggle)

        self.body = QtWidgets.QWidget()
        self.body.setVisible(False)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.btn)
        lay.addWidget(self.body)

        self.body_lay = QtWidgets.QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(10, 0, 10, 10)
        self.body_lay.setSpacing(8)

    def _on_toggle(self, checked: bool):
        self.body.setVisible(checked)
        self.btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.toggled.emit(checked)

    def setExpanded(self, expanded: bool):
        self.btn.setChecked(expanded)
        self._on_toggle(expanded)

    def addWidget(self, w: QtWidgets.QWidget):
        self.body_lay.addWidget(w)

    def addLayout(self, l: QtWidgets.QLayout):
        self.body_lay.addLayout(l)


def mk_duration_selector(default_custom_min=2):
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)

    group = QtWidgets.QButtonGroup()
    group.setExclusive(True)

    btns = {}

    for sec, txt in [(10, "10초"), (30, "30초"), (60, "1분")]:
        b = QtWidgets.QPushButton(txt)
        b.setCheckable(True)
        b.setMinimumHeight(34)
        group.addButton(b, sec)
        btns[sec] = b
        row.addWidget(b, 1)

    custom_btn = QtWidgets.QPushButton("선택(분)")
    custom_btn.setCheckable(True)
    custom_btn.setMinimumHeight(34)
    group.addButton(custom_btn, -2)
    btns["custom"] = custom_btn
    row.addWidget(custom_btn, 1)

    spin_min = QtWidgets.QSpinBox()
    spin_min.setRange(1, 999)
    spin_min.setValue(default_custom_min)
    spin_min.setSuffix(" 분")
    spin_min.setEnabled(False)
    spin_min.setMinimumHeight(34)
    spin_min.setMinimumWidth(90)
    row.addWidget(spin_min, 0)

    custom_btn.toggled.connect(spin_min.setEnabled)

    return row, group, btns, spin_min


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.det_cfg = DetectConfig()
        self.setup_cfg = SetupConfig()

        self.t = deque(maxlen=7200)
        self.ph = deque(maxlen=7200)
        self.vcc = deque(maxlen=7200)
        self.v_sensor = deque(maxlen=7200)

        self.ph_w = deque(maxlen=self.det_cfg.window_n)
        self.vcc_w = deque(maxlen=self.det_cfg.window_n)

        # Device-reported calibration
        self.cal_mask = None
        self.cal_v4 = None
        self.cal_v7 = None
        self.cal_v10 = None
        self.cal_a = None
        self.cal_b = None
        self.last_seg = "-"

        self.session: Optional[CaptureSession] = None
        self.pending_cal_target: Optional[float] = None

        self.reader = None
        self.stop_event = threading.Event()
        self.rx_queue = queue.Queue()
        self.connected = False

        # CSV export buffers
        self.meas_header = ["time", "duration_s", "ph_avg", "ph_std", "v_avg", "vcc_avg", "note"]
        self.cal_header = ["time", "target_ph", "duration_s", "ph_avg", "ph_std", "v_avg", "v_std", "vcc_avg", "note"]
        self.meas_rows = []
        self.cal_rows = []
        self.last_saved_paths = None

        self.setWindowTitle("pH Instrument")
        self._apply_style()
        self._build_ui()
        self._update_save_label()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

        self._refresh_ports()
        self._set_state_disconnected()

    def closeEvent(self, event):
        self._disconnect()
        event.accept()

    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        QLabel { color: #e6e6e6; font-size: 10px; }

        QFrame#Panel { background: #151a22; border: 1px solid #222a36; border-radius: 12px; }
        QFrame#Header { background: #111620; border: 1px solid #222a36; border-radius: 12px; }

        QLabel#HeaderTitle { font-size: 14px; font-weight: 950; letter-spacing: 0.4px; }
        QLabel#HeaderSub { color: #9aa6b2; font-size: 9px; font-weight: 800; }

        QPushButton {
            background: #1b2330; border: 1px solid #2a3444; border-radius: 10px;
            padding: 7px 8px; color: #e6e6e6; font-weight: 900; font-size: 10px;
        }
        QPushButton:hover { background: #222c3b; }
        QPushButton:pressed { background: #151c27; }
        QPushButton:checked { background: #1b2330; border: 1px solid #42506a; }
        QPushButton:disabled { color: #6f7a86; border-color: #1f2733; background: #141a22; }

        QComboBox {
            background: #121824; border: 1px solid #2a3444; border-radius: 10px;
            padding: 6px 8px; color: #e6e6e6; font-weight: 900; font-size: 10px;
        }
        QComboBox QAbstractItemView {
            background: #121824; border: 1px solid #2a3444; selection-background-color: #222c3b;
            color: #e6e6e6; font-size: 10px;
        }

        QPlainTextEdit {
            background: #0d111a; border: 1px solid #222a36; border-radius: 10px;
            color: #cfd8e3; padding: 8px; font-family: Consolas, Menlo, monospace; font-size: 9px;
        }

        QFrame#BigNumber { background: #0d111a; border: 1px solid #222a36; border-radius: 12px; }
        QLabel#BigNumberTitle { color: #9aa6b2; font-size: 9px; font-weight: 900; }
        QLabel#BigNumberUnit  { color: #9aa6b2; font-size: 10px; font-weight: 900; }

        QProgressBar {
            background: #0d111a; border: 1px solid #222a36; border-radius: 10px;
            color: #e6e6e6; text-align: center; font-weight: 900; height: 18px; font-size: 9px;
        }
        QProgressBar::chunk { background: #1b2330; border-radius: 10px; }

        QSpinBox, QDoubleSpinBox, QLineEdit {
            background: #121824; border: 1px solid #2a3444; border-radius: 10px;
            padding: 6px 8px; color: #e6e6e6; font-weight: 900; font-size: 10px;
        }

        QFrame#CollapsibleSection { background: transparent; }
        QToolButton#CollapseButton {
            background: #121824; border: 1px solid #222a36; border-radius: 10px;
            padding: 8px 10px; color: #cfd8e3; font-weight: 950;
            text-align: left;
        }
        QToolButton#CollapseButton:hover { background: #1b2330; }
        """)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QtWidgets.QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(56)
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.setSpacing(10)

        self.led_conn = LedDot(9)
        self.led_conn.setColor("#666666")

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(1)
        self.lbl_title = QtWidgets.QLabel("pH Instrument")
        self.lbl_title.setObjectName("HeaderTitle")
        self.lbl_sub = QtWidgets.QLabel("Meas: (not saved) | Cal: (not saved)")
        self.lbl_sub.setObjectName("HeaderSub")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_sub)

        self.lbl_statusline = QtWidgets.QLabel("STATUS: DISCONNECTED")
        self.lbl_statusline.setObjectName("HeaderSub")
        self.lbl_clock = QtWidgets.QLabel("--:--:--")
        self.lbl_clock.setObjectName("HeaderSub")
        self.btn_save = QtWidgets.QPushButton("SAVE CSV")
        self.btn_save.setFixedWidth(110)

        hl.addWidget(self.led_conn, 0, QtCore.Qt.AlignVCenter)
        hl.addLayout(title_box, 1)
        hl.addWidget(self.lbl_statusline, 0)
        hl.addWidget(self.btn_save, 0)
        hl.addWidget(self.lbl_clock, 0)
        root.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        # LEFT
        left = QtWidgets.QFrame()
        left.setObjectName("Panel")
        left.setMinimumWidth(440)
        left.setMaximumWidth(560)
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(10)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        self.cb_port = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("↻")
        self.btn_refresh.setFixedWidth(44)

        self.cb_baud = QtWidgets.QComboBox()
        self.cb_baud.addItems(["115200", "57600", "38400", "9600"])
        self.cb_baud.setCurrentText("115200")
        self.cb_baud.setFixedWidth(96)

        self.btn_connect = QtWidgets.QPushButton("CONNECT")
        self.btn_disconnect = QtWidgets.QPushButton("DISCONNECT")

        top_row.addWidget(self.cb_port, 1)
        top_row.addWidget(self.btn_refresh, 0)
        top_row.addWidget(self.cb_baud, 0)
        top_row.addWidget(self.btn_connect, 0)
        top_row.addWidget(self.btn_disconnect, 0)
        ll.addLayout(top_row)

        # LIVE
        live = QtWidgets.QFrame()
        live.setObjectName("Panel")
        lv = QtWidgets.QVBoxLayout(live)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        title = self._mk_section_title("LIVE")
        lv.addWidget(title)

        self.bn_ph = BigNumber("pH", "pH", big=True)
        lv.addWidget(self.bn_ph)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.bn_v = BigNumber("Sensor Voltage", "V", big=False)
        self.bn_vcc = BigNumber("VCC", "mV", big=False)
        grid.addWidget(self.bn_v, 0, 0)
        grid.addWidget(self.bn_vcc, 0, 1)

        lv.addLayout(grid)

        ind_row = QtWidgets.QHBoxLayout()
        ind_row.setSpacing(8)
        self.led_stable = LedDot(10)
        self.led_ready = LedDot(10)
        self.lbl_stable = QtWidgets.QLabel("STABLE: -")
        self.lbl_ready = QtWidgets.QLabel("READY: -")
        for lab in (self.lbl_stable, self.lbl_ready):
            lab.setStyleSheet("color:#9aa6b2; font-size: 10px; font-weight: 950;")
        ind_row.addWidget(self.led_stable, 0, QtCore.Qt.AlignVCenter)
        ind_row.addWidget(self.lbl_stable, 0, QtCore.Qt.AlignVCenter)
        ind_row.addSpacing(10)
        ind_row.addWidget(self.led_ready, 0, QtCore.Qt.AlignVCenter)
        ind_row.addWidget(self.lbl_ready, 0, QtCore.Qt.AlignVCenter)
        ind_row.addStretch(1)

        self.btn_read = QtWidgets.QPushButton("READ")
        self.btn_read.setMinimumHeight(36)

        lv.addLayout(ind_row)
        lv.addWidget(self.btn_read)

        self.lbl_cal_dev = QtWidgets.QLabel("Device CAL: -")
        self.lbl_cal_dev.setWordWrap(True)
        self.lbl_cal_dev.setStyleSheet("color:#cfd8e3; font-size: 9px; font-weight: 900;")
        lv.addWidget(self.lbl_cal_dev)

        ll.addWidget(live)

        # Collapsible sections
        self.sec_cal = CollapsibleSection("CALIBRATION (pH 4 / 7 / 10)")
        self.sec_meas = CollapsibleSection("MEASURE (average pH)")
        self.sec_setup = CollapsibleSection("SETUP")
        self.sec_log = CollapsibleSection("LOG")

        self.sec_cal.toggled.connect(lambda on: self._exclusive_expand(self.sec_cal, on))
        self.sec_meas.toggled.connect(lambda on: self._exclusive_expand(self.sec_meas, on))
        self.sec_setup.toggled.connect(lambda on: self._exclusive_expand(self.sec_setup, on))
        self.sec_log.toggled.connect(lambda on: self._exclusive_expand(self.sec_log, on))

        # CAL body
        self.lbl_cal_hint = QtWidgets.QLabel("pH 버튼 → 안내 확인 → 시간 선택 → START")
        self.lbl_cal_hint.setWordWrap(True)
        self.lbl_cal_hint.setStyleSheet("color:#9aa6b2; font-size: 9px; font-weight: 900;")
        self.sec_cal.addWidget(self.lbl_cal_hint)

        cal_btn_row = QtWidgets.QHBoxLayout()
        cal_btn_row.setSpacing(8)
        self.btn_cal4 = QtWidgets.QPushButton("CAL 4.00")
        self.btn_cal7 = QtWidgets.QPushButton("CAL 7.00")
        self.btn_cal10 = QtWidgets.QPushButton("CAL 10.00")
        cal_btn_row.addWidget(self.btn_cal4, 1)
        cal_btn_row.addWidget(self.btn_cal7, 1)
        cal_btn_row.addWidget(self.btn_cal10, 1)
        self.sec_cal.addLayout(cal_btn_row)

        cal_dur_row, self.grp_cal_dur, self.btns_cal_dur, self.sp_cal_custom_min = mk_duration_selector()
        self.sec_cal.addLayout(cal_dur_row)
        self.btns_cal_dur[10].setChecked(True)

        cal_ctrl = QtWidgets.QHBoxLayout()
        cal_ctrl.setSpacing(8)
        self.btn_cal_start = QtWidgets.QPushButton("START (CAL CAPTURE)")
        self.btn_cal_cancel = QtWidgets.QPushButton("CANCEL")
        cal_ctrl.addWidget(self.btn_cal_start, 1)
        cal_ctrl.addWidget(self.btn_cal_cancel, 1)
        self.sec_cal.addLayout(cal_ctrl)

        cal_ctrl2 = QtWidgets.QHBoxLayout()
        cal_ctrl2.setSpacing(8)
        self.btn_cal_reset = QtWidgets.QPushButton("RESET CAL")
        cal_ctrl2.addWidget(self.btn_cal_reset, 1)
        self.sec_cal.addLayout(cal_ctrl2)

        self.pb_cal = QtWidgets.QProgressBar()
        self.pb_cal.setRange(0, 1000)
        self.pb_cal.setValue(0)
        self.sec_cal.addWidget(self.pb_cal)

        self.lbl_cal_record = QtWidgets.QLabel("Last CAL capture: -")
        self.lbl_cal_record.setWordWrap(True)
        self.lbl_cal_record.setStyleSheet("color:#cfd8e3; font-size: 9px; font-weight: 900;")
        self.sec_cal.addWidget(self.lbl_cal_record)

        cal_note = QtWidgets.QHBoxLayout()
        cal_note.setSpacing(6)
        cal_note.addWidget(QtWidgets.QLabel("Note:"), 0)
        self.ed_cal_note = QtWidgets.QLineEdit()
        self.ed_cal_note.setPlaceholderText("calibration note (옵션)")
        cal_note.addWidget(self.ed_cal_note, 1)
        self.sec_cal.addLayout(cal_note)

        # MEASURE body
        meas_dur_row, self.grp_meas_dur, self.btns_meas_dur, self.sp_meas_custom_min = mk_duration_selector()
        self.sec_meas.addLayout(meas_dur_row)
        self.btns_meas_dur[10].setChecked(True)

        meas_ctrl = QtWidgets.QHBoxLayout()
        meas_ctrl.setSpacing(8)
        self.btn_meas_start = QtWidgets.QPushButton("START (MEASURE)")
        self.btn_meas_stop = QtWidgets.QPushButton("STOP")
        meas_ctrl.addWidget(self.btn_meas_start, 1)
        meas_ctrl.addWidget(self.btn_meas_stop, 1)
        self.sec_meas.addLayout(meas_ctrl)

        self.pb_meas = QtWidgets.QProgressBar()
        self.pb_meas.setRange(0, 1000)
        self.pb_meas.setValue(0)
        self.sec_meas.addWidget(self.pb_meas)

        self.lbl_meas_result = QtWidgets.QLabel("Result: -")
        self.lbl_meas_result.setWordWrap(True)
        self.lbl_meas_result.setStyleSheet("color:#cfd8e3; font-size: 10px; font-weight: 950;")
        self.sec_meas.addWidget(self.lbl_meas_result)

        meas_note = QtWidgets.QHBoxLayout()
        meas_note.setSpacing(6)
        meas_note.addWidget(QtWidgets.QLabel("Note:"), 0)
        self.ed_meas_note = QtWidgets.QLineEdit()
        self.ed_meas_note.setPlaceholderText("sample id / 조건 메모")
        meas_note.addWidget(self.ed_meas_note, 1)
        self.sec_meas.addLayout(meas_note)

        # SETUP body
        row_setup1 = QtWidgets.QHBoxLayout()
        row_setup1.setSpacing(8)
        row_setup1.addWidget(QtWidgets.QLabel("STABLE pH std thr:"), 0)
        self.sp_stable_thr = QtWidgets.QDoubleSpinBox()
        self.sp_stable_thr.setRange(0.001, 1.0)
        self.sp_stable_thr.setDecimals(3)
        self.sp_stable_thr.setValue(self.setup_cfg.stable_ph_std)
        row_setup1.addWidget(self.sp_stable_thr, 1)
        self.sec_setup.addLayout(row_setup1)

        row_setup2 = QtWidgets.QHBoxLayout()
        row_setup2.setSpacing(8)
        row_setup2.addWidget(QtWidgets.QLabel("Diag window (samples):"), 0)
        self.sp_win = QtWidgets.QSpinBox()
        self.sp_win.setRange(5, 300)
        self.sp_win.setValue(self.det_cfg.window_n)
        row_setup2.addWidget(self.sp_win, 1)
        self.sec_setup.addLayout(row_setup2)

        # LOG body
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(4000)
        self.txt_log.setMinimumHeight(160)
        self.sec_log.addWidget(self.txt_log)

        ll.addWidget(self.sec_cal)
        ll.addWidget(self.sec_meas)
        ll.addWidget(self.sec_setup)
        ll.addWidget(self.sec_log)

        self.sec_cal.setExpanded(False)
        self.sec_meas.setExpanded(False)
        self.sec_setup.setExpanded(False)
        self.sec_log.setExpanded(False)

        splitter.addWidget(left)

        # RIGHT graphs
        right = QtWidgets.QFrame()
        right.setObjectName("Panel")
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)

        pg.setConfigOptions(antialias=True)
        self.plot_ph = pg.PlotWidget(title="pH (Live)")
        self.plot_v = pg.PlotWidget(title="Sensor Voltage (V)")

        for pw in (self.plot_ph, self.plot_v):
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setBackground("#0d111a")

        self.curve_ph = self.plot_ph.plot([], [], pen=pg.mkPen(width=2))
        self.curve_v = self.plot_v.plot([], [], pen=pg.mkPen(width=2))

        rl.addWidget(self.plot_ph, 6)
        rl.addWidget(self.plot_v, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        # Signals
        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_read.clicked.connect(lambda: self._send("READ"))
        self.btn_save.clicked.connect(self._save_csv)

        self.btn_cal4.clicked.connect(lambda: self._select_cal_target(4.0))
        self.btn_cal7.clicked.connect(lambda: self._select_cal_target(7.0))
        self.btn_cal10.clicked.connect(lambda: self._select_cal_target(10.0))
        self.btn_cal_start.clicked.connect(self._start_cal_capture)
        self.btn_cal_cancel.clicked.connect(self._cancel_session)
        self.btn_cal_reset.clicked.connect(self._reset_calibration)

        self.btn_meas_start.clicked.connect(self._start_measure)
        self.btn_meas_stop.clicked.connect(self._cancel_session)

    def _mk_section_title(self, text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet("color:#cfd8e3; font-size: 10px; font-weight: 950;")
        return lab

    def _exclusive_expand(self, who: CollapsibleSection, on: bool):
        if not on:
            return
        for sec in (self.sec_cal, self.sec_meas, self.sec_setup, self.sec_log):
            if sec is not who:
                sec.setExpanded(False)

    def _get_selected_duration_sec(self, group: QtWidgets.QButtonGroup, spin_min: QtWidgets.QSpinBox) -> float:
        sel = group.checkedId()
        if sel == -2:   # custom minutes
            return float(spin_min.value()) * 60.0
        if sel > 0:
            return float(sel)
        return 10.0

    # ---------------- core helpers ----------------
    def _log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] {s}")

    def _update_save_label(self):
        if self.last_saved_paths:
            meas = self.last_saved_paths.get("meas") or "(not saved)"
            cal = self.last_saved_paths.get("cal") or "(not saved)"
            self.lbl_sub.setText(f"Meas: {meas} | Cal: {cal}")
        else:
            self.lbl_sub.setText("Meas: (not saved) | Cal: (not saved)")

    def _save_csv(self):
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "저장 폴더 선택", "")
        if not out_dir:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        meas_path = os.path.join(out_dir, f"ph_measure_{ts}.csv")
        cal_path = os.path.join(out_dir, f"ph_calibration_{ts}.csv")

        try:
            with open(meas_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self.meas_header)
                w.writerows(self.meas_rows)

            with open(cal_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self.cal_header)
                w.writerows(self.cal_rows)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAVE CSV", f"저장 실패: {e}")
            self._log(f"SAVE CSV failed: {e}")
            return

        self.last_saved_paths = {"log": None, "meas": meas_path, "cal": cal_path}
        self._update_save_label()
        self._log(f"SAVE CSV OK: {out_dir}")

    def _refresh_ports(self):
        ports = list_ports()
        cur = self.cb_port.currentText()
        self.cb_port.clear()
        self.cb_port.addItems(ports)
        if cur in ports:
            self.cb_port.setCurrentText(cur)
        self._log(f"Ports: {ports if ports else 'NONE'}")

    def _connect(self):
        port = self.cb_port.currentText().strip()
        if not port:
            self._log("No port selected.")
            return
        baud = int(self.cb_baud.currentText())

        self.stop_event.clear()
        self.reader = SerialReader(port, baud, self.rx_queue, self.stop_event)
        self.reader.start()

        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self.lbl_statusline.setText("STATUS: CONNECTING...")

    def _disconnect(self):
        if self.reader:
            self.stop_event.set()
            self.reader = None
        self._set_state_disconnected()

    def _send(self, cmd: str):
        if self.reader and self.reader.ser and self.reader.ser.is_open:
            self.reader.write_line(cmd)
            self._log(f"TX: {cmd}")
        else:
            self._log("TX failed: not connected.")

    def _set_state_disconnected(self):
        self.connected = False
        self.led_conn.setColor("#666666")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)

        for b in (self.btn_read, self.btn_cal4, self.btn_cal7, self.btn_cal10,
                  self.btn_cal_start, self.btn_cal_cancel, self.btn_cal_reset,
                  self.btn_meas_start, self.btn_meas_stop):
            b.setEnabled(False)

        self.lbl_statusline.setText("STATUS: DISCONNECTED")
        self._update_stable_ready(False, False, "OK")
        self._cancel_session(silent=True)

    def _set_state_connected(self, port: str, baud: str):
        self.connected = True
        self.led_conn.setColor("#1f8f4a")
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)

        for b in (self.btn_read, self.btn_cal4, self.btn_cal7, self.btn_cal10,
                  self.btn_cal_start, self.btn_cal_cancel, self.btn_cal_reset,
                  self.btn_meas_start, self.btn_meas_stop):
            b.setEnabled(True)

        self.lbl_statusline.setText(f"STATUS: CONNECTED ({port} @ {baud})")
        QtCore.QTimer.singleShot(350, lambda: self._send("CAL,STATUS"))

    # ---------------- Reset Calibration ----------------
    def _reset_calibration(self):
        if not self.connected:
            self._log("CAL RESET: not connected.")
            return

        r = QtWidgets.QMessageBox.question(
            self,
            "Calibration Reset",
            "저장된 보정값(EEPROM)을 초기화합니다.\n"
            "전원을 껐다 켜도 복구되지 않습니다.\n\n"
            "정말 초기화할까요?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if r != QtWidgets.QMessageBox.Yes:
            return

        self._send("CAL,CLEAR")

        self.pending_cal_target = None
        self.cal_mask = 0
        self.cal_v4 = None
        self.cal_v7 = None
        self.cal_v10 = None
        self.cal_a = None
        self.cal_b = None
        self.last_seg = "UNCAL"
        self.lbl_cal_hint.setText("pH 버튼 → 안내 확인 → 시간 선택 → START")
        self.lbl_cal_record.setText("Last CAL capture: -")
        self.pb_cal.setValue(0)
        self._update_device_cal_label()

        QtCore.QTimer.singleShot(200, lambda: self._send("CAL,STATUS"))
        self._log("CAL RESET requested.")

    # ---------------- READY / STABLE ----------------
    def _update_stable_ready(self, connected: bool, stable: bool, noise_level: str):
        if not connected:
            self.led_stable.setColor("#666666")
            self.led_ready.setColor("#666666")
            self.lbl_stable.setText("STABLE: -")
            self.lbl_ready.setText("READY: -")
            return

        if stable:
            self.led_stable.setColor("#1f8f4a")
            self.lbl_stable.setText("STABLE: YES")
        else:
            self.led_stable.setColor("#b36b00")
            self.lbl_stable.setText("STABLE: NO")

        if noise_level in ("POWER", "SIGNAL"):
            self.led_ready.setColor("#c00000" if noise_level == "POWER" else "#b36b00")
            self.lbl_ready.setText("READY: NO")
        else:
            self.led_ready.setColor("#1f8f4a" if stable else "#b36b00")
            self.lbl_ready.setText("READY: YES" if stable else "READY: NO")

    # ---------------- stats helpers ----------------
    @staticmethod
    def _std(vals: List[float]) -> float:
        clean = [x for x in vals if is_finite_number(x)]
        n = len(clean)
        if n < 2:
            return 0.0
        mean = sum(clean) / n
        var = sum((x - mean) ** 2 for x in clean) / (n - 1)
        return var ** 0.5

    @staticmethod
    def _avg(vals: List[float]) -> Optional[float]:
        clean = [x for x in vals if is_finite_number(x)]
        if not clean:
            return None
        return sum(clean) / len(clean)

    def _diagnose(self):
        vcc_std = self._std(list(self.vcc_w)) if len(self.vcc_w) >= 5 else 0.0
        ph_std = self._std(list(self.ph_w)) if len(self.ph_w) >= 5 else 0.0
        if vcc_std > self.det_cfg.vcc_std_mV:
            return "POWER", vcc_std, ph_std
        if ph_std > self.det_cfg.ph_std:
            return "SIGNAL", vcc_std, ph_std
        return "OK", vcc_std, ph_std

    def _compute_ph_from_voltage(self, v_sensor: Optional[float]) -> Tuple[Optional[float], str]:
        ph_val, seg = calc_ph_from_voltage(
            v_sensor,
            self.cal_mask,
            self.cal_v4,
            self.cal_v7,
            self.cal_v10,
        )
        if not is_finite_number(ph_val):
            return None, seg
        return ph_val, seg

    # ---------------- Calibration workflow ----------------
    def _select_cal_target(self, ph_target: float):
        self.pending_cal_target = ph_target
        QtWidgets.QMessageBox.information(
            self,
            "Calibration",
            f"pH{int(ph_target)} 값을 넣어주세요.\n"
            "안정화 후 시간(10초/30초/1분/선택 분) 선택 → START"
        )
        self._log(f"CAL: target selected pH{ph_target}")
        self.lbl_cal_hint.setText(f"pH{int(ph_target)} 값을 넣어주세요. 시간(10초/30초/1분/선택 분) 선택 후 START.")
        self.sec_cal.setExpanded(True)

    def _start_cal_capture(self):
        if not self.connected:
            self._log("CAL: not connected.")
            return
        if self.pending_cal_target is None:
            QtWidgets.QMessageBox.warning(self, "Calibration", "먼저 CAL 4/7/10 중 하나를 선택하세요.")
            return
        if self.session is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "이미 캡처가 진행 중입니다. CANCEL 후 진행하세요.")
            return

        dur = self._get_selected_duration_sec(self.grp_cal_dur, self.sp_cal_custom_min)
        self.session = CaptureSession(kind="CAL", duration_s=dur, t0=time.time(), target_ph=self.pending_cal_target)
        self.pb_cal.setValue(0)
        self._log(f"CAL: START capture target=pH{self.pending_cal_target} duration={dur}s")

    # ---------------- Measure workflow ----------------
    def _start_measure(self):
        if not self.connected:
            self._log("MEASURE: not connected.")
            return
        if self.session is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "이미 캡처가 진행 중입니다. STOP/CANCEL 후 진행하세요.")
            return

        dur = self._get_selected_duration_sec(self.grp_meas_dur, self.sp_meas_custom_min)
        self.session = CaptureSession(kind="MEAS", duration_s=dur, t0=time.time())
        self.pb_meas.setValue(0)
        self.lbl_meas_result.setText("Result: measuring...")
        self._log(f"MEASURE: START duration={dur}s")
        self.sec_meas.setExpanded(True)

    def _cancel_session(self, silent: bool = False):
        self.session = None
        self.pb_meas.setValue(0)
        self.pb_cal.setValue(0)
        if not silent:
            self._log("SESSION: canceled")

    # ---------------- UI label helpers ----------------
    def _update_device_cal_label(self):
        mode = calmask_to_mode(self.cal_mask) if self.cal_mask is not None else "UNKNOWN"
        v4 = f"{self.cal_v4:.4f}V" if is_finite_number(self.cal_v4) else "-"
        v7 = f"{self.cal_v7:.4f}V" if is_finite_number(self.cal_v7) else "-"
        v10 = f"{self.cal_v10:.4f}V" if is_finite_number(self.cal_v10) else "-"
        a = f"{self.cal_a:.6f}" if is_finite_number(self.cal_a) else "-"
        b = f"{self.cal_b:.6f}" if is_finite_number(self.cal_b) else "-"
        seg = self.last_seg if self.last_seg else "-"
        self.lbl_cal_dev.setText(
            f"Device CAL: {mode} | mask={self.cal_mask if self.cal_mask is not None else '-'} | "
            f"V4={v4} V7={v7} V10={v10} | a={a} b={b} | SEG={seg}"
        )

    # ---------------- Tick ----------------
    def _tick(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S"))

        self.setup_cfg.stable_ph_std = float(self.sp_stable_thr.value())
        new_win = int(self.sp_win.value())
        if new_win != self.det_cfg.window_n:
            self.det_cfg.window_n = new_win
            self.ph_w = deque(self.ph_w, maxlen=new_win)
            self.vcc_w = deque(self.vcc_w, maxlen=new_win)

        updated_graph = False

        # progress
        if self.session is not None:
            elapsed = time.time() - self.session.t0
            p = min(1.0, max(0.0, elapsed / max(0.001, self.session.duration_s)))
            if self.session.kind == "MEAS":
                self.pb_meas.setValue(int(p * 1000))
            else:
                self.pb_cal.setValue(int(p * 1000))

        while True:
            try:
                typ, payload = self.rx_queue.get_nowait()
            except queue.Empty:
                break

            if typ == "SYS":
                msg = payload
                if msg.startswith("CONNECTED"):
                    _, port, baud = msg.split(",", 2)
                    self._set_state_connected(port, baud)
                elif msg.startswith("DISCONNECTED"):
                    self._set_state_disconnected()
                elif msg.startswith("ERROR"):
                    self._log(f"SYSTEM: {msg}")
                    self.lbl_statusline.setText("STATUS: ERROR")
                else:
                    self._log(f"SYSTEM: {msg}")
                continue

            if typ == "LINE":
                raw = payload
                msg = parse_line(raw)
                if not msg:
                    continue

                if msg["type"] == "CAL_STATUS":
                    self.cal_mask = to_int(msg.get("MASK", None), self.cal_mask)
                    self.cal_v4 = to_float(msg.get("V4", None), self.cal_v4)
                    self.cal_v7 = to_float(msg.get("V7", None), self.cal_v7)
                    self.cal_v10 = to_float(msg.get("V10", None), self.cal_v10)
                    self.cal_a = to_float(msg.get("A", None), self.cal_a)
                    self.cal_b = to_float(msg.get("B", None), self.cal_b)
                    self._update_device_cal_label()
                    self._log(f"RX: {msg['raw']}")
                    continue

                if msg["type"] in ("OK", "ERR", "BOOT", "HELP", "MODE", "SAMPLE_HZ", "TX_HZ"):
                    self._log(f"RX: {msg['raw']}")
                    continue

                if msg["type"] == "RAW":
                    continue

                if msg["type"] in ("T", "S"):
                    ms = msg.get("ms", None) if msg["type"] == "T" else None
                    vcc_mV = to_int(msg.get("VCC_MV", None), None)
                    v_sensor = to_float(msg.get("V", None), None)
                    calmask = to_int(msg.get("CALMASK", None), None)

                    if calmask is not None:
                        self.cal_mask = calmask

                    ph_val, seg = self._compute_ph_from_voltage(v_sensor)
                    self.last_seg = seg

                    t_sec = (ms / 1000.0) if ms is not None else time.time()

                    self.t.append(t_sec)
                    self.ph.append(math.nan if ph_val is None else ph_val)
                    self.v_sensor.append(math.nan if v_sensor is None else v_sensor)
                    self.vcc.append(math.nan if vcc_mV is None else float(vcc_mV))

                    if ph_val is not None:
                        self.ph_w.append(ph_val)
                    if vcc_mV is not None:
                        self.vcc_w.append(float(vcc_mV))

                    level, vcc_std, ph_std = self._diagnose()

                    stable = False
                    if (ph_val is not None) and (len(self.ph_w) >= 5):
                        stable = (ph_std <= self.setup_cfg.stable_ph_std)
                    self._update_stable_ready(self.connected, stable, level)

                    if ph_val is None:
                        self.bn_ph.setValue("UNCAL" if v_sensor is not None else "--")
                    else:
                        self.bn_ph.setValue(f"{ph_val:0.3f}")

                    self.bn_v.setValue(f"{v_sensor:0.4f}" if is_finite_number(v_sensor) else "--")
                    self.bn_vcc.setValue(f"{vcc_mV:d}" if vcc_mV is not None else "--")

                    calmode = calmask_to_mode(self.cal_mask) if self.cal_mask is not None else "UNKNOWN"
                    self.lbl_statusline.setText(
                        f"STATUS: {'CONNECTED' if self.connected else 'DISCONNECTED'} | CAL={calmode} | SEG={self.last_seg}"
                    )
                    self._update_device_cal_label()

                    # session sampling
                    if self.session is not None:
                        elapsed = time.time() - self.session.t0

                        if ph_val is not None:
                            self.session.ph_samples.append(ph_val)
                        if v_sensor is not None:
                            self.session.v_samples.append(v_sensor)
                        if vcc_mV is not None:
                            self.session.vcc_samples.append(float(vcc_mV))

                        if elapsed >= self.session.duration_s:
                            self._finalize_session()

                    updated_graph = True

        # Graph update
        if updated_graph and len(self.t) >= 2:
            t0 = self.t[0]
            x = [tt - t0 for tt in self.t]

            self.curve_ph.setData(x, list(self.ph))
            self.curve_v.setData(x, list(self.v_sensor))

            self.plot_ph.enableAutoRange(x=True, y=True)
            self.plot_v.enableAutoRange(x=True, y=True)

    def _finalize_session(self):
        if self.session is None:
            return

        s = self.session
        ph_avg = self._avg(s.ph_samples)
        ph_std = self._std(s.ph_samples) if s.ph_samples else 0.0
        v_avg = self._avg(s.v_samples)
        v_std = self._std(s.v_samples) if s.v_samples else 0.0
        vcc_avg = self._avg(s.vcc_samples)
        dur = s.duration_s

        if s.kind == "MEAS":
            note = self.ed_meas_note.text().strip()
            if ph_avg is None:
                self.lbl_meas_result.setText("Result: no valid pH samples")
            else:
                self.lbl_meas_result.setText(f"Result: pH avg={ph_avg:.3f}  std={ph_std:.3f}  (t={int(dur)}s)")
            self.meas_rows.append([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                int(dur),
                f"{ph_avg:.3f}" if ph_avg is not None else "",
                f"{ph_std:.3f}",
                f"{v_avg:.4f}" if v_avg is not None else "",
                f"{vcc_avg:.1f}" if vcc_avg is not None else "",
                note
            ])
            self._log(f"MEASURE DONE: avg={ph_avg} std={ph_std} n={len(s.ph_samples)}")

        elif s.kind == "CAL":
            target = s.target_ph
            note = self.ed_cal_note.text().strip()

            self.lbl_cal_record.setText(
                f"Last CAL capture: pH{int(target) if target else '-'} | "
                f"Vavg={v_avg:.4f} Vstd={v_std:.4f} | t={int(dur)}s"
            )

            self.cal_rows.append([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{target:.1f}" if target is not None else "",
                int(dur),
                f"{ph_avg:.3f}" if ph_avg is not None else "",
                f"{ph_std:.3f}",
                f"{v_avg:.4f}" if v_avg is not None else "",
                f"{v_std:.4f}",
                f"{vcc_avg:.1f}" if vcc_avg is not None else "",
                note
            ])
            self._log(f"CAL DONE (PC): target={target} Vavg={v_avg} n={len(s.v_samples)}")

            if (target is not None) and (v_avg is not None):
                self._send(f"CAL,SET,{int(target)},{v_avg:.4f}")
                QtCore.QTimer.singleShot(150, lambda: self._send("CAL,STATUS"))

        self.session = None
        self.pb_meas.setValue(0)
        self.pb_cal.setValue(0)


def main():
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont("Segoe UI")
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    font.setPointSize(9)
    app.setFont(font)

    win = MainWindow()
    win.resize(1550, 950)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
import sys
import time
import re
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import serial
import serial.tools.list_ports

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters as pg_exporters
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
LINE_RE = re.compile(r'^S,CH,(\d+),(.+)$')


def list_ports() -> List[str]:
    return [p.device for p in serial.tools.list_ports.comports()]


def now_str() -> str:
    return time.strftime("%H:%M:%S")


def parse_kv_payload(payload: str) -> Dict[str, str]:
    parts = [p.strip() for p in payload.split(",")]
    out = {}
    for i in range(0, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1]
    return out


def safe_int(d: Dict[str, str], k: str, default: int = 0) -> int:
    try:
        return int(d.get(k, default))
    except Exception:
        return default


def channels_to_mask(ch_list: List[int]) -> int:
    m = 0
    for ch in ch_list:
        if 1 <= ch <= 4:
            m |= (1 << (ch - 1))
    return m


def moving_average(y: List[float], win: int) -> List[float]:
    """Simple moving average, keeps length same (edge: min-period)."""
    win = int(win)
    if win <= 1 or len(y) == 0:
        return list(y)
    win = min(win, len(y))
    out = []
    s = 0.0
    q = deque()
    for v in y:
        q.append(float(v))
        s += float(v)
        if len(q) > win:
            s -= q.popleft()
        out.append(s / len(q))
    return out


# -----------------------------
# UI: Instrument-like Panel (Collapsible)
# -----------------------------
class InstrumentPanel(QtWidgets.QFrame):
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str, collapsible: bool = True, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self.setProperty("panel", "1")
        self._collapsible = bool(collapsible)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        if self._collapsible:
            self.btn_title = QtWidgets.QToolButton()
            self.btn_title.setProperty("panelTitle", "1")
            self.btn_title.setText(title)
            self.btn_title.setCheckable(True)
            self.btn_title.setChecked(not collapsed)
            self.btn_title.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self.btn_title.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
            self.btn_title.clicked.connect(self._on_toggle_clicked)
            header.addWidget(self.btn_title, 1)
        else:
            self.lbl_title = QtWidgets.QLabel(title)
            self.lbl_title.setProperty("panelTitleLabel", "1")
            header.addWidget(self.lbl_title, 1)

        self.header_right = QtWidgets.QWidget()
        self.header_right_l = QtWidgets.QHBoxLayout(self.header_right)
        self.header_right_l.setContentsMargins(0, 0, 0, 0)
        self.header_right_l.setSpacing(8)
        header.addWidget(self.header_right, 0)

        root.addLayout(header)

        self.content = QtWidgets.QWidget()
        self.content_l = QtWidgets.QVBoxLayout(self.content)
        self.content_l.setContentsMargins(0, 0, 0, 0)
        self.content_l.setSpacing(8)
        root.addWidget(self.content, 1)

        self.set_collapsed(collapsed if self._collapsible else False)

    def set_collapsed(self, collapsed: bool):
        if not self._collapsible:
            self.content.setVisible(True)
            return
        collapsed = bool(collapsed)
        self.content.setVisible(not collapsed)
        self.btn_title.blockSignals(True)
        self.btn_title.setChecked(not collapsed)
        self.btn_title.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
        self.btn_title.blockSignals(False)
        self.toggled.emit(not collapsed)

    def is_collapsed(self) -> bool:
        if not self._collapsible:
            return False
        return not self.content.isVisible()

    def _on_toggle_clicked(self):
        expanded = self.btn_title.isChecked()
        self.btn_title.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)
        self.toggled.emit(expanded)


# -----------------------------
# Baseline + Ct (instrument-like)  (kept for Excel / plot, UI shows averages only)
# -----------------------------
@dataclass
class BaselineFit:
    start_cycle: int
    end_cycle: int
    a: float
    b: float
    baseline_line: List[float]
    delta: List[float]


def _linfit(x: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(x)
    if n <= 1:
        return 0.0, (y[0] if y else 0.0)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    den = (n * sxx - sx * sx)
    if abs(den) < 1e-12:
        return 0.0, sy / n
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    return a, b


def baseline_fit_linear(R: List[float], start_cycle: int, end_cycle: int) -> Optional[BaselineFit]:
    n = len(R)
    if n < 3:
        return None
    start_cycle = max(1, int(start_cycle))
    end_cycle = min(n, int(end_cycle))
    if end_cycle <= start_cycle:
        return None

    cycles = list(range(1, n + 1))
    xw = [float(c) for c in range(start_cycle, end_cycle + 1)]
    yw = [R[c - 1] for c in range(start_cycle, end_cycle + 1)]
    a, b = _linfit(xw, yw)

    baseline_line = [(a * float(c) + b) for c in cycles]
    delta = [R[i] - baseline_line[i] for i in range(n)]
    return BaselineFit(start_cycle, end_cycle, a, b, baseline_line, delta)


def auto_threshold_from_baseline(delta: List[float], base_start: int, base_end: int, k_sigma: float) -> float:
    base_start = max(1, int(base_start))
    base_end = min(len(delta), int(base_end))
    if base_end <= base_start:
        base_start = 1
        base_end = max(2, min(len(delta), 10))
    seg = delta[base_start - 1:base_end]
    if len(seg) < 2:
        return float(seg[0] if seg else 0.0)

    m = sum(seg) / len(seg)
    var = sum((x - m) ** 2 for x in seg) / max(1, (len(seg) - 1))
    s = math.sqrt(max(var, 1e-12))
    return m + float(k_sigma) * s


def calc_ct(delta: List[float], threshold: float) -> Optional[float]:
    if len(delta) < 2:
        return None
    thr = float(threshold)
    for i in range(1, len(delta)):
        y0 = float(delta[i - 1])
        y1 = float(delta[i])
        if y0 < thr <= y1 and (y1 - y0) != 0:
            frac = (thr - y0) / (y1 - y0)
            return (i) + frac
        if y0 == thr:
            return float(i)
    return None


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Frame:
    t_host: float
    ch: int
    led_en: int
    vpulse_mv: int
    ready: int
    ton_ms: int
    vext_en: int
    vext_mv: int
    mon_mv: int
    fluo_mv: int
    mon_raw: int
    fluo_raw: int
    tick_ms: int


@dataclass
class LedSchedule:
    enabled: bool = False
    delay_s: int = 0
    on_s: int = 5
    period_s: int = 30
    duration_s: int = 300


# -----------------------------
# Serial worker (QThread)
# -----------------------------
class SerialWorker(QtCore.QThread):
    line_received = QtCore.pyqtSignal(str)
    status_changed = QtCore.pyqtSignal(bool, str)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self._running = True
        self._ser: Optional[serial.Serial] = None
        self._tx_queue = deque()

    def stop(self):
        self._running = False

    @QtCore.pyqtSlot(str)
    def send_line(self, s: str):
        self._tx_queue.append(s)

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self.status_changed.emit(True, f"Connected: {self.port} @ {self.baud}")
        except Exception as e:
            self.status_changed.emit(False, f"Open failed: {e}")
            return

        buf = b""
        while self._running:
            # TX
            try:
                while self._tx_queue:
                    msg = self._tx_queue.popleft()
                    if not msg.endswith("\n"):
                        msg += "\n"
                    self._ser.write(msg.encode("utf-8", errors="ignore"))
            except Exception as e:
                self.status_changed.emit(False, f"TX error: {e}")
                break

            # RX
            try:
                data = self._ser.read(1024)
                if data:
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        s = line.decode("utf-8", errors="ignore").strip()
                        if s:
                            self.line_received.emit(s)
            except Exception as e:
                self.status_changed.emit(False, f"RX error: {e}")
                break

        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self.status_changed.emit(False, "Disconnected")


# -----------------------------
# Command scheduler (rate limit)
# -----------------------------
class CommandScheduler(QtCore.QObject):
    def __init__(self, send_func, min_gap_ms: int = 25, parent=None):
        super().__init__(parent)
        self._send_func = send_func
        self._min_gap = int(min_gap_ms)
        self._q = deque()
        self._t = QtCore.QTimer(self)
        self._t.timeout.connect(self._drain_once)
        self._t.start(5)
        self._last_tx_ms = 0

    def set_gap(self, ms: int):
        self._min_gap = max(0, int(ms))

    def push(self, cmd: str):
        self._q.append(cmd)

    def clear(self):
        self._q.clear()

    def _drain_once(self):
        if not self._q:
            return
        now_ms = int(time.time() * 1000)
        if (now_ms - self._last_tx_ms) < self._min_gap:
            return
        cmd = self._q.popleft()
        self._send_func(cmd)
        self._last_tx_ms = now_ms


# -----------------------------
# NORMAL Measure (per-channel concurrent)
# -----------------------------
@dataclass
class NormalMeasureTask:
    ch: int
    stage: str  # "DELAY" -> "ACQ"
    stage_t0: float
    delay_s: float
    window_s: float
    acc_mon: List[int]
    acc_fluo: List[int]
    prev_led_on: Optional[int] = None


# -----------------------------
# Excel FLUO plotter panel (used inside separate window)
# -----------------------------
class ExcelFluoPlotPanel(QtWidgets.QWidget):
    """
    Load xlsx saved by this app and plot FLUO only.
    - sheet select
    - channel checkboxes
    - smoothing (moving average)
    - save png button
    """
    def __init__(self, log_func, parent=None):
        super().__init__(parent)
        self._log = log_func
        self._xlsx_path: Optional[str] = None
        self._sheets: Dict[str, pd.DataFrame] = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Controls
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.btn_load = QtWidgets.QPushButton("Load Excel (.xlsx)")
        self.btn_load.clicked.connect(self._on_load_clicked)

        self.btn_save_png = QtWidgets.QPushButton("Save PNG")
        self.btn_save_png.clicked.connect(self._on_save_png)

        self.cb_sheet = QtWidgets.QComboBox()
        self.cb_sheet.currentTextChanged.connect(self._replot)

        self.cb_ch = []
        for i in range(4):
            cb = QtWidgets.QCheckBox(f"CH{i+1}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._replot)
            self.cb_ch.append(cb)

        self.sp_smooth = QtWidgets.QSpinBox()
        self.sp_smooth.setRange(1, 9999)
        self.sp_smooth.setValue(1)
        self.sp_smooth.valueChanged.connect(self._replot)

        row.addWidget(self.btn_load)
        row.addWidget(self.btn_save_png)
        row.addWidget(QtWidgets.QLabel("Sheet"))
        row.addWidget(self.cb_sheet, 1)
        row.addSpacing(10)
        for cb in self.cb_ch:
            row.addWidget(cb)
        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("Smoothing(win)"))
        row.addWidget(self.sp_smooth)
        lay.addLayout(row)

        # Plot
        self.plot = pg.PlotWidget(title="Excel FLUO (only)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        self.plot.setLabel("left", "FLUO (mV)")
        self.plot.setLabel("bottom", "t / index / cycle")
        lay.addWidget(self.plot, 1)

        pens = [
            pg.mkPen((255, 0, 0), width=2),
            pg.mkPen((0, 128, 255), width=2),
            pg.mkPen((0, 180, 80), width=2),
            pg.mkPen((200, 120, 0), width=2),
        ]
        self.curves: Dict[int, pg.PlotDataItem] = {}
        for ch in range(1, 5):
            self.curves[ch] = self.plot.plot([], [], pen=pens[ch-1], name=f"CH{ch}")

        self.lbl_info = QtWidgets.QLabel("No file loaded.")
        self.lbl_info.setStyleSheet("color:#b9c3d6; font-weight:700;")
        lay.addWidget(self.lbl_info, 0)

    # --- external API (MainWindow can call) ---
    def load_excel(self, path: str):
        if not path:
            return
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            self._sheets.clear()
            for name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=name, engine="openpyxl")
                    self._sheets[name] = df
                except Exception:
                    pass
            self._xlsx_path = path

            self.cb_sheet.blockSignals(True)
            self.cb_sheet.clear()
            for name in self._sheets.keys():
                self.cb_sheet.addItem(name)
            self.cb_sheet.blockSignals(False)

            preferred = None
            for cand in ["RECORD", "QPCR", "NORMAL_MEASURE"]:
                if cand in self._sheets:
                    preferred = cand
                    break
            if preferred:
                self.cb_sheet.setCurrentText(preferred)
            elif self.cb_sheet.count() > 0:
                self.cb_sheet.setCurrentIndex(0)

            self._log(f"Excel loaded: {path} (sheets={list(self._sheets.keys())})")
            self.lbl_info.setText(f"Loaded: {path}")
            self._replot()
        except Exception as e:
            self._log(f"Excel load failed: {e}")
            self.lbl_info.setText(f"Load failed: {e}")

    def _on_load_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Excel", "", "Excel (*.xlsx)")
        if not path:
            return
        self.load_excel(path)

    def _on_save_png(self):
        if self.plot is None:
            return
        default = f"fluo_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot as PNG", default, "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            exporter = pg_exporters.ImageExporter(self.plot.plotItem)
            exporter.export(path)
            self._log(f"PNG saved: {path}")
        except Exception as e:
            self._log(f"PNG save failed: {e}")

    def _pick_xy_and_fluo_col(self, df: pd.DataFrame) -> Tuple[List[float], str]:
        cols = set(df.columns.astype(str))

        y_candidates = [
            "fluo_mv",              # RECORD
            "fluo_avg_mv",          # NORMAL_MEASURE
            "light_fluo_mean_mv",   # QPCR
            "dark_fluo_mean_mv",    # QPCR (fallback)
            "R_light_minus_dark",   # QPCR (related)
            "delta",                # QPCR delta (fallback)
        ]
        ycol = None
        for c in y_candidates:
            if c in cols:
                ycol = c
                break
        if ycol is None:
            lower_map = {str(c).lower(): str(c) for c in df.columns}
            for c in y_candidates:
                if c.lower() in lower_map:
                    ycol = lower_map[c.lower()]
                    break
        if ycol is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return list(range(len(df))), ""
            ycol = str(num_cols[0])

        x_candidates = [
            "record_t_rel_s",  # RECORD
            "t_rel_s",         # QPCR
            "cycle",           # QPCR
            "tick_ms",
        ]
        xcol = None
        for c in x_candidates:
            if c in cols:
                xcol = c
                break
        if xcol is None:
            x = list(range(len(df)))
        else:
            x = pd.to_numeric(df[xcol], errors="coerce").ffill().fillna(0).tolist()

        return [float(v) for v in x], ycol

    def _replot(self):
        sheet = self.cb_sheet.currentText().strip()
        if not sheet or sheet not in self._sheets:
            for ch in range(1, 5):
                self.curves[ch].setData([], [])
            return

        df = self._sheets[sheet]
        if df is None or df.empty:
            for ch in range(1, 5):
                self.curves[ch].setData([], [])
            self.lbl_info.setText(f"Loaded: {self._xlsx_path} / sheet={sheet} (empty)")
            return

        x, ycol = self._pick_xy_and_fluo_col(df)
        if not ycol:
            self.lbl_info.setText(f"Loaded: {self._xlsx_path} / sheet={sheet} (no numeric FLUO column)")
            for ch in range(1, 5):
                self.curves[ch].setData([], [])
            return

        smooth_win = int(self.sp_smooth.value())
        has_ch = "ch" in set(df.columns.astype(str))

        plotted_any = False
        for ch in range(1, 5):
            if not self.cb_ch[ch - 1].isChecked():
                self.curves[ch].setData([], [])
                continue

            if has_ch:
                dch = df[df["ch"] == ch]
                if dch.empty:
                    self.curves[ch].setData([], [])
                    continue
                xch, ycol2 = self._pick_xy_and_fluo_col(dch)
                y = pd.to_numeric(dch[ycol2], errors="coerce").ffill().fillna(0).tolist()
                y = [float(v) for v in y]
                y_sm = moving_average(y, smooth_win)
                self.curves[ch].setData(xch, y_sm)
                plotted_any = True
            else:
                y = pd.to_numeric(df[ycol], errors="coerce").ffill().fillna(0).tolist()
                y = [float(v) for v in y]
                y_sm = moving_average(y, smooth_win)
                if ch == 1:
                    self.curves[ch].setData(x, y_sm)
                    plotted_any = True
                else:
                    self.curves[ch].setData([], [])

        self.plot.setTitle(f"Excel FLUO only | sheet={sheet} | y={ycol} | smooth={smooth_win}")
        if plotted_any:
            self.lbl_info.setText(f"Loaded: {self._xlsx_path} / sheet={sheet} | y={ycol}")
        else:
            self.lbl_info.setText(f"Loaded: {self._xlsx_path} / sheet={sheet} | (no selected data)")


class ExcelFluoPlotWindow(QtWidgets.QWidget):
    """Separate window for Excel FLUO plot."""
    def __init__(self, log_func, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Excel FLUO Plot (Saved xlsx)")
        self.resize(1200, 750)
        lay = QtWidgets.QVBoxLayout(self)
        self.panel = ExcelFluoPlotPanel(log_func)
        lay.addWidget(self.panel, 1)

    def load_excel(self, path: str):
        self.panel.load_excel(path)


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    MODE_NORMAL = "NORMAL"
    MODE_RECORD = "RECORD"
    MODE_QPCR = "QPCR"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("light2 - NORMAL / RECORD / QPCR (4CH)")
        self.resize(1750, 1100)

        self.worker: Optional[SerialWorker] = None
        self.cmd_sched = CommandScheduler(self._send_immediate, min_gap_ms=25)

        # realtime buffers
        self.win_points = 600
        self.x = list(range(self.win_points))
        self.buf_vpulse = [deque([0] * self.win_points, maxlen=self.win_points) for _ in range(4)]
        self.buf_mon_mv = [deque([0] * self.win_points, maxlen=self.win_points) for _ in range(4)]
        self.buf_fluo_mv = [deque([0] * self.win_points, maxlen=self.win_points) for _ in range(4)]
        self.last_frame: Dict[int, Frame] = {}

        # data storage
        self.record_rows: List[dict] = []
        self.qpcr_rows: List[dict] = []
        self.normal_rows: List[dict] = []

        # schedules (RECORD)
        self.led_sched: List[LedSchedule] = [LedSchedule() for _ in range(4)]

        # mode
        self.mode = self.MODE_NORMAL

        # record running
        self.record_running = False
        self.record_t0 = 0.0
        self.record_end_epoch = 0.0
        self._last_sent_led_mask = 0
        self.capture_hz = 5
        self._last_save_epoch = [0.0, 0.0, 0.0, 0.0]

        # ✅ RECORD running average (display averages only)
        self.record_acc_n = [0, 0, 0, 0]
        self.record_acc_mon = [0.0, 0.0, 0.0, 0.0]
        self.record_acc_fluo = [0.0, 0.0, 0.0, 0.0]
        self.lbl_rec_n: List[QtWidgets.QLabel] = []
        self.lbl_rec_mon_avg: List[QtWidgets.QLabel] = []
        self.lbl_rec_fluo_avg: List[QtWidgets.QLabel] = []

        # ---- QPCR state machine ----
        self.qpcr_running = False
        self.qpcr_t0 = 0.0
        self.qpcr_cycle_idx = 0
        self.qpcr_active_ch_list: List[int] = []
        self.qpcr_ch_ptr = 0
        self.qpcr_state = "IDLE"
        self.qpcr_state_t = 0.0

        self.qpcr_dark_samples: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
        self.qpcr_light_samples: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
        self.qpcr_mon_samples: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}

        self.qpcr_R: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
        self.qpcr_delta: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
        self.qpcr_ct: Dict[int, Optional[float]] = {1: None, 2: None, 3: None, 4: None}
        self.qpcr_threshold: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        self.qpcr_plot_win: Optional[QtWidgets.QWidget] = None
        self.qpcr_plot: Optional[pg.PlotWidget] = None
        self.qpcr_plot_curve: Dict[int, pg.PlotDataItem] = {}
        self.qpcr_thr_line: Dict[int, pg.InfiniteLine] = {}

        # ✅ QPCR averages display label per channel (D/L/R/M means)
        self.lbl_qp_avg: List[QtWidgets.QLabel] = []

        # NORMAL measure tasks (concurrent)
        self.meas_tasks: Dict[int, NormalMeasureTask] = {}

        # Excel plot window (separate)
        self.excel_win: Optional[ExcelFluoPlotWindow] = None

        # timers
        self.logic_timer = QtCore.QTimer(self)
        self.logic_timer.timeout.connect(self._logic_tick)

        self.qpcr_timer = QtCore.QTimer(self)
        self.qpcr_timer.timeout.connect(self._qpcr_tick)

        self.normal_timer = QtCore.QTimer(self)
        self.normal_timer.timeout.connect(self._normal_measure_tick)
        self.normal_timer.start(20)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._refresh_plots)
        self.ui_timer.start(100)

        self._extra_windows = []

        self._apply_instrument_style()
        self._build_ui()
        self._refresh_ports()
        self._on_mode_changed(self.MODE_NORMAL)

    # ---------------- Style ----------------
    def _apply_instrument_style(self):
        QtWidgets.QApplication.setStyle("Fusion")
        self.setStyleSheet("""
            QWidget { font-size: 10pt; color: #e7eefc; background: #0b0f14; }

            QFrame[panel="1"] {
                border: 1px solid #2b3240;
                border-radius: 12px;
                background: #0f131a;
            }

            QLabel[panelTitleLabel="1"] {
                font-size: 15pt;
                font-weight: 800;
                padding: 0px 2px;
                color: #dbe6ff;
            }

            QToolButton[panelTitle="1"] {
                font-size: 15pt;
                font-weight: 800;
                padding: 2px 2px;
                border: 0px;
                background: transparent;
                color: #dbe6ff;
            }
            QToolButton[panelTitle="1"]:hover { color: #ffffff; }

            QPushButton {
                border: 1px solid #2f3a4c;
                border-radius: 10px;
                padding: 8px 12px;
                background: #1a1f27;
                font-weight: 700;
            }
            QPushButton:hover { background: #222a35; }
            QPushButton:pressed { background: #131821; }
            QPushButton:disabled { color:#70809b; background:#10141b; border:1px solid #202836; }

            QPushButton#btn_record, QPushButton#btn_qpcr {
                font-size: 11pt;
                padding: 8px 12px;
                border-radius: 10px;
            }

            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #2f3a4c;
                border-radius: 8px;
                padding: 5px 8px;
                background: #10141b;
            }

            QCheckBox { spacing: 8px; }

            QLabel[lamp="1"] {
                min-width: 14px; min-height: 14px;
                max-width: 14px; max-height: 14px;
                border-radius: 7px;
                border: 1px solid #2f3a4c;
                background: #3a4150;
            }
            QLabel[lamp="1"][state="on"]   { background: #00c853; }
            QLabel[lamp="1"][state="warn"] { background: #ffab00; }
            QLabel[lamp="1"][state="err"]  { background: #ff5252; }

            QPlainTextEdit {
                border: 1px solid #2b3240;
                border-radius: 10px;
                background: #070a0f;
                color: #d6def2;
            }

            /* ✅ Port/Mode 패널만 글자 더 작게 */
            QFrame#pnl_port QLabel[panelTitleLabel="1"],
            QFrame#pnl_mode QLabel[panelTitleLabel="1"],
            QFrame#pnl_port QToolButton[panelTitle="1"],
            QFrame#pnl_mode QToolButton[panelTitle="1"] {
                font-size: 12pt;
                font-weight: 800;
            }

            /* ✅ Port/Mode 패널 내부 컨트롤(라벨/콤보/버튼 등) 폰트 축소 */
            QFrame#pnl_port QWidget,
            QFrame#pnl_mode QWidget {
                font-size: 9pt;
            }

            /* Mode 패널의 큰 버튼(Record/QPCR)도 같이 축소 */
            QFrame#pnl_mode QPushButton#btn_record,
            QFrame#pnl_mode QPushButton#btn_qpcr {
                font-size: 9pt;
                padding: 6px 10px;
            }
        """)

    # ---------------- UI build ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # =========================
        # Row 1: Port / Mode
        # =========================
        top_row = QtWidgets.QWidget()
        top_l = QtWidgets.QHBoxLayout(top_row)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(10)

        self.pnl_port = InstrumentPanel("Port:", collapsible=False)
        self.pnl_mode = InstrumentPanel("Mode:", collapsible=False)

        # ✅ QSS target
        self.pnl_port.setObjectName("pnl_port")
        self.pnl_mode.setObjectName("pnl_mode")

        top_l.addWidget(self.pnl_port, 1)
        top_l.addWidget(self.pnl_mode, 1)
        root.addWidget(top_row, 0)

        # Port panel content
        port_row = QtWidgets.QHBoxLayout()
        port_row.setContentsMargins(0, 0, 0, 0)
        port_row.setSpacing(8)

        self.cb_port = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.lbl_status_lamp = QtWidgets.QLabel("")
        self.lbl_status_lamp.setProperty("lamp", "1")
        self.lbl_status_lamp.setProperty("state", "err")

        self.lbl_status = QtWidgets.QLabel("Not connected")
        self.lbl_status.setStyleSheet("color:#b9c3d6;")

        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.btn_connect.clicked.connect(self._toggle_connect)

        port_row.addWidget(QtWidgets.QLabel("Port"))
        port_row.addWidget(self.cb_port, 1)
        port_row.addWidget(self.btn_refresh)
        port_row.addWidget(self.btn_connect)
        port_row.addSpacing(6)
        port_row.addWidget(self.lbl_status_lamp, 0)
        port_row.addWidget(self.lbl_status, 2)

        self.pnl_port.content_l.addLayout(port_row)

        # Mode panel content
        mode_grid = QtWidgets.QGridLayout()
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setHorizontalSpacing(8)
        mode_grid.setVerticalSpacing(6)

        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems([self.MODE_NORMAL, self.MODE_RECORD, self.MODE_QPCR])
        self.cb_mode.currentTextChanged.connect(self._on_mode_changed)

        self.btn_stream = QtWidgets.QPushButton("STREAM ON")
        self.btn_stream.setCheckable(True)
        self.btn_stream.setChecked(True)
        self.btn_stream.clicked.connect(self._toggle_stream)

        self.btn_record = QtWidgets.QPushButton("RECORD START")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._toggle_record)

        self.btn_qpcr = QtWidgets.QPushButton("QPCR START")
        self.btn_qpcr.setObjectName("btn_qpcr")
        self.btn_qpcr.setCheckable(True)
        self.btn_qpcr.clicked.connect(self._toggle_qpcr)

        self.btn_save_excel = QtWidgets.QPushButton("Save Excel")
        self.btn_save_excel.clicked.connect(self._save_excel)

        # Excel plot separate window open button
        self.btn_excel_plot = QtWidgets.QPushButton("Excel Plot")
        self.btn_excel_plot.clicked.connect(self._open_excel_plot_window)

        self.lbl_run = QtWidgets.QLabel("IDLE")
        self.lbl_run.setStyleSheet("color:#b9c3d6; font-weight:700;")

        mode_grid.addWidget(QtWidgets.QLabel("MODE"), 0, 0)
        mode_grid.addWidget(self.cb_mode, 0, 1)
        mode_grid.addWidget(self.btn_stream, 0, 2)
        mode_grid.addWidget(self.lbl_run, 0, 3)

        mode_grid.addWidget(self.btn_record, 1, 1)
        mode_grid.addWidget(self.btn_qpcr, 1, 2)
        mode_grid.addWidget(self.btn_save_excel, 1, 3)
        mode_grid.addWidget(self.btn_excel_plot, 1, 4)

        self.pnl_mode.content_l.addLayout(mode_grid)

        self.pnl_port.setMaximumHeight(115)
        self.pnl_mode.setMaximumHeight(115)

        # =========================
        # Row 2: Normal Measure
        # =========================
        self.pnl_normal = InstrumentPanel("Normal Measure", collapsible=True, collapsed=False)
        root.addWidget(self.pnl_normal, 0)

        meas_row = QtWidgets.QHBoxLayout()
        meas_row.setContentsMargins(0, 0, 0, 0)
        meas_row.setSpacing(8)

        self.sp_meas_delay_ms = QtWidgets.QSpinBox()
        self.sp_meas_delay_ms.setRange(0, 5000)
        self.sp_meas_delay_ms.setValue(300)

        self.sp_meas_window_ms = QtWidgets.QSpinBox()
        self.sp_meas_window_ms.setRange(200, 20000)
        self.sp_meas_window_ms.setValue(1500)

        self.btn_meas_stop = QtWidgets.QPushButton("STOP MEASURE ALL")
        self.btn_meas_stop.clicked.connect(self._normal_measure_stop_all)
        self.btn_meas_stop.setEnabled(False)

        meas_row.addWidget(QtWidgets.QLabel("Delay(ms)"))
        meas_row.addWidget(self.sp_meas_delay_ms)
        meas_row.addWidget(QtWidgets.QLabel("Window(ms)"))
        meas_row.addWidget(self.sp_meas_window_ms)
        meas_row.addWidget(self.btn_meas_stop)
        meas_row.addStretch(1)

        self.pnl_normal.content_l.addLayout(meas_row)

        # Channel grid
        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        headers = [
            "CH", "LED ON(3.3V)", "MEASURE", "VEXTEN", "VEXT(mV)",
            "VPULSE(mV)", "MON(mV)", "FLUO(mV)", "MON_AVG", "FLUO_AVG",
            "MON_RAW", "FLUO_RAW", "READY"
        ]
        for c, h in enumerate(headers):
            lab = QtWidgets.QLabel(h)
            lab.setStyleSheet("font-weight: 800; color:#b9c3d6;")
            grid.addWidget(lab, 0, c)

        self.btn_led = []
        self.btn_measure = []
        self.btn_vexten = []
        self.sp_vext_mv = []
        self.sl_vext = []

        self.lbl_vpulse = []
        self.lbl_mon_mv = []
        self.lbl_fluo_mv = []
        self.lbl_mon_avg = []
        self.lbl_fluo_avg = []
        self.lbl_mon_raw = []
        self.lbl_fluo_raw = []
        self.lbl_ready = []

        for i in range(4):
            ch = i + 1
            grid.addWidget(QtWidgets.QLabel(str(ch)), ch, 0)

            b_led = QtWidgets.QPushButton("OFF")
            b_led.setCheckable(True)
            b_led.clicked.connect(lambda checked, c=ch: self._cmd_led(c, checked))
            self.btn_led.append(b_led)
            grid.addWidget(b_led, ch, 1)

            b_meas = QtWidgets.QPushButton("MEASURE")
            b_meas.clicked.connect(lambda _=False, c=ch: self._normal_measure_start(c))
            self.btn_measure.append(b_meas)
            grid.addWidget(b_meas, ch, 2)

            b_ve = QtWidgets.QPushButton("OFF")
            b_ve.setCheckable(True)
            b_ve.clicked.connect(lambda checked, c=ch: self._cmd_vexten(c, checked))
            self.btn_vexten.append(b_ve)
            grid.addWidget(b_ve, ch, 3)

            vbox = QtWidgets.QHBoxLayout()
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)

            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(0, 1800)
            sl.setValue(1500)
            self.sl_vext.append(sl)

            sp = QtWidgets.QSpinBox()
            sp.setRange(0, 1800)
            sp.setValue(1500)
            sp.setFixedWidth(90)
            self.sp_vext_mv.append(sp)

            def apply_vext_from_ui(cch: int, mv: int):
                mv = max(0, min(1800, int(mv)))
                self.sp_vext_mv[cch - 1].blockSignals(True)
                self.sl_vext[cch - 1].blockSignals(True)
                self.sp_vext_mv[cch - 1].setValue(mv)
                self.sl_vext[cch - 1].setValue(mv)
                self.sp_vext_mv[cch - 1].blockSignals(False)
                self.sl_vext[cch - 1].blockSignals(False)
                self._cmd_vext(cch, mv)

            sl.valueChanged.connect(lambda v, cch=ch: apply_vext_from_ui(cch, v))
            sp.valueChanged.connect(lambda v, cch=ch: apply_vext_from_ui(cch, v))

            vbox.addWidget(sl, 1)
            vbox.addWidget(sp, 0)
            wv = QtWidgets.QWidget()
            wv.setLayout(vbox)
            grid.addWidget(wv, ch, 4)

            lv = QtWidgets.QLabel("0")
            lm = QtWidgets.QLabel("0")
            lf = QtWidgets.QLabel("0")
            lma = QtWidgets.QLabel("-")
            lfa = QtWidgets.QLabel("-")
            lmr = QtWidgets.QLabel("0")
            lfr = QtWidgets.QLabel("0")
            lr = QtWidgets.QLabel("0")

            lv.setStyleSheet("color:#ff6b6b; font-weight:800;")
            lm.setStyleSheet("color:#4dabff; font-weight:800;")
            lf.setStyleSheet("color:#38d9a9; font-weight:800;")
            lma.setStyleSheet("color:#a5d8ff; font-weight:800;")
            lfa.setStyleSheet("color:#8ce99a; font-weight:800;")
            lr.setStyleSheet("color:#b9c3d6; font-weight:800;")

            self.lbl_vpulse.append(lv)
            self.lbl_mon_mv.append(lm)
            self.lbl_fluo_mv.append(lf)
            self.lbl_mon_avg.append(lma)
            self.lbl_fluo_avg.append(lfa)
            self.lbl_mon_raw.append(lmr)
            self.lbl_fluo_raw.append(lfr)
            self.lbl_ready.append(lr)

            col = 5
            grid.addWidget(lv, ch, col); col += 1
            grid.addWidget(lm, ch, col); col += 1
            grid.addWidget(lf, ch, col); col += 1
            grid.addWidget(lma, ch, col); col += 1
            grid.addWidget(lfa, ch, col); col += 1
            grid.addWidget(lmr, ch, col); col += 1
            grid.addWidget(lfr, ch, col); col += 1
            grid.addWidget(lr, ch, col)

        normal_scroll = QtWidgets.QScrollArea()
        normal_scroll.setWidgetResizable(True)
        normal_scroll.setWidget(grid_widget)
        normal_scroll.setMinimumHeight(240)
        normal_scroll.setMaximumHeight(320)
        normal_scroll.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.pnl_normal.content_l.addWidget(normal_scroll)

        # =========================
        # Stream / Record
        # =========================
        self.pnl_stream = InstrumentPanel("Stream / Record", collapsible=True, collapsed=True)
        root.addWidget(self.pnl_stream, 0)

        setrow = QtWidgets.QGridLayout()
        setrow.setContentsMargins(0, 0, 0, 0)
        setrow.setHorizontalSpacing(8)
        setrow.setVerticalSpacing(6)

        self.sp_period = QtWidgets.QSpinBox()
        self.sp_period.setRange(10, 5000)
        self.sp_period.setValue(50)
        self.sp_period.valueChanged.connect(lambda v: self._queue(f"SET PERIOD {v}"))

        self.sp_delayon = QtWidgets.QSpinBox()
        self.sp_delayon.setRange(0, 5000)
        self.sp_delayon.setValue(300)
        self.sp_delayon.valueChanged.connect(lambda v: self._queue(f"SET DELAYON {v}"))

        self.sp_avg = QtWidgets.QSpinBox()
        self.sp_avg.setRange(1, 512)
        self.sp_avg.setValue(64)
        self.sp_avg.valueChanged.connect(lambda v: self._queue(f"SET AVG {v}"))

        self.sp_dummy = QtWidgets.QSpinBox()
        self.sp_dummy.setRange(0, 50)
        self.sp_dummy.setValue(6)
        self.sp_dummy.valueChanged.connect(lambda v: self._queue(f"SET DUMMY {v}"))

        self.cb_dark = QtWidgets.QCheckBox("DARK measure (READY=0도 측정)")
        self.cb_dark.setChecked(True)
        self.cb_dark.stateChanged.connect(lambda _: self._queue(f"SET DARK {1 if self.cb_dark.isChecked() else 0}"))

        self.sp_vdda = QtWidgets.QSpinBox()
        self.sp_vdda.setRange(2500, 3600)
        self.sp_vdda.setValue(3300)
        self.sp_vdda.valueChanged.connect(lambda v: self._queue(f"SET VDDA {v}"))

        self.sp_cmd_gap = QtWidgets.QSpinBox()
        self.sp_cmd_gap.setRange(0, 200)
        self.sp_cmd_gap.setValue(25)
        self.sp_cmd_gap.valueChanged.connect(lambda v: self.cmd_sched.set_gap(v))

        r = 0
        setrow.addWidget(QtWidgets.QLabel("PERIOD(ms)"), r, 0); setrow.addWidget(self.sp_period, r, 1)
        setrow.addWidget(QtWidgets.QLabel("DELAYON(ms)"), r, 2); setrow.addWidget(self.sp_delayon, r, 3)
        setrow.addWidget(QtWidgets.QLabel("AVG"), r, 4); setrow.addWidget(self.sp_avg, r, 5)
        setrow.addWidget(QtWidgets.QLabel("DUMMY"), r, 6); setrow.addWidget(self.sp_dummy, r, 7)

        r = 1
        setrow.addWidget(self.cb_dark, r, 0, 1, 3)
        setrow.addWidget(QtWidgets.QLabel("VDDA(mV)"), r, 3); setrow.addWidget(self.sp_vdda, r, 4)
        setrow.addWidget(QtWidgets.QLabel("CMD GAP(ms)"), r, 5); setrow.addWidget(self.sp_cmd_gap, r, 6)

        self.pnl_stream.content_l.addLayout(setrow)

        cap_row = QtWidgets.QHBoxLayout()
        cap_row.setContentsMargins(0, 0, 0, 0)
        cap_row.setSpacing(8)

        self.sp_capture_hz = QtWidgets.QSpinBox()
        self.sp_capture_hz.setRange(1, 200)
        self.sp_capture_hz.setValue(5)
        self.sp_capture_hz.valueChanged.connect(self._on_capture_hz_changed)

        cap_row.addWidget(QtWidgets.QLabel("RECORD 저장 빈도(회/초/채널)"))
        cap_row.addWidget(self.sp_capture_hz)
        cap_row.addStretch(1)
        self.pnl_stream.content_l.addLayout(cap_row)

        # ✅ RECORD 평균 요약(채널별): Samples / MON_AVG / FLUO_AVG
        rec_sum_group = QtWidgets.QWidget()
        rec_sum = QtWidgets.QGridLayout(rec_sum_group)
        rec_sum.setContentsMargins(0, 0, 0, 0)
        rec_sum.setHorizontalSpacing(12)
        rec_sum.setVerticalSpacing(6)

        headers2 = ["CH", "Samples", "MON_AVG(mV)", "FLUO_AVG(mV)"]
        for c, h in enumerate(headers2):
            lab = QtWidgets.QLabel(h)
            lab.setStyleSheet("font-weight:800; color:#b9c3d6;")
            rec_sum.addWidget(lab, 0, c)

        self.lbl_rec_n = []
        self.lbl_rec_mon_avg = []
        self.lbl_rec_fluo_avg = []

        for i in range(4):
            ch = i + 1
            rec_sum.addWidget(QtWidgets.QLabel(f"CH{ch}"), ch, 0)

            ln = QtWidgets.QLabel("-")
            lm = QtWidgets.QLabel("-")
            lf = QtWidgets.QLabel("-")
            ln.setStyleSheet("font-weight:800; color:#dbe6ff;")
            lm.setStyleSheet("font-weight:800; color:#4dabff;")
            lf.setStyleSheet("font-weight:800; color:#38d9a9;")

            self.lbl_rec_n.append(ln)
            self.lbl_rec_mon_avg.append(lm)
            self.lbl_rec_fluo_avg.append(lf)

            rec_sum.addWidget(ln, ch, 1)
            rec_sum.addWidget(lm, ch, 2)
            rec_sum.addWidget(lf, ch, 3)

        self.pnl_stream.content_l.addWidget(rec_sum_group)

        # schedule table
        sch_group = QtWidgets.QWidget()
        sch_layout = QtWidgets.QGridLayout(sch_group)
        sch_layout.setContentsMargins(0, 0, 0, 0)
        sch_layout.setHorizontalSpacing(8)
        sch_layout.setVerticalSpacing(6)

        sch_headers = ["LED", "Enable", "Start Delay (s)", "LED ON (s)", "Period (s)", "Duration (s)"]
        for c, h in enumerate(sch_headers):
            lab = QtWidgets.QLabel(h)
            lab.setStyleSheet("font-weight:800; color:#b9c3d6;")
            sch_layout.addWidget(lab, 0, c)

        self.cb_led_en: List[QtWidgets.QCheckBox] = []
        self.sp_delay_s: List[QtWidgets.QSpinBox] = []
        self.sp_on_s: List[QtWidgets.QSpinBox] = []
        self.sp_period_s: List[QtWidgets.QSpinBox] = []
        self.sp_dur_s: List[QtWidgets.QSpinBox] = []

        for i in range(4):
            rowi = i + 1
            sch_layout.addWidget(QtWidgets.QLabel(f"LED{i+1}"), rowi, 0)

            cb = QtWidgets.QCheckBox()
            cb.setChecked(True)
            self.cb_led_en.append(cb)
            sch_layout.addWidget(cb, rowi, 1)

            sp_d = QtWidgets.QSpinBox()
            sp_d.setRange(0, 86400)
            sp_d.setValue(0)
            self.sp_delay_s.append(sp_d)
            sch_layout.addWidget(sp_d, rowi, 2)

            sp_on = QtWidgets.QSpinBox()
            sp_on.setRange(1, 3600)
            sp_on.setValue(5)
            self.sp_on_s.append(sp_on)
            sch_layout.addWidget(sp_on, rowi, 3)

            sp_p = QtWidgets.QSpinBox()
            sp_p.setRange(1, 86400)
            sp_p.setValue(30)
            self.sp_period_s.append(sp_p)
            sch_layout.addWidget(sp_p, rowi, 4)

            sp_t = QtWidgets.QSpinBox()
            sp_t.setRange(1, 86400)
            sp_t.setValue(300)
            self.sp_dur_s.append(sp_t)
            sch_layout.addWidget(sp_t, rowi, 5)

        sch_scroll = QtWidgets.QScrollArea()
        sch_scroll.setWidgetResizable(True)
        sch_scroll.setWidget(sch_group)
        sch_scroll.setMinimumHeight(160)
        sch_scroll.setMaximumHeight(220)
        sch_scroll.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.pnl_stream.content_l.addWidget(sch_scroll)

        # =========================
        # Row 4: QPCR
        # =========================
        self.pnl_qpcr = InstrumentPanel("QPCR Measure", collapsible=True, collapsed=True)
        root.addWidget(self.pnl_qpcr, 0)

        qp_group = QtWidgets.QWidget()
        qp = QtWidgets.QGridLayout(qp_group)
        qp.setContentsMargins(0, 0, 0, 0)
        qp.setHorizontalSpacing(8)
        qp.setVerticalSpacing(6)

        qp.addWidget(QtWidgets.QLabel("CH"), 0, 0)
        qp.addWidget(QtWidgets.QLabel("Enable"), 0, 1)
        qp.addWidget(QtWidgets.QLabel("OFF(ms) (dark)"), 0, 2)
        qp.addWidget(QtWidgets.QLabel("ON(ms)"), 0, 3)
        qp.addWidget(QtWidgets.QLabel("Acquire(ms)"), 0, 4)
        qp.addWidget(QtWidgets.QLabel("Baseline start"), 0, 5)
        qp.addWidget(QtWidgets.QLabel("Baseline end"), 0, 6)
        qp.addWidget(QtWidgets.QLabel("Avg (D/L/R/M)"), 0, 7)

        self.cb_qp_en: List[QtWidgets.QCheckBox] = []
        self.sp_qp_off_ms: List[QtWidgets.QSpinBox] = []
        self.sp_qp_on_ms: List[QtWidgets.QSpinBox] = []
        self.sp_qp_acq_ms: List[QtWidgets.QSpinBox] = []
        self.sp_base_s: List[QtWidgets.QSpinBox] = []
        self.sp_base_e: List[QtWidgets.QSpinBox] = []
        self.lbl_qp_avg = []

        for i in range(4):
            ch = i + 1
            qp.addWidget(QtWidgets.QLabel(f"CH{ch}"), ch, 0)

            cb = QtWidgets.QCheckBox()
            cb.setChecked(True)
            self.cb_qp_en.append(cb)
            qp.addWidget(cb, ch, 1)

            sp_off = QtWidgets.QSpinBox()
            sp_off.setRange(0, 10000)
            sp_off.setValue(200)
            self.sp_qp_off_ms.append(sp_off)
            qp.addWidget(sp_off, ch, 2)

            sp_on = QtWidgets.QSpinBox()
            sp_on.setRange(50, 10000)
            sp_on.setValue(400)
            self.sp_qp_on_ms.append(sp_on)
            qp.addWidget(sp_on, ch, 3)

            sp_acq = QtWidgets.QSpinBox()
            sp_acq.setRange(20, 10000)
            sp_acq.setValue(200)
            self.sp_qp_acq_ms.append(sp_acq)
            qp.addWidget(sp_acq, ch, 4)

            bs = QtWidgets.QSpinBox()
            bs.setRange(1, 500)
            bs.setValue(3)
            self.sp_base_s.append(bs)
            qp.addWidget(bs, ch, 5)

            be = QtWidgets.QSpinBox()
            be.setRange(2, 500)
            be.setValue(15)
            self.sp_base_e.append(be)
            qp.addWidget(be, ch, 6)

            avg = QtWidgets.QLabel("-")
            avg.setStyleSheet("font-weight:800; color:#dbe6ff;")
            self.lbl_qp_avg.append(avg)
            qp.addWidget(avg, ch, 7)

        qp_scroll = QtWidgets.QScrollArea()
        qp_scroll.setWidgetResizable(True)
        qp_scroll.setWidget(qp_group)
        qp_scroll.setMinimumHeight(160)
        qp_scroll.setMaximumHeight(220)
        qp_scroll.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.pnl_qpcr.content_l.addWidget(qp_scroll)

        opt = QtWidgets.QGridLayout()
        opt.setContentsMargins(0, 0, 0, 0)
        opt.setHorizontalSpacing(8)
        opt.setVerticalSpacing(6)

        self.sp_qp_cycles = QtWidgets.QSpinBox()
        self.sp_qp_cycles.setRange(1, 500)
        self.sp_qp_cycles.setValue(40)

        self.sp_qp_inter_gap_ms = QtWidgets.QSpinBox()
        self.sp_qp_inter_gap_ms.setRange(0, 10000)
        self.sp_qp_inter_gap_ms.setValue(50)

        self.cb_auto_thr = QtWidgets.QCheckBox("Auto threshold (mean + k*std in baseline)")
        self.cb_auto_thr.setChecked(True)

        self.sp_k_sigma = QtWidgets.QDoubleSpinBox()
        self.sp_k_sigma.setRange(1.0, 30.0)
        self.sp_k_sigma.setValue(10.0)
        self.sp_k_sigma.setSingleStep(0.5)

        self.cb_use_delta = QtWidgets.QCheckBox("Plot ΔR (baseline-subtracted)")
        self.cb_use_delta.setChecked(True)

        self.cb_manual_thr = QtWidgets.QCheckBox("Manual threshold")
        self.cb_manual_thr.setChecked(False)

        self.sp_thr_manual = QtWidgets.QDoubleSpinBox()
        self.sp_thr_manual.setRange(-1e9, 1e9)
        self.sp_thr_manual.setValue(100.0)
        self.sp_thr_manual.setDecimals(3)

        self.btn_recalc_ct = QtWidgets.QPushButton("Recalc Ct")
        self.btn_recalc_ct.clicked.connect(self._qpcr_recalc_all)

        opt.addWidget(QtWidgets.QLabel("Cycles (N)"), 0, 0)
        opt.addWidget(self.sp_qp_cycles, 0, 1)
        opt.addWidget(QtWidgets.QLabel("Inter-CH gap(ms)"), 0, 2)
        opt.addWidget(self.sp_qp_inter_gap_ms, 0, 3)

        opt.addWidget(self.cb_use_delta, 1, 0, 1, 2)
        opt.addWidget(self.cb_auto_thr, 1, 2, 1, 2)
        opt.addWidget(QtWidgets.QLabel("k="), 1, 4)
        opt.addWidget(self.sp_k_sigma, 1, 5)
        opt.addWidget(self.btn_recalc_ct, 1, 6)

        opt.addWidget(self.cb_manual_thr, 2, 2)
        opt.addWidget(QtWidgets.QLabel("THR="), 2, 3)
        opt.addWidget(self.sp_thr_manual, 2, 4)

        self.pnl_qpcr.content_l.addLayout(opt)

        # panel heights
        self.pnl_normal.setMaximumHeight(420)
        self.pnl_stream.setMaximumHeight(520)
        self.pnl_qpcr.setMaximumHeight(420)

        # =========================
        # Row 5: Realtime Graph + Log (MON + FLUO only)
        # =========================
        self.pnl_graph = InstrumentPanel("Graph (Realtime)", collapsible=False)
        root.addWidget(self.pnl_graph, 10)

        pg.setConfigOptions(antialias=True)

        self.p_mon = pg.PlotWidget(title="MON (mV) - 4CH")
        self.p_fluo = pg.PlotWidget(title="FLUO (mV) - 4CH")

        for p in (self.p_mon, self.p_fluo):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setXRange(0, self.win_points - 1, padding=0)
            p.addLegend()

        pens = [
            pg.mkPen((255, 0, 0), width=2),
            pg.mkPen((0, 128, 255), width=2),
            pg.mkPen((0, 180, 80), width=2),
            pg.mkPen((200, 120, 0), width=2),
        ]

        self.cur_mon = []
        self.cur_fluo = []
        for i in range(4):
            ch = i + 1
            self.cur_mon.append(self.p_mon.plot(self.x, list(self.buf_mon_mv[i]), pen=pens[i], name=f"CH{ch}"))
            self.cur_fluo.append(self.p_fluo.plot(self.x, list(self.buf_fluo_mv[i]), pen=pens[i], name=f"CH{ch}"))

        plot_layout = QtWidgets.QHBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(self.p_mon, 1)
        plot_layout.addWidget(self.p_fluo, 1)

        plot_container = QtWidgets.QWidget()
        plot_container.setLayout(plot_layout)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(8000)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(plot_container)
        split.addWidget(self.txt_log)
        split.setStretchFactor(0, 8)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 220])

        self.pnl_graph.content_l.addWidget(split, 1)

    # ---------------- excel plot window ----------------
    def _open_excel_plot_window(self):
        if self.excel_win is None:
            self.excel_win = ExcelFluoPlotWindow(self._log)
            self._extra_windows.append(self.excel_win)
        self.excel_win.show()
        self.excel_win.raise_()
        self.excel_win.activateWindow()

    # ---------------- serial ops ----------------
    def _refresh_ports(self):
        self.cb_port.clear()
        for p in list_ports():
            self.cb_port.addItem(p)

    def _toggle_connect(self):
        if self.worker and self.worker.isRunning():
            self._stop_all_runs()
            self.cmd_sched.clear()
            self.worker.stop()
            self.worker.wait(700)
            self.worker = None
            self.btn_connect.setText("Connect")
            self._set_lamp("err")
            self._log("Disconnected by user")
            return

        port = self.cb_port.currentText().strip()
        if not port:
            self._log("No port selected")
            return

        self.worker = SerialWorker(port, 115200)
        self.worker.line_received.connect(self._on_line)
        self.worker.status_changed.connect(self._on_status)
        self.worker.start()
        self.btn_connect.setText("Disconnect")

    def _set_lamp(self, st: str):
        self.lbl_status_lamp.setProperty("state", st)
        self.lbl_status_lamp.style().unpolish(self.lbl_status_lamp)
        self.lbl_status_lamp.style().polish(self.lbl_status_lamp)
        self.lbl_status_lamp.update()

    def _on_status(self, ok: bool, msg: str):
        self.lbl_status.setText(msg)
        self._log(msg)
        self._set_lamp("on" if ok else "err")
        if ok:
            self._queue("STREAM 1")
            self._queue(f"SET PERIOD {self.sp_period.value()}")
            self._queue(f"SET DELAYON {self.sp_delayon.value()}")
            self._queue(f"SET AVG {self.sp_avg.value()}")
            self._queue(f"SET DUMMY {self.sp_dummy.value()}")
            self._queue(f"SET DARK {1 if self.cb_dark.isChecked() else 0}")
            self._queue("LEDMASK 15")
            self._queue("LEDS 15 0")

    def _send_immediate(self, s: str):
        if self.worker and self.worker.isRunning():
            self.worker.send_line(s)
            self._log(f"TX: {s}")

    def _queue(self, s: str):
        if self.worker and self.worker.isRunning():
            self.cmd_sched.push(s)
            self._log(f"Q: {s}")

    # ---------------- mode handling ----------------
    def _on_mode_changed(self, m: str):
        if self.record_running or self.qpcr_running or self.meas_tasks:
            self._stop_all_runs()

        self.mode = m
        self.lbl_run.setText(f"MODE={m} / IDLE")

        if m == self.MODE_NORMAL:
            self.pnl_normal.setEnabled(True)
            self.pnl_stream.setEnabled(False)
            self.pnl_qpcr.setEnabled(False)

            self.pnl_normal.set_collapsed(False)
            self.pnl_stream.set_collapsed(True)
            self.pnl_qpcr.set_collapsed(True)

            self.btn_record.setEnabled(False)
            self.btn_qpcr.setEnabled(False)
            for b in self.btn_led:
                b.setEnabled(True)
            for b in self.btn_measure:
                b.setEnabled(True)

        elif m == self.MODE_RECORD:
            self.pnl_normal.setEnabled(False)
            self.pnl_stream.setEnabled(True)
            self.pnl_qpcr.setEnabled(False)

            self.pnl_normal.set_collapsed(True)
            self.pnl_stream.set_collapsed(False)
            self.pnl_qpcr.set_collapsed(True)

            self.btn_record.setEnabled(True)
            self.btn_qpcr.setEnabled(False)
            for b in self.btn_led:
                b.setEnabled(False)
            for b in self.btn_measure:
                b.setEnabled(False)

        elif m == self.MODE_QPCR:
            self.pnl_normal.setEnabled(False)
            self.pnl_stream.setEnabled(False)
            self.pnl_qpcr.setEnabled(True)

            self.pnl_normal.set_collapsed(True)
            self.pnl_stream.set_collapsed(True)
            self.pnl_qpcr.set_collapsed(False)

            self.btn_record.setEnabled(False)
            self.btn_qpcr.setEnabled(True)
            for b in self.btn_led:
                b.setEnabled(False)
            for b in self.btn_measure:
                b.setEnabled(False)

    def _stop_all_runs(self):
        if self.record_running:
            self.btn_record.setChecked(False)
            self._toggle_record()
        if self.qpcr_running:
            self.btn_qpcr.setChecked(False)
            self._toggle_qpcr()
        self._normal_measure_stop_all()

    # ---------------- commands (NORMAL manual) ----------------
    def _cmd_led(self, ch: int, on: bool):
        self.btn_led[ch - 1].setText("ON" if on else "OFF")
        self._queue(f"LED {ch} {1 if on else 0}")

    def _cmd_vexten(self, ch: int, on: bool):
        self.btn_vexten[ch - 1].setText("ON" if on else "OFF")
        self._queue(f"VEXTEN {ch} {1 if on else 0}")

    def _cmd_vext(self, ch: int, mv: int):
        mv = max(0, min(1800, int(mv)))
        self._queue(f"VEXT {ch} {mv}")

    def _toggle_stream(self):
        on = self.btn_stream.isChecked()
        self.btn_stream.setText("STREAM ON" if on else "STREAM OFF")
        self._queue(f"STREAM {1 if on else 0}")

    # ---------------- NORMAL: Measure (concurrent) ----------------
    def _normal_measure_start(self, ch: int):
        if self.mode != self.MODE_NORMAL:
            self._log("MEASURE: only allowed in NORMAL mode")
            return
        if not (self.worker and self.worker.isRunning()):
            self._log("MEASURE: Not connected.")
            return
        if ch in self.meas_tasks:
            self._log(f"MEASURE: CH{ch} already running")
            return

        delay_ms = int(self.sp_meas_delay_ms.value())
        window_ms = int(self.sp_meas_window_ms.value())

        prev_led = None
        fr = self.last_frame.get(ch)
        if fr is not None:
            prev_led = int(fr.led_en)

        self.meas_tasks[ch] = NormalMeasureTask(
            ch=ch,
            stage="DELAY",
            stage_t0=time.time(),
            delay_s=delay_ms / 1000.0,
            window_s=window_ms / 1000.0,
            acc_mon=[],
            acc_fluo=[],
            prev_led_on=prev_led,
        )

        self.btn_meas_stop.setEnabled(True)
        self.btn_measure[ch - 1].setEnabled(False)
        self.btn_measure[ch - 1].setText("MEASURING...")

        self._log(f"MEASURE START: CH{ch} (delay={delay_ms}ms, window={window_ms}ms)")
        self._queue(f"LED {ch} 1")

    def _normal_measure_stop_one(self, ch: int, reason: str = ""):
        task = self.meas_tasks.get(ch)
        if not task:
            return

        if task.prev_led_on is not None:
            self._queue(f"LED {ch} {1 if task.prev_led_on else 0}")
        else:
            self._queue(f"LED {ch} 0")

        self.btn_measure[ch - 1].setEnabled(True)
        self.btn_measure[ch - 1].setText("MEASURE")

        del self.meas_tasks[ch]
        self._log(f"MEASURE STOP: CH{ch}" + (f" ({reason})" if reason else ""))

        if not self.meas_tasks:
            self.btn_meas_stop.setEnabled(False)

    def _normal_measure_stop_all(self):
        if not self.meas_tasks:
            self.btn_meas_stop.setEnabled(False)
            return
        chs = sorted(list(self.meas_tasks.keys()))
        for ch in chs:
            self._normal_measure_stop_one(ch, reason="stop all")
        self.btn_meas_stop.setEnabled(False)

    def _normal_measure_tick(self):
        if not self.meas_tasks:
            return

        now = time.time()

        for ch in sorted(list(self.meas_tasks.keys())):
            task = self.meas_tasks.get(ch)
            if not task:
                continue

            if task.stage == "DELAY":
                if (now - task.stage_t0) >= task.delay_s:
                    task.stage = "ACQ"
                    task.stage_t0 = now
                    self._log(f"MEASURE ACQ: CH{ch} collecting READY frames...")
                continue

            if task.stage == "ACQ":
                fr = self.last_frame.get(ch)
                if fr and fr.ready == 1:
                    task.acc_mon.append(int(fr.mon_mv))
                    task.acc_fluo.append(int(fr.fluo_mv))

                if (now - task.stage_t0) >= task.window_s:
                    n = len(task.acc_mon)
                    mon_avg = sum(task.acc_mon) / n if n else 0.0
                    fluo_avg = sum(task.acc_fluo) / n if n else 0.0

                    self.lbl_mon_avg[ch - 1].setText(f"{mon_avg:.1f}")
                    self.lbl_fluo_avg[ch - 1].setText(f"{fluo_avg:.1f}")

                    self.normal_rows.append({
                        "host_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ch": ch,
                        "delay_ms": int(self.sp_meas_delay_ms.value()),
                        "window_ms": int(self.sp_meas_window_ms.value()),
                        "samples": n,
                        "mon_avg_mv": float(mon_avg),
                        "fluo_avg_mv": float(fluo_avg),
                    })

                    self._log(f"MEASURE DONE: CH{ch} samples={n}, MON_AVG={mon_avg:.1f}mV, FLUO_AVG={fluo_avg:.1f}mV")
                    self._normal_measure_stop_one(ch, reason="done")

    # ---------------- RECORD ----------------
    def _on_capture_hz_changed(self, v: int):
        self.capture_hz = int(v)
        self._log(f"RECORD capture set: {self.capture_hz} shots/sec per channel")

    def _read_ui_schedules(self) -> Tuple[List[int], float]:
        enabled_ch = []
        max_end = 0.0

        for i in range(4):
            sch = self.led_sched[i]
            sch.enabled = self.cb_led_en[i].isChecked()
            sch.delay_s = int(self.sp_delay_s[i].value())
            sch.on_s = int(self.sp_on_s[i].value())
            sch.period_s = int(self.sp_period_s[i].value())
            sch.duration_s = int(self.sp_dur_s[i].value())

            if sch.period_s <= 0:
                sch.period_s = 1
            if sch.on_s <= 0:
                sch.on_s = 1
            if sch.duration_s <= 0:
                sch.duration_s = 1

            if sch.enabled:
                enabled_ch.append(i + 1)
                end = self.record_t0 + sch.delay_s + sch.duration_s
                if end > max_end:
                    max_end = end

        return enabled_ch, max_end

    def _record_log_summary(self):
        for ch in range(1, 5):
            idx = ch - 1
            n = self.record_acc_n[idx]
            if n <= 0:
                continue
            mon_avg = self.record_acc_mon[idx] / n
            fluo_avg = self.record_acc_fluo[idx] / n
            self._log(f"RECORD SUMMARY CH{ch}: samples={n}, MON_AVG={mon_avg:.1f}mV, FLUO_AVG={fluo_avg:.1f}mV")

    def _toggle_record(self):
        if self.mode != self.MODE_RECORD:
            self.btn_record.setChecked(False)
            return

        want_on = self.btn_record.isChecked()

        if want_on:
            if not (self.worker and self.worker.isRunning()):
                self._log("RECORD: Not connected.")
                self.btn_record.setChecked(False)
                return

            if not self.btn_stream.isChecked():
                self.btn_stream.setChecked(True)
                self._toggle_stream()

            self.record_rows.clear()
            self.record_t0 = time.time()

            # ✅ reset averages
            self.record_acc_n = [0, 0, 0, 0]
            self.record_acc_mon = [0.0, 0.0, 0.0, 0.0]
            self.record_acc_fluo = [0.0, 0.0, 0.0, 0.0]
            for i in range(4):
                if self.lbl_rec_n:
                    self.lbl_rec_n[i].setText("-")
                    self.lbl_rec_mon_avg[i].setText("-")
                    self.lbl_rec_fluo_avg[i].setText("-")

            enabled_ch, max_end = self._read_ui_schedules()
            if not enabled_ch:
                self._log("RECORD: No LED enabled.")
                self.btn_record.setChecked(False)
                return

            self.capture_hz = int(self.sp_capture_hz.value())
            self._last_save_epoch = [0.0, 0.0, 0.0, 0.0]

            self.record_end_epoch = max_end
            self.record_running = True
            self._last_sent_led_mask = 0

            self._queue(f"LEDMASK {channels_to_mask(enabled_ch)}")
            self._queue("LEDS 15 0")

            self.btn_record.setText("RECORD STOP")
            self.lbl_run.setText(f"MODE=RECORD / RUN enabled={enabled_ch}")
            self._log(f"RECORD START enabled={enabled_ch}")
            self.logic_timer.start(50)

        else:
            self.logic_timer.stop()
            self._queue("LEDS 15 0")
            self._queue("LEDMASK 15")
            self.record_running = False

            self.btn_record.setText("RECORD START")
            self.lbl_run.setText("MODE=RECORD / IDLE")
            self._record_log_summary()
            self._log(f"RECORD STOP (rows={len(self.record_rows)})")

    def _should_led_on_now(self, ch: int, now: float) -> bool:
        idx = ch - 1
        sch = self.led_sched[idx]
        if not sch.enabled:
            return False

        win_start = self.record_t0 + sch.delay_s
        win_end = win_start + sch.duration_s

        if now < win_start or now >= win_end:
            return False

        if sch.on_s >= sch.period_s:
            return True

        elapsed = now - win_start
        phase = elapsed % float(sch.period_s)
        return phase < float(sch.on_s)

    def _logic_tick(self):
        if not self.record_running:
            return

        now = time.time()
        if now >= self.record_end_epoch:
            self._log("RECORD: End time reached.")
            self.btn_record.setChecked(False)
            self._toggle_record()
            return

        mask = 0
        for ch in range(1, 5):
            if self._should_led_on_now(ch, now):
                mask |= (1 << (ch - 1))

        if mask != self._last_sent_led_mask:
            self._last_sent_led_mask = mask
            self._queue("LEDS 15 0")
            if mask != 0:
                QtCore.QTimer.singleShot(40, lambda m=mask: self._queue(f"LEDS {m} 1"))

    def _should_record_frame(self, fr: Frame) -> bool:
        if not self.record_running:
            return False
        if fr.ready != 1:
            return False
        now = time.time()
        if not self._should_led_on_now(fr.ch, now):
            return False

        hz = max(1, int(self.capture_hz))
        min_dt = 1.0 / float(hz)
        idx = fr.ch - 1
        if (now - self._last_save_epoch[idx]) < min_dt:
            return False
        self._last_save_epoch[idx] = now
        return True

    # ---------------- QPCR ----------------
    def _qpcr_build_plot_window(self):
        if self.qpcr_plot_win is not None:
            try:
                self.qpcr_plot_win.close()
            except Exception:
                pass
            self.qpcr_plot_win = None

        w = QtWidgets.QWidget()
        w.setWindowTitle("QPCR Amplification: ΔR + Threshold + Ct (saved to Excel)")
        w.resize(1250, 760)
        layout = QtWidgets.QVBoxLayout(w)

        self.qpcr_plot = pg.PlotWidget(title="Amplification Plot (x fixed to Cycle N)")
        self.qpcr_plot.showGrid(x=True, y=True, alpha=0.3)
        self.qpcr_plot.addLegend()
        self.qpcr_plot.setLabel("bottom", "Cycle")
        self.qpcr_plot.setLabel("left", "Signal (mV, relative)")

        layout.addWidget(self.qpcr_plot, 1)

        pens = [
            pg.mkPen((255, 0, 0), width=2),
            pg.mkPen((0, 128, 255), width=2),
            pg.mkPen((0, 180, 80), width=2),
            pg.mkPen((200, 120, 0), width=2),
        ]

        self.qpcr_plot_curve.clear()
        self.qpcr_thr_line.clear()

        for ch in range(1, 5):
            curve = self.qpcr_plot.plot([], [], pen=pens[ch - 1], name=f"CH{ch}")
            self.qpcr_plot_curve[ch] = curve
            thr = pg.InfiniteLine(angle=0, movable=False)
            thr.setVisible(False)
            self.qpcr_plot.addItem(thr)
            self.qpcr_thr_line[ch] = thr

        self.qpcr_plot_win = w
        self._extra_windows.append(w)
        w.show()

    def _qpcr_plot_update(self):
        if not self.qpcr_plot:
            return
        use_delta = self.cb_use_delta.isChecked()
        N = int(self.sp_qp_cycles.value())

        self.qpcr_plot.setXRange(1, N, padding=0.0)
        x_full = list(range(1, N + 1))

        for ch in range(1, 5):
            if ch not in self.qpcr_active_ch_list:
                self.qpcr_plot_curve[ch].setData([], [])
                self.qpcr_thr_line[ch].setVisible(False)
                continue

            y_src = self.qpcr_delta[ch] if (use_delta and self.qpcr_delta[ch]) else self.qpcr_R[ch]
            y = [float("nan")] * N
            for i, v in enumerate(y_src[:N]):
                y[i] = float(v)

            self.qpcr_plot_curve[ch].setData(x_full, y)

            thr = self.qpcr_threshold.get(ch, None)
            if thr is not None:
                self.qpcr_thr_line[ch].setValue(float(thr))
                self.qpcr_thr_line[ch].setVisible(True)

    def _qpcr_set_state(self, st: str):
        self.qpcr_state = st
        self.qpcr_state_t = time.time()

        if not self.qpcr_running:
            return

        cycles = int(self.sp_qp_cycles.value())
        if self.qpcr_cycle_idx >= cycles:
            return

        ch = self.qpcr_active_ch_list[self.qpcr_ch_ptr]
        bit = 1 << (ch - 1)

        if st == "ENTER_CH":
            self._queue(f"LEDMASK {bit}")
            self._queue("SET DARK 1")
            self._queue("LEDS 15 0")
            self.qpcr_dark_samples[ch] = []
            self.qpcr_light_samples[ch] = []
            self.qpcr_mon_samples[ch] = []
            return

        if st == "LED_ON_WAIT":
            self._queue("SET DARK 0")
            self._queue("LEDS 15 0")
            QtCore.QTimer.singleShot(40, lambda b=bit: self._queue(f"LEDS {b} 1"))
            return

        if st == "LED_OFF_WAIT":
            self._queue(f"LEDS {bit} 0")
            return

    def _toggle_qpcr(self):
        if self.mode != self.MODE_QPCR:
            self.btn_qpcr.setChecked(False)
            return

        want_on = self.btn_qpcr.isChecked()

        if want_on:
            if not (self.worker and self.worker.isRunning()):
                self._log("QPCR: Not connected.")
                self.btn_qpcr.setChecked(False)
                return

            if not self.btn_stream.isChecked():
                self.btn_stream.setChecked(True)
                self._toggle_stream()

            self.qpcr_rows.clear()
            self.qpcr_running = True
            self.qpcr_t0 = time.time()
            self.qpcr_cycle_idx = 0
            self.qpcr_ch_ptr = 0

            self.qpcr_active_ch_list = [i + 1 for i in range(4) if self.cb_qp_en[i].isChecked()]
            if not self.qpcr_active_ch_list:
                self._log("QPCR: No channel enabled.")
                self.btn_qpcr.setChecked(False)
                self.qpcr_running = False
                return

            for ch in range(1, 5):
                self.qpcr_R[ch] = []
                self.qpcr_delta[ch] = []
                self.qpcr_ct[ch] = None
                self.qpcr_threshold[ch] = 0.0
                self.lbl_qp_avg[ch - 1].setText("-")

            self._queue("LEDS 15 0")

            self.btn_qpcr.setText("QPCR STOP")
            self.lbl_run.setText(f"MODE=QPCR / RUN ch={self.qpcr_active_ch_list}")
            self._log(f"QPCR START ch={self.qpcr_active_ch_list}, cycles={self.sp_qp_cycles.value()}")

            self._qpcr_build_plot_window()
            self._qpcr_set_state("ENTER_CH")
            self.qpcr_timer.start(20)

        else:
            self.qpcr_timer.stop()
            self._queue("LEDS 15 0")
            self._queue("LEDMASK 15")
            self._queue("SET DARK 1")
            self.qpcr_running = False
            self.btn_qpcr.setText("QPCR START")
            self.lbl_run.setText("MODE=QPCR / IDLE")
            self._log(f"QPCR STOP (rows={len(self.qpcr_rows)})")

    def _qpcr_recalc_one(self, ch: int):
        R = self.qpcr_R[ch]
        if len(R) < 3:
            return

        bs = int(self.sp_base_s[ch - 1].value())
        be = int(self.sp_base_e[ch - 1].value())
        fit = baseline_fit_linear(R, bs, be)
        if fit is None:
            return
        self.qpcr_delta[ch] = list(fit.delta)

        if self.cb_manual_thr.isChecked():
            thr = float(self.sp_thr_manual.value())
        else:
            k = float(self.sp_k_sigma.value())
            thr = auto_threshold_from_baseline(self.qpcr_delta[ch], bs, be, k)
        self.qpcr_threshold[ch] = thr

        ct = calc_ct(self.qpcr_delta[ch], thr)
        self.qpcr_ct[ch] = ct

    def _qpcr_recalc_all(self):
        for ch in self.qpcr_active_ch_list:
            self._qpcr_recalc_one(ch)
        self._qpcr_plot_update()

    def _qpcr_tick(self):
        if not self.qpcr_running:
            return

        cycles = int(self.sp_qp_cycles.value())
        gap_ms = int(self.sp_qp_inter_gap_ms.value())

        if self.qpcr_cycle_idx >= cycles:
            self._log("QPCR: cycles done.")
            self.btn_qpcr.setChecked(False)
            self._toggle_qpcr()
            return

        ch = self.qpcr_active_ch_list[self.qpcr_ch_ptr]
        off_ms = int(self.sp_qp_off_ms[ch - 1].value())
        on_ms = int(self.sp_qp_on_ms[ch - 1].value())
        acq_ms = min(int(self.sp_qp_acq_ms[ch - 1].value()), on_ms)

        now = time.time()
        t_ms = (now - self.qpcr_state_t) * 1000.0

        if self.qpcr_state == "ENTER_CH":
            self.qpcr_state = "DARK_WAIT"
            self.qpcr_state_t = time.time()
            return

        if self.qpcr_state == "DARK_WAIT":
            fr = self.last_frame.get(ch)
            if fr:
                self.qpcr_dark_samples[ch].append(float(fr.fluo_mv))
            if t_ms >= max(0, off_ms):
                self._qpcr_set_state("LED_ON_WAIT")
            return

        if self.qpcr_state == "LED_ON_WAIT":
            wait_before_acq = max(0, on_ms - acq_ms)
            if t_ms >= wait_before_acq:
                self.qpcr_state = "ACQ_WAIT"
                self.qpcr_state_t = time.time()
            return

        if self.qpcr_state == "ACQ_WAIT":
            fr = self.last_frame.get(ch)
            if fr and fr.ready == 1:
                self.qpcr_light_samples[ch].append(float(fr.fluo_mv))
                self.qpcr_mon_samples[ch].append(float(fr.mon_mv))

            if t_ms >= max(0, acq_ms):
                dark_mean = sum(self.qpcr_dark_samples[ch]) / max(1, len(self.qpcr_dark_samples[ch]))
                light_mean = sum(self.qpcr_light_samples[ch]) / max(1, len(self.qpcr_light_samples[ch])) if self.qpcr_light_samples[ch] else 0.0
                mon_mean = sum(self.qpcr_mon_samples[ch]) / max(1, len(self.qpcr_mon_samples[ch])) if self.qpcr_mon_samples[ch] else 0.0
                R = float(light_mean - dark_mean)

                # ✅ UI: averages only
                self.lbl_qp_avg[ch - 1].setText(f"D{dark_mean:.1f} L{light_mean:.1f} R{R:.1f} M{mon_mean:.1f}")

                self.qpcr_R[ch].append(R)
                self._qpcr_recalc_one(ch)

                y_plot = self.qpcr_delta[ch][-1] if self.qpcr_delta[ch] else R
                self.qpcr_rows.append({
                    "host_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "t_rel_s": time.time() - self.qpcr_t0,
                    "cycle": len(self.qpcr_R[ch]),
                    "ch": ch,
                    "dark_fluo_mean_mv": float(dark_mean),
                    "light_fluo_mean_mv": float(light_mean),
                    "R_light_minus_dark": float(R),
                    "delta": float(y_plot),
                    "threshold": float(self.qpcr_threshold.get(ch, 0.0)),
                    "ct": ("" if self.qpcr_ct[ch] is None else float(self.qpcr_ct[ch])),
                    "mon_mean_mv": float(mon_mean),
                    "off_ms": off_ms,
                    "on_ms": on_ms,
                    "acq_ms": acq_ms,
                    "dark_samples": len(self.qpcr_dark_samples[ch]),
                    "light_samples": len(self.qpcr_light_samples[ch]),
                })

                self._qpcr_plot_update()
                self._qpcr_set_state("LED_OFF_WAIT")
            return

        if self.qpcr_state == "LED_OFF_WAIT":
            if t_ms >= max(0, gap_ms):
                self.qpcr_ch_ptr += 1
                if self.qpcr_ch_ptr >= len(self.qpcr_active_ch_list):
                    self.qpcr_ch_ptr = 0
                    self.qpcr_cycle_idx += 1
                    self._log(f"QPCR cycle {self.qpcr_cycle_idx}/{cycles} done.")

                if self.qpcr_cycle_idx >= cycles:
                    self._log("QPCR: cycles done.")
                    self.btn_qpcr.setChecked(False)
                    self._toggle_qpcr()
                    return

                self._qpcr_set_state("ENTER_CH")
            return

    # ---------------- RX parsing ----------------
    def _on_line(self, s: str):
        if s.startswith("S,CH,"):
            self._handle_stream_line(s)
        else:
            self._log(f"RX: {s}")

    def _handle_stream_line(self, s: str):
        m = LINE_RE.match(s)
        if not m:
            return

        ch = int(m.group(1))
        kv = parse_kv_payload(m.group(2))

        fr = Frame(
            t_host=time.time(),
            ch=ch,
            led_en=safe_int(kv, "LED_EN"),
            vpulse_mv=safe_int(kv, "VPULSE_MV"),
            ready=safe_int(kv, "READY"),
            ton_ms=safe_int(kv, "TON_MS"),
            vext_en=safe_int(kv, "VEXT_EN"),
            vext_mv=safe_int(kv, "VEXT_MV"),
            mon_mv=safe_int(kv, "MON_MV"),
            fluo_mv=safe_int(kv, "FLUO_MV"),
            mon_raw=safe_int(kv, "MON_RAW"),
            fluo_raw=safe_int(kv, "FLUO_RAW"),
            tick_ms=safe_int(kv, "TICK_MS"),
        )
        self.last_frame[ch] = fr

        i = ch - 1
        if 0 <= i < 4:
            self.buf_vpulse[i].append(fr.vpulse_mv)
            self.buf_mon_mv[i].append(fr.mon_mv)
            self.buf_fluo_mv[i].append(fr.fluo_mv)

            self.lbl_vpulse[i].setText(str(fr.vpulse_mv))
            self.lbl_mon_mv[i].setText(str(fr.mon_mv))
            self.lbl_fluo_mv[i].setText(str(fr.fluo_mv))
            self.lbl_mon_raw[i].setText(str(fr.mon_raw))
            self.lbl_fluo_raw[i].setText(str(fr.fluo_raw))
            self.lbl_ready[i].setText("1" if fr.ready else "0")

        if self.mode == self.MODE_RECORD and self._should_record_frame(fr):
            sch = self.led_sched[fr.ch - 1]
            now_epoch = time.time()
            self.record_rows.append({
                "host_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "record_t_rel_s": now_epoch - self.record_t0,
                "ch": fr.ch,
                "led_en": fr.led_en,
                "vpulse_mv": fr.vpulse_mv,
                "ready": fr.ready,
                "ton_ms": fr.ton_ms,
                "vext_en": fr.vext_en,
                "vext_mv": fr.vext_mv,
                "mon_mv": fr.mon_mv,
                "fluo_mv": fr.fluo_mv,
                "mon_raw": fr.mon_raw,
                "fluo_raw": fr.fluo_raw,
                "tick_ms": fr.tick_ms,
                "sched_delay_s": sch.delay_s,
                "sched_on_s": sch.on_s,
                "sched_period_s": sch.period_s,
                "sched_duration_s": sch.duration_s,
            })

            # ✅ running average update (display averages only)
            idx = fr.ch - 1
            self.record_acc_n[idx] += 1
            self.record_acc_mon[idx] += float(fr.mon_mv)
            self.record_acc_fluo[idx] += float(fr.fluo_mv)
            n = self.record_acc_n[idx]
            if n > 0 and self.lbl_rec_n:
                mon_avg = self.record_acc_mon[idx] / n
                fluo_avg = self.record_acc_fluo[idx] / n
                self.lbl_rec_n[idx].setText(str(n))
                self.lbl_rec_mon_avg[idx].setText(f"{mon_avg:.1f}")
                self.lbl_rec_fluo_avg[idx].setText(f"{fluo_avg:.1f}")

    # ---------------- Excel save ----------------
    def _save_excel(self):
        sheets = {}
        if self.normal_rows:
            sheets["NORMAL_MEASURE"] = pd.DataFrame(self.normal_rows)
        if self.record_rows:
            sheets["RECORD"] = pd.DataFrame(self.record_rows)
        if self.qpcr_rows:
            sheets["QPCR"] = pd.DataFrame(self.qpcr_rows)

        if not sheets:
            self._log("No data to save.")
            return

        default_name = f"light2_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Excel", default_name, "Excel (*.xlsx)"
        )
        if not path:
            return

        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name[:31], index=False)

        self._log(f"Saved: {path}")

        # open Excel plot window and auto load
        self._open_excel_plot_window()
        if self.excel_win is not None:
            self.excel_win.load_excel(path)

    # ---------------- plot refresh (realtime) ----------------
    def _refresh_plots(self):
        for i in range(4):
            self.cur_mon[i].setData(self.x, list(self.buf_mon_mv[i]))
            self.cur_fluo[i].setData(self.x, list(self.buf_fluo_mv[i]))

        if self.qpcr_running:
            self._qpcr_plot_update()

    # ---------------- log ----------------
    def _log(self, s: str):
        self.txt_log.appendPlainText(f"[{now_str()}] {s}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

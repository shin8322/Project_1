import sys
import os
import json
import time
import re
import threading
from dataclasses import dataclass
from typing import Optional, List

import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtWidgets


# -----------------------------
# Utils
# -----------------------------
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def ts() -> str:
    return time.strftime("%H:%M:%S")

def list_ports():
    return list(serial.tools.list_ports.comports())

def extract_numbers(text: str) -> List[str]:
    return NUM_RE.findall(text)

def app_state_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".yeonjung_serial_tool")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "state_numeric_log.json")

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def is_int_string(s: str) -> bool:
    try:
        int(s.strip())
        return True
    except Exception:
        return False


# -----------------------------
# Persistence
# -----------------------------
@dataclass
class AppState:
    last_port: str = ""
    last_baud: int = 115200
    last_databits: int = 8
    last_stopbits: int = 1
    last_parity: str = "None"
    use_crlf: bool = False
    send_rows: List[dict] = None
    history: List[str] = None
    airrep_count: int = 1

    def to_json(self) -> dict:
        return {
            "last_port": self.last_port,
            "last_baud": self.last_baud,
            "last_databits": self.last_databits,
            "last_stopbits": self.last_stopbits,
            "last_parity": self.last_parity,
            "use_crlf": self.use_crlf,
            "send_rows": self.send_rows or [],
            "history": self.history or [],
            "airrep_count": int(self.airrep_count),
        }

    @staticmethod
    def from_json(d: dict) -> "AppState":
        s = AppState()
        s.last_port = d.get("last_port", "")
        s.last_baud = int(d.get("last_baud", 115200))
        s.last_databits = int(d.get("last_databits", 8))
        s.last_stopbits = int(d.get("last_stopbits", 1))
        s.last_parity = d.get("last_parity", "None")
        s.use_crlf = bool(d.get("use_crlf", False))
        s.send_rows = d.get("send_rows", [])
        s.history = d.get("history", [])
        s.airrep_count = int(d.get("airrep_count", 1))
        return s

def load_state() -> AppState:
    p = app_state_path()
    if not os.path.exists(p):
        return AppState(send_rows=[], history=[], airrep_count=1)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return AppState.from_json(json.load(f))
    except Exception:
        return AppState(send_rows=[], history=[], airrep_count=1)

def save_state(st: AppState):
    p = app_state_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st.to_json(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -----------------------------
# Serial worker (QThread)
# -----------------------------
class SerialWorker(QtCore.QThread):
    rx_raw = QtCore.pyqtSignal(bytes)
    rx_line = QtCore.pyqtSignal(str)
    tx_done = QtCore.pyqtSignal(bytes)
    status = QtCore.pyqtSignal(bool, str)

    def __init__(self, port: str, baud: int, databits: int, stopbits: int, parity: str, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.databits = databits
        self.stopbits = stopbits
        self.parity = parity

        self._ser: Optional[serial.Serial] = None
        self._running = True
        self._rx_buf = bytearray()
        self._tx_lock = threading.Lock()
        self._tx_queue: List[bytes] = []

    def run(self):
        try:
            bytesize = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}[self.databits]
            stop = serial.STOPBITS_ONE if self.stopbits == 1 else serial.STOPBITS_TWO
            par = {"None": serial.PARITY_NONE, "Even": serial.PARITY_EVEN, "Odd": serial.PARITY_ODD}[self.parity]

            self._ser = serial.Serial(
                self.port, self.baud,
                bytesize=bytesize, stopbits=stop, parity=par,
                timeout=0.05, write_timeout=1.0,
                rtscts=False, dsrdtr=False
            )
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self.status.emit(True, f"Open: {self.port} @ {self.baud}")
        except Exception as e:
            self.status.emit(False, f"Open failed: {e}")
            return

        while self._running:
            try:
                data = self._ser.read(512)
                if data:
                    self.rx_raw.emit(data)
                    self._rx_buf.extend(data)
                    self._drain_lines()

                payload = None
                with self._tx_lock:
                    if self._tx_queue:
                        payload = self._tx_queue.pop(0)
                if payload:
                    self._ser.write(payload)
                    self._ser.flush()
                    self.tx_done.emit(payload)

            except Exception as e:
                self.status.emit(False, f"Serial error: {e}")
                break

        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self.status.emit(False, "Closed")

    def stop(self):
        self._running = False

    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    def set_rts(self, on: bool):
        if self._ser:
            try:
                self._ser.rts = on
            except Exception:
                pass

    def set_dtr(self, on: bool):
        if self._ser:
            try:
                self._ser.dtr = on
            except Exception:
                pass

    @QtCore.pyqtSlot(bytes)
    def enqueue_write(self, payload: bytes):
        if not self._ser or not self._ser.is_open:
            return
        with self._tx_lock:
            self._tx_queue.append(payload)

    def _drain_lines(self):
        while True:
            idx_n = self._rx_buf.find(b"\n")
            idx_r = self._rx_buf.find(b"\r")
            candidates = [i for i in (idx_n, idx_r) if i != -1]
            if not candidates:
                break
            cut = min(candidates)
            raw = self._rx_buf[:cut]
            del self._rx_buf[:cut + 1]
            line = raw.decode("utf-8", errors="ignore")
            if line != "":
                self.rx_line.emit(line)


# -----------------------------
# UI elements
# -----------------------------
class SendRow(QtWidgets.QWidget):
    send_clicked = QtCore.pyqtSignal(int)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.idx = idx
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)

        self.chk = QtWidgets.QCheckBox()
        self.edit = QtWidgets.QLineEdit()
        self.btn = QtWidgets.QPushButton("Send")

        self.edit.setPlaceholderText("ASCII command/value (e.g., SM CH, 1500, done, AR, SET, AIRREP 3)")
        self.btn.clicked.connect(lambda: self.send_clicked.emit(self.idx))

        h.addWidget(self.chk)
        h.addWidget(self.edit, 1)
        h.addWidget(self.btn)

    def to_dict(self) -> dict:
        return {"checked": self.chk.isChecked(), "text": self.edit.text()}

    def from_dict(self, d: dict):
        self.chk.setChecked(bool(d.get("checked", False)))
        self.edit.setText(d.get("text", ""))


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    NUM_ROWS = 9

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yeonjung Serial Tool - Servo Continuous + AR/SET/STM")

        self.worker: Optional[SerialWorker] = None
        self.state = load_state()

        self._run_running = False
        self._run_thread = None

        self._history = list(self.state.history or [])
        self._hist_idx = len(self._history)

        # Mode flags (for logging/UX)
        self.servo_mode_active = False

        # SET 이후 숫자(반복횟수) 1개를 받는 펌웨어 흐름 지원
        self.awaiting_rep_after_set = False

        # Servo mapping defaults (270-deg servo assumption)
        self.servo_min_us = 500
        self.servo_max_us = 2500
        self.servo_range_deg = 270.0

        # Slider send rate limit
        self._servo_pending_send = False
        self._servo_send_timer = QtCore.QTimer(self)
        self._servo_send_timer.setSingleShot(True)
        self._servo_send_timer.timeout.connect(self._servo_send_if_pending)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left: Port controls
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        port_box = QtWidgets.QGroupBox("Port")
        pg = QtWidgets.QGridLayout(port_box)

        self.cmb_port = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")

        self.cmb_baud = QtWidgets.QComboBox()
        self.cmb_baud.addItems(["115200", "57600", "38400", "19200", "9600"])
        self.cmb_baud.setCurrentText(str(self.state.last_baud or 115200))

        self.cmb_data = QtWidgets.QComboBox()
        self.cmb_data.addItems(["8", "7", "6", "5"])
        self.cmb_data.setCurrentText(str(self.state.last_databits or 8))

        self.cmb_stop = QtWidgets.QComboBox()
        self.cmb_stop.addItems(["1", "2"])
        self.cmb_stop.setCurrentText(str(self.state.last_stopbits or 1))

        self.cmb_parity = QtWidgets.QComboBox()
        self.cmb_parity.addItems(["None", "Even", "Odd"])
        self.cmb_parity.setCurrentText(self.state.last_parity or "None")

        self.btn_open = QtWidgets.QPushButton("Open")
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setEnabled(False)

        self.chk_rts = QtWidgets.QCheckBox("RTS")
        self.chk_dtr = QtWidgets.QCheckBox("DTR")

        self.rb_lf = QtWidgets.QRadioButton("LF (\\n)")
        self.rb_crlf = QtWidgets.QRadioButton("CRLF (\\r\\n)")
        self.rb_crlf.setChecked(bool(self.state.use_crlf))
        self.rb_lf.setChecked(not bool(self.state.use_crlf))

        self.lbl_status = QtWidgets.QLabel("Status: Closed")

        pg.addWidget(QtWidgets.QLabel("Device"), 0, 0)
        pg.addWidget(self.cmb_port, 0, 1)
        pg.addWidget(self.btn_refresh, 0, 2)

        pg.addWidget(QtWidgets.QLabel("Baud"), 1, 0)
        pg.addWidget(self.cmb_baud, 1, 1, 1, 2)

        pg.addWidget(QtWidgets.QLabel("Data"), 2, 0)
        pg.addWidget(self.cmb_data, 2, 1, 1, 2)

        pg.addWidget(QtWidgets.QLabel("Stop"), 3, 0)
        pg.addWidget(self.cmb_stop, 3, 1, 1, 2)

        pg.addWidget(QtWidgets.QLabel("Parity"), 4, 0)
        pg.addWidget(self.cmb_parity, 4, 1, 1, 2)

        pg.addWidget(self.btn_open, 5, 1)
        pg.addWidget(self.btn_close, 5, 2)

        pg.addWidget(self.chk_rts, 6, 1)
        pg.addWidget(self.chk_dtr, 6, 2)

        pg.addWidget(self.rb_lf, 7, 1, 1, 2)
        pg.addWidget(self.rb_crlf, 8, 1, 1, 2)

        pg.addWidget(self.lbl_status, 9, 0, 1, 3)

        left.addWidget(port_box)
        left.addStretch(1)

        # Right: upper controls + lower log
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(splitter, 1)

        # Upper area
        upper = QtWidgets.QWidget()
        uv = QtWidgets.QVBoxLayout(upper)
        uv.setContentsMargins(0, 0, 0, 0)

        # Quick command buttons
        cmd_box = QtWidgets.QGroupBox("Quick Commands (Firmware modes: SM/STM/AR/SET)")
        cb = QtWidgets.QGridLayout(cmd_box)

        self.btn_sm_ch = QtWidgets.QPushButton("SM CH (Servo mode)")
        self.btn_sm_done = QtWidgets.QPushButton("DONE (done)")
        self.btn_ar = QtWidgets.QPushButton("AR (send AIRREP/REP then AR)")
        self.btn_set = QtWidgets.QPushButton("SET (enter REP mode)")

        self.btn_stm_ch = QtWidgets.QPushButton("STM CH (Stepper cycles)")
        self.btn_stm_in = QtWidgets.QPushButton("STM IN CH (Infinity)")
        self.btn_stm_set = QtWidgets.QPushButton("STM SET (Microstep)")

        cb.addWidget(self.btn_sm_ch, 0, 0)
        cb.addWidget(self.btn_sm_done, 0, 1)
        cb.addWidget(self.btn_ar, 0, 2)
        cb.addWidget(self.btn_set, 0, 3)

        cb.addWidget(self.btn_stm_ch, 1, 0)
        cb.addWidget(self.btn_stm_in, 1, 1)
        cb.addWidget(self.btn_stm_set, 1, 2)

        # Stepper helpers
        self.spin_stm_cycle = QtWidgets.QSpinBox()
        self.spin_stm_cycle.setRange(-1000000, 1000000)
        self.spin_stm_cycle.setValue(50)
        self.btn_send_stm_cycle = QtWidgets.QPushButton("Send cycle (STM CH + value)")

        self.cmb_step_mode = QtWidgets.QComboBox()
        self.cmb_step_mode.addItems(["0 (Full)", "1 (1/2)", "2 (1/4)", "3 (1/8)", "4 (1/16)", "5 (1/32)"])
        self.cmb_step_mode.setCurrentIndex(0)
        self.btn_send_step_mode = QtWidgets.QPushButton("Send step mode (STM SET + mode)")

        cb.addWidget(QtWidgets.QLabel("Cycle"), 2, 0)
        cb.addWidget(self.spin_stm_cycle, 2, 1)
        cb.addWidget(self.btn_send_stm_cycle, 2, 2, 1, 2)

        cb.addWidget(QtWidgets.QLabel("Microstep"), 3, 0)
        cb.addWidget(self.cmb_step_mode, 3, 1)
        cb.addWidget(self.btn_send_step_mode, 3, 2, 1, 2)

        # AIRREP control
        self.spin_airrep = QtWidgets.QSpinBox()
        self.spin_airrep.setRange(0, 50)  # 0 허용
        self.spin_airrep.setValue(int(self.state.airrep_count if self.state.airrep_count is not None else 1))
        self.btn_airrep_apply = QtWidgets.QPushButton("Apply AIRREP/REP")

        cb.addWidget(QtWidgets.QLabel("AIRREP (count)"), 4, 0)
        cb.addWidget(self.spin_airrep, 4, 1)
        cb.addWidget(self.btn_airrep_apply, 4, 2, 1, 2)

        # NEW: 0~9 quick buttons
        quick_box = QtWidgets.QGroupBox("AIRREP Quick (0~9)")
        qb = QtWidgets.QHBoxLayout(quick_box)
        qb.setContentsMargins(10, 10, 10, 10)
        qb.setSpacing(8)

        self.btn_airrep_quick: List[QtWidgets.QPushButton] = []
        for n in range(10):
            b = QtWidgets.QPushButton(str(n))
            b.setFixedSize(62, 40)  # <-- 여기서 더 키우고 싶으면 숫자만 조절
            f = b.font()
            f.setPointSize(10)  # <-- 폰트도 크게
            f.setBold(True)
            b.setFont(f)
            b.clicked.connect(lambda _=False, nn=n: self.on_airrep_quick(nn))
            self.btn_airrep_quick.append(b)
            qb.addWidget(b)

        cb.addWidget(quick_box, 5, 0, 1, 4)

        self.lbl_airrep_hint = QtWidgets.QLabel(
            "Hint: If you pressed SET, Apply sends just the number. Otherwise, sends 'AIRREP n'."
        )
        cb.addWidget(self.lbl_airrep_hint, 6, 0, 1, 4)

        uv.addWidget(cmd_box)

        # Servo continuous control panel
        servo_box = QtWidgets.QGroupBox("Servo Continuous (send numeric PWM repeatedly until done)")
        sb = QtWidgets.QGridLayout(servo_box)

        self.spin_servo_min = QtWidgets.QSpinBox()
        self.spin_servo_min.setRange(0, 10000)
        self.spin_servo_min.setValue(self.servo_min_us)

        self.spin_servo_max = QtWidgets.QSpinBox()
        self.spin_servo_max.setRange(0, 10000)
        self.spin_servo_max.setValue(self.servo_max_us)

        self.spin_servo_deg = QtWidgets.QDoubleSpinBox()
        self.spin_servo_deg.setRange(1.0, 720.0)
        self.spin_servo_deg.setDecimals(1)
        self.spin_servo_deg.setValue(self.servo_range_deg)

        self.spin_servo_pwm = QtWidgets.QSpinBox()
        self.spin_servo_pwm.setRange(0, 10000)
        self.spin_servo_pwm.setValue(1500)

        self.slider_servo_pwm = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_servo_pwm.setRange(self.servo_min_us, self.servo_max_us)
        self.slider_servo_pwm.setValue(1500)

        self.chk_servo_auto = QtWidgets.QCheckBox("Auto send on change")
        self.chk_servo_auto.setChecked(True)

        self.btn_servo_send = QtWidgets.QPushButton("Send PWM")
        self.lbl_servo_info = QtWidgets.QLabel("PWM: 1500 us | Angle: 0.0° | Mode: OFF")

        sb.addWidget(QtWidgets.QLabel("Min(us)"), 0, 0)
        sb.addWidget(self.spin_servo_min, 0, 1)
        sb.addWidget(QtWidgets.QLabel("Max(us)"), 0, 2)
        sb.addWidget(self.spin_servo_max, 0, 3)
        sb.addWidget(QtWidgets.QLabel("Range(°)"), 0, 4)
        sb.addWidget(self.spin_servo_deg, 0, 5)

        sb.addWidget(QtWidgets.QLabel("PWM(us)"), 1, 0)
        sb.addWidget(self.spin_servo_pwm, 1, 1)
        sb.addWidget(self.chk_servo_auto, 1, 2)
        sb.addWidget(self.btn_servo_send, 1, 3, 1, 3)

        sb.addWidget(self.slider_servo_pwm, 2, 0, 1, 6)
        sb.addWidget(self.lbl_servo_info, 3, 0, 1, 6)

        uv.addWidget(servo_box)

        # Manual send rows
        send_box = QtWidgets.QGroupBox("Manual Send Rows (ASCII only)")
        sv = QtWidgets.QVBoxLayout(send_box)

        self.rows: List[SendRow] = []
        for i in range(self.NUM_ROWS):
            r = SendRow(i)
            r.send_clicked.connect(self.on_send_row)
            self.rows.append(r)
            sv.addWidget(r)

        bottom = QtWidgets.QHBoxLayout()
        self.spin_repeat = QtWidgets.QSpinBox()
        self.spin_repeat.setRange(1, 100000)
        self.spin_repeat.setValue(1)
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(0, 60000)
        self.spin_interval.setValue(100)
        self.spin_interval.setSuffix(" ms")

        self.btn_run_checked = QtWidgets.QPushButton("Run Checked")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.lbl_prog = QtWidgets.QLabel("0 / 0")

        bottom.addWidget(QtWidgets.QLabel("Repeat"))
        bottom.addWidget(self.spin_repeat)
        bottom.addWidget(QtWidgets.QLabel("Interval"))
        bottom.addWidget(self.spin_interval)
        bottom.addWidget(self.lbl_prog)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_run_checked)
        bottom.addWidget(self.btn_cancel)
        sv.addLayout(bottom)

        quick = QtWidgets.QHBoxLayout()
        self.quick_edit = QtWidgets.QLineEdit()
        self.quick_edit.setPlaceholderText("Quick send: type and Enter (↑/↓ history)")
        self.btn_quick = QtWidgets.QPushButton("Send")
        quick.addWidget(self.quick_edit, 1)
        quick.addWidget(self.btn_quick)
        sv.addLayout(quick)

        uv.addWidget(send_box)
        splitter.addWidget(upper)

        # Lower area: log
        lower = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(lower)
        lv.setContentsMargins(0, 0, 0, 0)

        log_opts = QtWidgets.QHBoxLayout()
        self.chk_numeric_only = QtWidgets.QCheckBox("Numeric only")
        self.chk_numeric_only.setChecked(True)

        self.chk_show_tx = QtWidgets.QCheckBox("TX")
        self.chk_show_tx.setChecked(True)
        self.chk_show_rx = QtWidgets.QCheckBox("RX")
        self.chk_show_rx.setChecked(True)

        self.chk_show_raw_hex = QtWidgets.QCheckBox("RAW HEX (debug)")
        self.chk_show_raw_hex.setChecked(False)

        self.btn_clear_log = QtWidgets.QPushButton("Clear")

        log_opts.addWidget(self.chk_numeric_only)
        log_opts.addWidget(self.chk_show_tx)
        log_opts.addWidget(self.chk_show_rx)
        log_opts.addWidget(self.chk_show_raw_hex)
        log_opts.addStretch(1)
        log_opts.addWidget(self.btn_clear_log)
        lv.addLayout(log_opts)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        lv.addWidget(self.log_view, 1)

        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # Signals
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_open.clicked.connect(self.open_port)
        self.btn_close.clicked.connect(self.close_port)
        self.chk_rts.toggled.connect(self.on_rts)
        self.chk_dtr.toggled.connect(self.on_dtr)
        self.btn_clear_log.clicked.connect(self.log_view.clear)

        self.btn_run_checked.clicked.connect(self.run_checked)
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.btn_quick.clicked.connect(self.quick_send)
        self.quick_edit.returnPressed.connect(self.quick_send)

        # Quick command buttons
        self.btn_sm_ch.clicked.connect(self.send_sm_ch)
        self.btn_sm_done.clicked.connect(self.send_sm_done)
        self.btn_set.clicked.connect(self.send_set_and_enter_rep_mode)

        self.btn_stm_ch.clicked.connect(lambda: self.enqueue_send_ascii("STM CH"))
        self.btn_stm_in.clicked.connect(lambda: self.enqueue_send_ascii("STM IN CH"))
        self.btn_stm_set.clicked.connect(lambda: self.enqueue_send_ascii("STM SET"))
        self.btn_send_stm_cycle.clicked.connect(self.send_stm_cycle_sequence)
        self.btn_send_step_mode.clicked.connect(self.send_step_mode_sequence)

        # AIRREP apply + AR sequence
        self.btn_airrep_apply.clicked.connect(self.send_airrep_or_rep_only)
        self.btn_ar.clicked.connect(self.send_ar_with_rep)

        # Servo UI signals
        self.spin_servo_min.valueChanged.connect(self.on_servo_map_changed)
        self.spin_servo_max.valueChanged.connect(self.on_servo_map_changed)
        self.spin_servo_deg.valueChanged.connect(self.on_servo_map_changed)

        self.spin_servo_pwm.valueChanged.connect(self.on_servo_pwm_spin_changed)
        self.slider_servo_pwm.valueChanged.connect(self.on_servo_pwm_slider_changed)
        self.btn_servo_send.clicked.connect(self.send_servo_pwm_now)

        # Restore
        self.refresh_ports(select_port=self.state.last_port)
        self.restore_rows()
        self._update_servo_info()

    # -----------------------------
    # Persistence hooks
    # -----------------------------
    def closeEvent(self, e):
        self.state.last_port = self.cmb_port.currentData() or ""
        self.state.last_baud = int(self.cmb_baud.currentText())
        self.state.last_databits = int(self.cmb_data.currentText())
        self.state.last_stopbits = int(self.cmb_stop.currentText())
        self.state.last_parity = self.cmb_parity.currentText()
        self.state.use_crlf = self.rb_crlf.isChecked()
        self.state.send_rows = [r.to_dict() for r in self.rows]
        self.state.history = self._history[-200:]
        self.state.airrep_count = int(self.spin_airrep.value())
        save_state(self.state)

        try:
            self.close_port()
        except Exception:
            pass
        super().closeEvent(e)

    def restore_rows(self):
        if self.state.send_rows:
            for i, d in enumerate(self.state.send_rows):
                if i < len(self.rows):
                    self.rows[i].from_dict(d)
            return

        seed = [
            "SET", "3", "AR", "SM CH", "1500",
            "1600", "1700", "done", "STM SET"
        ]
        for i, s in enumerate(seed):
            if i < len(self.rows):
                self.rows[i].edit.setText(s)

    # -----------------------------
    # Log behavior
    # -----------------------------
    def log(self, s: str):
        self.log_view.appendPlainText(s)

    def _servo_angle_from_pwm(self, pwm_us: int) -> float:
        mn = int(self.spin_servo_min.value())
        mx = int(self.spin_servo_max.value())
        deg = float(self.spin_servo_deg.value())
        if mx <= mn:
            return 0.0
        x = (pwm_us - mn) / float(mx - mn)
        x = clamp(x, 0.0, 1.0)
        return x * deg

    def log_tx(self, text: str):
        if not self.chk_show_tx.isChecked():
            return
        self._log_filtered("TX", text)

    def log_rx(self, text: str):
        if not self.chk_show_rx.isChecked():
            return
        self._log_filtered("RX", text)

    def _log_filtered(self, direction: str, text: str):
        if self.chk_numeric_only.isChecked():
            if direction == "TX" and self.servo_mode_active and is_int_string(text):
                pwm = int(text.strip())
                ang = self._servo_angle_from_pwm(pwm)
                self.log(f"[{ts()}] TX: {pwm} us, {ang:.1f}°")
                return

            nums = extract_numbers(text)
            if not nums:
                return
            self.log(f"[{ts()}] {direction}: " + ", ".join(nums))
        else:
            self.log(f"[{ts()}] {direction}: {text}")

    def eol(self) -> bytes:
        return b"\r\n" if self.rb_crlf.isChecked() else b"\n"

    def ensure_open(self) -> bool:
        if not self.worker or not self.worker.is_open():
            QtWidgets.QMessageBox.warning(self, "Serial", "Port is not open.")
            return False
        return True

    # -----------------------------
    # Port
    # -----------------------------
    def refresh_ports(self, select_port: str = ""):
        self.cmb_port.clear()
        ports = list_ports()
        sel_index = -1
        for i, p in enumerate(ports):
            self.cmb_port.addItem(f"{p.device} | {p.description}", userData=p.device)
            if select_port and p.device == select_port:
                sel_index = i
        if not ports:
            self.cmb_port.addItem("(no ports)", userData=None)
        if sel_index >= 0:
            self.cmb_port.setCurrentIndex(sel_index)

    def open_port(self):
        dev = self.cmb_port.currentData()
        if not dev:
            QtWidgets.QMessageBox.warning(self, "Port", "No port selected.")
            return

        baud = int(self.cmb_baud.currentText())
        databits = int(self.cmb_data.currentText())
        stopbits = int(self.cmb_stop.currentText())
        parity = self.cmb_parity.currentText()

        self.worker = SerialWorker(dev, baud, databits, stopbits, parity)
        self.worker.status.connect(self.on_status)
        self.worker.rx_raw.connect(self.on_rx_raw)
        self.worker.rx_line.connect(self.on_rx_line)
        self.worker.tx_done.connect(self.on_tx_done)
        self.worker.start()

        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)

    def close_port(self):
        if self.worker:
            self.worker.stop()
            try:
                self.worker.wait(300)
            except Exception:
                pass
            self.worker = None
        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.lbl_status.setText("Status: Closed")

    @QtCore.pyqtSlot(bool, str)
    def on_status(self, opened: bool, msg: str):
        self.lbl_status.setText(f"Status: {msg}")
        self.log(f"[{ts()}] STATUS: {msg}")
        if not opened:
            self.btn_open.setEnabled(True)
            self.btn_close.setEnabled(False)

    def on_rts(self, on: bool):
        if self.worker and self.worker.is_open():
            self.worker.set_rts(on)

    def on_dtr(self, on: bool):
        if self.worker and self.worker.is_open():
            self.worker.set_dtr(on)

    # -----------------------------
    # TX helpers
    # -----------------------------
    def enqueue_send_ascii(self, text: str, append_eol: bool = True):
        if not self.ensure_open():
            return
        payload = text.encode("ascii", errors="ignore")
        if append_eol:
            payload += self.eol()

        self.log_tx(text)
        self.worker.enqueue_write(payload)

        if text.strip():
            self._history.append(text.strip())
            self._history = self._history[-200:]
            self._hist_idx = len(self._history)

    @QtCore.pyqtSlot(bytes)
    def on_tx_done(self, payload: bytes):
        if self.chk_show_raw_hex.isChecked():
            self.log(f"[{ts()}] TX RAW: {payload.hex(' ')}")

    # -----------------------------
    # RX
    # -----------------------------
    @QtCore.pyqtSlot(bytes)
    def on_rx_raw(self, data: bytes):
        if self.chk_show_raw_hex.isChecked():
            self.log(f"[{ts()}] RX RAW: {data.hex(' ')}")

    @QtCore.pyqtSlot(str)
    def on_rx_line(self, line: str):
        if "SM DONE" in line:
            self.servo_mode_active = False
            self._update_servo_info()
        self.log_rx(line)

    # -----------------------------
    # AIRREP Quick Buttons (0~9)
    # -----------------------------
    def on_airrep_quick(self, n: int):
        """0~9 버튼 클릭 시 spin_airrep 값을 설정."""
        self.spin_airrep.setValue(int(n))

    # -----------------------------
    # SET / AIRREP(또는 REP 숫자) / AR
    # -----------------------------
    def send_set_and_enter_rep_mode(self):
        self.enqueue_send_ascii("SET")
        self.awaiting_rep_after_set = True
        self.log(f"[{ts()}] INFO: Awaiting REP number after SET (Apply will send number-only).")

    def send_airrep_or_rep_only(self):
        rep = int(self.spin_airrep.value())
        rep = max(0, min(50, rep))  # 0 허용
        self.spin_airrep.setValue(rep)

        if self.awaiting_rep_after_set:
            self.enqueue_send_ascii(str(rep))   # 숫자만
            self.awaiting_rep_after_set = False
            self.log(f"[{ts()}] INFO: REP sent as number-only. (awaiting_rep_after_set cleared)")
        else:
            self.enqueue_send_ascii(f"AIRREP {rep}")

    def send_ar_with_rep(self):
        rep = int(self.spin_airrep.value())
        rep = max(0, min(50, rep))
        self.spin_airrep.setValue(rep)

        if self.awaiting_rep_after_set:
            self.enqueue_send_ascii(str(rep))   # 숫자만
            self.awaiting_rep_after_set = False
            self.enqueue_send_ascii("AR")
        else:
            self.enqueue_send_ascii(f"AIRREP {rep}")
            self.enqueue_send_ascii("AR")

    # -----------------------------
    # Manual send rows / Quick send
    # -----------------------------
    def on_send_row(self, idx: int):
        r = self.rows[idx]
        text = r.edit.text().strip()
        if not text:
            return
        self.enqueue_send_ascii(text, append_eol=True)

    def quick_send(self):
        text = self.quick_edit.text().strip()
        if not text:
            return
        self.enqueue_send_ascii(text, append_eol=True)

    # -----------------------------
    # Run checked rows
    # -----------------------------
    def run_checked(self):
        if self._run_running:
            return
        if not self.ensure_open():
            return

        selected = [r for r in self.rows if r.chk.isChecked() and r.edit.text().strip()]
        if not selected:
            QtWidgets.QMessageBox.warning(self, "Run Checked", "No checked rows.")
            return

        repeat = int(self.spin_repeat.value())
        interval_ms = int(self.spin_interval.value())
        total = repeat * len(selected)

        self._run_running = True
        self.btn_run_checked.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_prog.setText(f"0 / {total}")

        def run():
            sent = 0
            for _ in range(repeat):
                if not self._run_running:
                    break
                for r in selected:
                    if not self._run_running:
                        break
                    try:
                        self.enqueue_send_ascii(r.edit.text().strip(), append_eol=True)
                    except Exception:
                        pass
                    sent += 1
                    QtCore.QMetaObject.invokeMethod(
                        self.lbl_prog, "setText", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"{sent} / {total}")
                    )
                    if interval_ms > 0:
                        time.sleep(interval_ms / 1000.0)
            QtCore.QMetaObject.invokeMethod(self, "_run_done", QtCore.Qt.QueuedConnection)

        self._run_thread = threading.Thread(target=run, daemon=True)
        self._run_thread.start()

    def cancel_run(self):
        self._run_running = False

    @QtCore.pyqtSlot()
    def _run_done(self):
        self._run_running = False
        self.btn_run_checked.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    # -----------------------------
    # Servo mode / controls
    # -----------------------------
    def send_sm_ch(self):
        self.enqueue_send_ascii("SM CH")
        self.servo_mode_active = True
        self._update_servo_info()

    def send_sm_done(self):
        self.enqueue_send_ascii("done")
        self.servo_mode_active = False
        self._update_servo_info()

    def on_servo_map_changed(self):
        mn = int(self.spin_servo_min.value())
        mx = int(self.spin_servo_max.value())
        if mx <= mn:
            mx = mn + 1
            self.spin_servo_max.setValue(mx)

        self.slider_servo_pwm.blockSignals(True)
        self.slider_servo_pwm.setRange(mn, mx)
        self.slider_servo_pwm.blockSignals(False)

        cur = int(self.spin_servo_pwm.value())
        cur = int(clamp(cur, mn, mx))
        self.spin_servo_pwm.setValue(cur)
        self.slider_servo_pwm.setValue(cur)
        self._update_servo_info()

    def on_servo_pwm_spin_changed(self, v: int):
        mn = int(self.spin_servo_min.value())
        mx = int(self.spin_servo_max.value())
        v = int(clamp(v, mn, mx))

        if self.slider_servo_pwm.value() != v:
            self.slider_servo_pwm.blockSignals(True)
            self.slider_servo_pwm.setValue(v)
            self.slider_servo_pwm.blockSignals(False)

        self._update_servo_info()
        if self.chk_servo_auto.isChecked():
            self._schedule_servo_send()

    def on_servo_pwm_slider_changed(self, v: int):
        if self.spin_servo_pwm.value() != v:
            self.spin_servo_pwm.blockSignals(True)
            self.spin_servo_pwm.setValue(v)
            self.spin_servo_pwm.blockSignals(False)

        self._update_servo_info()
        if self.chk_servo_auto.isChecked():
            self._schedule_servo_send()

    def _update_servo_info(self):
        pwm = int(self.spin_servo_pwm.value())
        ang = self._servo_angle_from_pwm(pwm)
        mode = "ON" if self.servo_mode_active else "OFF"
        self.lbl_servo_info.setText(f"PWM: {pwm} us | Angle: {ang:.1f}° | Mode: {mode}")

    def _schedule_servo_send(self):
        self._servo_pending_send = True
        if not self._servo_send_timer.isActive():
            self._servo_send_timer.start(40)

    def _servo_send_if_pending(self):
        if not self._servo_pending_send:
            return
        self._servo_pending_send = False
        self.send_servo_pwm_now()

    def send_servo_pwm_now(self):
        if not self.ensure_open():
            return
        if not self.servo_mode_active:
            self.log(f"[{ts()}] INFO: Servo mode is OFF (send 'SM CH' first if firmware requires).")
        pwm = int(self.spin_servo_pwm.value())
        self.enqueue_send_ascii(str(pwm))

    # -----------------------------
    # Stepper helper sequences
    # -----------------------------
    def send_stm_cycle_sequence(self):
        cyc = int(self.spin_stm_cycle.value())
        self.enqueue_send_ascii("STM CH")
        self.enqueue_send_ascii(str(cyc))

    def send_step_mode_sequence(self):
        mode_idx = self.cmb_step_mode.currentIndex()
        self.enqueue_send_ascii("STM SET")
        self.enqueue_send_ascii(str(mode_idx))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1280, 900)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

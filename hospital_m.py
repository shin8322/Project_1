import sys
import os
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, List

import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtWidgets


def ts() -> str:
    return time.strftime("%H:%M:%S")


def list_ports():
    return list(serial.tools.list_ports.comports())


def app_state_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".ar_control_tool")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "state.json")


@dataclass
class AppState:
    last_port: str = ""
    last_baud: int = 115200
    last_databits: int = 8
    last_stopbits: int = 1
    last_parity: str = "None"
    use_crlf: bool = False
    airrep_count: int = 1

    def to_json(self) -> dict:
        return {
            "last_port": self.last_port,
            "last_baud": int(self.last_baud),
            "last_databits": int(self.last_databits),
            "last_stopbits": int(self.last_stopbits),
            "last_parity": self.last_parity,
            "use_crlf": bool(self.use_crlf),
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
        s.airrep_count = int(d.get("airrep_count", 1))
        return s


def load_state() -> AppState:
    p = app_state_path()
    if not os.path.exists(p):
        return AppState()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return AppState.from_json(json.load(f))
    except Exception:
        return AppState()


def save_state(st: AppState):
    p = app_state_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st.to_json(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class SerialWorker(QtCore.QThread):
    rx_line = QtCore.pyqtSignal(str)
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
                    self._rx_buf.extend(data)
                    self._drain_lines()

                payload = None
                with self._tx_lock:
                    if self._tx_queue:
                        payload = self._tx_queue.pop(0)
                if payload:
                    self._ser.write(payload)
                    self._ser.flush()

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

    @QtCore.pyqtSlot(bytes)
    def enqueue_write(self, payload: bytes):
        if not self._ser or not self._ser.is_open:
            return
        with self._tx_lock:
            self._tx_queue.append(payload)

    def _drain_lines(self):
        while True:
            idx = self._rx_buf.find(b"\n")
            if idx == -1:
                idx = self._rx_buf.find(b"\r")
            if idx == -1:
                break
            raw = self._rx_buf[:idx]
            del self._rx_buf[:idx + 1]
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                self.rx_line.emit(line)


class MainWindow(QtWidgets.QMainWindow):
    AIRREP_MAX = 50

    def __init__(self):
        super().__init__()
        self.state = load_state()
        self.worker: Optional[SerialWorker] = None

        self.setWindowTitle("AR Control Tool (SET / AIRREP / AR)")

        # ✅ 초기 실행 창 크기 확대
        self.resize(1280, 820)
        self.setMinimumSize(1100, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---------------- Serial Port ----------------
        port_box = QtWidgets.QGroupBox("Serial Port")
        pg = QtWidgets.QGridLayout(port_box)
        pg.setHorizontalSpacing(10)
        pg.setVerticalSpacing(8)
        pg.setColumnStretch(0, 0)
        pg.setColumnStretch(1, 1)
        pg.setColumnStretch(2, 0)

        self.cmb_port = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setFixedWidth(140)

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

        self.rb_lf = QtWidgets.QRadioButton("LF (\\n)")
        self.rb_crlf = QtWidgets.QRadioButton("CRLF (\\r\\n)")
        self.rb_crlf.setChecked(bool(self.state.use_crlf))
        self.rb_lf.setChecked(not bool(self.state.use_crlf))

        self.btn_open = QtWidgets.QPushButton("Open")
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_open.setFixedWidth(120)
        self.btn_close.setFixedWidth(120)
        self.btn_close.setEnabled(False)

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

        eol_row = QtWidgets.QHBoxLayout()
        eol_row.addWidget(self.rb_lf)
        eol_row.addSpacing(20)
        eol_row.addWidget(self.rb_crlf)
        eol_row.addStretch(1)
        eol_wrap = QtWidgets.QWidget()
        eol_wrap.setLayout(eol_row)
        pg.addWidget(eol_wrap, 5, 1, 1, 2)

        oc_row = QtWidgets.QHBoxLayout()
        oc_row.addWidget(self.btn_open)
        oc_row.addWidget(self.btn_close)
        oc_row.addStretch(1)
        oc_wrap = QtWidgets.QWidget()
        oc_wrap.setLayout(oc_row)
        pg.addWidget(oc_wrap, 6, 1, 1, 2)

        pg.addWidget(self.lbl_status, 7, 0, 1, 3)
        root.addWidget(port_box)

        # ---------------- Control ----------------
        ctrl_box = QtWidgets.QGroupBox("Control")
        cg = QtWidgets.QGridLayout(ctrl_box)
        cg.setContentsMargins(10, 14, 10, 10)
        cg.setHorizontalSpacing(10)
        cg.setVerticalSpacing(10)

        cg.setColumnStretch(0, 2)
        cg.setColumnStretch(1, 0)
        cg.setColumnStretch(2, 1)
        cg.setColumnStretch(3, 2)

        self.btn_set = QtWidgets.QPushButton("SET")
        self.lbl_airrep = QtWidgets.QLabel(f"AIRREP Count (0~{self.AIRREP_MAX})")

        self.spin_airrep = QtWidgets.QSpinBox()
        self.spin_airrep.setRange(0, self.AIRREP_MAX)
        self.spin_airrep.setValue(int(self.state.airrep_count or 1))

        self.btn_airrep_apply = QtWidgets.QPushButton("AIRREP Apply")

        for b in (self.btn_set, self.btn_airrep_apply):
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            b.setMinimumHeight(42)

        cg.addWidget(self.btn_set, 0, 0)
        cg.addWidget(self.lbl_airrep, 0, 1)
        cg.addWidget(self.spin_airrep, 0, 2)
        cg.addWidget(self.btn_airrep_apply, 0, 3)

        quick_box = QtWidgets.QGroupBox("AIRREP Quick (0~9)")
        qh = QtWidgets.QHBoxLayout(quick_box)
        qh.setContentsMargins(10, 14, 10, 10)
        qh.setSpacing(8)

        self.quick_btns: List[QtWidgets.QPushButton] = []
        for d in range(10):
            b = QtWidgets.QPushButton(str(d))
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            b.setMinimumHeight(52)
            b.clicked.connect(lambda _=False, dd=d: self._set_and_apply_airrep(dd))
            self.quick_btns.append(b)
            qh.addWidget(b, 1)

        quick_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        cg.addWidget(quick_box, 1, 0, 1, 4)

        self.btn_ar_zero = QtWidgets.QPushButton("AR (AIR 0회 동작)")
        self.btn_ar_apply = QtWidgets.QPushButton("AR (설정 횟수 적용 후 동작)")
        for b in (self.btn_ar_zero, self.btn_ar_apply):
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            b.setMinimumHeight(44)

        cg.addWidget(self.btn_ar_zero, 2, 0, 1, 2)
        cg.addWidget(self.btn_ar_apply, 2, 2, 1, 2)

        root.addWidget(ctrl_box)

        # ---------------- Log ----------------
        log_box = QtWidgets.QGroupBox("Log")
        lv = QtWidgets.QVBoxLayout(log_box)
        lv.setContentsMargins(10, 14, 10, 10)
        lv.setSpacing(8)

        opt = QtWidgets.QHBoxLayout()
        self.chk_show_tx = QtWidgets.QCheckBox("TX")
        self.chk_show_rx = QtWidgets.QCheckBox("RX")
        self.chk_show_tx.setChecked(True)
        self.chk_show_rx.setChecked(True)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setFixedWidth(110)

        opt.addWidget(self.chk_show_tx)
        opt.addWidget(self.chk_show_rx)
        opt.addStretch(1)
        opt.addWidget(self.btn_clear)
        lv.addLayout(opt)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        lv.addWidget(self.log_view, 1)

        root.addWidget(log_box, 1)

        # ---------------- signals ----------------
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_open.clicked.connect(self.open_port)
        self.btn_close.clicked.connect(self.close_port)
        self.btn_clear.clicked.connect(self.log_view.clear)

        self.btn_set.clicked.connect(lambda: self.send_ascii("SET"))
        self.btn_airrep_apply.clicked.connect(self.send_airrep_only)
        self.btn_ar_zero.clicked.connect(self.send_ar_zero)
        self.btn_ar_apply.clicked.connect(self.send_ar_with_airrep)

        self.refresh_ports(select_port=self.state.last_port)

    def closeEvent(self, e):
        self.state.last_port = self.cmb_port.currentData() or ""
        self.state.last_baud = int(self.cmb_baud.currentText())
        self.state.last_databits = int(self.cmb_data.currentText())
        self.state.last_stopbits = int(self.cmb_stop.currentText())
        self.state.last_parity = self.cmb_parity.currentText()
        self.state.use_crlf = self.rb_crlf.isChecked()
        self.state.airrep_count = int(self.spin_airrep.value())
        save_state(self.state)

        try:
            self.close_port()
        except Exception:
            pass
        super().closeEvent(e)

    def eol(self) -> bytes:
        return b"\r\n" if self.rb_crlf.isChecked() else b"\n"

    def log(self, s: str):
        self.log_view.appendPlainText(s)

    def ensure_open(self) -> bool:
        if not self.worker or not self.worker.is_open():
            QtWidgets.QMessageBox.warning(self, "Serial", "Port is not open.")
            return False
        return True

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
        self.worker.rx_line.connect(self.on_rx_line)
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
    @QtCore.pyqtSlot(str)
    def on_rx_line(self, line: str):
        if self.chk_show_rx.isChecked():
            self.log(f"[{ts()}] RX: {line}")

    def send_ascii(self, text: str):
        if not self.ensure_open():
            return
        payload = text.encode("ascii", errors="ignore") + self.eol()
        if self.chk_show_tx.isChecked():
            self.log(f"[{ts()}] TX: {text}")
        self.worker.enqueue_write(payload)

    def send_airrep_only(self):
        rep = int(self.spin_airrep.value())
        self.send_ascii(f"AIRREP {rep}")

    def send_ar_zero(self):
        self.send_ascii("AIRREP 0")
        self.send_ascii("AR")

    def send_ar_with_airrep(self):
        rep = int(self.spin_airrep.value())
        self.send_ascii(f"AIRREP {rep}")
        self.send_ascii("AR")

    def _set_and_apply_airrep(self, d: int):
        self.spin_airrep.setValue(int(d))
        self.send_airrep_only()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

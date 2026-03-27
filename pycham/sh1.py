import os
import re
import numpy as np
import pandas as pd
from tkinter import filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import ttkbootstrap as ttk
from ttkbootstrap.constants import *


def _norm_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_voltage_current(csv_path: str):
    # Try with header
    try:
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        norm = {c: _norm_colname(c) for c in cols}

        v_candidates = [c for c in cols if norm[c] in ("v", "voltage") or "volt" in norm[c]]
        i_candidates = [c for c in cols if norm[c] in ("i", "current", "a") or "curr" in norm[c] or "amp" in norm[c]]

        if v_candidates and i_candidates:
            v_col = v_candidates[0]
            i_col = i_candidates[0]
            v = pd.to_numeric(df[v_col], errors="coerce").to_numpy()
            i = pd.to_numeric(df[i_col], errors="coerce").to_numpy()
            mask = np.isfinite(v) & np.isfinite(i)
            v, i = v[mask], i[mask]
            if len(v) < 2:
                raise ValueError("전압/전류 유효 데이터가 2개 이상 필요합니다.")
            return v, i, str(v_col), str(i_col)
    except Exception:
        pass

    # Assume no header
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV에 최소 2개 컬럼(전압, 전류)이 필요합니다. (1열=전압, 2열=전류)")
    v = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    i = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    mask = np.isfinite(v) & np.isfinite(i)
    v, i = v[mask], i[mask]
    if len(v) < 2:
        raise ValueError("유효한 전압/전류 데이터가 2개 이상 필요합니다.")
    return v, i, "col0(V)", "col1(I)"


def compute_baseline(v, i, idx1, idx2):
    v1, i1 = float(v[idx1]), float(i[idx1])
    v2, i2 = float(v[idx2]), float(i[idx2])
    if v1 == v2:
        raise ValueError("두 점의 전압이 같아서 계산이 불가능합니다.")
    m = (i2 - i1) / (v2 - v1)
    return v1, i1, v2, i2, m


def baseline_area_and_curve(v, i, idx1, idx2, mode="above"):
    v1, i1, v2, i2, m = compute_baseline(v, i, idx1, idx2)
    vmin, vmax = (v1, v2) if v1 < v2 else (v2, v1)

    mask = (v >= vmin) & (v <= vmax)
    vv = v[mask].astype(float)
    ii = i[mask].astype(float)
    if len(vv) < 2:
        raise ValueError("선택한 두 점 사이 구간에 데이터가 충분하지 않습니다.")

    order = np.argsort(vv)
    vv = vv[order]
    ii = ii[order]

    ib = i1 + m * (vv - v1)
    diff = ii - ib

    if mode == "above":
        diff_int = np.maximum(diff, 0.0)
    elif mode == "abs":
        diff_int = np.abs(diff)
    elif mode == "signed":
        diff_int = diff
    else:
        raise ValueError("mode는 'above', 'signed', 'abs' 중 하나여야 합니다.")

    area = np.trapezoid(diff_int, vv)
    return float(area), vv, ib, ii, float(m)


class ModernIVApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("I-V Analyzer")
        self.geometry("1280x860")
        self.minsize(1100, 720)

        self.csv_path = None
        self.v = None
        self.i = None

        # Selection
        self.current_points = []
        self.current_markers = []

        # Two baselines
        self.baselines = [
            {"line": None, "fill": None, "label": None, "area": None, "slope": None, "idxs": None},
            {"line": None, "fill": None, "label": None, "area": None, "slope": None, "idxs": None},
        ]
        self.ratio_label = None

        # User-added lines (unlimited)
        self.user_lines = []   # list of dicts: {"line": artist, "label": artist, "p": ((x1,y1),(x2,y2))}
        self.user_line_counter = 0

        # UI state
        self.area_mode = ttk.StringVar(value="above")
        self.var_all_xticks = ttk.BooleanVar(value=False)
        self.var_all_yticks = ttk.BooleanVar(value=False)

        # Draw mode: baseline vs user line
        self.draw_mode = ttk.StringVar(value="baseline")  # "baseline" or "userline"

        self._build_ui()
        self._bind_plot_events()
        self._set_status("Open CSV → Plot → (Baseline/User Line 모드 선택) 후 그래프 클릭 2번으로 선 생성")

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=(14, 12))
        top.pack(side=TOP, fill=X)

        ttk.Button(top, text="Open CSV", bootstyle=PRIMARY, command=self.open_csv).pack(side=LEFT)
        self.btn_plot = ttk.Button(top, text="Plot", bootstyle=SUCCESS, command=self.plot_data, state=DISABLED)
        self.btn_plot.pack(side=LEFT, padx=(10, 0))
        ttk.Button(top, text="Clear (C)", bootstyle=SECONDARY, command=lambda: self.reset_all(keep_plot=True)).pack(side=LEFT, padx=(10, 0))
        ttk.Button(top, text="Toggle Theme", bootstyle=INFO, command=self.toggle_theme).pack(side=LEFT, padx=(10, 0))

        self.file_badge = ttk.Label(top, text="No file", bootstyle="secondary", anchor=W)
        self.file_badge.pack(side=LEFT, padx=(14, 0), fill=X, expand=True)

        ttk.Separator(self).pack(side=TOP, fill=X)

        body = ttk.Frame(self, padding=(14, 12))
        body.pack(side=TOP, fill=BOTH, expand=True)

        left = ttk.Frame(body, width=340)
        left.pack(side=LEFT, fill=Y, padx=(0, 12))
        left.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side=LEFT, fill=BOTH, expand=True)

        # Draw Mode
        lf_draw = ttk.Labelframe(left, text="Draw mode", padding=10)
        lf_draw.pack(fill=X, pady=(0, 10))
        ttk.Radiobutton(lf_draw, text="Baseline (2 lines max + ratios)", variable=self.draw_mode, value="baseline").pack(anchor=W, pady=2)
        ttk.Radiobutton(lf_draw, text="User Line (unlimited)", variable=self.draw_mode, value="userline").pack(anchor=W, pady=2)
        ttk.Label(lf_draw, text="Tip: U = undo last user line", bootstyle="secondary").pack(anchor=W, pady=(8, 0))

        # Area mode (applies to baseline only)
        lf_mode = ttk.Labelframe(left, text="Area mode (baseline only)", padding=10)
        lf_mode.pack(fill=X, pady=(0, 10))
        ttk.Radiobutton(lf_mode, text="Above baseline only", variable=self.area_mode, value="above").pack(anchor=W, pady=2)
        ttk.Radiobutton(lf_mode, text="Signed area", variable=self.area_mode, value="signed").pack(anchor=W, pady=2)
        ttk.Radiobutton(lf_mode, text="Absolute area", variable=self.area_mode, value="abs").pack(anchor=W, pady=2)

        lf_ticks = ttk.Labelframe(left, text="Axis ticks", padding=10)
        lf_ticks.pack(fill=X, pady=(0, 10))
        ttk.Checkbutton(lf_ticks, text="Show all x ticks (V)", variable=self.var_all_xticks).pack(anchor=W, pady=2)
        ttk.Checkbutton(lf_ticks, text="Show all y ticks (I)", variable=self.var_all_yticks).pack(anchor=W, pady=2)

        lf_help = ttk.Labelframe(left, text="Controls", padding=10)
        lf_help.pack(fill=BOTH, expand=True)
        help_text = (
            "Baseline mode:\n"
            "• Click 2 pts → BL1\n"
            "• Click 2 pts → BL2\n"
            "• Auto area + ratios\n\n"
            "User Line mode:\n"
            "• Click 2 pts → add line\n"
            "• Repeat unlimited\n\n"
            "Shortcuts:\n"
            "• C : Clear all\n"
            "• ESC : Clear current selection\n"
            "• U : Undo last user line\n\n"
            "Note: Toolbar Pan/Zoom must be OFF."
        )
        ttk.Label(lf_help, text=help_text, justify=LEFT).pack(anchor=W)

        # Plot
        self.fig = Figure(figsize=(9.5, 6.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load CSV to plot I-V")
        self.ax.set_xlabel("Voltage (V)")
        self.ax.set_ylabel("Current (A)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill=X)

        # Status bar
        ttk.Separator(self).pack(side=TOP, fill=X)
        self.status_var = ttk.StringVar(value="")
        status = ttk.Frame(self, padding=(14, 10))
        status.pack(side=BOTTOM, fill=X)
        ttk.Label(status, textvariable=self.status_var, anchor=W).pack(side=LEFT, fill=X, expand=True)

    def toggle_theme(self):
        cur = self.style.theme.name
        self.style.theme_use("darkly" if cur != "darkly" else "flatly")

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # ---------- Actions ----------
    def open_csv(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            self._set_status("파일 선택이 취소되었습니다.")
            return

        try:
            v, i, v_col, i_col = load_voltage_current(path)
            self.csv_path = path
            self.v, self.i = v, i

            self.file_badge.configure(text=f"{os.path.basename(path)}   (V={v_col}, I={i_col}, N={len(v)})")
            self.btn_plot.configure(state=NORMAL)

            self.ax.clear()
            self.ax.set_title("Ready — press Plot")
            self.ax.set_xlabel("Voltage (V)")
            self.ax.set_ylabel("Current (A)")
            self.reset_all(keep_plot=True)
            self.canvas.draw_idle()

            self._set_status(f"CSV 로드 완료: {os.path.basename(path)}")
        except Exception as e:
            ttk.dialogs.Messagebox.show_error(str(e), title="Load error")
            self._set_status("CSV 로드 실패.")

    def plot_data(self):
        if self.v is None or self.i is None:
            ttk.dialogs.Messagebox.show_warning("먼저 CSV 파일을 선택하세요.", title="Order")
            return

        self.ax.clear()
        self.ax.set_title(f"I-V Curve — {os.path.basename(self.csv_path)}")
        self.ax.set_xlabel("Voltage (V)")
        self.ax.set_ylabel("Current (A)")

        self.ax.plot(self.v, self.i, linewidth=1.4, zorder=2)
        self.ax.scatter(self.v, self.i, s=18, zorder=3)

        if self.var_all_xticks.get():
            vx = np.unique(self.v)
            self.ax.set_xticks(vx)
            self.ax.set_xticklabels([f"{x:g}" for x in vx], rotation=90)
        if self.var_all_yticks.get():
            iy = np.unique(self.i)
            self.ax.set_yticks(iy)
            self.ax.set_yticklabels([f"{y:g}" for y in iy])

        self.fig.tight_layout()
        self.reset_all(keep_plot=True)
        self.canvas.draw_idle()
        self._set_status("Plot 완료. 선택한 Draw mode에 따라 그래프에서 2점을 클릭해 선을 만드세요.")

    # ---------- Interaction ----------
    def _bind_plot_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        k = (event.key or "").lower()
        if k == "c":
            self.reset_all(keep_plot=True)
            self.canvas.draw_idle()
            self._set_status("초기화 완료.")
        elif k == "escape":
            self._clear_current_selection()
            self.canvas.draw_idle()
            self._set_status("현재 선택만 초기화.")
        elif k == "u":
            self.undo_last_user_line()
            self.canvas.draw_idle()

    def on_plot_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.v is None or self.i is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = self._nearest_point_index(event.xdata, event.ydata)
        if self.current_points and idx == self.current_points[-1]:
            self._set_status("같은 점이 다시 선택되었습니다. 다른 점을 선택하세요.")
            return

        # If baseline mode and both baselines exist, next action restarts baselines only (keep user lines)
        if self.draw_mode.get() == "baseline":
            if self.baselines[0]["idxs"] is not None and self.baselines[1]["idxs"] is not None:
                self._remove_baseline(0)
                self._remove_baseline(1)
                if self.ratio_label is not None:
                    try:
                        self.ratio_label.remove()
                    except Exception:
                        pass
                    self.ratio_label = None

        self.current_points.append(idx)
        self._draw_current_markers()

        if len(self.current_points) == 1:
            self._set_status("첫 점 선택됨. 두 번째 점을 클릭하세요.")
            self.canvas.draw_idle()
            return

        # Two points selected -> create line
        idx1, idx2 = self.current_points[0], self.current_points[1]

        if self.draw_mode.get() == "userline":
            # Add arbitrary user line
            self._render_user_line(idx1, idx2)
            self._clear_current_selection()
            self._set_status("User line 추가됨. 계속 2점을 클릭해 더 추가할 수 있습니다. (U: undo)")
            self.canvas.draw_idle()
            return

        # Baseline mode
        which = 0 if self.baselines[0]["idxs"] is None else 1
        try:
            area, vv, ib, ii_used, slope = baseline_area_and_curve(self.v, self.i, idx1, idx2, mode=self.area_mode.get())
        except Exception as e:
            ttk.dialogs.Messagebox.show_error(str(e), title="Baseline error")
            self._clear_current_selection()
            self.canvas.draw_idle()
            return

        self._render_baseline(which, idx1, idx2, vv, ib, ii_used, area, slope)
        self._clear_current_selection()

        if self.baselines[0]["idxs"] is not None and self.baselines[1]["idxs"] is not None:
            self._render_ratios()
            self._set_status(self._build_status_two_done())
        else:
            self._set_status("Baseline #1 완료. 이제 baseline #2 두 점을 클릭하세요.")

        self.canvas.draw_idle()

    def _nearest_point_index(self, x, y) -> int:
        dx = self.v - x
        dy = self.i - y
        return int(np.argmin(dx * dx + dy * dy))

    def _draw_current_markers(self):
        for a in self.current_markers:
            try:
                a.remove()
            except Exception:
                pass
        self.current_markers.clear()

        for idx in self.current_points:
            a = self.ax.scatter([self.v[idx]], [self.i[idx]], s=130, zorder=50)
            self.current_markers.append(a)

    def _clear_current_selection(self):
        self.current_points.clear()
        for a in self.current_markers:
            try:
                a.remove()
            except Exception:
                pass
        self.current_markers.clear()

    # ---------- Baselines ----------
    def _remove_baseline(self, which: int):
        b = self.baselines[which]
        for key in ("line", "fill", "label"):
            if b.get(key) is not None:
                try:
                    b[key].remove()
                except Exception:
                    pass
                b[key] = None
        b["area"] = None
        b["slope"] = None
        b["idxs"] = None

    def _render_baseline(self, which, idx1, idx2, vv, ib, ii_used, area, slope):
        self._remove_baseline(which)

        mode = self.area_mode.get()
        if mode == "above":
            where = (ii_used >= ib)
            fill = self.ax.fill_between(vv, ib, ii_used, where=where, interpolate=True, alpha=0.22, zorder=5)
        else:
            fill = self.ax.fill_between(vv, ib, ii_used, interpolate=True, alpha=0.16, zorder=5)

        x1, y1 = float(self.v[idx1]), float(self.i[idx1])
        x2, y2 = float(self.v[idx2]), float(self.i[idx2])
        (line,) = self.ax.plot([x1, x2], [y1, y2], linewidth=3.0, zorder=60)

        xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        label = self.ax.text(
            xm, ym,
            f"BL{which+1}  Area={area:g} A·V   Slope={slope:g} A/V",
            fontsize=10, zorder=70
        )

        self.baselines[which].update({
            "line": line, "fill": fill, "label": label,
            "area": float(area), "slope": float(slope), "idxs": (idx1, idx2)
        })

    def _render_ratios(self):
        if self.ratio_label is not None:
            try:
                self.ratio_label.remove()
            except Exception:
                pass
            self.ratio_label = None

        a1, a2 = self.baselines[0]["area"], self.baselines[1]["area"]
        s1, s2 = self.baselines[0]["slope"], self.baselines[1]["slope"]
        area_ratio = np.inf if a1 == 0 else (a2 / a1)
        slope_ratio = np.inf if s1 == 0 else (s2 / s1)

        txt = f"Area2/Area1 = {area_ratio:g}    |    Slope2/Slope1 = {slope_ratio:g}"
        self.ratio_label = self.ax.text(
            0.02, 0.98, txt, transform=self.ax.transAxes,
            va="top", fontsize=11, zorder=200
        )

    def _build_status_two_done(self) -> str:
        a1, a2 = self.baselines[0]["area"], self.baselines[1]["area"]
        s1, s2 = self.baselines[0]["slope"], self.baselines[1]["slope"]
        area_ratio = np.inf if a1 == 0 else (a2 / a1)
        slope_ratio = np.inf if s1 == 0 else (s2 / s1)
        return f"Done — Area1={a1:g}, Area2={a2:g}, Area2/Area1={area_ratio:g} | Slope1={s1:g}, Slope2={s2:g}, Slope2/Slope1={slope_ratio:g}"

    # ---------- User Lines ----------
    def _render_user_line(self, idx1, idx2):
        x1, y1 = float(self.v[idx1]), float(self.i[idx1])
        x2, y2 = float(self.v[idx2]), float(self.i[idx2])

        self.user_line_counter += 1
        name = f"UL{self.user_line_counter}"

        (line,) = self.ax.plot([x1, x2], [y1, y2], linewidth=2.2, linestyle="--", zorder=80)
        xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        label = self.ax.text(xm, ym, name, fontsize=10, zorder=90)

        self.user_lines.append({"line": line, "label": label, "p": ((x1, y1), (x2, y2)), "name": name})

    def undo_last_user_line(self):
        if not self.user_lines:
            self._set_status("삭제할 user line이 없습니다.")
            return
        last = self.user_lines.pop()
        for key in ("line", "label"):
            try:
                last[key].remove()
            except Exception:
                pass
        self._set_status(f"마지막 user line 삭제됨: {last.get('name')}")

    # ---------- Reset ----------
    def reset_all(self, keep_plot=False):
        self._clear_current_selection()
        self._remove_baseline(0)
        self._remove_baseline(1)

        if self.ratio_label is not None:
            try:
                self.ratio_label.remove()
            except Exception:
                pass
            self.ratio_label = None

        # remove user lines too
        for ul in self.user_lines:
            for key in ("line", "label"):
                try:
                    ul[key].remove()
                except Exception:
                    pass
        self.user_lines.clear()
        self.user_line_counter = 0

        if not keep_plot:
            self.ax.clear()
            self.ax.set_title("Load CSV to plot I-V")
            self.ax.set_xlabel("Voltage (V)")
            self.ax.set_ylabel("Current (A)")


if __name__ == "__main__":
    app = ModernIVApp()
    app.mainloop()

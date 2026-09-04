"""Двухэтапный графический интерфейс анализатора тензограмм.

Этап 1 — "Файл и участок":
    * загрузка файла тензограммы, отображение записи целиком;
    * ввод параметров участка (время события, окно до/после);
    * кнопка "Выбрать" — показывает выбранный участок в увеличенном
      масштабе для визуальной проверки.

Этап 2 — "Анализ":
    * настройка параметров расчёта (длительность "нулевого" участка,
      частота среза первой формы, ширина окрестности поиска динамики);
    * кнопка "Анализ" — расчёт и отображение результатов с графиком.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import PIL._tkinter_finder  # noqa: F401  (нужен PyInstaller для сборки exe)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import core


class TensogramApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Анализатор тензограмм")
        self.geometry("1100x780")
        self.minsize(900, 650)

        self.t_full: np.ndarray | None = None
        self.v_full: np.ndarray | None = None
        self.file_path: str | None = None

        self.ts_segment: np.ndarray | None = None
        self.vs_segment: np.ndarray | None = None

        self._build_widgets()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        # -- top: plot area (shared across stages) --------------------
        plot_frame = ttk.Frame(container)
        plot_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        self.figure = Figure(figsize=(9, 4.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # -- bottom: notebook with stage 1 / stage 2 controls ---------
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=8)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Файл и участок")
        self.notebook.add(self.tab2, text="2. Анализ")
        self.notebook.tab(self.tab2, state="disabled")

        self._build_stage1(self.tab1)
        self._build_stage2(self.tab2)

        self.status_var = tk.StringVar(value="Откройте файл тензограммы.")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_stage1(self, parent: ttk.Frame) -> None:
        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=4)
        ttk.Button(row0, text="Открыть файл...", command=self.on_open_file).pack(side=tk.LEFT)
        self.file_label_var = tk.StringVar(value="Файл не выбран")
        ttk.Label(row0, textvariable=self.file_label_var).pack(side=tk.LEFT, padx=10)

        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="Время события, с:").grid(row=0, column=0, sticky="w", padx=4)
        self.event_time_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.event_time_var, width=12).grid(row=0, column=1, padx=4)

        ttk.Label(row1, text="Окно до, с:").grid(row=0, column=2, sticky="w", padx=4)
        self.before_var = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self.before_var, width=8).grid(row=0, column=3, padx=4)

        ttk.Label(row1, text="Окно после, с:").grid(row=0, column=4, sticky="w", padx=4)
        self.after_var = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self.after_var, width=8).grid(row=0, column=5, padx=4)

        ttk.Button(row1, text="Выбрать", command=self.on_select_segment).grid(
            row=0, column=6, padx=12
        )

    def _build_stage2(self, parent: ttk.Frame) -> None:
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="Длительность нулевого участка, с:").grid(
            row=0, column=0, sticky="w", padx=4
        )
        self.baseline_dur_var = tk.StringVar(value="2")
        ttk.Entry(row1, textvariable=self.baseline_dur_var, width=8).grid(row=0, column=1, padx=4)

        ttk.Label(row1, text="Частота среза 1-й формы, Гц:").grid(
            row=0, column=2, sticky="w", padx=4
        )
        self.cutoff_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.cutoff_var, width=10).grid(row=0, column=3, padx=4)
        ttk.Button(row1, text="Авто", command=self.on_auto_cutoff).grid(row=0, column=4, padx=4)

        ttk.Label(row1, text="Окрестность поиска динамики, с:").grid(
            row=0, column=5, sticky="w", padx=4
        )
        self.neighborhood_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.neighborhood_var, width=8).grid(row=0, column=6, padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=4)
        ttk.Button(row2, text="Анализ", command=self.on_analyze).pack(side=tk.LEFT)

        results = ttk.LabelFrame(parent, text="Результаты")
        results.pack(fill=tk.X, pady=6)

        self.result_static_var = tk.StringVar(value="Макс. статическое значение: —")
        self.result_dynamic_var = tk.StringVar(value="Макс. динамическое значение: —")
        self.result_kd_var = tk.StringVar(value="Коэффициент динамичности: —")

        ttk.Label(results, textvariable=self.result_static_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=6, pady=2
        )
        ttk.Label(results, textvariable=self.result_dynamic_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=6, pady=2
        )
        ttk.Label(results, textvariable=self.result_kd_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=6, pady=2
        )

    # ------------------------------------------------------------------
    # Stage 1 handlers
    # ------------------------------------------------------------------
    def on_open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл тензограммы",
            filetypes=[
                ("Текстовые файлы", "*.txt *.gru *.dat *.csv"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        try:
            t, v = core.load_tensogram(path)
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))
            return
        except OSError as exc:
            messagebox.showerror("Ошибка загрузки", f"Не удалось открыть файл:\n{exc}")
            return

        self.t_full = t
        self.v_full = v
        self.file_path = path
        self.file_label_var.set(f"{path}  ({len(t)} точек, {t[0]:.2f}..{t[-1]:.2f} с)")
        self.notebook.tab(self.tab2, state="disabled")
        self.ts_segment = None
        self.vs_segment = None

        default_event = t[len(t) // 2]
        self.event_time_var.set(f"{default_event:.2f}")

        self._plot_full_record()
        self.status_var.set(
            "Файл загружен. Укажите время события и окна до/после, затем нажмите 'Выбрать'."
        )

    def _reset_single_axis(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def _plot_full_record(self) -> None:
        assert self.t_full is not None and self.v_full is not None
        self._reset_single_axis()
        self.ax.plot(self.t_full, self.v_full, linewidth=0.6, color="tab:blue")
        self.ax.set_title("Тензограмма (вся запись)")
        self.ax.set_xlabel("Время, с")
        self.ax.set_ylabel("Величина")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _parse_stage1_inputs(self) -> tuple[float, float, float] | None:
        try:
            event_time = float(self.event_time_var.get().replace(",", "."))
            before = float(self.before_var.get().replace(",", "."))
            after = float(self.after_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror(
                "Неверные параметры", "Время события и окна должны быть числами."
            )
            return None
        return event_time, before, after

    def on_select_segment(self) -> None:
        if self.t_full is None:
            messagebox.showwarning("Нет данных", "Сначала откройте файл тензограммы.")
            return
        params = self._parse_stage1_inputs()
        if params is None:
            return
        event_time, before, after = params

        try:
            ts, vs = core.extract_segment(self.t_full, self.v_full, event_time, before, after)
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка выбора участка", str(exc))
            return

        self.ts_segment = ts
        self.vs_segment = vs

        self._plot_segment_preview(event_time)

        self.notebook.tab(self.tab2, state="normal")
        self.notebook.select(self.tab2)
        self.cutoff_var.set("")
        self.neighborhood_var.set("")
        self.result_static_var.set("Макс. статическое значение: —")
        self.result_dynamic_var.set("Макс. динамическое значение: —")
        self.result_kd_var.set("Коэффициент динамичности: —")
        self.status_var.set(
            "Участок выбран. Настройте параметры расчёта и нажмите 'Анализ'."
        )

    def _plot_segment_preview(self, event_time: float) -> None:
        assert self.ts_segment is not None
        self._reset_single_axis()
        self.ax.plot(self.ts_segment, self.vs_segment, linewidth=1.0, color="tab:blue")
        self.ax.axvline(event_time, color="tab:red", linestyle="--", linewidth=1, label="событие")
        self.ax.set_title("Выбранный участок тензограммы")
        self.ax.set_xlabel("Время, с")
        self.ax.set_ylabel("Величина")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Stage 2 handlers
    # ------------------------------------------------------------------
    def _parse_stage2_inputs(self) -> tuple[float, float | None, float | None] | None:
        try:
            baseline_dur = float(self.baseline_dur_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Неверные параметры", "Длительность нулевого участка — число.")
            return None

        cutoff_txt = self.cutoff_var.get().strip()
        cutoff_hz = None
        if cutoff_txt:
            try:
                cutoff_hz = float(cutoff_txt.replace(",", "."))
            except ValueError:
                messagebox.showerror("Неверные параметры", "Частота среза должна быть числом.")
                return None

        neigh_txt = self.neighborhood_var.get().strip()
        neighborhood_s = None
        if neigh_txt:
            try:
                neighborhood_s = float(neigh_txt.replace(",", "."))
            except ValueError:
                messagebox.showerror("Неверные параметры", "Окрестность должна быть числом.")
                return None

        return baseline_dur, cutoff_hz, neighborhood_s

    def on_auto_cutoff(self) -> None:
        if self.ts_segment is None:
            return
        params = self._parse_stage2_inputs()
        if params is None:
            return
        baseline_dur, _cutoff_hz, _neigh = params
        try:
            baseline = core.segment_baseline(self.ts_segment, self.vs_segment, baseline_dur)
            vz = self.vs_segment - baseline
            fs = core.sampling_rate(self.ts_segment)
            duration = self.ts_segment[-1] - self.ts_segment[0]
            f1 = core.detect_first_mode_frequency(vz, fs, duration)
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        self.cutoff_var.set(f"{f1 * 1.6:.4f}")
        self.neighborhood_var.set(f"{max(1.0 / f1, 1.0):.3f}")
        self.status_var.set(
            f"Автоопределение: частота первой формы ~{f1:.4f} Гц (период ~{1/f1:.2f} с)."
        )

    def on_analyze(self) -> None:
        if self.ts_segment is None:
            messagebox.showwarning("Нет участка", "Сначала выберите участок на первом этапе.")
            return
        params = self._parse_stage2_inputs()
        if params is None:
            return
        baseline_dur, cutoff_hz, neighborhood_s = params

        try:
            result = core.analyze_segment(
                self.ts_segment,
                self.vs_segment,
                baseline_dur=baseline_dur,
                cutoff_hz=cutoff_hz,
                neighborhood_s=neighborhood_s,
            )
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка анализа", str(exc))
            return

        self.result_static_var.set(
            f"Макс. статическое значение: {result.static_max:.4f}  "
            f"(t = {result.t_static_max:.3f} с)"
        )
        self.result_dynamic_var.set(
            f"Макс. динамическое значение: {result.dynamic_max:.4f}  "
            f"(t = {result.t_dynamic_max:.3f} с)"
        )
        self.result_kd_var.set(
            f"Коэффициент динамичности: {result.dynamic_coefficient:.4f}"
        )
        if not self.cutoff_var.get().strip():
            self.cutoff_var.set(f"{result.cutoff_hz:.4f}")
        if not self.neighborhood_var.get().strip():
            self.neighborhood_var.set(f"{result.neighborhood_s:.3f}")

        self._plot_result(result)
        self.status_var.set(
            f"Расчёт выполнен. Ноль участка = {result.baseline:.4f}; "
            f"частота 1-й формы = {result.first_mode_freq:.4f} Гц "
            f"(период {result.first_mode_period:.2f} с)."
        )

    def _plot_result(self, result: core.AnalysisResult) -> None:
        self.figure.clear()
        ax_overview = self.figure.add_subplot(211)
        ax_zoom = self.figure.add_subplot(212)
        self.ax = ax_overview

        def _draw(ax, ts, vz, static_curve, title):
            ax.plot(ts, vz, linewidth=0.8, color="tab:blue", label="сигнал (динамика, за вычетом нуля)")
            ax.plot(
                ts, static_curve, linewidth=1.8, color="tab:orange",
                label="1-я форма (квазистатика)",
            )
            ax.axvspan(
                result.t_static_max - result.neighborhood_s,
                result.t_static_max + result.neighborhood_s,
                color="tab:green", alpha=0.12, label="окрестность поиска динамики",
            )
            ax.plot(
                [result.t_static_max], [result.static_max], "o", color="tab:orange",
                markersize=9, zorder=5, label="макс. статическое",
            )
            ax.plot(
                [result.t_dynamic_max], [result.dynamic_max], "^", color="tab:red",
                markersize=10, zorder=5, label="макс. динамическое",
            )
            ax.set_title(title)
            ax.set_xlabel("Время, с")
            ax.set_ylabel("Величина (от нуля участка)")
            ax.grid(True, alpha=0.3)

        # Верхний график: весь участок целиком.
        _draw(ax_overview, result.ts, result.vz, result.static_curve, "Весь участок: динамика и 1-я форма (статика)")
        ax_overview.legend(loc="best", fontsize=8)

        # Нижний график: увеличенный вид вокруг статического/динамического
        # максимума — для визуальной проверки корректности значений.
        half_span = max(result.neighborhood_s * 2.5, result.first_mode_period * 0.75, 1.0)
        t_center = result.t_static_max
        mask = np.abs(result.ts - t_center) <= half_span
        if mask.sum() < 4:
            mask = np.ones_like(result.ts, dtype=bool)
        _draw(
            ax_zoom,
            result.ts[mask],
            result.vz[mask],
            result.static_curve[mask],
            "Увеличенный вид: проверка максимумов",
        )

        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    app = TensogramApp()
    app.mainloop()


if __name__ == "__main__":
    main()

"""Двухэтапный графический интерфейс анализатора тензограмм.

Этап 1 — "Файл и участок":
    * загрузка файла тензограммы, отображение записи целиком (масштаб
      и панорамирование — стандартной панелью инструментов matplotlib);
    * выбор участка — вручную (время события + окна до/после) или
      протяжкой мыши прямо по графику записи (чекбокс "Выбор мышью");
    * кнопка "Выбрать" — показывает выбранный участок в увеличенном
      масштабе для визуальной проверки.

Этап 2 — "Анализ":
    * название точки расчёта и комментарий — для идентификации;
    * настройка параметров расчёта (длительность "нулевого" участка,
      частота среза первой формы, ширина окрестности поиска динамики);
    * настройка спектрального анализа (задержка от максимума и
      длительность окна) — для определения частот собственных
      колебаний по свободным колебаниям после прохода нагрузки;
    * кнопка "Анализ" — расчёт и отображение результатов на сетке из
      4 графиков (обзор, увеличенный вид, окно и спектр);
    * кнопка "Сохранить отчёт" — сохраняет PDF-отчёт, CSV со значениями
      и рядами данных и PNG-изображения графиков в подпапку с названием
      точки расчёта.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import PIL._tkinter_finder  # noqa: F401  (нужен PyInstaller для сборки exe)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from . import core, plotting, report


class TensogramApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Анализатор тензограмм")
        self.geometry("1150x860")
        self.minsize(950, 700)

        self.t_full: np.ndarray | None = None
        self.v_full: np.ndarray | None = None
        self.file_path: str | None = None

        self.ts_segment: np.ndarray | None = None
        self.vs_segment: np.ndarray | None = None

        self.last_result: core.AnalysisResult | None = None
        self.last_spectral: core.SpectralResult | None = None

        self.span_selector: SpanSelector | None = None
        self._selection_patch = None
        self._stage1_showing_full = False

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

        self.figure = Figure(figsize=(9, 5.5), dpi=100)
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

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.status_var = tk.StringVar(value="Откройте файл тензограммы.")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_stage1(self, parent: ttk.Frame) -> None:
        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=4)
        ttk.Button(row0, text="Открыть файл...", command=self.on_open_file).pack(side=tk.LEFT)
        self.file_label_var = tk.StringVar(value="Файл не выбран")
        ttk.Label(row0, textvariable=self.file_label_var).pack(side=tk.LEFT, padx=10)

        row_nodata = ttk.Frame(parent)
        row_nodata.pack(fill=tk.X, pady=2)
        self.exclude_no_data_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_nodata,
            text="Не отображать участки без записи (значение ≡ 0) длительностью от, с:",
            variable=self.exclude_no_data_var,
            command=self.on_toggle_exclude_no_data,
        ).pack(side=tk.LEFT)
        self.exclude_no_data_min_dur_var = tk.StringVar(value=str(core.DEFAULT_NO_DATA_MIN_DURATION_S))
        ttk.Entry(row_nodata, textvariable=self.exclude_no_data_min_dur_var, width=6).pack(
            side=tk.LEFT, padx=4
        )
        self.exclude_no_data_min_dur_var.trace_add("write", self._on_exclude_no_data_param_changed)
        ttk.Label(
            row_nodata,
            text="(только для отображения; на данные и расчёт не влияет)",
        ).pack(side=tk.LEFT, padx=6)

        row_mouse = ttk.Frame(parent)
        row_mouse.pack(fill=tk.X, pady=2)
        self.mouse_select_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_mouse,
            text="Выбор мышью на графике (протянуть по горизонтали; для масштабирования/"
            "панорамирования сначала снимите этот флажок)",
            variable=self.mouse_select_var,
            command=self.on_toggle_mouse_select,
        ).pack(side=tk.LEFT)

        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="а) время события, с:").grid(row=0, column=0, sticky="w", padx=4)
        self.event_time_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.event_time_var, width=12).grid(row=0, column=1, padx=4)

        ttk.Label(row1, text="окно до, с:").grid(row=0, column=2, sticky="w", padx=4)
        self.before_var = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self.before_var, width=8).grid(row=0, column=3, padx=4)

        ttk.Label(row1, text="окно после, с:").grid(row=0, column=4, sticky="w", padx=4)
        self.after_var = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self.after_var, width=8).grid(row=0, column=5, padx=4)

        for var in (self.event_time_var, self.before_var, self.after_var):
            var.trace_add("write", self._on_convenience_changed)

        row_range = ttk.Frame(parent)
        row_range.pack(fill=tk.X, pady=4)

        ttk.Label(row_range, text="б) либо протяните мышью по графику — диапазон, с: от").grid(
            row=0, column=0, sticky="w", padx=4
        )
        self.range_from_var = tk.StringVar(value="")
        ttk.Entry(row_range, textvariable=self.range_from_var, width=12).grid(
            row=0, column=1, padx=4
        )
        ttk.Label(row_range, text="до").grid(row=0, column=2, sticky="w", padx=4)
        self.range_to_var = tk.StringVar(value="")
        ttk.Entry(row_range, textvariable=self.range_to_var, width=12).grid(
            row=0, column=3, padx=4
        )

        for var in (self.range_from_var, self.range_to_var):
            var.trace_add("write", self._on_range_changed)

        ttk.Button(row_range, text="Выбрать", command=self.on_select_segment).grid(
            row=0, column=4, padx=12
        )
        ttk.Label(
            row_range,
            text="— именно этот диапазон используется далее для анализа",
        ).grid(row=0, column=5, sticky="w", padx=4)

    def _build_stage2(self, parent: ttk.Frame) -> None:
        row_id = ttk.Frame(parent)
        row_id.pack(fill=tk.X, pady=4)
        ttk.Label(row_id, text="Название точки расчёта:").grid(row=0, column=0, sticky="nw", padx=4)
        self.point_name_var = tk.StringVar(value="")
        ttk.Entry(row_id, textvariable=self.point_name_var, width=30).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Label(row_id, text="Комментарий:").grid(row=0, column=2, sticky="nw", padx=4)
        self.comment_text = tk.Text(row_id, width=40, height=2)
        self.comment_text.grid(row=0, column=3, sticky="w", padx=4)

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
        self.neighborhood_var = tk.StringVar(value="1")
        ttk.Entry(row1, textvariable=self.neighborhood_var, width=8).grid(row=0, column=6, padx=4)

        row_spec = ttk.Frame(parent)
        row_spec.pack(fill=tk.X, pady=4)
        ttk.Label(row_spec, text="Спектральный анализ — задержка от максимума, с:").grid(
            row=0, column=0, sticky="w", padx=4
        )
        self.spec_delay_var = tk.StringVar(value=str(core.DEFAULT_SPECTRAL_DELAY_S))
        ttk.Entry(row_spec, textvariable=self.spec_delay_var, width=8).grid(row=0, column=1, padx=4)

        ttk.Label(row_spec, text="Длительность окна спектра, с:").grid(
            row=0, column=2, sticky="w", padx=4
        )
        self.spec_window_var = tk.StringVar(value=str(core.DEFAULT_SPECTRAL_WINDOW_S))
        ttk.Entry(row_spec, textvariable=self.spec_window_var, width=8).grid(row=0, column=3, padx=4)

        ttk.Button(row_spec, text="Обновить", command=self.on_refresh_spectral).grid(
            row=0, column=4, padx=4
        )

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=4)
        ttk.Button(row2, text="Анализ", command=self.on_analyze).pack(side=tk.LEFT)
        self.save_report_btn = ttk.Button(
            row2, text="Сохранить отчёт...", command=self.on_save_report, state="disabled"
        )
        self.save_report_btn.pack(side=tk.LEFT, padx=10)

        results = ttk.LabelFrame(parent, text="Результаты")
        results.pack(fill=tk.X, pady=6)

        self.result_static_var = tk.StringVar(value="Макс. статическое значение: —")
        self.result_dynamic_var = tk.StringVar(value="Макс. динамическое значение: —")
        self.result_kd_var = tk.StringVar(value="Коэффициент динамичности: —")
        self.result_spectral_var = tk.StringVar(value="Спектральный анализ: —")

        for var in (
            self.result_static_var,
            self.result_dynamic_var,
            self.result_kd_var,
            self.result_spectral_var,
        ):
            ttk.Label(results, textvariable=var, font=("TkDefaultFont", 10, "bold")).pack(
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
        self.last_result = None
        self.last_spectral = None
        self.save_report_btn.config(state="disabled")

        self._plot_full_record()
        self._setup_span_selector()

        default_event = t[len(t) // 2]
        self.event_time_var.set(f"{default_event:.2f}")  # triggers range fields via trace

        self.status_var.set(
            "Файл загружен. Укажите время события и окна до/после, либо включите "
            "'Выбор мышью' и протяните участок на графике — предполагаемый диапазон "
            "подсвечивается на графике. Нажмите 'Выбрать', чтобы применить его к анализу."
        )

    def _reset_single_axis(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._selection_patch = None

    def _get_display_v_full(self) -> np.ndarray:
        assert self.v_full is not None
        if not self.exclude_no_data_var.get():
            return self.v_full
        try:
            min_dur = float(self.exclude_no_data_min_dur_var.get().replace(",", "."))
        except ValueError:
            min_dur = core.DEFAULT_NO_DATA_MIN_DURATION_S
        return core.mask_no_data_runs(self.t_full, self.v_full, min_duration_s=min_dur)

    def _plot_full_record(self) -> None:
        assert self.t_full is not None and self.v_full is not None
        self._reset_single_axis()
        plotting.draw_full_record(self.ax, self.t_full, self._get_display_v_full())
        self._stage1_showing_full = True
        self._update_stage1_highlight()
        self.canvas.draw_idle()

    def on_toggle_exclude_no_data(self) -> None:
        if self.t_full is not None and self._stage1_showing_full:
            self._plot_full_record()
            self._setup_span_selector()

    def _on_exclude_no_data_param_changed(self, *_args) -> None:
        if self.exclude_no_data_var.get():
            self.on_toggle_exclude_no_data()

    def _setup_span_selector(self) -> None:
        self.span_selector = SpanSelector(
            self.ax,
            self._on_span_select,
            direction="horizontal",
            useblit=True,
            interactive=False,
            props=dict(alpha=0.2, facecolor="tab:red"),
        )
        self.span_selector.active = self.mouse_select_var.get()

    def _on_tab_changed(self, _event=None) -> None:
        if self.t_full is None:
            return
        if self.notebook.index(self.notebook.select()) == 0 and not self._stage1_showing_full:
            self._plot_full_record()
            self._setup_span_selector()

    def on_toggle_mouse_select(self) -> None:
        if self.span_selector is not None:
            self.span_selector.active = self.mouse_select_var.get()
        if self.mouse_select_var.get():
            self.status_var.set(
                "Режим выбора мышью включён: протяните область на графике полной записи."
            )
        else:
            self.status_var.set(
                "Режим выбора мышью выключен: доступны масштабирование и панорамирование."
            )

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        if self.t_full is None or xmax <= xmin:
            return
        event_time = 0.5 * (xmin + xmax)
        before = event_time - xmin
        after = xmax - event_time
        self.event_time_var.set(f"{event_time:.3f}")
        self.before_var.set(f"{before:.3f}")
        self.after_var.set(f"{after:.3f}")
        # Точные границы (без промежуточных округлений через событие/окна) —
        # они являются итоговым диапазоном для анализа.
        self.range_from_var.set(f"{xmin:.3f}")
        self.range_to_var.set(f"{xmax:.3f}")
        self.status_var.set(
            f"Диапазон выбран мышью: {xmin:.3f} .. {xmax:.3f} с. "
            "Нажмите 'Выбрать', чтобы применить его к анализу."
        )

    def _on_convenience_changed(self, *_args) -> None:
        """Время события + окна до/после -> поля диапазона (одно направление)."""
        try:
            event_time = float(self.event_time_var.get().replace(",", "."))
            before = float(self.before_var.get().replace(",", "."))
            after = float(self.after_var.get().replace(",", "."))
        except ValueError:
            return
        self.range_from_var.set(f"{event_time - before:.3f}")
        self.range_to_var.set(f"{event_time + after:.3f}")

    def _on_range_changed(self, *_args) -> None:
        self._update_stage1_highlight()

    def _update_stage1_highlight(self) -> None:
        """Подсвечивает на графике полной записи текущий (ещё не применённый)
        диапазон, не заменяя сам график — данные не 'обрезаются' до нажатия
        кнопки 'Выбрать'."""
        if not self._stage1_showing_full or self.t_full is None:
            return
        if self._selection_patch is not None:
            try:
                self._selection_patch.remove()
            except (ValueError, NotImplementedError):
                pass
            self._selection_patch = None
        try:
            t_from = float(self.range_from_var.get().replace(",", "."))
            t_to = float(self.range_to_var.get().replace(",", "."))
        except ValueError:
            self.canvas.draw_idle()
            return
        if t_to > t_from:
            self._selection_patch = self.ax.axvspan(
                t_from, t_to, color="tab:red", alpha=0.15, zorder=0
            )
        self.canvas.draw_idle()

    def _parse_range_inputs(self) -> tuple[float, float] | None:
        try:
            t_from = float(self.range_from_var.get().replace(",", "."))
            t_to = float(self.range_to_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror(
                "Неверные параметры", "Границы диапазона ('от', 'до') должны быть числами."
            )
            return None
        if t_to <= t_from:
            messagebox.showerror(
                "Неверные параметры", "Правая граница диапазона должна быть больше левой."
            )
            return None
        return t_from, t_to

    def on_select_segment(self) -> None:
        if self.t_full is None:
            messagebox.showwarning("Нет данных", "Сначала откройте файл тензограммы.")
            return
        params = self._parse_range_inputs()
        if params is None:
            return
        t_from, t_to = params

        try:
            ts, vs = core.extract_window(self.t_full, self.v_full, t_from, t_to)
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка выбора участка", str(exc))
            return

        self.ts_segment = ts
        self.vs_segment = vs
        self.last_result = None
        self.last_spectral = None
        self.save_report_btn.config(state="disabled")

        self._plot_segment_preview()

        self.notebook.tab(self.tab2, state="normal")
        self.notebook.select(self.tab2)
        self.cutoff_var.set("")
        self.neighborhood_var.set("1")
        self.result_static_var.set("Макс. статическое значение: —")
        self.result_dynamic_var.set("Макс. динамическое значение: —")
        self.result_kd_var.set("Коэффициент динамичности: —")
        self.result_spectral_var.set("Спектральный анализ: —")
        self.status_var.set(
            f"Участок [{t_from:.3f} .. {t_to:.3f}] с выбран. Настройте параметры "
            "расчёта и нажмите 'Анализ'."
        )

    def _plot_segment_preview(self) -> None:
        assert self.ts_segment is not None
        self._reset_single_axis()
        self._stage1_showing_full = False
        plotting.draw_segment_preview(self.ax, self.ts_segment, self.vs_segment)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Stage 2 handlers
    # ------------------------------------------------------------------
    def _parse_stage2_inputs(
        self,
    ) -> tuple[float, float | None, float | None, float, float] | None:
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

        spec_params = self._parse_spectral_inputs()
        if spec_params is None:
            return None
        spec_delay, spec_window = spec_params

        return baseline_dur, cutoff_hz, neighborhood_s, spec_delay, spec_window

    def _parse_spectral_inputs(self) -> tuple[float, float] | None:
        try:
            spec_delay = float(self.spec_delay_var.get().replace(",", "."))
            spec_window = float(self.spec_window_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror(
                "Неверные параметры",
                "Параметры спектрального анализа (задержка, длительность окна) должны быть числами.",
            )
            return None
        return spec_delay, spec_window

    def on_auto_cutoff(self) -> None:
        if self.ts_segment is None:
            return
        params = self._parse_stage2_inputs()
        if params is None:
            return
        baseline_dur, _cutoff_hz, _neigh, _delay, _window = params
        try:
            baseline = core.segment_baseline(self.ts_segment, self.vs_segment, baseline_dur)
            vz = self.vs_segment - baseline
            fs = core.sampling_rate(self.ts_segment)
            duration = self.ts_segment[-1] - self.ts_segment[0]
            f1 = core.detect_first_mode_frequency(vz, fs, duration)
            cutoff = core.suggest_cutoff_frequency(vz, fs, duration)
        except core.TensogramError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        self.cutoff_var.set(f"{cutoff:.4f}")
        self.status_var.set(
            f"Автоопределение: частота первой формы ~{f1:.4f} Гц (период ~{1/f1:.2f} с), "
            f"частота среза ~{cutoff:.4f} Гц."
        )

    def on_analyze(self) -> None:
        if self.ts_segment is None:
            messagebox.showwarning("Нет участка", "Сначала выберите участок на первом этапе.")
            return
        params = self._parse_stage2_inputs()
        if params is None:
            return
        baseline_dur, cutoff_hz, neighborhood_s, spec_delay, spec_window = params

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

        spectral = self._compute_spectral(result, spec_delay, spec_window)

        self.last_result = result
        self.last_spectral = spectral
        self.save_report_btn.config(state="normal")

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
        self._update_spectral_result_label(spectral)

        if not self.cutoff_var.get().strip():
            self.cutoff_var.set(f"{result.cutoff_hz:.4f}")
        if not self.neighborhood_var.get().strip():
            self.neighborhood_var.set(f"{result.neighborhood_s:.3f}")

        plotting.draw_results_grid(self.figure, result, spectral)
        self.canvas.draw_idle()
        self.status_var.set(
            f"Расчёт выполнен. Ноль участка = {result.baseline:.4f}; "
            f"частота 1-й формы = {result.first_mode_freq:.4f} Гц "
            f"(период {result.first_mode_period:.2f} с)."
        )

    def _compute_spectral(
        self, result: core.AnalysisResult, spec_delay: float, spec_window: float
    ) -> core.SpectralResult | None:
        try:
            return core.spectral_analysis(
                self.t_full,
                self.v_full,
                result.baseline,
                result.t_static_max,
                delay_s=spec_delay,
                window_s=spec_window,
            )
        except core.TensogramError as exc:
            messagebox.showwarning(
                "Спектральный анализ",
                f"Не удалось выполнить спектральный анализ: {exc}",
            )
            return None

    def _update_spectral_result_label(self, spectral: core.SpectralResult | None) -> None:
        if spectral is not None and len(spectral.peak_freqs) > 0:
            freqs_txt = ", ".join(f"{f:.3f}" for f in spectral.peak_freqs)
            self.result_spectral_var.set(f"Спектральный анализ — частоты, Гц: {freqs_txt}")
        else:
            self.result_spectral_var.set("Спектральный анализ: не выполнен")

    def on_refresh_spectral(self) -> None:
        if self.last_result is None:
            messagebox.showwarning(
                "Нет результата", "Сначала выполните основной анализ ('Анализ')."
            )
            return
        spec_params = self._parse_spectral_inputs()
        if spec_params is None:
            return
        spec_delay, spec_window = spec_params

        spectral = self._compute_spectral(self.last_result, spec_delay, spec_window)
        self.last_spectral = spectral
        self._update_spectral_result_label(spectral)

        plotting.draw_results_grid(self.figure, self.last_result, spectral)
        self.canvas.draw_idle()

        if spectral is not None:
            self.status_var.set(
                f"Спектр пересчитан для окна {spectral.t_start:.2f}..{spectral.t_end:.2f} с."
            )

    def on_save_report(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("Нет результата", "Сначала выполните анализ.")
            return

        point_name = self.point_name_var.get().strip()
        comment = self.comment_text.get("1.0", "end").strip()

        initial_dir = os.path.dirname(self.file_path) if self.file_path else None

        base_dir = filedialog.askdirectory(
            title="Выберите папку, в которой будет создана подпапка с отчётом",
            initialdir=initial_dir,
        )
        if not base_dir:
            return

        try:
            out_dir = report.save_report(
                base_dir, point_name, comment, self.last_result, self.last_spectral
            )
        except OSError as exc:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить отчёт:\n{exc}")
            return

        self.status_var.set(f"Отчёт сохранён в: {out_dir}")
        messagebox.showinfo("Отчёт сохранён", f"Отчёт сохранён в папку:\n{out_dir}")


def main() -> None:
    app = TensogramApp()
    app.mainloop()


if __name__ == "__main__":
    main()

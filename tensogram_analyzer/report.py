"""Сохранение отчёта по расчёту: PDF (сводка + графики), CSV (значения и
временные ряды) и отдельные изображения графиков (PNG)."""

from __future__ import annotations

import csv
import datetime
import os
import re

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import core, plotting


def sanitize_dirname(name: str) -> str:
    name = name.strip() or "точка_расчёта"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120] if name else "точка_расчёта"


def _summary_rows(
    point_name: str,
    comment: str,
    result: core.AnalysisResult,
    spectral: core.SpectralResult | None,
) -> list[tuple[str, str]]:
    rows = [
        ("Название точки", point_name),
        ("Комментарий", comment),
        ("Дата/время формирования отчёта", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("", ""),
        ("Ноль участка (baseline)", f"{result.baseline:.6g}"),
        ("Частота первой формы, Гц", f"{result.first_mode_freq:.6g}"),
        ("Период первой формы, с", f"{result.first_mode_period:.6g}"),
        ("Частота среза фильтра, Гц", f"{result.cutoff_hz:.6g}"),
        ("Окрестность поиска динамики, с", f"{result.neighborhood_s:.6g}"),
        ("", ""),
        ("Максимальное статическое значение", f"{result.static_max:.6g}"),
        ("Время максимального статического значения, с", f"{result.t_static_max:.6g}"),
        ("Максимальное динамическое значение", f"{result.dynamic_max:.6g}"),
        ("Время максимального динамического значения, с", f"{result.t_dynamic_max:.6g}"),
        ("Коэффициент динамичности", f"{result.dynamic_coefficient:.6g}"),
    ]
    if spectral is not None:
        rows += [
            ("", ""),
            ("Задержка от максимума для спектра, с", f"{spectral.delay_s:.6g}"),
            ("Длительность окна спектра, с", f"{spectral.window_s:.6g}"),
            ("Окно спектрального анализа, с", f"{spectral.t_start:.3f} .. {spectral.t_end:.3f}"),
        ]
        for i, (f, a) in enumerate(zip(spectral.peak_freqs, spectral.peak_amps), start=1):
            rows.append((f"Пик спектра #{i}, Гц (амплитуда)", f"{f:.6g} ({a:.4g})"))
    return rows


def write_values_csv(
    path: str,
    point_name: str,
    comment: str,
    result: core.AnalysisResult,
    spectral: core.SpectralResult | None,
) -> None:
    rows = _summary_rows(point_name, comment, result, spectral)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Параметр", "Значение"])
        writer.writerows(rows)


def write_segment_csv(path: str, result: core.AnalysisResult) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["t, с", "измеренное значение", "значение от нуля участка", "1-я форма (статика)"])
        for row in zip(result.ts, result.vs, result.vz, result.static_curve):
            writer.writerow([f"{row[0]:.6g}", f"{row[1]:.6g}", f"{row[2]:.6g}", f"{row[3]:.6g}"])


def write_spectrum_csv(path: str, spectral: core.SpectralResult) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["частота, Гц", "амплитуда"])
        for f, a in zip(spectral.freqs, spectral.spectrum):
            writer.writerow([f"{f:.6g}", f"{a:.6g}"])


def write_spectral_window_csv(path: str, spectral: core.SpectralResult) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["t, с", "значение от нуля участка"])
        for t, v in zip(spectral.ts, spectral.vz):
            writer.writerow([f"{t:.6g}", f"{v:.6g}"])


def _save_panel_png(path: str, draw_fn, *args) -> None:
    fig = Figure(figsize=(7, 4.5), dpi=150)
    ax = fig.add_subplot(111)
    draw_fn(ax, *args)
    fig.tight_layout()
    fig.savefig(path)


def save_report(
    base_dir: str,
    point_name: str,
    comment: str,
    result: core.AnalysisResult,
    spectral: core.SpectralResult | None,
) -> str:
    """Создаёт подпапку `base_dir/<название точки>` и сохраняет в неё
    report.pdf, CSV-файлы со значениями и рядами данных, а также
    отдельные PNG-изображения графиков. Возвращает путь к папке."""
    dirname = sanitize_dirname(point_name)
    out_dir = os.path.join(base_dir, dirname)
    os.makedirs(out_dir, exist_ok=True)

    write_values_csv(os.path.join(out_dir, "values.csv"), point_name, comment, result, spectral)
    write_segment_csv(os.path.join(out_dir, "segment_data.csv"), result)
    if spectral is not None:
        write_spectrum_csv(os.path.join(out_dir, "spectrum.csv"), spectral)
        write_spectral_window_csv(os.path.join(out_dir, "spectral_window.csv"), spectral)

    _save_panel_png(os.path.join(out_dir, "overview.png"), plotting.draw_overview, result)
    _save_panel_png(os.path.join(out_dir, "zoom.png"), plotting.draw_zoom, result)
    if spectral is not None:
        _save_panel_png(
            os.path.join(out_dir, "spectral_window.png"), plotting.draw_spectral_window, spectral
        )
        _save_panel_png(os.path.join(out_dir, "spectrum.png"), plotting.draw_spectrum, spectral)

    pdf_path = os.path.join(out_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig_text = Figure(figsize=(8.27, 11.69), dpi=150)  # A4
        ax = fig_text.add_subplot(111)
        ax.axis("off")
        title = point_name.strip() or "(без названия)"
        lines = [f"Отчёт по анализу тензограммы: {title}", ""]
        if comment.strip():
            lines.append("Комментарий:")
            lines.append(comment.strip())
            lines.append("")
        for key, value in _summary_rows(point_name, comment, result, spectral):
            if key == "":
                lines.append("")
            else:
                lines.append(f"{key}: {value}")
        ax.text(
            0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, family="monospace", wrap=True,
        )
        pdf.savefig(fig_text)

        fig_grid = Figure(figsize=(11, 8), dpi=150)
        plotting.draw_results_grid(fig_grid, result, spectral)
        pdf.savefig(fig_grid)

    return out_dir

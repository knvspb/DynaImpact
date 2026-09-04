"""Функции отрисовки графиков, общие для интерактивного GUI и отчёта.

Каждая функция рисует в переданные оси (Axes) и не создаёт фигуру сама —
это позволяет использовать один и тот же код и для совмещённого вида
результатов в GUI (несколько Axes на одной Figure), и для отдельных
изображений при сохранении отчёта (PNG/PDF).
"""

from __future__ import annotations

import numpy as np

from . import core


def draw_full_record(ax, t: np.ndarray, v: np.ndarray) -> None:
    ax.plot(t, v, linewidth=0.6, color="tab:blue")
    ax.set_title("Тензограмма (вся запись)")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Величина")
    ax.grid(True, alpha=0.3)


def draw_segment_preview(ax, ts: np.ndarray, vs: np.ndarray, event_time: float) -> None:
    ax.plot(ts, vs, linewidth=1.0, color="tab:blue")
    ax.axvline(event_time, color="tab:red", linestyle="--", linewidth=1, label="событие")
    ax.set_title("Выбранный участок тензограммы")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Величина")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _draw_static_dynamic(ax, ts, vz, static_curve, result: core.AnalysisResult, title: str) -> None:
    ax.plot(ts, vz, linewidth=0.8, color="tab:blue", label="сигнал (динамика, за вычетом нуля)")
    ax.plot(ts, static_curve, linewidth=1.8, color="tab:orange", label="1-я форма (квазистатика)")
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


def draw_overview(ax, result: core.AnalysisResult, legend: bool = True) -> None:
    _draw_static_dynamic(
        ax, result.ts, result.vz, result.static_curve, result,
        "Весь участок: динамика и 1-я форма (статика)",
    )
    if legend:
        ax.legend(loc="best", fontsize=8)


def draw_zoom(ax, result: core.AnalysisResult) -> None:
    half_span = max(result.neighborhood_s * 2.5, result.first_mode_period * 0.75, 1.0)
    t_center = result.t_static_max
    mask = np.abs(result.ts - t_center) <= half_span
    if mask.sum() < 4:
        mask = np.ones_like(result.ts, dtype=bool)
    _draw_static_dynamic(
        ax, result.ts[mask], result.vz[mask], result.static_curve[mask], result,
        "Увеличенный вид: проверка максимумов",
    )


def draw_spectral_window(ax, spectral: core.SpectralResult) -> None:
    ax.plot(spectral.ts, spectral.vz, linewidth=0.9, color="tab:purple")
    ax.set_title(
        f"Окно спектрального анализа "
        f"({spectral.t_start:.2f}..{spectral.t_end:.2f} с)"
    )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Величина (от нуля участка)")
    ax.grid(True, alpha=0.3)


def draw_spectrum(ax, spectral: core.SpectralResult, max_freq: float | None = None) -> None:
    freqs = spectral.freqs
    spectrum = spectral.spectrum
    if max_freq is None:
        if len(spectral.peak_freqs) > 0:
            max_freq = min(freqs.max(), spectral.peak_freqs.max() * 2.5)
        else:
            max_freq = freqs.max()
        max_freq = max(max_freq, 1.0)
    mask = freqs <= max_freq
    ax.plot(freqs[mask], spectrum[mask], linewidth=1.0, color="tab:purple")
    for f, a in zip(spectral.peak_freqs, spectral.peak_amps):
        if f <= max_freq:
            ax.plot([f], [a], "v", color="tab:red", markersize=7)
            ax.annotate(
                f"{f:.3f} Гц", (f, a), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8, color="tab:red",
            )
    ax.set_title("Спектр (окно после нагрузки)")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда")
    ax.grid(True, alpha=0.3)


def draw_results_grid(figure, result: core.AnalysisResult, spectral: core.SpectralResult | None) -> None:
    """Строит стандартную сетку 2x2 результатов на переданной Figure."""
    figure.clear()
    ax_overview = figure.add_subplot(221)
    ax_zoom = figure.add_subplot(222)
    ax_spec_time = figure.add_subplot(223)
    ax_spec_freq = figure.add_subplot(224)

    draw_overview(ax_overview, result)
    draw_zoom(ax_zoom, result)

    if spectral is not None:
        draw_spectral_window(ax_spec_time, spectral)
        draw_spectrum(ax_spec_freq, spectral)
    else:
        for ax, msg in (
            (ax_spec_time, "Спектральный анализ не выполнен"),
            (ax_spec_freq, "Спектральный анализ не выполнен"),
        ):
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    figure.tight_layout()

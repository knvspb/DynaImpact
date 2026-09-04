"""Загрузка тензограммы и расчёт статического/динамического значений.

Терминология и метод соответствуют практике испытаний сооружений
(мостов и т.п.) по цифровым тензограммам:

* "Тензограмма" — запись время/величина (деформация, прогиб и т.п.).
* Ноль записи (участка) принимается равным среднему значению величины
  на небольших ("тихих", безнагрузочных) отрезках в начале и в конце
  ВЫБРАННОГО УЧАСТКА (а не всей записи).
* "Первая форма колебания" — составляющая сигнала с наибольшим периодом
  (наименьшей частотой) среди значимых колебательных составляющих.
  Она выделяется низкочастотной фильтрацией (после удаления линейного
  тренда и вычитания нуля) и её экстремум на участке принимается за
  МАКСИМАЛЬНОЕ СТАТИЧЕСКОЕ значение.
* В окрестности момента этого статического максимума (ширина окрестности
  задаётся оператором) в НЕФИЛЬТРОВАННОМ (полном) сигнале ищется
  максимальное по модулю значение — МАКСИМАЛЬНОЕ ДИНАМИЧЕСКОЕ значение.
* Коэффициент динамичности = |динамическое| / |статическое|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, detrend


class TensogramError(Exception):
    """Ошибки загрузки/обработки тензограммы."""


# --------------------------------------------------------------------------
# Загрузка файла
# --------------------------------------------------------------------------

def load_tensogram(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Читает текстовый файл вида "время<пробел/таб>значение" построчно.

    Поддерживается запятая как десятичный разделитель, произвольное
    количество пробелов/табуляций между колонками, пустые строки и
    нечисловые заголовки (пропускаются).
    """
    times: list[float] = []
    values: list[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue
            try:
                t_val = float(parts[0])
                v_val = float(parts[1])
            except ValueError:
                continue
            times.append(t_val)
            values.append(v_val)

    if len(times) < 4:
        raise TensogramError(
            "Не удалось прочитать данные из файла. Ожидается текстовый "
            "файл со столбцами: время, значение."
        )

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)

    order = np.argsort(t, kind="mergesort")
    return t[order], v[order]


# --------------------------------------------------------------------------
# Выделение участка
# --------------------------------------------------------------------------

def extract_segment(
    t: np.ndarray, v: np.ndarray, event_time: float, before: float, after: float
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает данные на отрезке [event_time - before, event_time + after]."""
    if before < 0 or after < 0:
        raise TensogramError("Окна 'до' и 'после' не могут быть отрицательными.")
    t_start = event_time - before
    t_end = event_time + after
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 4:
        raise TensogramError(
            "На выбранном участке недостаточно точек данных. Проверьте "
            "время события и границы записи "
            f"({t[0]:.3f} .. {t[-1]:.3f} с)."
        )
    return t[mask], v[mask]


# --------------------------------------------------------------------------
# Ноль участка
# --------------------------------------------------------------------------

def segment_baseline(ts: np.ndarray, vs: np.ndarray, baseline_dur: float = 2.0) -> float:
    """Среднее значение по 'тихим' отрезкам в начале и в конце УЧАСТКА."""
    duration = ts[-1] - ts[0]
    dur = min(baseline_dur, duration / 2) if duration > 0 else baseline_dur
    if dur <= 0:
        dur = baseline_dur
    m_start = ts <= ts[0] + dur
    m_end = ts >= ts[-1] - dur
    v_start = vs[m_start].mean() if m_start.any() else vs[0]
    v_end = vs[m_end].mean() if m_end.any() else vs[-1]
    return 0.5 * (v_start + v_end)


# --------------------------------------------------------------------------
# Выделение первой формы колебаний (наибольший период)
# --------------------------------------------------------------------------

def sampling_rate(ts: np.ndarray) -> float:
    dt = float(np.median(np.diff(ts)))
    if dt <= 0:
        raise TensogramError("Некорректный шаг времени в данных.")
    return 1.0 / dt


def detect_first_mode_frequency(
    vz: np.ndarray, fs: float, duration: float, threshold_ratio: float = 0.2
) -> float:
    """Оценивает частоту первой формы колебаний (наименьшая значимая частота).

    Учитываются только частоты, для которых на участке укладывается не
    менее двух периодов (иначе колебание нельзя достоверно выделить), и
    амплитуда которых не менее `threshold_ratio` от максимальной в спектре.
    """
    n = len(vz)
    if n < 8 or duration <= 0:
        return 1.0

    vz_detrended = detrend(vz, type="linear")
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(vz_detrended * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    f_min = 2.0 / duration
    valid = freqs >= f_min
    spec_valid = spectrum[valid]
    freq_valid = freqs[valid]

    if len(spec_valid) == 0 or spec_valid.max() <= 0:
        return f_min

    threshold = threshold_ratio * spec_valid.max()
    candidates = freq_valid[spec_valid >= threshold]
    if len(candidates) == 0:
        return float(freq_valid[np.argmax(spec_valid)])
    return float(candidates.min())


def lowpass_filter(vz: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Фазово-нейтральная низкочастотная фильтрация (Баттерворт, filtfilt)."""
    nyq = fs / 2.0
    cutoff = min(max(cutoff_hz, 1e-6), nyq * 0.98)
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(vz) <= padlen:
        return vz.copy()
    return filtfilt(b, a, vz)


# --------------------------------------------------------------------------
# Полный расчёт
# --------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    baseline: float
    first_mode_freq: float
    first_mode_period: float
    cutoff_hz: float
    neighborhood_s: float

    static_max: float
    t_static_max: float

    dynamic_max: float
    t_dynamic_max: float

    dynamic_coefficient: float

    ts: np.ndarray
    vz: np.ndarray  # исходный сигнал участка за вычетом нуля
    static_curve: np.ndarray  # выделенная первая форма (статика)


def analyze_segment(
    ts: np.ndarray,
    vs: np.ndarray,
    baseline_dur: float = 2.0,
    cutoff_hz: float | None = None,
    cutoff_multiplier: float = 1.6,
    neighborhood_s: float | None = None,
) -> AnalysisResult:
    """Выполняет полный расчёт по выбранному участку тензограммы.

    Параметры `cutoff_hz` и `neighborhood_s`, если заданы явно (из GUI),
    имеют приоритет над автоматически определяемыми значениями.
    """
    if len(ts) < 8:
        raise TensogramError("Недостаточно точек для анализа участка.")

    baseline = segment_baseline(ts, vs, baseline_dur)
    vz = vs - baseline

    fs = sampling_rate(ts)
    duration = ts[-1] - ts[0]

    f1 = detect_first_mode_frequency(vz, fs, duration)

    if cutoff_hz is None or cutoff_hz <= 0:
        cutoff_hz = f1 * cutoff_multiplier

    static_curve = lowpass_filter(vz, fs, cutoff_hz)

    i_static = int(np.argmax(np.abs(static_curve)))
    static_max = float(static_curve[i_static])
    t_static_max = float(ts[i_static])

    if neighborhood_s is None or neighborhood_s <= 0:
        neighborhood_s = max(1.0 / f1, 1.0) if f1 > 0 else 1.0

    mask = np.abs(ts - t_static_max) <= neighborhood_s
    if not mask.any():
        mask = np.ones_like(ts, dtype=bool)

    ts_local = ts[mask]
    vz_local = vz[mask]
    i_dynamic_local = int(np.argmax(np.abs(vz_local)))
    dynamic_max = float(vz_local[i_dynamic_local])
    t_dynamic_max = float(ts_local[i_dynamic_local])

    if static_max == 0:
        dynamic_coefficient = float("nan")
    else:
        dynamic_coefficient = abs(dynamic_max) / abs(static_max)

    return AnalysisResult(
        baseline=baseline,
        first_mode_freq=f1,
        first_mode_period=(1.0 / f1 if f1 > 0 else float("inf")),
        cutoff_hz=cutoff_hz,
        neighborhood_s=neighborhood_s,
        static_max=static_max,
        t_static_max=t_static_max,
        dynamic_max=dynamic_max,
        t_dynamic_max=t_dynamic_max,
        dynamic_coefficient=dynamic_coefficient,
        ts=ts,
        vz=vz,
        static_curve=static_curve,
    )

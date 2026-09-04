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
* Динамический коэффициент = |динамическое| / |статическое|.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks


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
# Формат .ana + .anp (бинарные данные + текстовый заголовок)
# --------------------------------------------------------------------------

_ANP_KEY_ORDER = [
    "COMMENT", "GAIN", "ABSVOLT", "FRQ", "TMI", "FRL", "FRH", "FORMAT",
    "START", "DATE", "CHANNEL", "MAXLEVEL", "SENSE", "CONVERT", "AMPL",
    "REFER", "AFCH", "DC", "ID",
]

_ANP_LABELS = {
    "COMMENT": "Комментарий прибора",
    "GAIN": "Коэффициент усиления (GAIN)",
    "ABSVOLT": "ABSVOLT",
    "FRQ": "Частота дискретизации, Гц",
    "FRL": "Фильтр, нижняя частота, Гц",
    "FRH": "Фильтр, верхняя частота, Гц",
    "FORMAT": "Формат данных",
    "START": "Время начала записи",
    "DATE": "Дата записи",
    "CHANNEL": "Канал",
    "MAXLEVEL": "MAXLEVEL",
    "SENSE": "SENSE",
    "CONVERT": "Единицы измерения",
    "AMPL": "AMPL",
    "ID": "ID канала",
}


def _find_anp_companion(path: str) -> tuple[str, str]:
    """По пути к .ana или .anp находит пару (путь_к_ana, путь_к_anp)."""
    base, ext = os.path.splitext(path)
    ext_low = ext.lower()
    if ext_low == ".ana":
        ana_path, anp_path = path, base + ".anp"
    elif ext_low == ".anp":
        ana_path, anp_path = base + ".ana", path
    else:
        raise TensogramError(
            "Ожидается файл с расширением .ana или .anp (парный текстовый "
            "заголовок должен лежать рядом, с тем же именем)."
        )

    if not os.path.isfile(ana_path):
        raise TensogramError(f"Не найден файл данных: {os.path.basename(ana_path)}")
    if not os.path.isfile(anp_path):
        raise TensogramError(f"Не найден файл заголовка: {os.path.basename(anp_path)}")
    return ana_path, anp_path


def _parse_anp_header(anp_path: str) -> dict:
    """Разбирает текстовый заголовок .anp.

    Значения полей (в частности CONVERT — единицы измерения) в файлах
    этого прибора закодированы в Windows-1251, а не в UTF-8.
    """
    with open(anp_path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("cp1251", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    header: dict = {"title": lines[0] if lines else ""}
    for line in lines[1:]:
        parts = line.split(None, 1)
        key = parts[0].upper()
        value = parts[1].strip() if len(parts) > 1 else ""
        header[key] = value
    return header


def _format_anp_comment(header: dict, ana_path: str) -> str:
    lines = [f"Файл: {os.path.basename(ana_path)}"]
    if header.get("title"):
        lines.append(f"Канал/точка: {header['title']}")
    if header.get("DATE") or header.get("START"):
        lines.append(f"Начало записи: {header.get('DATE', '')} {header.get('START', '')}".strip())
    for key in _ANP_KEY_ORDER:
        if key == "COMMENT":
            continue
        if key in header and header[key] not in ("", None) and key not in ("DATE", "START"):
            label = _ANP_LABELS.get(key, key)
            value = header[key].replace("\t", "  ")
            lines.append(f"{label}: {value}")
    if header.get("COMMENT"):
        lines.append(f"Комментарий прибора: {header['COMMENT']}")
    return "\n".join(lines)


def load_ana_anp(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Загружает тензограмму из пары файлов .ana (бинарные данные) +
    .anp (текстовый заголовок с параметрами записи).

    `path` может указывать на любой из двух файлов пары. Данные в .ana
    (при FORMAT f2, единственном пока опробованном варианте) — это
    массив чисел float32 (little-endian), уже в инженерных единицах
    (масштабирование по GAIN/ABSVOLT в прибор уже применено при записи).
    Частота дискретизации берётся из поля FRQ заголовка.

    Возвращает (время, значение, текст с сопроводительной информацией
    из заголовка — для отображения оператору, например в комментарии).
    """
    ana_path, anp_path = _find_anp_companion(path)
    header = _parse_anp_header(anp_path)

    fmt = header.get("FORMAT", "").strip().lower()
    if fmt not in ("f2", "f4", "float", "float32"):
        raise TensogramError(
            f"Неподдерживаемый формат данных .ana: FORMAT={header.get('FORMAT')!r}. "
            "Поддерживается только 32-битный float (FORMAT f2/f4)."
        )

    try:
        freq_hz = float(header.get("FRQ", "").split()[0])
    except (ValueError, IndexError):
        raise TensogramError("В заголовке .anp не найдена частота дискретизации (FRQ).")
    if freq_hz <= 0:
        raise TensogramError("Некорректная частота дискретизации (FRQ) в заголовке .anp.")

    with open(ana_path, "rb") as fh:
        raw = fh.read()
    if len(raw) % 4 != 0:
        raise TensogramError(
            f"Размер файла {ana_path!r} не кратен 4 байтам — не похоже на массив float32."
        )
    v = np.frombuffer(raw, dtype="<f4").astype(float)
    if len(v) < 4:
        raise TensogramError("Файл .ana не содержит достаточного количества данных.")

    t = np.arange(len(v)) / freq_hz

    comment = _format_anp_comment(header, ana_path)
    return t, v, comment


DEFAULT_NO_DATA_MIN_DURATION_S = 5.0


def mask_no_data_runs(
    t: np.ndarray,
    v: np.ndarray,
    min_duration_s: float = DEFAULT_NO_DATA_MIN_DURATION_S,
    eps: float = 1e-9,
) -> np.ndarray:
    """Возвращает копию `v`, где длинные участки отсутствия записи
    заменены на NaN — только для отображения (не меняет исходные данные,
    участвующие в расчёте).

    Длительная запись часто содержит интервалы, где канал физически не
    писал данные и это выражается строго нулевым значением (в отличие
    от кратковременных переходов сигнала через ноль при реальных
    колебаниях, которые длятся доли секунды). Такие интервалы, если их
    показывать наравне с реальными данными, вынуждают вертикальную ось
    графика захватывать 0, из-за чего интересующие колебания (например,
    в диапазоне 150..185) сжимаются в узкую полосу графика. Участки
    короче `min_duration_s` (обычные переходы через ноль) не трогаются.
    """
    is_zero = np.abs(v) <= eps
    if not is_zero.any():
        return v

    v_out = v.astype(float, copy=True)
    n = len(v)
    i = 0
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            duration = t[j - 1] - t[i] if j > i else 0.0
            if duration >= min_duration_s:
                v_out[i:j] = np.nan
            i = j
        else:
            i += 1
    return v_out


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


def extract_window(
    t: np.ndarray, v: np.ndarray, t_start: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает данные на явно заданном отрезке [t_start, t_end]."""
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 4:
        raise TensogramError(
            "Недостаточно точек данных на окне "
            f"[{t_start:.3f} .. {t_end:.3f}] с. Проверьте параметры "
            f"спектрального анализа и границы записи ({t[0]:.3f} .. {t[-1]:.3f} с)."
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


def fft_spectrum(v: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Амплитудный спектр сигнала (после удаления линейного тренда и
    оконного взвешивания Ханна). Общий помощник для всех спектральных
    вычислений в модуле."""
    n = len(v)
    v_detrended = detrend(v, type="linear")
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(v_detrended * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spectrum


MAX_FIRST_MODE_FREQ_HZ = 1.0
"""Верхняя граница физически правдоподобной частоты первой формы колебаний.

Для реальных сооружений (мостов и т.п.) частота первой (основной) формы
собственных колебаний практически всегда не превышает 1 Гц; более
высокочастотные пики в спектре — это либо старшие формы/вибрации, либо
шум, и не должны приниматься за первую форму, даже если формально
оказываются самой низкочастотной значимой составляющей на "тихом"
(слабо нагруженном) участке.
"""


def first_mode_candidates(
    vz: np.ndarray,
    fs: float,
    duration: float,
    max_freq: float = MAX_FIRST_MODE_FREQ_HZ,
    n: int = 8,
) -> list[tuple[float, float]]:
    """Возвращает все правдоподобные кандидаты в частоту первой формы —
    локальные пики спектра в диапазоне [2/длительность, `max_freq`] Гц,
    отсортированные по частоте (до `n` штук, отобранных по амплитуде).

    Список отдаётся оператору целиком (см. `AnalysisResult.first_mode_candidates`)
    для ручной калибровки — окончательный выбор одной частоты из них
    делает `choose_first_mode_frequency`.
    """
    n_samples = len(vz)
    if n_samples < 8 or duration <= 0:
        return []

    freqs, spectrum = fft_spectrum(vz, fs)

    # Пики ищутся по ПОЛНОМУ спектру, а не по уже обрезанному до
    # [f_min, max_freq] срезу: иначе бин ровно на границе среза не имеет
    # соседа с одной стороны в обрезанном массиве и find_peaks никогда не
    # распознает его как пик, даже если он — истинный (и часто самый
    # значимый) локальный максимум спектра.
    all_peak_idx, _ = find_peaks(spectrum)

    f_min = 2.0 / duration
    in_range = (freqs[all_peak_idx] >= f_min) & (freqs[all_peak_idx] <= max_freq)
    peak_idx = all_peak_idx[in_range]

    if len(peak_idx) == 0:
        valid = (freqs >= f_min) & (freqs <= max_freq)
        if not valid.any():
            valid = freqs >= f_min
        if not valid.any() or spectrum[valid].max() <= 0:
            return []
        peak_idx = np.where(valid)[0][[np.argmax(spectrum[valid])]]

    order = np.argsort(spectrum[peak_idx])[::-1][:n]
    chosen = peak_idx[order]
    chosen = chosen[np.argsort(freqs[chosen])]
    return [(float(freqs[i]), float(spectrum[i])) for i in chosen]


def choose_first_mode_frequency(
    candidates: list[tuple[float, float]],
    duration: float,
    threshold_ratio: float = 0.2,
    expected_freq: float | None = None,
) -> float:
    """Выбирает одну частоту первой формы из списка кандидатов.

    Если задана `expected_freq` (физически ожидаемая частота — например,
    из условия "период = время прохождения автомобиля по пролёту"),
    берётся ближайший к ней кандидат: это привязывает выбор к физике
    задачи, а не только к тому, какой пик спектра случайно оказался
    самым низкочастотным значимым. Иначе используется прежняя эвристика:
    наименьшая частота среди кандидатов с амплитудой не менее
    `threshold_ratio` от максимальной.
    """
    if not candidates:
        return 2.0 / duration if duration > 0 else 1.0

    if expected_freq is not None and expected_freq > 0:
        return min(candidates, key=lambda c: abs(c[0] - expected_freq))[0]

    max_amp = max(a for _, a in candidates)
    significant = [f for f, a in candidates if a >= threshold_ratio * max_amp]
    return min(significant) if significant else candidates[0][0]


def detect_first_mode_frequency(
    vz: np.ndarray,
    fs: float,
    duration: float,
    threshold_ratio: float = 0.2,
    expected_freq: float | None = None,
) -> float:
    """Оценивает частоту первой формы колебаний.

    Учитываются только частоты, для которых на участке укладывается не
    менее двух периодов (иначе колебание нельзя достоверно выделить), и
    которые не превышают `MAX_FIRST_MODE_FREQ_HZ` (см. её описание —
    реальная первая форма сооружения не бывает выше ~1 Гц). См.
    `choose_first_mode_frequency` за тем, как из кандидатов выбирается
    итоговое значение (в том числе с учётом `expected_freq`).
    """
    candidates = first_mode_candidates(vz, fs, duration)
    return choose_first_mode_frequency(candidates, duration, threshold_ratio, expected_freq)


MAX_CUTOFF_SEARCH_HZ = 3.0
"""Верхняя граница поиска "провала" в спектре для частоты среза.

На слабо нагруженных ("тихих") участках, где нет выраженной формы
нагружения, гармоники первой формы могут быть едва заметны на фоне
шума, и поиск провала способен случайно "уехать" далеко по частоте —
вплоть до реальных высокочастотных форм/вибраций конструкции (тогда
частота среза получалась бы завышенной на порядок и более, например
8-12 Гц вместо ожидаемых для типичного проезда автомобиля ~0.2-0.5 Гц).
Ограничение поиска разумным диапазоном (первые несколько герц) не даёт
этому случиться; если в этом диапазоне подходящего провала не нашлось,
используется запасной вариант `fallback_multiplier * f1`.
"""


def suggest_cutoff_frequency(
    vz: np.ndarray,
    fs: float,
    duration: float,
    f1: float | None = None,
    gap_ratio: float = 0.03,
    gap_span_hz: float = 0.3,
    fallback_multiplier: float = 1.6,
) -> float:
    """Подбирает частоту среза, отделяющую первую форму (и её гармоники,
    формирующие форму нарастания/спада статического воздействия) от
    более высокочастотных форм колебаний конструкции.

    Одиночный "горб" статического воздействия, как правило, несинусоидален
    и поэтому даёт в спектре не одну частоту, а основную частоту первой
    формы и набор её гармоник убывающей амплитуды. Простой срез сразу
    после первой формы (например, в 1.5–2 раза выше её частоты) обрезает
    и эти гармоники, занижая максимум статической составляющей.

    Вместо этого ищется первый достаточно широкий провал в спектре
    (участок, где амплитуда падает ниже `gap_ratio` от максимума и
    остаётся такой на протяжении не менее `gap_span_hz`) после частоты
    первой формы — он отделяет затухающие гармоники формы нагружения от
    старших форм/вибраций конструкции. Частота среза берётся на границе
    этого провала.

    Если такого провала не находится (спектр непрерывный, без чёткого
    разделения), используется запасной вариант: `fallback_multiplier`,
    умноженный на частоту первой формы.

    Если `f1` передана явно (например, итоговый выбор
    `choose_first_mode_frequency`, в том числе с учётом физически
    ожидаемой частоты), используется она — чтобы срез и сама частота
    первой формы были согласованы между собой, а не определялись двумя
    независимыми эвристиками.
    """
    n = len(vz)
    if n < 8 or duration <= 0:
        return 1.0

    freqs, spectrum = fft_spectrum(vz, fs)

    f_min = 2.0 / duration
    valid = (freqs >= f_min) & (freqs <= MAX_FIRST_MODE_FREQ_HZ)
    if not valid.any():
        valid = freqs >= f_min
    if not valid.any() or spectrum[valid].max() <= 0:
        return f_min * fallback_multiplier

    if f1 is None:
        spec_valid = spectrum[valid]
        freq_valid = freqs[valid]
        f1 = float(freq_valid[np.argmax(spec_valid)])

    threshold = gap_ratio * spectrum[valid].max()
    below_threshold = spectrum < threshold
    candidate_idx = np.where((freqs > f1) & (freqs <= MAX_CUTOFF_SEARCH_HZ) & below_threshold)[0]

    for i in candidate_idx:
        span_mask = (freqs >= freqs[i]) & (freqs <= freqs[i] + gap_span_hz)
        if span_mask.sum() > 3 and below_threshold[span_mask].all():
            return float(freqs[i])

    return f1 * fallback_multiplier


def lowpass_filter(vz: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Фазово-нейтральная низкочастотная фильтрация (Баттерворт, filtfilt)."""
    nyq = fs / 2.0
    cutoff = min(max(cutoff_hz, 1e-6), nyq * 0.98)
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(vz) <= padlen:
        return vz.copy()
    return filtfilt(b, a, vz)


def find_zero_crossing_bounds(ts: np.ndarray, curve: np.ndarray, t_peak: float) -> tuple[float, float]:
    """От момента `t_peak` ищет ближайшие пересечения нуля кривой `curve`
    слева и справа — границы "горба" (от нуля до нуля), содержащего пик.
    """
    i_peak = int(np.argmin(np.abs(ts - t_peak)))
    n = len(curve)
    sign_peak = 1.0 if curve[i_peak] >= 0 else -1.0

    i_left = i_peak
    while i_left > 0 and np.sign(curve[i_left - 1]) == sign_peak:
        i_left -= 1

    i_right = i_peak
    while i_right < n - 1 and np.sign(curve[i_right + 1]) == sign_peak:
        i_right += 1

    return float(ts[i_left]), float(ts[i_right])


def compute_vehicle_speed(
    span_length_m: float, wheelbase_m: float, crossing_period_s: float
) -> tuple[float, float]:
    """Оценивает фактическую скорость автомобиля по времени прохождения
    пролёта.

    Пока хотя бы одна ось автомобиля находится на пролёте, он проходит
    расстояние "длина пролёта + база автомобиля" (от въезда передней оси
    до съезда задней). Время этого прохождения — период "от нуля до
    нуля" квазистатической кривой (см. `find_zero_crossing_bounds`).

    Возвращает (скорость, м/с; скорость, км/ч).
    """
    if crossing_period_s <= 0:
        raise TensogramError("Период прохождения (время между нулями) должен быть положительным.")
    if span_length_m <= 0:
        raise TensogramError("Длина пролёта должна быть положительной.")
    if wheelbase_m < 0:
        raise TensogramError("База автомобиля не может быть отрицательной.")

    distance_m = span_length_m + wheelbase_m
    speed_m_s = distance_m / crossing_period_s
    speed_kmh = speed_m_s * 3.6
    return speed_m_s, speed_kmh


@dataclass
class SpeedInfo:
    span_length_m: float
    wheelbase_m: float
    design_speed_kmh: float
    crossing_period_s: float
    actual_speed_m_s: float
    actual_speed_kmh: float


def compute_speed_info(
    span_length_m: float, wheelbase_m: float, design_speed_kmh: float, crossing_period_s: float
) -> SpeedInfo:
    """Сопоставляет расчётную скорость (введённую оператором) и
    фактическую (по времени прохождения пролёта, см. `compute_vehicle_speed`)."""
    speed_m_s, speed_kmh = compute_vehicle_speed(span_length_m, wheelbase_m, crossing_period_s)
    return SpeedInfo(
        span_length_m=span_length_m,
        wheelbase_m=wheelbase_m,
        design_speed_kmh=design_speed_kmh,
        crossing_period_s=crossing_period_s,
        actual_speed_m_s=speed_m_s,
        actual_speed_kmh=speed_kmh,
    )


def expected_first_mode_frequency(
    span_length_m: float | None, wheelbase_m: float | None, design_speed_kmh: float | None
) -> float | None:
    """Ожидаемая частота первой формы, исходя из условия: период первой
    формы равен времени прохождения автомобилем пролёта на расчётной
    скорости. Возвращает None, если каких-то из трёх исходных данных нет
    или они некорректны (тогда выбор частоты опирается только на спектр).
    """
    if span_length_m is None or wheelbase_m is None or design_speed_kmh is None:
        return None
    if span_length_m <= 0 or wheelbase_m < 0 or design_speed_kmh <= 0:
        return None
    distance_m = span_length_m + wheelbase_m
    speed_m_s = design_speed_kmh / 3.6
    expected_period_s = distance_m / speed_m_s
    if expected_period_s <= 0:
        return None
    return 1.0 / expected_period_s


# --------------------------------------------------------------------------
# Полный расчёт
# --------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    baseline: float
    first_mode_freq: float
    first_mode_period: float
    first_mode_candidates: list[tuple[float, float]]  # (частота, амплитуда), для калибровки
    expected_first_mode_freq: float | None  # из длины пролёта/скорости/базы, если заданы
    cutoff_hz: float
    neighborhood_s: float

    static_max: float
    t_static_max: float

    dynamic_max: float
    t_dynamic_max: float

    dynamic_coefficient: float

    t_hump_start: float  # границы "горба" статики (от нуля до нуля), содержащего t_static_max
    t_hump_end: float
    crossing_period_s: float  # = t_hump_end - t_hump_start

    ts: np.ndarray
    vs: np.ndarray  # исходный (не зануленный) сигнал участка
    vz: np.ndarray  # исходный сигнал участка за вычетом нуля
    static_curve: np.ndarray  # выделенная первая форма (статика)


DEFAULT_NEIGHBORHOOD_S = 1.0


def analyze_segment(
    ts: np.ndarray,
    vs: np.ndarray,
    baseline_dur: float = 2.0,
    cutoff_hz: float | None = None,
    cutoff_multiplier: float = 1.6,
    neighborhood_s: float | None = None,
    expected_first_mode_freq: float | None = None,
) -> AnalysisResult:
    """Выполняет полный расчёт по выбранному участку тензограммы.

    Параметры `cutoff_hz` и `neighborhood_s`, если заданы явно (из GUI),
    имеют приоритет над автоматически определяемыми значениями.

    `expected_first_mode_freq` — физически ожидаемая частота первой формы
    (из длины пролёта, базы автомобиля и расчётной скорости, см.
    `expected_first_mode_frequency`); если задана, используется как
    ориентир при выборе среди кандидатов в частоту первой формы.
    """
    if len(ts) < 8:
        raise TensogramError("Недостаточно точек для анализа участка.")

    baseline = segment_baseline(ts, vs, baseline_dur)
    vz = vs - baseline

    fs = sampling_rate(ts)
    duration = ts[-1] - ts[0]

    candidates = first_mode_candidates(vz, fs, duration)
    f1 = choose_first_mode_frequency(candidates, duration, expected_freq=expected_first_mode_freq)

    if cutoff_hz is None or cutoff_hz <= 0:
        cutoff_hz = suggest_cutoff_frequency(
            vz, fs, duration, f1=f1, fallback_multiplier=cutoff_multiplier
        )

    static_curve = lowpass_filter(vz, fs, cutoff_hz)

    i_static = int(np.argmax(np.abs(static_curve)))
    static_max = float(static_curve[i_static])
    t_static_max = float(ts[i_static])

    if neighborhood_s is None or neighborhood_s <= 0:
        neighborhood_s = DEFAULT_NEIGHBORHOOD_S

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

    t_hump_start, t_hump_end = find_zero_crossing_bounds(ts, static_curve, t_static_max)

    return AnalysisResult(
        baseline=baseline,
        first_mode_freq=f1,
        first_mode_period=(1.0 / f1 if f1 > 0 else float("inf")),
        first_mode_candidates=candidates,
        expected_first_mode_freq=expected_first_mode_freq,
        cutoff_hz=cutoff_hz,
        neighborhood_s=neighborhood_s,
        static_max=static_max,
        t_static_max=t_static_max,
        dynamic_max=dynamic_max,
        t_dynamic_max=t_dynamic_max,
        dynamic_coefficient=dynamic_coefficient,
        t_hump_start=t_hump_start,
        t_hump_end=t_hump_end,
        crossing_period_s=t_hump_end - t_hump_start,
        ts=ts,
        vs=vs,
        vz=vz,
        static_curve=static_curve,
    )


# --------------------------------------------------------------------------
# Спектральный анализ (определение частот собственных колебаний)
# --------------------------------------------------------------------------

DEFAULT_SPECTRAL_DELAY_S = 5.0
DEFAULT_SPECTRAL_WINDOW_S = 10.0


@dataclass
class SpectralResult:
    t_start: float
    t_end: float
    delay_s: float
    window_s: float

    ts: np.ndarray
    vz: np.ndarray  # окно сигнала (за вычетом нуля участка), использованное для БПФ

    freqs: np.ndarray
    spectrum: np.ndarray

    peak_freqs: np.ndarray
    peak_amps: np.ndarray


def spectral_analysis(
    t_full: np.ndarray,
    v_full: np.ndarray,
    baseline: float,
    t_reference: float,
    delay_s: float = DEFAULT_SPECTRAL_DELAY_S,
    window_s: float = DEFAULT_SPECTRAL_WINDOW_S,
    n_peaks: int = 5,
) -> SpectralResult:
    """Спектральный анализ участка записи для определения частот
    собственных колебаний конструкции.

    Окно берётся из ПОЛНОЙ записи (а не только из ранее выбранного
    участка), начиная через `delay_s` секунд после `t_reference`
    (обычно — момент максимального статического значения) и длиной
    `window_s` секунд. Такая задержка нужна, чтобы окно спектрального
    анализа приходилось на свободные колебания конструкции уже после
    прохода нагрузки, без искажения спектра частотным составом самого
    воздействия.
    """
    if delay_s < 0:
        raise TensogramError("Задержка от максимума не может быть отрицательной.")
    if window_s <= 0:
        raise TensogramError("Длительность окна спектрального анализа должна быть положительной.")

    t_start = t_reference + delay_s
    t_end = t_start + window_s

    tw, vw = extract_window(t_full, v_full, t_start, t_end)
    vzw = vw - baseline

    fs = sampling_rate(tw)
    freqs, spectrum = fft_spectrum(vzw, fs)

    valid = freqs > 0
    freq_valid = freqs[valid]
    spec_valid = spectrum[valid]

    if len(spec_valid) == 0:
        peak_freqs = np.array([])
        peak_amps = np.array([])
    else:
        peak_idx, _ = find_peaks(spec_valid)
        if len(peak_idx) == 0:
            peak_idx = np.array([int(np.argmax(spec_valid))])
        order = np.argsort(spec_valid[peak_idx])[::-1][:n_peaks]
        chosen = peak_idx[order]
        sort_by_freq = np.argsort(freq_valid[chosen])
        peak_freqs = freq_valid[chosen][sort_by_freq]
        peak_amps = spec_valid[chosen][sort_by_freq]

    return SpectralResult(
        t_start=t_start,
        t_end=t_end,
        delay_s=delay_s,
        window_s=window_s,
        ts=tw,
        vz=vzw,
        freqs=freqs,
        spectrum=spectrum,
        peak_freqs=peak_freqs,
        peak_amps=peak_amps,
    )


DEFAULT_SPECTRAL_SWEEP_STEP_S = 0.5


def sweep_spectral_delays(
    t_full: np.ndarray,
    v_full: np.ndarray,
    baseline: float,
    t_reference: float,
    window_s: float,
    delay_start_s: float,
    delay_end_s: float,
    delay_step_s: float = DEFAULT_SPECTRAL_SWEEP_STEP_S,
    n_peaks: int = 5,
) -> list[SpectralResult]:
    """Выполняет спектральный анализ для набора задержек от `t_reference`
    (от `delay_start_s` до `delay_end_s` с шагом `delay_step_s`, при
    фиксированной длине окна `window_s`).

    Полезно для отделения устойчивых частот собственных колебаний
    конструкции (которые проявляются в спектре стабильно, независимо от
    того, с какой задержкой после максимума взято окно) от случайных
    или переходных пиков, характерных лишь для отдельных задержек.

    Окна, выходящие за пределы записи, пропускаются (не прерывают всю
    развёртку); если ни одна задержка не дала результата, поднимается
    TensogramError.
    """
    if delay_step_s <= 0:
        raise TensogramError("Шаг изменения задержки должен быть положительным.")
    if delay_end_s < delay_start_s:
        raise TensogramError("Конечная задержка не может быть меньше начальной.")

    results: list[SpectralResult] = []
    n_steps = int(round((delay_end_s - delay_start_s) / delay_step_s))
    for i in range(n_steps + 1):
        delay = delay_start_s + i * delay_step_s
        try:
            res = spectral_analysis(
                t_full, v_full, baseline, t_reference,
                delay_s=delay, window_s=window_s, n_peaks=n_peaks,
            )
        except TensogramError:
            continue
        results.append(res)

    if not results:
        raise TensogramError(
            "Не удалось выполнить развёртку по задержке — окно спектра выходит "
            "за пределы записи для всех заданных значений задержки."
        )
    return results

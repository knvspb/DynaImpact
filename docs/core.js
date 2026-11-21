/* core.js — численное ядро веб-версии анализатора тензограмм.
 *
 * Прямой порт tensogram_analyzer/core.py (Python). Работает и в браузере
 * (через <script src="core.js">, экспортирует window.TensoCore), и в
 * Node.js (для автономного тестирования: module.exports).
 */
(function (root) {
  "use strict";

  class TensogramError extends Error {}

  // ------------------------------------------------------------------
  // Загрузка / разбор файлов
  // ------------------------------------------------------------------

  function parseTextTensogram(text) {
    const times = [];
    const values = [];
    const lines = text.split(/\r\n|\r|\n/);
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) continue;
      const parts = line.replace(/,/g, ".").split(/\s+/);
      if (parts.length < 2) continue;
      const t = parseFloat(parts[0]);
      const v = parseFloat(parts[1]);
      if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
      times.push(t);
      values.push(v);
    }
    if (times.length < 4) {
      throw new TensogramError(
        "Не удалось прочитать данные из файла. Ожидается текстовый файл со столбцами: время, значение."
      );
    }
    const order = times.map((_, i) => i).sort((a, b) => times[a] - times[b]);
    const t = new Float64Array(order.length);
    const v = new Float64Array(order.length);
    order.forEach((idx, i) => {
      t[i] = times[idx];
      v[i] = values[idx];
    });
    return { t, v };
  }

  const ANP_LABELS = {
    COMMENT: "Комментарий прибора",
    GAIN: "Коэффициент усиления (GAIN)",
    ABSVOLT: "ABSVOLT",
    FRQ: "Частота дискретизации, Гц",
    FRL: "Фильтр, нижняя частота, Гц",
    FRH: "Фильтр, верхняя частота, Гц",
    FORMAT: "Формат данных",
    CHANNEL: "Канал",
    MAXLEVEL: "MAXLEVEL",
    SENSE: "SENSE",
    CONVERT: "Единицы измерения",
    AMPL: "AMPL",
    ID: "ID канала",
  };
  const ANP_KEY_ORDER = [
    "GAIN", "ABSVOLT", "FRQ", "TMI", "FRL", "FRH", "FORMAT", "CHANNEL",
    "MAXLEVEL", "SENSE", "CONVERT", "AMPL", "REFER", "AFCH", "DC", "ID",
  ];

  function parseAnpHeader(text) {
    const lines = text.split(/\r\n|\r|\n/).map((l) => l.trim()).filter(Boolean);
    const header = { title: lines.length ? lines[0] : "" };
    for (let i = 1; i < lines.length; i++) {
      const m = lines[i].match(/^(\S+)\s*(.*)$/);
      if (!m) continue;
      header[m[1].toUpperCase()] = (m[2] || "").trim();
    }
    return header;
  }

  function formatAnpComment(header, fileName) {
    const lines = [`Файл: ${fileName}`];
    if (header.title) lines.push(`Канал/точка: ${header.title}`);
    const dateStart = `${header.DATE || ""} ${header.START || ""}`.trim();
    if (dateStart) lines.push(`Начало записи: ${dateStart}`);
    for (const key of ANP_KEY_ORDER) {
      if (header[key] !== undefined && header[key] !== "") {
        const label = ANP_LABELS[key] || key;
        lines.push(`${label}: ${header[key].replace(/\t/g, "  ")}`);
      }
    }
    if (header.COMMENT) lines.push(`Комментарий прибора: ${header.COMMENT}`);
    return lines.join("\n");
  }

  function parseAnaBinary(arrayBuffer, freqHz) {
    if (arrayBuffer.byteLength % 4 !== 0) {
      throw new TensogramError("Размер файла .ana не кратен 4 байтам — не похоже на массив float32.");
    }
    const n = arrayBuffer.byteLength / 4;
    if (n < 4) throw new TensogramError("Файл .ana не содержит достаточного количества данных.");
    const view = new DataView(arrayBuffer);
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = view.getFloat32(i * 4, true);
    const t = new Float64Array(n);
    for (let i = 0; i < n; i++) t[i] = i / freqHz;
    return { t, v };
  }

  // ------------------------------------------------------------------
  // Исключение участков без записи (только для отображения)
  // ------------------------------------------------------------------

  const DEFAULT_NO_DATA_MIN_DURATION_S = 5.0;

  function maskNoDataRuns(t, v, minDurationS, eps) {
    minDurationS = minDurationS === undefined ? DEFAULT_NO_DATA_MIN_DURATION_S : minDurationS;
    eps = eps === undefined ? 1e-9 : eps;
    const out = Float64Array.from(v);
    const n = v.length;
    let i = 0;
    while (i < n) {
      if (Math.abs(v[i]) <= eps) {
        let j = i;
        while (j < n && Math.abs(v[j]) <= eps) j++;
        const duration = j > i ? t[j - 1] - t[i] : 0;
        if (duration >= minDurationS) {
          for (let k = i; k < j; k++) out[k] = NaN;
        }
        i = j;
      } else {
        i++;
      }
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Выделение участка
  // ------------------------------------------------------------------

  function extractWindow(t, v, tStart, tEnd) {
    const idx = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] >= tStart && t[i] <= tEnd) idx.push(i);
    }
    if (idx.length < 4) {
      throw new TensogramError(
        `Недостаточно точек данных на окне [${tStart.toFixed(3)} .. ${tEnd.toFixed(3)}] с.`
      );
    }
    const tt = new Float64Array(idx.length);
    const vv = new Float64Array(idx.length);
    idx.forEach((ix, i) => { tt[i] = t[ix]; vv[i] = v[ix]; });
    return { t: tt, v: vv };
  }

  // ------------------------------------------------------------------
  // Ноль участка
  // ------------------------------------------------------------------

  function segmentBaseline(ts, vs, baselineDur) {
    baselineDur = baselineDur === undefined ? 2.0 : baselineDur;
    const duration = ts[ts.length - 1] - ts[0];
    let dur = duration > 0 ? Math.min(baselineDur, duration / 2) : baselineDur;
    if (dur <= 0) dur = baselineDur;
    let sumStart = 0, nStart = 0, sumEnd = 0, nEnd = 0;
    const t0 = ts[0], t1 = ts[ts.length - 1];
    for (let i = 0; i < ts.length; i++) {
      if (ts[i] <= t0 + dur) { sumStart += vs[i]; nStart++; }
      if (ts[i] >= t1 - dur) { sumEnd += vs[i]; nEnd++; }
    }
    const vStart = nStart ? sumStart / nStart : vs[0];
    const vEnd = nEnd ? sumEnd / nEnd : vs[vs.length - 1];
    return 0.5 * (vStart + vEnd);
  }

  // ------------------------------------------------------------------
  // Частота дискретизации
  // ------------------------------------------------------------------

  function samplingRate(ts) {
    const diffs = [];
    for (let i = 1; i < ts.length; i++) diffs.push(ts[i] - ts[i - 1]);
    diffs.sort((a, b) => a - b);
    const dt = diffs[Math.floor(diffs.length / 2)];
    if (!(dt > 0)) throw new TensogramError("Некорректный шаг времени в данных.");
    return 1.0 / dt;
  }

  // ------------------------------------------------------------------
  // БПФ-спектр (прямой DFT: rfft-эквивалент, точные бины k/N*fs)
  // ------------------------------------------------------------------

  function linearDetrend(v) {
    const n = v.length;
    if (n < 2) return Float64Array.from(v);
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
      sumX += i; sumY += v[i]; sumXY += i * v[i]; sumXX += i * i;
    }
    const denom = n * sumXX - sumX * sumX;
    const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
    const intercept = (sumY - slope * sumX) / n;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = v[i] - (slope * i + intercept);
    return out;
  }

  function hanningWindow(n) {
    const w = new Float64Array(n);
    if (n === 1) { w[0] = 1; return w; }
    for (let i = 0; i < n; i++) {
      w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    }
    return w;
  }

  function fftSpectrum(v, fs) {
    const n = v.length;
    const detrended = linearDetrend(v);
    const win = hanningWindow(n);
    const windowed = new Float64Array(n);
    for (let i = 0; i < n; i++) windowed[i] = detrended[i] * win[i];

    const nBins = Math.floor(n / 2) + 1;
    const freqs = new Float64Array(nBins);
    const spectrum = new Float64Array(nBins);
    const base = (-2 * Math.PI) / n;
    for (let k = 0; k < nBins; k++) {
      let re = 0, im = 0;
      const w = base * k;
      for (let j = 0; j < n; j++) {
        const angle = w * j;
        re += windowed[j] * Math.cos(angle);
        im += windowed[j] * Math.sin(angle);
      }
      freqs[k] = (k * fs) / n;
      spectrum[k] = Math.hypot(re, im);
    }
    return { freqs, spectrum };
  }

  // ------------------------------------------------------------------
  // Частота первой формы: кандидаты + выбор
  // ------------------------------------------------------------------

  const MAX_FIRST_MODE_FREQ_HZ = 1.0;
  const MAX_CUTOFF_SEARCH_HZ = 3.0;

  function findLocalMaxima(y) {
    const idx = [];
    for (let i = 1; i < y.length - 1; i++) {
      if (y[i] > y[i - 1] && y[i] > y[i + 1]) idx.push(i);
    }
    return idx;
  }

  function firstModeCandidates(vz, fs, duration, maxFreq, n) {
    maxFreq = maxFreq === undefined ? MAX_FIRST_MODE_FREQ_HZ : maxFreq;
    n = n === undefined ? 8 : n;
    if (vz.length < 8 || duration <= 0) return [];

    const { freqs, spectrum } = fftSpectrum(vz, fs);
    const allPeaks = findLocalMaxima(spectrum);
    const fMin = 2.0 / duration;
    let peakIdx = allPeaks.filter((i) => freqs[i] >= fMin && freqs[i] <= maxFreq);

    if (peakIdx.length === 0) {
      const valid = [];
      for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] >= fMin && freqs[i] <= maxFreq) valid.push(i);
      }
      const pool = valid.length ? valid : freqs.map((_, i) => i).filter((i) => freqs[i] >= fMin);
      if (!pool.length) return [];
      let best = pool[0];
      for (const i of pool) if (spectrum[i] > spectrum[best]) best = i;
      if (spectrum[best] <= 0) return [];
      peakIdx = [best];
    }

    peakIdx.sort((a, b) => spectrum[b] - spectrum[a]);
    peakIdx = peakIdx.slice(0, n);
    peakIdx.sort((a, b) => freqs[a] - freqs[b]);
    return peakIdx.map((i) => [freqs[i], spectrum[i]]);
  }

  function chooseFirstModeFrequency(candidates, duration, thresholdRatio, expectedFreq) {
    thresholdRatio = thresholdRatio === undefined ? 0.2 : thresholdRatio;
    if (!candidates.length) return duration > 0 ? 2.0 / duration : 1.0;

    if (expectedFreq !== undefined && expectedFreq !== null && expectedFreq > 0) {
      let best = candidates[0];
      for (const c of candidates) {
        if (Math.abs(c[0] - expectedFreq) < Math.abs(best[0] - expectedFreq)) best = c;
      }
      return best[0];
    }

    const maxAmp = Math.max(...candidates.map((c) => c[1]));
    const significant = candidates.filter((c) => c[1] >= thresholdRatio * maxAmp);
    if (significant.length) {
      return significant.reduce((m, c) => Math.min(m, c[0]), Infinity);
    }
    return candidates[0][0];
  }

  function detectFirstModeFrequency(vz, fs, duration, thresholdRatio, expectedFreq) {
    const candidates = firstModeCandidates(vz, fs, duration);
    return chooseFirstModeFrequency(candidates, duration, thresholdRatio, expectedFreq);
  }

  // ------------------------------------------------------------------
  // Частота среза
  // ------------------------------------------------------------------

  function suggestCutoffFrequency(vz, fs, duration, f1, gapRatio, gapSpanHz, fallbackMultiplier) {
    gapRatio = gapRatio === undefined ? 0.03 : gapRatio;
    gapSpanHz = gapSpanHz === undefined ? 0.3 : gapSpanHz;
    fallbackMultiplier = fallbackMultiplier === undefined ? 1.6 : fallbackMultiplier;

    if (vz.length < 8 || duration <= 0) return 1.0;

    const { freqs, spectrum } = fftSpectrum(vz, fs);
    const fMin = 2.0 / duration;
    let valid = [];
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= fMin && freqs[i] <= MAX_FIRST_MODE_FREQ_HZ) valid.push(i);
    }
    if (!valid.length) {
      for (let i = 0; i < freqs.length; i++) if (freqs[i] >= fMin) valid.push(i);
    }
    const maxValidAmp = valid.length ? Math.max(...valid.map((i) => spectrum[i])) : 0;
    if (!valid.length || maxValidAmp <= 0) return fMin * fallbackMultiplier;

    if (f1 === undefined || f1 === null) {
      let best = valid[0];
      for (const i of valid) if (spectrum[i] > spectrum[best]) best = i;
      f1 = freqs[best];
    }

    const threshold = gapRatio * maxValidAmp;
    const below = spectrum.map((s) => s < threshold);
    const candidateIdx = [];
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] > f1 && freqs[i] <= MAX_CUTOFF_SEARCH_HZ && below[i]) candidateIdx.push(i);
    }

    for (const i of candidateIdx) {
      const spanLo = freqs[i], spanHi = freqs[i] + gapSpanHz;
      let count = 0, allBelow = true;
      for (let j = 0; j < freqs.length; j++) {
        if (freqs[j] >= spanLo && freqs[j] <= spanHi) {
          count++;
          if (!below[j]) { allBelow = false; break; }
        }
      }
      if (count > 3 && allBelow) return freqs[i];
    }
    return f1 * fallbackMultiplier;
  }

  // ------------------------------------------------------------------
  // Низкочастотный фильтр Баттерворта 2-го порядка (filtfilt)
  // ------------------------------------------------------------------

  function butterLowpass2Coeffs(cutoffHz, fs) {
    const nyq = fs / 2.0;
    const cutoff = Math.min(Math.max(cutoffHz, 1e-6), nyq * 0.98);
    const w0 = (2 * Math.PI * cutoff) / fs;
    const Q = Math.SQRT1_2;
    const alpha = Math.sin(w0) / (2 * Q);
    const cosw0 = Math.cos(w0);
    const a0 = 1 + alpha;
    const b0 = (1 - cosw0) / 2 / a0;
    const b1 = (1 - cosw0) / a0;
    const b2 = (1 - cosw0) / 2 / a0;
    const a1 = (-2 * cosw0) / a0;
    const a2 = (1 - alpha) / a0;
    return { b0, b1, b2, a1, a2 };
  }

  function biquadFilter(x, c) {
    const n = x.length;
    const y = new Float64Array(n);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < n; i++) {
      const xn = x[i];
      const yn = c.b0 * xn + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
      y[i] = yn;
      x2 = x1; x1 = xn; y2 = y1; y1 = yn;
    }
    return y;
  }

  function filtfiltBiquad(x, c) {
    const padlen = 6;
    const n = x.length;
    if (n <= padlen) return Float64Array.from(x);
    const left = new Float64Array(padlen);
    for (let i = 0; i < padlen; i++) left[i] = 2 * x[0] - x[padlen - i];
    const right = new Float64Array(padlen);
    for (let i = 0; i < padlen; i++) right[i] = 2 * x[n - 1] - x[n - 2 - i];
    const padded = new Float64Array(padlen * 2 + n);
    padded.set(left, 0);
    padded.set(x, padlen);
    padded.set(right, padlen + n);

    let y = biquadFilter(padded, c);
    y = y.reverse();
    y = biquadFilter(y, c);
    y = y.reverse();
    return y.slice(padlen, padlen + n);
  }

  function lowpassFilter(vz, fs, cutoffHz, order) {
    // order фиксирован на 2 (как в Python-версии); параметр оставлен для совместимости сигнатуры.
    const c = butterLowpass2Coeffs(cutoffHz, fs);
    return filtfiltBiquad(vz, c);
  }

  // ------------------------------------------------------------------
  // Границы "горба" (от нуля до нуля) и скорость
  // ------------------------------------------------------------------

  function findZeroCrossingBounds(ts, curve, tPeak) {
    let iPeak = 0, best = Infinity;
    for (let i = 0; i < ts.length; i++) {
      const d = Math.abs(ts[i] - tPeak);
      if (d < best) { best = d; iPeak = i; }
    }
    const n = curve.length;
    const signPeak = curve[iPeak] >= 0 ? 1 : -1;
    let iLeft = iPeak;
    while (iLeft > 0 && Math.sign(curve[iLeft - 1]) === signPeak) iLeft--;
    let iRight = iPeak;
    while (iRight < n - 1 && Math.sign(curve[iRight + 1]) === signPeak) iRight++;
    return [ts[iLeft], ts[iRight]];
  }

  function computeVehicleSpeed(spanLengthM, wheelbaseM, crossingPeriodS) {
    if (crossingPeriodS <= 0) throw new TensogramError("Период прохождения (время между нулями) должен быть положительным.");
    if (spanLengthM <= 0) throw new TensogramError("Длина пролёта должна быть положительной.");
    if (wheelbaseM < 0) throw new TensogramError("База автомобиля не может быть отрицательной.");
    const distanceM = spanLengthM + wheelbaseM;
    const speedMs = distanceM / crossingPeriodS;
    return { speedMs, speedKmh: speedMs * 3.6 };
  }

  function computeSpeedInfo(spanLengthM, wheelbaseM, designSpeedKmh, crossingPeriodS) {
    const { speedMs, speedKmh } = computeVehicleSpeed(spanLengthM, wheelbaseM, crossingPeriodS);
    return {
      spanLengthM, wheelbaseM, designSpeedKmh, crossingPeriodS,
      actualSpeedMs: speedMs, actualSpeedKmh: speedKmh,
    };
  }

  function expectedFirstModeFrequency(spanLengthM, wheelbaseM, designSpeedKmh) {
    if (spanLengthM == null || wheelbaseM == null || designSpeedKmh == null) return null;
    if (!(spanLengthM > 0) || wheelbaseM < 0 || !(designSpeedKmh > 0)) return null;
    const distanceM = spanLengthM + wheelbaseM;
    const speedMs = designSpeedKmh / 3.6;
    const expectedPeriodS = distanceM / speedMs;
    if (!(expectedPeriodS > 0)) return null;
    return 1.0 / expectedPeriodS;
  }

  // ------------------------------------------------------------------
  // Полный расчёт по участку
  // ------------------------------------------------------------------

  const DEFAULT_NEIGHBORHOOD_S = 1.0;

  function analyzeSegment(ts, vs, opts) {
    opts = opts || {};
    const baselineDur = opts.baselineDur === undefined ? 2.0 : opts.baselineDur;
    let cutoffHz = opts.cutoffHz;
    const cutoffMultiplier = opts.cutoffMultiplier === undefined ? 1.6 : opts.cutoffMultiplier;
    let neighborhoodS = opts.neighborhoodS;
    const expectedFirstModeFreq = opts.expectedFirstModeFreq === undefined ? null : opts.expectedFirstModeFreq;

    if (ts.length < 8) throw new TensogramError("Недостаточно точек для анализа участка.");

    const baseline = segmentBaseline(ts, vs, baselineDur);
    const vz = new Float64Array(ts.length);
    for (let i = 0; i < ts.length; i++) vz[i] = vs[i] - baseline;

    const fs = samplingRate(ts);
    const duration = ts[ts.length - 1] - ts[0];

    const candidates = firstModeCandidates(vz, fs, duration);
    const f1 = chooseFirstModeFrequency(candidates, duration, 0.2, expectedFirstModeFreq);

    if (cutoffHz == null || cutoffHz <= 0) {
      cutoffHz = suggestCutoffFrequency(vz, fs, duration, f1, undefined, undefined, cutoffMultiplier);
    }

    const staticCurve = lowpassFilter(vz, fs, cutoffHz);

    let iStatic = 0;
    for (let i = 1; i < staticCurve.length; i++) {
      if (Math.abs(staticCurve[i]) > Math.abs(staticCurve[iStatic])) iStatic = i;
    }
    const staticMax = staticCurve[iStatic];
    const tStaticMax = ts[iStatic];

    if (neighborhoodS == null || neighborhoodS <= 0) neighborhoodS = DEFAULT_NEIGHBORHOOD_S;

    const localIdx = [];
    for (let i = 0; i < ts.length; i++) {
      if (Math.abs(ts[i] - tStaticMax) <= neighborhoodS) localIdx.push(i);
    }
    const pool = localIdx.length ? localIdx : ts.map((_, i) => i);
    let iDyn = pool[0];
    for (const i of pool) if (Math.abs(vz[i]) > Math.abs(vz[iDyn])) iDyn = i;
    const dynamicMax = vz[iDyn];
    const tDynamicMax = ts[iDyn];

    const dynamicCoefficient = staticMax === 0 ? NaN : Math.abs(dynamicMax) / Math.abs(staticMax);

    const [tHumpStart, tHumpEnd] = findZeroCrossingBounds(ts, staticCurve, tStaticMax);

    return {
      baseline,
      firstModeFreq: f1,
      firstModePeriod: f1 > 0 ? 1.0 / f1 : Infinity,
      firstModeCandidates: candidates,
      expectedFirstModeFreq,
      cutoffHz,
      neighborhoodS,
      staticMax, tStaticMax,
      dynamicMax, tDynamicMax,
      dynamicCoefficient,
      tHumpStart, tHumpEnd,
      crossingPeriodS: tHumpEnd - tHumpStart,
      ts, vs, vz, staticCurve,
    };
  }

  // ------------------------------------------------------------------
  // Спектральный анализ (частоты собственных колебаний)
  // ------------------------------------------------------------------

  const DEFAULT_SPECTRAL_DELAY_S = 5.0;
  const DEFAULT_SPECTRAL_WINDOW_S = 10.0;
  const DEFAULT_SPECTRAL_SWEEP_STEP_S = 0.5;

  function spectralAnalysis(tFull, vFull, baseline, tReference, delayS, windowS, nPeaks) {
    delayS = delayS === undefined ? DEFAULT_SPECTRAL_DELAY_S : delayS;
    windowS = windowS === undefined ? DEFAULT_SPECTRAL_WINDOW_S : windowS;
    nPeaks = nPeaks === undefined ? 5 : nPeaks;
    if (delayS < 0) throw new TensogramError("Задержка от максимума не может быть отрицательной.");
    if (windowS <= 0) throw new TensogramError("Длительность окна спектрального анализа должна быть положительной.");

    const tStart = tReference + delayS;
    const tEnd = tStart + windowS;
    const { t: tw, v: vw } = extractWindow(tFull, vFull, tStart, tEnd);
    const vzw = new Float64Array(tw.length);
    for (let i = 0; i < tw.length; i++) vzw[i] = vw[i] - baseline;

    const fs = samplingRate(tw);
    const { freqs, spectrum } = fftSpectrum(vzw, fs);

    const validIdx = [];
    for (let i = 0; i < freqs.length; i++) if (freqs[i] > 0) validIdx.push(i);

    let peakFreqs = [], peakAmps = [];
    if (validIdx.length) {
      let peakIdx = findLocalMaxima(spectrum).filter((i) => freqs[i] > 0);
      if (!peakIdx.length) {
        let best = validIdx[0];
        for (const i of validIdx) if (spectrum[i] > spectrum[best]) best = i;
        peakIdx = [best];
      }
      peakIdx.sort((a, b) => spectrum[b] - spectrum[a]);
      peakIdx = peakIdx.slice(0, nPeaks);
      peakIdx.sort((a, b) => freqs[a] - freqs[b]);
      peakFreqs = peakIdx.map((i) => freqs[i]);
      peakAmps = peakIdx.map((i) => spectrum[i]);
    }

    return {
      tStart, tEnd, delayS, windowS,
      ts: tw, vz: vzw, freqs, spectrum,
      peakFreqs, peakAmps,
    };
  }

  function sweepSpectralDelays(tFull, vFull, baseline, tReference, windowS, delayStartS, delayEndS, delayStepS, nPeaks) {
    delayStepS = delayStepS === undefined ? DEFAULT_SPECTRAL_SWEEP_STEP_S : delayStepS;
    nPeaks = nPeaks === undefined ? 5 : nPeaks;
    if (delayStepS <= 0) throw new TensogramError("Шаг изменения задержки должен быть положительным.");
    if (delayEndS < delayStartS) throw new TensogramError("Конечная задержка не может быть меньше начальной.");

    const results = [];
    const nSteps = Math.round((delayEndS - delayStartS) / delayStepS);
    for (let i = 0; i <= nSteps; i++) {
      const delay = delayStartS + i * delayStepS;
      try {
        results.push(spectralAnalysis(tFull, vFull, baseline, tReference, delay, windowS, nPeaks));
      } catch (e) {
        if (!(e instanceof TensogramError)) throw e;
      }
    }
    if (!results.length) {
      throw new TensogramError("Не удалось выполнить развёртку по задержке — окно спектра выходит за пределы записи для всех заданных значений задержки.");
    }
    return results;
  }

  // ------------------------------------------------------------------
  // Экспорт
  // ------------------------------------------------------------------

  const TensoCore = {
    TensogramError,
    parseTextTensogram,
    parseAnpHeader,
    formatAnpComment,
    parseAnaBinary,
    maskNoDataRuns,
    DEFAULT_NO_DATA_MIN_DURATION_S,
    extractWindow,
    segmentBaseline,
    samplingRate,
    fftSpectrum,
    linearDetrend,
    hanningWindow,
    MAX_FIRST_MODE_FREQ_HZ,
    MAX_CUTOFF_SEARCH_HZ,
    firstModeCandidates,
    chooseFirstModeFrequency,
    detectFirstModeFrequency,
    suggestCutoffFrequency,
    lowpassFilter,
    findZeroCrossingBounds,
    computeVehicleSpeed,
    computeSpeedInfo,
    expectedFirstModeFrequency,
    DEFAULT_NEIGHBORHOOD_S,
    analyzeSegment,
    DEFAULT_SPECTRAL_DELAY_S,
    DEFAULT_SPECTRAL_WINDOW_S,
    DEFAULT_SPECTRAL_SWEEP_STEP_S,
    spectralAnalysis,
    sweepSpectralDelays,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = TensoCore;
  } else {
    root.TensoCore = TensoCore;
  }
})(typeof window !== "undefined" ? window : globalThis);

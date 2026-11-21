/* app.js — UI-логика веб-версии анализатора тензограмм (использует core.js + Plotly). */
(function () {
  "use strict";
  const C = window.TensoCore;

  // ------------------------------------------------------------------
  // Состояние
  // ------------------------------------------------------------------
  let tFull = null, vFull = null, fileLabel = "";
  let tsSegment = null, vsSegment = null;
  let lastResult = null, lastSpectral = null, lastSpeedInfo = null;
  let lastSweepResults = null;
  let stage1ShowingFull = false;
  let fullPlotReady = false;

  // ------------------------------------------------------------------
  // DOM helpers
  // ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  function setStatus(text) { $("statusBar").textContent = text; }
  function parseNum(str) {
    const s = (str || "").trim();
    if (!s) return null;
    const v = parseFloat(s.replace(",", "."));
    return Number.isFinite(v) ? v : null;
  }
  function fmt(x, digits) {
    digits = digits === undefined ? 4 : digits;
    return Number.isFinite(x) ? x.toFixed(digits) : "—";
  }

  // ------------------------------------------------------------------
  // Вкладки
  // ------------------------------------------------------------------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "tab1" && tFull && !stage1ShowingFull) {
        renderFullPlot();
      }
    });
  });

  // ------------------------------------------------------------------
  // Этап 1: загрузка файла
  // ------------------------------------------------------------------
  $("fileBtn").addEventListener("click", () => $("fileInput").click());

  $("fileInput").addEventListener("change", async (ev) => {
    const files = Array.from(ev.target.files || []);
    if (!files.length) return;
    try {
      let result;
      const anaFile = files.find((f) => /\.ana$/i.test(f.name));
      const anpFile = files.find((f) => /\.anp$/i.test(f.name));
      if (anaFile || anpFile) {
        if (!anaFile || !anpFile) {
          throw new C.TensogramError(
            "Для формата .ana/.anp выберите ОБА файла сразу (зажмите Ctrl/Cmd при выборе)."
          );
        }
        const anpBuf = await anpFile.arrayBuffer();
        const anpText = new TextDecoder("windows-1251").decode(anpBuf);
        const header = C.parseAnpHeader(anpText);
        const fmtField = (header.FORMAT || "").trim().toLowerCase();
        if (!["f2", "f4", "float", "float32"].includes(fmtField)) {
          throw new C.TensogramError(`Неподдерживаемый формат данных .ana: FORMAT=${header.FORMAT}`);
        }
        const freqHz = parseFloat((header.FRQ || "").trim().split(/\s+/)[0]);
        if (!(freqHz > 0)) throw new C.TensogramError("В заголовке .anp не найдена частота дискретизации (FRQ).");
        const anaBuf = await anaFile.arrayBuffer();
        const parsed = C.parseAnaBinary(anaBuf, freqHz);
        result = { t: parsed.t, v: parsed.v, comment: C.formatAnpComment(header, anaFile.name) };
        fileLabel = `${anaFile.name} + ${anpFile.name}`;
      } else {
        const text = await files[0].text();
        const parsed = C.parseTextTensogram(text);
        result = { t: parsed.t, v: parsed.v, comment: null };
        fileLabel = files[0].name;
      }

      tFull = result.t;
      vFull = result.v;
      $("fileLabel").textContent = `${fileLabel} (${tFull.length} точек, ${tFull[0].toFixed(2)}..${tFull[tFull.length - 1].toFixed(2)} с)`;
      if (result.comment) $("comment").value = result.comment;

      tsSegment = null; vsSegment = null;
      lastResult = null; lastSpectral = null; lastSpeedInfo = null; lastSweepResults = null;
      $("tabBtn2").disabled = true; $("tabBtn3").disabled = true;
      $("saveCsvBtn").disabled = true; $("printReportBtn").disabled = true;

      renderFullPlot();
      const mid = tFull[Math.floor(tFull.length / 2)];
      $("eventTime").value = mid.toFixed(2);
      recomputeRangeFromConvenience();

      setStatus(
        "Файл загружен. Укажите время события и окна до/после, либо включите 'Выбор мышью' " +
        "и протяните участок на графике. Нажмите 'Выбрать', чтобы применить диапазон к анализу."
      );
    } catch (e) {
      alert("Ошибка загрузки: " + e.message);
    } finally {
      ev.target.value = "";
    }
  });

  // ------------------------------------------------------------------
  // Этап 1: исключение нулевых участков (только отображение)
  // ------------------------------------------------------------------
  function getDisplayVFull() {
    if (!$("excludeNoDataChk").checked) return vFull;
    const dur = parseNum($("excludeNoDataDur").value);
    return C.maskNoDataRuns(tFull, vFull, dur == null ? C.DEFAULT_NO_DATA_MIN_DURATION_S : dur);
  }
  $("excludeNoDataChk").addEventListener("change", () => { if (stage1ShowingFull) renderFullPlot(); });
  $("excludeNoDataDur").addEventListener("input", () => { if (stage1ShowingFull && $("excludeNoDataChk").checked) renderFullPlot(); });

  // ------------------------------------------------------------------
  // Этап 1: полный график + выбор мышью
  // ------------------------------------------------------------------
  function currentHighlightShape() {
    const from = parseNum($("rangeFrom").value);
    const to = parseNum($("rangeTo").value);
    if (from == null || to == null || to <= from) return [];
    return [{
      type: "rect", xref: "x", yref: "paper", x0: from, x1: to, y0: 0, y1: 1,
      fillcolor: "red", opacity: 0.15, line: { width: 0 },
    }];
  }

  function renderFullPlot() {
    stage1ShowingFull = true;
    const displayV = getDisplayVFull();
    const trace = {
      x: tFull, y: displayV, type: "scattergl", mode: "lines",
      line: { width: 1, color: "#1a6fb0" }, name: "запись",
    };
    const layout = {
      title: "Тензограмма (вся запись)",
      xaxis: { title: "Время, с" },
      yaxis: { title: "Величина" },
      dragmode: $("mouseSelectChk").checked ? "select" : "zoom",
      selectdirection: "h",
      shapes: currentHighlightShape(),
      margin: { t: 40, r: 20, b: 40, l: 55 },
    };
    if (!fullPlotReady) {
      Plotly.newPlot("fullPlot", [trace], layout, { responsive: true }).then(() => {
        fullPlotReady = true;
        document.getElementById("fullPlot").on("plotly_selected", onSpanSelected);
      });
    } else {
      Plotly.react("fullPlot", [trace], layout, { responsive: true });
    }
  }

  function updateHighlight() {
    if (!stage1ShowingFull || !fullPlotReady) return;
    Plotly.relayout("fullPlot", { shapes: currentHighlightShape() });
  }

  $("mouseSelectChk").addEventListener("change", () => {
    if (fullPlotReady) {
      Plotly.relayout("fullPlot", { dragmode: $("mouseSelectChk").checked ? "select" : "zoom" });
    }
    setStatus($("mouseSelectChk").checked
      ? "Режим выбора мышью включён: протяните область на графике полной записи."
      : "Режим выбора мышью выключен: доступны масштабирование и панорамирование.");
  });

  function onSpanSelected(eventData) {
    if (!eventData || !eventData.range || !eventData.range.x) return;
    const [xmin, xmax] = eventData.range.x;
    if (!(xmax > xmin)) return;
    const eventTime = 0.5 * (xmin + xmax);
    $("eventTime").value = eventTime.toFixed(3);
    $("windowBefore").value = (eventTime - xmin).toFixed(3);
    $("windowAfter").value = (xmax - eventTime).toFixed(3);
    $("rangeFrom").value = xmin.toFixed(3);
    $("rangeTo").value = xmax.toFixed(3);
    updateHighlight();
    setStatus(`Диапазон выбран мышью: ${xmin.toFixed(3)} .. ${xmax.toFixed(3)} с. Нажмите 'Выбрать'.`);
  }

  // ------------------------------------------------------------------
  // Этап 1: событие/окна -> диапазон; диапазон -> подсветка
  // ------------------------------------------------------------------
  function recomputeRangeFromConvenience() {
    const eventTime = parseNum($("eventTime").value);
    const before = parseNum($("windowBefore").value);
    const after = parseNum($("windowAfter").value);
    if (eventTime == null || before == null || after == null) return;
    $("rangeFrom").value = (eventTime - before).toFixed(3);
    $("rangeTo").value = (eventTime + after).toFixed(3);
    updateHighlight();
  }
  ["eventTime", "windowBefore", "windowAfter"].forEach((id) => {
    $(id).addEventListener("input", recomputeRangeFromConvenience);
  });
  ["rangeFrom", "rangeTo"].forEach((id) => {
    $(id).addEventListener("input", updateHighlight);
  });

  function parseVehicleParams() {
    return [parseNum($("spanLength").value), parseNum($("wheelbase").value), parseNum($("designSpeed").value)];
  }

  // ------------------------------------------------------------------
  // Этап 1: кнопка "Выбрать"
  // ------------------------------------------------------------------
  $("selectBtn").addEventListener("click", () => {
    if (!tFull) { alert("Сначала откройте файл тензограммы."); return; }
    const from = parseNum($("rangeFrom").value);
    const to = parseNum($("rangeTo").value);
    if (from == null || to == null) { alert("Границы диапазона должны быть числами."); return; }
    if (to <= from) { alert("Правая граница диапазона должна быть больше левой."); return; }
    try {
      const seg = C.extractWindow(tFull, vFull, from, to);
      tsSegment = seg.t; vsSegment = seg.v;
    } catch (e) {
      alert("Ошибка выбора участка: " + e.message);
      return;
    }
    lastResult = null; lastSpectral = null; lastSpeedInfo = null;
    $("saveCsvBtn").disabled = true; $("printReportBtn").disabled = true; $("tabBtn3").disabled = true;

    renderSegmentPreview();
    $("cutoffHz").value = ""; $("neighborhood").value = "1";
    $("resStatic").textContent = "Макс. статическое значение: —";
    $("resDynamic").textContent = "Макс. динамическое значение: —";
    $("resKd").textContent = "Динамический коэффициент: —";
    $("resSpectral").textContent = "Спектральный анализ: —";
    $("resSpeed").textContent = "Скорость движения: —";
    $("infoFirstMode").textContent = "Кандидаты в частоту первой формы появятся после анализа.";
    $("infoSpectral").textContent = "Метод определения частот и дискретизация спектра появятся после анализа.";

    $("tabBtn2").disabled = false;
    document.querySelector('.tab-btn[data-tab="tab2"]').click();
    stage1ShowingFull = false;
    setStatus(`Участок [${from.toFixed(3)} .. ${to.toFixed(3)}] с выбран. Настройте параметры расчёта и нажмите 'Анализ'.`);
  });

  function renderSegmentPreview() {
    $("previewPlot").style.display = "";
    $("resultsPlot").style.display = "none";
    Plotly.newPlot("previewPlot", [{
      x: Array.from(tsSegment), y: Array.from(vsSegment), type: "scattergl", mode: "lines",
      line: { width: 1, color: "#1a6fb0" },
    }], {
      title: "Выбранный участок тензограммы (предпросмотр — нажмите «Анализ»)",
      xaxis: { title: "Время, с" }, yaxis: { title: "Величина" },
      margin: { t: 40, r: 20, b: 40, l: 55 },
    }, { responsive: true });
  }

  // ------------------------------------------------------------------
  // Этап 2: параметры
  // ------------------------------------------------------------------
  function parseStage2Common() {
    const baselineDur = parseNum($("baselineDur").value);
    if (baselineDur == null) { alert("Длительность нулевого участка — число."); return null; }
    const cutoffTxt = $("cutoffHz").value.trim();
    let cutoffHz = null;
    if (cutoffTxt) {
      cutoffHz = parseNum(cutoffTxt);
      if (cutoffHz == null) { alert("Частота среза должна быть числом."); return null; }
    }
    const neighTxt = $("neighborhood").value.trim();
    let neighborhoodS = null;
    if (neighTxt) {
      neighborhoodS = parseNum(neighTxt);
      if (neighborhoodS == null) { alert("Окрестность должна быть числом."); return null; }
    }
    const specDelay = parseNum($("specDelay").value);
    const specWindow = parseNum($("specWindow").value);
    if (specDelay == null || specWindow == null) {
      alert("Параметры спектрального анализа (задержка, длительность окна) должны быть числами.");
      return null;
    }
    return { baselineDur, cutoffHz, neighborhoodS, specDelay, specWindow };
  }

  $("autoCutoffBtn").addEventListener("click", () => {
    if (!tsSegment) return;
    const params = parseStage2Common();
    if (!params) return;
    const [spanLength, wheelbase, designSpeed] = parseVehicleParams();
    const expectedFreq = C.expectedFirstModeFrequency(spanLength, wheelbase, designSpeed);
    try {
      const baseline = C.segmentBaseline(tsSegment, vsSegment, params.baselineDur);
      const vz = new Float64Array(tsSegment.length);
      for (let i = 0; i < tsSegment.length; i++) vz[i] = vsSegment[i] - baseline;
      const fs = C.samplingRate(tsSegment);
      const duration = tsSegment[tsSegment.length - 1] - tsSegment[0];
      const f1 = C.detectFirstModeFrequency(vz, fs, duration, undefined, expectedFreq);
      const cutoff = C.suggestCutoffFrequency(vz, fs, duration, f1);
      $("cutoffHz").value = cutoff.toFixed(4);
      const anchorNote = expectedFreq != null ? ` (ориентир по пролёту/скорости: ${expectedFreq.toFixed(4)} Гц)` : "";
      setStatus(`Автоопределение: частота первой формы ~${f1.toFixed(4)} Гц (период ~${(1 / f1).toFixed(2)} с)${anchorNote}, частота среза ~${cutoff.toFixed(4)} Гц.`);
    } catch (e) {
      alert("Ошибка: " + e.message);
    }
  });

  $("spectrumBarChk").addEventListener("change", () => {
    if (lastResult) renderResultsGrid(lastResult, lastSpectral);
  });

  // ------------------------------------------------------------------
  // Этап 2: Анализ
  // ------------------------------------------------------------------
  $("analyzeBtn").addEventListener("click", () => {
    if (!tsSegment) { alert("Сначала выберите участок на первом этапе."); return; }
    const params = parseStage2Common();
    if (!params) return;
    const [spanLength, wheelbase, designSpeed] = parseVehicleParams();
    const expectedFreq = C.expectedFirstModeFrequency(spanLength, wheelbase, designSpeed);

    let result;
    try {
      result = C.analyzeSegment(tsSegment, vsSegment, {
        baselineDur: params.baselineDur, cutoffHz: params.cutoffHz,
        neighborhoodS: params.neighborhoodS, expectedFirstModeFreq: expectedFreq,
      });
    } catch (e) {
      alert("Ошибка анализа: " + e.message);
      return;
    }

    let spectral = null;
    try {
      spectral = C.spectralAnalysis(tFull, vFull, result.baseline, result.tStaticMax, params.specDelay, params.specWindow);
    } catch (e) {
      console.warn("Спектральный анализ не выполнен:", e.message);
    }

    let speedInfo = null;
    if (spanLength != null && designSpeed != null) {
      try {
        speedInfo = C.computeSpeedInfo(spanLength, wheelbase || 0, designSpeed, result.crossingPeriodS);
      } catch (e) {
        console.warn("Скорость движения:", e.message);
      }
    }

    lastResult = result; lastSpectral = spectral; lastSpeedInfo = speedInfo;
    $("tabBtn3").disabled = false; $("saveCsvBtn").disabled = false; $("printReportBtn").disabled = false;

    $("resStatic").textContent = `Макс. статическое значение: ${fmt(result.staticMax)}  (t = ${fmt(result.tStaticMax, 3)} с)`;
    $("resDynamic").textContent = `Макс. динамическое значение: ${fmt(result.dynamicMax)}  (t = ${fmt(result.tDynamicMax, 3)} с)`;
    $("resKd").textContent = `Динамический коэффициент: ${fmt(result.dynamicCoefficient)}`;
    updateSpectralResultLabel(spectral);
    updateSpeedLabel(result, speedInfo);
    updateFirstModeInfo(result);

    if (!$("cutoffHz").value.trim()) $("cutoffHz").value = result.cutoffHz.toFixed(4);
    if (!$("neighborhood").value.trim()) $("neighborhood").value = result.neighborhoodS.toFixed(3);

    renderResultsGrid(result, spectral);
    setStatus(`Расчёт выполнен. Ноль участка = ${fmt(result.baseline)}; частота 1-й формы = ${fmt(result.firstModeFreq)} Гц (период ${fmt(result.firstModePeriod, 2)} с).`);
  });

  function updateFirstModeInfo(result) {
    const lines = [
      "Метод: локальные пики БПФ-спектра (после удаления линейного тренда),",
      `частоты до ${C.MAX_FIRST_MODE_FREQ_HZ.toFixed(2)} Гц.`,
    ];
    if (result.firstModeCandidates.length) {
      const txt = result.firstModeCandidates.map((c) => `${c[0].toFixed(4)} Гц (${c[1].toPrecision(3)})`).join(", ");
      lines.push(`Кандидаты (частота (амплитуда)): ${txt}.`);
    } else {
      lines.push("Значимых кандидатов не найдено.");
    }
    if (result.expectedFirstModeFreq != null) {
      lines.push(`Ожидание по пролёту/базе/скорости: ${result.expectedFirstModeFreq.toFixed(4)} Гц — выбран ближайший к нему кандидат.`);
    } else {
      lines.push("Выбрана наименьшая частота среди значимых кандидатов (укажите длину пролёта, базу и расчётную скорость на этапе 1, чтобы выбор опирался на физику проезда).");
    }
    lines.push(`Выбрано: ${result.firstModeFreq.toFixed(4)} Гц.`);
    $("infoFirstMode").textContent = lines.join("\n");
  }

  function updateSpeedLabel(result, speedInfo) {
    if (speedInfo) {
      $("resSpeed").textContent =
        `Скорость — расчётная: ${speedInfo.designSpeedKmh.toFixed(2)} км/ч (${(speedInfo.designSpeedKmh / 3.6).toFixed(2)} м/с); ` +
        `фактическая: ${speedInfo.actualSpeedKmh.toFixed(2)} км/ч (${speedInfo.actualSpeedMs.toFixed(2)} м/с) ` +
        `[период ${result.crossingPeriodS.toFixed(2)} с]`;
    } else {
      $("resSpeed").textContent = "Скорость движения: укажите длину пролёта и расчётную скорость на этапе 1";
    }
  }

  function updateSpectralResultLabel(spectral) {
    if (spectral && spectral.peakFreqs.length) {
      $("resSpectral").textContent = `Спектральный анализ — частоты, Гц: ${spectral.peakFreqs.map((f) => f.toFixed(3)).join(", ")}`;
    } else {
      $("resSpectral").textContent = "Спектральный анализ: не выполнен";
    }
    if (spectral) {
      const fs = C.samplingRate(spectral.ts);
      const dfRes = 1.0 / spectral.windowS;
      $("infoSpectral").textContent =
        "Метод: БПФ амплитудного спектра (удаление линейного тренда, окно Ханна)\n" +
        `Частота дискретизации: ${fs.toFixed(2)} Гц (${spectral.ts.length} отсчётов в окне ${spectral.windowS.toFixed(2)} с)\n` +
        `Дискретизация по частоте (разрешение спектра): Δf = 1/${spectral.windowS.toFixed(2)} с ≈ ${dfRes.toFixed(4)} Гц\n` +
        `Частота Найквиста: ${(fs / 2).toFixed(2)} Гц`;
    } else {
      $("infoSpectral").textContent = "Метод определения частот и дискретизация спектра появятся после анализа.";
    }
  }

  $("refreshSpectralBtn").addEventListener("click", () => {
    if (!lastResult) { alert("Сначала выполните основной анализ ('Анализ')."); return; }
    const specDelay = parseNum($("specDelay").value);
    const specWindow = parseNum($("specWindow").value);
    if (specDelay == null || specWindow == null) { alert("Задержка и длительность окна должны быть числами."); return; }
    try {
      lastSpectral = C.spectralAnalysis(tFull, vFull, lastResult.baseline, lastResult.tStaticMax, specDelay, specWindow);
    } catch (e) {
      alert("Ошибка: " + e.message);
      lastSpectral = null;
    }
    updateSpectralResultLabel(lastSpectral);
    renderResultsGrid(lastResult, lastSpectral);
    if (lastSpectral) setStatus(`Спектр пересчитан для окна ${lastSpectral.tStart.toFixed(2)}..${lastSpectral.tEnd.toFixed(2)} с.`);
  });

  // ------------------------------------------------------------------
  // Этап 2: графики результатов (2x2)
  // ------------------------------------------------------------------
  function zeroCrossingZoomRange(result) {
    let tLeft = result.tHumpStart, tRight = result.tHumpEnd;
    const span = Math.max(tRight - tLeft, 2 * result.neighborhoodS, 1.0);
    const pad = 0.12 * span;
    tLeft = Math.min(tLeft, result.tDynamicMax - result.neighborhoodS) - pad;
    tRight = Math.max(tRight, result.tDynamicMax + result.neighborhoodS) + pad;
    return [tLeft, tRight];
  }

  function staticDynamicTraces(result, xref, yref, axisSuffix) {
    const ts = Array.from(result.ts), vz = Array.from(result.vz), sc = Array.from(result.staticCurve);
    return [
      { x: ts, y: vz, type: "scattergl", mode: "lines", line: { width: 1, color: "#1a6fb0" }, name: "сигнал", xaxis: axisSuffix.x, yaxis: axisSuffix.y, showlegend: false },
      { x: ts, y: sc, type: "scattergl", mode: "lines", line: { width: 2, color: "#e07b00" }, name: "1-я форма", xaxis: axisSuffix.x, yaxis: axisSuffix.y, showlegend: false },
      { x: [result.tStaticMax], y: [result.staticMax], mode: "markers", marker: { size: 10, color: "#e07b00" }, name: "макс. статическое", xaxis: axisSuffix.x, yaxis: axisSuffix.y, showlegend: false },
      { x: [result.tDynamicMax], y: [result.dynamicMax], mode: "markers", marker: { size: 11, color: "#c0392b", symbol: "triangle-up" }, name: "макс. динамическое", xaxis: axisSuffix.x, yaxis: axisSuffix.y, showlegend: false },
    ];
  }

  function neighborhoodShape(result, axisSuffix) {
    return { type: "rect", xref: axisSuffix.x, yref: axisSuffix.y + " domain", x0: result.tStaticMax - result.neighborhoodS, x1: result.tStaticMax + result.neighborhoodS, y0: 0, y1: 1, fillcolor: "green", opacity: 0.12, line: { width: 0 } };
  }

  function spectralShape(spectral, axisSuffix) {
    if (!spectral) return null;
    return { type: "rect", xref: axisSuffix.x, yref: axisSuffix.y + " domain", x0: spectral.tStart, x1: spectral.tEnd, y0: 0, y1: 1, fillcolor: "purple", opacity: 0.12, line: { width: 0 } };
  }

  function renderResultsGrid(result, spectral) {
    const data = [];
    const shapes = [];
    const annotations = [];

    // --- subplot 1 (overview) ---
    data.push(...staticDynamicTraces(result, "x", "y", { x: "x", y: "y" }));
    shapes.push(neighborhoodShape(result, { x: "x", y: "y" }));
    const specShape1 = spectralShape(spectral, { x: "x", y: "y" });
    if (specShape1) shapes.push(specShape1);

    // --- subplot 2 (zoom, zero-to-zero) ---
    data.push(...staticDynamicTraces(result, "x2", "y2", { x: "x2", y: "y2" }));
    shapes.push(neighborhoodShape(result, { x: "x2", y: "y2" }));
    const [zLeft, zRight] = zeroCrossingZoomRange(result);

    // --- subplot 3 (spectral window, time domain) ---
    if (spectral) {
      data.push({
        x: Array.from(spectral.ts), y: Array.from(spectral.vz), type: "scattergl", mode: "lines",
        line: { width: 1, color: "#7a3fa0" }, xaxis: "x3", yaxis: "y3", showlegend: false,
      });
    }

    // --- subplot 4 (spectrum) ---
    let maxFreqShown = 10;
    if (spectral) {
      const style = $("spectrumBarChk").checked ? "bar" : "line";
      if (spectral.peakFreqs.length) {
        maxFreqShown = Math.min(spectral.freqs[spectral.freqs.length - 1], Math.max(...spectral.peakFreqs) * 2.5);
      } else {
        maxFreqShown = spectral.freqs[spectral.freqs.length - 1];
      }
      maxFreqShown = Math.max(maxFreqShown, 1.0);
      const freqs = [], amps = [];
      for (let i = 0; i < spectral.freqs.length; i++) {
        if (spectral.freqs[i] <= maxFreqShown) { freqs.push(spectral.freqs[i]); amps.push(spectral.spectrum[i]); }
      }
      const spectrumTrace = { x: freqs, y: amps, xaxis: "x4", yaxis: "y4", showlegend: false };
      if (style === "bar") {
        spectrumTrace.type = "bar";
        spectrumTrace.marker = { color: "#7a3fa0" };
      } else {
        spectrumTrace.type = "scattergl";
        spectrumTrace.mode = "lines";
        spectrumTrace.line = { width: 1, color: "#7a3fa0" };
      }
      data.push(spectrumTrace);
      // пики: разнесение подписей по уровням (минимизация наложения)
      const minGap = 0.11 * maxFreqShown;
      const placed = [];
      spectral.peakFreqs.forEach((f, idx) => {
        if (f > maxFreqShown) return;
        let level = 0;
        while (placed.some((p) => p.level === level && Math.abs(p.f - f) < minGap)) level++;
        placed.push({ f, level });
        data.push({ x: [f], y: [spectral.peakAmps[idx]], mode: "markers", marker: { size: 8, color: "#c0392b", symbol: "triangle-down" }, xaxis: "x4", yaxis: "y4", showlegend: false });
        annotations.push({
          x: f, y: 0.94 - level * 0.085, xref: "x4", yref: "y4 domain",
          text: f.toFixed(3), showarrow: false, font: { size: 10, color: "#c0392b" },
        });
      });
    }

    const layout = {
      grid: { rows: 2, columns: 2, pattern: "independent" },
      xaxis: { title: "Время, с", domain: [0, 0.46] },
      yaxis: { title: "Величина (от нуля)", domain: [0.56, 1] },
      xaxis2: { title: "Время, с", domain: [0.54, 1], range: [zLeft, zRight] },
      yaxis2: { title: "Величина (от нуля)", domain: [0.56, 1] },
      xaxis3: { title: "Время, с", domain: [0, 0.46] },
      yaxis3: { title: "Величина (от нуля)", domain: [0, 0.44] },
      xaxis4: { title: "Частота, Гц", domain: [0.54, 1], range: [0, maxFreqShown] },
      yaxis4: { title: "Амплитуда", domain: [0, 0.44] },
      shapes, annotations,
      margin: { t: 30, r: 20, b: 40, l: 55 },
      showlegend: false,
      title: "Весь участок (слева) / увеличенный вид от нуля до нуля (справа) / окно спектра и спектр (внизу)",
    };
    $("previewPlot").style.display = "none";
    $("resultsPlot").style.display = "";
    Plotly.newPlot("resultsPlot", data, layout, { responsive: true });
  }

  // ------------------------------------------------------------------
  // Этап 2: экспорт
  // ------------------------------------------------------------------
  function downloadBlob(filename, content, mime) {
    const blob = new Blob([content], { type: mime || "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(a.href); }, 1000);
  }

  function csvEscape(s) { return '"' + String(s).replace(/"/g, '""') + '"'; }

  function buildSummaryRows() {
    const r = lastResult;
    const rows = [
      ["Название точки", $("pointName").value],
      ["Комментарий", $("comment").value],
      ["Дата/время формирования отчёта", new Date().toLocaleString("ru-RU")],
      ["", ""],
      ["Ноль участка (baseline)", r.baseline],
      ["Частота первой формы, Гц", r.firstModeFreq],
      ["Период первой формы, с", r.firstModePeriod],
      ["Частота среза фильтра, Гц", r.cutoffHz],
      ["Окрестность поиска динамики, с", r.neighborhoodS],
      ["Кандидаты частоты первой формы (частота (амплитуда))", r.firstModeCandidates.map((c) => `${c[0].toFixed(4)} Гц (${c[1].toPrecision(4)})`).join(", ")],
    ];
    if (r.expectedFirstModeFreq != null) rows.push(["Ожидаемая частота первой формы, Гц", r.expectedFirstModeFreq]);
    rows.push(
      ["", ""],
      ["Максимальное статическое значение", r.staticMax],
      ["Время максимального статического значения, с", r.tStaticMax],
      ["Максимальное динамическое значение", r.dynamicMax],
      ["Время максимального динамического значения, с", r.tDynamicMax],
      ["Динамический коэффициент", r.dynamicCoefficient],
      ["", ""],
      ["Границы горба (от нуля до нуля), с", `${r.tHumpStart} .. ${r.tHumpEnd}`],
      ["Период прохождения (время между нулями), с", r.crossingPeriodS]
    );
    if (lastSpeedInfo) {
      rows.push(
        ["Длина пролёта, м", lastSpeedInfo.spanLengthM],
        ["База автомобиля, м", lastSpeedInfo.wheelbaseM],
        ["Расчётная скорость", `${lastSpeedInfo.designSpeedKmh} км/ч (${(lastSpeedInfo.designSpeedKmh / 3.6).toFixed(4)} м/с)`],
        ["Фактическая скорость", `${lastSpeedInfo.actualSpeedKmh.toFixed(4)} км/ч (${lastSpeedInfo.actualSpeedMs.toFixed(4)} м/с)`]
      );
    }
    if (lastSpectral) {
      rows.push(
        ["", ""],
        ["Задержка от максимума для спектра, с", lastSpectral.delayS],
        ["Длительность окна спектра, с", lastSpectral.windowS],
        ["Окно спектрального анализа, с", `${lastSpectral.tStart.toFixed(3)} .. ${lastSpectral.tEnd.toFixed(3)}`]
      );
      lastSpectral.peakFreqs.forEach((f, i) => {
        rows.push([`Пик спектра #${i + 1}, Гц (амплитуда)`, `${f.toFixed(4)} (${lastSpectral.peakAmps[i].toPrecision(4)})`]);
      });
    }
    return rows;
  }

  $("saveCsvBtn").addEventListener("click", () => {
    if (!lastResult) return;
    const name = ($("pointName").value || "точка_расчёта").replace(/[\\/:*?"<>|]+/g, "_");

    const valuesCsv = ["Параметр;Значение", ...buildSummaryRows().map((r) => `${csvEscape(r[0])};${csvEscape(r[1])}`)].join("\r\n");
    downloadBlob(`${name}_values.csv`, "﻿" + valuesCsv);

    const r = lastResult;
    const segRows = ["t, с;измеренное значение;значение от нуля участка;1-я форма (статика)"];
    for (let i = 0; i < r.ts.length; i++) segRows.push(`${r.ts[i]};${r.vs[i]};${r.vz[i]};${r.staticCurve[i]}`);
    setTimeout(() => downloadBlob(`${name}_segment_data.csv`, "﻿" + segRows.join("\r\n")), 300);

    if (lastSpectral) {
      const specRows = ["частота, Гц;амплитуда"];
      for (let i = 0; i < lastSpectral.freqs.length; i++) specRows.push(`${lastSpectral.freqs[i]};${lastSpectral.spectrum[i]}`);
      setTimeout(() => downloadBlob(`${name}_spectrum.csv`, "﻿" + specRows.join("\r\n")), 600);
    }
  });

  $("printReportBtn").addEventListener("click", async () => {
    if (!lastResult) return;
    const title = $("pointName").value.trim() || "(без названия)";
    const comment = $("comment").value.trim();
    const rowsHtml = buildSummaryRows().map((r) => r[0] === "" ? "<tr><td colspan=2>&nbsp;</td></tr>" : `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
    let imgHtml = "";
    try {
      const url = await Plotly.toImage("resultsPlot", { format: "png", width: 1100, height: 800 });
      imgHtml = `<h2>Графики</h2><img src="${url}" style="width:100%;">`;
    } catch (e) { /* ignore */ }

    const win = window.open("", "_blank");
    win.document.write(`<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт — ${title}</title>
      <style>
        body{font-family:Arial,sans-serif;padding:20px;color:#111;}
        table{border-collapse:collapse;width:100%;margin-top:10px;}
        td{border:1px solid #ccc;padding:4px 8px;font-size:13px;}
        h1{font-size:18px;} h2{font-size:15px;margin-top:20px;}
      </style></head><body>
      <h1>Отчёт по анализу тензограммы: ${title}</h1>
      ${comment ? `<p><b>Комментарий:</b> ${comment.replace(/\n/g, "<br>")}</p>` : ""}
      <table>${rowsHtml}</table>
      ${imgHtml}
      </body></html>`);
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 400);
  });

  // ------------------------------------------------------------------
  // Этап 3: развёртка
  // ------------------------------------------------------------------
  function parseSweepFilters() {
    return [parseNum($("sweepFreqMin").value), parseNum($("sweepFreqMax").value), parseNum($("sweepAmpMax").value)];
  }

  function renderSweepResults() {
    const [freqMin, freqMax, ampMax] = parseSweepFilters();
    const tbody = document.querySelector("#sweepTable tbody");
    tbody.innerHTML = "";
    const points = [];

    lastSweepResults.forEach((res) => {
      const peaks = [];
      for (let i = 0; i < res.peakFreqs.length; i++) {
        const f = res.peakFreqs[i], a = res.peakAmps[i];
        if (freqMin != null && f < freqMin) continue;
        if (freqMax != null && f > freqMax) continue;
        if (ampMax != null && a > ampMax) continue;
        peaks.push([f, a]);
        points.push([res.delayS, f, a]);
      }
      const tr = document.createElement("tr");
      const cells = [res.delayS.toFixed(3)];
      for (let i = 0; i < 5; i++) cells.push(peaks[i] ? peaks[i][0].toFixed(3) : "");
      cells.forEach((c) => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
      tbody.appendChild(tr);
    });

    Plotly.purge("sweepPlot");
    if (points.length) {
      Plotly.newPlot("sweepPlot", [{
        x: points.map((p) => p[0]), y: points.map((p) => p[1]),
        mode: "markers", type: "scattergl",
        marker: { size: 9, color: points.map((p) => p[2]), colorscale: "Viridis", showscale: true, colorbar: { title: "Амплитуда" }, line: { width: 0.5, color: "#000" } },
      }], {
        xaxis: { title: "Задержка от максимума, с" }, yaxis: { title: "Частота пика, Гц" },
        title: "Развёртка пиков спектра по задержке",
        margin: { t: 40, r: 20, b: 40, l: 55 },
      }, { responsive: true });
    } else {
      Plotly.newPlot("sweepPlot", [], { title: "Нет данных", margin: { t: 40, r: 20, b: 40, l: 55 } });
    }

    let filterNote = "";
    const parts = [];
    if (freqMin != null) parts.push(`частота ≥ ${freqMin} Гц`);
    if (freqMax != null) parts.push(`частота ≤ ${freqMax} Гц`);
    if (ampMax != null) parts.push(`амплитуда ≤ ${ampMax}`);
    if (parts.length) filterNote = `; фильтр: ${parts.join(", ")}`;
    $("sweepStatus").textContent = `Развёртка: ${lastSweepResults.length} задержек, показано ${points.length} пиков${filterNote}.`;
  }

  $("runSweepBtn").addEventListener("click", () => {
    if (!lastResult) { alert("Сначала выполните основной анализ ('Анализ') на вкладке 2."); return; }
    const windowS = parseNum($("sweepWindow").value);
    const from = parseNum($("sweepFrom").value);
    const to = parseNum($("sweepTo").value);
    const step = parseNum($("sweepStep").value);
    if (windowS == null || from == null || to == null || step == null) {
      alert("Длина окна, задержки 'от'/'до' и шаг должны быть числами.");
      return;
    }
    try {
      lastSweepResults = C.sweepSpectralDelays(tFull, vFull, lastResult.baseline, lastResult.tStaticMax, windowS, from, to, step);
    } catch (e) {
      alert("Ошибка развёртки: " + e.message);
      return;
    }
    renderSweepResults();
  });

  $("applySweepFilterBtn").addEventListener("click", () => {
    if (!lastSweepResults) { alert("Сначала нажмите 'Запустить развёртку'."); return; }
    renderSweepResults();
  });
})();

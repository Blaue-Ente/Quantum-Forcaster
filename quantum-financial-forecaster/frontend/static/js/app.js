"use strict";

const state = {
    assets: { stocks: [], crypto: [], forex: [] },
    priceChart: null,
    modelChart: null,
    weightChart: null,
};

const els = {
    category: document.getElementById("category"),
    symbol: document.getElementById("symbol"),
    horizon: document.getElementById("horizon"),
    forecastBtn: document.getElementById("forecastBtn"),
    loader: document.getElementById("loader"),
    results: document.getElementById("results"),
    errorBox: document.getElementById("errorBox"),
    statusPill: document.getElementById("statusPill"),
    statusText: document.getElementById("statusText"),
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
    try {
        const res = await fetch("/api/assets");
        state.assets = await res.json();
        updateSymbolOptions();
        setOnlineStatus(true);
    } catch (err) {
        showError("Не може да се свърже със сървъра. Стартирайте backend първо.");
        setOnlineStatus(false);
    }

    els.category.addEventListener("change", updateSymbolOptions);
    els.forecastBtn.addEventListener("click", runForecast);
}

function updateSymbolOptions() {
    const cat = els.category.value;
    const symbols = state.assets[cat] || [];
    els.symbol.innerHTML = symbols
        .map((s) => `<option value="${s}">${s}</option>`)
        .join("");
}

function setOnlineStatus(isOnline) {
    if (isOnline) {
        els.statusPill.classList.add("online");
        els.statusText.textContent = "online";
    } else {
        els.statusPill.classList.remove("online");
        els.statusText.textContent = "offline";
    }
}

function showError(msg) {
    els.errorBox.textContent = msg;
    els.errorBox.classList.remove("hidden");
    els.results.classList.add("hidden");
}

function clearError() {
    els.errorBox.classList.add("hidden");
}

async function runForecast() {
    clearError();
    els.results.classList.add("hidden");
    els.loader.classList.remove("hidden");
    els.forecastBtn.disabled = true;

    const symbol = els.symbol.value;
    const horizon = parseInt(els.horizon.value, 10) || 7;
    const url = `/api/forecast?symbol=${encodeURIComponent(symbol)}&horizon=${horizon}`;

    try {
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Грешка при прогнозата");
        }
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        els.loader.classList.add("hidden");
        els.forecastBtn.disabled = false;
    }
}

function renderResults(data) {
    els.results.classList.remove("hidden");

    // Header
    document.getElementById("resSymbol").textContent = data.symbol;
    document.getElementById("resMeta").textContent =
        `Хоризонт: ${data.horizon_days} дни · Последна цена: ${data.last_close.toFixed(4)}`;

    const dirBadge = document.getElementById("directionBadge");
    dirBadge.className = `direction-badge ${data.direction}`;
    document.getElementById("directionText").textContent =
        data.direction === "UP" ? "▲ ВЪЗХОДЯЩ" : "▼ НИЗХОДЯЩ";

    // Metrics
    document.getElementById("lastClose").textContent = fmt(data.last_close);
    document.getElementById("projectedClose").textContent = fmt(data.projected_close);
    const changeEl = document.getElementById("projectedChange");
    changeEl.textContent = `${data.projected_change_pct.toFixed(2)}%`;
    changeEl.style.color = data.projected_change_pct >= 0 ? "var(--success)" : "var(--danger)";
    document.getElementById("probUp").textContent = `${(data.probability_up * 100).toFixed(1)}%`;
    document.getElementById("confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    const regimeEl = document.getElementById("regime");
    regimeEl.textContent = data.regime.regime;
    regimeEl.style.color = regimeColor(data.regime.regime);

    // Price chart
    renderPriceChart(data);

    // Quantum params
    renderQuantumParams(data.quantum);

    // Model comparison chart
    renderModelChart(data);

    // Weight chart
    renderWeightChart(data);

    // Accuracy
    document.getElementById("qAccuracy").textContent =
        `${(data.quantum_accuracy * 100).toFixed(1)}%`;
    document.getElementById("cAccuracy").textContent =
        `${(data.classical_accuracy * 100).toFixed(1)}%`;
}

function fmt(v) {
    if (v >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
    return v.toFixed(4);
}

function regimeColor(regime) {
    return { trending: "var(--success)", volatile: "var(--danger)", "mean-reverting": "var(--warning)" }[regime] || "var(--text)";
}

function renderPriceChart(data) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    if (state.priceChart) state.priceChart.destroy();

    const hist = data.history;
    const labels = hist.map((h) => h.date);
    const closes = hist.map((h) => h.close);

    // Добавяме проекцията като последна точка
    const lastDate = new Date(labels[labels.length - 1]);
    lastDate.setDate(lastDate.getDate() + data.horizon_days);
    labels.push(lastDate.toISOString().slice(0, 10));
    closes.push(data.projected_close);

    const projIdx = closes.length - 1;
    const histData = closes.map((v, i) => (i < projIdx ? v : null));
    const projData = closes.map((v, i) => (i >= projIdx - 1 ? v : null));

    state.priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "История",
                    data: histData,
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.1)",
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                },
                {
                    label: "Прогноза",
                    data: projData,
                    borderColor: "#a855f7",
                    borderDash: [6, 4],
                    tension: 0.3,
                    pointRadius: 3,
                    borderWidth: 2,
                    fill: false,
                },
            ],
        },
        options: chartOpts(),
    });
}

function renderQuantumParams(q) {
    document.getElementById("nQubits").textContent = q.n_qubits;
    document.getElementById("fmDepth").textContent = q.feature_map_depth;
    document.getElementById("ansatzDepth").textContent = q.ansatz_depth;

    // Визуализация на кубитите
    const qubitRow = document.getElementById("qubitRow");
    qubitRow.innerHTML = "";
    for (let i = 0; i < q.n_qubits; i++) {
        const line = document.createElement("div");
        line.className = "qubit-line";
        line.innerHTML = `
            <span class="qubit-label">|q${i}⟩</span>
            <div class="qubit-track">
                <span class="qubit-gate" style="left: 15%">H</span>
                <span class="qubit-gate" style="left: 50%">Ry</span>
                <span class="qubit-gate" style="left: 85%">M</span>
            </div>
        `;
        qubitRow.appendChild(line);
    }
}

function renderModelChart(data) {
    const ctx = document.getElementById("modelChart").getContext("2d");
    if (state.modelChart) state.modelChart.destroy();

    state.modelChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["VQC (квантов)", "Random Forest", "Gradient Boost", "Ансамбъл"],
            datasets: [{
                label: "P(↑)",
                data: [
                    data.quantum.probability_up,
                    data.classical.rf_proba,
                    data.classical.gb_proba,
                    data.probability_up,
                ],
                backgroundColor: ["#6366f1", "#10b981", "#f59e0b", "#a855f7"],
                borderRadius: 8,
            }],
        },
        options: {
            ...chartOpts(),
            scales: {
                y: { ...chartOpts().scales.y, min: 0, max: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
                x: { ...chartOpts().scales.x },
            },
            plugins: { legend: { display: false } },
        },
    });
}

function renderWeightChart(data) {
    const ctx = document.getElementById("weightChart").getContext("2d");
    if (state.weightChart) state.weightChart.destroy();

    const w = data.weights;
    state.weightChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Квантов", "Класически"],
            datasets: [{
                data: [w.quantum * 100, w.classical * 100],
                backgroundColor: ["#6366f1", "#10b981"],
                borderColor: "rgba(10, 14, 26, 0.6)",
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            cutout: "65%",
            plugins: {
                legend: { position: "bottom", labels: { color: "#e2e8f0" } },
                tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw.toFixed(1)}%` } },
            },
        },
    });

    const regime = data.regime.regime;
    document.getElementById("weightsText").textContent =
        `Режим: ${regime} · Тежести адаптирани според волатилността`;
}

function chartOpts() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: "#e2e8f0" } },
        },
        scales: {
            x: { ticks: { color: "#94a3b8", maxTicksLimit: 8 }, grid: { color: "rgba(148, 163, 184, 0.1)" } },
            y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148, 163, 184, 0.1)" } },
        },
    };
}

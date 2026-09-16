/* Dependency-free client for the local, read-only explorer API. */
(() => {
  "use strict";

  const stateSelect = document.getElementById("state-select");
  const scaleToggle = document.getElementById("scale-toggle");
  const seriesSearch = document.getElementById("series-search");
  const seriesList = document.getElementById("series-list");
  const seriesCount = document.getElementById("series-count");
  const versionDate = document.getElementById("version-date");
  const versionWeekday = document.getElementById("version-weekday");
  const versionPrev = document.getElementById("version-prev");
  const versionNext = document.getElementById("version-next");
  const versionLatest = document.getElementById("version-latest");
  const versionPrevWednesday = document.getElementById("version-prev-wednesday");
  const versionNextWednesday = document.getElementById("version-next-wednesday");
  const versionKeep = document.getElementById("version-keep");
  const versionStatus = document.getElementById("version-status");
  const comparisons = document.getElementById("comparisons");
  const clearButton = document.getElementById("clear-button");
  const hubPresets = document.getElementById("hub-presets");
  const signalFilters = document.getElementById("signal-filters");
  const cadenceFilter = document.getElementById("cadence-filter");
  const vintageFilter = document.getElementById("vintage-filter");
  const supportFilter = document.getElementById("support-filter");
  const freshnessFilter = document.getElementById("freshness-filter");
  const filterCount = document.getElementById("filter-count");
  const resetFilters = document.getElementById("reset-filters");
  const plotTitle = document.getElementById("plot-title");
  const plotStatus = document.getElementById("plot-status");
  const indexSummary = document.getElementById("index-summary");
  const indexDetails = document.getElementById("index-details");
  const legend = document.getElementById("legend");
  const chart = document.getElementById("chart");
  const rangeControl = document.getElementById("range-control");
  const rangeStart = document.getElementById("range-start");
  const rangeEnd = document.getElementById("range-end");
  const rangeReset = document.getElementById("range-reset");
  const rangeOverview = document.getElementById("range-overview");
  const DAY = 86400000;
  const dateTime = day => Date.parse(`${day}T00:00:00Z`);
  const isoDate = time => new Date(time).toISOString().slice(0, 10);
  const weekday = (day, style = "long") => new Date(`${day}T00:00:00Z`)
    .toLocaleDateString(undefined, {weekday: style, timeZone: "UTC"});
  // Calendar Wednesday strictly before/after a day (today when showing latest).
  const today = () => new Date().toISOString().slice(0, 10);
  const wednesday = (day, direction) => {
    const time = dateTime(day || today());
    const weekdayIndex = new Date(time).getUTCDay();
    const shift = direction < 0 ? -(((weekdayIndex - 3 + 7) % 7) || 7) : (((3 - weekdayIndex + 7) % 7) || 7);
    return isoDate(time + shift * DAY);
  };
  const versionLabel = day => day ? `${weekday(day, "short")} ${day}` : "Latest";
  const tooltip = document.getElementById("tooltip");

  const model = {
    state: "",
    stateName: "",
    search: "",
    filters: {cadence: "", vintage: "", support: "native_state", freshness: ""},
    asOf: "",
    stateSupport: "native_state",
    locationRequest: 0,
    versionDates: [],
    comparisons: [],
    versionRequest: 0,
    total: 0,
    available: [],
    selected: new Map(),
    hidden: new Set(),
    plotted: [],
    lineColors: new Map(),
    // View range is independent of publication cutoff and full-series scaling.
    viewRange: null,
    dateExtent: null,
    listRequest: 0,
    plotRequest: 0,
  };

  async function requestJSON(path) {
    const response = await fetch(path, {headers: {"Accept": "application/json"}});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function escapeHTML(value) {
    return String(value).replace(/[&<>'"]/g, character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[character]);
  }

  async function initialize() {
    try {
      const overview = await requestJSON("/api/catalog");
      const points = overview.datasets.reduce((sum, item) => sum + item.point_count, 0);
      const selection = overview.selection;
      indexSummary.textContent = selection
        ? `${selection.indexed_source_groups}/${selection.logical_source_groups} source groups · ${formatNumber(selection.indexed_signal_choices)} signals · ${formatNumber(points)} indexed values`
        : `${overview.datasets.length} datasets · ${formatNumber(points)} indexed values`;
      stateSelect.innerHTML = overview.states.map(item =>
        `<option value="${item.code}">${escapeHTML(item.name)}</option>`
      ).join("");
      const built = overview.meta.built_at ? new Date(overview.meta.built_at).toLocaleString() : "unknown";
      indexDetails.innerHTML = `<div>Built ${escapeHTML(built)} from ${escapeHTML(overview.meta.artifact_count || "0")} raw artifacts. The raw repository remains unchanged.</div><div>${escapeHTML(overview.aggregation_note)}</div>`;
      if (selection) {
        const summary = document.createElement("div");
        summary.textContent = `Selection policy ${selection.policy_version}: ${selection.raw_catalog_datasets} raw datasets → ${selection.logical_source_groups} logical groups. ${selection.indexed_source_groups} groups have indexed data. Missing downloads: ${selection.missing_datasets.join(", ") || "none"}.`;
        indexDetails.appendChild(summary);
      }
      for (const warning of overview.warnings.filter(item => item.status !== "excluded")) {
        const line = document.createElement("div");
        line.className = "warning";
        line.textContent = `${warning.dataset_key} · ${warning.source_path}: ${warning.message || warning.status}`;
        indexDetails.appendChild(line);
      }
      if (!overview.states.length) {
        stateSelect.innerHTML = '<option>No state-level data indexed</option>';
        seriesList.innerHTML = '<div class="empty-state"><span>No plottable state series were found. Open Index details for skipped formats or schemas.</span></div>';
        return;
      }
      stateSelect.disabled = false;
      model.state = stateSelect.value;
      model.stateName = stateSelect.options[stateSelect.selectedIndex].text.replace(/\s+\([\d,]+ columns\)$/, "");
      syncLocationSupport();
      await loadSeries(true);
    } catch (error) {
      indexSummary.textContent = "Could not load index";
      showError(error);
    }
  }

  async function loadSeries() {
    if (!model.state) return;
    const token = ++model.listRequest;
    seriesCount.textContent = "Loading selected signals…";
    try {
      const params = new URLSearchParams({state: model.state, q: model.search, limit: "0", ...model.filters});
      const payload = await requestJSON(`/api/series?${params}`);
      if (token !== model.listRequest) return;
      model.total = payload.total;
      model.available = payload.items;
      renderSeriesList();
    } catch (error) {
      if (token === model.listRequest) showError(error);
    }
  }

  async function loadVersions() {
    const token = ++model.versionRequest;
    model.versionDates = [];
    renderVersionControls();
    if (!model.selected.size) return;
    try {
      const params = new URLSearchParams({state: model.state, series: [...model.selected.keys()].join(",")});
      const payload = await requestJSON(`/api/versions?${params}`);
      if (token !== model.versionRequest) return;
      model.versionDates = payload.dates;
      renderVersionControls();
    } catch (error) {
      if (token === model.versionRequest) showError(error);
    }
  }

  function renderVersionControls() {
    versionDate.value = model.asOf;
    versionWeekday.textContent = model.asOf ? weekday(model.asOf) : "";
    versionPrev.disabled = !model.versionDates.some(day => !model.asOf || day < model.asOf);
    versionNext.disabled = !model.asOf;
    versionPrevWednesday.disabled = !model.selected.size;
    versionNextWednesday.disabled = !model.asOf || wednesday(model.asOf, 1) > today();
    versionKeep.disabled = !model.asOf || !model.selected.size || model.comparisons.includes(model.asOf) || model.comparisons.length >= 5;
    versionStatus.textContent = model.asOf ? `Solid: latest available · dashed: ${weekday(model.asOf)} ${model.asOf}` : "Latest available values · choose a date to compare";
    comparisons.innerHTML = model.comparisons.map((day, index) =>
      `<button type="button" data-comparison="${index}" title="Remove comparison">${escapeHTML(versionLabel(day))} <span aria-hidden="true">×</span><span class="sr-only">Remove comparison</span></button>`
    ).join("");
  }

  function setVersion(day) {
    model.asOf = day;
    renderVersionControls();
    updatePlot();
  }

  function renderSeriesList() {
    const previouslyOpen = new Set(
      [...seriesList.querySelectorAll("details[open][data-tree-key]")]
        .map(element => element.dataset.treeKey)
    );
    seriesCount.textContent = `${formatNumber(new Set(model.available.map(item => item.signal_key)).size)} signals · ${formatNumber(model.total)} variants`;
    if (!model.available.length) {
      seriesList.innerHTML = '<div class="empty-state"><span>No matching selected signals. Try another spatial support filter.</span></div>';
    } else {
      seriesList.innerHTML = renderSeriesTree(previouslyOpen);
    }
  }

  function renderSeriesTree(previouslyOpen) {
    // Pathogen → source → signal → variants; pathogens in policy rank, the rest by title.
    const pathogens = [...groupBy(model.available, item => item.pathogen).values()]
      .sort((a, b) => a[0].pathogen_rank - b[0].pathogen_rank || a[0].pathogen_title.localeCompare(b[0].pathogen_title));
    return pathogens.map(pathogenItems => {
      const pathogenKey = treeKey("pathogen", pathogenItems[0].pathogen);
      const sources = [...groupBy(pathogenItems, item => item.source_group).values()]
        .sort((a, b) => a[0].source_group_title.localeCompare(b[0].source_group_title));
      const children = sources.map(items => {
        const key = treeKey(pathogenKey, items[0].source_group);
        const signals = [...groupBy(items, item => item.signal_key).values()]
          .sort((a, b) => a[0].signal_title.localeCompare(b[0].signal_title));
        return renderBranch({className: "series-dataset", key,
          label: items[0].source_group_title, meta: providers(items), count: signals.length,
          open: shouldOpen(items, key, previouslyOpen), children: signals.map(renderSignal).join("")});
      }).join("");
      return renderBranch({className: "series-pathogen", key: pathogenKey,
        label: pathogenItems[0].pathogen_title,
        count: new Set(pathogenItems.map(item => item.signal_key)).size,
        open: shouldOpen(pathogenItems, pathogenKey, previouslyOpen), children});
    }).join("");
  }

  function renderSignal(variants) {
    variants.sort((a, b) => {
      for (let i = 0; i < a.variant_rank.length; i++) {
        if (a.variant_rank[i] !== b.variant_rank[i]) return a.variant_rank[i] - b.variant_rank[i];
      }
      return a.variant_label.localeCompare(b.variant_label);
    });
    return `<div class="selected-signal" data-signal-key="${escapeHTML(variants[0].signal_key)}">
      <strong class="signal-title" title="${escapeHTML(variants[0].column_description || "")}">${escapeHTML(variants[0].signal_title)}</strong>
      <div class="signal-variants">${variants.map(item => renderSeriesOption(item, item.variant_label)).join("")}</div>
    </div>`;
  }

  function providers(items) {
    const names = {cdc: "CDC", delphi: "Delphi", hub: "Forecast Hub"};
    return [...new Set(items.map(item => names[item.provider_kind] || item.provider_kind))].join(" · ");
  }

  function renderSeriesOption(item, label) {
    // One line per variant; provenance and coverage appear as a card once checked.
    const checked = model.selected.has(item.id);
    const providerClass = item.provider_kind === "delphi" ? " delphi-option"
      : item.provider_kind === "hub" ? " hub-option" : "";
    const source = `${item.measure_id || item.value_column} · ${item.parent_dataset || ""}${item.parent_column ? ` · ${item.parent_column}` : ""}`;
    const card = checked ? `<span class="series-card">
        <span>${formatNumber(item.point_count)} points · ${escapeHTML(item.date_min)} to ${escapeHTML(item.date_max)}</span>
        <span>${escapeHTML(signalMetadata(item))}</span>
        <span>${escapeHTML(source)}</span>
      </span>` : "";
    return `<label class="series-option${providerClass}${checked ? " is-checked" : ""}" title="${escapeHTML(`${label}\n${signalMetadata(item)}\n${source}\n${formatNumber(item.point_count)} points · ${item.date_min} to ${item.date_max}`)}">
      <input type="checkbox" data-series-id="${item.id}"${checked ? " checked" : ""}>
      <span class="series-option-label">${escapeHTML(label)}</span>${card}
    </label>`;
  }

  function renderBranch({className, key, label, meta = "", count, open, children}) {
    const metadata = meta ? `<small>${escapeHTML(meta)}</small>` : "";
    return `<details class="series-branch ${className}" data-tree-key="${escapeHTML(key)}"${open ? " open" : ""}>
      <summary><span>${escapeHTML(label)}${metadata}</span><b>${formatNumber(count)}</b></summary>
      <div class="series-children">${children}</div>
    </details>`;
  }

  // Hub truth has fixed colors so it reads as the reference in every plot.
  const HUB_TRUTH_COLORS = {nhsn: "#000000", nssp: "#dc2626"};
  const darkTheme = window.matchMedia("(prefers-color-scheme: dark)");
  function seriesColor(item) {
    // A series and all of its vintages share one color; line style marks the version.
    if (item.provider_kind === "hub" && HUB_TRUTH_COLORS[item.source_group]) {
      return item.source_group === "nhsn" && darkTheme.matches ? "#f1f5f9" : HUB_TRUTH_COLORS[item.source_group];
    }
    if (!model.lineColors.has(item.id)) {
      const palette = ["#2563eb", "#e87516", "#26934b", "#9333ea", "#db2777",
        "#0891b2", "#a88708", "#64748b", "#7c3d12", "#4f46e5"];
      const index = model.lineColors.size;
      model.lineColors.set(item.id, palette[index]
        || `hsl(${((index - palette.length) * 137.508 + 45) % 360} 65% 45%)`);
    }
    return model.lineColors.get(item.id);
  }

  function lineDash(item) {
    return ["none", "7 4", "2 3", "10 3 2 3", "12 5", "4 6"][item.versionIndex % 6];
  }

  function signalMetadata(item) {
    const vintage = item.revision_mode === "initial_release" ? "frozen first release"
      : item.versioned ? "revision archive" : "current snapshot";
    const support = item.spatial_support || "native state";
    return `${item.temporal_resolution} · ${vintage} · ${support} · ${item.freshness}`;
  }

  function groupBy(items, keyFunction) {
    const groups = new Map();
    for (const item of items) {
      const key = keyFunction(item);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    return groups;
  }

  function treeKey(parent, value) { return `${parent}/${encodeURIComponent(value)}`; }
  function shouldOpen(items, key, previouslyOpen) {
    return Boolean(model.search) || previouslyOpen.has(key) || items.some(item => model.selected.has(item.id));
  }

  async function updatePlot() {
    const token = ++model.plotRequest;
    const ids = [...model.selected.keys()];
    if (!ids.length) {
      model.plotted = [];
      model.viewRange = null;
      model.dateExtent = null;
      rangeControl.hidden = true;
      tooltip.hidden = true;
      legend.innerHTML = "";
      plotTitle.textContent = "Choose a location and one or more columns";
      plotStatus.textContent = "";
      chart.innerHTML = '<div class="empty-state"><strong>No series selected</strong><span>Select signals on the left to overlay them here.</span></div>';
      return;
    }
    plotStatus.textContent = "Loading…";
    chart.classList.add("is-loading");
    try {
      const versions = [...new Set(["", model.asOf, ...model.comparisons])];
      const payloads = await Promise.all(versions.map(asOf => {
        const params = new URLSearchParams({state: model.state, series: ids.join(","),
          scale: "false", as_of: asOf || "latest"});
        return requestJSON(`/api/data?${params}`);
      }));
      if (token !== model.plotRequest) return;
      chart.classList.remove("is-loading");
      // Every displayed version is divided by the same series' latest maximum, so a
      // dated curve with a shorter archived history stays comparable to latest.
      const divisors = new Map(payloads[0].series.map(item => [item.id, item.max || 1]));
      const divisor = item => scaleToggle.checked ? divisors.get(item.id) || item.max || 1 : 1;
      model.plotted = payloads.flatMap((payload, versionIndex) => payload.series.map(item => ({
        ...item, points: item.points.map(([day, value, samples]) => [day, value / divisor(item), samples]),
        key: `${item.id}@${versions[versionIndex] || "latest"}`,
        // Visibility follows the active comparison as its date moves; pinned dates
        // and the latest reference keep independent visibility choices.
        visibilityKey: `${item.id}@${versions[versionIndex] && versions[versionIndex] === model.asOf
          ? "as-of" : versions[versionIndex] || "latest"}`,
        versionIndex,
        label: `${item.signal_title || item.label} · ${item.variant_label || ""} · ${versionLabel(versions[versionIndex])}`,
      })));
      plotTitle.textContent = payloads[0].state_name;
      const count = model.plotted.reduce((sum, item) => sum + item.points.length, 0);
      const empty = model.plotted.filter(item => !item.points.length).length;
      plotStatus.textContent = `${ids.length} signals · ${versions.length} displayed version${versions.length === 1 ? "" : "s"} · ${formatNumber(count)} points${empty ? ` · ${empty} unavailable for this location/date` : ""}${scaleToggle.checked ? " · scaled by each series' latest maximum" : ""}`;
      renderLegend();
      drawChart();
    } catch (error) {
      if (token === model.plotRequest) chart.classList.remove("is-loading");
      showError(error);
    }
  }

  function renderLegend() {
    legend.innerHTML = model.plotted.map((item, index) => {
      const hidden = model.hidden.has(item.visibilityKey || item.key);
      const maximum = item.max == null ? "" : ` · max ${formatValue(item.max)}`;
      return `<button class="legend-item" type="button" data-legend-id="${item.visibilityKey || item.key}" aria-pressed="${hidden ? "false" : "true"}" title="${escapeHTML([item.label, item.parent_dataset, item.parent_column, item.parent_transform, item.version_note].filter(Boolean).join(" · "))}">
        <svg class="swatch" viewBox="0 0 22 6" aria-hidden="true"><line x1="0" x2="22" y1="3" y2="3" stroke="${seriesColor(item)}" stroke-width="2.5" stroke-dasharray="${lineDash(item)}"></line></svg>
        <span class="legend-label">${escapeHTML(item.label)}${maximum}</span>
      </button>`;
    }).join("");
  }

  function drawChart() {
    tooltip.hidden = true;
    renderRangeControl();
    const visible = model.plotted.filter(item => !model.hidden.has(item.visibilityKey || item.key) && item.points.length);
    if (!visible.length) {
      chart.innerHTML = '<div class="empty-state"><strong>No visible data</strong><span>No values available for this location/date, or all lines are hidden.</span></div>';
      return;
    }
    const width = Math.max(320, chart.clientWidth);
    const height = Math.max(360, chart.clientHeight);
    const margin = {top: 14, right: 18, bottom: 48, left: 72};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const [xMin, xMax] = model.viewRange || model.dateExtent;
    const inRange = visible.map(item => ({...item, points: item.points.filter(point => {
      const time = dateTime(point[0]);
      return time >= xMin && time <= xMax;
    })}));
    let yMin = Infinity, yMax = -Infinity;
    for (const item of inRange) for (const point of item.points) {
      yMin = Math.min(yMin, point[1]); yMax = Math.max(yMax, point[1]);
    }
    if (!Number.isFinite(yMin)) {
      chart.innerHTML = '<div class="empty-state"><strong>No observations in this range</strong><span>Widen the date range or choose All dates.</span></div>';
      return;
    }
    if (yMin === yMax) { const pad = Math.abs(yMin || 1) * 0.1; yMin -= pad; yMax += pad; }
    const yPad = (yMax - yMin) * 0.06;
    yMin -= yPad; yMax += yPad;
    const x = value => margin.left + (value - xMin) / (xMax - xMin) * innerWidth;
    const y = value => margin.top + innerHeight - (value - yMin) / (yMax - yMin) * innerHeight;
    const xTicks = ticks(xMin, xMax, width < 560 ? 4 : 7);
    const yTicks = ticks(yMin, yMax, 6);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("aria-label", `${visible.length} time series for ${model.stateName}`);
    const markup = [`<defs><clipPath id="plot-clip"><rect x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}"></rect></clipPath></defs>`];
    for (const tick of yTicks) {
      markup.push(`<line class="gridline" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"></line>`);
      markup.push(`<text x="${margin.left - 9}" y="${y(tick) + 4}" text-anchor="end" class="axis">${escapeHTML(formatValue(tick))}</text>`);
    }
    for (const tick of xTicks) {
      markup.push(`<line class="gridline" x1="${x(tick)}" x2="${x(tick)}" y1="${margin.top}" y2="${margin.top + innerHeight}"></line>`);
      markup.push(`<text x="${x(tick)}" y="${height - 25}" text-anchor="middle" class="axis">${escapeHTML(formatDate(tick, xMax - xMin))}</text>`);
    }
    markup.push(`<rect class="frame" x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}" fill="none"></rect>`);
    visible.forEach(item => {
      const path = item.points.map((point, index) => `${index ? "L" : "M"}${x(Date.parse(`${point[0]}T00:00:00Z`)).toFixed(2)},${y(point[1]).toFixed(2)}`).join(" ");
      markup.push(`<path clip-path="url(#plot-clip)" class="series-line" data-series-id="${item.id}" d="${path}" stroke="${seriesColor(item)}" stroke-dasharray="${lineDash(item)}"></path>`);
    });
    markup.push(`<text class="axis-title" x="${margin.left + innerWidth / 2}" y="${height - 4}" text-anchor="middle">Date</text>`);
    markup.push(`<text class="axis-title" transform="translate(15 ${margin.top + innerHeight / 2}) rotate(-90)" text-anchor="middle">${scaleToggle.checked ? "Value ÷ latest series maximum" : "Raw value (mixed units possible)"}</text>`);
    markup.push(`<g id="hover-layer" hidden><line class="hover-line" y1="${margin.top}" y2="${margin.top + innerHeight}"></line></g>`);
    markup.push(`<rect class="hit-area" x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}"></rect>`);
    svg.innerHTML = markup.join("");
    chart.replaceChildren(svg);

    const hit = svg.querySelector(".hit-area");
    hit.addEventListener("pointermove", event => showTooltip(event, svg, inRange, {x, y, xMin, xMax, margin, innerWidth}));
    hit.addEventListener("pointerleave", () => { svg.querySelector("#hover-layer").hidden = true; tooltip.hidden = true; });
  }

  function renderRangeControl() {
    // Keep the overview extent stable when legend entries are hidden.
    let first = Infinity, last = -Infinity;
    for (const item of model.plotted) for (const point of item.points) {
      const time = dateTime(point[0]);
      first = Math.min(first, time); last = Math.max(last, time);
    }
    rangeControl.hidden = !Number.isFinite(first);
    if (rangeControl.hidden) { model.dateExtent = null; return; }
    if (first === last) { first -= DAY; last += DAY; }
    model.dateExtent = [first, last];
    if (model.viewRange) {
      const start = Math.max(first, model.viewRange[0]);
      const end = Math.min(last, model.viewRange[1]);
      model.viewRange = start < end ? [start, end] : null;
    }
    const [start, end] = model.viewRange || model.dateExtent;
    rangeStart.value = isoDate(start); rangeEnd.value = isoDate(end);
    rangeStart.min = isoDate(first); rangeStart.max = isoDate(end - DAY);
    rangeEnd.min = isoDate(start + DAY); rangeEnd.max = isoDate(last);
    const width = Math.max(260, rangeOverview.clientWidth);
    const x = time => 8 + (time - first) / (last - first) * (width - 16);
    const paths = model.plotted.filter(item => !model.hidden.has(item.visibilityKey || item.key)).map(item => {
      if (!item.points.length) return "";
      // Independently normalize the overview so signals with different units remain visible.
      let low = Infinity, high = -Infinity;
      for (const point of item.points) { low = Math.min(low, point[1]); high = Math.max(high, point[1]); }
      const path = item.points.map((point, index) =>
        `${index ? "L" : "M"}${x(dateTime(point[0])).toFixed(2)},${(54 - (point[1] - low) / (high - low || 1) * 44).toFixed(2)}`
      ).join(" ");
      return `<path d="${path}" fill="none" stroke="${seriesColor(item)}" stroke-width="1" opacity=".65"></path>`;
    }).join("");
    rangeOverview.innerHTML = `<svg viewBox="0 0 ${width} 86" aria-label="Full data history; each signal scaled independently">
      ${paths}
      <rect class="range-shade" x="8" y="4" width="${x(start) - 8}" height="56"></rect>
      <rect class="range-shade" x="${x(end)}" y="4" width="${width - 8 - x(end)}" height="56"></rect>
      <rect class="range-window" data-range-part="pan" x="${x(start)}" y="4" width="${x(end) - x(start)}" height="56"></rect>
      <rect class="range-handle" data-range-part="start" x="${x(start) - 5}" y="4" width="10" height="56" rx="3"></rect>
      <rect class="range-handle" data-range-part="end" x="${x(end) - 5}" y="4" width="10" height="56" rx="3"></rect>
      <text class="axis" x="8" y="80">${isoDate(first)}</text>
      <text class="axis" x="${width - 8}" y="80" text-anchor="end">${isoDate(last)}</text>
    </svg>`;
  }

  function setViewRange(start, end) {
    if (!model.dateExtent || !Number.isFinite(start) || !Number.isFinite(end)) return;
    const [first, last] = model.dateExtent;
    start = Math.max(first, Math.min(last - DAY, Math.round(start / DAY) * DAY));
    end = Math.min(last, Math.max(start + DAY, Math.round(end / DAY) * DAY));
    model.viewRange = start === first && end === last ? null : [start, end];
    drawChart();
  }

  for (const input of [rangeStart, rangeEnd]) input.addEventListener("change", () => {
    if (rangeStart.validity.valid && rangeEnd.validity.valid) {
      setViewRange(dateTime(rangeStart.value), dateTime(rangeEnd.value));
    }
  });
  rangeReset.addEventListener("click", () => { model.viewRange = null; drawChart(); });
  let rangeDrag = null;
  const pointerDate = event => {
    const bounds = rangeOverview.getBoundingClientRect();
    const [first, last] = model.dateExtent;
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left - 8) / (bounds.width - 16)));
    return Math.round((first + fraction * (last - first)) / DAY) * DAY;
  };
  rangeOverview.addEventListener("pointerdown", event => {
    if (event.button !== 0 || !model.dateExtent) return;
    event.preventDefault();
    rangeDrag = {
      part: event.target.dataset.rangePart || "new",
      anchor: pointerDate(event),
      range: [...(model.viewRange || model.dateExtent)],
      pointerId: event.pointerId,
    };
    rangeOverview.setPointerCapture(event.pointerId);
  });
  rangeOverview.addEventListener("pointermove", event => {
    if (!rangeDrag || event.pointerId !== rangeDrag.pointerId) return;
    const time = pointerDate(event);
    const [start, end] = rangeDrag.range;
    const [first, last] = model.dateExtent;
    if (rangeDrag.part === "start") setViewRange(Math.min(time, end - DAY), end);
    else if (rangeDrag.part === "end") setViewRange(start, Math.max(time, start + DAY));
    else if (rangeDrag.part === "pan") {
      const shift = Math.max(first - start, Math.min(last - end, time - rangeDrag.anchor));
      setViewRange(start + shift, end + shift);
    } else setViewRange(Math.min(time, rangeDrag.anchor), Math.max(time, rangeDrag.anchor));
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    rangeOverview.addEventListener(type, () => { rangeDrag = null; });
  }

  function showTooltip(event, svg, inRange, scale) {
    const bounds = svg.getBoundingClientRect();
    const cursorX = (event.clientX - bounds.left) * (svg.viewBox.baseVal.width / bounds.width);
    const timestamp = scale.xMin + (cursorX - scale.margin.left) / scale.innerWidth * (scale.xMax - scale.xMin);
    const layer = svg.querySelector("#hover-layer");
    const guideX = Math.max(scale.margin.left, Math.min(scale.margin.left + scale.innerWidth, cursorX));
    const rows = [];
    layer.innerHTML = `<line class="hover-line" x1="${guideX}" x2="${guideX}" y1="${scale.margin.top}" y2="${svg.viewBox.baseVal.height - scale.margin.bottom}"></line>`;
    inRange.forEach(item => {
      const nearest = nearestPoint(item.points, timestamp);
      if (!nearest) return;
      layer.insertAdjacentHTML("beforeend", `<circle class="hover-dot" cx="${scale.x(Date.parse(`${nearest[0]}T00:00:00Z`))}" cy="${scale.y(nearest[1])}" r="4" fill="${seriesColor(item)}"></circle>`);
      rows.push({item, point: nearest, color: seriesColor(item)});
    });
    layer.hidden = false;
    const dateLabel = new Date(timestamp).toLocaleDateString(undefined, {year: "numeric", month: "short", day: "numeric", timeZone: "UTC"});
    tooltip.innerHTML = `<time>${escapeHTML(dateLabel)}</time>${rows.map(row => `<div class="tooltip-row"><i style="background:${row.color}"></i><span>${escapeHTML(row.item.label)}${row.point[2] > 1 ? ` (n=${row.point[2]})` : ""}</span><b>${escapeHTML(formatValue(row.point[1]))}</b></div>`).join("")}`;
    tooltip.hidden = false;
    const panelBounds = chart.parentElement.getBoundingClientRect();
    const localX = event.clientX - panelBounds.left;
    const localY = event.clientY - panelBounds.top;
    tooltip.style.left = `${Math.min(localX + 14, panelBounds.width - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${Math.max(8, localY - tooltip.offsetHeight - 10)}px`;
  }

  function nearestPoint(points, timestamp) {
    if (!points.length) return null;
    let low = 0, high = points.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      const value = Date.parse(`${points[middle][0]}T00:00:00Z`);
      if (value < timestamp) low = middle + 1; else high = middle;
    }
    if (low > 0) {
      const left = Date.parse(`${points[low - 1][0]}T00:00:00Z`);
      const right = Date.parse(`${points[low][0]}T00:00:00Z`);
      if (Math.abs(left - timestamp) < Math.abs(right - timestamp)) return points[low - 1];
    }
    return points[low];
  }

  function ticks(minimum, maximum, count) {
    if (maximum > 1e11) {
      const step = (maximum - minimum) / Math.max(1, count - 1);
      return Array.from({length: count}, (_, index) => minimum + step * index);
    }
    const raw = (maximum - minimum) / Math.max(1, count - 1);
    const magnitude = 10 ** Math.floor(Math.log10(Math.abs(raw) || 1));
    const residual = raw / magnitude;
    const nice = residual >= 5 ? 10 : residual >= 2 ? 5 : residual >= 1 ? 2 : 1;
    const step = nice * magnitude;
    const start = Math.ceil(minimum / step) * step;
    const result = [];
    for (let value = start; value <= maximum + step * 0.001; value += step) result.push(value);
    return result;
  }

  function formatDate(timestamp, span) {
    const options = span > 1000 * 86400000 ? {year: "numeric"} : span > 180 * 86400000 ? {month: "short", year: "2-digit"} : {month: "short", day: "numeric"};
    return new Date(timestamp).toLocaleDateString(undefined, {...options, timeZone: "UTC"});
  }
  function formatNumber(value) { return new Intl.NumberFormat().format(value); }
  function formatValue(value) {
    const absolute = Math.abs(value);
    if ((absolute > 0 && absolute < 0.001) || absolute >= 1000000) return Number(value).toExponential(2);
    return new Intl.NumberFormat(undefined, {maximumFractionDigits: absolute < 10 ? 3 : 1}).format(value);
  }
  function showError(error) { plotStatus.textContent = error.message; }

  function syncLocationSupport() {
    const national = model.state === "US";
    supportFilter.disabled = national;
    supportFilter.value = national ? "national" : model.stateSupport;
    model.filters.support = supportFilter.value;
    const active = Object.values(model.filters).filter(Boolean).length;
    filterCount.textContent = active ? `(${active})` : "";
  }

  // Geography facets are ignored so a national series and its state counterpart share a key.
  const GEOGRAPHY_FACETS = new Set(["spatial_support", "trend_source", "geo_type"]);
  function locationVariantKey(item) {
    const dimensions = Object.entries(item.dimensions || {})
      .filter(([key]) => !GEOGRAPHY_FACETS.has(key))
      .sort(([a], [b]) => a.localeCompare(b));
    const path = item.source_path.replace(/geo_type=(?:nation|state)\b/, "geo_type=*");
    return JSON.stringify([item.dataset_key, path, item.value_column, dimensions]);
  }

  function isNationalBroadcast(item) { return Boolean(item.dimensions?.spatial_support); }

  async function resolveSelectionForLocation(token) {
    // Selections follow the location: native state series in a state, published US
    // series nationally. National context chosen deliberately in a state stays national.
    if (!model.selected.size) return true;
    const payload = await requestJSON(`/api/series?${new URLSearchParams({state: model.state, limit: "0"})}`);
    if (token !== model.locationRequest) return false;
    const national = model.state === "US";
    const counterparts = new Map(payload.items
      .filter(item => isNationalBroadcast(item) === national)
      .map(item => [locationVariantKey(item), item]));
    model.selected = new Map([...model.selected.values()].map(item => {
      const followsLocation = !item.explicitNational;
      const replacement = followsLocation && counterparts.get(locationVariantKey(item));
      const selected = replacement ? {...replacement, explicitNational: false} : item;
      if (selected.id !== item.id) {
        for (const key of [...model.hidden]) {
          if (key.startsWith(`${item.id}@`)) { model.hidden.delete(key); model.hidden.add(key.replace(`${item.id}@`, `${selected.id}@`)); }
        }
      }
      return [selected.id, selected];
    }));
    return true;
  }

  // Each Hub preset pairs the Hub's target data with the Delphi ground truth it is
  // derived from: NHSN weekly admissions and reported (unsmoothed) NSSP ED percentage.
  const HUB_PRESETS = {
    covid: {hub: "hub_covid_current", signals: ["nhsn:totalconfc19newadm", "nssp:percent_visits_covid"]},
    influenza: {hub: "hub_flusight_current", signals: ["nhsn:totalconfflunewadm", "nssp:percent_visits_influenza"]},
    rsv: {hub: "hub_rsv_current", signals: ["nhsn:totalconfrsvnewadm", "nssp:percent_visits_rsv"]},
  };

  async function applyHubPreset(name) {
    const preset = HUB_PRESETS[name];
    const token = ++model.locationRequest;
    try {
      const payload = await requestJSON(`/api/series?${new URLSearchParams({state: model.state, limit: "0"})}`);
      if (token !== model.locationRequest) return;
      const datasets = new Set([preset.hub, "delphi_nhsn", "delphi_nssp"]);
      const items = payload.items.filter(item => datasets.has(item.dataset_key)
        && preset.signals.includes(item.signal_key)
        && isNationalBroadcast(item) === (model.state === "US"));
      model.selected = new Map(items.map(item => [item.id, {...item, explicitNational: false}]));
      model.hidden.clear(); model.comparisons = []; model.lineColors.clear();
      // Admissions counts and ED percentages/proportions only overlay once scaled.
      scaleToggle.checked = true;
      renderSeriesList(); renderVersionControls(); loadVersions(); updatePlot();
    } catch (error) {
      if (token === model.locationRequest) showError(error);
    }
  }
  hubPresets.addEventListener("click", event => {
    const button = event.target.closest("[data-preset]");
    if (button) applyHubPreset(button.dataset.preset);
  });

  stateSelect.addEventListener("change", async () => {
    const token = ++model.locationRequest;
    model.state = stateSelect.value;
    model.stateName = stateSelect.options[stateSelect.selectedIndex].text.replace(/\s+\([\d,]+ columns\)$/, "");
    syncLocationSupport();
    // Invalidate outstanding responses immediately while resolving counterparts.
    ++model.plotRequest; ++model.versionRequest; ++model.listRequest;
    try {
      if (!await resolveSelectionForLocation(token)) return;
    } catch (error) {
      if (token !== model.locationRequest) return;
      showError(error);
    }
    loadVersions(); updatePlot(); await loadSeries();
  });
  scaleToggle.addEventListener("change", updatePlot);
  clearButton.addEventListener("click", () => {
    model.selected.clear(); model.hidden.clear(); model.comparisons = []; model.lineColors.clear();
    renderSeriesList(); loadVersions(); updatePlot();
  });
  versionDate.addEventListener("change", () => { if (versionDate.validity.valid) setVersion(versionDate.value); });
  versionLatest.addEventListener("click", () => setVersion(""));
  versionPrevWednesday.addEventListener("click", () => setVersion(wednesday(model.asOf, -1)));
  versionNextWednesday.addEventListener("click", () => setVersion(wednesday(model.asOf, 1)));
  versionPrev.addEventListener("click", () => {
    const dates = model.versionDates.filter(day => !model.asOf || day < model.asOf);
    if (dates.length) setVersion(dates[dates.length - 1]);
  });
  versionNext.addEventListener("click", () => setVersion(model.versionDates.find(day => day > model.asOf) || ""));
  versionKeep.addEventListener("click", () => {
    if (model.asOf && !model.comparisons.includes(model.asOf) && model.comparisons.length < 5) {
      model.comparisons.push(model.asOf);
      for (const id of model.selected.keys()) {
        const key = `${id}@${model.asOf}`;
        if (model.hidden.has(`${id}@as-of`)) model.hidden.add(key);
        else model.hidden.delete(key);
      }
    }
    renderVersionControls(); updatePlot();
  });
  comparisons.addEventListener("click", event => {
    const button = event.target.closest("[data-comparison]");
    if (!button) return;
    model.comparisons.splice(Number(button.dataset.comparison), 1);
    renderVersionControls(); updatePlot();
  });
  let searchTimer;
  seriesSearch.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { model.search = seriesSearch.value; loadSeries(true); }, 180);
  });
  signalFilters.addEventListener("change", () => {
    model.filters = {
      cadence: cadenceFilter.value,
      vintage: vintageFilter.value,
      support: supportFilter.value,
      freshness: freshnessFilter.value,
    };
    if (model.state !== "US") model.stateSupport = supportFilter.value;
    syncLocationSupport();
    loadSeries(true);
  });
  resetFilters.addEventListener("click", () => {
    for (const select of signalFilters.querySelectorAll("select")) select.value = "";
    model.stateSupport = "native_state";
    syncLocationSupport();
    signalFilters.dispatchEvent(new Event("change", {bubbles: true}));
  });
  seriesList.addEventListener("change", event => {
    const input = event.target.closest("input[data-series-id]");
    if (!input) return;
    const id = Number(input.dataset.seriesId);
    const item = model.available.find(candidate => candidate.id === id);
    if (input.checked && item) {
      if (model.selected.size >= 100) { input.checked = false; showError(new Error("Select up to 100 columns.")); return; }
      model.selected.set(id, {...item, explicitNational: model.state !== "US" && isNationalBroadcast(item)});
    } else {
      model.selected.delete(id);
      for (const key of model.hidden) if (key.startsWith(`${id}@`)) model.hidden.delete(key);
    }
    const option = input.closest(".series-option");
    if (item && option) {
      option.outerHTML = renderSeriesOption(item, item.variant_label);
      seriesList.querySelector(`input[data-series-id="${id}"]`)?.focus();
    }
    loadVersions(); updatePlot();
  });
  legend.addEventListener("click", event => {
    const button = event.target.closest("[data-legend-id]");
    if (!button) return;
    const id = button.dataset.legendId;
    if (model.hidden.has(id)) model.hidden.delete(id); else model.hidden.add(id);
    // When the active date is also pinned, the single displayed curve represents both.
    if (id.endsWith("@as-of") && model.comparisons.includes(model.asOf)) {
      const pinnedKey = id.replace(/@as-of$/, `@${model.asOf}`);
      if (model.hidden.has(id)) model.hidden.add(pinnedKey);
      else model.hidden.delete(pinnedKey);
    }
    renderLegend(); drawChart();
  });
  new ResizeObserver(() => { if (model.plotted.length) drawChart(); }).observe(chart);
  darkTheme.addEventListener("change", () => { if (model.plotted.length) { renderLegend(); drawChart(); } });
  initialize();
})();

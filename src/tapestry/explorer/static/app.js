/* Dependency-free client for the local, read-only explorer API. */
(() => {
  "use strict";

  const stateSelect = document.getElementById("state-select");
  const scaleToggle = document.getElementById("scale-toggle");
  const seriesSearch = document.getElementById("series-search");
  const seriesList = document.getElementById("series-list");
  const versionDate = document.getElementById("version-date");
  const versionWeekday = document.getElementById("version-weekday");
  const versionPrev = document.getElementById("version-prev");
  const versionNext = document.getElementById("version-next");
  const versionLatest = document.getElementById("version-latest");
  const versionPrevWednesday = document.getElementById("version-prev-wednesday");
  const versionNextWednesday = document.getElementById("version-next-wednesday");
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

  let pendingRequests = 0;
  async function requestJSON(path) {
    // Any outstanding API request shows the progress bar under the header.
    document.body.classList.toggle("is-busy", ++pendingRequests > 0);
    try {
      if (await staticData.available()) return await staticData.request(path);
      const response = await fetch(path, {headers: {"Accept": "application/json"}});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload;
    } finally {
      document.body.classList.toggle("is-busy", --pendingRequests > 0);
    }
  }

  /* Published (GitHub Pages) mode: no server, so the four API calls are answered
     from files written by `explore_covariates.py export` into ./data/. Revisions
     are one Parquet change log per series, downloaded whole and read with hyparquet. Resolution
     mirrors ExplorerIndex.data(): per event, the latest row released on or before
     the as-of date (versioned sources), events on or before it, null values hidden. */
  const staticData = (() => {
    const ROOT = "data/";
    const HYPARQUET = "https://cdn.jsdelivr.net/npm/hyparquet@1.31.0/+esm";
    const FRESHNESS_DAYS = {daily: 14, weekly: 35, monthly: 75, sample: 45};
    let detected, catalog, ranges, hyparquet, metadata;
    const files = new Map();
    const lists = new Map();
    const revisions = new Map();
    // GitHub Pages caches files for 10 minutes. The catalog is always revalidated and
    // its export time versions every other data URL, so a refresh never mixes exports.
    const versioned = name => `${ROOT}${name}?v=${encodeURIComponent(catalog?.meta?.exported_at || "")}`;
    const json = async name => {
      const response = await fetch(versioned(name));
      if (!response.ok) throw new Error(`Missing published file ${name} (HTTP ${response.status})`);
      return response.json();
    };
    const dayNumber = iso => Math.floor(dateTime(iso) / DAY);
    const dayString = number => isoDate(number * DAY);

    function available() {
      detected ??= fetch(ROOT + "catalog.json", {cache: "no-cache"}).then(async response => {
        if (!response.ok || !(response.headers.get("content-type") || "").includes("json")) return false;
        catalog = await response.json();
        return true;
      }).catch(() => false);
      return detected;
    }

    function list(state) {
      if (!lists.has(state)) {
        metadata ??= Promise.all([json("series.json"), json("locations.json")]);
        lists.set(state, metadata.then(([series, locations]) => (locations[state] || []).map(
          ([id, date_min, date_max, point_count, max_samples_per_point]) =>
            ({...series[id], date_min, date_max, point_count, max_samples_per_point}))));
      }
      return lists.get(state);
    }

    async function rows(id, state) {
      const key = `${id}|${state}`;
      if (!revisions.has(key)) {
        revisions.set(key, (async () => {
          ranges ??= json("ranges.json");
          const range = (await ranges)[key];
          if (!range) return [];
          hyparquet ??= import(HYPARQUET);
          // Whole-file download: GitHub Pages gzips responses, which breaks byte ranges.
          if (!files.has(id)) {
            files.set(id, fetch(versioned(`revisions/${id}.parquet`)).then(response => {
              if (!response.ok) throw new Error(`Missing published revisions for series ${id} (HTTP ${response.status})`);
              return response.arrayBuffer();
            }));
          }
          const [module, file] = await Promise.all([hyparquet, files.get(id)]);
          return module.parquetReadObjects({file, columns: ["event", "release", "value", "n"], rowStart: range[0], rowEnd: range[1]});
        })());
      }
      return revisions.get(key);
    }

    async function seriesRows(id, state) {
      const own = await rows(id, state);
      return state === "US" ? own : [...own, ...await rows(id, "US")];
    }

    function freshness(item) {
      const cutoff = isoDate(Date.now() - (FRESHNESS_DAYS[String(item.temporal_resolution).toLowerCase()] || 35) * DAY);
      return item.date_max >= cutoff ? "current" : "lagging";
    }

    async function series(params) {
      const state = params.get("state");
      const needle = (params.get("q") || "").trim().toLowerCase();
      const cadence = (params.get("cadence") || "").toLowerCase();
      const vintage = params.get("vintage") || "";
      const support = params.get("support") || "";
      const fresh = params.get("freshness") || "";
      const items = (await list(state)).map(item => ({...item, freshness: freshness(item)})).filter(item =>
        (!needle || [item.label, item.dataset_title, item.provider, item.source_path, item.value_column]
          .some(field => String(field || "").toLowerCase().includes(needle)))
        && (!cadence || String(item.temporal_resolution).toLowerCase() === cadence)
        && (!vintage || item.versioned === (vintage === "versioned"))
        && (support !== "native_state" || !item.dimensions.spatial_support)
        && (support !== "national" || item.dimensions.spatial_support === "national parent broadcast")
        && (!fresh || item.freshness === fresh));
      return {state, total: items.length, items};
    }

    async function versions(params) {
      const state = params.get("state");
      const ids = (params.get("series") || "").split(",").filter(Boolean).map(Number);
      const dates = new Set();
      for (const id of ids) {
        for (const row of await seriesRows(id, state)) dates.add(dayString(row.release ?? row.event));
      }
      return {state, dates: [...dates].sort()};
    }

    async function data(params) {
      const state = params.get("state");
      const asOf = params.get("as_of");
      const day = asOf && asOf !== "latest" ? asOf : null;
      const cutoff = day ? dayNumber(day) : null;
      const ids = (params.get("series") || "").split(",").filter(Boolean).map(Number);
      const items = new Map((await list(state)).map(item => [item.id, item]));
      const output = [];
      for (const id of ids) {
        let item = items.get(id);
        if (!item) item = (await list("US")).find(candidate => candidate.id === id);
        if (!item) continue;
        const latest = new Map();
        for (const row of await seriesRows(id, state)) {
          if (cutoff != null && row.event > cutoff) continue;
          if (cutoff != null && item.versioned && (row.release == null || row.release > cutoff)) continue;
          const current = latest.get(row.event);
          if (!current || (row.release ?? -Infinity) >= (current.release ?? -Infinity)) latest.set(row.event, row);
        }
        const points = [...latest.values()].filter(row => row.value != null)
          .sort((a, b) => a.event - b.event).map(row => [dayString(row.event), row.value, row.n]);
        const max = points.length ? Math.max(...points.map(point => point[1])) : null;
        output.push({...item, points, max, as_of: day,
          version_note: day && item.versioned && !points.length ? "No archived values available by this date." : undefined});
      }
      const location = catalog.states.find(entry => entry.code === state);
      return {state, state_name: location ? location.name : state, as_of: day, series: output};
    }

    async function request(path) {
      const url = new URL(path, location.href);
      const params = url.searchParams;
      switch (url.pathname.replace(/^.*\/api\//, "/api/")) {
        case "/api/catalog": return catalog;
        case "/api/series": return series(params);
        case "/api/versions": return versions(params);
        case "/api/data": return data(params);
        default: throw new Error(`Unsupported request in published mode: ${path}`);
      }
    }
    const exportedAt = () => (catalog?.meta?.exported_at || "").slice(0, 10);
    return {available, request, exportedAt};
  })();

  function escapeHTML(value) {
    return String(value).replace(/[&<>'"]/g, character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[character]);
  }

  async function initialize() {
    try {
      const overview = await requestJSON("api/catalog");
      const selection = overview.selection;
      stateSelect.innerHTML = overview.states.map(item =>
        `<option value="${item.code}">${escapeHTML(item.name)}</option>`
      ).join("");
      const built = overview.meta.exported_at ? `${new Date(overview.meta.exported_at).toLocaleString()} (published copy)`
        : overview.meta.built_at ? new Date(overview.meta.built_at).toLocaleString() : "unknown";
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
      showError(error);
    }
  }

  async function loadSeries() {
    if (!model.state) return;
    const token = ++model.listRequest;
    try {
      const params = new URLSearchParams({state: model.state, q: model.search, limit: "0", ...model.filters});
      const payload = await requestJSON(`api/series?${params}`);
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
      const payload = await requestJSON(`api/versions?${params}`);
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
          open: shouldOpen(items, key, previouslyOpen),
          children: signals.map(variants => renderSignal(variants, key, previouslyOpen)).join("")});
      }).join("");
      return renderBranch({className: "series-pathogen", key: pathogenKey,
        label: pathogenItems[0].pathogen_title,
        count: new Set(pathogenItems.map(item => item.signal_key)).size,
        open: shouldOpen(pathogenItems, pathogenKey, previouslyOpen), children});
    }).join("");
  }

  function renderSignal(variants, parentKey, previouslyOpen) {
    variants.sort((a, b) => {
      for (let i = 0; i < a.variant_rank.length; i++) {
        if (a.variant_rank[i] !== b.variant_rank[i]) return a.variant_rank[i] - b.variant_rank[i];
      }
      return a.variant_label.localeCompare(b.variant_label);
    });
    const key = treeKey(parentKey, variants[0].signal_key);
    return renderBranch({className: "series-signal", key, label: variants[0].signal_title,
      title: variants[0].column_description, count: variants.length,
      open: shouldOpen(variants, key, previouslyOpen),
      children: variants.map(item => renderSeriesOption(item, item.variant_label)).join("")});
  }

  function providers(items) {
    const names = {cdc: "CDC", delphi: "Delphi", hub: "Forecast Hub"};
    return [...new Set(items.map(item => names[item.provider_kind] || item.provider_kind))].join(" · ");
  }

  function renderSeriesOption(item, variantLabel) {
    // One line per variant, named as a path; provenance and coverage appear as a card once checked.
    const checked = model.selected.has(item.id);
    const label = variantLabel.split(" · ").join("/");
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

  function renderBranch({className, key, label, meta = "", title = "", count, open, children}) {
    const metadata = meta ? `<small>${escapeHTML(meta)}</small>` : "";
    return `<details class="series-branch ${className}" data-tree-key="${escapeHTML(key)}"${open ? " open" : ""}>
      <summary${title ? ` title="${escapeHTML(title)}"` : ""}><span>${escapeHTML(label)}${metadata}</span><b>${formatNumber(count)}</b></summary>
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

  // Legend and tooltip names: what is measured, then who publishes it.
  function dataName(item) {
    if (item.source_group === "nssp") return `${item.pathogen_title} ED visits`;
    return item.signal_title || item.label;
  }

  function sourceName(item) {
    const release = (item.variant_label || "").split(" · ")[0];
    let name = item.provider_kind === "delphi" ? "Delphi"
      : item.provider_kind === "hub" ? release.replace(/ target data$/, "")
      : release.startsWith("CDC") ? release
      : release.startsWith("cdc_") ? `CDC ${item.parent_dataset || ""}`.trim()
      : `CDC ${item.parent_dataset || ""} ${release.toLowerCase()}`.replace(/\s+/g, " ").trim();
    if (isNationalBroadcast(item) && model.state !== "US") name += " (national)";
    return name;
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
      plotTitle.textContent = "Choose a location and one or more signals";
      plotStatus.textContent = "";
      chart.innerHTML = '<div class="empty-state"><strong>No series selected</strong><span>Select signals on the left to overlay them here.</span></div>';
      return;
    }
    plotStatus.textContent = "";
    chart.classList.add("is-loading");
    try {
      const versions = [...new Set(["", model.asOf])];
      const payloads = await Promise.all(versions.map(asOf => {
        const params = new URLSearchParams({state: model.state, series: ids.join(","),
          scale: "false", as_of: asOf || "latest"});
        return requestJSON(`api/data?${params}`);
      }));
      if (token !== model.plotRequest) return;
      chart.classList.remove("is-loading");
      // Divide by mean over the dates every selected series covers: each series is
      // divided by its latest mean inside that common window, and the y-axis fits that
      // window, so a longer series may leave the canvas elsewhere. Revisions share the divisor.
      const withData = payloads[0].series.filter(item => item.points.length);
      const start = withData.reduce((day, item) => item.points[0][0] > day ? item.points[0][0] : day, "");
      const end = withData.reduce((day, item) => item.points.at(-1)[0] < day ? item.points.at(-1)[0] : day, "9999");
      model.commonWindow = withData.length && start <= end ? [start, end] : null;
      const divisors = new Map(payloads[0].series.map(item => {
        const values = model.commonWindow
          ? item.points.filter(([day]) => day >= start && day <= end).map(point => point[1]) : [];
        const all = values.length ? values : item.points.map(point => point[1]);
        const mean = all.length ? all.reduce((sum, value) => sum + value, 0) / all.length : 0;
        return [item.id, mean > 0 ? mean : 1];
      }));
      const divisor = item => scaleToggle.checked ? divisors.get(item.id) || item.max || 1 : 1;
      model.plotted = payloads.flatMap((payload, versionIndex) => payload.series.map(item => ({
        ...item, points: item.points.map(([day, value, samples]) => [day, value / divisor(item), samples]),
        key: `${item.id}@${versions[versionIndex] || "latest"}`,
        // Visibility follows the as-of curve as its date moves.
        visibilityKey: `${item.id}@${versionIndex ? "as-of" : "latest"}`,
        versionIndex,
        label: [dataName(item), sourceName(item), versionIndex ? versionLabel(versions[versionIndex]) : ""]
          .filter(Boolean).join(" / "),
        details: [item.signal_title, item.variant_label, versionLabel(versions[versionIndex])].join(" · "),
      }))).sort((a, b) => ids.indexOf(a.id) - ids.indexOf(b.id) || a.versionIndex - b.versionIndex);
      plotTitle.textContent = payloads[0].state_name;
      renderLegend();
      drawChart();
    } catch (error) {
      if (token === model.plotRequest) chart.classList.remove("is-loading");
      showError(error);
    }
  }

  function renderLegend() {
    // One column per series: real (latest) data first, its revision directly below.
    const groups = groupBy(model.plotted, item => item.id);
    legend.innerHTML = [...groups.values()].map(items => `<div class="legend-group">${items.map(item => {
      const hidden = model.hidden.has(item.visibilityKey || item.key);
      const maximum = item.max == null ? "" : `max ${formatValue(item.max)}`;
      return `<button class="legend-item${item.versionIndex ? " is-revision" : ""}" type="button" data-legend-id="${item.visibilityKey || item.key}" aria-pressed="${hidden ? "false" : "true"}" title="${escapeHTML([item.details, maximum, item.parent_dataset, item.parent_column, item.parent_transform, item.version_note].filter(Boolean).join(" · "))}">
        <svg class="swatch" viewBox="0 0 22 6" aria-hidden="true"><line x1="0" x2="22" y1="3" y2="3" stroke="${seriesColor(item)}" stroke-width="2.5" stroke-dasharray="${lineDash(item)}"></line></svg>
        <span class="legend-label">${escapeHTML(item.label)}</span>
      </button>`;
    }).join("")}</div>`).join("");
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
    if (scaleToggle.checked && model.commonWindow) {
      // Fit the y-axis to the common window; larger values elsewhere are clipped.
      const [start, end] = model.commonWindow;
      const common = inRange.flatMap(item => item.points.filter(([day]) => day >= start && day <= end).map(point => point[1]));
      if (common.length) { yMin = Math.min(0, ...common); yMax = Math.max(...common); }
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
    [...visible].sort((a, b) => b.versionIndex - a.versionIndex).forEach(item => {
      const path = item.points.map((point, index) => `${index ? "L" : "M"}${x(Date.parse(`${point[0]}T00:00:00Z`)).toFixed(2)},${y(point[1]).toFixed(2)}`).join(" ");
      markup.push(`<path clip-path="url(#plot-clip)" class="series-line" data-series-id="${item.id}" d="${path}" stroke="${seriesColor(item)}" stroke-dasharray="${lineDash(item)}"></path>`);
    });
    if (model.asOf) {
      // The chosen as-of date: data to its right was not yet published at that cutoff.
      const asOfTime = dateTime(model.asOf);
      if (asOfTime >= xMin && asOfTime <= xMax) {
        const asOfX = x(asOfTime);
        const anchor = asOfX > margin.left + innerWidth - 150 ? "end" : "start";
        markup.push(`<line class="as-of-line" x1="${asOfX}" x2="${asOfX}" y1="${margin.top}" y2="${margin.top + innerHeight}"></line>`);
        markup.push(`<text class="as-of-label" x="${asOfX + (anchor === "start" ? 6 : -6)}" y="${margin.top + 12}" text-anchor="${anchor}">As of ${escapeHTML(versionLabel(model.asOf))}</text>`);
      }
    }
    markup.push(`<text class="axis-title" x="${margin.left + innerWidth / 2}" y="${height - 4}" text-anchor="middle">Date</text>`);
    markup.push(`<text class="axis-title" transform="translate(15 ${margin.top + innerHeight / 2}) rotate(-90)" text-anchor="middle">${scaleToggle.checked ? (model.commonWindow ? `Value ÷ mean over ${model.commonWindow[0]} – ${model.commonWindow[1]}` : "Value ÷ series mean") : "Raw value (mixed units possible)"}</text>`);
    markup.push(`<g id="hover-layer" hidden><line class="hover-line" y1="${margin.top}" y2="${margin.top + innerHeight}"></line></g>`);
    markup.push(`<rect class="hit-area" x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}"><title>Click to set the as-of date</title></rect>`);
    svg.innerHTML = markup.join("");
    chart.replaceChildren(svg);

    const hit = svg.querySelector(".hit-area");
    hit.addEventListener("pointermove", event => showTooltip(event, svg, inRange, {x, y, xMin, xMax, margin, innerWidth}));
    hit.addEventListener("pointerleave", () => { svg.querySelector("#hover-layer").hidden = true; tooltip.hidden = true; });
    hit.addEventListener("click", event => {
      // Clicking the plot sets the as-of date to that day (never after today).
      const bounds = svg.getBoundingClientRect();
      const cursorX = (event.clientX - bounds.left) * (svg.viewBox.baseVal.width / bounds.width);
      const time = xMin + (cursorX - margin.left) / innerWidth * (xMax - xMin);
      const day = isoDate(Math.round(time / DAY) * DAY);
      setVersion(day > today() ? today() : day);
    });
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
    for (let value = start; value <= maximum + step * 0.001; value += step) result.push(value || 0);
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
    const payload = await requestJSON(`api/series?${new URLSearchParams({state: model.state, limit: "0"})}`);
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
        // The counterpart keeps the line color of the series it replaces.
        if (model.lineColors.has(item.id)) model.lineColors.set(selected.id, model.lineColors.get(item.id));
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
      const payload = await requestJSON(`api/series?${new URLSearchParams({state: model.state, limit: "0"})}`);
      if (token !== model.locationRequest) return;
      const datasets = new Set([preset.hub, "delphi_nhsn", "delphi_nssp"]);
      const items = payload.items.filter(item => datasets.has(item.dataset_key)
        && preset.signals.includes(item.signal_key)
        && isNationalBroadcast(item) === (model.state === "US"));
      model.selected = new Map(items.map(item => [item.id, {...item, explicitNational: false}]));
      model.hidden.clear(); model.lineColors.clear();
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
    model.selected.clear(); model.hidden.clear(); model.lineColors.clear();
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
    renderLegend(); drawChart();
  });
  // Dismissal is intentionally not stored: the disclaimer returns on every visit.
  document.getElementById("disclaimer-close").addEventListener("click", () => document.getElementById("disclaimer").remove());
  // The published copy is a frozen export: say so, and point to the local explorer.
  staticData.available().then(published => {
    const text = document.querySelector("#disclaimer p");
    if (!published || !text) return;
    const exported = staticData.exportedAt();
    const note = document.createElement("strong");
    note.textContent = ` This online version is not updated${exported ? ` (exported ${exported})` : ""}; use the local explorer as the ground-truth source.`;
    text.appendChild(note);
  });
  new ResizeObserver(() => { if (model.plotted.length) drawChart(); }).observe(chart);
  darkTheme.addEventListener("change", () => { if (model.plotted.length) { renderLegend(); drawChart(); } });
  initialize();
})();

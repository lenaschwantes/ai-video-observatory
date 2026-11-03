// Ranking chart: one row per tool, three views with animated transitions.
//   fpr    - real videos called AI (bars, lower is better)
//   auc    - separation ability (bars from the 0.5 coin-flip line)
//   region - Brazil vs global false positives (dumbbell)
// Data comes from <script type="application/json"> in the page; labels are
// passed in so the chart follows the page language.
/* global d3, tip */

function rankingChart(root, rows, L) {
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pct = d3.format(".0%");
  const dec = (x) => (x == null ? "n/a" : x.toFixed(2));
  const margin = { top: 8, right: 64, bottom: 28, left: 150 };
  const rowH = 44;
  const svg = d3.select(root).append("svg").attr("role", "group");
  const gx = svg.append("g").attr("class", "x-axis");
  const gRef = svg.append("g");
  const gRows = svg.append("g");
  let view = "fpr";
  let width = 0;

  function order(v) {
    if (v === "auc") return [...rows].filter((r) => r.auc != null).sort((a, b) => b.auc - a.auc);
    if (v === "region") return [...rows].filter((r) => r.fpr_br != null && r.fpr_global != null).sort((a, b) => b.fpr_br - a.fpr_br);
    return [...rows].sort((a, b) => b.fpr - a.fpr);
  }

  function draw(animate) {
    width = root.clientWidth;
    const narrow = width < 520;
    margin.left = narrow ? 128 : 150;
    const data = order(view);
    const height = margin.top + margin.bottom + data.length * rowH;
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height)
      .attr("aria-label", L[view + "_aria"]);
    const x = view === "auc"
      ? d3.scaleLinear().domain([0.5, 1]).range([margin.left, width - margin.right])
      : d3.scaleLinear().domain([0, Math.max(0.3, d3.max(data, (d) => Math.max(d.fpr, d.fpr_br ?? 0, d.fpr_global ?? 0)) * 1.1)]).nice()
          .range([margin.left, width - margin.right]);
    const y = (i) => margin.top + i * rowH + rowH / 2;
    const t = svg.transition().duration(animate ? 650 : 0).ease(d3.easeCubicInOut);
    const fmt = view === "auc" ? (v) => v.toFixed(2) : pct;

    gx.attr("transform", `translate(0,${height - margin.bottom})`);
    // First draw without a transition: d3 cannot interpolate tick attributes
    // that do not exist yet (logs "dy: NaN").
    (animate && gx.selectAll(".tick").size() ? gx.transition(t) : gx)
      .call(d3.axisBottom(x).ticks(narrow ? 3 : 5).tickFormat(fmt).tickSize(-(height - margin.top - margin.bottom)).tickPadding(8))
      .call((g) => g.select(".domain").remove());

    // coin-flip reference for AUC
    const ref = gRef.selectAll("g.ref").data(view === "auc" ? [0.5] : []);
    ref.exit().remove();
    const refEnter = ref.enter().append("g").attr("class", "ref");
    refEnter.append("text").attr("font-size", 11).attr("text-anchor", "start");
    refEnter.merge(ref).select("text").attr("x", x(0.5) + 4).attr("y", margin.top - 0).text("← " + L.coin_flip).attr("fill", css("--muted"));

    const g = gRows.selectAll("g.row").data(data, (d) => d.name);
    g.exit().transition(t).style("opacity", 0).remove();
    const enter = g.enter().append("g").attr("class", "row").style("opacity", 0)
      .attr("transform", (d, i) => `translate(0,${y(i)})`);
    enter.append("text").attr("class", "name").attr("x", margin.left - 12).attr("text-anchor", "end").attr("dy", "-0.1em").attr("font-size", 14);
    enter.append("text").attr("class", "kind").attr("x", margin.left - 12).attr("text-anchor", "end").attr("dy", "1.15em").attr("font-size", 11);
    enter.append("line").attr("class", "stem");
    enter.append("rect").attr("class", "bar").attr("height", 18).attr("y", -9).attr("rx", 4);
    enter.append("circle").attr("class", "dot-g").attr("r", 6);
    enter.append("circle").attr("class", "dot-b").attr("r", 6);
    enter.append("text").attr("class", "value-label").attr("dy", "0.35em").attr("font-size", 13);
    enter.append("rect").attr("class", "hit").attr("x", 0).attr("y", -rowH / 2).attr("height", rowH).attr("fill", "transparent");

    const all = enter.merge(g);
    all.transition(t).style("opacity", 1).attr("transform", (d, i) => `translate(0,${y(i)})`);
    all.select(".name").text((d) => d.name).attr("x", margin.left - 12).attr("fill", css("--ink")).attr("font-weight", 600);
    all.select(".kind").text((d) => L["kind_" + d.kind]).attr("x", margin.left - 12).attr("fill", css("--muted"));
    all.select(".hit").attr("width", width);

    const isDumb = view === "region";
    const bar = all.select(".bar").attr("fill", view === "fpr" ? css("--ai") : css("--series-1"));
    if (view === "auc") {
      bar.transition(t).attr("x", x(0.5)).attr("width", (d) => Math.max(0, x(d.auc) - x(0.5))).style("opacity", 1);
    } else if (view === "fpr") {
      bar.transition(t).attr("x", x(0)).attr("width", (d) => Math.max(2, x(d.fpr) - x(0))).style("opacity", 1);
    } else {
      bar.transition(t).attr("width", 0).style("opacity", 0);
    }
    all.select(".stem").attr("stroke", css("--axis")).attr("stroke-width", 2)
      .transition(t)
      .attr("x1", (d) => (isDumb ? x(Math.min(d.fpr_br, d.fpr_global)) : x(0)))
      .attr("x2", (d) => (isDumb ? x(Math.max(d.fpr_br, d.fpr_global)) : x(0)))
      .style("opacity", isDumb ? 1 : 0);
    all.select(".dot-g").attr("fill", css("--series-1")).attr("stroke", css("--surface")).attr("stroke-width", 2)
      .transition(t).attr("cx", (d) => x(isDumb ? d.fpr_global : 0)).style("opacity", isDumb ? 1 : 0);
    all.select(".dot-b").attr("fill", css("--series-2")).attr("stroke", css("--surface")).attr("stroke-width", 2)
      .transition(t).attr("cx", (d) => x(isDumb ? d.fpr_br : 0)).style("opacity", isDumb ? 1 : 0);

    all.select(".value-label").attr("fill", css("--ink"))
      .text((d) => (view === "auc" ? dec(d.auc) : view === "fpr" ? pct(d.fpr) : `${pct(d.fpr_br)} BR`))
      .transition(t)
      .attr("x", (d) => (view === "auc" ? x(d.auc) : view === "fpr" ? x(d.fpr) : x(Math.max(d.fpr_br, d.fpr_global))) + 10);

    const count = (d) => (d.real_flagged != null && d.n_real ? ` (${d.real_flagged}/${d.n_real})` : "");
    const details = (d) => `<strong>${d.name}</strong> <span style="opacity:.7">${d.model}</span><br>` +
      `${L.fpr}: ${pct(d.fpr)}${count(d)}<br>${L.auc}: ${dec(d.auc)}<br>${L.recall}: ${d.recall == null ? "n/a" : pct(d.recall)}` +
      (d.fpr_br != null ? `<br>${L.region_br}: ${pct(d.fpr_br)} · ${L.region_global}: ${pct(d.fpr_global)}` : "");
    const label = (d) => `${d.name}, ${L["kind_" + d.kind]}. ${L.fpr}: ${pct(d.fpr)}${count(d)}. ${L.auc}: ${dec(d.auc)}.` +
      (d.fpr_br != null ? ` ${L.region_br}: ${pct(d.fpr_br)}, ${L.region_global}: ${pct(d.fpr_global)}.` : "");
    const atElement = (el) => { const r = el.getBoundingClientRect(); return { clientX: r.left + 40, clientY: r.top + r.height / 2 }; };
    const highlight = (el, on) => d3.select(el).select(".bar").attr("opacity", on ? 0.8 : 1);

    // Each row is a focusable, labelled element: hover, keyboard focus and tap all show the details.
    all.attr("tabindex", 0).attr("role", "img").attr("aria-label", label)
      .on("focus", (ev, d) => { tip.show(details(d), atElement(ev.currentTarget)); highlight(ev.currentTarget, true); })
      .on("blur", (ev) => { tip.hide(); highlight(ev.currentTarget, false); });
    all.select(".hit")
      .on("mouseenter", (ev, d) => { tip.show(details(d), ev); highlight(ev.currentTarget.parentNode, true); })
      .on("mousemove", (ev) => tip.move(ev))
      .on("mouseleave", (ev) => { tip.hide(); highlight(ev.currentTarget.parentNode, false); })
      .on("click", (ev, d) => { ev.stopPropagation(); tip.show(details(d), ev); });
  }
  document.addEventListener("click", () => tip.hide());

  draw(false);
  new ResizeObserver(() => { if (Math.abs(root.clientWidth - width) > 4) draw(false); }).observe(root);
  document.addEventListener("themechange", () => draw(false));
  return {
    set(v) { view = v; draw(true); },
  };
}

// Wire a segmented control (buttons with data-view) to a chart.
function bindSegments(seg, chart, onChange) {
  seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    seg.querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    chart.set(b.dataset.view);
    if (onChange) onChange(b.dataset.view);
  }));
}

// Count a number up when it scrolls into view.
function countUp(el) {
  const target = parseFloat(el.dataset.value);
  const suffix = el.dataset.suffix || "";
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = Math.round(target) + suffix; return; }
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    const start = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - start) / 1100);
      el.textContent = Math.round(target * d3.easeCubicOut(k)) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
  io.observe(el);
}

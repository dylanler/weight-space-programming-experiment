(() => {
  "use strict";

  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const scenes = [...document.querySelectorAll("[data-scene]")];
  const projections = new WeakMap();

  function stopProjection(scene) {
    const frame = projections.get(scene);
    if (frame !== undefined) cancelAnimationFrame(frame);
    projections.delete(scene);
    scene.querySelector("[data-projection]")?.setAttribute("d", "M151 166 199 230");
  }

  function slideProjection(scene) {
    const arrow = scene.querySelector("[data-projection]");
    if (!arrow) return;
    const start = performance.now();

    function update(now) {
      const t = Math.max(0, Math.min(1, (now - start - 1150) / 900));
      const eased = 1 - (1 - t) ** 3;
      // Slide the tip from w to w-perp, parallel to u.
      arrow.setAttribute("d", `M151 166 ${303 - 104 * eased} ${152 + 78 * eased}`);
      if (t < 1) projections.set(scene, requestAnimationFrame(update));
      else projections.delete(scene);
    }

    update(start);
  }

  function play(target) {
    stopProjection(target);
    delete target.dataset.state;
    if (motion.matches) return;
    // Force a style flush so immediate replays restart CSS keyframes.
    target.getBoundingClientRect();
    target.dataset.state = "playing";
    slideProjection(target);
  }

  function make(tag, className, content) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (content !== undefined) item.textContent = content;
    return item;
  }

  function buildSearch(scene) {
    const plot = scene.querySelector(".search-plot");
    const edits = [...scene.querySelectorAll("tbody tr")].map(row => {
      const [rendered, correct] = row.lastElementChild.textContent.split("/").map(Number);
      return { name: row.cells[0].textContent, rendered, correct };
    });
    const scale = Math.max(...edits.map(edit => edit.rendered));
    plot.append(make("div", "micro search-key", "Rendered / visually correct"));
    edits.forEach(({ name, rendered, correct }, index) => {
      const node = make("div", "search-node");
      node.style.setProperty("--i", index);
      node.append(make("b", "", name));
      for (const [kind, value] of [["rendered", rendered], ["correct", correct]]) {
        const track = make("div", "search-track");
        const fill = make("span", `search-fill ${kind}`);
        fill.style.setProperty("--value", `${value / scale * 100}%`);
        track.append(fill);
        node.append(track);
      }
      node.append(make("div", "search-count", `${rendered} / ${correct}`));
      plot.append(node);
    });
    plot.hidden = false;
  }

  function buildResults(scene) {
    const ui = scene.querySelector(".results-ui");
    const chart = scene.querySelector(".chart");
    const status = scene.querySelector(".chart-status");
    const tables = new Map([...scene.querySelectorAll("[data-protocol]")].map(table => [table.dataset.protocol, table]));
    const data = row => [...row.cells].slice(0, 5).map(cell => cell.textContent.trim());

    function render(protocol, announce = false) {
      const table = tables.get(protocol);
      chart.replaceChildren(...[...table.tBodies[0].rows].map((row, index) => {
        const [name, base, edited] = data(row);
        const item = make("div", "chart-row");
        item.style.setProperty("--i", index);
        const pair = make("div", "chart-pair");
        for (const [model, value] of [["base", base], ["edited", edited]]) {
          const barRow = make("div", `chart-bar-row ${model}`);
          const bar = make("span", "chart-bar");
          bar.style.setProperty("--value", `${value}%`);
          bar.setAttribute("aria-hidden", "true");
          barRow.append(bar, make("span", "sr-only", `${model}: `), make("span", "chart-bar-value", value));
          pair.append(barRow);
        }
        item.append(make("span", "chart-label", name), pair);
        return item;
      }));

      const [, base, edited, fixed, broken] = data(table.tFoot.rows[0]);
      scene.querySelector(".fixed .flow-number").textContent = fixed;
      scene.querySelector(".broken .flow-number").textContent = broken;
      if (announce) status.textContent = `${table.caption.textContent}: ${base} base, ${edited} edited, ${fixed} fixed, ${broken} broken.`;
    }

    render(scene.querySelector("input:checked").value);
    scene.addEventListener("change", event => {
      if (!event.target.matches('input[name="protocol"]')) return;
      render(event.target.value, true);
      play(scene);
    });
    scene.querySelector(".result-details").open = false;
    ui.hidden = false;
  }

  const groups = new Map([
    ["tasks", ".task"],
    ["pipeline", ".pipeline-track li"],
  ]);
  const targetsFor = scene => {
    const selector = groups.get(scene.dataset.scene);
    return selector ? [...scene.querySelectorAll(selector)] : [scene];
  };
  const crossesTrigger = target => {
    const { top, bottom } = target.getBoundingClientRect();
    return top < innerHeight * .94 && bottom > 0;
  };

  const observer = typeof IntersectionObserver === "function" ? new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      play(entry.target);
    }
  }, { rootMargin: "0px 0px -6% 0px" }) : null;

  function watch(target) {
    if (observer) observer.observe(target);
    else if (crossesTrigger(target)) play(target);
  }

  for (const scene of scenes) {
    if (scene.dataset.scene === "search") buildSearch(scene);
    if (scene.dataset.scene === "results") buildResults(scene);
    const targets = targetsFor(scene);
    const replay = scene.querySelector(".replay");
    replay.hidden = motion.matches;
    replay.addEventListener("click", () => {
      for (const target of targets) {
        observer?.unobserve(target);
        if (target === scene || crossesTrigger(target)) play(target);
        else watch(target);
      }
    });
    if (!motion.matches) targets.forEach(watch);
  }

  motion.addEventListener("change", () => {
    observer?.disconnect();
    for (const scene of scenes) {
      scene.querySelector(".replay").hidden = motion.matches;
      for (const target of targetsFor(scene)) {
        delete target.dataset.state;
        stopProjection(target);
        if (!motion.matches) watch(target);
      }
    }
  });

  const progress = document.querySelector(".read-progress");
  let pending = false;
  function measureProgress() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      const length = document.documentElement.scrollHeight - innerHeight;
      progress.style.transform = `scaleX(${length > 0 ? Math.min(1, Math.max(0, scrollY / length)) : 0})`;
      pending = false;
    });
  }
  addEventListener("scroll", measureProgress, { passive: true });
  addEventListener("resize", measureProgress);
  measureProgress();
})();

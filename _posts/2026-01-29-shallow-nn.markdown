---
layout: single
title:  "Shallow Neural Network Intuition"
date:   2026-01-29 12:00:00 +0100
show_date: true
categories: ds
tags: ds
toc: false
---

In this post i want to summarize the intuition behind shallow networks
as shown by Simon J.D. Prince in *Understanding Deep Learning*. I extended the visual plots to include additional activation functions.

Stated by the [*Universal Approximation Theorem*](https://en.wikipedia.org/wiki/Universal_approximation_theorem),
for any continuous function there exists a shallow network 
that can approximate this function to any specified precision.

That is for input $$ x \in \mathbb{R}^{D_i} $$ to output $$ y \in \mathbb{R}^{D_0} $$
using hidden units:

$$
    h_d = a(\theta_{d0} + \sum_{i=1}{D_i} \theta_{di} x_i)
$$

Which are combined linearly to:

$$
    y_j = \phi_{j0} + \sum_{d=1}^{D} \phi_{jd} h_d
$$

This was proven by Cybenko in 1989 for sigmoid activations and later extended by Hornik
for a larger class of nonlinear activations  in 1991. Moshe Leshno et al. (1993) and Allan Pinkus (1999) proved that a network is a universal approximator if and only if the activation function is **non-polynomial**.

# Visual Intuition

We show this by using the following 3 unit network:

$$
    \begin{split}
        h_i = a[\theta_{i0} + \theta_{i1} x]
    \end{split}
$$

$$
    y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3
$$

<div class="nn-demo" id="nn-demo-1">
  <div class="nn-controls"></div>
  <div class="nn-plot"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(() => {
  const root = document.getElementById("nn-demo-1");
  const controlsHost = root.querySelector(".nn-controls");
  const plotHost = root.querySelector(".nn-plot");
  const style = document.createElement("style");

  style.textContent = `
    .nn-demo { width: 100%; box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    
    .nn-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px 10px;
      margin: 8px 0 10px;
      align-items: start;
    }
    .nn-ctl {
      padding: 7px 9px;
      border: 1px solid #ddd;
      border-radius: 12px;
      box-sizing: border-box;
      background: transparent;
    }
    .nn-ctl label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      line-height: 1.1;
      margin-bottom: 5px;
      gap: 10px;
      white-space: nowrap;
    }
    .nn-ctl input[type="range"] { width: 100%; height: 18px; }
    .nn-ctl select {
      width: 100%;
      padding: 6px 8px;
      border-radius: 10px;
      border: 1px solid #ddd;
      font-size: 12px;
      box-sizing: border-box;
      background: transparent;
    }

    .nn-plot {
      width: 100%;
      height: 700px; /* fallback; JS will overwrite */
    }

    @media (max-width: 520px) {
      .nn-title { font-size: 16px; }
      .nn-controls { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
    }
  `;
  document.head.appendChild(style);

  const N = 400;
  const start = -10, stop = 10;
  const X = Array.from({length: N}, (_, i) => start + (stop - start) * (i / (N - 1)));

  function lin(theta0, theta1) {
    return X.map(x => theta0 + theta1 * x);
  }

  function sigmoidScalar(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    } else {
      const z = Math.exp(x);
      return z / (1 + z);
    }
  }

  const LEAKY_ALPHA = 0.01;

  const activations = {
    relu: (x) => Math.max(0, x),
    leaky_relu: (x) => (x >= 0 ? x : LEAKY_ALPHA * x),
    sigmoid: (x) => sigmoidScalar(x),
    tanh: (x) => Math.tanh(x),
    heaviside: (x) => (x >= 0 ? 1 : 0),
    rect: (x) => Math.max(0, x),
  };

  function applyActivation(arr, actKey) {
    const f = activations[actKey] || activations.relu;
    return arr.map(f);
  }

  function forward(p) {
    const l1 = lin(p.theta10, p.theta11);
    const l2 = lin(p.theta20, p.theta21);
    const l3 = lin(p.theta30, p.theta31);

    const a1 = applyActivation(l1, p.activation);
    const a2 = applyActivation(l2, p.activation);
    const a3 = applyActivation(l3, p.activation);

    const z1 = a1.map(v => p.phi1 * v);
    const z2 = a2.map(v => p.phi2 * v);
    const z3 = a3.map(v => p.phi3 * v);

    const out = a1.map((_, i) =>
      p.phi0 + p.phi1 * a1[i] + p.phi2 * a2[i] + p.phi3 * a3[i]
    );

    return { l1, l2, l3, a1, a2, a3, z1, z2, z3, out };
  }

  function domains3Cols(gap=0.025) {
    const totalGap = 2 * gap;
    const w = (1 - totalGap) / 3;
    return [
      [0, w],
      [w + gap, 2*w + gap],
      [2*w + 2*gap, 1]
    ];
  }

  function domains4Rows(gap=0.045) {
    const totalGap = 3 * gap;
    const h = (1 - totalGap) / 4;
    const r1 = [1 - h, 1];
    const r2 = [1 - 2*h - gap, 1 - h - gap];
    const r3 = [1 - 3*h - 2*gap, 1 - 2*h - 2*gap];
    const r4 = [0, h];
    return [r1, r2, r3, r4];
  }

  const xD = domains3Cols(0.025);
  const yD = domains4Rows(0.045);

  function axisIndex(row, col) {
    return (row - 1) * 3 + col; // 1..9
  }

    const PARAMS = [
        // hidden unit 1: left-shifted, positive slope
        {id:"theta10", label:"theta10", min:-5, max:5, step:0.1, value:-2.0},
        {id:"theta11", label:"theta11", min:-5, max:5, step:0.1, value: 1.2},

        // hidden unit 2: centered, negative slope
        {id:"theta20", label:"theta20", min:-5, max:5, step:0.1, value: 0.5},
        {id:"theta21", label:"theta21", min:-5, max:5, step:0.1, value:-0.8},

        // hidden unit 3: right-shifted, steeper slope
        {id:"theta30", label:"theta30", min:-5, max:5, step:0.1, value: 2.5},
        {id:"theta31", label:"theta31", min:-5, max:5, step:0.1, value: 1.6},

        // output layer
        {id:"phi0", label:"phi0", min:-5, max:5, step:0.1, value: 0.3},
        {id:"phi1", label:"phi1", min:-5, max:5, step:0.1, value: 1.4},
        {id:"phi2", label:"phi2", min:-5, max:5, step:0.1, value:-1.1},
        {id:"phi3", label:"phi3", min:-5, max:5, step:0.1, value: 0.8},
    ];

  function setActivationNote(act, el) {
    if (act === "sigmoid") el.textContent = "range ~ [0,1]";
    else if (act === "tanh") el.textContent = "range ~ [-1,1]";
    else if (act === "heaviside") el.textContent = "outputs 0/1";
    else if (act === "leaky_relu") el.textContent = "keeps negatives";
    else el.textContent = "";
  }

  function makeControls() {
    controlsHost.innerHTML = "";

    // Activation selector
    const actBox = document.createElement("div");
    actBox.className = "nn-ctl";
    actBox.style.gridColumn = "1 / -1";
    actBox.innerHTML = `
      <label>
        <span>activation</span>
        <span id="act_note" style="color:#666; font-size:12px;"></span>
      </label>
      <select id="activation">
        <option value="relu">ReLU</option>
        <option value="leaky_relu">Leaky ReLU (α=${LEAKY_ALPHA})</option>
        <option value="sigmoid">Sigmoid</option>
        <option value="tanh">tanh</option>
        <option value="heaviside">Heaviside step</option>
      </select>
    `;
    controlsHost.appendChild(actBox);

    // Sliders
    for (const p of PARAMS) {
      const box = document.createElement("div");
      box.className = "nn-ctl";
      box.innerHTML = `
        <label>
          <span>${p.label}</span>
          <span id="val_${p.id}">${Number(p.value).toFixed(1)}</span>
        </label>
        <input id="${p.id}" type="range"
               min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}">
      `;
      controlsHost.appendChild(box);
    }

    const actSel = root.querySelector("#activation");
    const note = root.querySelector("#act_note");
    setActivationNote(actSel.value, note);
  }
  makeControls();

  function getParams() {
    const obj = { activation: root.querySelector("#activation").value };
    for (const p of PARAMS) {
      const v = parseFloat(root.querySelector("#" + p.id).value);
      obj[p.id] = v;
      root.querySelector("#val_" + p.id).textContent = v.toFixed(1);
    }
    return obj;
  }

  function trace(x, y, ax, ay, name) {
    return { x, y, mode: "lines", xaxis: ax, yaxis: ay, name, hoverinfo: "x+y" };
  }

  const p0 = getParams();
  const f0 = forward(p0);

  const data = [
    trace(X, f0.l1, "x",   "y",   "l1"),
    trace(X, f0.l2, "x2",  "y2",  "l2"),
    trace(X, f0.l3, "x3",  "y3",  "l3"),

    trace(X, f0.a1, "x4",  "y4",  "A(l1)"),
    trace(X, f0.a2, "x5",  "y5",  "A(l2)"),
    trace(X, f0.a3, "x6",  "y6",  "A(l3)"),

    trace(X, f0.z1, "x7",  "y7",  "phi1*A(l1)"),
    trace(X, f0.z2, "x8",  "y8",  "phi2*A(l2)"),
    trace(X, f0.z3, "x9",  "y9",  "phi3*A(l3)"),

    trace(X, f0.out,"x10", "y10", "output"),
  ];

  const layout = {
    title: { text: "Component Contributions", x: 0.02, font: { size: 16 } },
    margin: { l: 42, r: 12, t: 42, b: 32 },
    showlegend: false,
  };

  // Axes 1..9
  for (let r = 1; r <= 3; r++) {
    for (let c = 1; c <= 3; c++) {
      const idx = axisIndex(r, c); // 1..9
      const xa = (idx === 1) ? "xaxis" : "xaxis" + idx;
      const ya = (idx === 1) ? "yaxis" : "yaxis" + idx;

      layout[xa] = {
        domain: xD[c-1],
        anchor: "y" + (idx === 1 ? "" : idx),
        zeroline: false,
        showgrid: true,
        ticks: "outside",
        tickfont: { size: 10 },
      };
      layout[ya] = {
        domain: yD[r-1],
        anchor: "x" + (idx === 1 ? "" : idx),
        zeroline: true,
        showgrid: true,
        ticks: "outside",
        tickfont: { size: 10 },
      };
    }
  }

  layout["xaxis10"] = {
    domain: [0, 1],
    anchor: "y10",
    zeroline: false,
    showgrid: true,
    ticks: "outside",
    tickfont: { size: 10 },
  };
  layout["yaxis10"] = {
    domain: yD[3],
    anchor: "x10",
    zeroline: true,
    showgrid: true,
    ticks: "outside",
    tickfont: { size: 10 },
  };

  function computePlotHeight(containerWidth) {
    // Tune these numbers to taste:
    // keeps it proportional
    // clamps too tall/too small
    const ratio = 0.75;
    const h = containerWidth * ratio;
    const minH = 520;
    const maxH = 820;                   // cap height
    return Math.max(minH, Math.min(maxH, Math.round(h)));
  }

  function applySize() {
    const w = root.clientWidth || 900;
    const h = computePlotHeight(w);
    plotHost.style.height = h + "px";
    // Relayout height + trigger resize for crisp axes
    Plotly.relayout(plotHost, { height: h });
    Plotly.Plots.resize(plotHost);
  }

  Plotly.newPlot(plotHost, data, layout, { responsive: true }).then(() => {
    applySize();
  });

  const ro = new ResizeObserver(() => applySize());
  ro.observe(root);

  const actNameMap = {
    relu: "ReLU",
    leaky_relu: `LeakyReLU(α=${LEAKY_ALPHA})`,
    sigmoid: "sigmoid",
    tanh: "tanh",
    heaviside: "Heaviside",
    rect: "rect",
  };

  function updatePlot() {
    const actSel = root.querySelector("#activation");
    const note = root.querySelector("#act_note");
    setActivationNote(actSel.value, note);

    const p = getParams();
    const f = forward(p);

    Plotly.restyle(plotHost, {
      y: [
        f.l1, f.l2, f.l3,
        f.a1, f.a2, f.a3,
        f.z1, f.z2, f.z3,
        f.out
      ],
      name: [
        "l1","l2","l3",
        `${actNameMap[p.activation]}(l1)`, `${actNameMap[p.activation]}(l2)`, `${actNameMap[p.activation]}(l3)`,
        `phi1*${actNameMap[p.activation]}(l1)`, `phi2*${actNameMap[p.activation]}(l2)`, `phi3*${actNameMap[p.activation]}(l3)`,
        "output"
      ]
    });
  }

  for (const p of PARAMS) {
    root.querySelector("#" + p.id).addEventListener("input", updatePlot);
  }
  root.querySelector("#activation").addEventListener("change", updatePlot);
})();
</script>
<p></p>

# ReLU

With ReLU activation functions, a network with $$ D $$ hidden units represents a continuous piecewise linear function with at most $$ D $$ joints and $$ D+1 $$ linear regions.
Each hidden unit introduces a single joint where its activation switches between inactive and active.

![ReLU Joints](/assets/images/2026-01-29-shallow-nn/joints.png)

As more hidden units are added, the number of linear regions increases, allowing the network to approximate more complex functions.
Each linear region corresponds to a distinct activation pattern of the hidden units.
In regions where all units are inactive, the output reduces to the bias term.
Although a network with three hidden units can form four linear regions, only three of the slopes are independent, as the remaining slope is determined by the active units in the other regions.

![Not Activated Regions](/assets/images/2026-01-29-shallow-nn/not-activated.png)

## In Multiple Dimensions TBD

For multivariate output function:

$$
  \begin{split}
      y_1 = \phi_{10} + \phi_{11} h_1 + \phi_{12} h_2 + \phi_{13} h_3 + \phi_{14} h_4 \\
      y_2 = \phi_{20} + \phi_{21} h_1 + \phi_{22} h_2 + \phi_{23} h_3 + \phi_{24} h_4
  \end{split}
$$

For the intuition above this will create two functions, where the joints of the
linear regions coincide:

![Joints](/assets/images/2026-01-29-shallow-nn/multivariate-output.png)

<p></p>


![Shallow Water](/assets/images/2026-01-29-shallow-nn/shallow-water.jpg)
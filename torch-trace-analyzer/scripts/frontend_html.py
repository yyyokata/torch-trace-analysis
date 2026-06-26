#!/usr/bin/env python3
"""HTML flowchart rendering for the torch-trace-analyzer DAG.

This module renders the interactive HTML output (``generate_html_flowchart``,
``generate_html_flowchart_dual``, ``build_timing_data_from_trace`` and
``_generate_flowchart_html``).  All DAG analysis primitives live in the
sibling ``analyze_trace`` module which is imported explicitly below — there
is **no** dynamic ``importlib.util.spec_from_file_location`` re-exec, no
``globals().update`` and no module-name aliasing here.

Dependency direction is strictly one way:

    analyze_trace  ──provides─▶  ASTFrontend, _build_class_map, ...
    frontend_html  ──consumes──┘

``analyze_trace`` re-exports the public entry points of this module at the
end of its own file purely for backward compatibility with callers that
reach those names through ``analyze_trace`` (for example
``testset/test_dag_rules.py``).  That re-export is the only "back" reference
and it runs *after* every name imported below has been bound, so there is
no real circular dependency.
"""
from __future__ import annotations

import os
import sys
import json as _json
import re
from pathlib import Path

# When this module is imported by ``analyze_trace.py`` the script directory is
# already on ``sys.path``.  We add it defensively here so the module also
# imports cleanly when loaded standalone (e.g. ``python -c "import
# frontend_html"`` from the scripts/ directory or from a tooling harness).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RENDER_GROUP_JS_PATH = Path(_SCRIPT_DIR) / "render_group.js"
_RENDER_GROUP_JS_PLACEHOLDER = "__RENDER_GROUP_JS_PLACEHOLDER__"
# Canvas Phase 1 / Stage 1.1: the runtime render path switches from the legacy
# SVG `render_group.js` injection to a Pixi engine bundle + `render_canvas.js`.
_ENGINE_BUNDLE_PATH = Path(_SCRIPT_DIR) / "pixi_engine_bundle.js"
_ENGINE_BUNDLE_PLACEHOLDER = "__ENGINE_BUNDLE_PLACEHOLDER__"
_RENDER_CANVAS_JS_PATH = Path(_SCRIPT_DIR) / "render_canvas.js"
_RENDER_CANVAS_JS_PLACEHOLDER = "__RENDER_CANVAS_JS_PLACEHOLDER__"
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


FLOWCHART_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Module DAG Flowchart - Torch Trace Analyzer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { overflow-anchor: none; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; min-height: 100vh; }
/* Phase 2 step 5 — unified single-row sticky topbar (mockup4).  Replaces the
   old centered header / controls / standalone legend + floating breadcrumb. */
.topbar { position: sticky; top: 0; z-index: 200; height: 48px; box-sizing: border-box; margin: -20px -20px 16px -20px; padding: 0 16px; display: flex; align-items: center; justify-content: space-between; gap: 16px; background: #16213e; border-bottom: 1px solid rgba(255,255,255,0.08); box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
.topbar-left { display: flex; align-items: center; min-width: 0; overflow-x: auto; white-space: nowrap; }
.topbar-right { display: flex; align-items: center; gap: 8px; flex-shrink: 0; white-space: nowrap; }
.tb-capsule { background: #1e3a5f; color: #72c0ff; border-radius: 6px; padding: 4px 10px; font-size: 12px; line-height: 1.4; cursor: pointer; user-select: none; }
.tb-capsule::after { content: '▾'; margin-left: 5px; font-size: 10px; color: #72c0ff; }
.tb-capsule.tb-capsule-disabled { cursor: default; }
.tb-capsule.tb-capsule-disabled::after { color: #4a5568; }
.tb-nav-sep { color: #4a5568; margin: 0 6px; font-size: 13px; }
/* Phase 2 step 5 — Training / runstep capsule dropdown menu (multi-tab L1/L2
   switcher).  Absolutely positioned under the clicked capsule, above the
   sticky topbar (z-index 200). */
.tb-dropdown-menu { position: fixed; z-index: 300; background: #16213e; border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.5); padding: 4px; min-width: 140px; max-height: 60vh; overflow-y: auto; }
.tb-dropdown-item { padding: 6px 12px; margin: 2px 0; font-size: 12px; line-height: 1.4; color: #cbd5e1; background: #1e3a5f; border-radius: 4px; cursor: pointer; white-space: nowrap; }
.tb-dropdown-item:hover { background: #2a4d7a; color: #ffffff; }
.tb-dropdown-item.tb-dropdown-active { color: #72c0ff; font-weight: 600; }
/* Focus-path breadcrumb segments (Semantic Zoom) — appended into
   #dag-breadcrumb-nav after the static Training / runstep capsules. */
.focus-breadcrumb-seg { font-size: 12px; line-height: 1.4; }
.focus-breadcrumb-sep { color: #4a5568; margin: 0 6px; font-size: 13px; }
.focus-breadcrumb-clickable { color: #8892b0; cursor: pointer; }
.focus-breadcrumb-clickable:hover { color: #cbd5e1; }
.focus-breadcrumb-current { color: #ffffff; font-weight: 600; }
.tb-btn { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); color: #ccc; padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
.tb-btn:hover { background: rgba(255,255,255,0.15); color: #fff; }
.tb-btn.active { background: rgba(100,181,246,0.2); border-color: #64b5f6; color: #64b5f6; }
.tb-divider { width: 1px; height: 24px; background: rgba(255,255,255,0.15); margin: 0 6px; }
.tb-legend { display: flex; align-items: center; gap: 12px; }
.tb-legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #8892b0; }
.tb-legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.tb-legend-sq { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.tb-legend-dep { color: #2ecc71; font-weight: bold; }
.dag-container { width: 100%; overflow-x: hidden; position: relative; }
.dag-stage { display: block; margin: 0 auto; width: 100%; overflow: visible; overflow-anchor: none; }
.dag-stage canvas { display: block; width: 100%; }
.tooltip { position: fixed; background: #16213e; border: 1px solid rgba(100,181,246,0.3); border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #e0e0e0; pointer-events: none; z-index: 1000; max-width: 300px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); opacity: 0; transition: opacity 0.15s; }
.tooltip.visible { opacity: 1; }
.tooltip .tt-title { font-weight: 600; color: #64b5f6; margin-bottom: 4px; }
.tooltip .tt-row { margin-top: 2px; }
.summary { margin-top: 24px; padding: 14px 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }
.summary h3 { font-size: 13px; color: #64b5f6; margin-bottom: 6px; }
.summary p { font-size: 11px; color: #8892b0; line-height: 1.7; }
/* Side panel for click-to-source UX (module class definitions / dep edge evidence) */
.side-panel { position: fixed; top: 0; right: -560px; width: 540px; height: 100vh; background: #0f1626; border-left: 1px solid rgba(100,181,246,0.3); box-shadow: -8px 0 30px rgba(0,0,0,0.5); transition: right 0.25s ease; z-index: 2000; display: flex; flex-direction: column; }
.side-panel.open { right: 0; }
.side-panel-header { padding: 14px 18px; border-bottom: 1px solid rgba(255,255,255,0.08); display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
.side-panel-title { font-size: 13px; font-weight: 600; color: #64b5f6; word-break: break-word; }
.side-panel-subtitle { font-size: 10px; color: #8892b0; margin-top: 3px; font-family: monospace; word-break: break-all; }
.side-panel-close { background: none; border: none; color: #8892b0; font-size: 22px; cursor: pointer; padding: 0 6px; line-height: 1; }
.side-panel-close:hover { color: #fff; }
.side-panel-body { flex: 1 1 auto; overflow: auto; padding: 12px 0; }
.side-panel-section { margin-bottom: 14px; }
.side-panel-section h4 { font-size: 11px; color: #8892b0; padding: 0 18px 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.code-block { background: #050a14; margin: 0 12px; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 10px 0; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; font-size: 11.5px; line-height: 1.55; color: #d6e1f5; overflow: auto; max-height: 360px; }
.code-block .code-line { display: flex; padding: 0 8px; white-space: pre; }
.code-line .lineno { flex-shrink: 0; width: 48px; color: #4a5970; text-align: right; padding-right: 12px; user-select: none; }
.code-line .code-text { white-space: pre; flex: 1; }
.code-line .code-var-hit { background: rgba(255, 209, 102, 0.22); color: #fff4c2; border-radius: 3px; padding: 0 2px; box-shadow: inset 0 0 0 1px rgba(255, 209, 102, 0.18); }
.code-line.highlight { background: rgba(100,181,246,0.12); }
.code-line.highlight .lineno { color: #64b5f6; font-weight: bold; }
.code-line .code-text mark.var-mark { background: rgba(255, 209, 102, 0.25); color: #fff4c2; border-radius: 3px; padding: 0 2px; box-shadow: inset 0 0 0 1px rgba(255, 209, 102, 0.35); font-weight: 600; }
.code-line .code-text mark.var-mark.producer-mark { background: rgba(46, 204, 113, 0.30); color: #d5f5e3; box-shadow: inset 0 0 0 1px rgba(46, 204, 113, 0.40); }
.code-line .code-text mark.var-mark.consumer-mark { background: rgba(255, 209, 102, 0.25); color: #fff4c2; box-shadow: inset 0 0 0 1px rgba(255, 209, 102, 0.35); }
.dag-container.hover-active .leaf-node.node-dim,
.dag-container.hover-active .group-box.node-dim,
.dag-container.hover-active .io-node.node-dim,
.dag-container.hover-active .group-label.node-dim,
.dag-container.hover-active .group-timing.node-dim { opacity: 0.25 !important; }
.dag-container.hover-active .leaf-node.node-active,
.dag-container.hover-active .group-box.node-active,
.dag-container.hover-active .io-node.node-active { opacity: 1 !important; stroke: #FFD166 !important; stroke-width: 3 !important; }
.evidence-meta { padding: 6px 18px 10px; font-size: 11px; color: #aab4cf; }
.evidence-meta b { color: #64b5f6; }
.evidence-meta code { background: rgba(100,181,246,0.12); padding: 1px 5px; border-radius: 3px; font-family: monospace; color: #ffffff; }
.flow-block { margin: 0 12px 10px; border: 1px solid rgba(255,255,255,0.07); border-radius: 6px; overflow: hidden; }
.flow-block-head { display: flex; align-items: center; gap: 8px; padding: 7px 10px; cursor: pointer; user-select: none; background: rgba(100,181,246,0.06); font-size: 11px; color: #c0c8d8; }
.flow-block-head:hover { background: rgba(100,181,246,0.12); }
.flow-block-head .flow-toggle { color: #64b5f6; font-size: 10px; min-width: 10px; }
.flow-block-head .flow-dtype { background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px; font-family: monospace; font-size: 10px; color: #ffd166; }
.flow-block-head .flow-shape { font-family: monospace; font-size: 10px; color: #a8d8a8; }
.flow-block-body { display: none; }
.flow-block-body.open { display: block; }
.flow-step { padding: 6px 10px 0; }
.flow-step-code { margin: 4px 0 6px; }
.timing-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.timing-label { display: inline-flex; align-items: center; gap: 6px; }
.metric-help { display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; border: 1px solid rgba(100,181,246,0.45); color: #8fc5ff; font-size: 10px; line-height: 1; cursor: help; position: relative; }
.metric-help:hover { background: rgba(100,181,246,0.14); color: #d9eeff; }
.metric-help-popup { position: fixed; background: #1a2a4a; border: 1px solid rgba(100,181,246,0.5); border-radius: 6px; padding: 7px 10px; font-size: 11px; color: #cde; pointer-events: none; z-index: 9999; max-width: 260px; box-shadow: 0 4px 16px rgba(0,0,0,0.5); line-height: 1.5; display: none; }
.timing-value { color: #e7eefc; font-variant-numeric: tabular-nums; }
.truncation-note { font-size: 10px; color: #c77a3c; padding: 6px 18px 0; }
.iter14-render-progress-overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: rgba(26, 26, 46, 0.86);
    z-index: 3000;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transition: opacity 0.25s ease, visibility 0.25s ease;
}
.iter14-render-progress-overlay.visible {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
}
.iter14-render-progress-overlay.closing {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
}
.iter14-render-progress-card {
    width: min(420px, calc(100vw - 48px));
    padding: 22px 24px;
    border-radius: 14px;
    border: 1px solid rgba(124, 131, 253, 0.35);
    background: rgba(15, 22, 38, 0.94);
    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
}
.iter14-render-progress-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}
.iter14-render-progress-stage {
    font-size: 13px;
    color: #8892b0;
    margin-bottom: 14px;
}
.iter14-render-progress-track {
    width: 100%;
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
}
.iter14-render-progress-bar {
    width: 0%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #7c83fd, #64b5f6);
    transition: width 0.12s ease-out, background 0.18s ease;
}
.iter14-render-progress-percent {
    margin-top: 12px;
    font-size: 13px;
    font-weight: 600;
    color: #e7eefc;
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.iter14-render-progress-overlay.failed .iter14-render-progress-stage {
    color: #ffb4a8;
}
.iter14-render-progress-overlay.failed .iter14-render-progress-bar {
    background: linear-gradient(90deg, #ff6b6b, #ff8e53);
}
</style>
</head>
<body>
<!-- Phase 2 step 5 — unified single-row sticky topbar (mockup4).  Left:
     Training / runstep capsules + dynamic Semantic-Zoom focus path
     (#dag-breadcrumb-nav).  Right: graph controls + inline static legend. -->
<div class="topbar">
    <div class="topbar-left" id="dag-breadcrumb-nav">
        <span class="tb-capsule" id="tb-mode-capsule">Training</span>
        <span class="tb-nav-sep">›</span>
        <span class="tb-capsule tb-capsule-disabled" id="tb-runstep-capsule">runstep</span>
    </div>
    <div class="topbar-right">
        <button id="btn-expand-all" class="tb-btn">Expand All</button>
        <button id="btn-collapse-all" class="tb-btn">Collapse All</button>
        <button id="btn-fit" class="tb-btn">Fit to View</button>
        <span class="tb-divider"></span>
        <span class="tb-legend">
            <span class="tb-legend-item"><span class="tb-legend-dot" style="background:#4a6fa5"></span>Depth0</span>
            <span class="tb-legend-item"><span class="tb-legend-sq" style="background:#5b8c5a"></span>Depth1</span>
            <span class="tb-legend-item"><span class="tb-legend-sq" style="background:#8e6fad"></span>Depth2</span>
            <span class="tb-legend-item"><span class="tb-legend-sq" style="background:#c77a3c"></span>Depth3+</span>
            <span class="tb-legend-item"><span class="tb-legend-dep">━▶</span>dep</span>
        </span>
    </div>
</div>
<div class="dag-container" id="dag-container">
    <div id="dag-stage" class="dag-stage"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<div class="side-panel" id="side-panel">
    <div class="side-panel-header">
        <div>
            <div class="side-panel-title" id="sp-title">—</div>
            <div class="side-panel-subtitle" id="sp-subtitle"></div>
        </div>
        <button class="side-panel-close" id="sp-close" title="Close">×</button>
    </div>
    <div class="side-panel-body" id="sp-body"></div>
</div>
<div class="summary" id="summary"></div>
<div id="render-progress-overlay" class="iter14-render-progress-overlay" aria-hidden="true">
    <div class="iter14-render-progress-card">
        <div class="iter14-render-progress-title">正在渲染 DAG…</div>
        <div class="iter14-render-progress-stage" id="render-progress-stage">准备中…</div>
        <div class="iter14-render-progress-track">
            <div class="iter14-render-progress-bar" id="render-progress-bar"></div>
        </div>
        <div class="iter14-render-progress-percent" id="render-progress-percent">0%</div>
    </div>
</div>

<script id="engine-bundle">
__ENGINE_BUNDLE_PLACEHOLDER__
</script>
<script id="render-canvas-js">
__RENDER_CANVAS_JS_PLACEHOLDER__
</script>

<script>
const DATA = __FLOWCHART_DATA_PLACEHOLDER__;
// __SOURCE_MAP_PLACEHOLDER__

const groupMap = {};
const nodeMap = {};
const collapsedState = {};
// Phase 2 step 3: expose ``collapsedState`` for the canvas engine + tests.
// ``window.__canvasOnGroupToggle`` mutates this map; ``window.__canvas_collapsed_state``
// is the read alias used by UTs to assert toggle behaviour.
if (typeof window !== 'undefined') { window.__canvas_collapsed_state = collapsedState; }
// Phase 2 step 5 — Semantic Zoom: ``focusStack`` is the *single source of
// truth* for drill-down focus state (see design/frontend_canvas_phase2.md
// §6.3).  An empty array means full graph; a non-empty array's last element
// is the current focus root.  ``window.__canvas_focus_stack`` mirrors the
// ``__canvas_collapsed_state`` pattern so UTs can assert focus state without
// touching the engine.
const focusStack = [];
if (typeof window !== 'undefined') { window.__canvas_focus_stack = focusStack; }
let groupLayout = {};
let edgeDomRegistry = [];
const edgeByNodeId = new Map();
const edgeByGroupId = new Map();
const edgeDomByKey = new Map();
let prevActiveItems = [];
let prevActiveNodeIds = new Set();
let prevActiveEdgeKeys = new Set();
let prevActiveGroupNodeIds = new Set();
let prevActiveGroupIds = new Set();
let prevDimEdgeKeys = new Set();
let prevDimGroupNodeIds = new Set();
let prevDimGroupIds = new Set();
let lastHoveredGid = null;
let focusedEdgePath = null;
let hoveredEdgeKey = null;
let hoveredEdges = [];
let hoveredEdgeIdx = 0;
let hoveredGroupId = null;
let hoveredNodeId = null;
let nodeDomRegistry = new Map();
const groupDomRegistry = new Map();
let renderGeneration = 0;
const nodeAncestorGroups = new Map();
const LONG_EDGE_MIN_SPAN = 260;

function edgeKey(edge) {
    if (!edge) return '';
    return [edge.from || '', edge.to || '', edge.type || '', edge.parent_class || ''].join('||');
}

function bundleKey(direction, nid) {
    return `${direction}::${nid || ''}`;
}

function collectAllRenderableEdges() {
    return [...(DATA.edges || [])];
}

function buildEdgeBundleState() {
    const pairBuckets = new Map();
    const sourceBuckets = new Map();
    const targetBuckets = new Map();
    for (const e of collectAllRenderableEdges()) {
        const fanKey = `${e.from || ''}=>${e.to || ''}`;
        if (!pairBuckets.has(fanKey)) pairBuckets.set(fanKey, []);
        pairBuckets.get(fanKey).push(e);

        const outKey = bundleKey('out', e.from || '');
        const inKey = bundleKey('in', e.to || '');
        if (!sourceBuckets.has(outKey)) sourceBuckets.set(outKey, []);
        if (!targetBuckets.has(inKey)) targetBuckets.set(inKey, []);
        sourceBuckets.get(outKey).push(e);
        targetBuckets.get(inKey).push(e);
    }

    const edgeMeta = new Map();
    for (const list of pairBuckets.values()) {
        const total = list.length;
        list.forEach((e, idx) => {
            const base = total <= 1 ? 0 : (idx - (total - 1) / 2);
            const sourceCount = (sourceBuckets.get(bundleKey('out', e.from || '')) || []).length || 1;
            const targetCount = (targetBuckets.get(bundleKey('in', e.to || '')) || []).length || 1;
            const spread = Math.max(sourceCount, targetCount, total);
            edgeMeta.set(edgeKey(e), {
                bundleIndex: idx,
                bundleTotal: total,
                bundleOffset: base * Math.min(18, 7 + spread * 1.2),
                sourceFanout: sourceCount,
                targetFanin: targetCount,
            });
        });
    }
    return { edgeMeta };
}
const EDGE_BUNDLE_STATE = buildEdgeBundleState();
const EDGE_BUNDLE_META = EDGE_BUNDLE_STATE.edgeMeta;

function isEdgeBundled(edge) {
    return false;
}

function isEdgeVisible(edge) {
    return true;
}

function resolveCollapsedAncestor(nodeId) {
    // nodeAncestorGroups 存储顺序：从根到近（外层 group 在前）
    // 从前往后遍历找最外层的已折叠祖先。
    //
    // Phase 2 step 5 (problem 5 — ReturnVal edge): in focus mode the focus
    // root subtree is cascade-expanded, but the focus root's *parent* chain
    // stays collapsed.  A boundary-port node id (e.g. a Group out_port that
    // backs a ReturnVal return edge) carries the full ancestor chain
    // ``[root, ..., parentGroup, focusGroup]``.  Walking from index 0 would
    // hit the collapsed ``parentGroup`` and redirect the focus group's own
    // out-port onto the parent card.  Anchoring the scan at the current focus
    // root makes the resolution honour the locally-expanded subtree: only the
    // focus root and the ancestors *below* it are considered, so an out-port
    // owned by the (expanded) focus group resolves to the port id itself and
    // ``portPointForEndpoint`` can anchor it to the focus group's real box.
    const ancestors = nodeAncestorGroups.get(nodeId) || [];
    let startIdx = 0;
    if (focusStack && focusStack.length > 0) {
        const rootId = focusStack[focusStack.length - 1];
        const idx = ancestors.findIndex(a => String(a) === String(rootId));
        if (idx !== -1) startIdx = idx;
    }
    for (let i = startIdx; i < ancestors.length; i++) {
        if (collapsedState[ancestors[i]]) return ancestors[i];
    }
    return nodeId;
}

function toggleBundle(bundleId) {
    return;
}

function registerBundleWidget(group, bundleId) {
    return;
}

function renderBundleWidget(bundleId, cx, cy, direction) {
    return;
}

function getActiveEdgeItems() {
    if (focusedEdgePath) return edgeDomRegistry.filter(i => i.key === focusedEdgePath);
    if (hoveredEdgeKey) return edgeDomRegistry.filter(i => i.key === hoveredEdgeKey);
    if (hoveredNodeId != null) return edgeByNodeId.get(hoveredNodeId) ?? [];
    if (hoveredGroupId != null) return edgeByGroupId.get(hoveredGroupId) ?? [];
    return [];
}

function setEdgeItemFocusState(item, state) {
    if (!item || !item.path) return;
    const p = item.path;
    if (state === 'active') {
        p.classList.add('edge-active');
        p.classList.remove('edge-dim');
        syncLongEdgeDisplay(p, true);
        return;
    }
    if (state === 'dim') {
        p.classList.add('edge-dim');
        p.classList.remove('edge-active');
        syncLongEdgeDisplay(p, false);
        return;
    }
    p.classList.remove('edge-active', 'edge-dim');
    syncLongEdgeDisplay(p, false);
}

function clearActiveEdgeOverlay() {
    return;
}

function syncActiveEdgeOverlay(item) {
    return;
}

function _applyNodeFocusStyle(el, state) {
    if (state === 'active') {
        el.classList.add('node-active');
        el.classList.remove('node-dim');
        return;
    }
    if (state === 'dim') {
        el.classList.add('node-dim');
        el.classList.remove('node-active');
        return;
    }
    el.classList.remove('node-active', 'node-dim');
}

function setNodeFocusStateById(nodeId, state) {
    const els = nodeDomRegistry.get(nodeId) || [];
    for (const el of els) {
        _applyNodeFocusStyle(el, state);
    }
}

function setGroupFocusStateById(gid, state) {
    const dom = groupDomRegistry.get(gid);
    if (!dom) return;
    for (const el of Object.values(dom)) {
        if (!el) continue;
        _applyNodeFocusStyle(el, state);
    }
}

function syncLongEdgeDisplay(path, isActive) {
    if (!path || path.dataset.longEdge !== '1') {
        return;
    }
    if (isActive) {
        path.classList.add('edge-full');
        if (path.dataset.fullDasharray) {
            path.setAttribute('stroke-dasharray', path.dataset.fullDasharray);
        } else {
            path.removeAttribute('stroke-dasharray');
        }
        if (path.dataset.fullDashoffset) {
            path.setAttribute('stroke-dashoffset', path.dataset.fullDashoffset);
        } else {
            path.removeAttribute('stroke-dashoffset');
        }
    } else {
        path.classList.remove('edge-full');
        if (path.dataset.truncatedDasharray) {
            path.setAttribute('stroke-dasharray', path.dataset.truncatedDasharray);
        }
        if (path.dataset.truncatedDashoffset) {
            path.setAttribute('stroke-dashoffset', path.dataset.truncatedDashoffset);
        } else {
            path.removeAttribute('stroke-dashoffset');
        }
    }
}

function getAncestorGroups(nodeId) {
    return nodeAncestorGroups.get(nodeId) || [];
}

function isNodeInGroupScope(nodeId, gid) {
    if (nodeId == null || gid == null) return false;
    if (nodeId === gid) return true;
    const ancs = getAncestorGroups(nodeId);
    return ancs && ancs.includes(gid);
}

function isIONodeId(nodeId) {
    if (nodeId === null || nodeId === undefined) return false;
    return (DATA.input_node_ids || []).includes(nodeId)
        || (DATA.param_node_ids || []).includes(nodeId)
        || (DATA.const_node_ids || []).includes(nodeId)
        || (DATA.output_node_ids || []).includes(nodeId);
}

function computeActiveNodeIds(activeItems) {
    const activeNodeIds = new Set();
    const activeGroupIds = new Set();
    const ioFocusedNodes = new Set();
    const ioHoverActive = isIONodeId(hoveredNodeId);

    if (hoveredNodeId != null) {
        if (ioHoverActive) {
            ioFocusedNodes.add(hoveredNodeId);
        } else {
            activeNodeIds.add(hoveredNodeId);
        }
        for (const gid of getAncestorGroups(hoveredNodeId)) {
            activeGroupIds.add(gid);
        }
    }

    for (const item of activeItems) {
        const edge = item.edge || {};
        const fromIsIO = isIONodeId(edge.from);
        const toIsIO = isIONodeId(edge.to);
        if (fromIsIO || toIsIO) {
            if (fromIsIO) ioFocusedNodes.add(edge.from);
            if (toIsIO) ioFocusedNodes.add(edge.to);
            const peerNodeId = fromIsIO ? edge.to : edge.from;
            for (const gid of getAncestorGroups(peerNodeId)) {
                activeGroupIds.add(gid);
            }
            continue;
        }
        activeNodeIds.add(edge.from);
        activeNodeIds.add(edge.to);
        for (const gid of getAncestorGroups(edge.from)) {
            activeGroupIds.add(gid);
        }
        for (const gid of getAncestorGroups(edge.to)) {
            activeGroupIds.add(gid);
        }
    }

    for (const nid of ioFocusedNodes) {
        activeNodeIds.add(nid);
    }
    if (hoveredGroupId != null) {
        activeGroupIds.add(hoveredGroupId);
    }
    return new Set([...activeNodeIds, ...activeGroupIds]);
}

function clearEdgeHoverFocusState() {
    for (const item of prevActiveItems) {
        setEdgeItemFocusState(item, 'clear');
    }
    for (const nid of prevActiveNodeIds) {
        setNodeFocusStateById(nid, 'clear');
        setGroupFocusStateById(nid, 'clear');
    }
    prevActiveItems = [];
    prevActiveNodeIds = new Set();
    clearActiveEdgeOverlay();
    const dagContainer = document.getElementById('dag-container');
    if (dagContainer) dagContainer.classList.remove('hover-active');
}

function applyGroupFocusState(hoveredGid) {
    clearEdgeHoverFocusState();
    if (hoveredGid == null) {
        for (const key of prevActiveEdgeKeys) {
            setEdgeItemFocusState(edgeDomByKey.get(key), 'clear');
        }
        for (const key of prevDimEdgeKeys) {
            setEdgeItemFocusState(edgeDomByKey.get(key), 'clear');
        }
        for (const nid of prevActiveGroupNodeIds) {
            setNodeFocusStateById(nid, 'clear');
        }
        for (const nid of prevDimGroupNodeIds) {
            setNodeFocusStateById(nid, 'clear');
        }
        for (const gid of prevActiveGroupIds) {
            setGroupFocusStateById(gid, 'clear');
        }
        for (const gid of prevDimGroupIds) {
            setGroupFocusStateById(gid, 'clear');
        }
        prevActiveEdgeKeys = new Set();
        prevActiveGroupNodeIds = new Set();
        prevActiveGroupIds = new Set();
        prevDimEdgeKeys = new Set();
        prevDimGroupNodeIds = new Set();
        prevDimGroupIds = new Set();
        lastHoveredGid = null;
        const dagContainer = document.getElementById('dag-container');
        if (dagContainer) dagContainer.classList.remove('hover-active');
        return;
    }

    const activeEdgeItems = edgeByGroupId.get(hoveredGid) ?? [];
    const activeEdgeKeys = new Set(activeEdgeItems.map(i => i.key));
    const activeNodeIds = new Set();
    const activeGroupIds = new Set([hoveredGid]);
    for (const item of activeEdgeItems) {
        const f = item.edge.from;
        const t = item.edge.to;
        if (f != null) {
            activeNodeIds.add(f);
            for (const g of getAncestorGroups(f)) activeGroupIds.add(g);
        }
        if (t != null) {
            activeNodeIds.add(t);
            for (const g of getAncestorGroups(t)) activeGroupIds.add(g);
        }
    }

    const combinedActiveNodeIds = new Set([...activeNodeIds, ...activeGroupIds]);
    const nextDimEdgeKeys = new Set();
    const nextDimGroupNodeIds = new Set();
    const nextDimGroupIds = new Set();

    for (const key of prevActiveEdgeKeys) {
        if (!activeEdgeKeys.has(key)) {
            setEdgeItemFocusState(edgeDomByKey.get(key), 'dim');
            nextDimEdgeKeys.add(key);
        }
    }
    for (const key of activeEdgeKeys) {
        if (!prevActiveEdgeKeys.has(key) || prevDimEdgeKeys.has(key)) {
            setEdgeItemFocusState(edgeDomByKey.get(key), 'active');
        }
    }
    for (const key of prevDimEdgeKeys) {
        if (!activeEdgeKeys.has(key)) nextDimEdgeKeys.add(key);
    }

    for (const nid of prevActiveGroupNodeIds) {
        if (!combinedActiveNodeIds.has(nid)) {
            setNodeFocusStateById(nid, 'dim');
            nextDimGroupNodeIds.add(nid);
        }
    }
    for (const nid of combinedActiveNodeIds) {
        if (!prevActiveGroupNodeIds.has(nid) || prevDimGroupNodeIds.has(nid)) {
            setNodeFocusStateById(nid, 'active');
        }
    }
    for (const nid of prevDimGroupNodeIds) {
        if (!combinedActiveNodeIds.has(nid)) nextDimGroupNodeIds.add(nid);
    }

    for (const gid of prevActiveGroupIds) {
        if (!activeGroupIds.has(gid)) {
            setGroupFocusStateById(gid, 'dim');
            nextDimGroupIds.add(gid);
        }
    }
    for (const gid of activeGroupIds) {
        if (!prevActiveGroupIds.has(gid) || prevDimGroupIds.has(gid)) {
            setGroupFocusStateById(gid, 'active');
        }
    }
    for (const gid of prevDimGroupIds) {
        if (!activeGroupIds.has(gid)) nextDimGroupIds.add(gid);
    }

    prevActiveEdgeKeys = activeEdgeKeys;
    prevActiveGroupNodeIds = combinedActiveNodeIds;
    prevActiveGroupIds = activeGroupIds;
    prevDimEdgeKeys = nextDimEdgeKeys;
    prevDimGroupNodeIds = nextDimGroupNodeIds;
    prevDimGroupIds = nextDimGroupIds;
    lastHoveredGid = hoveredGid;
    const dagContainer = document.getElementById('dag-container');
    if (dagContainer) dagContainer.classList.add('hover-active');
}

function applyEdgeFocusState() {
    if (hoveredGroupId != null) {
        applyGroupFocusState(hoveredGroupId);
        clearActiveEdgeOverlay();
        return;
    }
    if (lastHoveredGid != null) {
        applyGroupFocusState(null);
    }

    const newItems = getActiveEdgeItems();
    const activeEdgeKeys = new Set(newItems.map(item => item.key));
    const newNodeIds = computeActiveNodeIds(newItems);
    const prevActiveEdgeKeysLocal = new Set(prevActiveItems.map(item => item.key));
    const hasActive = newItems.length > 0 || hoveredNodeId != null || focusedEdgePath != null || hoveredEdgeKey != null;

    if (!hasActive) {
        clearEdgeHoverFocusState();
        return;
    }

    for (const item of prevActiveItems) {
        if (!activeEdgeKeys.has(item.key)) {
            setEdgeItemFocusState(item, 'clear');
        }
    }
    for (const item of newItems) {
        if (!prevActiveEdgeKeysLocal.has(item.key)) {
            setEdgeItemFocusState(item, 'active');
        }
    }

    for (const id of prevActiveNodeIds) {
        if (!newNodeIds.has(id)) {
            setNodeFocusStateById(id, 'clear');
            setGroupFocusStateById(id, 'clear');
        }
    }
    for (const id of newNodeIds) {
        if (!prevActiveNodeIds.has(id)) {
            setNodeFocusStateById(id, 'active');
            setGroupFocusStateById(id, 'active');
        }
    }

    const dagContainer = document.getElementById('dag-container');
    if (dagContainer) {
        if (hasActive) dagContainer.classList.add('hover-active');
        else dagContainer.classList.remove('hover-active');
    }
    prevActiveItems = newItems;
    prevActiveNodeIds = newNodeIds;
    syncActiveEdgeOverlay(newItems[0] || null);
}

function clearEdgeFocus() {
    focusedEdgePath = null;
    hoveredEdgeKey = null;
    hoveredEdges = [];
    hoveredEdgeIdx = 0;
    hoveredGroupId = null;
    hoveredNodeId = null;
    applyEdgeFocusState();
}

function setEdgeFocus(pathOrKey) {
    const activeKey = typeof pathOrKey === 'string' ? pathOrKey : (edgeDomRegistry.find(x => x.path === pathOrKey)?.key || null);
    focusedEdgePath = activeKey;
    hoveredNodeId = null;
    applyEdgeFocusState();
}

function bindGroupHover(el, gid) {
    if (!el || !gid) return;
    el.addEventListener('mouseenter', (e) => {
        e.stopPropagation();
        if (hoveredGroupId === gid) return;
        hoveredNodeId = null;
        hoveredGroupId = gid;
        showTooltip(e, groupMap[gid]);
        applyEdgeFocusState();
    });
    el.addEventListener('mouseleave', (e) => {
        e.stopPropagation();
        hoveredGroupId = null;
        hideTooltip();
        applyEdgeFocusState();
    });
}

function bindIONodeHover(el, nid) {
    if (!el || !nid) return;
    el.addEventListener('mouseenter', () => {
        hoveredGroupId = null;
        hoveredEdgeKey = null;
        hoveredEdges = [];
        hoveredEdgeIdx = 0;
        hoveredNodeId = nid;
        applyEdgeFocusState();
    });
    el.addEventListener('mouseleave', () => {
        if (hoveredNodeId === nid) {
            hoveredNodeId = null;
            applyEdgeFocusState();
        }
    });
}

function registerNodeDom(nid, el) {
    if (!nid || !el) return;
    if (!nodeDomRegistry.has(nid)) nodeDomRegistry.set(nid, []);
    nodeDomRegistry.get(nid).push(el);
}

function registerGroupDom(gid, key, el) {
    if (!gid || !el) return;
    if (!groupDomRegistry.has(gid)) groupDomRegistry.set(gid, {});
    groupDomRegistry.get(gid)[key] = el;
}

function indexGroupAncestors(groups, ancestors = []) {
    for (const g of (groups || [])) {
        const nextAncestors = ancestors.concat(g.id);
        nodeAncestorGroups.set(g.id, ancestors.slice());
        for (const nid of (g.children_nodes || [])) {
            nodeAncestorGroups.set(nid, nextAncestors.slice());
        }
        for (const port of (g.in_ports || [])) {
            if (port && port.node_id !== null && port.node_id !== undefined) {
                nodeAncestorGroups.set(port.node_id, nextAncestors.slice());
            }
        }
        for (const port of (g.out_ports || [])) {
            if (port && port.node_id !== null && port.node_id !== undefined) {
                nodeAncestorGroups.set(port.node_id, nextAncestors.slice());
            }
        }
        indexGroupAncestors((g.children_group_ids || []).map(id => groupMap[id]).filter(g => g != null), nextAncestors);
    }
}


DATA.groups.forEach(g => groupMap[g.id] = g);
DATA.nodes.forEach(n => nodeMap[n.id] = n);
indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));
(DATA.io_groups || []).forEach(g => {
    (g.member_ids || []).forEach(nid => nodeAncestorGroups.set(nid, [g.id]));
});
// Default: depth > 1, native, and synthetic A/B groups start collapsed
DATA.groups.forEach(g => {
    collapsedState[g.id] = g.depth >= 2 || g.is_native === true
        || g.synthetic_type === 'function_group'
        || g.synthetic_type === 'callloc_group';
});
// Top-level IO groups (Input/Param/Const) default to their adapter-provided collapsed state
(DATA.io_groups || []).forEach(g => { if (!(g.id in collapsedState)) collapsedState[g.id] = g.collapsed; });

// Phase 2 step 4: hand DATA + collapsedState references to render_canvas.js so
// it can run the incremental render path (computeVisibleScene + diffAndPatch)
// without going back through canvasRenderPhase1 / resetScene.  The setter is
// only present once render_canvas.js has loaded; it is invoked once here at
// startup and never again.
if (typeof window !== 'undefined' && typeof window.__canvasSetIncrementalContext === 'function') {
    window.__canvasSetIncrementalContext({ data: DATA, collapsedState: collapsedState, focusStack: focusStack });
}

function formatDur(us) {
    if (us >= 1e6) return (us/1e6).toFixed(3) + ' s';
    if (us >= 1e3) return (us/1e3).toFixed(2) + ' ms';
    return us.toFixed(1) + ' µs';
}

function getNodeColor(n) {
    if (!DATA.has_timing || !n.has_timing) {
        const depthColors = ['#4a6fa5', '#5b8c5a', '#8e6fad', '#c77a3c', '#4aa5a0', '#a05c7b'];
        return depthColors[(n.depth || 0) % depthColors.length];
    }
    if (n.role === 'worker') {
        if (n.pct > 20) return '#e74c3c';
        if (n.pct > 10) return '#e67e22';
        return '#f39c12';
    }
    if (n.pct > 20) return '#2980b9';
    if (n.pct > 10) return '#27ae60';
    if (n.pct > 5) return '#8e44ad';
    return '#5a6c7d';
}

function getGroupBorderColor(g) {
    if (!DATA.has_timing || !g.has_timing) {
        const depthColors = ['rgba(74,111,165,0.5)', 'rgba(91,140,90,0.5)', 'rgba(142,111,173,0.5)', 'rgba(199,122,60,0.5)'];
        return depthColors[(g.depth || 0) % depthColors.length];
    }
    if (g.pct > 20) return 'rgba(41,128,185,0.6)';
    if (g.pct > 10) return 'rgba(39,174,96,0.5)';
    return 'rgba(142,111,173,0.4)';
}

// Layout algorithm: layered (Sugiyama-style) horizontal placement
// Each expanded group lays out its children in horizontal rows ("ranks")
// derived from internal data dependency edges. This spreads sibling modules
// sideways instead of stacking 30+ children in a single tall column.
const LAYOUT = {
    nodeW: 150, nodeH: 36,
    groupPadX: 22, groupPadTop: 36, groupPadBottom: 18,
    // Reserve a "lane gutter" on each side of the group so skip-rank edges
    // can route around intermediate rows without crossing through children.
    laneGutter: 36,
    rowGap: 30,        // vertical gap between consecutive ranks (rows)
    siblingGap: 22,    // horizontal gap between siblings in the same rank
    portR: 4,
    maxRowWidth: 1100  // soft cap; ranks wider than this are wrapped onto multiple rows
};

// Compute longest-path rank for each child of a group based on visible data-flow
// edges. Roots (no incoming edge) get rank 0; rank(v) = max(rank(u)+1)
// over all edges u→v. This produces a layered DAG layout where data flows
// strictly top-to-bottom and siblings of the same rank line up horizontally.
function computeRanks(group) {
    const callOrder = group.call_order || [];
    const ids = callOrder.map(c => c.id);
    const idSet = new Set(ids);
    const inEdges = {};   // id -> [src ids]
    const outEdges = {};  // id -> [dst ids]
    ids.forEach(id => { inEdges[id] = []; outEdges[id] = []; });

    function resolveAncestorInSet(nodeId, candidateSet) {
        if (candidateSet.has(nodeId)) return nodeId;
        const ancestors = nodeAncestorGroups.get(nodeId) || [];
        let result = null;
        for (let i = 0; i < ancestors.length; i++) {
            if (candidateSet.has(ancestors[i])) result = ancestors[i];
        }
        return result;
    }

    const edges = [];
    const edgeKeySet = new Set();
    for (const e of (DATA.edges || [])) {
        const fromId = resolveAncestorInSet(e.from, idSet);
        const toId = resolveAncestorInSet(e.to, idSet);
        if (!fromId || !toId || fromId === toId) {
            continue;
        }
        const key = `${fromId}:${toId}`;
        if (edgeKeySet.has(key)) {
            continue;
        }
        edgeKeySet.add(key);
        edges.push({ from: fromId, to: toId });
    }

    edges.forEach(e => {
        outEdges[e.from].push(e.to);
        inEdges[e.to].push(e.from);
    });

    // Topological-ish longest-path rank assignment with cycle safety.
    const rank = {};
    const callIndex = {};
    callOrder.forEach((c, i) => { callIndex[c.id] = i; });
    // Initial: rank 0 for nodes with no in-edges
    const queue = [];
    ids.forEach(id => {
        if (inEdges[id].length === 0) { rank[id] = 0; queue.push(id); }
    });
    // Iterative relaxation, guarded by an iteration cap to avoid infinite loops
    let safety = ids.length * (edges.length + 4) + 16;
    while (queue.length && safety-- > 0) {
        const u = queue.shift();
        const ru = rank[u] || 0;
        for (const v of outEdges[u]) {
            const rv = (rank[v] === undefined) ? -1 : rank[v];
            if (ru + 1 > rv) {
                rank[v] = ru + 1;
                queue.push(v);
            }
        }
    }
    // Any unreached node (e.g. inside a cycle): place it after its largest
    // predecessor's rank, falling back to call order index.
    for (const id of ids) {
        if (rank[id] === undefined) {
            const preds = inEdges[id].map(p => rank[p]).filter(r => r !== undefined);
            if (preds.length) {
                rank[id] = Math.max(...preds) + 1;
            } else {
                // No edge information at all -> use call index as fallback rank.
                // This degrades gracefully to the old single-column layout.
                rank[id] = callIndex[id] || 0;
            }
        }
    }

    return { rank, edges, callIndex };
}

// Order children within each rank using the median heuristic to reduce
// edge crossings between consecutive ranks.
function orderRanks(rankInfo, childSizes) {
    const { rank, edges, callIndex } = rankInfo;
    const childById = {};
    childSizes.forEach(c => { childById[c.id] = c; });
    // Group children by rank
    const layers = {};
    let maxRank = 0;
    Object.entries(rank).forEach(([id, r]) => {
        if (!layers[r]) layers[r] = [];
        layers[r].push(id);
        if (r > maxRank) maxRank = r;
    });
    // Initial order = call order (stable starting point)
    for (let r = 0; r <= maxRank; r++) {
        if (!layers[r]) { layers[r] = []; continue; }
        layers[r].sort((a, b) => (callIndex[a] || 0) - (callIndex[b] || 0));
    }
    // Two passes of median-heuristic to refine ordering
    const buildAdj = () => {
        const inAdj = {}, outAdj = {};
        edges.forEach(e => {
            (outAdj[e.from] = outAdj[e.from] || []).push(e.to);
            (inAdj[e.to] = inAdj[e.to] || []).push(e.from);
        });
        return { inAdj, outAdj };
    };
    const { inAdj, outAdj } = buildAdj();
    const positionInLayer = (id) => {
        const r = rank[id];
        const layer = layers[r];
        return layer.indexOf(id);
    };
    for (let pass = 0; pass < 2; pass++) {
        // Top-down: order by median of predecessors
        for (let r = 1; r <= maxRank; r++) {
            if (!layers[r] || layers[r].length <= 1) continue;
            layers[r].sort((a, b) => {
                const pa = (inAdj[a] || []).map(positionInLayer).filter(p => p >= 0);
                const pb = (inAdj[b] || []).map(positionInLayer).filter(p => p >= 0);
                const ma = pa.length ? pa.reduce((s,x)=>s+x,0)/pa.length : (callIndex[a] || 0);
                const mb = pb.length ? pb.reduce((s,x)=>s+x,0)/pb.length : (callIndex[b] || 0);
                if (ma === mb) return (callIndex[a] || 0) - (callIndex[b] || 0);
                return ma - mb;
            });
        }
        // Bottom-up: order by median of successors
        for (let r = maxRank - 1; r >= 0; r--) {
            if (!layers[r] || layers[r].length <= 1) continue;
            layers[r].sort((a, b) => {
                const sa = (outAdj[a] || []).map(positionInLayer).filter(p => p >= 0);
                const sb = (outAdj[b] || []).map(positionInLayer).filter(p => p >= 0);
                const ma = sa.length ? sa.reduce((s,x)=>s+x,0)/sa.length : (callIndex[a] || 0);
                const mb = sb.length ? sb.reduce((s,x)=>s+x,0)/sb.length : (callIndex[b] || 0);
                if (ma === mb) return (callIndex[a] || 0) - (callIndex[b] || 0);
                return ma - mb;
            });
        }
    }
    return layers;
}

function layoutGroup(gid, containerWidth) {
    if (!containerWidth || containerWidth <= 0) {
        throw new Error('layoutGroup: containerWidth must be a positive number');
    }
    const effectiveMaxRowWidth = Math.max(containerWidth - 100, 400);
    const g = groupMap[gid];
    if (!g) return { w: LAYOUT.nodeW, h: LAYOUT.nodeH };

    if (collapsedState[gid]) {
        // Collapsed: render as a single node-like box
        const w = LAYOUT.nodeW + 20;
        const h = LAYOUT.nodeH + 8;
        groupLayout[gid] = { w, h, collapsed: true };
        return { w, h };
    }

    // Expanded: lay out children using layered DAG placement
    const callOrder = g.call_order || [];
    if (callOrder.length === 0) {
        const w = LAYOUT.nodeW + 40;
        const h = LAYOUT.nodeH + LAYOUT.groupPadTop + LAYOUT.groupPadBottom;
        groupLayout[gid] = { w, h, collapsed: false, childPositions: [] };
        return { w, h };
    }

    // 1. Recursively layout each child to obtain its bounding box
    const childSizes = [];
    for (const item of callOrder) {
        if (item.type === 'node') {
            const node = nodeMap[item.id];
            const h = node && node.has_timing ? 36 : 28;
            childSizes.push({ id: item.id, type: 'node', w: LAYOUT.nodeW, h });
        } else {
            const sz = layoutGroup(item.id, containerWidth);
            childSizes.push({ id: item.id, type: 'group', w: sz.w, h: sz.h });
        }
    }
    const childById = {};
    childSizes.forEach(c => { childById[c.id] = c; });

    // 2. Compute rank (vertical layer) for each child via longest-path on internal edges
    const rankInfo = computeRanks(g);
    const layers = orderRanks(rankInfo, childSizes);
    const maxRank = Math.max(0, ...Object.keys(layers).map(Number));

    // 3. For each rank, lay children left-to-right; track row widths and heights.
    //    If a single rank has too many children to fit horizontally, wrap it
    //    onto multiple physical rows (a "wrapped layer") so we don't overflow
    //    the parent container.
    const rowLayouts = []; // each: y, h, rows (each row: items, totalW)
    let cy = LAYOUT.groupPadTop;
    let maxRowW = 0;
    const maxW = effectiveMaxRowWidth;
    for (let r = 0; r <= maxRank; r++) {
        const layerIds = layers[r] || [];
        if (layerIds.length === 0) continue;
        const sizes = layerIds.map(id => childById[id]).filter(Boolean);

        // Greedy wrap: pack items into as few rows as possible while staying
        // under maxW. Always at least one item per row even if it's wider.
        const wrapped = [];  // each: items, totalW, h
        let curRow = []; let curW = 0;
        for (const c of sizes) {
            const tentative = curW + (curRow.length ? LAYOUT.siblingGap : 0) + c.w;
            if (curRow.length && tentative > maxW) {
                wrapped.push({ items: curRow, totalW: curW, h: Math.max(...curRow.map(x => x.h)) });
                curRow = [c]; curW = c.w;
            } else {
                curRow.push(c); curW = tentative;
            }
        }
        if (curRow.length) wrapped.push({ items: curRow, totalW: curW, h: Math.max(...curRow.map(x => x.h)) });

        const layerStartY = cy;
        for (const wr of wrapped) {
            rowLayouts.push({ y: cy, h: wr.h, items: wr.items, totalW: wr.totalW, rank: r });
            if (wr.totalW > maxRowW) maxRowW = wr.totalW;
            cy += wr.h + LAYOUT.rowGap;
        }
    }
    cy -= LAYOUT.rowGap;  // remove trailing gap

    // Detect skip-rank edges in this group; if any exist, reserve gutter
    // space on both sides so the rank-aware edge router has clear routing
    // lanes without crossing children.
    let hasSkipEdges = false;
    {
        const rankOf = rankInfo.rank;
        for (const ed of rankInfo.edges) {
            const ra = rankOf[ed.from], rb = rankOf[ed.to];
            if (ra !== undefined && rb !== undefined && Math.abs(rb - ra) > 1) {
                hasSkipEdges = true;
                break;
            }
        }
    }
    const gutter = hasSkipEdges ? LAYOUT.laneGutter : 0;

    const groupW = Math.max(maxRowW + (LAYOUT.groupPadX + gutter) * 2, LAYOUT.nodeW + 60);
    const groupH = cy + LAYOUT.groupPadBottom;

    // 4. Place each row centered horizontally inside the group
    const childPositions = [];
    for (const row of rowLayouts) {
        let x = (groupW - row.totalW) / 2;
        // Vertically center each item within the row band (rows can have mixed heights)
        for (const it of row.items) {
            const y = row.y + (row.h - it.h) / 2;
            childPositions.push({ id: it.id, type: it.type, x, y, w: it.w, h: it.h, rank: row.rank });
            x += it.w + LAYOUT.siblingGap;
        }
    }

    groupLayout[gid] = {
        w: groupW, h: groupH, collapsed: false, childPositions,
        rowLayouts, rankInfo
    };
    return { w: groupW, h: groupH };
}

const IO_W = 140;
const IO_H = 40;
const IO_GAP = 36;
const IO_PILL_GAP = 18;
const EXPAND_PADDING = 24;
const EXPAND_PILL_W = 100;
const EXPAND_COLS = 3;
const EXPAND_FRAME_PAD = 16;
const EXPAND_COL_GAP = 20;
const EXPAND_OUTER_PAD = 24;
const EXPANDED_IO_H = 28;
const EXPANDED_IO_GAP = 16;
const COLLAPSE_BUTTON_W = 70;
const COLLAPSE_BUTTON_H = 22;
const COLLAPSE_BOTTOM_PADDING = 10;
let ACTIVE_FLOWCHART_LAYOUT_CONTEXT = null;

function getIOLayoutConfig() {
    return {
        ioW: IO_W,
        ioH: IO_H,
        ioGap: IO_GAP,
        pillGap: IO_PILL_GAP,
        EXPAND_PADDING,
        EXPAND_PILL_W,
        EXPAND_COLS,
        EXPAND_FRAME_PAD,
        EXPAND_COL_GAP,
        EXPAND_OUTER_PAD,
        EXPANDED_IO_H,
        EXPANDED_IO_GAP,
        COLLAPSE_BUTTON_W,
        COLLAPSE_BUTTON_H,
        COLLAPSE_BOTTOM_PADDING,
    };
}

function computeIOGroupExpandedLayout(memberCount, availableSvgW) {
    const availableWInput = Number(availableSvgW);
    if (!Number.isFinite(availableWInput)) {
        throw new Error(`computeIOGroupExpandedLayout got invalid availableSvgW: ${availableSvgW}`);
    }
    if (memberCount === 0) return { cols: EXPAND_COLS, pillW: EXPAND_PILL_W, memberRows: 0, height: 0 };
    const availableW = availableWInput - 2 * EXPAND_FRAME_PAD;
    const cols = Math.max(EXPAND_COLS, Math.floor(availableW / (EXPAND_PILL_W + IO_PILL_GAP)));
    const pillW = Math.floor((availableW - (cols - 1) * IO_PILL_GAP) / cols);
    const memberRows = Math.ceil(memberCount / cols);
    const memberAreaH = memberRows * EXPANDED_IO_H + (memberRows - 1) * EXPANDED_IO_GAP;
    const height = memberAreaH + EXPANDED_IO_GAP + COLLAPSE_BUTTON_H + COLLAPSE_BOTTOM_PADDING;
    return { cols, pillW, memberRows, height };
}

function calcIOGroupExpandedHeight(ioGroup, availableW) {
    const width = Number(availableW);
    if (!Number.isFinite(width)) {
        throw new Error(`calcIOGroupExpandedHeight got invalid availableW: ${availableW}`);
    }
    const memberCount = (ioGroup.member_ids || []).length;
    return computeIOGroupExpandedLayout(memberCount, width).height;
}

function getIOGroupCollapsedState(ioGroup) {
    return (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
}

function calcIORowWidth(row, availableW) {
    const width = Number(availableW);
    if (!Number.isFinite(width)) {
        throw new Error(`calcIORowWidth got invalid availableW: ${availableW}`);
    }
    if (!row.items || row.items.length === 0) return 0;
    let result = 0;
    let pendingInline = 0;
    const flushInline = () => {
        if (pendingInline > 0) {
            result = Math.max(result, pendingInline * IO_W + (pendingInline - 1) * IO_PILL_GAP);
            pendingInline = 0;
        }
    };
    for (const item of row.items) {
        if (item.isIOGroup === true) {
            const ioGroup = item.ioGroup;
            if (!getIOGroupCollapsedState(ioGroup)) {
                flushInline();
                const memberCount = (ioGroup.member_ids || []).length;
                const expandedW = memberCount > 0
                    ? Math.min(width * 0.85, memberCount * (IO_W + IO_PILL_GAP) - IO_PILL_GAP)
                    : 0;
                result = Math.max(result, expandedW);
                continue;
            }
        }
        pendingInline += 1;
    }
    flushInline();
    return result;
}

function calcIORowHeight(row, svgW) {
    const width = Number(svgW);
    if (!Number.isFinite(width)) {
        throw new Error(`calcIORowHeight got invalid svgW: ${svgW}`);
    }
    if (!row.items || row.items.length === 0) return 0;
    const expandedGroups = row.items.filter(item => item.isIOGroup === true && !getIOGroupCollapsedState(item.ioGroup));
    const hasInlineRow = row.items.some(item => item.isIOGroup !== true || getIOGroupCollapsedState(item.ioGroup));
    let height = hasInlineRow ? IO_H : 0;
    if (expandedGroups.length > 0) {
        const groupCount = expandedGroups.length;
        const colW = (width - (groupCount - 1) * EXPAND_COL_GAP) / groupCount;
        const expandedH = Math.max(...expandedGroups.map(item => calcIOGroupExpandedHeight(item.ioGroup, colW)));
        height += (height > 0 ? IO_GAP : 0) + expandedH;
    }
    return height;
}

function collectIORenderTasks(row, startY) {
    if (!ACTIVE_FLOWCHART_LAYOUT_CONTEXT) {
        throw new Error('collectIORenderTasks called before computeFlowchartLayout initialized layout context');
    }
    const { svgW } = ACTIVE_FLOWCHART_LAYOUT_CONTEXT;
    if (!row || !row.items || row.items.length === 0) return { tasks: [], height: 0 };
    let y = startY;
    const tasks = [];
    const inlineItems = [];
    const expandedItems = [];
    for (const item of row.items) {
        if (item.isIOGroup === true) {
            const ioGroup = item.ioGroup;
            if (!getIOGroupCollapsedState(ioGroup)) {
                expandedItems.push(item);
                continue;
            }
        }
        inlineItems.push(item);
    }

    if (inlineItems.length > 0) {
        const rowWidth = inlineItems.length * IO_W + (inlineItems.length - 1) * IO_PILL_GAP;
        let left = (svgW - rowWidth) / 2;
        const cy = y + IO_H / 2;
        for (const item of inlineItems) {
            if (item.isIOGroup === true) {
                tasks.push({
                    type: 'io',
                    taskKind: 'io_group',
                    ioGroup: { ...item.ioGroup, _collapsed: true },
                    cx: left + IO_W / 2,
                    cy,
                    w: IO_W,
                    h: IO_H,
                    availableW: svgW,
                });
            } else {
                const nid = item.id;
                const node = nodeMap[nid];
                const baseText = node ? node.class_name : item.defaultSublabel;
                const sublabel = (node && node.has_timing)
                    ? `${baseText} · ${node.pct.toFixed(1)}%`
                    : baseText;
                tasks.push({
                    type: 'io',
                    taskKind: 'io_pill',
                    nid,
                    subtype: item.subtype,
                    cx: left + IO_W / 2,
                    cy,
                    w: IO_W,
                    h: IO_H,
                    label: item.label,
                    sublabel,
                    fillColor: item.fillColor,
                });
            }
            left += IO_W + IO_PILL_GAP;
        }
        y += IO_H;
    }

    if (expandedItems.length > 0) {
        if (y > startY) y += IO_GAP;
        const groupCount = expandedItems.length;
        const colW = (svgW - (groupCount - 1) * EXPAND_COL_GAP) / groupCount;
        const expandedHeight = Math.max(...expandedItems.map(item => calcIOGroupExpandedHeight(item.ioGroup, colW)));
        let left = 0;
        for (const item of expandedItems) {
            tasks.push({
                type: 'io',
                taskKind: 'io_group',
                ioGroup: { ...item.ioGroup, _collapsed: false },
                cx: left + colW / 2,
                cy: y + EXPANDED_IO_H / 2,
                w: IO_W,
                h: IO_H,
                availableW: colW,
            });
            left += colW + EXPAND_COL_GAP;
        }
        y += expandedHeight;
    }
    return { tasks, height: y - startY };
}

function computeFlowchartLayout(data, containerWidth) {
    if (!containerWidth || containerWidth <= 0) {
        throw new Error('computeFlowchartLayout: containerWidth must be a positive number');
    }
    const rootIds = Array.isArray(data.root_groups) ? data.root_groups : [];
    const rootSizes = rootIds.map(rid => {
        const sz = layoutGroup(rid, containerWidth);
        return { id: rid, ...sz };
    });
    const topIOItems = (data.io_groups && data.io_groups.length > 0)
        ? data.io_groups
            .filter(g => g.io_subtype !== 'output')
            .map(g => ({ isIOGroup: true, ioGroup: g }))
        : [
            ...(data.input_node_ids || []).map(id => ({ id, subtype: 'input', label: 'Input', defaultSublabel: 'network input', fillColor: 'rgba(46,204,113,0.55)' })),
            ...(data.param_node_ids || []).map(id => ({ id, subtype: 'param', label: 'Param', defaultSublabel: 'model param', fillColor: 'rgba(155,89,182,0.55)' })),
            ...(data.const_node_ids || []).map(id => ({ id, subtype: 'const', label: 'Const', defaultSublabel: 'const value', fillColor: 'rgba(241,196,15,0.55)' })),
        ];
    const outputIOGroup = (data.io_groups || []).find(g => g.io_subtype === 'output');
    const bottomIOItems = outputIOGroup
        ? [{ isIOGroup: true, ioGroup: outputIOGroup }]
        : (data.output_node_ids || []).map(id => ({
              id,
              subtype: 'output',
              label: 'Result',
              defaultSublabel: 'result output',
              fillColor: 'rgba(231,76,60,0.55)'
          }));
    const topIORows = topIOItems.length > 0 ? [{ items: topIOItems }] : [];
    const bottomIORows = bottomIOItems.length > 0 ? [{ items: bottomIOItems }] : [];
    const allIORows = topIORows.concat(bottomIORows);
    const maxRootW = rootSizes.length ? Math.max(...rootSizes.map(r => r.w)) : LAYOUT.nodeW;
    const maxRootH = rootSizes.length ? Math.max(...rootSizes.map(r => r.h)) : LAYOUT.nodeH;
    const provisionalSvgW = Math.max(maxRootW + 80, 480);
    const expandedGroupCount = (data.io_groups || []).filter(g => !getIOGroupCollapsedState(g)).length;
    const minIOColW = EXPAND_COLS * EXPAND_PILL_W + (EXPAND_COLS - 1) * IO_PILL_GAP + 2 * EXPAND_FRAME_PAD;
    const minIOTotalW = expandedGroupCount > 0
        ? expandedGroupCount * minIOColW + (expandedGroupCount - 1) * EXPAND_COL_GAP + 2 * EXPAND_OUTER_PAD
        : 0;
    const maxIORowW = allIORows.length > 0 ? Math.max(...allIORows.map(row => calcIORowWidth(row, provisionalSvgW))) : 0;
    const svgW = Math.max(provisionalSvgW, maxIORowW + 80, minIOTotalW);
    const topIOHeight = topIORows.length > 0 ? topIORows.reduce((acc, row, idx) => acc + calcIORowHeight(row, svgW) + (idx > 0 ? IO_GAP : 0), 0) : 0;
    const bottomIOHeight = bottomIORows.length > 0 ? bottomIORows.reduce((acc, row, idx) => acc + calcIORowHeight(row, svgW) + (idx > 0 ? IO_GAP : 0), 0) : 0;
    const rootStartY = 30 + (topIOHeight > 0 ? topIOHeight + IO_GAP : 0);
    const svgH = 30 + topIOHeight + (topIOHeight > 0 ? IO_GAP : 0) + maxRootH + (bottomIOHeight > 0 ? IO_GAP + bottomIOHeight : 0) + 40;
    const rootPositions = [];
    if (rootSizes.length > 0) {
        const rs = rootSizes[0];
        rootPositions.push({ id: rs.id, x: (svgW - rs.w) / 2, y: rootStartY, w: rs.w, h: rs.h });
    }
    ACTIVE_FLOWCHART_LAYOUT_CONTEXT = {
        data,
        svgW,
        svgH,
        provisionalSvgW,
        rootStartY,
        topIORows,
        bottomIORows,
    };
    const ioTasks = [];
    let topIOY = 30;
    for (const row of topIORows) {
        const plan = collectIORenderTasks(row, topIOY);
        ioTasks.push(...plan.tasks);
        topIOY += plan.height + IO_GAP;
    }
    const rootEntry = rootPositions[0];
    let bottomIOY = rootEntry ? (rootEntry.y + rootEntry.h + IO_GAP) : (rootStartY + maxRootH + IO_GAP);
    for (const row of bottomIORows) {
        const plan = collectIORenderTasks(row, bottomIOY);
        ioTasks.push(...plan.tasks);
        bottomIOY += plan.height + IO_GAP;
    }
    return {
        svgW,
        svgH,
        provisionalSvgW,
        rootStartY,
        rootSizes,
        rootPositions,
        rootEntry,
        ioTasks,
        topIORows,
        bottomIORows,
        topIOHeight,
        bottomIOHeight,
    };
}

function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function getRenderProgressElements() {
    const overlay = document.getElementById('render-progress-overlay');
    const bar = document.getElementById('render-progress-bar');
    const stage = document.getElementById('render-progress-stage');
    const percent = document.getElementById('render-progress-percent');
    if (!overlay || !bar || !stage || !percent) {
        throw new Error('render progress overlay DOM is incomplete');
    }
    return { overlay, bar, stage, percent };
}

function setRenderProgress(percent, stageText) {
    const { overlay, bar, stage, percent: percentEl } = getRenderProgressElements();
    const normalized = Math.max(0, Math.min(100, Number(percent)));
    if (!Number.isFinite(normalized)) {
        throw new Error(`setRenderProgress got invalid percent: ${percent}`);
    }
    overlay.dataset.progress = normalized.toFixed(1);
    bar.style.width = `${normalized}%`;
    stage.textContent = stageText;
    percentEl.textContent = `${Math.round(normalized)}%`;
}

function showRenderProgress(stageText) {
    const { overlay } = getRenderProgressElements();
    overlay.classList.remove('closing', 'failed');
    overlay.classList.add('visible');
    overlay.setAttribute('aria-hidden', 'false');
    setRenderProgress(0, stageText);
}

async function hideRenderProgress() {
    const { overlay } = getRenderProgressElements();
    const ownerGeneration = overlay.dataset.renderGeneration || '';
    if (ownerGeneration !== String(renderGeneration)) {
        overlay.classList.add('closing');
        overlay.classList.remove('visible');
        overlay.setAttribute('aria-hidden', 'true');
        return;
    }
    setRenderProgress(100, '渲染完成');
    await nextFrame();
    if (ownerGeneration !== String(renderGeneration)) {
        overlay.classList.add('closing');
        overlay.classList.remove('visible');
        overlay.setAttribute('aria-hidden', 'true');
        return;
    }
    overlay.classList.add('closing');
    overlay.classList.remove('visible');
    await new Promise((resolve) => setTimeout(resolve, 180));
    if (ownerGeneration !== String(renderGeneration)) {
        overlay.setAttribute('aria-hidden', 'true');
        return;
    }
    overlay.setAttribute('aria-hidden', 'true');
}

function assertActiveRenderGeneration(generation, stageText) {
    if (typeof generation !== 'number' || !Number.isFinite(generation)) {
        throw new Error(`invalid render generation ${generation} during ${stageText}`);
    }
    if (typeof renderGeneration === 'undefined') {
        throw new Error('renderGeneration is unavailable');
    }
    if (generation !== renderGeneration) {
        return false;
    }
    return true;
}
if (typeof window !== 'undefined') {
    window.__canvasAssertGeneration = assertActiveRenderGeneration;
}

async function runChunked(items, handler, options) {
    const opts = options || {};
    const batchSize = Math.max(1, Number(opts.batchSize || 1));
    const phaseStart = Number(opts.phaseStart || 0);
    const phaseEnd = Number(opts.phaseEnd || phaseStart);
    const stageText = opts.stageText || '正在渲染…';
    const generation = Number(opts.generation);
    const allowedTypes = Array.isArray(opts.allowedTypes) && opts.allowedTypes.length > 0
        ? opts.allowedTypes
        : ['group', 'node', 'io', 'edge'];
    const allowedTypeSet = new Set(allowedTypes);
    if (!assertActiveRenderGeneration(generation, stageText)) {
        return false;
    }
    const total = items.length;
    if (total === 0) {
        setRenderProgress(phaseEnd, stageText);
        await nextFrame();
        return assertActiveRenderGeneration(generation, stageText);
    }
    let processed = 0;
    while (processed < total) {
        if (!assertActiveRenderGeneration(generation, stageText)) {
            return false;
        }
        const upperBound = Math.min(processed + batchSize, total);
        for (let idx = processed; idx < upperBound; idx++) {
            const item = items[idx];
            if (!item || typeof item.type !== 'string' || !allowedTypeSet.has(item.type)) {
                const actualType = item && typeof item.type === 'string' ? item.type : '<missing>';
                throw new Error(`runChunked got unknown task type: ${actualType}`);
            }
            if (!assertActiveRenderGeneration(generation, stageText)) {
                return false;
            }
            await handler(item, idx);
        }
        processed = upperBound;
        const progress = phaseStart + (processed / total) * (phaseEnd - phaseStart);
        setRenderProgress(progress, stageText);
        await nextFrame();
    }
    return assertActiveRenderGeneration(generation, stageText);
}

function invokeRender(renderOpts) {
    const promise = render(renderOpts);
    promise.catch((err) => {
        setTimeout(() => { throw err; }, 0);
    });
    return promise;
}

async function render(renderOpts) {
    if (typeof window !== 'undefined' && typeof window.__canvasRenderPhase1 === 'function') {
        return window.__canvasRenderPhase1(DATA, renderOpts);
    }
    throw new Error('Canvas render entry __canvasRenderPhase1 is missing; engine bundle or render_canvas.js not loaded');
}

// Phase 2 step 4: ``invokeIncrementalRender`` is the post-toggle render entry.
// Step 4 routes it through ``__canvasInvokeIncrementalRender`` which captures
// the current ``engine.visible*`` sets, computes the next visible scene from
// ``DATA + collapsedState``, and drives the canvas bundle's incremental
// diff/auto-fit path.  This path MUST NOT touch ``resetScene()`` or the legacy
// full ``invokeRender`` pipeline.
function invokeIncrementalRender(ctx) {
    if (typeof window === 'undefined' || typeof window.__canvasInvokeIncrementalRender !== 'function') {
        throw new Error('Canvas incremental render entry __canvasInvokeIncrementalRender is missing; render_canvas.js must be loaded before the inline runtime');
    }
    window.__canvasInvokeIncrementalRender(ctx);
}

// Phase 2 step 4: ``expandAll`` / ``collapseAll`` mutate ``collapsedState`` in
// place and dispatch the incremental render path.  They never touch the
// ``invokeRender`` (full pipeline) path, which is reserved for the very first
// page load and explicit user-driven re-fits.
function expandAll() {
    DATA.groups.forEach(g => collapsedState[g.id] = false);
    (DATA.io_groups || []).forEach(g => { collapsedState[g.id] = false; });
    invokeIncrementalRender({ reason: 'expand-all' });
}

function collapseAll() {
    DATA.groups.forEach(g => { if (g.depth >= 1) collapsedState[g.id] = true; });
    (DATA.io_groups || []).forEach(g => { collapsedState[g.id] = true; });
    invokeIncrementalRender({ reason: 'collapse-all' });
}

// Phase 2 step 3: click / dblclick handlers on group hit boxes forward to
// these inline-runtime globals (render_canvas.js' engine.onGroupToggle /
// onGroupSelect read them at call time).  Missing groupMap[gid] is a hard
// error — no silent fallback.
if (typeof window !== 'undefined') {
    window.__canvasOnGroupToggle = function (gid) {
        if (!groupMap[gid]) {
            throw new Error('__canvasOnGroupToggle: unknown group id ' + gid);
        }
        collapsedState[gid] = !collapsedState[gid];
        invokeIncrementalRender({ reason: 'toggle', gid: gid });
    };
    window.__canvasOnGroupSelect = function (gid) {
        const g = groupMap[gid];
        if (!g) {
            throw new Error('__canvasOnGroupSelect: unknown group id ' + gid);
        }
        const engine = (typeof window.__canvasEnginePhase1 === 'function') ? window.__canvasEnginePhase1() : null;
        if (engine) { engine.selectedGroupId = gid; }
        showGroupPanel(g);
    };
}

// Phase 2 step 5 — Semantic Zoom helpers.
//
// ``cascadeExpandSubtree(gid)`` flips ``collapsedState[descendant] = false``
// for ``gid`` and every group reachable through ``children_group_ids``.  This
// is invoked the moment we enter focus so that ``computeVisibleScene`` (in
// render_canvas.js) sees a fully-expanded subtree under the focus root and
// emits the entire local semantic closure on the very first frame.  The
// recursion ignores ``DATA.io_groups`` (those are sibling rows, not children
// of any group), and silently skips unknown ids — they are emitted by the
// upstream backend and a missing one is a backend bug, not a frontend
// concern, so we surface it as a hard error.
function cascadeExpandSubtree(gid) {
    const g = groupMap[gid];
    if (!g) {
        throw new Error('cascadeExpandSubtree: unknown group id ' + gid);
    }
    collapsedState[gid] = false;
    const queue = (g.children_group_ids || []).slice();
    while (queue.length > 0) {
        const cgid = queue.shift();
        const cg = groupMap[cgid];
        if (!cg) {
            throw new Error('cascadeExpandSubtree: child group ' + cgid + ' missing from groupMap');
        }
        collapsedState[cgid] = false;
        (cg.children_group_ids || []).forEach(nextId => queue.push(nextId));
    }
}

// ``buildBreadcrumbLabel(gid)`` returns a short, human-readable label for the
// breadcrumb segment associated with ``gid``.  Falls back to the raw id only
// when neither ``label`` nor ``name`` is populated on the group record.
function buildBreadcrumbLabel(gid) {
    const g = groupMap[gid];
    if (!g) {
        throw new Error('buildBreadcrumbLabel: unknown group id ' + gid);
    }
    return g.label || g.name || g.id;
}

// ``initTopbarRunstep()`` labels the topbar runstep capsule from the model
// root(s).  Single-runstep traces keep the ▾ disabled (CSS only); the label is
// purely informational and never drives any trace processing.
function initTopbarRunstep() {
    if (typeof document === 'undefined') { return; }
    const cap = document.getElementById('tb-runstep-capsule');
    if (!cap) { return; }
    const meta = (typeof DATA !== 'undefined' && DATA && DATA.meta) ? DATA.meta : {};
    if (meta.roots && meta.roots.length > 0) {
        cap.textContent = meta.roots.join(' / ');
    }
}

// ``renderBreadcrumb()`` rebuilds the Semantic-Zoom focus path inside the
// unified topbar nav (#dag-breadcrumb-nav).  The static Training / runstep
// capsules are authored in HTML and never touched here — only the dynamic
// focus-path segments (one ``›`` separator + one label per focus level) are
// managed.  The deepest level renders as the current (white, non-clickable)
// segment; shallower levels are clickable to jump back up via
// ``setFocusToDepth()``.  Returning to the full graph is done via ESC or a
// right double-click on the focus root (Semantic Zoom protocol), so there is
// no '全图' home affordance here anymore.
let __focusBreadcrumbEls = [];
function renderBreadcrumb() {
    if (typeof document === 'undefined') { return; }
    const host = document.getElementById('dag-breadcrumb-nav');
    if (!host) {
        throw new Error('renderBreadcrumb: #dag-breadcrumb-nav element missing');
    }
    // Remove the focus-path segments appended by the previous call, leaving the
    // static Training / runstep capsules in place.
    for (let i = 0; i < __focusBreadcrumbEls.length; i++) {
        const el = __focusBreadcrumbEls[i];
        if (el && el.parentNode === host && typeof host.removeChild === 'function') {
            host.removeChild(el);
        }
    }
    __focusBreadcrumbEls = [];
    for (let i = 0; i < focusStack.length; i++) {
        const sep = document.createElement('span');
        sep.className = 'focus-breadcrumb-sep';
        sep.textContent = '›';
        host.appendChild(sep);
        __focusBreadcrumbEls.push(sep);
        const seg = document.createElement('span');
        seg.className = 'focus-breadcrumb-seg';
        seg.textContent = buildBreadcrumbLabel(focusStack[i]);
        seg.dataset.depth = String(i + 1);
        if (i < focusStack.length - 1) {
            seg.classList.add('focus-breadcrumb-clickable');
            const targetDepth = i + 1;
            seg.addEventListener('click', () => setFocusToDepth(targetDepth));
        } else {
            seg.classList.add('focus-breadcrumb-current');
        }
        host.appendChild(seg);
        __focusBreadcrumbEls.push(seg);
    }
}

// ``enterFocus(gid)`` pushes ``gid`` onto ``focusStack``, cascade-expands the
// focus subtree, refreshes the breadcrumb, and triggers the incremental
// render.  Re-entering the SAME current focus root is a no-op.
function enterFocus(gid) {
    if (gid === undefined || gid === null) {
        throw new Error('enterFocus: gid is required');
    }
    const g = groupMap[gid];
    if (!g) {
        throw new Error('enterFocus: unknown group id ' + gid);
    }
    if (focusStack.length > 0 && String(focusStack[focusStack.length - 1]) === String(gid)) {
        return;
    }
    // Problem 2: a *leaf* group (no nested child groups) has nothing to drill
    // into.  Pushing it onto the focus stack used to relayout its single card to
    // fill the viewport — the reported vertical-stretch artefact — and, because
    // the stretched card ends up fully covered by its own child nodes, left no
    // group box for the user to right-double-click, so focus could not be
    // exited.  Leaf groups therefore only surface their side panel (acting as a
    // plain selection); they never push the focus stack or cascade-expand.
    if (((g.children_group_ids || []).length) === 0) {
        showGroupPanel(g);
        return;
    }
    cascadeExpandSubtree(gid);
    focusStack.push(String(gid));
    renderBreadcrumb();
    invokeIncrementalRender({ reason: 'focus-enter', gid: gid });
}

// ``exitFocus()`` pops one level off ``focusStack``.  Empty stack → no-op
// (keeps the keyboard handler trivially idempotent).  Cascade-expansion is
// NOT reverted — the user accepts the persistent expansion (NaN explicit,
// 2026-06-25).
function exitFocus() {
    if (focusStack.length === 0) { return; }
    focusStack.pop();
    renderBreadcrumb();
    invokeIncrementalRender({ reason: 'focus-exit' });
}

// ``setFocusToDepth(depth)`` truncates ``focusStack`` to ``depth`` entries
// (depth=0 → full graph).  Used by breadcrumb segment clicks.  Calling with
// the current depth is a no-op so successive clicks on the active segment do
// not trigger a redundant re-render.
function setFocusToDepth(depth) {
    if (typeof depth !== 'number' || depth < 0 || depth > focusStack.length) {
        throw new Error('setFocusToDepth: depth out of range: ' + depth + ' (stack=' + focusStack.length + ')');
    }
    if (depth === focusStack.length) { return; }
    focusStack.length = depth;
    renderBreadcrumb();
    invokeIncrementalRender({ reason: 'focus-jump', depth: depth });
}

// ``__focusNowMs()`` is a monotonic-ish millisecond clock used by the right
// double-click gestures (group-box drill-down and the stage-level focus exit).
// Falls back to 0 in the degenerate environment where Date is unavailable so
// the gesture math stays defined.
function __focusNowMs() {
    return (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
}
// Timestamp of the most recent group-box focus gesture (enter/exit).  The
// stage-level right-double-click guard consults it to avoid double-handling a
// gesture the group box already consumed.
let __lastGroupFocusGestureAt = 0;

if (typeof window !== 'undefined') {
    // Engine callbacks for right-double-click on a group box.  Pure
    // forwarders so render_canvas.js does not need to know about
    // ``focusStack`` directly.  Each gesture also stamps
    // ``__lastGroupFocusGestureAt`` so the stage-level right-double-click guard
    // (bindCanvasContextMenuGuard) can tell "the group box already handled this
    // gesture" apart from "right-double-click on empty / covered canvas".
    window.__canvasOnEnterFocus = function (gid) { __lastGroupFocusGestureAt = __focusNowMs(); enterFocus(gid); };
    window.__canvasOnExitFocus = function () { __lastGroupFocusGestureAt = __focusNowMs(); exitFocus(); };
    // Convenience hook for tests + breadcrumb integration tests.
    window.__canvasSetFocusToDepth = function (depth) { setFocusToDepth(depth); };
}

function showGroupPanel(g) {
    const sp = document.getElementById('side-panel');
    const body = document.getElementById('sp-body');
    const titleEl = document.getElementById('sp-title');
    const subtitleEl = document.getElementById('sp-subtitle');
    if (!sp || !body || !titleEl || !subtitleEl) {
        throw new Error('showGroupPanel: side-panel DOM is missing');
    }
    titleEl.textContent = g.class_name || g.label || ('group ' + g.id);
    subtitleEl.textContent = (g.attr_name || '') + ' \u00B7 gid=' + g.id;

    const collapsed = !!collapsedState[g.id];
    const childrenCount = ((g.children_group_ids || []).length) + ((g.children_nodes || []).length);
    const kernelMs = Number(g && g.kernel_us != null ? g.kernel_us : (g.dur_us || 0)) / 1000.0;
    const fwdMs = Number(g && g.fwd_kernel_us || 0) / 1000.0;
    const bwdMs = Number(g && g.bwd_kernel_us || 0) / 1000.0;
    const otherMs = Number(g && g.other_kernel_us || 0) / 1000.0;

    let bodyHtml = '<div class="side-panel-section"><h4>Group</h4>'
        + `<div class="evidence-meta"><b>gid:</b> ${escapeHtml(String(g.id))}</div>`
        + `<div class="evidence-meta"><b>label:</b> ${escapeHtml(g.label || '')}</div>`
        + `<div class="evidence-meta"><b>type:</b> ${escapeHtml(String(g.node_type || g.synthetic_type || 'module'))}</div>`
        + `<div class="evidence-meta"><b>depth:</b> ${escapeHtml(String(g.depth))}</div>`
        + `<div class="evidence-meta"><b>collapsed:</b> ${collapsed ? 'true' : 'false'}</div>`
        + `<div class="evidence-meta"><b>children:</b> ${escapeHtml(String(childrenCount))}</div>`
        + '</div>';

    if (g.has_timing || g.has_phase_timing) {
        bodyHtml += '<div class="side-panel-section"><h4>Timing</h4>'
            + renderTimingRow('Kernel', kernelMs, 'kernel')
            + renderTimingRow('Forward', fwdMs, 'forward')
            + renderTimingRow('Backward', bwdMs, 'backward')
            + renderTimingRow('Other', otherMs, 'other')
            + '</div>';
    }

    if (g.src_file) {
        const startLine = g.src_start_line || '';
        const endLine = g.src_end_line || '';
        bodyHtml += '<div class="side-panel-section"><h4>Source location</h4>'
            + `<div class="evidence-meta"><b>file:</b> ${escapeHtml(String(g.src_file))}</div>`
            + `<div class="evidence-meta"><b>lines:</b> ${escapeHtml(String(startLine))}\u2013${escapeHtml(String(endLine))}</div>`
            + '</div>';
    }

    body.innerHTML = bodyHtml;
    sp.classList.add('open');
}

function toggleGroup(gid) {
    collapsedState[gid] = !collapsedState[gid];
    invokeRender();
}

function computeSelfMs(item) {
    const excPct = Number(item && item.exc_pct || 0);
    const stepUs = Number(DATA && DATA.meta && DATA.meta.step_dur_us || 0);
    return (excPct > 0 && stepUs > 0) ? (excPct * stepUs / 100.0 / 1000.0) : 0;
}

function fmtTimingMs(value, forceShow) {
    const num = Number(value || 0);
    if (num > 0) return `${num.toFixed(1)} ms`;
    if (forceShow) return `0.0 ms`;
    return '—';
}

function timingHelpText(kind) {
    const help = {
        kernel: 'Kernel time. The cumulative GPU kernel time attributed to this module instance (forward + backward + other-phase kernels). Excludes optimizer kernels and host walltime.',
        forward: 'Forward time. GPU kernel time classified into the forward phase for this module instance.',
        backward: 'Backward time. GPU kernel time classified into the backward phase for this module instance.',
        other: 'Other-phase kernel time. GPU kernel time attributed to this module that is neither forward nor backward (e.g. communication or fallback).'
    };
    return help[kind] || '';
}

function renderTimingRow(label, value, kind) {
    return `<div class="evidence-meta timing-row"><span class="timing-label"><b>${label}:</b><span class="metric-help" data-help="${escapeHtml(timingHelpText(kind))}">ⓘ</span></span><span class="timing-value">${escapeHtml(fmtTimingMs(value, true))}</span></div>`;
}

function showTooltip(e, n) {
    const tt = document.getElementById('tooltip');
    let html = `<div class="tt-title">${n.attr_name || n.label || n.class_name} <span style="opacity:0.6">(${n.class_name || n.label || 'Module'})</span></div>`;
    // Kernel-only contract: dur_us mirrors kernel_us; fwd/bwd/other are the
    // only phase splits we expose. No host walltime / overhead displayed.
    const kernelMs = Number(n && n.kernel_us != null ? n.kernel_us : (n.dur_us || 0)) / 1000.0;
    const fwdMs = Number(n && n.fwd_kernel_us || 0) / 1000.0;
    const bwdMs = Number(n && n.bwd_kernel_us || 0) / 1000.0;
    const otherMs = Number(n && n.other_kernel_us || 0) / 1000.0;
    html += `<div class="tt-row">Kernel: ${fmtTimingMs(kernelMs, true)}</div>`;
    html += `<div class="tt-row">Forward: ${fmtTimingMs(fwdMs, true)}</div>`;
    html += `<div class="tt-row">Backward: ${fmtTimingMs(bwdMs, true)}</div>`;
    html += `<div class="tt-row">Other: ${fmtTimingMs(otherMs, true)}</div>`;
    tt.innerHTML = html;
    tt.style.left = (e.clientX + 12) + 'px';
    tt.style.top = (e.clientY + 12) + 'px';
    tt.classList.add('visible');
}

function hideTooltip() {
    document.getElementById('tooltip').classList.remove('visible');
}

// ----------------------------------------------------------------------
// Side panel: click-to-source UX
// ----------------------------------------------------------------------
function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderCodeBlock(text, startLine, highlightLine, varNames, markClass) {
    if (!text) return '<div class="code-block"><div class="code-line"><span class="code-text" style="color:#7a849c">(no source available)</span></div></div>';
    const lines = text.split('\n');
    // varNames may be a single string, an array, or undefined. We highlight
    // each tracked variable using a word-boundary regex so substrings (e.g.
    // `x` inside `xxx`) are not mistakenly marked.
    let names = [];
    if (Array.isArray(varNames)) {
        names = varNames.filter(v => v && /^[A-Za-z_][A-Za-z0-9_]*$/.test(v));
    } else if (typeof varNames === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(varNames)) {
        names = [varNames];
    }
    // Deduplicate while preserving order.
    names = Array.from(new Set(names));
    const extraMarkClass = (typeof markClass === 'string' && markClass.trim()) ? (' ' + markClass.trim()) : '';
    function highlightVars(s) {
        let out = escapeHtml(s);
        for (const v of names) {
            const re = new RegExp('\\b' + v.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + '\\b', 'g');
            out = out.replace(re, '<mark class="var-mark' + extraMarkClass + '">' + v + '</mark>');
        }
        return out;
    }
    const html = lines.map((line, i) => {
        const lineno = startLine + i;
        const cls = (highlightLine && lineno === highlightLine) ? 'code-line highlight' : 'code-line';
        return `<div class="${cls}"><span class="lineno">${lineno}</span><span class="code-text">${highlightVars(line)}</span></div>`;
    }).join('');
    return `<div class="code-block">${html}</div>`;
}


function showSourcePanel(g) {
    const sp = document.getElementById('side-panel');
    const body = document.getElementById('sp-body');
    if (!sp || !body) return;

    document.getElementById('sp-title').textContent = g.class_name || 'Source';
    document.getElementById('sp-subtitle').textContent = g.attr_name || '';

    let bodyHtml = '';
    if (g.attr_name) {
        bodyHtml += `<div class="evidence-meta"><b>attr_name:</b> ${escapeHtml(g.attr_name)}</div>`;
    }
    bodyHtml += `<div class="evidence-meta"><b>class_name:</b> ${escapeHtml(g.class_name)}</div>`;

    const kernelMs = Number(g && g.kernel_us != null ? g.kernel_us : (g.dur_us || 0)) / 1000.0;
    const fwdMs = Number(g && g.fwd_kernel_us || 0) / 1000.0;
    const bwdMs = Number(g && g.bwd_kernel_us || 0) / 1000.0;
    const otherMs = Number(g && g.other_kernel_us || 0) / 1000.0;
    if (g.has_timing || g.has_phase_timing) {
        bodyHtml += '<div class="side-panel-section"><h4>Timing</h4>' +
            renderTimingRow('Kernel', kernelMs, 'kernel') +
            renderTimingRow('Forward', fwdMs, 'forward') +
            renderTimingRow('Backward', bwdMs, 'backward') +
            renderTimingRow('Other', otherMs, 'other') +
            '</div>';
    }

    const sections = [];
    const sourceMap = (typeof SOURCE_MAP !== 'undefined') ? SOURCE_MAP : null;
    if (g.class_def_loc && sourceMap && sourceMap[g.class_def_loc.file]) {
        const lines = sourceMap[g.class_def_loc.file].split('\n');
        const startIdx = g.class_def_loc.line - 1;
        const classIndent = lines[startIdx].search(/\S/);
        let endIdx = startIdx + 1;
        while (endIdx < lines.length) {
            const line = lines[endIdx];
            if (line.trim() === '' || line.trim().startsWith('#')) { endIdx++; continue; }
            if (line.search(/\S/) <= classIndent) break;
            endIdx++;
        }
        const snippet = lines.slice(startIdx, endIdx).join('\n');
        sections.push({ title: 'Class ' + g.class_name + ' (definition)', snippet, startLine: g.class_def_loc.line, file: g.class_def_loc.file });
    }
    if (g.def_loc && sourceMap && sourceMap[g.def_loc.file]) {
        const lines = sourceMap[g.def_loc.file].split('\n');
        const startIdx = g.def_loc.line - 1;
        const snippet = lines.slice(Math.max(0, startIdx - 1), startIdx + 4).join('\n');
        sections.push({ title: 'Constructor site', snippet, startLine: Math.max(1, g.def_loc.line - 1), file: g.def_loc.file });
    }
    if (g.src_file && sourceMap && sourceMap[g.src_file]) {
        const lines = sourceMap[g.src_file].split('\n');
        const startIdx = g.src_start_line - 1;
        const snippet = lines.slice(Math.max(0, startIdx - 2), startIdx + 3).join('\n');
        sections.push({ title: 'Call site', snippet, startLine: Math.max(1, g.src_start_line - 2), file: g.src_file });
    }

    if (sections.length === 0) {
        bodyHtml += '<p style="color:#888;padding:0 18px">No source available.</p>';
    } else {
        bodyHtml += sections.map(s => `
            <div class="side-panel-section">
                <h4>${escapeHtml(s.title)}</h4>
                <div class="evidence-meta" style="margin-bottom:6px"><b>file:</b> ${escapeHtml(s.file)} &nbsp;·&nbsp; <b>line:</b> ${escapeHtml(String(s.startLine))}</div>
                <pre class="code-block">${escapeHtml(s.snippet)}</pre>
            </div>
        `).join('');
    }
    body.innerHTML = bodyHtml;
    sp.classList.add('open');
}

function toggleFlowBlock(bodyId, headEl) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    const open = body.classList.toggle('open');
    const toggle = headEl.querySelector('.flow-toggle');
    if (toggle) toggle.textContent = open ? '▼' : '▶';
}

function showEdgePanel(edge) {
    const sp = document.getElementById('side-panel');
    const formatEdgeEndpoint = (node) => {
        if (!node) return '?';
        const parts = [node.class_name || '?', node.attr_name || '?'];
        const callLoc = node.call_loc;
        if (callLoc) {
            const fileName = callLoc.file ? String(callLoc.file).split('/').pop() : '';
            const locText = fileName && callLoc.line ? `${fileName}:${callLoc.line}` : fileName;
            if (locText) parts.push(locText);
        }
        return parts.join(' · ');
    };
    const parentClass = edge.parent_class || '';
    document.getElementById('sp-title').textContent = 'Edge';
    const sub = parentClass ? `data dependency in ${parentClass}.forward()` : 'data dependency';
    document.getElementById('sp-subtitle').textContent = sub;

    let bodyHtml = '<div class="side-panel-section"><h4>Endpoints</h4>' +
        `<div class="evidence-meta"><b>From:</b> ${escapeHtml(formatEdgeEndpoint(edge.from_node))}</div>` +
        `<div class="evidence-meta"><b>To:</b> ${escapeHtml(formatEdgeEndpoint(edge.to_node))}</div>` +
        '</div>';

    const sourceMap = (typeof SOURCE_MAP !== 'undefined') ? SOURCE_MAP : null;
    const flows = edge.flows || {};
    const idxKeys = Object.keys(flows).sort((a, b) => Number(a) - Number(b));

    if (idxKeys.length > 0) {
        bodyHtml += '<div class="side-panel-section"><h4>Flows</h4>';
        const autoExpand = idxKeys.length <= 1;

        idxKeys.forEach(idx => {
            const flow = flows[idx];
            const shapeStr = (flow.shape && flow.shape.length > 0) ? '[' + flow.shape.join('\u00d7') + ']' : '';
            const dtypeShort = (flow.dtype || '').replace('torch.', '');
            const bodyId = `flow-body-${edge.from}-${edge.to}-${idx}`;

            bodyHtml += `<div class="flow-block">` +
                `<div class="flow-block-head" onclick="toggleFlowBlock('${bodyId}', this)">` +
                `<span class="flow-toggle">${autoExpand ? '▼' : '▶'}</span>` +
                `<span>idx\u202f=\u202f${escapeHtml(String(idx))}</span>` +
                (dtypeShort ? `<span class="flow-dtype">${escapeHtml(dtypeShort)}</span>` : '') +
                (shapeStr ? `<span class="flow-shape">${escapeHtml(shapeStr)}</span>` : '') +
                `</div>` +
                `<div class="flow-block-body${autoExpand ? ' open' : ''}" id="${escapeHtml(bodyId)}">`;

            (flow.steps || []).forEach((step, si) => {
                const loc = step.loc || {};
                const varName = step.var || '';
                const file = loc.file || '';
                const line = loc.line;
                const fileName = file.split('/').pop();
                const locLabel = fileName && line ? `${fileName}:${line}` : (fileName || '');

                let codeHtml = '';
                if (sourceMap && file && sourceMap[file] && line) {
                    const allLines = sourceMap[file].split('\n');
                    // 只显示命中行（单行），除非该行以 \ 结尾（Python 续行）则往下扩展
                    let endLine = line;
                    while (endLine < allLines.length && (allLines[endLine - 1] || '').trimEnd().endsWith('\\')) {
                        endLine++;
                    }
                    const snippet = allLines.slice(line - 1, endLine).join('\n');
                    codeHtml = renderCodeBlock(snippet, line, line, varName ? [varName] : [], 'var-mark');
                }

                bodyHtml += `<div class="flow-step">` +
                    `<div class="evidence-meta">` +
                    `<b>step\u202f${si}</b>` +
                    (varName ? `\u2002<code>${escapeHtml(varName)}</code>` : `\u2002<span style="color:#555">(no\u00a0var)</span>`) +
                    (locLabel ? `\u2002<span class="lineage-loc">${escapeHtml(locLabel)}</span>` : '') +
                    `</div>` +
                    (codeHtml ? `<div class="flow-step-code">${codeHtml}</div>` : '') +
                    `</div>`;
            });

            bodyHtml += `</div></div>`; // flow-block-body + flow-block
        });

        bodyHtml += '</div>'; // side-panel-section
    }

    document.getElementById('sp-body').innerHTML = bodyHtml;
    sp.classList.add('open');
}

document.getElementById('sp-close').addEventListener('click', () => {
    document.getElementById('side-panel').classList.remove('open');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Phase 2 step 5 — Semantic Zoom: ESC pops a focus level first.
        // Only when the focus stack is empty does ESC fall through to the
        // legacy "close side panel" behaviour, matching the priority order
        // in design/frontend_canvas_phase2.md §6.2 ("focus 退栈优先，再
        // side-panel 关闭").
        if (focusStack.length > 0) {
            exitFocus();
            return;
        }
        document.getElementById('side-panel').classList.remove('open');
    }
});

// Phase 2 step 5 — Semantic Zoom: the right double-click drill-down gesture
// happens on the Pixi canvas, which lives inside #dag-stage.  Browsers fire a
// native ``contextmenu`` on right-mouse-up; the Pixi-level ``rightclick``
// handler cannot always cancel it, so suppress it at the DOM container level
// too.  This is purely a UX guard (no system menu over the canvas) and never
// touches trace data.
//
// Problem 2: the group box's own right-double-click exits focus, but once a
// drilled-in subtree fills the viewport its boundary group box can be fully
// covered by child nodes, leaving the user no box to right-double-click.  We
// therefore ALSO recognise a right double-click at the *stage* level (two
// ``contextmenu`` events within RIGHT_DBLCLICK_MS and RIGHT_DBLCLICK_DIST px)
// and, when a focus is active, pop one level.  A gesture the group box already
// handled (stamped on ``__lastGroupFocusGestureAt``) is ignored here so we
// never double-handle the same physical double-click.
(function bindCanvasContextMenuGuard() {
    const stage = document.getElementById('dag-stage');
    if (!stage || typeof stage.addEventListener !== 'function') { return; }
    const RIGHT_DBLCLICK_MS = 350;
    const RIGHT_DBLCLICK_DIST = 6;
    const GROUP_GESTURE_GUARD_MS = 400;
    let lastCtxTime = 0;
    let lastCtxX = 0;
    let lastCtxY = 0;
    stage.addEventListener('contextmenu', function (e) {
        if (e && typeof e.preventDefault === 'function') { e.preventDefault(); }
        const now = __focusNowMs();
        const x = (e && typeof e.clientX === 'number') ? e.clientX : 0;
        const y = (e && typeof e.clientY === 'number') ? e.clientY : 0;
        const dt = lastCtxTime > 0 ? (now - lastCtxTime) : Infinity;
        const dist = Math.abs(x - lastCtxX) + Math.abs(y - lastCtxY);
        const isDouble = dt <= RIGHT_DBLCLICK_MS && dist <= RIGHT_DBLCLICK_DIST;
        lastCtxTime = now; lastCtxX = x; lastCtxY = y;
        if (!isDouble) { return; }
        // The group box already consumed this exact double-click — don't pop a
        // second level at the stage level.
        if (now - __lastGroupFocusGestureAt <= GROUP_GESTURE_GUARD_MS) { return; }
        if (focusStack.length > 0) {
            // Reset the recogniser so an immediate third right-click does not
            // re-trigger off this same gesture.
            lastCtxTime = 0;
            exitFocus();
        }
    });
})();

document.getElementById('btn-expand-all').addEventListener('click', () => {
    expandAll();
});
document.getElementById('btn-collapse-all').addEventListener('click', () => {
    collapseAll();
});
document.getElementById('btn-fit').addEventListener('click', () => {
    if (typeof window.__canvasEnginePhase1 !== 'function') {
        throw new Error('Canvas engine accessor __canvasEnginePhase1 is missing');
    }
    const engine = window.__canvasEnginePhase1();
    if (!engine || !engine.viewportController || typeof engine.viewportController.fitToView !== 'function') {
        throw new Error('Canvas viewport controller is unavailable');
    }
    if (!engine.contentBounds) {
        throw new Error('Canvas content bounds are unavailable');
    }
    const container = document.getElementById('dag-container');
    const containerWidth = container && Number(container.clientWidth);
    const containerHeight = container && Number(container.clientHeight);
    engine.viewportController.fitToView(engine.contentBounds, containerWidth, containerHeight);
});

invokeRender();

// Phase 2 step 5 — populate the topbar runstep capsule from the model meta and
// render the (initially empty) Semantic-Zoom focus path once at startup.
if (typeof document !== 'undefined' && document.getElementById('dag-breadcrumb-nav')) {
    initTopbarRunstep();
    renderBreadcrumb();
    // Wire the L1/L2 capsule dropdowns.  No-op on the single-runstep base page
    // (ALL_TAB_DATA / DATA_TRAIN are only defined by the multi-/dual-tab wrappers,
    // whose switch scripts call refreshTopbarCapsules() again after they load).
    refreshTopbarCapsules();
}

// ── Topbar L1/L2 capsule dropdowns (Phase 2 step 5) ─────────────────────────
// The legacy two-row tab bar (.iter14-tabs / .multi-tabs-l1) has been removed;
// the Training (L1) and runstep (L2) switchers now live in the unified topbar as
// dropdown menus.  ``refreshTopbarCapsules()`` (re)labels the two capsules and
// (re)binds their click handlers from the active dataset model.  It is wired by
// the multi-/dual-tab switch scripts (which define ALL_TAB_DATA / DATA_TRAIN);
// on the single-runstep base page it leaves both capsules inert.
function _closeTopbarDropdown() {
    if (typeof document === 'undefined') { return; }
    const existing = document.getElementById('tb-active-dropdown');
    if (existing && existing.parentNode && typeof existing.parentNode.removeChild === 'function') {
        existing.parentNode.removeChild(existing);
    }
}
function _openTopbarDropdown(anchorEl, items, activeIdx, onSelect) {
    if (typeof document === 'undefined') { return; }
    _closeTopbarDropdown();
    const menu = document.createElement('div');
    menu.className = 'tb-dropdown-menu';
    menu.id = 'tb-active-dropdown';
    for (let i = 0; i < items.length; i++) {
        const it = document.createElement('div');
        it.className = 'tb-dropdown-item' + (i === activeIdx ? ' tb-dropdown-active' : '');
        it.textContent = items[i];
        const idx = i;
        it.addEventListener('click', (ev) => {
            if (ev && typeof ev.stopPropagation === 'function') { ev.stopPropagation(); }
            _closeTopbarDropdown();
            onSelect(idx);
        });
        menu.appendChild(it);
    }
    document.body.appendChild(menu);
    // Anchor the menu just under the clicked capsule when geometry is available.
    if (anchorEl && typeof anchorEl.getBoundingClientRect === 'function') {
        const r = anchorEl.getBoundingClientRect();
        menu.style.left = r.left + 'px';
        menu.style.top = (r.bottom + 4) + 'px';
    }
    // One-shot outside-click dismissal (deferred so the opening click does not
    // immediately re-close it).
    setTimeout(() => {
        if (typeof document.addEventListener === 'function') {
            document.addEventListener('click', _closeTopbarDropdown, { once: true });
        }
    }, 0);
}
function _capitalizeMode(s) {
    return (typeof s === 'string' && s.length) ? (s.charAt(0).toUpperCase() + s.slice(1)) : s;
}
function refreshTopbarCapsules() {
    if (typeof document === 'undefined') { return; }
    const modeCap = document.getElementById('tb-mode-capsule');
    const runCap = document.getElementById('tb-runstep-capsule');
    const hasMulti = (typeof ALL_TAB_DATA !== 'undefined') && ALL_TAB_DATA;
    const hasDual = (typeof DATA_TRAIN !== 'undefined') && (typeof DATA_INFER !== 'undefined');
    // L1 (Training / Inference) capsule.
    if (modeCap) {
        if (hasMulti) {
            const modes = Object.keys(ALL_TAB_DATA);
            modeCap.textContent = _capitalizeMode(ACTIVE_L1);
            if (modes.length > 1) {
                modeCap.classList.remove('tb-capsule-disabled');
                modeCap.onclick = (ev) => {
                    if (ev && typeof ev.stopPropagation === 'function') { ev.stopPropagation(); }
                    _openTopbarDropdown(modeCap, modes.map(_capitalizeMode), modes.indexOf(ACTIVE_L1), (i) => activateL1(modes[i]));
                };
            } else {
                modeCap.classList.add('tb-capsule-disabled');
                modeCap.onclick = null;
            }
        } else if (hasDual) {
            const order = ['train', 'infer'];
            const cur = (typeof DATA !== 'undefined' && DATA === DATA_INFER) ? 'infer' : 'train';
            modeCap.textContent = (cur === 'infer') ? 'Inference' : 'Training';
            modeCap.classList.remove('tb-capsule-disabled');
            modeCap.onclick = (ev) => {
                if (ev && typeof ev.stopPropagation === 'function') { ev.stopPropagation(); }
                _openTopbarDropdown(modeCap, ['Training', 'Inference'], order.indexOf(cur), (i) => {
                    if (typeof window !== 'undefined' && typeof window.__iter14_activateTab === 'function') {
                        window.__iter14_activateTab(order[i]);
                    }
                    refreshTopbarCapsules();
                });
            };
        } else {
            // single-runstep base page: nothing to switch.
            modeCap.classList.add('tb-capsule-disabled');
            modeCap.onclick = null;
        }
    }
    // L2 (runstep) capsule.  Problem 5 root-cause fix: every branch must set the
    // disabled state EXPLICITLY so a dataset switch (activateTab / activateL1 /
    // activateL2 all re-run refreshTopbarCapsules) never leaves the capsule's
    // ``tb-capsule-disabled`` class in a stale state inherited from the previous
    // dataset.  Previously only the ``hasMulti`` branch touched runCap, so in
    // dual / single-runstep mode the class was whatever the prior render left it.
    if (runCap) {
        if (hasMulti) {
            const steps = ALL_TAB_DATA[ACTIVE_L1] || [];
            const curIdx = ACTIVE_L2[ACTIVE_L1] || 0;
            const labelOf = (e) => String((e && e.label != null) ? e.label : 'runstep');
            runCap.textContent = steps.length ? labelOf(steps[curIdx]) : 'runstep';
            if (steps.length > 1) {
                runCap.classList.remove('tb-capsule-disabled');
                runCap.onclick = (ev) => {
                    if (ev && typeof ev.stopPropagation === 'function') { ev.stopPropagation(); }
                    _openTopbarDropdown(runCap, steps.map(labelOf), curIdx, (i) => activateL2(ACTIVE_L1, i));
                };
            } else {
                runCap.classList.add('tb-capsule-disabled');
                runCap.onclick = null;
            }
        } else {
            // dual (Training/Inference only) and single-runstep base page: there
            // is no runstep switcher, so the capsule is always disabled and inert.
            runCap.classList.add('tb-capsule-disabled');
            runCap.onclick = null;
        }
    }
}

// ── metric-help custom tooltip ──────────────────────────────────────────────
(function() {
    const popup = document.createElement('div');
    popup.className = 'metric-help-popup';
    popup.id = 'metric-help-popup';
    document.body.appendChild(popup);

    document.addEventListener('mouseover', function(e) {
        const el = e.target.closest('.metric-help[data-help]');
        if (!el) return;
        const text = el.getAttribute('data-help');
        if (!text) return;
        popup.textContent = text;
        popup.style.display = 'block';
        positionPopup(e);
    });
    document.addEventListener('mousemove', function(e) {
        if (popup.style.display === 'block') positionPopup(e);
    });
    document.addEventListener('mouseout', function(e) {
        const el = e.target.closest('.metric-help[data-help]');
        if (el) popup.style.display = 'none';
    });
    function positionPopup(e) {
        const pad = 10;
        let x = e.clientX + pad;
        let y = e.clientY + pad;
        const pw = popup.offsetWidth || 200;
        const ph = popup.offsetHeight || 40;
        if (x + pw > window.innerWidth - pad) x = e.clientX - pw - pad;
        if (y + ph > window.innerHeight - pad) y = e.clientY - ph - pad;
        popup.style.left = x + 'px';
        popup.style.top  = y + 'px';
    }
})();
</script>
</body>
</html>"""


def _collect_source_files(data: dict) -> set[str]:
    """遍历 groups，收集所有 loc 字段引用的 file 路径（去重）。"""
    files: set[str] = set()
    for group in data.get("groups", []):
        src_file = group.get("src_file")
        if src_file:
            files.add(src_file)
        for loc_field in ("call_loc", "def_loc", "class_def_loc"):
            loc = group.get(loc_field)
            if isinstance(loc, dict):
                f = loc.get("file")
                if f:
                    files.add(f)
    # edges → flows → steps → loc
    for edge in data.get("edges", []):
        for flow in (edge.get("flows") or {}).values():
            for step in (flow.get("steps") or []):
                loc = step.get("loc")
                if isinstance(loc, dict) and loc.get("file"):
                    files.add(loc["file"])
    return files


def _build_source_map(file_paths: set[str]) -> dict[str, str]:
    """读取文件内容，返回 {path: content}，不存在的文件跳过。"""
    source_map: dict[str, str] = {}
    for path in file_paths:
        try:
            source_map[path] = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
    return source_map


def _inject_source_map(html_content: str, source_files: set[str]) -> str:
    source_map = _build_source_map(source_files)
    source_map_js = "const SOURCE_MAP = " + _json.dumps(source_map, ensure_ascii=True).replace("</", "<\\/") + ";"
    return html_content.replace("// __SOURCE_MAP_PLACEHOLDER__", source_map_js)


def _replace_data_preamble(base_html: str, preamble: str, *, context: str) -> str:
    replaced_html, replace_count = re.subn(
        r"const DATA = .*?(?=\nconst groupMap = \{\};)",
        lambda _match: preamble,
        base_html,
        count=1,
        flags=re.DOTALL,
    )
    if replace_count != 1:
        raise RuntimeError(f"failed to replace DATA preamble for {context}")
    return replaced_html


def render_flowchart_to_file(flowchart_data, output_path):
    html_content = _generate_flowchart_html(flowchart_data)
    source_files = _collect_source_files(flowchart_data)
    html_content = _inject_source_map(html_content, source_files)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


def render_dual_flowchart_to_file(data_train, data_infer, output_path):
    html_content = _generate_flowchart_html_dual(data_train, data_infer)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


def render_multi_tab_flowchart_to_file(
    tabs: dict[str, list[dict]],
    output_path: str,
) -> None:
    """
    tabs: {"training": [{"label": "step_forward", "data": <adapted_dict>}, ...],
           "inference": [{"label": "predict_online", "error": "...", "warnings": [...]}, ...]}
    每个 mode 下的 list 可以有任意数量 step，各出一张图对应一个 L2 tab。
    """
    html_content = _generate_flowchart_html_multi(tabs)
    source_files: set[str] = set()
    for items in tabs.values():
        for item in items:
            if "data" in item:
                source_files.update(_collect_source_files(item["data"]))

    html_content = _inject_source_map(html_content, source_files)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


def _generate_flowchart_html_multi(tabs: dict[str, list[dict]]) -> str:
    if not tabs:
        raise ValueError("tabs must not be empty")
    first_mode = list(tabs.keys())[0]
    if not tabs[first_mode]:
        raise ValueError(f"tabs[{first_mode!r}] must contain at least one entry")
    default_mode = None
    default_idx = None
    default_data = None
    for mode, items in tabs.items():
        if not items:
            raise ValueError(f"tabs[{mode!r}] must contain at least one entry")
        for idx, item in enumerate(items):
            if "label" not in item:
                raise ValueError(f"tab item for mode={mode!r} must contain label")
            if "data" not in item and "error" not in item:
                raise ValueError(f"tab item for mode={mode!r} must contain data or error")
            if default_data is None and "data" in item:
                default_mode = mode
                default_idx = idx
                default_data = item["data"]

    if default_data is None or default_mode is None or default_idx is None:
        raise ValueError("at least one tab item must contain data")

    base_html = _generate_flowchart_html(default_data)

    all_tab_data = {mode: items for mode, items in tabs.items()}
    active_l2 = {mode: 0 for mode in tabs.keys()}
    active_l2[default_mode] = default_idx

    all_tab_data_json = _json.dumps(all_tab_data, ensure_ascii=True).replace("</", "<\\/")
    active_l2_json = _json.dumps(active_l2, ensure_ascii=True)
    data_preamble = (
        "const ALL_TAB_DATA = " + all_tab_data_json + ";\n"
        f'let ACTIVE_L1 = {default_mode!r};\n'
        "const ACTIVE_L2 = " + active_l2_json + ";\n"
        "let DATA = (ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]].data || ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]]);\n"
        "// __SOURCE_MAP_PLACEHOLDER__\n"
    )
    base_html = _replace_data_preamble(
        base_html,
        data_preamble,
        context="multi-tab flowchart html",
    )

    # Phase 2 step 5 — the legacy two-row tab header (L1 Training/Inference tabs
    # + L2 runstep tabs) has been removed.  L1/L2 switching is now driven by the
    # unified topbar's Training and runstep capsule dropdowns (see the inline
    # runtime's ``refreshTopbarCapsules`` / dropdown wiring), which reuse
    # ``ALL_TAB_DATA`` + ``activateL1`` / ``activateL2`` below.  No tab-row HTML
    # or CSS is injected anymore.

    switch_js = (
        "\n<script>\n"
        "function _resolveTabPayload(tabEntry) {\n"
        "    if (!tabEntry) {\n"
        "        throw new Error('multi-tab switch got empty tab entry');\n"
        "    }\n"
        "    if (tabEntry.error) {\n"
        "        return tabEntry;\n"
        "    }\n"
        "    return tabEntry.data || tabEntry;\n"
        "}\n"
        "function _ensureDagShell() {\n"
        "    var dagContainer = document.getElementById(\"dag-container\");\n"
        "    if (!dagContainer) return;\n"
        "    if (!document.getElementById(\"dag-stage\")) {\n"
        "        dagContainer.innerHTML = '<div class=\"dag-stage\" id=\"dag-stage\"></div>';\n"
        "    }\n"
        "}\n"
        "function _showErrorPanel(errMsg, warnings) {\n"
        "    var metaInfo = document.getElementById(\"meta-info\");\n"
        "    if (metaInfo) metaInfo.textContent = 'DAG build failed for this step';\n"
        "    var modeBadge = document.getElementById(\"mode-badge\");\n"
        "    if (modeBadge) modeBadge.innerHTML = '<span class=\"mode-badge\" style=\"background:rgba(248,81,73,0.15);color:#f85149;border-color:rgba(248,81,73,0.35)\">⚠ Build Error</span>';\n"
        "    var legend = document.getElementById(\"legend\");\n"
        "    if (legend) legend.innerHTML = warnings.length ? '<div class=\"legend-item\" style=\"color:#f0ad4e\">⚠ Build warnings: ' + escapeHtml(String(warnings.length)) + '</div>' : '';\n"
        "    var summary = document.getElementById(\"summary\");\n"
        "    if (summary) summary.innerHTML = '<h3>⚠ DAG Build Failed</h3><p>This runstep did not produce a renderable DAG. See the error details below.</p>';\n"
        "    var dagContainer = document.getElementById(\"dag-container\");\n"
        "    if (!dagContainer) return;\n"
        "    var warningHtml = warnings.length ? '<details><summary style=\"cursor:pointer;color:#f0ad4e\">⚠ Build Warnings (' + escapeHtml(String(warnings.length)) + ')</summary><pre style=\"font-size:11px;color:#f0ad4e;white-space:pre-wrap\">' + warnings.map(function (w) { return escapeHtml(w); }).join('\\n') + '</pre></details>' : '';\n"
        "    dagContainer.innerHTML = '<div style=\"padding:40px;font-family:monospace\"><div style=\"color:#f85149;font-size:16px;margin-bottom:16px\">⚠ DAG build failed for this step</div><pre style=\"background:#161b22;padding:16px;border-radius:6px;color:#ffa657;font-size:12px;white-space:pre-wrap;overflow:auto\">' + escapeHtml(errMsg) + '</pre>' + warningHtml + '</div>';\n"
        "}\n"
        "function _resetSharedState() {\n"
        "    if (typeof groupMap === \"object\" && groupMap) {\n"
        "        Object.keys(groupMap).forEach(k => { delete groupMap[k]; });\n"
        "    }\n"
        "    if (typeof nodeMap === \"object\" && nodeMap) {\n"
        "        Object.keys(nodeMap).forEach(k => { delete nodeMap[k]; });\n"
        "    }\n"
        "    if (typeof collapsedState === \"object\" && collapsedState) {\n"
        "        Object.keys(collapsedState).forEach(k => { delete collapsedState[k]; });\n"
        "    }\n"
        "    try { if (typeof nodeAncestorGroups !== \"undefined\" && nodeAncestorGroups && nodeAncestorGroups.clear) nodeAncestorGroups.clear(); } catch (e) {}\n"
        "    try { if (typeof nodeDomRegistry !== \"undefined\" && nodeDomRegistry && nodeDomRegistry.clear) nodeDomRegistry.clear(); } catch (e) {}\n"
        "    try { if (typeof groupDomRegistry !== \"undefined\" && groupDomRegistry && groupDomRegistry.clear) groupDomRegistry.clear(); } catch (e) {}\n"
        "    try { if (typeof edgeDomRegistry !== \"undefined\") edgeDomRegistry.length = 0; } catch (e) {}\n"
        "    try { if (typeof edgeByNodeId !== \"undefined\" && edgeByNodeId && edgeByNodeId.clear) edgeByNodeId.clear(); } catch (e) {}\n"
        "    try { if (typeof edgeByGroupId !== \"undefined\" && edgeByGroupId && edgeByGroupId.clear) edgeByGroupId.clear(); } catch (e) {}\n"
        "    try { if (typeof edgeDomByKey !== \"undefined\" && edgeDomByKey && edgeDomByKey.clear) edgeDomByKey.clear(); } catch (e) {}\n"
        "    try { if (typeof groupLayout !== \"undefined\") { Object.keys(groupLayout).forEach(k => delete groupLayout[k]); } } catch (e) {}\n"
        "    try { hoveredEdgeKey = null; hoveredEdges = []; hoveredEdgeIdx = 0; hoveredNodeId = null; hoveredGroupId = null; focusedEdgePath = null; prevActiveItems = []; prevActiveNodeIds = new Set(); prevActiveEdgeKeys = new Set(); prevActiveGroupNodeIds = new Set(); prevActiveGroupIds = new Set(); prevDimEdgeKeys = new Set(); prevDimGroupNodeIds = new Set(); prevDimGroupIds = new Set(); lastHoveredGid = null; } catch (e) {}\n"
        "    try { var sp = document.getElementById(\"side-panel\"); if (sp) sp.classList.remove(\"open\"); } catch (e) {}\n"
        "}\n"
        "function _rebuildIndices() {\n"
        "    DATA.groups.forEach(g => groupMap[g.id] = g);\n"
        "    DATA.nodes.forEach(n => nodeMap[n.id] = n);\n"
        "    if (typeof indexGroupAncestors === \"function\" && DATA.root_groups) {\n"
        "        indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));\n"
        "    }\n"
        # Re-seed io_group member -> io_group ancestor mapping that
        # _resetSharedState() cleared.  Mirrors the initial-load setup in
        # _generate_flowchart_html.  Without this, nodes living as members of
        # a collapsed io_group lose their ancestor mapping after a tab switch,
        # so resolveCollapsedAncestor() falls through to the raw node id
        # (never registered in nodePortMap for collapsed io_groups) and
        # drawGlobalEdges throws "global edge endpoint missing".
        "    (DATA.io_groups || []).forEach(g => {\n"
        "        (g.member_ids || []).forEach(nid => nodeAncestorGroups.set(nid, [g.id]));\n"
        "    });\n"
        "    DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2 || g.is_native === true || g.synthetic_type === 'function_group' || g.synthetic_type === 'callloc_group'; });\n"
        # Re-seed io_group collapsedState defaults that _resetSharedState() cleared.
        "    (DATA.io_groups || []).forEach(g => { if (!(g.id in collapsedState)) collapsedState[g.id] = g.collapsed; });\n"
        "}\n"
        "function activateL1(mode) {\n"
        "    ACTIVE_L1 = mode;\n"
        "    if (focusStack.length > 0) {\n"
        "        focusStack.length = 0;\n"
        "        renderBreadcrumb();\n"
        "    }\n"
        "    switchDataset(ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]]);\n"
        "    if (typeof refreshTopbarCapsules === 'function') { refreshTopbarCapsules(); }\n"
        "}\n"
        "function activateL2(mode, idx) {\n"
        "    ACTIVE_L2[mode] = idx;\n"
        "    if (mode === ACTIVE_L1) {\n"
        "        if (focusStack.length > 0) {\n"
        "            focusStack.length = 0;\n"
        "            renderBreadcrumb();\n"
        "        }\n"
        "        switchDataset(ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]]);\n"
        "    }\n"
        "    if (typeof refreshTopbarCapsules === 'function') { refreshTopbarCapsules(); }\n"
        "}\n"
        "function switchDataset(nextTabEntry) {\n"
        "    var nextData = _resolveTabPayload(nextTabEntry);\n"
        "    if (DATA === nextData) return;\n"
        "    _resetSharedState();\n"
        "    DATA = nextData;\n"
        "    if (DATA && DATA.error) {\n"
        "        _showErrorPanel(DATA.error, DATA.warnings || []);\n"
        "        return;\n"
        "    }\n"
        "    _ensureDagShell();\n"
        "    _rebuildIndices();\n"
        "    invokeRender();\n"
        "}\n"
        # Multi-tab mode: now that ALL_TAB_DATA / ACTIVE_L1 / ACTIVE_L2 exist,
        # relabel the topbar Training + runstep capsules and turn them into the
        # active dropdowns (the base inline runtime saw them as undefined at its
        # own load time and left them inert).
        "if (typeof refreshTopbarCapsules === 'function') { refreshTopbarCapsules(); }\n"
        "</script>\n"
    )
    if "</body>" in base_html:
        base_html = base_html.replace("</body>", switch_js + "</body>", 1)
    else:
        base_html = base_html + switch_js
    return base_html


def _generate_flowchart_html_dual(data_train, data_infer):
    """Iter14 (revised): build a single HTML page hosting both Training and
    Inference DAGs as switchable tabs.

    Earlier revision used iframes with ``data:text/html;base64`` URLs. That
    approach broke for large models (5547919 infer DAG is ~5 MB base64) because
    Chromium silently refuses to render data: URLs above its top-frame limit
    (≈2 MB for navigation), so the infer tab simply showed a blank pane.

    New approach: produce **one** inner HTML using the standard
    ``_generate_flowchart_html`` template, embed BOTH datasets as
    ``DATA_TRAIN`` / ``DATA_INFER`` JSON literals, and let JS switch the active
    DAG by re-pointing ``DATA`` and re-running the existing initialization +
    ``render()`` pipeline. This keeps the proven render logic untouched while
    making tab switching truly instant — no iframe size limit, no duplicated
    DOM, no cross-tab leakage (each switch fully resets ``groupMap`` /
    ``nodeMap`` / ``collapsedState`` / ``nodeAncestorGroups``).
    """

    # Use the train dataset to render the base template; the JSON payload will
    # be replaced with a dual-data preamble below.
    base_html = _generate_flowchart_html(data_train)

    train_json = _json.dumps(data_train, ensure_ascii=True)
    infer_json = _json.dumps(data_infer, ensure_ascii=True)

    # ---- 1. swap `const DATA = {...}` for a dual-data preamble --------------
    # The single-tab template uses the regex:
    #   const DATA = ... (?=\nconst groupMap = \{\};)
    # so we anchor to the same boundary.
    dual_preamble = (
        "const DATA = " + train_json + ";\n"
        "const DATA_TRAIN = " + train_json + ";\n"
        "const DATA_INFER = " + infer_json + ";\n"
    )
    try:
        base_html = _replace_data_preamble(
            base_html,
            dual_preamble,
            context="dual-tab flowchart html",
        )
    except RuntimeError:
        # Fall back to legacy iframe shell (extremely defensive — should never
        # trigger because _generate_flowchart_html always emits the marker).
        print("  ⚠️ 双 Tab: 未在内嵌模板中找到 DATA 注入锚点，回退为单 Tab(train) 输出")
        return base_html

    # ---- 2. legacy two-row tab bar removed -------------------------------
    # Phase 2 step 5 — the standalone ``.iter14-tabs`` train/infer tab row that
    # used to be injected after <body> is gone.  The unified topbar (rendered by
    # the base single-tab template) is the only header now.  ``activateTab`` is
    # retained below so the proven reset/rebuild/render switch pipeline stays
    # intact, but no tab-row HTML or CSS is injected.

    # ---- 3. inject tab-switching JS at the very end (after final render()) --
    switch_js = (
        '\n<script>\n'
        '(function(){\n'
        '  function _resetSharedState(){\n'
        '    // Reset every piece of mutable global state the inner template\n'
        '    // built from DATA so the new tab\'s DAG renders cleanly.\n'
        '    if (typeof groupMap === "object" && groupMap){\n'
        '      Object.keys(groupMap).forEach(k => { delete groupMap[k]; });\n'
        '    }\n'
        '    if (typeof nodeMap === "object" && nodeMap){\n'
        '      Object.keys(nodeMap).forEach(k => { delete nodeMap[k]; });\n'
        '    }\n'
        '    if (typeof collapsedState === "object" && collapsedState){\n'
        '      Object.keys(collapsedState).forEach(k => { delete collapsedState[k]; });\n'
        '    }\n'
        '    try { if (typeof nodeAncestorGroups !== "undefined" && nodeAncestorGroups && nodeAncestorGroups.clear) nodeAncestorGroups.clear(); } catch(e){}\n'
        '    try { if (typeof nodeDomRegistry !== "undefined" && nodeDomRegistry && nodeDomRegistry.clear) nodeDomRegistry.clear(); } catch(e){}\n'
        '    try { if (typeof groupDomRegistry !== "undefined" && groupDomRegistry && groupDomRegistry.clear) groupDomRegistry.clear(); } catch(e){}\n'
        '    try { if (typeof edgeDomRegistry !== "undefined") edgeDomRegistry.length = 0; } catch(e){}\n'
        '    try { if (typeof edgeByNodeId !== "undefined" && edgeByNodeId && edgeByNodeId.clear) edgeByNodeId.clear(); } catch(e){}\n'
        '    try { if (typeof edgeByGroupId !== "undefined" && edgeByGroupId && edgeByGroupId.clear) edgeByGroupId.clear(); } catch(e){}\n'
        '    try { if (typeof edgeDomByKey !== "undefined" && edgeDomByKey && edgeDomByKey.clear) edgeDomByKey.clear(); } catch(e){}\n'
        '    try { if (typeof groupLayout !== "undefined") { Object.keys(groupLayout).forEach(k => delete groupLayout[k]); } } catch(e){}\n'
        '    try { hoveredEdgeKey = null; hoveredEdges = []; hoveredEdgeIdx = 0; hoveredNodeId = null; hoveredGroupId = null; focusedEdgePath = null; prevActiveItems = []; prevActiveNodeIds = new Set(); prevActiveEdgeKeys = new Set(); prevActiveGroupNodeIds = new Set(); prevActiveGroupIds = new Set(); prevDimEdgeKeys = new Set(); prevDimGroupNodeIds = new Set(); prevDimGroupIds = new Set(); lastHoveredGid = null; } catch(e){}\n'
        '    // Close any open side panel from the previous tab.\n'
        '    try { var sp = document.getElementById("side-panel"); if (sp) sp.classList.remove("open"); } catch(e){}\n'
        '  }\n'
        '  function _rebuildIndices(){\n'
        '    DATA.groups.forEach(g => groupMap[g.id] = g);\n'
        '    DATA.nodes.forEach(n => nodeMap[n.id] = n);\n'
        '    if (typeof indexGroupAncestors === "function" && DATA.root_groups){\n'
        '      indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));\n'
        '    }\n'
        # Re-seed io_group member -> io_group ancestor mapping that
        # _resetSharedState() cleared (see _generate_flowchart_html_multi for
        # the full root-cause comment).
        '    (DATA.io_groups || []).forEach(g => {\n'
        '      (g.member_ids || []).forEach(nid => nodeAncestorGroups.set(nid, [g.id]));\n'
        '    });\n'
        '    DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2 || g.is_native === true; });\n'
        # Re-seed io_group collapsedState defaults that _resetSharedState() cleared.
        '    (DATA.io_groups || []).forEach(g => { if (!(g.id in collapsedState)) collapsedState[g.id] = g.collapsed; });\n'
        '  }\n'
        '  function activateTab(mode){\n'
        '    var nextData = (mode === "infer") ? DATA_INFER : DATA_TRAIN;\n'
        '    if (DATA === nextData) return;\n'
        '    if (focusStack.length > 0) {\n'
        '      focusStack.length = 0;\n'
        '      renderBreadcrumb();\n'
        '    }\n'
        '    _resetSharedState();\n'
        '    DATA = nextData;\n'
        '    _ensureDagShell();\n'
        '    _rebuildIndices();\n'
        '    if (typeof invokeRender === "function") invokeRender();\n'
        '    if (typeof refreshTopbarCapsules === "function") refreshTopbarCapsules();\n'
        '  }\n'
        '  // Expose for debugging.\n'
        '  window.__iter14_activateTab = activateTab;\n'
        '  // The standalone .iter14-tabs tab row was removed; the Training /\n'
        '  // Inference switch now lives in the unified topbar mode capsule.\n'
        '  if (typeof refreshTopbarCapsules === "function") refreshTopbarCapsules();\n'
        '})();\n'
        '</script>\n'
    )
    if '</body>' in base_html:
        base_html = base_html.replace('</body>', switch_js + '</body>', 1)
    else:
        base_html = base_html + switch_js

    return base_html


def _generate_flowchart_html(data):
    # ensure_ascii=True is safer for character handling
    data_json = _json.dumps(data, ensure_ascii=True)
    html_template = FLOWCHART_HTML_TEMPLATE

    # Canvas Phase 1 / Stage 1.1: inject the Pixi engine bundle and
    # render_canvas.js.  These two placeholders are the only runtime render
    # injection path now; a missing placeholder is a hard error (no fallback).
    engine_bundle_js = _ENGINE_BUNDLE_PATH.read_text(encoding="utf-8")
    if _ENGINE_BUNDLE_PLACEHOLDER not in html_template:
        raise RuntimeError("engine bundle placeholder is missing from FLOWCHART_HTML_TEMPLATE")
    html_template = html_template.replace(_ENGINE_BUNDLE_PLACEHOLDER, engine_bundle_js, 1)

    render_canvas_js = _RENDER_CANVAS_JS_PATH.read_text(encoding="utf-8")
    if _RENDER_CANVAS_JS_PLACEHOLDER not in html_template:
        raise RuntimeError("render_canvas.js placeholder is missing from FLOWCHART_HTML_TEMPLATE")
    html_template = html_template.replace(_RENDER_CANVAS_JS_PLACEHOLDER, render_canvas_js, 1)

    # The legacy render_group.js slot is intentionally NOT fed any runtime code.
    # Stage 1.1 stops injecting the SVG group renderer entirely; if the template
    # still carries the placeholder it can only collapse to an empty string.
    if _RENDER_GROUP_JS_PLACEHOLDER in html_template:
        html_template = html_template.replace(_RENDER_GROUP_JS_PLACEHOLDER, "", 1)

    # Replace the embedded DATA payload using the stable script markers rather
    # than a naive non-greedy `{...};` regex: the serialized JSON contains many
    # nested `};` substrings inside code snippets, which can cause partial
    # matches and leave stale template data in place.
    html_template = html_template.replace(
        'const DATA = __FLOWCHART_DATA_PLACEHOLDER__;',
        'const DATA = ' + data_json + ';',
        1
    )

    return html_template



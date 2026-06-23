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
from collections import defaultdict
from pathlib import Path

# When this module is imported by ``analyze_trace.py`` the script directory is
# already on ``sys.path``.  We add it defensively here so the module also
# imports cleanly when loaded standalone (e.g. ``python -c "import
# frontend_html"`` from the scripts/ directory or from a tooling harness).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RENDER_GROUP_JS_PATH = Path(_SCRIPT_DIR) / "render_group.js"
_RENDER_GROUP_JS_PLACEHOLDER = "__RENDER_GROUP_JS_PLACEHOLDER__"
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
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; min-height: 100vh; }
.header { text-align: center; margin-bottom: 24px; }
.header h1 { font-size: 22px; color: #ffffff; margin-bottom: 6px; }
.header .meta { font-size: 12px; color: #8892b0; }
.header .mode-badge { display: inline-block; font-size: 11px; padding: 3px 12px; border-radius: 12px; margin-top: 6px; }
.header .mode-structure { background: rgba(39, 174, 96, 0.2); color: #27ae60; border: 1px solid rgba(39, 174, 96, 0.3); }
.header .mode-timing { background: rgba(41, 128, 185, 0.2); color: #64b5f6; border: 1px solid rgba(41, 128, 185, 0.3); }
.controls { display: flex; gap: 10px; justify-content: center; margin-bottom: 16px; }
.controls button { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); color: #ccc; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
.controls button:hover { background: rgba(255,255,255,0.15); color: #fff; }
.controls button.active { background: rgba(100,181,246,0.2); border-color: #64b5f6; color: #64b5f6; }
.dag-container { width: 100%; overflow: auto; position: relative; contain: layout style; }
.dag-svg { display: block; margin: 0 auto; will-change: transform; }
.dag-svg .group-box { fill: rgba(255,255,255,0.03); stroke: rgba(255,255,255,0.12); stroke-width: 1.5; rx: 10; cursor: pointer; transition: fill 0.2s; }
.dag-svg .group-box:hover { fill: rgba(255,255,255,0.06); stroke: rgba(100,181,246,0.4); }
.dag-svg .group-box.collapsed { fill: rgba(100,181,246,0.08); stroke: rgba(100,181,246,0.3); stroke-dasharray: 5,3; }
.dag-svg .group-label { font-size: 11px; fill: #8892b0; font-weight: 600; pointer-events: none; }
.dag-svg .group-timing { font-size: 10px; fill: #64b5f6; pointer-events: none; }
.dag-svg .group-toggle { font-size: 10px; fill: #555; pointer-events: auto; cursor: pointer; }
.dag-svg .group-info-hit { fill: rgba(100,181,246,0.001); stroke: rgba(100,181,246,0.45); stroke-width: 1; pointer-events: auto; cursor: pointer; }
.dag-svg .group-info-hit:hover { fill: rgba(100,181,246,0.12); stroke: rgba(100,181,246,0.9); }
.dag-svg .group-info-text { fill: rgba(100,181,246,0.82); font-size: 10px; font-weight: 700; pointer-events: none; text-anchor: middle; dominant-baseline: middle; }
.dag-svg .leaf-node { rx: 6; ry: 6; stroke-width: 1.5; cursor: default; transition: stroke 0.2s, stroke-width 0.2s; }
.dag-svg .leaf-node:hover { stroke: #ffffff; stroke-width: 2.5; }
.dag-svg .node-label { font-size: 11px; fill: #ffffff; font-weight: 500; pointer-events: none; text-anchor: middle; }
.dag-svg .node-sublabel { font-size: 9px; fill: rgba(255,255,255,0.7); pointer-events: none; text-anchor: middle; }
.dag-svg .edge-path { fill: none; stroke: rgba(255,255,255,0.2); stroke-width: 1.7; opacity: 0.82; transition: stroke 0.18s ease, stroke-width 0.18s ease, opacity 0.18s ease; }
.dag-svg .edge-path.internal { stroke: rgba(100,181,246,0.46); stroke-width: 1.35; }
.dag-svg .edge-path.dep { stroke: rgba(46,204,113,0.62); stroke-width: 1.9; }
.dag-svg .edge-path.flow { stroke: rgba(255,255,255,0.28); stroke-width: 1.35; stroke-dasharray: 4,3; }
.dag-svg .port { fill: rgba(255,255,255,0.15); stroke: rgba(255,255,255,0.3); stroke-width: 1; }
.tooltip { position: fixed; background: #16213e; border: 1px solid rgba(100,181,246,0.3); border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #e0e0e0; pointer-events: none; z-index: 1000; max-width: 300px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); opacity: 0; transition: opacity 0.15s; }
.tooltip.visible { opacity: 1; }
.tooltip .tt-title { font-weight: 600; color: #64b5f6; margin-bottom: 4px; }
.tooltip .tt-row { margin-top: 2px; }
.legend { display: flex; gap: 16px; justify-content: center; margin-bottom: 16px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #8892b0; }
.legend-dot { width: 10px; height: 10px; border-radius: 3px; }
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
.dag-svg .group-clickable { cursor: pointer; }
.dag-svg .edge-path { pointer-events: stroke; cursor: pointer; }
.dag-svg .edge-path:hover { stroke-width: 3.4; opacity: 1 !important; }
.dag-container.hover-active .edge-path:not(.edge-active) { opacity: 0.14 !important; }
.dag-container.hover-active .edge-path.edge-active { opacity: 1 !important; }
.dag-container.hover-active .edge-path.dep.edge-active { stroke: #7CFFB2 !important; }
.dag-container.hover-active .edge-path.flow.edge-active { stroke: #FFD166 !important; }
.dag-container.hover-active .edge-path.internal.edge-active { stroke: #7FD1FF !important; }
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
<div class="header">
    <h1>🔥 Module Architecture DAG</h1>
    <div class="meta" id="meta-info"></div>
    <div id="mode-badge"></div>
</div>
<div class="controls">
    <button id="btn-expand-all">Expand All</button>
    <button id="btn-collapse-all">Collapse All</button>
    <button id="btn-fit">Fit to View</button>
</div>
<div class="legend" id="legend"></div>
<div class="dag-container" id="dag-container">
    <svg class="dag-svg" id="dag-svg"></svg>
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

<script>
const DATA = __FLOWCHART_DATA_PLACEHOLDER__;
// __SOURCE_MAP_PLACEHOLDER__

const groupMap = {};
const nodeMap = {};
const collapsedState = {};
let groupLayout = {};
let nodePortMap = {};
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
let edgeOverlayLayer = null;
let activeEdgeOverlayPath = null;
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
    // 从前往后遍历找最外层的已折叠祖先
    const ancestors = nodeAncestorGroups.get(nodeId) || [];
    for (let i = 0; i < ancestors.length; i++) {
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
    if (activeEdgeOverlayPath && activeEdgeOverlayPath.parentNode) {
        const parent = activeEdgeOverlayPath.parentNode;
        if (typeof parent.removeChild === 'function') {
            parent.removeChild(activeEdgeOverlayPath);
        } else if (Array.isArray(parent.children)) {
            parent.children = parent.children.filter(child => child !== activeEdgeOverlayPath);
        }
        activeEdgeOverlayPath.parentNode = null;
    }
    activeEdgeOverlayPath = null;
}

function syncActiveEdgeOverlay(item) {
    if (!item || !item.path) {
        clearActiveEdgeOverlay();
        return;
    }
    if (!edgeOverlayLayer) {
        throw new Error('edge overlay layer is not initialized');
    }
    const sourcePath = item.path;
    const overlayPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    overlayPath.setAttribute('d', sourcePath.getAttribute('d') || '');
    for (const attrName of ['class', 'opacity', 'stroke-dasharray', 'stroke-dashoffset']) {
        const attrValue = sourcePath.getAttribute(attrName);
        if (attrValue !== null && attrValue !== undefined) {
            overlayPath.setAttribute(attrName, attrValue);
        }
    }
    for (const dataName of ['longEdge', 'fullDasharray', 'fullDashoffset', 'truncatedDasharray', 'truncatedDashoffset']) {
        if (sourcePath.dataset && sourcePath.dataset[dataName] !== undefined) {
            overlayPath.dataset[dataName] = sourcePath.dataset[dataName];
        }
    }
    overlayPath.style.pointerEvents = 'none';
    setEdgeItemFocusState({ path: overlayPath }, 'active');
    clearActiveEdgeOverlay();
    edgeOverlayLayer.appendChild(overlayPath);
    activeEdgeOverlayPath = overlayPath;
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
        syncArrowPolygonStyle(path);
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
    syncArrowPolygonStyle(path);
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

function layoutGroup(gid) {
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
            const sz = layoutGroup(item.id);
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
    const maxW = LAYOUT.maxRowWidth;
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
    setRenderProgress(100, '渲染完成');
    await nextFrame();
    if (ownerGeneration !== String(renderGeneration)) {
        throw new Error(`hideRenderProgress aborted by newer render generation: ${ownerGeneration} -> ${renderGeneration}`);
    }
    overlay.classList.add('closing');
    overlay.classList.remove('visible');
    await new Promise((resolve) => setTimeout(resolve, 180));
    if (ownerGeneration !== String(renderGeneration)) {
        throw new Error(`hideRenderProgress completion overtaken by render generation: ${ownerGeneration} -> ${renderGeneration}`);
    }
    overlay.setAttribute('aria-hidden', 'true');
}

function assertActiveRenderGeneration(generation, stageText) {
    if (generation !== renderGeneration) {
        throw new Error(`render generation ${generation} expired during ${stageText}`);
    }
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
    assertActiveRenderGeneration(generation, stageText);
    const total = items.length;
    if (total === 0) {
        setRenderProgress(phaseEnd, stageText);
        await nextFrame();
        assertActiveRenderGeneration(generation, stageText);
        return;
    }
    let processed = 0;
    while (processed < total) {
        assertActiveRenderGeneration(generation, stageText);
        const upperBound = Math.min(processed + batchSize, total);
        for (let idx = processed; idx < upperBound; idx++) {
            const item = items[idx];
            if (!item || typeof item.type !== 'string' || !allowedTypeSet.has(item.type)) {
                const actualType = item && typeof item.type === 'string' ? item.type : '<missing>';
                throw new Error(`runChunked got unknown task type: ${actualType}`);
            }
            assertActiveRenderGeneration(generation, stageText);
            await handler(item, idx);
        }
        processed = upperBound;
        const progress = phaseStart + (processed / total) * (phaseEnd - phaseStart);
        setRenderProgress(progress, stageText);
        await nextFrame();
    }
    assertActiveRenderGeneration(generation, stageText);
}

function invokeRender() {
    const promise = render();
    promise.catch((err) => {
        setTimeout(() => { throw err; }, 0);
    });
    return promise;
}

async function render() {
    const generation = ++renderGeneration;
    showRenderProgress('正在准备渲染状态…');
    const { overlay } = getRenderProgressElements();
    overlay.dataset.renderGeneration = String(generation);
    await nextFrame();
    try {
        groupLayout = {};
        nodePortMap = {};
        edgeDomRegistry = [];
        edgeByNodeId.clear();
        edgeByGroupId.clear();
        edgeDomByKey.clear();
        prevActiveItems = [];
        prevActiveNodeIds = new Set();
        prevActiveEdgeKeys = new Set();
        prevActiveGroupNodeIds = new Set();
        prevActiveGroupIds = new Set();
        prevDimEdgeKeys = new Set();
        prevDimGroupNodeIds = new Set();
        prevDimGroupIds = new Set();
        lastHoveredGid = null;
        nodeDomRegistry = new Map();
        groupDomRegistry.clear();
        focusedEdgePath = null;
        hoveredEdgeKey = null;
        hoveredEdges = [];
        hoveredEdgeIdx = 0;
        hoveredGroupId = null;
        hoveredNodeId = null;
        edgeOverlayLayer = null;
        clearActiveEdgeOverlay();
        hideOverlapBadge();
        assertActiveRenderGeneration(generation, '准备阶段');
        setRenderProgress(5, '正在准备渲染状态…');
        await nextFrame();

        const rootSizes = DATA.root_groups.map(rid => {
            const sz = layoutGroup(rid);
            return { id: rid, ...sz };
        });

        const ioW = 140, ioH = 40;
        const ioGap = 36;
        const pillGap = 18;
        const topIOItems = (DATA.io_groups && DATA.io_groups.length > 0)
            ? DATA.io_groups
                .filter(g => g.io_subtype !== 'output')
                .map(g => ({ isIOGroup: true, ioGroup: g }))
            : [
                ...(DATA.input_node_ids || []).map(id => ({ id, label: 'Input', defaultSublabel: 'network input', fillColor: 'rgba(46,204,113,0.55)' })),
                ...(DATA.param_node_ids || []).map(id => ({ id, label: 'Param', defaultSublabel: 'model param', fillColor: 'rgba(155,89,182,0.55)' })),
                ...(DATA.const_node_ids || []).map(id => ({ id, label: 'Const', defaultSublabel: 'const value', fillColor: 'rgba(241,196,15,0.55)' })),
            ];
        const outputIOGroup = (DATA.io_groups || []).find(g => g.io_subtype === 'output');
        const bottomIOItems = outputIOGroup
            ? [{ isIOGroup: true, ioGroup: outputIOGroup }]
            : (DATA.output_node_ids || []).map(id => ({
                  id, label: 'Result', defaultSublabel: 'result output',
                  fillColor: 'rgba(231,76,60,0.55)'
              }));
        const topIORows = topIOItems.length > 0 ? [{ items: topIOItems }] : [];
        const bottomIORows = bottomIOItems.length > 0 ? [{ items: bottomIOItems }] : [];
        const allIORows = topIORows.concat(bottomIORows);
        const maxRootW = rootSizes.length ? Math.max(...rootSizes.map(r => r.w)) : LAYOUT.nodeW;
        const maxRootH = rootSizes.length ? Math.max(...rootSizes.map(r => r.h)) : LAYOUT.nodeH;
        const provisionalSvgW = Math.max(maxRootW + 80, 480);
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
        const expandedGroupCount = (DATA.io_groups || []).filter(g =>
            !((g.id in collapsedState) ? collapsedState[g.id] : g.collapsed)
        ).length;
        const minIOColW = EXPAND_COLS * EXPAND_PILL_W + (EXPAND_COLS - 1) * pillGap + 2 * EXPAND_FRAME_PAD;
        const minIOTotalW = expandedGroupCount > 0
            ? expandedGroupCount * minIOColW + (expandedGroupCount - 1) * EXPAND_COL_GAP + 2 * EXPAND_OUTER_PAD
            : 0;
        const computeIOGroupExpandedLayout = (memberCount, availableSvgW = provisionalSvgW) => {
            if (memberCount === 0) return { cols: EXPAND_COLS, pillW: EXPAND_PILL_W, memberRows: 0, height: 0 };
            const availableW = availableSvgW - 2 * EXPAND_FRAME_PAD;
            const cols = Math.max(EXPAND_COLS, Math.floor(availableW / (EXPAND_PILL_W + pillGap)));
            const pillW = Math.floor((availableW - (cols - 1) * pillGap) / cols);
            const memberRows = Math.ceil(memberCount / cols);
            const memberAreaH = memberRows * EXPANDED_IO_H + (memberRows - 1) * EXPANDED_IO_GAP;
            const height = memberAreaH + EXPANDED_IO_GAP + COLLAPSE_BUTTON_H + COLLAPSE_BOTTOM_PADDING;
            return { cols, pillW, memberRows, height };
        };
        const calcIOGroupExpandedHeight = (ioGroup, availableW = provisionalSvgW) => {
            const memberCount = (ioGroup.member_ids || []).length;
            return computeIOGroupExpandedLayout(memberCount, availableW).height;
        };
        const calcIORowWidth = (row, availableW = provisionalSvgW) => {
            if (!row.items || row.items.length === 0) return 0;
            let width = 0;
            let pendingInline = 0;
            const flushInline = () => {
                if (pendingInline > 0) {
                    width = Math.max(width, pendingInline * ioW + (pendingInline - 1) * pillGap);
                    pendingInline = 0;
                }
            };
            for (const item of row.items) {
                if (item.isIOGroup === true) {
                    const ioGroup = item.ioGroup;
                    const isCollapsed = (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
                    if (!isCollapsed) {
                        flushInline();
                        const memberCount = (ioGroup.member_ids || []).length;
                        const expandedW = memberCount > 0
                            ? Math.min(availableW * 0.85, memberCount * (ioW + pillGap) - pillGap)
                            : 0;
                        width = Math.max(width, expandedW);
                        continue;
                    }
                }
                pendingInline += 1;
            }
            flushInline();
            return width;
        };
        const maxIORowW = allIORows.length > 0 ? Math.max(...allIORows.map(row => calcIORowWidth(row))) : 0;
        const svgW = Math.max(provisionalSvgW, maxIORowW + 80, minIOTotalW);
        const calcIORowHeight = (row) => {
            if (!row.items || row.items.length === 0) return 0;
            const expandedGroups = row.items.filter(item => {
                if (item.isIOGroup !== true) return false;
                const ioGroup = item.ioGroup;
                const isCollapsed = (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
                return !isCollapsed;
            });
            const hasInlineRow = row.items.some(item => {
                if (item.isIOGroup !== true) return true;
                const ioGroup = item.ioGroup;
                const isCollapsed = (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
                return isCollapsed;
            });
            let height = hasInlineRow ? ioH : 0;
            if (expandedGroups.length > 0) {
                const groupCount = expandedGroups.length;
                const colW = (svgW - (groupCount - 1) * EXPAND_COL_GAP) / groupCount;
                const expandedH = Math.max(...expandedGroups.map(item => calcIOGroupExpandedHeight(item.ioGroup, colW)));
                height += (height > 0 ? ioGap : 0) + expandedH;
            }
            return height;
        };
        const topIOHeight = topIORows.length > 0 ? topIORows.reduce((acc, row, idx) => acc + calcIORowHeight(row) + (idx > 0 ? ioGap : 0), 0) : 0;
        const bottomIOHeight = bottomIORows.length > 0 ? bottomIORows.reduce((acc, row, idx) => acc + calcIORowHeight(row) + (idx > 0 ? ioGap : 0), 0) : 0;
        const rootStartY = 30 + (topIOHeight > 0 ? topIOHeight + ioGap : 0);
        const svgH = 30 + topIOHeight + (topIOHeight > 0 ? ioGap : 0) + maxRootH + (bottomIOHeight > 0 ? ioGap + bottomIOHeight : 0) + 40;
        assertActiveRenderGeneration(generation, '布局阶段');
        setRenderProgress(25, '正在计算 DAG 布局…');
        await nextFrame();

        const svg = document.getElementById('dag-svg');
        svg.onclick = () => clearEdgeFocus();
        svg.setAttribute('width', svgW);
        svg.setAttribute('height', svgH);
        svg.innerHTML = `<g id="edge-overlay-layer"></g>`;
        edgeOverlayLayer = document.getElementById('edge-overlay-layer');
        if (!edgeOverlayLayer || edgeOverlayLayer.tagName !== 'g') {
            edgeOverlayLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            edgeOverlayLayer.setAttribute('id', 'edge-overlay-layer');
            svg.appendChild(edgeOverlayLayer);
        }
        clearActiveEdgeOverlay();
        assertActiveRenderGeneration(generation, 'SVG 初始化阶段');
        setRenderProgress(30, '正在初始化画布…');
        await nextFrame();

        const rootPositions = [];
        if (rootSizes.length > 0) {
            const rs = rootSizes[0];
            const rx = (svgW - rs.w) / 2;
            const ry = rootStartY;
            rootPositions.push({ id: rs.id, x: rx, y: ry, w: rs.w, h: rs.h });
        }

        function appendGroupInfoButton(g, cx, cy, titleText = '') {
            if (!g || !g.src_file) return;
            const hit = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            hit.setAttribute('cx', cx);
            hit.setAttribute('cy', cy - 1);
            hit.setAttribute('r', 9);
            hit.setAttribute('class', 'group-info-hit');
            hit.addEventListener('click', (e) => {
                e.stopPropagation();
                showSourcePanel(g);
            });
            svg.appendChild(hit);

            const info = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            info.setAttribute('x', cx);
            info.setAttribute('y', cy - 1);
            info.setAttribute('class', 'group-toggle group-info-text');
            info.textContent = 'i';
            if (titleText) {
                const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                titleEl.textContent = titleText;
                info.appendChild(titleEl);
            }
            svg.appendChild(info);
        }

        __RENDER_GROUP_JS_PLACEHOLDER__

        function renderNodeAt(nid, nx, ny, w, h) {
            const n = nodeMap[nid];
            if (!n) return;
            const color = getNodeColor(n);

            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', nx); rect.setAttribute('y', ny);
            rect.setAttribute('width', w); rect.setAttribute('height', h);
            rect.setAttribute('class', 'leaf-node');
            rect.setAttribute('fill', color);
            rect.setAttribute('stroke', 'rgba(255,255,255,0.15)');
            rect.style.cursor = 'pointer';
            rect.dataset.nid = nid;
            rect.addEventListener('mouseenter', (e) => showTooltip(e, n));
            rect.addEventListener('mouseleave', hideTooltip);
            rect.addEventListener('click', (e) => { e.stopPropagation(); showSourcePanel(n); });
            svg.appendChild(rect);
            registerNodeDom(nid, rect);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', nx + w/2); label.setAttribute('y', ny + h/2);
            label.setAttribute('class', 'node-label');
            label.style.pointerEvents = 'none';
            label.textContent = n.class_name;
            svg.appendChild(label);

            let subText = '';
            if (n.has_timing) subText = `${n.pct.toFixed(1)}%`;
            if (subText) {
                const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                sub.setAttribute('x', nx + w/2); sub.setAttribute('y', ny + h/2 + 11);
                sub.setAttribute('class', 'node-sublabel');
                sub.style.pointerEvents = 'none';
                sub.textContent = subText;
                svg.appendChild(sub);
            }

            nodePortMap[nid] = { cx: nx + w/2, cy: ny + h/2 };
            nodePortMap[nid + '__in'] = { cx: nx + w/2, cy: ny };
            nodePortMap[nid + '__out'] = { cx: nx + w/2, cy: ny + h };
        }

        function configureLongEdgeDisplay(path) {
            if (!path) return;
            let total = NaN;
            try {
                total = path.getTotalLength();
            } catch (err) {
                total = NaN;
            }
            if (!Number.isFinite(total) || total < LONG_EDGE_MIN_SPAN) return;
            const originalDasharray = path.getAttribute('stroke-dasharray') || '';
            const originalDashoffset = path.getAttribute('stroke-dashoffset') || '';
            const stubLen = Math.max(40, Math.min(72, total * 0.12));
            const gapLen = Math.max(80, total - stubLen * 2);
            path.dataset.longEdge = '1';
            path.dataset.fullDasharray = originalDasharray;
            path.dataset.fullDashoffset = originalDashoffset;
            path.dataset.truncatedDasharray = `${stubLen} ${gapLen} ${stubLen}`;
            if (originalDashoffset) {
                path.dataset.truncatedDashoffset = originalDashoffset;
            }
            path.classList.add('long-edge');
            path.setAttribute('stroke-dasharray', path.dataset.truncatedDasharray);
        }

        function buildDirectEdgePath(x1, y1, x2, y2, routeMeta) {
            const dy = y2 - y1;
            const dx = x2 - x1;
            if (Math.abs(dy) < 3 && Math.abs(dx) < 3) return null;
            const meta = routeMeta || {};
            const offset = meta.bundleOffset || 0;
            const sourceFanout = meta.sourceFanout || 1;
            const targetFanin = meta.targetFanin || 1;
            const sideBias = offset === 0 ? (dx >= 0 ? 1 : -1) : (offset > 0 ? 1 : -1);
            let d;
            if (dy > 8) {
                const cp = Math.max(24, Math.min(Math.abs(dy) * 0.34 + Math.abs(offset) * 0.7, 96));
                const c1x = x1 + offset;
                const c2x = x2 + offset;
                d = `M${x1},${y1} C${c1x},${y1+cp} ${c2x},${y2-cp} ${x2},${y2}`;
            } else if (dy < -8) {
                const horizontal = sideBias * Math.max(52, Math.abs(dx) * 0.42 + 34 + Math.abs(offset));
                const rise = Math.max(18, Math.min(72, Math.abs(offset) + 24));
                d = `M${x1},${y1} C${x1+horizontal},${y1-rise} ${x2+horizontal},${y2+rise} ${x2},${y2}`;
            } else {
                const midX = (x1 + x2) / 2 + offset;
                const midY = (y1 + y2) / 2 + 14 + Math.abs(offset) * 0.28;
                d = `M${x1},${y1} Q${midX},${midY} ${x2},${y2}`;
            }
            return { d };
        }

        function buildIntraGroupEdgePath(fr, to, routeCtx, edgeData) {
            const { groupLeft, groupRight, childLeftEdge, childRightEdge, skipLaneCounter } = routeCtx;
            const x1 = fr.x + fr.w / 2, y1 = fr.y + fr.h;
            const x2 = to.x + to.w / 2, y2 = to.y;
            const fromRank = fr.rank, toRank = to.rank;
            const routeMeta = EDGE_BUNDLE_META.get(edgeKey(edgeData)) || null;
            if (toRank < 0) {
                return buildDirectEdgePath(x1, y1, x2, y2, routeMeta);
            }
            const span = Math.abs(toRank - fromRank);

            if (span <= 1 && y2 > y1) {
                return buildDirectEdgePath(x1, y1, x2, y2, routeMeta);
            }
            if (span === 0) {
                return {
                    d: `M${x1},${y1} C${x1},${y1+14} ${x2},${y2+14} ${x2},${y2}`,
                };
            }

            const groupMidX = (childLeftEdge + childRightEdge) / 2;
            const avgX = (x1 + x2) / 2;
            const preferRight = avgX >= groupMidX;
            let laneIndex;
            if (preferRight) {
                laneIndex = skipLaneCounter.right++;
            } else {
                laneIndex = skipLaneCounter.left++;
            }
            const laneStep = 9;
            const laneX = preferRight
                ? Math.min(groupRight - 4, childRightEdge + 14 + (laneIndex % 4) * laneStep)
                : Math.max(groupLeft + 4, childLeftEdge - 14 - (laneIndex % 4) * laneStep);

            const verticalApproach = 18;
            const d = [
                `M${x1},${y1}`,
                `C${x1},${y1 + verticalApproach} ${laneX},${y1 + verticalApproach} ${laneX},${y1 + verticalApproach + 10}`,
                `L${laneX},${y2 - verticalApproach - 10}`,
                `C${laneX},${y2 - verticalApproach} ${x2},${y2 - verticalApproach} ${x2},${y2}`
            ].join(' ');
            return { d, opacity: '0.7' };
        }

        function createEdgePathElement(d) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', d);
            return path;
        }

        function getEdgeArrowColor(type) {
            if (type === 'dep') return 'rgba(46,204,113,0.6)';
            if (type === 'internal') return 'rgba(100,181,246,0.5)';
            return 'rgba(255,255,255,0.3)';
        }

        function makeArrowhead(path, color) {
            if (!path) throw new Error('makeArrowhead requires a path element');
            const len = path.getTotalLength();
            const p2 = path.getPointAtLength(len);
            const p1 = path.getPointAtLength(Math.max(0, len - 8));
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const angle = Math.atan2(dy, dx);
            const L = 8;
            const W = 3;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            const pts = [
                [p2.x, p2.y],
                [p2.x - L * cos + W * sin, p2.y - L * sin - W * cos],
                [p2.x - L * cos - W * sin, p2.y - L * sin + W * cos],
            ].map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
            const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            poly.setAttribute('points', pts);
            poly.setAttribute('fill', color);
            poly.setAttribute('pointer-events', 'none');
            return poly;
        }

        function syncArrowPolygonStyle(path) {
            if (!path || !path._arrowPoly) return;
            const poly = path._arrowPoly;
            const computed = window.getComputedStyle(path);
            poly.setAttribute('fill', computed.stroke || path.dataset.arrowColor || 'rgba(255,255,255,0.3)');
            poly.setAttribute('opacity', computed.opacity || path.getAttribute('opacity') || '1');
            const display = computed.display === 'none' ? 'none' : '';
            poly.style.display = display;
            poly.style.visibility = computed.visibility === 'hidden' ? 'hidden' : 'visible';
        }

        function createEdgeHitboxElement(d, key) {
            const hitbox = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            hitbox.setAttribute('d', d);
            hitbox.setAttribute('stroke', 'transparent');
            hitbox.setAttribute('stroke-width', '20');
            hitbox.setAttribute('fill', 'none');
            hitbox.setAttribute('pointer-events', 'stroke');
            hitbox.setAttribute('opacity', '0');
            hitbox.setAttribute('data-edge-key', key);
            return hitbox;
        }

        function finalizeEdgeRendering(path, hitbox) {
            svg.appendChild(path);
            if (path._arrowPoly) svg.appendChild(path._arrowPoly);
            if (hitbox) svg.appendChild(hitbox);
            configureLongEdgeDisplay(path);
            syncArrowPolygonStyle(path);
        }

        function renderEdge(edgeSpec) {
            let pathSpec = null;
            if (edgeSpec.routingMode === 'direct') {
                pathSpec = buildDirectEdgePath(edgeSpec.x1, edgeSpec.y1, edgeSpec.x2, edgeSpec.y2, edgeSpec.routeMeta);
            } else if (edgeSpec.routingMode === 'intra_group') {
                pathSpec = buildIntraGroupEdgePath(edgeSpec.fr, edgeSpec.to, edgeSpec.routeCtx, edgeSpec.edgeData);
            } else {
                throw new Error(`Unknown edge routing mode: ${edgeSpec.routingMode}`);
            }
            if (!pathSpec) return;
            const key = edgeKey(edgeSpec.edgeData);
            const path = createEdgePathElement(pathSpec.d);
            const hitbox = createEdgeHitboxElement(pathSpec.d, key);
            applyEdgePresentation(path, edgeSpec.type, pathSpec.opacity ?? null);
            bindEdgeInteractions(path, hitbox, edgeSpec.edgeData);
            finalizeEdgeRendering(path, hitbox);
        }

        function registerEdgeDom(path, edgeData) {
            if (!path || !edgeData) return;
            const item = { path, edge: edgeData, key: edgeKey(edgeData) };
            edgeDomRegistry.push(item);
            edgeDomByKey.set(item.key, item);
            for (const nid of [edgeData.from, edgeData.to]) { if (nid == null) continue; if (!edgeByNodeId.has(nid)) edgeByNodeId.set(nid, []); edgeByNodeId.get(nid).push(item); }
            for (const nid of [edgeData.from, edgeData.to]) { if (nid == null) continue; for (const gid of (nodeAncestorGroups.get(nid) ?? [])) { if (!edgeByGroupId.has(gid)) edgeByGroupId.set(gid, []); edgeByGroupId.get(gid).push(item); } }
        }

        function applyEdgePresentation(path, type, opacity=null) {
            path.setAttribute('class', `edge-path ${type || ''}`);
            path.setAttribute('style', 'pointer-events: none');
            path.dataset.arrowColor = getEdgeArrowColor(type);
            path._arrowPoly = makeArrowhead(path, path.dataset.arrowColor);
            if (opacity !== null) {
                path.setAttribute('opacity', opacity);
            }
            syncArrowPolygonStyle(path);
        }

        function showOverlapBadge(e, idx, total) {
            let badge = document.getElementById('overlap-badge');
            if (!badge) {
                badge = document.createElement('div');
                badge.id = 'overlap-badge';
                badge.style.cssText = 'position:fixed;background:rgba(30,30,40,0.92);color:#e0e0e0;'
                    + 'font-size:12px;padding:3px 8px;border-radius:4px;pointer-events:none;'
                    + 'z-index:9999;white-space:nowrap;border:1px solid rgba(255,255,255,0.15);';
                document.body.appendChild(badge);
            }
            badge.textContent = `${idx + 1}/${total}  ·  scroll to switch`;
            badge.style.left = (e.clientX + 14) + 'px';
            badge.style.top  = (e.clientY - 10) + 'px';
            badge.style.display = 'block';
        }
        function hideOverlapBadge() {
            const badge = document.getElementById('overlap-badge');
            if (badge) badge.style.display = 'none';
        }
        function updateOverlapBadge() {
            const badge = document.getElementById('overlap-badge');
            if (!badge || hoveredEdges.length <= 1) return;
            badge.textContent = `${hoveredEdgeIdx + 1}/${hoveredEdges.length}  ·  scroll to switch`;
        }

        function getHoveredEdgesFromEvent(e) {
            const hitEls = document.elementsFromPoint(e.clientX, e.clientY);
            const seen = new Set();
            return hitEls
                .filter(el => el && typeof el.hasAttribute === 'function' && el.hasAttribute('data-edge-key'))
                .map(el => el.getAttribute('data-edge-key'))
                .filter(key => {
                    if (!key || seen.has(key)) return false;
                    seen.add(key);
                    return true;
                })
                .map(key => edgeDomByKey.get(key))
                .filter(Boolean);
        }

        function bindEdgeInteractions(path, hitbox, edgeData) {
            if (!path || !edgeData || !hitbox) return;
            const key = edgeKey(edgeData);
            const hoverTargets = [path, hitbox];
            path._hitbox = hitbox;

            const handleMouseEnter = (e) => {
                hoveredEdges = getHoveredEdgesFromEvent(e);
                hoveredEdgeIdx = hoveredEdges.findIndex(item => item.key === key);
                if (hoveredEdgeIdx < 0) hoveredEdgeIdx = 0;
                hoveredEdgeKey = hoveredEdges.length > 0 ? hoveredEdges[hoveredEdgeIdx].key : key;
                applyEdgeFocusState();
                if (hoveredEdges.length > 1) showOverlapBadge(e, hoveredEdgeIdx, hoveredEdges.length);
                const handleMouseMove = (ev) => {
                    const badge = document.getElementById('overlap-badge');
                    if (badge && badge.style.display !== 'none') {
                        badge.style.left = (ev.clientX + 14) + 'px';
                        badge.style.top  = (ev.clientY - 10) + 'px';
                    }
                };
                for (const target of hoverTargets) {
                    target._overlapMoveHandler = handleMouseMove;
                    target.addEventListener('mousemove', handleMouseMove);
                }
            };

            const handleMouseLeave = () => {
                for (const target of hoverTargets) {
                    if (target._overlapMoveHandler) {
                        target.removeEventListener('mousemove', target._overlapMoveHandler);
                        target._overlapMoveHandler = null;
                    }
                }
                hoveredEdges = [];
                hoveredEdgeIdx = 0;
                hoveredEdgeKey = null;
                hideOverlapBadge();
                applyEdgeFocusState();
            };

            const handleWheel = (e) => {
                if (hoveredEdges.length <= 1) return;
                e.preventDefault();
                hoveredEdgeIdx = (hoveredEdgeIdx + (e.deltaY > 0 ? 1 : -1) + hoveredEdges.length) % hoveredEdges.length;
                hoveredEdgeKey = hoveredEdges[hoveredEdgeIdx].key;
                applyEdgeFocusState();
                updateOverlapBadge();
                showEdgePanel(hoveredEdges[hoveredEdgeIdx].edge);
            };

            const handleClick = (e) => {
                e.stopPropagation();
                const alreadyActive = focusedEdgePath === key;
                if (alreadyActive) {
                    focusedEdgePath = null;
                    applyEdgeFocusState();
                } else {
                    setEdgeFocus(key);
                }
                showEdgePanel(edgeData);
            };

            for (const target of hoverTargets) {
                target.addEventListener('mouseenter', handleMouseEnter);
                target.addEventListener('mouseleave', handleMouseLeave);
                target.addEventListener('wheel', handleWheel, { passive: false });
                target.addEventListener('click', handleClick);
            }
            registerEdgeDom(path, edgeData);
        }

        function renderGroupTask(task) {
            if (task.type === 'node') {
                renderNodeAt(task.nid, task.x, task.y, task.w, task.h);
                return;
            }
            if (task.type !== 'group') {
                throw new Error(`renderGroupTask got unsupported task type: ${task.type}`);
            }
            const ctx = task.ctx;
            if (!ctx || !ctx.g || !ctx.pos) {
                throw new Error(`renderGroupTask got invalid context for task: ${task.taskKind || '<missing>'}`);
            }
            if (task.taskKind === 'collapsed_group') {
                RENDER_GROUP_EXPORTS.renderCollapsedGroupBox(ctx);
                RENDER_GROUP_EXPORTS.renderCollapsedGroupLabel(ctx);
                RENDER_GROUP_EXPORTS.renderCollapsedGroupTiming(ctx);
                RENDER_GROUP_EXPORTS.renderCollapsedGroupInfoButton(ctx);
                RENDER_GROUP_EXPORTS.registerCollapsedGroupPorts(ctx);
                return;
            }
            if (task.taskKind === 'expanded_group_shell') {
                RENDER_GROUP_EXPORTS.renderExpandedGroupBox(ctx);
                const { headerText } = RENDER_GROUP_EXPORTS.renderExpandedGroupHeaderLabel(ctx);
                RENDER_GROUP_EXPORTS.renderExpandedGroupInfoButton(ctx, headerText);
                RENDER_GROUP_EXPORTS.renderExpandedGroupTiming(ctx);
                return;
            }
            if (task.taskKind === 'expanded_group_ports') {
                RENDER_GROUP_EXPORTS.registerExpandedGroupPorts(ctx);
                return;
            }
            throw new Error(`renderGroupTask got unknown taskKind: ${task.taskKind}`);
        }

        function collectGroupRenderTasks(items) {
            const tasks = [];
            function visitGroup(gid, ox, oy) {
                const ctx = RENDER_GROUP_EXPORTS.getGroupRenderContext(gid, ox, oy);
                if (!ctx || !ctx.pos) {
                    throw new Error(`collectGroupRenderTasks missing layout for group: ${gid}`);
                }
                if (ctx.pos.collapsed) {
                    tasks.push({ type: 'group', taskKind: 'collapsed_group', ctx });
                    return;
                }
                tasks.push({ type: 'group', taskKind: 'expanded_group_shell', ctx });
                for (const child of (ctx.pos.childPositions || [])) {
                    if (child.type === 'node') {
                        tasks.push({
                            type: 'node',
                            taskKind: 'node',
                            nid: child.id,
                            x: ctx.ox + child.x,
                            y: ctx.oy + child.y,
                            w: child.w,
                            h: child.h,
                        });
                    } else if (child.type === 'group') {
                        visitGroup(child.id, ctx.ox + child.x, ctx.oy + child.y);
                    } else {
                        throw new Error(`collectGroupRenderTasks got unknown child type: ${child.type}`);
                    }
                }
                tasks.push({ type: 'group', taskKind: 'expanded_group_ports', ctx });
            }
            for (const item of items) {
                visitGroup(item.id, item.x, item.y);
            }
            return tasks;
        }

        function collectGlobalEdgeTasks() {
            const tasks = [];
            for (const edge of (DATA.edges || [])) {
                if (!isEdgeVisible(edge)) {
                    continue;
                }
                tasks.push({ type: 'edge', taskKind: 'global_edge', edgeData: edge });
            }
            return tasks;
        }

        function renderIOPill(nid, cx, cy, w, h, label, sublabel, fillColor) {
            const n = nodeMap[nid];
            const x = cx - w/2;
            const y = cy - h/2;
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x); rect.setAttribute('y', y);
            rect.setAttribute('width', w); rect.setAttribute('height', h);
            rect.setAttribute('rx', h / 2); rect.setAttribute('ry', h / 2);
            rect.setAttribute('class', 'io-node');
            rect.setAttribute('fill', fillColor);
            rect.setAttribute('stroke', 'rgba(255,255,255,0.35)');
            rect.setAttribute('stroke-width', '1.5');
            if (n) {
                rect.dataset.nid = nid;
                rect.addEventListener('mouseenter', (e) => showTooltip(e, n));
                rect.addEventListener('mouseleave', hideTooltip);
                bindIONodeHover(rect, nid);
            }
            svg.appendChild(rect);
            registerNodeDom(nid, rect);

            const lab = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            lab.setAttribute('x', cx); lab.setAttribute('y', sublabel ? cy - 2 : cy + 4);
            lab.setAttribute('class', 'node-label');
            lab.setAttribute('font-weight', '700');
            lab.textContent = label;
            svg.appendChild(lab);

            if (sublabel) {
                const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                sub.setAttribute('x', cx); sub.setAttribute('y', cy + 11);
                sub.setAttribute('class', 'node-sublabel');
                sub.textContent = sublabel;
                svg.appendChild(sub);
            }

            nodePortMap[nid] = { cx, cy };
            nodePortMap[nid + '__in'] = { cx, cy: y };
            nodePortMap[nid + '__out'] = { cx, cy: y + h };
        }

        const IO_GROUP_FILL = {
            input: 'rgba(46,204,113,0.55)',
            param: 'rgba(155,89,182,0.55)',
            const: 'rgba(241,196,15,0.55)',
            output: 'rgba(231,76,60,0.55)',
        };
        const IO_GROUP_MEMBER_LABEL = { input: 'Input', param: 'Param', const: 'Const', output: 'Result' };

        function renderIOGroupPill(ioGroup, cx, cy, w, h, availableW = svgW) {
            const fillColor = IO_GROUP_FILL[ioGroup.io_subtype] || 'rgba(127,140,141,0.55)';
            const memberLabel = IO_GROUP_MEMBER_LABEL[ioGroup.io_subtype] || ioGroup.io_subtype;
            const isCollapsed = (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
            const renderCollapseButton = (bcx, bcy) => {
                const bx = bcx - COLLAPSE_BUTTON_W / 2;
                const by = bcy - COLLAPSE_BUTTON_H / 2;
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', bx); rect.setAttribute('y', by);
                rect.setAttribute('width', COLLAPSE_BUTTON_W); rect.setAttribute('height', COLLAPSE_BUTTON_H);
                rect.setAttribute('rx', COLLAPSE_BUTTON_H / 2); rect.setAttribute('ry', COLLAPSE_BUTTON_H / 2);
                rect.setAttribute('class', 'io-node io-group');
                rect.setAttribute('fill', 'transparent');
                rect.setAttribute('stroke', 'rgba(255,255,255,0.45)');
                rect.setAttribute('stroke-width', '1.5');
                rect.style.cursor = 'pointer';
                rect.dataset.ioGroupId = ioGroup.id;
                rect.addEventListener('dblclick', (e) => {
                    e.stopPropagation();
                    collapsedState[ioGroup.id] = true;
                    invokeRender();
                });
                svg.appendChild(rect);

                const lab = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                lab.setAttribute('x', bcx); lab.setAttribute('y', bcy + 4);
                lab.setAttribute('class', 'node-label');
                lab.setAttribute('font-weight', '700');
                lab.textContent = '▲ 收起';
                lab.style.pointerEvents = 'none';
                svg.appendChild(lab);
            };
            if (!isCollapsed) {
                const members = ioGroup.member_ids || [];
                if (members.length === 0) return 0;
                const { cols, pillW, memberRows, height: expandedHeight } = computeIOGroupExpandedLayout(members.length, availableW);
                const startY = cy - EXPANDED_IO_H / 2;
                const frameX = cx - availableW / 2 + EXPAND_FRAME_PAD - 8;
                const frameY = startY - 8;
                const frameW = availableW - 2 * (EXPAND_FRAME_PAD - 8);
                const frameH = expandedHeight + 8;
                const frame = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                frame.setAttribute('x', frameX); frame.setAttribute('y', frameY);
                frame.setAttribute('width', frameW); frame.setAttribute('height', frameH);
                frame.setAttribute('rx', 10); frame.setAttribute('ry', 10);
                frame.setAttribute('fill', 'rgba(255,255,255,0.03)');
                frame.setAttribute('stroke', fillColor);
                frame.setAttribute('stroke-width', '1.5');
                frame.setAttribute('stroke-dasharray', '5,4');
                svg.appendChild(frame);

                const truncateSublabel = (text) => {
                    const limit = Math.max(1, Math.floor(pillW / 6.5));
                    return text.length > limit ? `${text.slice(0, Math.max(0, limit - 1))}…` : text;
                };
                for (let rowIdx = 0; rowIdx < memberRows; rowIdx++) {
                    const first = rowIdx * cols;
                    const rowMemberCount = Math.max(0, Math.min(cols, members.length - first));
                    const rowWidth = rowMemberCount * pillW + (rowMemberCount - 1) * pillGap;
                    let left = cx - rowWidth / 2;
                    const rowCy = startY + rowIdx * (EXPANDED_IO_H + EXPANDED_IO_GAP) + EXPANDED_IO_H / 2;
                    for (let idx = 0; idx < rowMemberCount; idx++) {
                        const memberId = members[first + idx];
                        const node = nodeMap[memberId];
                        const baseText = node ? node.class_name : memberLabel;
                        const sublabel = (node && node.has_timing)
                            ? `${baseText} · ${node.pct.toFixed(1)}%`
                            : baseText;
                        renderIOPill(memberId, left + pillW / 2, rowCy, pillW, EXPANDED_IO_H, memberLabel, truncateSublabel(sublabel), fillColor);
                        left += pillW + pillGap;
                    }
                }
                const collapseRowCy = frameY + frameH - COLLAPSE_BOTTOM_PADDING - COLLAPSE_BUTTON_H / 2;
                renderCollapseButton(cx, collapseRowCy);
                return expandedHeight;
            }
            const x = cx - w / 2;
            const y = cy - h / 2;
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x); rect.setAttribute('y', y);
            rect.setAttribute('width', w); rect.setAttribute('height', h);
            rect.setAttribute('rx', h / 2); rect.setAttribute('ry', h / 2);
            rect.setAttribute('class', 'io-node io-group');
            rect.setAttribute('fill', fillColor);
            rect.setAttribute('stroke', 'rgba(255,255,255,0.35)');
            rect.setAttribute('stroke-width', '1.5');
            rect.style.cursor = 'pointer';
            rect.dataset.ioGroupId = ioGroup.id;
            rect.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                collapsedState[ioGroup.id] = false;
                invokeRender();
            });
            svg.appendChild(rect);

            const lab = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            lab.setAttribute('x', cx); lab.setAttribute('y', cy + 4);
            lab.setAttribute('class', 'node-label');
            lab.setAttribute('font-weight', '700');
            lab.textContent = `▶ ${ioGroup.label}`;
            lab.style.pointerEvents = 'none';
            svg.appendChild(lab);

            nodePortMap[ioGroup.id] = { cx, cy };
            nodePortMap[ioGroup.id + '__in'] = { cx, cy: y };
            nodePortMap[ioGroup.id + '__out'] = { cx, cy: y + h };
            return h;
        }

        function collectIORenderTasks(row, startY) {
            if (!row || !row.items || row.items.length === 0) return { tasks: [], height: 0 };
            let y = startY;
            const tasks = [];
            const inlineItems = [];
            const expandedItems = [];
            for (const item of row.items) {
                if (item.isIOGroup === true) {
                    const ioGroup = item.ioGroup;
                    const isCollapsed = (ioGroup.id in collapsedState) ? collapsedState[ioGroup.id] : ioGroup.collapsed;
                    if (!isCollapsed) {
                        expandedItems.push(item);
                        continue;
                    }
                }
                inlineItems.push(item);
            }

            if (inlineItems.length > 0) {
                const rowWidth = inlineItems.length * ioW + (inlineItems.length - 1) * pillGap;
                let left = (svgW - rowWidth) / 2;
                const cy = y + ioH / 2;
                for (const item of inlineItems) {
                    if (item.isIOGroup === true) {
                        tasks.push({ type: 'io', taskKind: 'io_group', ioGroup: item.ioGroup, cx: left + ioW / 2, cy, w: ioW, h: ioH, availableW: svgW });
                    } else {
                        const nid = item.id;
                        const node = nodeMap[nid];
                        const baseText = node ? node.class_name : item.defaultSublabel;
                        const sublabel = (node && node.has_timing)
                            ? `${baseText} · ${node.pct.toFixed(1)}%`
                            : baseText;
                        tasks.push({ type: 'io', taskKind: 'io_pill', nid, cx: left + ioW / 2, cy, w: ioW, h: ioH, label: item.label, sublabel, fillColor: item.fillColor });
                    }
                    left += ioW + pillGap;
                }
                y += ioH;
            }

            if (expandedItems.length > 0) {
                if (y > startY) y += ioGap;
                const groupCount = expandedItems.length;
                const colW = (svgW - (groupCount - 1) * EXPAND_COL_GAP) / groupCount;
                const expandedHeight = Math.max(...expandedItems.map(item => calcIOGroupExpandedHeight(item.ioGroup, colW)));
                let left = 0;
                for (const item of expandedItems) {
                    tasks.push({ type: 'io', taskKind: 'io_group', ioGroup: item.ioGroup, cx: left + colW / 2, cy: y + EXPANDED_IO_H / 2, w: ioW, h: ioH, availableW: colW });
                    left += colW + EXPAND_COL_GAP;
                }
                y += expandedHeight;
            }
            return { tasks, height: y - startY };
        }

        function renderIOTask(task) {
            if (task.type !== 'io') {
                throw new Error(`renderIOTask got unsupported task type: ${task.type}`);
            }
            if (task.taskKind === 'io_group') {
                renderIOGroupPill(task.ioGroup, task.cx, task.cy, task.w, task.h, task.availableW);
                return;
            }
            if (task.taskKind === 'io_pill') {
                renderIOPill(task.nid, task.cx, task.cy, task.w, task.h, task.label, task.sublabel, task.fillColor);
                return;
            }
            throw new Error(`renderIOTask got unknown taskKind: ${task.taskKind}`);
        }

        const groupTasks = collectGroupRenderTasks(rootPositions);
        const ioTasks = [];
        let topIOY = 30;
        for (const row of topIORows) {
            const plan = collectIORenderTasks(row, topIOY);
            ioTasks.push(...plan.tasks);
            topIOY += plan.height + ioGap;
        }
        const rootEntry = rootPositions[0];
        let bottomIOY = rootEntry ? (rootEntry.y + rootEntry.h + ioGap) : (rootStartY + maxRootH + ioGap);
        for (const row of bottomIORows) {
            const plan = collectIORenderTasks(row, bottomIOY);
            ioTasks.push(...plan.tasks);
            bottomIOY += plan.height + ioGap;
        }

        await runChunked(groupTasks.concat(ioTasks), async (task) => {
            if (task.type === 'io') {
                renderIOTask(task);
                return;
            }
            renderGroupTask(task);
        }, {
            batchSize: 80,
            phaseStart: 30,
            phaseEnd: 60,
            stageText: '正在渲染模块节点…',
            generation,
            allowedTypes: ['group', 'node', 'io'],
        });

        const globalEdgeTasks = collectGlobalEdgeTasks();

        await runChunked(globalEdgeTasks, async (task) => {
            if (task.type !== 'edge') {
                throw new Error(`edge renderer got unsupported task type: ${task.type}`);
            }
            if (task.taskKind === 'global_edge') {
                const fromId = resolveCollapsedAncestor(task.edgeData.from);
                const toId = resolveCollapsedAncestor(task.edgeData.to);
                if (fromId === toId) {
                    return;
                }
                const fromPos = nodePortMap[fromId + '__out'] || nodePortMap[fromId];
                const toPos = nodePortMap[toId + '__in'] || nodePortMap[toId];
                if (!fromPos || !toPos) {
                    throw new Error(`global edge endpoint missing: ${task.edgeData.from} -> ${task.edgeData.to}`);
                }
                renderEdge({
                    routingMode: 'direct',
                    x1: fromPos.cx, y1: fromPos.cy,
                    x2: toPos.cx, y2: toPos.cy,
                    type: task.edgeData.type || 'dep',
                    edgeData: task.edgeData,
                    routeMeta: EDGE_BUNDLE_META.get(edgeKey(task.edgeData)) || null,
                });
                return;
            }
            throw new Error(`edge renderer got unknown taskKind: ${task.taskKind}`);
        }, {
            batchSize: 180,
            phaseStart: 60,
            phaseEnd: 90,
            stageText: '正在渲染依赖边…',
            generation,
            allowedTypes: ['edge'],
        });
        if (!edgeOverlayLayer) {
            throw new Error('edge overlay layer missing after edge render');
        }
        svg.appendChild(edgeOverlayLayer);

        setRenderProgress(90, '正在更新图例和摘要…');
        await nextFrame();
        const meta = DATA.meta;
        if (DATA.has_timing) {
            document.getElementById('mode-badge').innerHTML = '<span class="mode-badge mode-timing">📊 Structure + Timing</span>';
            document.getElementById('meta-info').textContent = `Device: ${meta.device} | Step: ${meta.step_dur_str} | Modules: ${meta.num_modules}`;
        } else {
            document.getElementById('mode-badge').innerHTML = '<span class="mode-badge mode-structure">🏗️ Static Structure (source code)</span>';
            document.getElementById('meta-info').textContent = `Modules: ${meta.num_modules} | Root: ${meta.roots ? meta.roots.join(", ") : "N/A"}`;
        }

        const legendDiv = document.getElementById('legend');
        if (DATA.has_timing) {
            legendDiv.innerHTML = `
                <div class="legend-item"><div class="legend-dot" style="background:#2980b9"></div>&gt;20%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>10-20%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#8e44ad"></div>5-10%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#5a6c7d"></div>&lt;5%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Worker &gt;20%</div>
                <div class="legend-item"><div class="legend-dot" style="background:#e67e22"></div>Worker 10-20%</div>`;
        } else {
            legendDiv.innerHTML = `
                <div class="legend-item"><div class="legend-dot" style="background:#4a6fa5"></div>Depth 0</div>
                <div class="legend-item"><div class="legend-dot" style="background:#5b8c5a"></div>Depth 1</div>
                <div class="legend-item"><div class="legend-dot" style="background:#8e6fad"></div>Depth 2</div>
                <div class="legend-item"><div class="legend-dot" style="background:#c77a3c"></div>Depth 3+</div>
                <div class="legend-item" style="margin-left: 12px;"><span style="color:#64b5f6">▶</span> Click to expand</div>
                <div class="legend-item" style="margin-left: 12px;"><span style="color:rgba(46,204,113,0.8)">━━▶</span> Data dependency</div>
                <div class="legend-item"><span style="color:rgba(255,255,255,0.3)">╌╌▶</span> Sequential (fallback)</div>`;
        }

        const summaryDiv = document.getElementById('summary');
        const allNodes = DATA.nodes;
        const allGroups = DATA.groups;
        if (DATA.has_timing) {
            const topN = [...allNodes, ...allGroups].filter(x => x.has_timing).sort((a,b) => b.pct - a.pct).slice(0,5);
            summaryDiv.innerHTML = `<h3>📊 Top Modules by Time</h3><p>${
                topN.map(x => `<b>${x.label || x.class_name}</b> ${x.pct.toFixed(1)}%`).join(' → ')
            }</p>`;
        } else {
            summaryDiv.innerHTML = `<h3>🏗️ Architecture Summary</h3><p>Module count: ${allNodes.length + allGroups.length} | Expandable containers: ${allGroups.length} | Leaf nodes: ${allNodes.length}<br><i>Click ▶ collapsed containers to expand. Provide --trace-file for timing overlay.</i></p>`;
        }
        assertActiveRenderGeneration(generation, '收尾阶段');
        setRenderProgress(98, '正在更新图例和摘要…');
        await nextFrame();
        assertActiveRenderGeneration(generation, '完成阶段');
        await hideRenderProgress();
    } catch (err) {
        if (renderGeneration === generation) {
            const progressEls = getRenderProgressElements();
            progressEls.overlay.classList.remove('closing');
            progressEls.overlay.classList.add('visible', 'failed');
            progressEls.overlay.setAttribute('aria-hidden', 'false');
            const lastProgress = Number(progressEls.overlay.dataset.progress || 0);
            setRenderProgress(Number.isFinite(lastProgress) ? Math.min(99, lastProgress) : 0, '渲染失败，请查看 Console 错误');
        }
        throw err;
    }
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
    if (e.key === 'Escape') document.getElementById('side-panel').classList.remove('open');
});

document.getElementById('btn-expand-all').addEventListener('click', () => {
    DATA.groups.forEach(g => collapsedState[g.id] = false);
    invokeRender();
});
document.getElementById('btn-collapse-all').addEventListener('click', () => {
    DATA.groups.forEach(g => { if (g.depth >= 1) collapsedState[g.id] = true; });
    invokeRender();
});
document.getElementById('btn-fit').addEventListener('click', () => {
    const container = document.getElementById('dag-container');
    const svgEl = document.getElementById('dag-svg');
    const sw = parseInt(svgEl.getAttribute('width'));
    const cw = container.clientWidth;
    if (sw > cw) {
        svgEl.style.transform = `scale(${(cw / sw).toFixed(3)})`;
        svgEl.style.transformOrigin = 'top left';
    } else {
        svgEl.style.transform = '';
    }
});

invokeRender();

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


def render_flowchart_to_file(flowchart_data, output_path):
    html_content = _generate_flowchart_html(flowchart_data)
    source_files = _collect_source_files(flowchart_data)
    source_map = _build_source_map(source_files)
    source_map_js = "const SOURCE_MAP = " + _json.dumps(source_map, ensure_ascii=True).replace("</", "<\\/") + ";"
    html_content = html_content.replace("// __SOURCE_MAP_PLACEHOLDER__", source_map_js)
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
    
    source_map = _build_source_map(source_files)
    source_map_js = "const SOURCE_MAP = " + _json.dumps(source_map, ensure_ascii=True).replace("</", "<\\/") + ";"
    html_content = html_content.replace("// __SOURCE_MAP_PLACEHOLDER__", source_map_js)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


def _generate_flowchart_html_multi(tabs: dict[str, list[dict]]) -> str:
    import json as _json
    import re as _re

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
    all_tab_labels = {mode: [str(item["label"]) for item in items] for mode, items in tabs.items()}
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
    base_html, n_sub = _re.subn(
        r"const DATA = .*?(?=\nconst groupMap = \{\};)",
        lambda _m: data_preamble,
        base_html,
        count=1,
        flags=_re.DOTALL,
    )
    if n_sub != 1:
        raise RuntimeError("failed to replace DATA preamble for multi-tab flowchart html")

    tab_styles = (
        "<style>\n"
        ".multi-tabs-l1 {\n"
        "    position: sticky; top: 0; z-index: 200;\n"
        "    display: flex; align-items: center;\n"
        "    height: 48px; padding: 0 16px;\n"
        "    background: #0d1117; border-bottom: 1px solid #21262d;\n"
        "}\n"
        ".multi-tab-l1 {\n"
        "    height: 48px; padding: 0 20px;\n"
        "    background: none; border: none; border-bottom: 2px solid transparent;\n"
        "    color: #8b949e; font-size: 14px; cursor: pointer;\n"
        "    transition: color .15s, border-color .15s;\n"
        "}\n"
        ".multi-tab-l1.active { color: #64b5f6; border-bottom-color: #64b5f6; background: rgba(100,181,246,.07); }\n"
        ".multi-tab-l1:hover:not(.active) { color: #c9d1d9; }\n"
        ".multi-tabs-l2-container {\n"
        "    background: #0d1117; border-bottom: 1px solid #161b22;\n"
        "    padding: 4px 0 4px 32px;\n"
        "}\n"
        ".multi-l1-panel { display: none; }\n"
        ".multi-l1-panel.active { display: block; }\n"
        ".multi-tabs-l2 { display: flex; gap: 6px; flex-wrap: wrap; }\n"
        ".multi-tab-l2 {\n"
        "    height: 28px; padding: 0 12px;\n"
        "    background: #161b22; border: 1px solid #30363d; border-radius: 14px;\n"
        "    color: #8b949e; font-size: 12px; cursor: pointer;\n"
        "    transition: background .15s, color .15s;\n"
        "}\n"
        ".multi-tab-l2.active { background: #1f3a5f; border-color: #388bfd; color: #64b5f6; }\n"
        ".multi-tab-l2:hover:not(.active) { background: #21262d; color: #c9d1d9; }\n"
        "</style>\n"
    )
    if "</head>" in base_html:
        base_html = base_html.replace("</head>", tab_styles + "</head>", 1)
    else:
        base_html = tab_styles + base_html

    l1_buttons: list[str] = []
    l2_panels: list[str] = []
    for idx, (mode, labels) in enumerate(all_tab_labels.items()):
        is_active = idx == 0
        l1_class = "multi-tab-l1 active" if is_active else "multi-tab-l1"
        l1_buttons.append(
            f'<button class="{l1_class}" data-l1="{mode}" onclick="activateL1({mode!r})">{mode.title()}</button>'
        )
        l2_buttons: list[str] = []
        for step_idx, label in enumerate(labels):
            l2_class = "multi-tab-l2 active" if step_idx == 0 else "multi-tab-l2"
            l2_buttons.append(
                f'<button class="{l2_class}" data-l1="{mode}" onclick="activateL2({mode!r}, {step_idx})">{label}</button>'
            )
        panel_class = "multi-l1-panel active" if is_active else "multi-l1-panel"
        l2_panels.append(
            f'<div class="{panel_class}" data-l1-panel="{mode}"><div class="multi-tabs-l2">' + "".join(l2_buttons) + "</div></div>"
        )
    tab_html = (
        '<div class="multi-tabs-l1">' + "".join(l1_buttons) + "</div>\n"
        '<div class="multi-tabs-l2-container">' + "".join(l2_panels) + "</div>\n"
    )
    base_html, body_sub = _re.subn(r"(<body[^>]*>)", r"\1\n" + tab_html, base_html, count=1)
    if body_sub != 1:
        raise RuntimeError("failed to inject multi-tab bar into flowchart html body")

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
        "    if (!document.getElementById(\"dag-svg\")) {\n"
        "        dagContainer.innerHTML = '<svg class=\"dag-svg\" id=\"dag-svg\"></svg>';\n"
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
        "    try { if (typeof nodePortMap !== \"undefined\") { Object.keys(nodePortMap).forEach(k => delete nodePortMap[k]); } } catch (e) {}\n"
        "    try { hoveredEdgeKey = null; hoveredEdges = []; hoveredEdgeIdx = 0; hoveredNodeId = null; hoveredGroupId = null; focusedEdgePath = null; prevActiveItems = []; prevActiveNodeIds = new Set(); prevActiveEdgeKeys = new Set(); prevActiveGroupNodeIds = new Set(); prevActiveGroupIds = new Set(); prevDimEdgeKeys = new Set(); prevDimGroupNodeIds = new Set(); prevDimGroupIds = new Set(); lastHoveredGid = null; } catch (e) {}\n"
        "    try { var sp = document.getElementById(\"side-panel\"); if (sp) sp.classList.remove(\"open\"); } catch (e) {}\n"
        "}\n"
        "function _rebuildIndices() {\n"
        "    DATA.groups.forEach(g => groupMap[g.id] = g);\n"
        "    DATA.nodes.forEach(n => nodeMap[n.id] = n);\n"
        "    if (typeof indexGroupAncestors === \"function\" && DATA.root_groups) {\n"
        "        indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));\n"
        "    }\n"
        "    DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2 || g.is_native === true || g.synthetic_type === 'function_group' || g.synthetic_type === 'callloc_group'; });\n"
        "}\n"
        "function activateL1(mode) {\n"
        "    ACTIVE_L1 = mode;\n"
        "    document.querySelectorAll('.multi-tab-l1').forEach(b => {\n"
        "        b.classList.toggle('active', b.dataset.l1 === mode);\n"
        "    });\n"
        "    document.querySelectorAll('.multi-l1-panel').forEach(p => {\n"
        "        p.classList.toggle('active', p.dataset.l1Panel === mode);\n"
        "    });\n"
        "    document.querySelectorAll(`.multi-tab-l2[data-l1=\"${mode}\"]`).forEach((b, i) => {\n"
        "        b.classList.toggle('active', i === ACTIVE_L2[mode]);\n"
        "    });\n"
        "    switchDataset(ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]]);\n"
        "}\n"
        "function activateL2(mode, idx) {\n"
        "    ACTIVE_L2[mode] = idx;\n"
        "    if (mode === ACTIVE_L1) {\n"
        "        document.querySelectorAll(`.multi-tab-l2[data-l1=\"${mode}\"]`).forEach((b, i) => {\n"
        "            b.classList.toggle('active', i === idx);\n"
        "        });\n"
        "        switchDataset(ALL_TAB_DATA[ACTIVE_L1][ACTIVE_L2[ACTIVE_L1]]);\n"
        "    }\n"
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
    import json as _json
    import re as _re

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
    new_html, n_sub = _re.subn(
        r'const DATA = .*?(?=\nconst groupMap = \{\};)',
        lambda m: dual_preamble,
        base_html,
        count=1,
        flags=_re.DOTALL,
    )
    if n_sub != 1:
        # Fall back to legacy iframe shell (extremely defensive — should never
        # trigger because _generate_flowchart_html always emits the marker).
        print("  ⚠️ 双 Tab: 未在内嵌模板中找到 DATA 注入锚点，回退为单 Tab(train) 输出")
        return base_html
    base_html = new_html

    # ---- 2. inject the tab bar UI right after <body> ------------------------
    summary = lambda d: {
        "nodes": len(d.get("nodes", [])),
        "groups": len(d.get("groups", [])),
        "edges": len(d.get("edges", [])),
    }
    s_train = summary(data_train)
    s_infer = summary(data_infer)

    tab_styles = (
        "<style>\n"
        "  .iter14-tabs { display: flex; align-items: stretch; gap: 0;"
        " padding: 0 16px; background: #14142b;"
        " border-bottom: 1px solid rgba(255,255,255,0.08);"
        " margin: -20px -20px 16px -20px; position: sticky; top: 0; z-index: 50; }\n"
        "  .iter14-tab { background: transparent; border: none; color: #8892b0;"
        " padding: 14px 22px; font-size: 14px; cursor: pointer;"
        " position: relative; transition: all 0.2s; font-weight: 500;"
        " letter-spacing: 0.3px; outline: none; font-family: inherit; }\n"
        "  .iter14-tab:hover { color: #cfd6e8; background: rgba(100,181,246,0.06); }\n"
        "  .iter14-tab.active { color: #64b5f6; background: rgba(100,181,246,0.10); }\n"
        "  .iter14-tab.active::after { content: \"\"; position: absolute;"
        " left: 12px; right: 12px; bottom: -1px; height: 3px;"
        " background: #64b5f6; border-radius: 2px 2px 0 0; }\n"
        "  .iter14-tab-stats { color: #5d6a85; font-size: 11px; margin-left: 8px;"
        " font-weight: 400; }\n"
        "  .iter14-tab.active .iter14-tab-stats { color: #93b9e6; }\n"
        "  .iter14-tab-spacer { flex: 1; }\n"
        "  .iter14-tab-info { color: #5d6a85; font-size: 11px; padding: 14px 8px; }\n"
        "</style>\n"
    )
    tab_html = (
        '<div class="iter14-tabs" role="tablist">\n'
        '  <button class="iter14-tab active" data-iter14-tab="train" role="tab"'
        ' aria-selected="true" type="button">训练图'
        '<span class="iter14-tab-stats">' + str(s_train["nodes"]) + ' nodes · '
        + str(s_train["groups"]) + ' groups · ' + str(s_train["edges"]) + ' edges</span></button>\n'
        '  <button class="iter14-tab" data-iter14-tab="infer" role="tab"'
        ' aria-selected="false" type="button">推理图'
        '<span class="iter14-tab-stats">' + str(s_infer["nodes"]) + ' nodes · '
        + str(s_infer["groups"]) + ' groups · ' + str(s_infer["edges"]) + ' edges</span></button>\n'
        '  <div class="iter14-tab-spacer"></div>\n'
        '  <div class="iter14-tab-info">点击 Tab 切换训练/推理 DAG</div>\n'
        '</div>\n'
    )

    # Insert styles right before </head>; insert tab bar right after <body>.
    if '</head>' in base_html:
        base_html = base_html.replace('</head>', tab_styles + '</head>', 1)
    else:
        base_html = tab_styles + base_html
    base_html = _re.sub(r'(<body[^>]*>)', r'\1\n' + tab_html, base_html, count=1)

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
        '    try { if (typeof nodePortMap !== "undefined") { Object.keys(nodePortMap).forEach(k => delete nodePortMap[k]); } } catch(e){}\n'
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
        '    DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2 || g.is_native === true; });\n'
        '  }\n'
        '  function activateTab(mode){\n'
        '    var btns = document.querySelectorAll(".iter14-tab");\n'
        '    btns.forEach(function(b){\n'
        '      var on = b.getAttribute("data-iter14-tab") === mode;\n'
        '      b.classList.toggle("active", on);\n'
        '      b.setAttribute("aria-selected", on ? "true" : "false");\n'
        '    });\n'
        '    var nextData = (mode === "infer") ? DATA_INFER : DATA_TRAIN;\n'
        '    if (DATA === nextData) return;\n'
        '    _resetSharedState();\n'
        '    DATA = nextData;\n'
        '    _rebuildIndices();\n'
        '    if (typeof invokeRender === "function") invokeRender();\n'
        '  }\n'
        '  document.querySelectorAll(".iter14-tab").forEach(function(b){\n'
        '    b.addEventListener("click", function(){\n'
        '      activateTab(b.getAttribute("data-iter14-tab"));\n'
        '    });\n'
        '  });\n'
        '  // Expose for debugging.\n'
        '  window.__iter14_activateTab = activateTab;\n'
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
    render_group_js = _RENDER_GROUP_JS_PATH.read_text(encoding="utf-8")
    if _RENDER_GROUP_JS_PLACEHOLDER not in html_template:
        raise RuntimeError("render_group.js placeholder is missing from FLOWCHART_HTML_TEMPLATE")
    html_template = html_template.replace(_RENDER_GROUP_JS_PLACEHOLDER, render_group_js, 1)
    # NOTE: The legacy "Total X% · Y" → totalUs reflow patch that previously
    # lived here is gone.  After the kernel-field migration the in-template
    # SVG label already reads `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`
    # directly (g.dur_us mirrors kernel_us), so no post-hoc replace is
    # needed.


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



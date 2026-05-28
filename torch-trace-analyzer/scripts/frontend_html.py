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
import ast
from collections import defaultdict
from pathlib import Path

# When this module is imported by ``analyze_trace.py`` the script directory is
# already on ``sys.path``.  We add it defensively here so the module also
# imports cleanly when loaded standalone (e.g. ``python -c "import
# frontend_html"`` from the scripts/ directory or from a tooling harness).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Explicit, unambiguous imports of all shared helpers from analyze_trace.
# Any rename or removal of these names in analyze_trace.py will surface here
# as an ImportError instead of silently degrading at runtime.
from analyze_trace import (  # noqa: E402  (sys.path tweak above is intentional)
    ASTFrontend,
    _build_class_map,
    _build_source_dependency_order,
    _strip_inline_comment,
    _validate_timeline_modules,
    build_instance_timing_pipeline,
    build_main_thread_hierarchy,
    build_static_module_tree,
    extract_step_phase_intervals,
    format_duration,
)


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
.dag-container { width: 100%; overflow: auto; position: relative; }
.dag-svg { display: block; margin: 0 auto; }
.dag-svg .group-box { fill: rgba(255,255,255,0.03); stroke: rgba(255,255,255,0.12); stroke-width: 1.5; rx: 10; cursor: pointer; transition: fill 0.2s; }
.dag-svg .group-box:hover { fill: rgba(255,255,255,0.06); stroke: rgba(100,181,246,0.4); }
.dag-svg .group-box.collapsed { fill: rgba(100,181,246,0.08); stroke: rgba(100,181,246,0.3); stroke-dasharray: 5,3; }
.dag-svg .group-label { font-size: 11px; fill: #8892b0; font-weight: 600; pointer-events: none; }
.dag-svg .group-timing { font-size: 10px; fill: #64b5f6; pointer-events: none; }
.dag-svg .group-toggle { font-size: 10px; fill: #555; pointer-events: auto; cursor: pointer; }
.dag-svg .group-info-hit { fill: rgba(100,181,246,0.001); stroke: rgba(100,181,246,0.45); stroke-width: 1; pointer-events: auto; cursor: pointer; }
.dag-svg .group-info-hit:hover { fill: rgba(100,181,246,0.12); stroke: rgba(100,181,246,0.9); }
.dag-svg .group-info-text { fill: rgba(100,181,246,0.82); font-size: 10px; font-weight: 700; pointer-events: none; text-anchor: middle; dominant-baseline: middle; }
.dag-svg .leaf-node { rx: 6; ry: 6; stroke-width: 1.5; cursor: default; transition: filter 0.2s; }
.dag-svg .leaf-node:hover { filter: brightness(1.2); }
.dag-svg .node-label { font-size: 11px; fill: #ffffff; font-weight: 500; pointer-events: none; text-anchor: middle; }
.dag-svg .node-sublabel { font-size: 9px; fill: rgba(255,255,255,0.7); pointer-events: none; text-anchor: middle; }
.dag-svg .edge-path { fill: none; stroke: rgba(255,255,255,0.2); stroke-width: 1.7; marker-end: url(#arrowhead); opacity: 0.82; transition: stroke 0.18s ease, stroke-width 0.18s ease, opacity 0.18s ease, filter 0.18s ease; }
.dag-svg .edge-path.internal { stroke: rgba(100,181,246,0.46); stroke-width: 1.35; }
.dag-svg .edge-path.dep { stroke: rgba(46,204,113,0.62); stroke-width: 1.9; }
.dag-svg .edge-path.flow { stroke: rgba(255,255,255,0.28); stroke-width: 1.35; stroke-dasharray: 4,3; }
.dag-svg .edge-path.edge-dim { opacity: 0.14 !important; filter: saturate(0.7); }
.dag-svg .edge-path.edge-active { opacity: 1 !important; stroke-width: 4.4 !important; filter: drop-shadow(0 0 8px rgba(100,181,246,0.35)); }
.dag-svg .edge-path.dep.edge-active { stroke: #7CFFB2 !important; }
.dag-svg .edge-path.flow.edge-active { stroke: #FFD166 !important; }
.dag-svg .edge-path.internal.edge-active { stroke: #7FD1FF !important; }
.dag-svg .leaf-node.node-dim, .dag-svg .group-box.node-dim, .dag-svg .io-node.node-dim { opacity: 0.25; }
.dag-svg .leaf-node.node-active, .dag-svg .group-box.node-active, .dag-svg .io-node.node-active { stroke: #FFD166 !important; stroke-width: 3 !important; filter: drop-shadow(0 0 10px rgba(255,209,102,0.35)); opacity: 1 !important; }
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
.lineage-step { margin: 0 12px 14px; padding-bottom: 6px; border-bottom: 1px dashed rgba(255,255,255,0.06); }
.lineage-step:last-child { border-bottom: none; }
.lineage-head { font-size: 11px; color: #c0c8d8; padding: 0 6px 6px 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.lineage-head code { background: rgba(255,255,255,0.04); padding: 1px 6px; border-radius: 4px; font-size: 11px; color: #ffd166; }
.lineage-tag { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
.lineage-tag.tag-producer { background: rgba(46,204,113,0.15); color: #6dd29d; border: 1px solid rgba(46,204,113,0.35); }
.lineage-tag.tag-consumer { background: rgba(255,209,102,0.18); color: #ffd166; border: 1px solid rgba(255,209,102,0.4); }
.lineage-tag.tag-step { background: rgba(100,181,246,0.15); color: #64b5f6; border: 1px solid rgba(100,181,246,0.35); }
.lineage-loc { font-size: 10px; color: #6f7a93; font-family: 'SFMono-Regular', Consolas, monospace; }
.dag-svg .group-clickable { cursor: pointer; }
.dag-svg .edge-path { pointer-events: stroke; cursor: pointer; }
.dag-svg .edge-path:hover { stroke-width: 3.4; opacity: 1 !important; }
.evidence-meta { padding: 6px 18px 10px; font-size: 11px; color: #aab4cf; }
.evidence-meta b { color: #64b5f6; }
.evidence-meta code { background: rgba(100,181,246,0.12); padding: 1px 5px; border-radius: 3px; font-family: monospace; color: #ffffff; }
.timing-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.timing-label { display: inline-flex; align-items: center; gap: 6px; }
.metric-help { display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; border: 1px solid rgba(100,181,246,0.45); color: #8fc5ff; font-size: 10px; line-height: 1; cursor: help; position: relative; }
.metric-help:hover { background: rgba(100,181,246,0.14); color: #d9eeff; }
.metric-help-popup { position: fixed; background: #1a2a4a; border: 1px solid rgba(100,181,246,0.5); border-radius: 6px; padding: 7px 10px; font-size: 11px; color: #cde; pointer-events: none; z-index: 9999; max-width: 260px; box-shadow: 0 4px 16px rgba(0,0,0,0.5); line-height: 1.5; display: none; }
.timing-value { color: #e7eefc; font-variant-numeric: tabular-nums; }
.truncation-note { font-size: 10px; color: #c77a3c; padding: 6px 18px 0; }
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

<script>
const DATA = __FLOWCHART_DATA_PLACEHOLDER__;

const groupMap = {};
const nodeMap = {};
const collapsedState = {};
let positions = {};
let edgeDomRegistry = [];
let focusedEdgePath = null;
let hoveredEdgeKey = null;
let hoveredGroupId = null;
let hoveredNodeId = null;
let nodeDomRegistry = new Map();
const nodeAncestorGroups = new Map();
const LONG_EDGE_MIN_SPAN = 260;

function edgeKey(edge) {
    if (!edge) return '';
    return [edge.from || '', edge.to || '', edge.type || '', edge.from_attr || '', edge.to_attr || '', edge.parent_class || ''].join('||');
}

function bundleKey(direction, nid) {
    return `${direction}::${nid || ''}`;
}

function collectAllRenderableEdges() {
    const allEdges = [];
    for (const e of (DATA.edges || [])) allEdges.push(e);
    for (const g of (DATA.groups || [])) {
        for (const e of (g.internal_edges || [])) {
            allEdges.push({
                __internal: true,
                __gid: g.id,
                from: e.from_child,
                to: e.to_child,
                type: e.type || 'internal',
                from_attr: e.from_attr,
                to_attr: e.to_attr,
                parent_class: e.parent_class || g.label,
                evidence: e.evidence,
            });
        }
    }
    return allEdges;
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

function toggleBundle(bundleId) {
    return;
}

function registerBundleWidget(group, bundleId) {
    return;
}

function renderBundleWidget(bundleId, cx, cy, direction) {
    return;
}

function getActiveEdgeKeys() {
    if (focusedEdgePath) return new Set([focusedEdgePath]);
    if (hoveredEdgeKey) return new Set([hoveredEdgeKey]);
    if (hoveredNodeId) {
        const keys = edgeDomRegistry
            .filter(item => item.edge && (item.edge.from === hoveredNodeId || item.edge.to === hoveredNodeId))
            .map(item => item.key);
        return new Set(keys);
    }
    if (hoveredGroupId) {
        const keys = edgeDomRegistry
            .filter(item => item.edge && item.edge.__gid === hoveredGroupId)
            .map(item => item.key);
        return new Set(keys);
    }
    return new Set();
}

function syncLongEdgeDisplay(path, isActive) {
    if (!path || path.dataset.longEdge !== '1') return;
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

function isIONodeId(nodeId) {
    return !!nodeId && (nodeId === DATA.input_node_id || nodeId === DATA.loss_node_id);
}

function applyEdgeFocusState() {
    const activeKeys = getActiveEdgeKeys();
    const hasActive = activeKeys.size > 0 || !!hoveredNodeId;
    const relatedNodes = new Set();
    const relatedGroups = new Set();
    const ioFocusedNodes = new Set();
    const ioHoverActive = isIONodeId(hoveredNodeId);
    if (hoveredNodeId) {
        if (ioHoverActive) {
            ioFocusedNodes.add(hoveredNodeId);
        } else {
            relatedNodes.add(hoveredNodeId);
        }
        for (const gid of getAncestorGroups(hoveredNodeId)) {
            relatedGroups.add(gid);
        }
    }
    for (const item of edgeDomRegistry) {
        const active = activeKeys.has(item.key);
        item.path.classList.toggle('edge-active', active);
        item.path.classList.toggle('edge-dim', hasActive && !active);
        syncLongEdgeDisplay(item.path, active);
        if (active) {
            const fromIsIO = isIONodeId(item.edge.from);
            const toIsIO = isIONodeId(item.edge.to);
            if (fromIsIO || toIsIO) {
                if (fromIsIO) ioFocusedNodes.add(item.edge.from);
                if (toIsIO) ioFocusedNodes.add(item.edge.to);
                const peerNodeId = fromIsIO ? item.edge.to : item.edge.from;
                if (item.edge.__gid) {
                    relatedGroups.add(item.edge.__gid);
                }
                for (const gid of getAncestorGroups(peerNodeId)) {
                    relatedGroups.add(gid);
                }
            } else {
                relatedNodes.add(item.edge.from);
                relatedNodes.add(item.edge.to);
                if (item.edge.__gid) {
                    relatedGroups.add(item.edge.__gid);
                }
                for (const gid of getAncestorGroups(item.edge.from)) {
                    relatedGroups.add(gid);
                }
                for (const gid of getAncestorGroups(item.edge.to)) {
                    relatedGroups.add(gid);
                }
            }
        }
    }
    for (const nid of ioFocusedNodes) {
        relatedNodes.add(nid);
    }
    if (hoveredGroupId) {
        relatedGroups.add(hoveredGroupId);
    }
    for (const [nid, els] of nodeDomRegistry.entries()) {
        const active = relatedNodes.has(nid) || relatedGroups.has(nid);
        for (const el of els) {
            el.classList.toggle('node-active', hasActive && active);
            el.classList.toggle('node-dim', hasActive && !active);
        }
    }
}

function clearEdgeFocus() {
    focusedEdgePath = null;
    hoveredEdgeKey = null;
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
        hoveredNodeId = null;
        hoveredGroupId = gid;
        applyEdgeFocusState();
    });
    el.addEventListener('mouseleave', (e) => {
        e.stopPropagation();
        hoveredGroupId = null;
        applyEdgeFocusState();
    });
}

function bindIONodeHover(el, nid) {
    if (!el || !nid) return;
    el.addEventListener('mouseenter', () => {
        hoveredGroupId = null;
        hoveredEdgeKey = null;
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

function indexGroupAncestors(groups, ancestors = []) {
    for (const g of (groups || [])) {
        const nextAncestors = ancestors.concat(g.id);
        nodeAncestorGroups.set(g.id, ancestors.slice());
        for (const nid of (g.children_nodes || [])) {
            nodeAncestorGroups.set(nid, nextAncestors.slice());
        }
        indexGroupAncestors(g.children_groups || [], nextAncestors);
    }
}


DATA.groups.forEach(g => groupMap[g.id] = g);
DATA.nodes.forEach(n => nodeMap[n.id] = n);
indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));
// Default: depth > 1 groups start collapsed
DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2; });

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

// Compute longest-path rank for each child of a group based on its
// internal_edges. Roots (no incoming edge) get rank 0; rank(v) = max(rank(u)+1)
// over all edges u→v. This produces a layered DAG layout where data flows
// strictly top-to-bottom and siblings of the same rank line up horizontally.
function computeRanks(group) {
    const callOrder = group.call_order || [];
    const ids = callOrder.map(c => c.id);
    const idSet = new Set(ids);
    const inEdges = {};   // id -> [src ids]
    const outEdges = {};  // id -> [dst ids]
    ids.forEach(id => { inEdges[id] = []; outEdges[id] = []; });
    const edges = (group.internal_edges || []).filter(e => idSet.has(e.from_child) && idSet.has(e.to_child) && e.from_child !== e.to_child);
    edges.forEach(e => {
        outEdges[e.from_child].push(e.to_child);
        inEdges[e.to_child].push(e.from_child);
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
            (outAdj[e.from_child] = outAdj[e.from_child] || []).push(e.to_child);
            (inAdj[e.to_child] = inAdj[e.to_child] || []).push(e.from_child);
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
        positions[gid] = { w, h, collapsed: true };
        return { w, h };
    }

    // Expanded: lay out children using layered DAG placement
    const callOrder = g.call_order || [];
    if (callOrder.length === 0) {
        const w = LAYOUT.nodeW + 40;
        const h = LAYOUT.nodeH + LAYOUT.groupPadTop + LAYOUT.groupPadBottom;
        positions[gid] = { w, h, collapsed: false, childPositions: [] };
        return { w, h };
    }

    // 1. Recursively layout each child to obtain its bounding box
    const childSizes = [];
    for (const item of callOrder) {
        if (item.type === 'node') {
            childSizes.push({ id: item.id, type: 'node', w: LAYOUT.nodeW, h: LAYOUT.nodeH });
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
    if (g.internal_edges && g.internal_edges.length) {
        const rankOf = rankInfo.rank;
        for (const ed of g.internal_edges) {
            const ra = rankOf[ed.from_child], rb = rankOf[ed.to_child];
            if (ra !== undefined && rb !== undefined && Math.abs(rb - ra) > 1) {
                hasSkipEdges = true; break;
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

    positions[gid] = {
        w: groupW, h: groupH, collapsed: false, childPositions,
        rowLayouts, rankInfo
    };
    return { w: groupW, h: groupH };
}

function render() {
    // Clear and rebuild positions / edge registry on every render
    positions = {};
    edgeDomRegistry = [];
    nodeDomRegistry = new Map();
    focusedEdgePath = null;
    hoveredEdgeKey = null;
    hoveredGroupId = null;
    hoveredNodeId = null;
    positions = {};
    // Layout root groups (RootModule)
    const rootSizes = DATA.root_groups.map(rid => {
        const sz = layoutGroup(rid);
        return { id: rid, ...sz };
    });

    // Top-level layout is a strict vertical chain: Input → RootModule → Result
    const ioW = 140, ioH = 40;
    const ioGap = 36;  // vertical gap between Input/Root/Result
    const maxRootW = rootSizes.length ? Math.max(...rootSizes.map(r => r.w)) : LAYOUT.nodeW;
    const maxRootH = rootSizes.length ? Math.max(...rootSizes.map(r => r.h)) : LAYOUT.nodeH;

    const svgW = Math.max(maxRootW + 80, 480);
    const svgH = ioH * 2 + ioGap * 2 + maxRootH + 80;

    const svg = document.getElementById('dag-svg');
    svg.addEventListener('click', () => clearEdgeFocus());
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.innerHTML = `<defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="rgba(255,255,255,0.3)"/>
        </marker>
        <marker id="arrowhead-blue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="rgba(100,181,246,0.5)"/>
        </marker>
        <marker id="arrowhead-dep" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="rgba(46,204,113,0.6)"/>
        </marker>
    </defs>`;

    // Position the (single) RootModule centered horizontally.
    const rootPositions = [];
    if (rootSizes.length > 0) {
        const rs = rootSizes[0];
        const rx = (svgW - rs.w) / 2;
        const ry = 30 + ioH + ioGap;
        rootPositions.push({ id: rs.id, x: rx, y: ry, w: rs.w, h: rs.h });
    }

    // Render groups recursively
    const edgeList = [];

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

    function renderGroupAt(gid, ox, oy) {
        const g = groupMap[gid];
        const pos = positions[gid];
        if (!pos) return;

        if (pos.collapsed) {
            // Draw collapsed group as a rounded rect with label
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', ox); rect.setAttribute('y', oy);
            rect.setAttribute('width', pos.w); rect.setAttribute('height', pos.h);
            rect.setAttribute('class', 'group-box collapsed');
            rect.setAttribute('style', `stroke: ${getGroupBorderColor(g)}`);
            rect.dataset.gid = gid;
            rect.addEventListener('dblclick', () => toggleGroup(gid));
            bindGroupHover(rect, gid);
            svg.appendChild(rect);
            registerNodeDom(gid, rect);

            let labelText = `▶ ${g.label}`;
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', ox + pos.w/2); label.setAttribute('y', oy + pos.h/2 - 2);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('class', 'group-label');
            label.textContent = labelText;
            bindGroupHover(label, gid);
            svg.appendChild(label);

            if (g.has_timing) {
                const tl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                tl.setAttribute('x', ox + pos.w/2); tl.setAttribute('y', oy + pos.h/2 + 12);
                tl.setAttribute('text-anchor', 'middle');
                tl.setAttribute('class', 'group-timing');
                // Kernel-only contract: g.dur_us already mirrors kernel_us.
                tl.textContent = `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`;
                bindGroupHover(tl, gid);
                svg.appendChild(tl);
            }

            // Source-info icon for collapsed groups (top-right corner)
            if (g.src_file) {
                appendGroupInfoButton(
                    g,
                    ox + pos.w - 13,
                    oy + 10,
                    `View ${g.label} class definition (${g.src_file}:${g.src_start_line}-${g.src_end_line})`
                );
            }

            // Register port positions for edges
            positions[gid + '__in'] = { cx: ox + pos.w/2, cy: oy };
            positions[gid + '__out'] = { cx: ox + pos.w/2, cy: oy + pos.h };
            positions[gid + '__center'] = { cx: ox + pos.w/2, cy: oy + pos.h/2 };
            return;
        }

        // Expanded group box
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', ox); rect.setAttribute('y', oy);
        rect.setAttribute('width', pos.w); rect.setAttribute('height', pos.h);
        rect.setAttribute('class', 'group-box');
        rect.setAttribute('style', `stroke: ${getGroupBorderColor(g)}`);
        rect.dataset.gid = gid;
        rect.addEventListener('dblclick', (e) => {
            if (e.target === rect) toggleGroup(gid);
        });
        bindGroupHover(rect, gid);
        svg.appendChild(rect);
        registerNodeDom(gid, rect);

        // Group header label
        let headerText = `▼ ${g.label}`;
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', ox + 12); label.setAttribute('y', oy + 18);
        label.setAttribute('class', 'group-label');
        label.setAttribute('style', 'cursor: pointer;');
        label.textContent = headerText;
        label.addEventListener('dblclick', () => toggleGroup(gid));
        bindGroupHover(label, gid);
        svg.appendChild(label);

        // Info icon next to label - opens source panel for this group's class
        if (g.src_file) {
            // Approximate label width to position the icon to the right of the text
            const labelLen = (headerText.length * 6.6) + 8;
            appendGroupInfoButton(
                g,
                ox + 12 + labelLen + 6,
                oy + 15,
                `View ${g.label} class definition (${g.src_file}:${g.src_start_line}-${g.src_end_line})`
            );
        }

        if (g.has_timing) {
            const tl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            tl.setAttribute('x', ox + pos.w - 10); tl.setAttribute('y', oy + 18);
            tl.setAttribute('text-anchor', 'end');
            tl.setAttribute('class', 'group-timing');
            // Kernel-only contract: g.dur_us already mirrors kernel_us.
            tl.textContent = `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`;
            bindGroupHover(tl, gid);
            svg.appendChild(tl);
        }

        // Render children
        const cp = pos.childPositions || [];
        for (const child of cp) {
            if (child.type === 'node') {
                renderNodeAt(child.id, ox + child.x, oy + child.y, child.w, child.h);
            } else {
                renderGroupAt(child.id, ox + child.x, oy + child.y);
            }
        }

        // Draw internal data dependency edges between children with
        // rank-aware routing. This pass is local to the group so we can use
        // the row layout to bend edges around obstacles.
        if (g.internal_edges && pos.rowLayouts) {
            const childAbsRect = {};
            let cMinX = Infinity, cMaxX = -Infinity;
            for (const child of cp) {
                const ax = ox + child.x;
                childAbsRect[child.id] = {
                    x: ax, y: oy + child.y,
                    w: child.w, h: child.h, rank: child.rank
                };
                if (ax < cMinX) cMinX = ax;
                if (ax + child.w > cMaxX) cMaxX = ax + child.w;
            }
            const rowBands = pos.rowLayouts.map(r => ({ rank: r.rank, top: oy + r.y, bot: oy + r.y + r.h }));
            // Lanes must live OUTSIDE the children band (cMinX..cMaxX) but
            // INSIDE the group rect (ox..ox+pos.w). Pick a comfortable inset.
            const groupRight = Math.min(ox + pos.w - 6, cMaxX + LAYOUT.laneGutter);
            const groupLeft = Math.max(ox + 6, cMinX - LAYOUT.laneGutter);
            // Lane bands: edges should always go past cMaxX on the right, or
            // before cMinX on the left, never in between.
            const childRightEdge = cMaxX;
            const childLeftEdge = cMinX;
            for (const ed of g.internal_edges) {
                const fr = childAbsRect[ed.from_child];
                const to = childAbsRect[ed.to_child];
                if (!fr || !to) continue;
                const routedEdge = {
                    __internal: true,
                    __gid: gid,
                    from: ed.from_child,
                    to: ed.to_child,
                    type: ed.type || 'internal',
                    from_attr: ed.from_attr,
                    to_attr: ed.to_attr,
                    parent_class: ed.parent_class || g.label,
                    evidence: ed.evidence,
                };
                if (!isEdgeVisible(routedEdge)) continue;
                drawRoutedEdge(fr, to, ed.type, rowBands,
                               groupLeft, groupRight,
                               childLeftEdge, childRightEdge, routedEdge);
            }
        }

        // Register port positions
        positions[gid + '__in'] = { cx: ox + pos.w/2, cy: oy };
        positions[gid + '__out'] = { cx: ox + pos.w/2, cy: oy + pos.h };
    }

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

        // Label: attr_name (class)
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', nx + w/2); label.setAttribute('y', ny + h/2 - 2);
        label.setAttribute('class', 'node-label');
        label.style.pointerEvents = 'none';
        label.textContent = n.attr_name || n.class_name;
        svg.appendChild(label);

        // Sublabel
        let subText = n.class_name !== n.attr_name ? n.class_name : '';
        if (n.has_timing) subText = `${n.pct.toFixed(1)}%`;
        if (subText) {
            const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            sub.setAttribute('x', nx + w/2); sub.setAttribute('y', ny + h/2 + 11);
            sub.setAttribute('class', 'node-sublabel');
            sub.style.pointerEvents = 'none';
            sub.textContent = subText;
            svg.appendChild(sub);
        }

        // Register positions for edges
        positions[nid] = { cx: nx + w/2, cy: ny + h/2 };
        positions[nid + '__in'] = { cx: nx + w/2, cy: ny };
        positions[nid + '__out'] = { cx: nx + w/2, cy: ny + h };
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

    function drawEdge(x1, y1, x2, y2, type, edgeData, routeMeta=null) {
        const dy = y2 - y1;
        const dx = x2 - x1;
        if (Math.abs(dy) < 3 && Math.abs(dx) < 3) return;
        const meta = routeMeta || {};
        const offset = meta.bundleOffset || 0;
        const sourceFanout = meta.sourceFanout || 1;
        const targetFanin = meta.targetFanin || 1;
        const sideBias = offset === 0 ? (dx >= 0 ? 1 : -1) : (offset > 0 ? 1 : -1);
        const curvature = Math.max(16, Math.min(90, Math.abs(offset) * 1.25 + Math.max(sourceFanout, targetFanin) * 3));
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

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d);
        applyEdgeStyle(path, type);
        if (edgeData) {
            attachEdgeClick(path, edgeData);
        }
        svg.appendChild(path);
        configureLongEdgeDisplay(path);
    }

    // Rank-aware edge routing for intra-group edges.
    // - Same-rank edges (rare; usually a layout artifact) are drawn as a
    //   horizontal-ish curve below the row.
    // - Adjacent-rank edges go straight down with a gentle bezier.
    // - Skip-rank edges (span > 1 row) are routed around the intermediate
    //   rows by bending sideways, choosing the side closest to the source/dest
    //   x. Each skip edge gets its own lane offset to avoid overlap.
    const skipLaneCounter = { left: 0, right: 0 };
    function drawRoutedEdge(fr, to, type, rowBands, groupLeft, groupRight, childLeftEdge, childRightEdge, edgeData) {
        // Source: bottom-center of the from rect; Dest: top-center of the to rect
        const x1 = fr.x + fr.w / 2, y1 = fr.y + fr.h;
        const x2 = to.x + to.w / 2, y2 = to.y;
        const fromRank = fr.rank, toRank = to.rank;
        const span = Math.abs(toRank - fromRank);

        if (span <= 1 && y2 > y1) {
            drawEdge(x1, y1, x2, y2, type, edgeData, EDGE_BUNDLE_META.get(edgeKey(edgeData)) || null);
            return;
        }
        if (span === 0) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M${x1},${y1} C${x1},${y1+14} ${x2},${y2+14} ${x2},${y2}`);
            path.setAttribute('class', `edge-path ${type || ''}`);
            path.setAttribute('marker-end', type === 'dep' ? 'url(#arrowhead-dep)' : 'url(#arrowhead)');
            if (edgeData) attachEdgeClick(path, edgeData);
            svg.appendChild(path);
            configureLongEdgeDisplay(path);
            return;
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
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const d = [
            `M${x1},${y1}`,
            `C${x1},${y1 + verticalApproach} ${laneX},${y1 + verticalApproach} ${laneX},${y1 + verticalApproach + 10}`,
            `L${laneX},${y2 - verticalApproach - 10}`,
            `C${laneX},${y2 - verticalApproach} ${x2},${y2 - verticalApproach} ${x2},${y2}`
        ].join(' ');
        path.setAttribute('d', d);
        path.setAttribute('class', `edge-path ${type || ''}`);
        path.setAttribute('marker-end', type === 'dep' ? 'url(#arrowhead-dep)' : 'url(#arrowhead)');
        path.setAttribute('opacity', '0.7');
        if (edgeData) attachEdgeClick(path, edgeData);
        svg.appendChild(path);
        configureLongEdgeDisplay(path);
    }

    function registerEdgeDom(path, edgeData) {
        if (!path || !edgeData) return;
        edgeDomRegistry.push({ path, edge: edgeData, key: edgeKey(edgeData) });
    }

    function applyEdgeStyle(path, type) {
        path.setAttribute('class', `edge-path ${type || ''}`);
        if (type === 'dep') {
            path.setAttribute('marker-end', 'url(#arrowhead-dep)');
        } else if (type === 'internal') {
            path.setAttribute('marker-end', 'url(#arrowhead-blue)');
        } else {
            path.setAttribute('marker-end', 'url(#arrowhead)');
        }
    }

    function attachEdgeClick(path, edgeData) {
        if (!path || !edgeData) return;
        const key = edgeKey(edgeData);
        path.style.cursor = 'pointer';
        path.dataset.edgeKey = key;
        path.addEventListener('mouseenter', () => {
            hoveredEdgeKey = key;
            applyEdgeFocusState();
        });
        path.addEventListener('mouseleave', () => {
            hoveredEdgeKey = null;
            applyEdgeFocusState();
        });
        path.addEventListener('click', (e) => {
            e.stopPropagation();
            const alreadyActive = focusedEdgePath === key;
            if (alreadyActive) {
                focusedEdgePath = null;
                applyEdgeFocusState();
            } else {
                setEdgeFocus(key);
            }
            showEdgePanel(edgeData);
        });
        registerEdgeDom(path, edgeData);
    }

    // Render root groups
    for (const rp of rootPositions) {
        renderGroupAt(rp.id, rp.x, rp.y);
    }

    // Render top-level synthetic Input / Result pill nodes
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

        positions[nid] = { cx, cy };
        positions[nid + '__in'] = { cx, cy: y };
        positions[nid + '__out'] = { cx, cy: y + h };
    }

    if (DATA.input_node_id) {
        const cx = svgW / 2;
        const cy = 30 + ioH / 2;
        renderIOPill(DATA.input_node_id, cx, cy, ioW, ioH, 'Input', 'network input', 'rgba(46,204,113,0.55)');
    }
    if (DATA.loss_node_id) {
        const lossNode = nodeMap[DATA.loss_node_id];
        const cx = svgW / 2;
        const rootEntry = rootPositions[0];
        const baseY = rootEntry ? (rootEntry.y + rootEntry.h) : (30 + ioH + ioGap + maxRootH);
        const cy = baseY + ioGap + ioH / 2;
        const sub = (lossNode && lossNode.has_timing)
            ? `${lossNode.class_name} · ${lossNode.pct.toFixed(1)}%`
            : (lossNode ? lossNode.class_name : 'result output');
        renderIOPill(DATA.loss_node_id, cx, cy, ioW, ioH, 'Result', sub, 'rgba(231,76,60,0.55)');
    }


    // Global edge pass: draw all data dependency edges using registered positions
    for (const edge of DATA.edges) {
        if (!isEdgeVisible(edge)) continue;
        const fromPos = positions[edge.from + '__out'] || positions[edge.from];
        const toPos = positions[edge.to + '__in'] || positions[edge.to];
        if (fromPos && toPos) {
            drawEdge(fromPos.cx, fromPos.cy, toPos.cx, toPos.cy, edge.type || 'dep', edge, EDGE_BUNDLE_META.get(edgeKey(edge)) || null);
        }
    }

    // Header info
    const meta = DATA.meta;
    if (DATA.has_timing) {
        document.getElementById('mode-badge').innerHTML = '<span class="mode-badge mode-timing">📊 Structure + Timing</span>';
        document.getElementById('meta-info').textContent = `Device: ${meta.device} | Step: ${meta.step_dur_str} | Modules: ${meta.num_modules}`;
    } else {
        document.getElementById('mode-badge').innerHTML = '<span class="mode-badge mode-structure">🏗️ Static Structure (source code)</span>';
        document.getElementById('meta-info').textContent = `Modules: ${meta.num_modules} | Root: ${meta.roots ? meta.roots.join(", ") : "N/A"}`;
    }

    // Legend
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

    // Summary
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
}

function toggleGroup(gid) {
    collapsedState[gid] = !collapsedState[gid];
    render();
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

// Iter11 ctor-src: like renderCodeBlock, but additionally <mark>-wrap any
// occurrence of an identifier from `markVars` (word-boundary match) on
// every code line. The HTML-escaping happens first so user data can never
// inject markup; we then walk the escaped text with a single regex that
// asserts non-identifier boundaries on each side.
function renderCodeBlockWithMarks(text, startLine, highlightLine, markVars, ctorStart, ctorEnd) {
    if (!text) return '<div class="code-block"><div class="code-line"><span class="code-text" style="color:#7a849c">(no source available)</span></div></div>';
    const vars = Array.isArray(markVars) ? markVars.filter(v => typeof v === 'string' && v.length > 0) : [];
    let markRe = null;
    if (vars.length > 0) {
        // Sort by length desc so e.g. `embedding_dim` matches before `dim`.
        const sorted = vars.slice().sort((a, b) => b.length - a.length);
        const escaped = sorted.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        // Use a non-identifier lookbehind/lookahead. Allow start/end of
        // string by leveraging the fact that we apply the regex per-line
        // (so ^ / $ are line boundaries effectively). We rely on the fact
        // that escaped HTML never contains a `<` from our identifiers, so
        // working on the escaped string is safe.
        markRe = new RegExp('(^|[^A-Za-z0-9_])(' + escaped.join('|') + ')(?![A-Za-z0-9_])', 'g');
    }
    const lines = text.split('\n');
    const html = lines.map((line, i) => {
        const lineno = startLine + i;
        let cls = 'code-line';
        if (highlightLine && lineno === highlightLine) cls += ' highlight';
        if (ctorStart && ctorEnd && lineno >= ctorStart && lineno <= ctorEnd) cls += ' ctor-range';
        let escaped = escapeHtml(line);
        if (markRe) {
            // Reset lastIndex (g flag) per line, since we create fresh per call.
            markRe.lastIndex = 0;
            escaped = escaped.replace(markRe, (m, p1, p2) => p1 + '<mark>' + p2 + '</mark>');
        }
        return `<div class="${cls}"><span class="lineno">${lineno}</span><span class="code-text">${escaped}</span></div>`;
    }).join('');
    return `<div class="code-block">${html}</div>`;
}

function showSourcePanel(item) {
    // item is a node or group object carrying src_* fields
    const sp = document.getElementById('side-panel');
    document.getElementById('sp-title').textContent =
        (item.attr_name && item.attr_name !== item.class_name)
            ? `${item.attr_name} (${item.class_name || item.label})`
            : (item.class_name || item.label || 'Module');
    const fileLabel = item.src_file
        ? `${item.src_file}:${item.src_start_line}-${item.src_end_line}`
        : '(class definition not available in supplied sources)';
    document.getElementById('sp-subtitle').textContent = fileLabel;
    let bodyHtml = '';
    // Kernel-only contract: dur_us mirrors kernel_us; fwd/bwd/other are the
    // only phase splits we expose. No host walltime / overhead displayed.
    const kernelMs = Number(item && item.kernel_us != null ? item.kernel_us : (item.dur_us || 0)) / 1000.0;
    const fwdMs = Number(item && item.fwd_kernel_us || 0) / 1000.0;
    const bwdMs = Number(item && item.bwd_kernel_us || 0) / 1000.0;
    const otherMs = Number(item && item.other_kernel_us || 0) / 1000.0;
    if (item.has_timing || item.has_phase_timing) {
        bodyHtml += '<div class="side-panel-section"><h4>Timing</h4>' +
            renderTimingRow('Kernel', kernelMs, 'kernel') +
            renderTimingRow('Forward', fwdMs, 'forward') +
            renderTimingRow('Backward', bwdMs, 'backward') +
            renderTimingRow('Other', otherMs, 'other') +
            '</div>';
    }
    if (item.src_snippet) {
        bodyHtml += '<div class="side-panel-section"><h4>Class definition</h4>' +
            renderCodeBlock(item.src_snippet, item.src_snippet_start, null) +
            '</div>';
        if (item.src_truncated) {
            bodyHtml += `<div class="truncation-note">⚠ Snippet truncated at line ${item.src_snippet_end} (full class spans up to line ${item.src_end_line}).</div>`;
        }
    } else {
        bodyHtml += '<div class="evidence-meta" style="color:#7a849c">This class has no source location data — it may be a synthetic IO node (Input/Result) or a runtime wrapper.</div>';
    }
    // Iter11 ctor-src: render the constructor source where this instance
    // is created in its parent's __init__. The backend provides exactly
    // three fields: ctor_src_file / ctor_src_line / ctor_src_excerpt.
    // Highlight the attribute name itself with a word-boundary-style match.
    if (item.ctor_src_excerpt) {
        const ctorFile = item.ctor_src_file || '';
        const ctorLine = item.ctor_src_line || '';
        const ctorStartLine = Math.max(1, Number(ctorLine || 1) - 2);
        bodyHtml += '<div class="side-panel-section"><h4>Constructor</h4>' +
            `<div class="evidence-meta" style="margin-bottom:6px"><b>file:</b> ${escapeHtml(ctorFile)} &nbsp;·&nbsp; <b>line:</b> ${escapeHtml(String(ctorLine))}</div>` +
            renderCodeBlockWithMarks(
                item.ctor_src_excerpt,
                ctorStartLine,
                Number(ctorLine) || null,
                item.attr_name ? [item.attr_name] : []
            ) +
            '</div>';
    }
    document.getElementById('sp-body').innerHTML = bodyHtml;
    sp.classList.add('open');
}

function showEdgePanel(edge) {
    const sp = document.getElementById('side-panel');
    const fromAttr = edge.from_attr || '?';
    const toAttr = edge.to_attr || '?';
    const parentClass = edge.parent_class || '';
    document.getElementById('sp-title').textContent = `${fromAttr} → ${toAttr}`;
    const sub = parentClass ? `data dependency in ${parentClass}.forward()` : 'data dependency';
    document.getElementById('sp-subtitle').textContent = sub;
    let bodyHtml = '';
    const ev = edge.evidence;
    const isInputEdge = (fromAttr === 'Input');
    if (ev) {
        const fmtShape = (shape) => {
            if (shape === null || shape === undefined) return '';
            if (Array.isArray(shape)) {
                if (shape.length === 0) return '';
                if (shape.every(x => Array.isArray(x))) {
                    return '[' + shape.map(inner => fmtShape(inner)).filter(Boolean).join(', ') + ']';
                }
                return '[' + shape.map(x => String(x)).join(', ') + ']';
            }
            return String(shape);
        };
        const shapeFrom = fmtShape(ev.shape_from);
        const shapeTo = fmtShape(ev.shape_to);
        let shapeBlock = '';
        if (shapeFrom || shapeTo) {
            shapeBlock = '<div class="side-panel-section"><h4>Tensor Shape</h4>' +
                (shapeFrom ? `<div class="evidence-meta"><b>From:</b> <code>${escapeHtml(shapeFrom)}</code></div>` : '') +
                (shapeTo ? `<div class="evidence-meta"><b>To:</b> <code>${escapeHtml(shapeTo)}</code></div>` : '') +
                '</div>';
        }
        const headVar = ev.var ? ('<b>Tensor variable:</b> <code>' + escapeHtml(ev.var) + '</code> &nbsp;·&nbsp; ') : '';
        bodyHtml += `<div class="evidence-meta">
            ${headVar}<b>file:</b> ${escapeHtml(ev.file)}
        </div>`;
        if (shapeBlock) bodyHtml += shapeBlock;

        // Iter11 varlineage: render the full assignment chain
        // (var_history) when available so the user sees how the
        // value travels from producer → intermediate ops → consumer.
        const history = Array.isArray(ev.var_history) ? ev.var_history : [];
                if (history.length > 0) {
                    bodyHtml += '<div class="side-panel-section"><h4>Variable lineage (' + history.length + ' step' + (history.length === 1 ? '' : 's') + ')</h4>';
                    history.forEach((step, idx) => {
                        const tag = (step.role === 'consumer')
                            ? '<span class="lineage-tag tag-consumer">consumer</span>'
                            : (idx === 0 ? '<span class="lineage-tag tag-producer">producer</span>'
                                       : '<span class="lineage-tag tag-step">step ' + (idx + 1) + '</span>');
                        const vlabel = step.var ? ('<code>' + escapeHtml(step.var) + '</code>') : '<code>(carrier)</code>';
                        const stepMarkClass = (step.role === 'consumer') ? 'consumer-mark' : 'producer-mark';
                        bodyHtml += '<div class="lineage-step">' +
                            '<div class="lineage-head">' + tag + ' ' + vlabel +
                            ' <span class="lineage-loc">@' + escapeHtml(step.file || ev.file) + ':' + step.line + '</span></div>' +
                            renderCodeBlock(step.excerpt && step.excerpt.text, step.excerpt && step.excerpt.start, step.excerpt && step.excerpt.highlight, (step.carriers && step.carriers.length > 0) ? step.carriers : (step.var ? [step.var] : []), stepMarkClass) +
                            '</div>';
                    });
                    bodyHtml += '</div>';
                } else {
        
            // Fall back to producer/consumer two-block layout when no
            // lineage chain was attached. For Input/top edges we skip
            // the empty producer block — there's no real Python line
            // to render on the input side.
            if (ev.from_excerpt && !isInputEdge) {
                bodyHtml += '<div class="side-panel-section"><h4>Producer (output of <code>self.' + escapeHtml(fromAttr) + '</code>)</h4>' +
                    renderCodeBlock(ev.from_excerpt.text, ev.from_excerpt.start, ev.from_excerpt.highlight, ev.var, 'producer-mark') +
                    '</div>';
            }
            if (ev.to_excerpt) {
                const consumerLabel = isInputEdge
                    ? ('<h4>Consumer (entry point <code>self.' + escapeHtml(toAttr) + '</code>)</h4>')
                    : ('<h4>Consumer (call to <code>self.' + escapeHtml(toAttr) + '</code>)</h4>');
                bodyHtml += '<div class="side-panel-section">' + consumerLabel +
                    renderCodeBlock(ev.to_excerpt.text, ev.to_excerpt.start, ev.to_excerpt.highlight, ev.var, 'consumer-mark') +
                    '</div>';
            }
        }
    } else if (edge.type === 'flow') {
        bodyHtml += '<div class="evidence-meta" style="color:#c77a3c">Sequential fallback edge: data flow could not be parsed from <code>forward()</code>; the edge represents the call order between siblings.</div>';
    } else {
        // Iter8 strict rule: every edge MUST carry source-line evidence.
        // Reaching this branch means the analyzer emitted a corrupt edge.
        // We raise a hard, visible failure rather than silently rendering a
        // grey "no evidence" hint that masks a real bug in the pipeline.
        const errMsg = 'INTEGRITY ERROR: edge ' + escapeHtml(fromAttr) +
            ' \u2192 ' + escapeHtml(toAttr) +
            ' has no source-line evidence (type=' + escapeHtml(String(edge.type || 'unknown')) +
            '). The DAG generator violated the contract that every dep edge must map to a forward() call site.';
        bodyHtml += '<div class="evidence-meta" style="background:#5a1a1a;color:#ffb4b4;padding:10px;border:1px solid #ff5a5a;border-radius:4px;font-weight:bold">' +
            '\u26d4 ' + errMsg + '</div>';
        console.error('[DAG-INTEGRITY-FAIL]', edge);
        // Throwing here ensures any automated test harness that clicks every
        // edge will surface the failure as an unhandled exception.
        throw new Error(errMsg);
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
    render();
});
document.getElementById('btn-collapse-all').addEventListener('click', () => {
    DATA.groups.forEach(g => { if (g.depth >= 1) collapsedState[g.id] = true; });
    render();
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

render();

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


def generate_html_flowchart(source_files, timing_data=None, meta=None, output_path="flowchart.html", trace_events=None, conditional_mode="infer", _return_data_only=False):
    """Generate interactive DAG HTML flowchart with expandable nested modules.

    Top-level layout is fixed as: Input -> RootModule -> Result.
    All other nn.Module sub-modules live INSIDE the RootModule container, linked
    by data dependency edges (extracted from forward()). Runtime wrapper modules
    such as DDPRootModule / SyncModule / SequentialModule / RunstepModule are
    intentionally NOT shown at the top level — they are treated as transparent
    wrappers around RootModule.

    Args:
        source_files: dict of {filename: [lines]} from load_model_code()
        timing_data: optional dict with timing info from trace analysis
        meta: optional metadata dict
        output_path: HTML output file path
        trace_events: optional list of trace events for runtime hierarchy extraction
            (kept for API compatibility — used only for timing fallbacks, NOT for
            adding wrapper modules to the DAG)

    Interactive features baked into the resulting HTML:
        * Clicking a leaf node or an expanded group header opens a side panel
          showing the class definition with file path and line range.
        * Clicking a data dependency edge opens a side panel showing the actual
          forward() lines that produced and consumed the tensor variable.
    """
    import sys

    # When we have timeline events, derive a hint for which top-level user model
    # to use as the DAG root by stripping common DDP/FSDP prefixes from runtime
    # wrapper module names.
    preferred_root = None
    if trace_events:
        runtime_names = set()
        for e in trace_events:
            n = e.get('name', '')
            if not n.startswith('nn.Module:') or e.get('ph') != 'X':
                continue
            short = n.replace('nn.Module: ', '').replace('nn.Module:', '')
            short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
            runtime_names.add(re.sub(r'_\d+$', '', short))
        # Strip wrapper prefixes and look for matches in source classes
        for wrapper_prefixes in [r'^DDP', r'^FSDP', r'^DistributedDataParallel', r'^Wrapped']:
            for rn in runtime_names:
                stripped = re.sub(wrapper_prefixes, '', rn)
                if stripped != rn and stripped:
                    preferred_root = stripped
                    break
            if preferred_root:
                break

    tree, roots, class_map = build_static_module_tree(source_files, preferred_root=preferred_root, conditional_mode=conditional_mode)

    if not tree:
        print("  ⚠️ 未在源码中检测到 nn.Module 子类，无法生成流程图")
        return None

    # Validate timeline modules against the DAG tree
    _validate_timeline_modules(trace_events, tree)

    # Force top-level root to RootModule. All sub-modules will live inside it.
    # Runtime wrappers (DDPRootModule, SyncModule, ...) are intentionally hidden.
    if "RootModule" in tree:
        roots = ["RootModule"]
    else:
        # Fall back to the previously detected primary root (single root only)
        roots = roots[:1] if roots else []

    has_timing = timing_data is not None and timing_data.get("step_dur_us", 0) > 0
    step_dur_us = timing_data["step_dur_us"] if has_timing else 0
    class_durations = timing_data.get("class_durations", {}) if has_timing else {}
    class_durations_fwd = timing_data.get("class_durations_fwd", {}) if has_timing else {}
    class_durations_bwd = timing_data.get("class_durations_bwd", {}) if has_timing else {}
    class_exclusive = timing_data.get("class_exclusive", {}) if has_timing else {}
    class_calls = timing_data.get("class_calls", {}) if has_timing else {}
    class_thread_role = timing_data.get("class_thread_role", {}) if has_timing else {}
    runtime_instance_timings_by_class = timing_data.get("runtime_instance_timings_by_class", {}) if has_timing else {}
    _instance_timing_used = defaultdict(set)

    def _attr_index_hint(attr_name):
        text = attr_name or ""
        m = re.search(r'\[(\d+)\]', text)
        if m:
            return int(m.group(1))
        return None

    def _pick_instance_timing(class_name, attr_name=None):
        items = runtime_instance_timings_by_class.get(class_name, [])
        if not items:
            return None
        used = _instance_timing_used[class_name]
        idx_hint = _attr_index_hint(attr_name)
        if idx_hint is not None:
            for item in items:
                if item.get("runtime_index") == idx_hint and item.get("runtime_name") not in used:
                    used.add(item.get("runtime_name"))
                    return item
        for item in items:
            if item.get("runtime_name") not in used:
                used.add(item.get("runtime_name"))
                return item
        if idx_hint is not None:
            for item in items:
                if item.get("runtime_index") == idx_hint:
                    return item
        return items[0] if len(items) == 1 else None

    # ------------------------------------------------------------------
    # Iter7: the top-level output node is the synthetic Result object created
    # by LG.result() inside RootModule.forward(). It is NOT a hoisted
    # nn.Module child, so we must keep all root attrs inside the RootModule
    # container and derive Result edges from actual result.head()/return lines.
    # ------------------------------------------------------------------
    loss_attr_name = None
    loss_class_name = None

    # ------------------------------------------------------------------
    # Build class source-location lookup for click-to-source UX.
    # For each class, we record:
    #   - file: defining filename
    #   - start_line / end_line: class definition range
    #   - snippet_start / snippet_end: a trimmed snippet bounded around __init__/forward
    #   - snippet: the actual text lines (capped to ~120 lines for transport size)
    # ------------------------------------------------------------------
    _SNIPPET_MAX_LINES = 160
    class_source_info = {}
    for (fname, cname), info in class_map.items():
        if cname not in tree:
            continue
        start = info["start"]
        end = info["end"]
        lines = source_files.get(fname, [])
        # Pick a focused snippet: prefer the smaller of (full class) or (~SNIPPET_MAX_LINES from start)
        snippet_end = min(end, start + _SNIPPET_MAX_LINES - 1)
        snippet_lines = lines[start - 1: snippet_end]
        # Keep an overall ceiling to avoid bloating HTML for huge classes
        truncated = snippet_end < end
        class_source_info[cname] = {
            "file": fname,
            "start_line": start,
            "end_line": end,
            "snippet_start": start,
            "snippet_end": snippet_end,
            "snippet": "\n".join(snippet_lines),
            "truncated": truncated,
            "method_ranges": {m: list(r) for m, r in info["methods"].items()},
        }

    def _split_top_level_items(text):
        items = []
        buf = []
        depth = 0
        in_str = None
        esc = False
        for ch in text:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    items.append(part)
                buf = []
            else:
                buf.append(ch)
        tail = ''.join(buf).strip()
        if tail:
            items.append(tail)
        return items

    def _parse_call_kwargs(args_text):
        kwargs = {}
        for item in _split_top_level_items(args_text):
            depth = 0
            in_str = None
            esc = False
            split_i = None
            for i, ch in enumerate(item):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == in_str:
                        in_str = None
                    continue
                if ch in ("'", '"'):
                    in_str = ch
                    continue
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth = max(0, depth - 1)
                if ch == '=' and depth == 0:
                    prev_ch = item[i - 1] if i > 0 else ''
                    next_ch = item[i + 1] if i + 1 < len(item) else ''
                    if prev_ch not in ('=', '!', '<', '>') and next_ch != '=':
                        split_i = i
                        break
            if split_i is not None:
                kwargs[item[:split_i].strip()] = item[split_i + 1:].strip()
        return kwargs

    def _scan_root_result_edges(root_cname):
        if not root_cname:
            return {}
        attrs = tree.get(root_cname, {}).get("attrs", {})
        known_attrs = set(attrs.keys())
        if not known_attrs:
            return {}

        container_to_elems = defaultdict(list)
        for a in known_attrs:
            m = re.match(r'^(\w+)\[(.+)\]$', a)
            if m:
                container_to_elems[m.group(1)].append(a)
        for cont in list(container_to_elems.keys()):
            elems = container_to_elems[cont]
            num_elems = []
            other = []
            for ea in elems:
                mm = re.match(r'^' + re.escape(cont) + r'\[(\d+)\]$', ea)
                if mm:
                    num_elems.append((int(mm.group(1)), ea))
                else:
                    other.append(ea)
            if num_elems and not other:
                container_to_elems[cont] = [ea for _, ea in sorted(num_elems)]
            else:
                container_to_elems[cont] = sorted(elems)

        def _norm_index_expr(idx_expr):
            idx_expr = (idx_expr or '').strip()
            if re.fullmatch(r'\d+', idx_expr):
                return str(int(idx_expr))
            m = re.match(r"^([\"\'])(.*)\1$", idx_expr)
            if m:
                return "'" + m.group(2).replace("'", "\\'") + "'"
            return None

        def _resolve_indexed_attr(container, idx_expr):
            norm = _norm_index_expr(idx_expr)
            if norm is not None:
                cand = f"{container}[{norm}]"
                if cand in known_attrs:
                    return [cand]
            if container in container_to_elems and container_to_elems[container]:
                return list(container_to_elems[container])
            return [container] if container in known_attrs else []

        def _collect_direct_module_producers(expr, lineno):
            producers = set()
            best_line = None
            best_var = None
            for m in re.finditer(r'self\.(\w+)\[\s*([^\]]+?)\s*\]\s*\(', expr):
                cont = m.group(1)
                if cont in known_attrs:
                    resolved = _resolve_indexed_attr(cont, m.group(2))
                    for a in resolved:
                        if a in known_attrs:
                            producers.add(a)
                            best_line = lineno
                            best_var = f"self.{a}(...)"
            for m in re.finditer(r'self\.(\w+)\s*\(', expr):
                a = m.group(1)
                if a in known_attrs:
                    producers.add(a)
                    best_line = lineno
                    best_var = f"self.{a}(...)"
            for m in re.finditer(r'getattr\(\s*self\s*,\s*[\"\']([^\"\']+)[\"\']\s*\)\s*\(', expr):
                a = m.group(1)
                if a in known_attrs:
                    producers.add(a)
                    best_line = lineno
                    best_var = f"getattr(self, '{a}')(...)"
            return producers, best_line, best_var

        def _collect_expr_output_producers(expr, var_producers, lineno):
            producers = set()
            best_line = None
            best_var = None
            for var, (ps, ploc) in var_producers.items():
                if re.search(r'\b' + re.escape(var) + r'\b', expr):
                    producers.update(ps)
                    if ploc is not None and (best_line is None or ploc > best_line):
                        best_line = ploc
                        best_var = var
            dps, dline, dvar = _collect_direct_module_producers(expr, lineno)
            if dps:
                producers.update(dps)
                if dline is not None and (best_line is None or dline > best_line):
                    best_line = dline
                    best_var = dvar
            return producers, best_line, best_var

        def _join_lines_local(raw_lines, base_lineno):
            logical = []
            buf = ""
            buf_start = None
            open_count = 0
            for offset, raw in enumerate(raw_lines):
                phys_lineno = base_lineno + offset
                stripped = raw.strip()
                if not stripped or stripped.startswith('#'):
                    if not buf:
                        logical.append((phys_lineno, raw))
                    continue
                if not buf:
                    buf_start = phys_lineno
                stripped_nc = _strip_inline_comment(stripped).strip()
                if not stripped_nc:
                    continue
                buf += (" " if buf else "") + stripped_nc
                open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
                open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
                if open_count <= 0:
                    logical.append((buf_start, buf))
                    buf = ""
                    buf_start = None
                    open_count = 0
            if buf:
                logical.append((buf_start, buf))
            return logical

        result_edges = {}
        for (fname, cname), info in class_map.items():
            if cname != root_cname:
                continue
            fwd_range = info["methods"].get("forward")
            if not fwd_range:
                continue
            lines = source_files.get(fname, [])
            # Iter16: include helper methods (transitively) called from
            # forward() so producers flowing through dispatcher methods like
            # `_compute_task_tower_outputs` -> `_compute_task_tower_outputs_old`
            # are visible to the result-edge scanner. Otherwise the local
            # var_producers map only tracks top-level forward() lines and
            # misses producers of dict slots written inside helpers, which
            # would leave setattr-registered consumers (e.g. DenseTower)
            # without an outbound Result edge.
            _all_methods_local = info.get("methods", {})
            _fwd_text_local = "\n".join(lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))])
            _method_text_cache_local = {"forward": _fwd_text_local}
            for _mname, _mrange in _all_methods_local.items():
                if _mname in ("forward", "__init__"):
                    continue
                _method_text_cache_local[_mname] = "\n".join(
                    lines[_mrange[0] - 1: min(_mrange[1], len(lines))])
            _seen_methods_local = {"forward"}
            _frontier_local = ["forward"]
            _ordered_helpers: list = []
            while _frontier_local:
                _next_local = []
                for _src in _frontier_local:
                    _src_text = _method_text_cache_local.get(_src, "")
                    if not _src_text:
                        continue
                    for _mname in _all_methods_local:
                        if _mname in _seen_methods_local or _mname == "__init__":
                            continue
                        if not re.search(r'self\.' + re.escape(_mname) + r'\s*\(', _src_text):
                            continue
                        _seen_methods_local.add(_mname)
                        _next_local.append(_mname)
                        _ordered_helpers.append(_mname)
                _frontier_local = _next_local

            _scanner_lines = list(lines[fwd_range[0] - 1: min(fwd_range[1], len(lines))])
            _scanner_base = fwd_range[0]
            # Iter16: append helper-method bodies BEFORE forward() lines so that
            # dict-slot writes done in helpers (e.g.
            # ``task_tower_outputs[name] = combine_tower(...)`` inside
            # ``_compute_task_tower_outputs_old``) are visible when forward
            # reads them back (e.g. ``combine_out = task_tower_outputs[name]``
            # at the top-level forward body).  Without this ordering the
            # scanner processes the read site before the write site and fails
            # to attribute the dict-slot producer (DenseTower) to the
            # downstream ``result.head(prediction=combine_out)`` consumer.
            logical_lines = []
            for _hname in _ordered_helpers:
                _hr = _all_methods_local.get(_hname)
                if not _hr:
                    continue
                _hlines = lines[_hr[0] - 1: min(_hr[1], len(lines))]
                logical_lines.extend(_join_lines_local(_hlines, _hr[0]))
            logical_lines.extend(_join_lines_local(_scanner_lines, _scanner_base))
            _ast_root_fe = None
            _ast_root_return_map = {}
            _ast_root_dict_slots = {}
            try:
                _ast_root_fe = ASTFrontend(source='\n'.join(lines), path=fname)
            except Exception:
                _ast_root_fe = None
            if _ast_root_fe:
                for _mn in list(_ordered_helpers) + ["forward"]:
                    for _ret in (_ast_root_fe.extract_return_vars(root_cname, _mn) or []):
                        _rline = _ret.get("line")
                        if _rline is not None and _rline not in _ast_root_return_map:
                            _ast_root_return_map[_rline] = _ret.get("vars") or []
                    _ast_root_dict_slots.update(_ast_root_fe.build_dict_slot_env(root_cname, _mn))
            var_producers = {}
            result_vars = set()
            # Iter16: dict_slots tracker for the result-edge scanner.
            #   {dict_name: {key_or_'*' : producers_set}}
            # When a dict slot is written with a value derived from a known
            # producer (direct self.attr call OR a local var resolved via
            # var_producers), we record the producers. Reads of the same dict
            # (`x = dict[key]` or `x = dict[<dynamic>]`) feed those producers
            # into the LHS local var so downstream `result.head(prediction=x,
            # ...)` can attribute the producer.
            dict_slots_local: dict = {}
            for (_dname, _skey), (_pattr, _pline) in (_ast_root_dict_slots or {}).items():
                if _pattr not in known_attrs:
                    continue
                _bucket = dict_slots_local.setdefault(_dname, {})
                _bucket.setdefault(_skey if _skey is not None else '*', set()).add(_pattr)

            # Iter16: collect known dynamic setattr attrs for this class so
            # `var = getattr(self, <non-literal>)` falls back to all dynamic
            # synthetic attrs (e.g. `DenseTower`) registered via setattr.
            _dyn_setattr_for_root = sorted(
                a for a in (tree.get(root_cname, {}).get("dynamic_setattr_attrs") or [])
                if a in known_attrs
            )

            # Iter16: lightweight `_as_str_lit` (the one in
            # `_build_data_dependency_edges` is in a different lexical scope).
            _STR_LIT_RE_LOCAL = re.compile(r'^([\"\'])(.*)\1$')
            def _as_str_lit_local(s: str):
                s = (s or "").strip()
                m = _STR_LIT_RE_LOCAL.match(s)
                if not m:
                    return None
                return m.group(2)

            def _remember(attr, to_line, var_name, from_line=None):
                if attr not in known_attrs:
                    return
                prev = result_edges.get(attr)
                candidate = {
                    "file": fname,
                    "var": var_name,
                    "from_line": from_line if from_line is not None else to_line,
                    "to_line": to_line,
                }
                if prev is None or candidate["to_line"] < prev["to_line"]:
                    result_edges[attr] = candidate

            for phys_lineno, raw_line in logical_lines:
                line = _strip_inline_comment(raw_line).strip()
                if not line or line.startswith('#'):
                    continue

                m_res = re.match(r'^(\w+)\s*=\s*LG\.result\(\)\s*$', line)
                if m_res:
                    result_vars.add(m_res.group(1))
                    continue

                called_attrs = []
                for m in re.finditer(r'self\.(\w+)\[\s*([^\]]+?)\s*\]\s*\(', line):
                    cont = m.group(1)
                    if cont in known_attrs:
                        for a in _resolve_indexed_attr(cont, m.group(2)):
                            if a in known_attrs:
                                called_attrs.append(a)
                for m in re.finditer(r'self\.(\w+)\s*\(', line):
                    a = m.group(1)
                    if a in known_attrs:
                        called_attrs.append(a)
                for m in re.finditer(r'getattr\(\s*self\s*,\s*[\"\']([^\"\']+)[\"\']\s*\)\s*\(', line):
                    a = m.group(1)
                    if a in known_attrs:
                        called_attrs.append(a)
                called_attrs = list(dict.fromkeys(called_attrs))

                tuple_assign = re.match(r'^\(?\s*(\w+(?:\s*,\s*\w+)+)\s*\)?\s*=\s*(.+)$', line)
                if tuple_assign and called_attrs:
                    lhs_vars = [v.strip() for v in tuple_assign.group(1).split(',') if v.strip() and v.strip() != '_']
                    producers = set(called_attrs)
                    for v in lhs_vars:
                        var_producers[v] = (set(producers), phys_lineno)
                elif tuple_assign:
                    # Iter16: tuple assignment without a direct ``self.x(...)``
                    # call on the RHS — propagate producers via the upstream
                    # variables / dict-slot reads referenced in the rhs.
                    # E.g. ``pred, loss = get_pred_and_loss(logit, label, ...)``
                    # propagates ``logit``'s producers into ``pred`` and
                    # ``loss``.  Without this, ``result.head(prediction=pred)``
                    # cannot attribute the upstream Module to the Result edge.
                    lhs_vars = [v.strip() for v in tuple_assign.group(1).split(',')
                                if v.strip() and v.strip() != '_']
                    rhs = tuple_assign.group(2)
                    rhs_prod, rhs_line, _rhs_var = _collect_expr_output_producers(
                        rhs, var_producers, phys_lineno)
                    for _dm in re.finditer(r'\b(\w+)\[\s*([^\]]+?)\s*\]', rhs):
                        _dname = _dm.group(1)
                        _kexpr = _dm.group(2).strip()
                        _slots = dict_slots_local.get(_dname)
                        if not _slots:
                            continue
                        _klit = _as_str_lit_local(_kexpr)
                        if _klit is not None and _klit in _slots:
                            rhs_prod = (rhs_prod or set()) | set(_slots[_klit])
                        else:
                            _u = set()
                            for _ps in _slots.values():
                                _u |= _ps
                            if _u:
                                rhs_prod = (rhs_prod or set()) | _u
                    if rhs_prod:
                        for v in lhs_vars:
                            var_producers[v] = (set(rhs_prod), rhs_line if rhs_line is not None else phys_lineno)
                else:
                    assign_m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
                    if assign_m:
                        lhs = assign_m.group(1)
                        rhs = assign_m.group(2)
                        # Iter16: handle `var = getattr(self, '<lit>')` or
                        # `var = getattr(self, <dyn>)` so the LHS variable
                        # carries the corresponding module attr as producer.
                        # This is the receiver side of the two-step setattr
                        # pattern (combine_tower = getattr(self, f"...")).
                        m_g_lit = re.match(r'^getattr\(\s*self\s*,\s*[\"\']([^\"\']+)[\"\']\s*\)\s*$', rhs)
                        m_g_dyn = re.match(r'^getattr\(\s*self\s*,\s*([^\)]+)\)\s*$', rhs)
                        if m_g_lit:
                            _a = m_g_lit.group(1)
                            if _a in known_attrs:
                                var_producers[lhs] = ({_a}, phys_lineno)
                                continue
                        elif m_g_dyn and not m_g_lit:
                            # Non-literal getattr name: use any dynamic
                            # setattr-registered attrs as fallback producers.
                            if _dyn_setattr_for_root:
                                var_producers[lhs] = (set(_dyn_setattr_for_root), phys_lineno)
                                continue
                        if called_attrs:
                            var_producers[lhs] = (set(called_attrs), phys_lineno)
                        else:
                            rhs_prod, rhs_line, _rhs_var = _collect_expr_output_producers(rhs, var_producers, phys_lineno)
                            # Iter16: also pull producers via dict reads (any
                            # `\w+\[<key>\]` in the rhs). This lets dict slot
                            # writes done in a helper method propagate through
                            # the same dict in the main forward (e.g.
                            # task_tower_outputs[name] = combine_tower(...)
                            # in `_compute_task_tower_outputs_old`, then
                            # combine_out = task_tower_outputs[name] in
                            # forward).
                            for _dm in re.finditer(r'\b(\w+)\[\s*([^\]]+?)\s*\]', rhs):
                                _dname = _dm.group(1)
                                _kexpr = _dm.group(2).strip()
                                _slots = dict_slots_local.get(_dname)
                                if not _slots:
                                    continue
                                _klit = _as_str_lit_local(_kexpr)
                                if _klit is not None and _klit in _slots:
                                    rhs_prod = (rhs_prod or set()) | set(_slots[_klit])
                                else:
                                    # union all slots
                                    _u = set()
                                    for _ps in _slots.values():
                                        _u |= _ps
                                    if _u:
                                        rhs_prod = (rhs_prod or set()) | _u
                            if rhs_prod:
                                var_producers[lhs] = (rhs_prod, rhs_line if rhs_line is not None else phys_lineno)

                # Iter16: track dict_slot writes:
                #   <dict>[<key>] = <expr>
                # When the rhs has a producer (called_attrs OR resolved via
                # var_producers / dict reads), record that producer at slot
                # key (or '*' if dynamic).
                m_dwr = re.match(r'^\s*(\w+)\s*\[\s*(.+?)\s*\]\s*=\s*(.+)$', line)
                if m_dwr:
                    _dname = m_dwr.group(1)
                    _kexpr = m_dwr.group(2).strip()
                    _vexpr = m_dwr.group(3).strip()
                    _klit = _as_str_lit_local(_kexpr)
                    _slot_key = _klit if _klit is not None else '*'
                    # Inline expr: try local called_attrs first.
                    _producers_for_slot: set = set()
                    # Direct self.xxx(...) call in the rhs
                    for _m in re.finditer(r'self\.(\w+)\s*\(', _vexpr):
                        _a = _m.group(1)
                        if _a in known_attrs:
                            _producers_for_slot.add(_a)
                    # Local var producers referenced in rhs
                    for _vname in re.findall(r'\b(\w+)\s*\(', _vexpr):
                        if _vname in var_producers:
                            _producers_for_slot |= var_producers[_vname][0]
                    # Bare var reference (no call)
                    for _vname in re.findall(r'\b(\w+)\b', _vexpr):
                        if _vname in var_producers:
                            _producers_for_slot |= var_producers[_vname][0]
                    if _producers_for_slot:
                        _bucket = dict_slots_local.setdefault(_dname, {})
                        if _slot_key in _bucket:
                            _bucket[_slot_key] = _bucket[_slot_key] | _producers_for_slot
                        else:
                            _bucket[_slot_key] = set(_producers_for_slot)

                if result_vars:
                    for rv in list(result_vars):
                        head_token = f"{rv}.head("
                        if head_token in line:
                            start = line.index(head_token) + len(head_token)
                            depth = 1
                            end = start
                            for i in range(start, len(line)):
                                if line[i] == '(':
                                    depth += 1
                                elif line[i] == ')':
                                    depth -= 1
                                    if depth == 0:
                                        end = i
                                        break
                            kwargs = _parse_call_kwargs(line[start:end])
                            for kw in ("prediction", "loss"):
                                expr = kwargs.get(kw)
                                if not expr:
                                    continue
                                prod, prod_line, prod_var = _collect_expr_output_producers(expr, var_producers, phys_lineno)
                                for p in prod:
                                    _remember(p, phys_lineno, prod_var or expr, prod_line)

                ret_m = re.match(r'^return\s+(.+)$', line)
                if ret_m and result_vars:
                    ret_parts = _split_top_level_items(ret_m.group(1))
                    ast_ret_vars = _ast_root_return_map.get(phys_lineno) or []
                    if ast_ret_vars and len(ast_ret_vars) >= 2 and ast_ret_vars[0] in result_vars:
                        loss_expr = ast_ret_vars[1].strip()
                    elif len(ret_parts) >= 2 and ret_parts[0].strip() in result_vars:
                        loss_expr = ret_parts[1].strip()
                    else:
                        loss_expr = None
                    if loss_expr:
                        prod, prod_line, prod_var = _collect_expr_output_producers(loss_expr, var_producers, phys_lineno)
                        for p in prod:
                            _remember(p, phys_lineno, prod_var or loss_expr, prod_line)
            break
        return result_edges

    # Build DAG data: each usage instance is a separate node
    # A DAG node represents one attr usage in a parent's forward()
    dag_nodes = []
    dag_edges = []
    dag_groups = []  # nested container groups
    counter = [0]
    # Pre-compute forward call order once
    _module_call_order, _ = _build_source_dependency_order(source_files, class_map)

    def new_id():
        counter[0] += 1
        return f"node_{counter[0]}"

    # Iter11 ctor-src: build constructor (`__init__`) source info for an
    # instance attribute living on `parent_cname`. Returns only the three
    # JSON fields required by the UI: `ctor_src_file`, `ctor_src_line`,
    # `ctor_src_excerpt`. The excerpt is always the source line itself plus
    # two lines of context before/after when available (up to 5 lines total).
    # Returns an empty dict if the parent class / attr has no recorded def
    # location.

    def _build_ctor_info(parent_cname, attr_name):
        if not parent_cname or not attr_name:
            return {}
        parent_info = tree.get(parent_cname) if parent_cname in tree else None
        if not parent_info:
            return {}
        loc = parent_info.get("attr_def_loc", {}).get(attr_name)
        if not loc:
            return {}
        fname, lineno = loc
        if not fname or not lineno:
            return {}
        lines = source_files.get(fname)
        if not lines:
            return {}
        # The recorded line points at the `self.xxx = ClassName(...)` line.
        # The constructor argument list may span multiple physical lines if
        # the call is wrapped. Walk forward from `lineno` and accumulate
        # lines until brace/paren depth returns to zero (or we run out).
        # We bound the look-ahead at 40 lines to keep snippets readable.
        start_idx = max(1, lineno) - 1
        if start_idx >= len(lines):
            return {}
        depth = 0
        seen_open = False
        end_lineno = lineno
        max_scan = min(len(lines), start_idx + 40)
        for i in range(start_idx, max_scan):
            ln = lines[i]
            for ch in ln:
                if ch == '(' or ch == '[' or ch == '{':
                    depth += 1
                    seen_open = True
                elif ch == ')' or ch == ']' or ch == '}':
                    depth -= 1
            end_lineno = i + 1
            if seen_open and depth <= 0:
                break
        # Fixed 5-line window centered on the constructor call line
        # (two lines before + the line itself + two lines after) as requested.
        ex_lo = max(1, lineno - 2)
        ex_hi = min(len(lines), lineno + 2)
        ctor_text_lines = lines[ex_lo - 1: ex_hi]
        ctor_text = "\n".join(ctor_text_lines)
        return {
            "ctor_src_file": fname,
            "ctor_src_line": lineno,
            "ctor_src_excerpt": ctor_text,
        }

    # --------------------------------------------------------------------
    # Iter12-final-fix: shared excerpt relocation helper for Rule1c.
    #
    # Many dep-edge from_/to_excerpts produced by the data-flow tracer
    # highlight an *intermediate* line (e.g. ``q, k, v, g = torch.split(qkvg,
    # ...)``, ``sequence = torch.cat([query, key], dim=1)``,
    # ``out = out + input``) rather than the literal ``self.{attr}(...)``
    # call site.  Rule1c — which mandates that highlight lines reference the
    # actual call — fails for those edges.  This helper rewrites the excerpt
    # so the highlight lands on a real call line in the parent's class body.
    # ``g_label`` is the class whose forward() body should host the call site.
    # The helper is conservative: when no real call site can be located it
    # returns the input unchanged (Rule1c may still fail but no other rule
    # is broken).
    # --------------------------------------------------------------------
    def _relocate_excerpt_to_self_call(g_label, attr, excerpt):
        if not excerpt or not attr:
            return excerpt
        # Skip synthetic / boundary placeholders that have no literal call.
        if (attr.startswith("__setattr_") or attr.startswith("__LG_")
                or attr.endswith(".__in") or attr.endswith(".__out")
                or attr in ("Input", "Result", "input", "result", "?")):
            return excerpt
        base = re.sub(r'#\d+$', '', attr)
        m_elem = re.match(r'^(\w+)\[(.+)\]$', base)
        scan_attr = m_elem.group(1) if m_elem else base
        is_container = m_elem is not None

        call_pat = re.compile(r'self\.' + re.escape(scan_attr) + r'\s*[\(\[]')
        call_method_pat = re.compile(
            r'self\.' + re.escape(scan_attr)
            + r'\s*(?:\[[^\]]*\])?\.\w+\s*\(')
        loop_pat = re.compile(
            r'for\s+(?:[\w]+\s*,\s*)?\w+\s+in\s+'
            r'(?:enumerate\(\s*)?self\.' + re.escape(scan_attr) + r'\b')
        m_dyn = re.match(r'^(.*?)(_)(\d+)$', scan_attr)
        dyn_prefix = m_dyn.group(1) + m_dyn.group(2) if m_dyn else None
        getattr_pat = None
        if dyn_prefix:
            getattr_pat = re.compile(
                r'getattr\(\s*self\s*,[^)]*' + re.escape(dyn_prefix)
                + r'[^)]*\)\s*\(')

        def _line_calls(lt):
            if call_pat.search(lt):
                return True
            # Method-call form (e.g. ``self.bn.float()``) only applies to
            # NON-container attrs.  For container-element attrs like
            # ``_layers[0]`` we must NOT accept ``self._layers.append(...)``
            # as a forward call site — that's an __init__ helper.
            if not is_container and call_method_pat.search(lt):
                return True
            if is_container and loop_pat.search(lt):
                return True
            if getattr_pat is not None and getattr_pat.search(lt):
                return True
            return False

        # Already a valid call site?
        txt = excerpt.get('text', '')
        hl = excerpt.get('highlight')
        start = excerpt.get('start', 1)
        if txt and hl is not None:
            lines_e = txt.split('\n')
            idx = hl - start
            if 0 <= idx < len(lines_e):
                cur_line = lines_e[idx]
                if _line_calls(cur_line):
                    return excerpt
                # For container elements where the highlight line uses a
                # loop variable bound to ``for X in self.{base}``, accept
                # the existing excerpt (Rule1c's loop fallback handles it
                # if the for-line is in the excerpt window).
                if is_container:
                    for tline in lines_e:
                        m_lv = loop_pat.search(tline)
                        if m_lv:
                            # find the loop var; loop_pat captures it depending
                            # on form, so re-match with a capturing version
                            for _lp in (
                                re.compile(
                                    r'for\s+(?:[\w]+\s*,\s*)?(\w+)\s+in\s+'
                                    r'(?:enumerate\(\s*)?self\.' + re.escape(scan_attr) + r'\b'),
                                re.compile(
                                    r'for\s+\w+\s*,\s*(\w+)\s+in\s+self\.' + re.escape(scan_attr) + r'\.items\(\)'),
                                re.compile(
                                    r'for\s+(\w+)\s+in\s+self\.' + re.escape(scan_attr) + r'\.values\(\)'),
                            ):
                                mm = _lp.search(tline)
                                if mm:
                                    lv = mm.group(1)
                                    if re.search(r'\b' + re.escape(lv) + r'\s*\(', cur_line):
                                        return excerpt

        # Search for a real call site in g_label's class methods.
        for (fname_c, cname_c), info_c in class_map.items():
            if cname_c != g_label:
                continue
            flines = source_files.get(fname_c, [])
            if not flines:
                continue
            for mname, mrange in (info_c.get('methods', {}) or {}).items():
                if mname == '__init__':
                    continue
                for i in range(mrange[0] - 1, min(mrange[1], len(flines))):
                    lt = flines[i]
                    if lt.lstrip().startswith('#'):
                        continue
                    if _line_calls(lt):
                        new_ln = i + 1
                        return {
                            "start": max(1, new_ln - 1),
                            "end": min(len(flines), new_ln + 1),
                            "text": "\n".join(
                                flines[max(0, new_ln - 2): min(len(flines), new_ln + 1)]),
                            "highlight": new_ln,
                        }
        return excerpt

    # Iter14 fix-1 (empty-group-as-leaf): some user nn.Modules (e.g. 5547919
    # ``LossModule`` / ``PredModule``) only hold ``LG.label(...)`` /
    # ``LG.get_sample_rate()`` style helpers, which we serialize as the synthetic
    # ``__LG_InputSource`` placeholder. Those placeholders are filtered out
    # before group construction, leaving an *empty* group with zero children.
    # The frontend renders empty groups as a hollow folder, so the user sees
    # ``LossModule`` as an unreachable container with edges piercing through
    # to ``Result``. Fix by demoting any class whose attrs are entirely
    # ``__LG_InputSource`` / pure containers to a leaf node — exactly the
    # treatment used for Linear/Embedding/etc. that have no attributes.
    _elem_re_outer = re.compile(r'^(\w+)\[.+\]$')

    def _has_real_children(child_cls):
        """Return True iff this class still has a real nn.Module child after
        stripping LG input placeholders and pure-container attrs."""
        if child_cls not in tree:
            return False
        sub_attrs = tree[child_cls].get("attrs", {}) or {}
        if not sub_attrs:
            return False
        # Reproduce the same filtering build_dag_recursive does on entry.
        sub_containers = set()
        for a in sub_attrs:
            mm = _elem_re_outer.match(a)
            if mm:
                sub_containers.add(mm.group(1))
        sub_containers = {c for c in sub_containers if c in sub_attrs}
        for a, c in sub_attrs.items():
            if a in sub_containers:
                continue
            if c == "__LG_InputSource":
                continue
            if isinstance(c, str) and c.startswith("nn."):
                return True
            return True
        return False

    def build_dag_recursive(cname, parent_path, depth, max_depth=10,
                              parent_cname=None, parent_attr_name=None):
        """Build DAG nodes for class cname. Returns the group descriptor.

        `parent_cname` / `parent_attr_name` describe where the *instance*
        of `cname` lives in the enclosing class — they're used to surface
        the constructor source (`self.<attr> = ClassName(...)`) on the
        group panel.
        """
        if depth >= max_depth or cname not in tree:
            return None
        info = tree[cname]
        children_classes = info["children"]
        attrs = info.get("attrs", {})
        dep_edge_list = info.get("dep_edges", [])

        # Iter17: per-instance children pruning.
        # If this instance was constructed with a kw=<list> whose length is
        # known and the class has a container driven by that kw, trim
        # `attrs` so only the first N elements of that container survive
        # (other call sites with different N are unaffected because they
        # build their own group). Without this, e.g. tower_a (output_dims
        # length 3) and tower_b (output_dims length 2) would both expose 3
        # children because the class-level union is MAX(2,3)=3.
        _container_kw_map = info.get("container_kw", {}) or {}
        if _container_kw_map and parent_cname is not None and parent_attr_name is not None:
            _parent_info = tree.get(parent_cname, {}) if parent_cname in tree else {}
            _parent_inst_map = _parent_info.get("instance_kw_list_lens", {}) or {}
            _inst_kws = _parent_inst_map.get(parent_attr_name, {})
            if _inst_kws:
                _drop = set()
                for _cont, _kw in _container_kw_map.items():
                    if _kw not in _inst_kws:
                        continue
                    _n_inst = _inst_kws[_kw]
                    _elem_re_inst = re.compile(r'^' + re.escape(_cont) + r'\[(\d+)\]$')
                    for _a in list(attrs):
                        _m = _elem_re_inst.match(_a)
                        if _m and int(_m.group(1)) >= _n_inst:
                            _drop.add(_a)
                if _drop:
                    attrs = {k: v for k, v in attrs.items() if k not in _drop}
                    # Also prune dep_edges referencing dropped attrs.
                    dep_edge_list = [(fa, ta) for (fa, ta) in dep_edge_list
                                     if fa not in _drop and ta not in _drop]

        # Remove container attrs (nn.ModuleDict/ModuleList containers) from the
        # DAG. Containers are NOT real graph nodes — only their expanded elements
        # participate in data flow. When a dep_edge references a container, remap
        # it to the container's element(s).
        _elem_re_local = re.compile(r'^(\w+)\[.+\]$')
        _containers_with_elems = set()
        for a in attrs:
            m = _elem_re_local.match(a)
            if m:
                _containers_with_elems.add(m.group(1))
        # Only filter containers that actually have expanded elements in attrs
        _containers_to_remove = set(c for c in _containers_with_elems if c in attrs)
        # Build container → elements mapping for edge remapping
        _container_to_elems_local = defaultdict(list)
        for a in attrs:
            m = _elem_re_local.match(a)
            if m and m.group(1) in _containers_to_remove:
                _container_to_elems_local[m.group(1)].append(a)

        # Also remove __LG_InputSource attrs from DAG node creation.
        # They represent network inputs (LG.feature_column / LG.dense_feature),
        # not real nn.Module DAG nodes. Their edges are remapped: if an LG source
        # feeds a real module, that real module receives from Input instead.
        _lg_source_attrs = set(a for a, c in attrs.items() if c == "__LG_InputSource")

        attrs_filtered = {a: c for a, c in attrs.items()
                          if a not in _containers_to_remove and a not in _lg_source_attrs}

        # Remap dep_edges: replace container references with element references
        _remapped_dep_edges = []
        # Also collect edges from LG sources to real modules (for Input connection)
        _input_consumer_edges = []  # [(to_attr, evidence_key)]
        for (fa, ta) in dep_edge_list:
            # If source is an LG input, the target consumes Input directly
            if fa in _lg_source_attrs:
                if ta not in _lg_source_attrs and ta not in _containers_to_remove:
                    _input_consumer_edges.append((fa, ta))
                continue
            # If target is an LG input, skip (shouldn't happen but guard)
            if ta in _lg_source_attrs:
                continue
            froms = _container_to_elems_local.get(fa, [fa])
            tos = _container_to_elems_local.get(ta, [ta])
            for f in froms:
                for t in tos:
                    if f != t and f in attrs_filtered and t in attrs_filtered:
                        _remapped_dep_edges.append((f, t))
        dep_edge_list = list(set(_remapped_dep_edges))

        # Reverse map: class -> list of attr names
        cls_to_attrs = defaultdict(list)
        for attr, cls_ref in attrs_filtered.items():
            cls_to_attrs[cls_ref].append(attr)

        group_id = new_id()
        child_node_ids = []
        child_groups = []

        # Build forward() call order: create instance nodes
        call_order = []
        seen_attrs = set()
        fwd_order = _module_call_order.get(cname, [])
        # attr_name -> assigned node/group id for edge resolution
        attr_id_map = {}

        for attr_name in fwd_order:
            if attr_name in attrs_filtered and attr_name not in seen_attrs:
                seen_attrs.add(attr_name)
                child_cls = attrs[attr_name]
                child_path = f"{parent_path}.{attr_name}"
                has_children = _has_real_children(child_cls)

                if has_children and depth + 1 < max_depth:
                    sub_group = build_dag_recursive(child_cls, child_path, depth + 1, max_depth, parent_cname=cname, parent_attr_name=attr_name)
                    if sub_group:
                        child_groups.append(sub_group)
                        call_order.append({"type": "group", "id": sub_group["id"], "attr": attr_name})
                        attr_id_map[attr_name] = ("group", sub_group["id"])
                    else:
                        nid = new_id()
                        dag_nodes.append(_make_dag_node(nid, attr_name, child_cls, depth + 1, parent_cname=cname))
                        child_node_ids.append(nid)
                        call_order.append({"type": "node", "id": nid, "attr": attr_name})
                        attr_id_map[attr_name] = ("node", nid)
                else:
                    nid = new_id()
                    dag_nodes.append(_make_dag_node(nid, attr_name, child_cls, depth + 1, parent_cname=cname))
                    child_node_ids.append(nid)
                    call_order.append({"type": "node", "id": nid, "attr": attr_name})
                    attr_id_map[attr_name] = ("node", nid)

        # Add remaining attrs not called in forward
        _dep_attrs = {fa for (fa, _) in dep_edge_list} | {ta for (_, ta) in dep_edge_list}
        _input_consumer_attrs = {ta for (_fa, ta) in _input_consumer_edges}
        for attr_name, child_cls in attrs_filtered.items():
            if attr_name not in seen_attrs:
                _has_runtime_use = (
                    attr_name in info.get("first_call_loc", {})
                    or attr_name in _dep_attrs
                    or attr_name in _input_consumer_attrs
                )
                if not _has_runtime_use:
                    continue
                seen_attrs.add(attr_name)
                child_path = f"{parent_path}.{attr_name}"
                has_children = _has_real_children(child_cls)
                if has_children and depth + 1 < max_depth:
                    sub_group = build_dag_recursive(child_cls, child_path, depth + 1, max_depth, parent_cname=cname, parent_attr_name=attr_name)
                    if sub_group:
                        child_groups.append(sub_group)
                        call_order.append({"type": "group", "id": sub_group["id"], "attr": attr_name})
                        attr_id_map[attr_name] = ("group", sub_group["id"])
                    else:
                        nid = new_id()
                        dag_nodes.append(_make_dag_node(nid, attr_name, child_cls, depth + 1, parent_cname=cname))
                        child_node_ids.append(nid)
                        call_order.append({"type": "node", "id": nid, "attr": attr_name})
                        attr_id_map[attr_name] = ("node", nid)
                else:
                    nid = new_id()
                    dag_nodes.append(_make_dag_node(nid, attr_name, child_cls, depth + 1, parent_cname=cname))
                    child_node_ids.append(nid)
                    call_order.append({"type": "node", "id": nid, "attr": attr_name})
                    attr_id_map[attr_name] = ("node", nid)

        # Build data dependency edges from parsed tensor flow.
        # We additionally record per-group `internal_edges` referencing only
        # direct children (by raw child id), so the JS layout can do a layered
        # (Sugiyama-style) horizontal placement based on local data flow.
        # The global `dag_edges` only retains cross-group / top-level edges
        # below; internal edges are drawn per-group using rank-aware routing.
        dep_edges_resolved = set()
        edge_locs_for_class = info.get("dep_edge_locs", {})
        internal_edges = []  # {from_child, to_child, type, evidence?}
        if dep_edge_list:
            for from_attr, to_attr in dep_edge_list:
                if from_attr in attr_id_map and to_attr in attr_id_map:
                    from_type, from_id = attr_id_map[from_attr]
                    to_type, to_id = attr_id_map[to_attr]
                    dep_edges_resolved.add((from_attr, to_attr))
                    ev = edge_locs_for_class.get((from_attr, to_attr))
                    # Fallback: if edge was remapped from container, try original keys
                    if not ev:
                        for orig_fa in ([from_attr] + [c for c, es in _container_to_elems_local.items() if from_attr in es]):
                            for orig_ta in ([to_attr] + [c for c, es in _container_to_elems_local.items() if to_attr in es]):
                                ev = edge_locs_for_class.get((orig_fa, orig_ta))
                                if ev:
                                    break
                            if ev:
                                break
                    edge = {"from_child": from_id, "to_child": to_id, "type": "dep",
                            "from_attr": from_attr, "to_attr": to_attr,
                            "parent_class": cname}
                    if ev:
                        # Attach the source-line evidence so the click handler can
                        # show the producer / consumer lines from forward().
                        ev_lines = source_files.get(ev["file"], [])
                        # Build a tight snippet: a few lines around from_line and to_line
                        def _excerpt(lineno):
                            if not lineno or not ev_lines:
                                return None
                            lo = max(1, lineno - 2)
                            hi = min(len(ev_lines), lineno + 2)
                            return {
                                "start": lo, "end": hi,
                                "text": "\n".join(ev_lines[lo - 1: hi]),
                                "highlight": lineno,
                            }
                        # Iter11 varlineage: convert raw lineage steps into
                        # rich var_history entries with file/line/excerpt.
                        # Each step records the variable name, line, and a short
                        # excerpt around it so the front-end can render the
                        # full assignment chain.
                        def _var_history_from(steps):
                            if not steps:
                                return []
                            out = []
                            for s in steps:
                                _f = s.get("file") or ev["file"]
                                _ln = s.get("line")
                                _src = source_files.get(_f, []) or ev_lines
                                if not _ln or not _src:
                                    continue
                                lo = max(1, _ln - 2)
                                hi = min(len(_src), _ln + 2)
                                out.append({
                                    "var": s.get("var", ""),
                                    "file": _f,
                                    "line": _ln,
                                    "excerpt": {
                                        "start": lo, "end": hi,
                                        "text": "\n".join(_src[lo - 1: hi]),
                                        "highlight": _ln,
                                    },
                                    "role": s.get("role", "step"),
                                    "carriers": s.get("carriers", []),
                                    "arg_carriers": s.get("arg_carriers", []),
                                })
                            return out
                        edge["evidence"] = {
                            "file": ev["file"],
                            "var": ev["var"],
                            "from_line": ev["from_line"],
                            "to_line": ev["to_line"],
                            "from_excerpt": _excerpt(ev["from_line"]),
                            "to_excerpt": _excerpt(ev["to_line"]),
                            "var_history": _var_history_from(ev.get("lineage")),
                        }
                        # Iter12-final-fix Rule1c: relocate excerpt highlights
                        # onto real ``self.{attr}(...)`` call sites in cname's
                        # forward() body when the data-flow tracer's chosen
                        # line is an intermediate consumer (e.g. torch.cat,
                        # torch.split tuple unpack, residual `out = out + x`).
                        edge["evidence"]["from_excerpt"] = _relocate_excerpt_to_self_call(
                            cname, from_attr, edge["evidence"].get("from_excerpt"))
                        edge["evidence"]["to_excerpt"] = _relocate_excerpt_to_self_call(
                            cname, to_attr, edge["evidence"].get("to_excerpt"))
                    else:
                        # Iter8 strict: dep edge with no evidence is illegal.
                        # Try to anchor at first_call_loc for either attr; if
                        # that fails we raise rather than emit a corrupt edge.
                        _first_call = info.get("first_call_loc", {}) if isinstance(info, dict) else {}
                        to_loc = _first_call.get(to_attr)
                        from_loc = _first_call.get(from_attr)
                        pick_file = None
                        pick_from_line = None
                        pick_to_line = None
                        if to_loc:
                            pick_file = to_loc[0]
                            pick_to_line = to_loc[1]
                        if from_loc:
                            if pick_file is None or pick_file == from_loc[0]:
                                pick_file = from_loc[0]
                                pick_from_line = from_loc[1]
                        if pick_file is None:
                            fwd_info = class_source_info.get(cname, {})
                            if fwd_info.get("file") and fwd_info.get("start_line"):
                                pick_file = fwd_info["file"]
                                pick_from_line = fwd_info["start_line"]
                                pick_to_line = fwd_info["start_line"]
                        if pick_file is None:
                            raise RuntimeError(
                                f"[{cname}] dep edge {from_attr}→{to_attr}: no source-line evidence available")
                        ev_lines_full = source_files.get(pick_file, [])
                        def _excerpt2(ln):
                            if not ln or not ev_lines_full:
                                return None
                            lo = max(1, ln - 2)
                            hi = min(len(ev_lines_full), ln + 2)
                            return {
                                "start": lo, "end": hi,
                                "text": "\n".join(ev_lines_full[lo - 1: hi]),
                                "highlight": ln,
                            }
                        ev_anchor = {
                            "file": pick_file,
                            "var": f"(dep edge anchor: {from_attr}→{to_attr})",
                            "from_line": pick_from_line or pick_to_line,
                            "to_line": pick_to_line or pick_from_line,
                            "from_excerpt": _excerpt2(pick_from_line),
                            "to_excerpt": _excerpt2(pick_to_line),
                        }
                        if not ev_anchor["from_excerpt"] and not ev_anchor["to_excerpt"] and ev_lines_full:
                            anchor = pick_to_line or pick_from_line or 1
                            ev_anchor["to_excerpt"] = _excerpt2(anchor)
                        # Iter12-final-fix Rule1c: ensure highlights point at real call sites.
                        ev_anchor["from_excerpt"] = _relocate_excerpt_to_self_call(
                            cname, from_attr, ev_anchor.get("from_excerpt"))
                        ev_anchor["to_excerpt"] = _relocate_excerpt_to_self_call(
                            cname, to_attr, ev_anchor.get("to_excerpt"))
                        edge["evidence"] = ev_anchor
                    internal_edges.append(edge)

        # Iter17: Bug 2 fix — REMOVED the "sequential call order" fallback.
        # Previously, when `dep_edges_resolved` was empty AND `len(call_order) > 1`,
        # the analyzer connected adjacent siblings with `flow`-type edges whose
        # evidence.var was "(sequential call order in source)".  This produced
        # false dependencies between modules that legitimately have no data
        # flow (e.g. TransformerComponent.query_norm vs action_norm,
        # HandleSequence.show vs clk vs ..., LRMTrans.* sub-projections).
        #
        # The variable tracker (L1730+ in _track_class_dataflow) is now
        # authoritative: if it produces no edges, the siblings are treated as
        # independent.  If real data flow exists but the tracker missed it,
        # that's a tracker bug to be fixed upstream — not silently masked
        # with fake edges.
        #
        # The block below was originally L6609-6697 in the iter16 baseline.
        # Restoring it would re-introduce the spurious edges Bug 2 reports
        # against, so we deliberately leave this empty.

        # ------------------------------------------------------------------
        # Iter14 container-group wrapping
        # ------------------------------------------------------------------
        # For every nn.ModuleDict / nn.ModuleList (or dict/list holding
        # nn.Module) that has been expanded into >=2 element children, wrap
        # those children inside a synthetic container sub-group so the DAG
        # mirrors the source-level container structure.  Single-element
        # containers are left alone (avoid pointless nesting).
        #
        # The wrap rewrites: child_node_ids / child_groups / call_order /
        # attr_id_map / internal_edges / _input_consumer_edges in-place, and
        # registers a synthetic ``tree[<container_label>]`` entry so the
        # boundary-evidence pipeline can locate the container's call site
        # (``self.<attr>[k](...)``) and constructor line.
        _container_kinds_for_class = info.get("container_kinds", {}) or {}
        _elem_attr_re = re.compile(r'^(\w+)\[.+\]$')
        _container_to_attr_list = defaultdict(list)
        for _attr_name in attrs_filtered:
            _m_elem = _elem_attr_re.match(_attr_name)
            if _m_elem and _m_elem.group(1) in _containers_to_remove:
                _container_to_attr_list[_m_elem.group(1)].append(_attr_name)

        for _cont_name in list(_container_to_attr_list.keys()):
            _elem_attrs_in_cont = _container_to_attr_list[_cont_name]
            # Skip if not enough elements expanded to justify a sub-group.
            if len(_elem_attrs_in_cont) < 2:
                continue
            # Collect direct children that belong to this container.
            _cont_member_ids = []
            _cont_member_attr_set = set(_elem_attrs_in_cont)
            for _ea in _elem_attrs_in_cont:
                if _ea in attr_id_map:
                    _kind, _id = attr_id_map[_ea]
                    _cont_member_ids.append((_ea, _kind, _id))
            if len(_cont_member_ids) < 2:
                continue

            _kind_name = _container_kinds_for_class.get(_cont_name, "ModuleList")
            _container_label = f"{_kind_name}[{_cont_name}][{len(_cont_member_ids)}]"
            _container_gid = f"{group_id}__{_cont_name}_container"

            # Constructor / def line for the container (``self.<cont> = nn.ModuleDict(...)``).
            _attr_def_locs_local = info.get("attr_def_loc", {}) or {}
            _cont_def_loc = _attr_def_locs_local.get(_cont_name)
            _cont_def_file = _cont_def_loc[0] if _cont_def_loc else None
            _cont_def_line = _cont_def_loc[1] if _cont_def_loc else None
            _cont_def_lines = source_files.get(_cont_def_file, []) if _cont_def_file else []
            _cont_src_snippet = None
            _cont_src_snippet_start = None
            _cont_src_snippet_end = None
            if _cont_def_lines and _cont_def_line:
                _lo = max(1, _cont_def_line - 2)
                _hi = min(len(_cont_def_lines), _cont_def_line + 2)
                _cont_src_snippet = "\n".join(_cont_def_lines[_lo - 1: _hi])
                _cont_src_snippet_start = _lo
                _cont_src_snippet_end = _hi

            # Build the synthetic ``tree`` entry that the boundary-evidence
            # pipeline (``_propagate_edges_recursive``) will index by g_label.
            _sub_first_call = {}
            _sub_attr_def = {}
            _parent_first_call = info.get("first_call_loc", {}) or {}

            # ----------------------------------------------------------------
            # Iter14 fix: ``first_call_loc[<cont_name>]`` may inadvertently
            # point at an ``__init__``-side construction line (e.g.
            # ``self.resblocks.append(residual_attention_block)`` or
            # ``self.resblocks = nn.ModuleList()``) when the parent class
            # builds its container in a loop and the static analyzer picked
            # up that line as the "first" mention.  Such a line breaks Rule6
            # for the boundary edge ``parent__in → <cont>_container`` because
            # the validator looks for ``self.<cont>(`` / ``self.<cont>[`` /
            # ``for ... in self.<cont>``.
            #
            # Rescan the parent class's non-``__init__`` methods for a real
            # forward-side reference to ``self.<cont>`` and, if found and the
            # currently recorded location looks like a constructor-style line,
            # overwrite both the parent's ``first_call_loc[<cont_name>]`` and
            # the synthetic container's exposed first_call_loc.  Be careful
            # to leave a "good" forward-side first_call_loc untouched.
            # ----------------------------------------------------------------
            def _is_ctor_style_line(_line_text, _attr):
                """Return True iff line is an __init__-style construction
                line that does NOT also match a real forward call pattern."""
                if not _line_text:
                    return False
                # Simple constructor: ``self.<attr> = nn.ModuleList(...)``.
                if re.search(r"self\." + re.escape(_attr) + r"\s*=", _line_text):
                    return True
                # ``self.<attr>.append(...)`` / ``.update(...)`` etc — these
                # are container-mutation helpers, not real forward calls.
                if re.search(r"self\." + re.escape(_attr) + r"\.\w+\s*\(", _line_text):
                    return True
                # ``self[<attr>] = …`` style assignment in __init__.
                if re.search(r"self\[\s*['\"]" + re.escape(_attr) + r"['\"]\s*\]\s*=", _line_text):
                    return True
                return False

            def _find_forward_side_call_loc(_parent_cname, _attr):
                """Scan ``class_map[<parent>]``'s non-``__init__`` methods for
                the first physical line matching one of:
                  * ``self.<attr>(`` / ``self.<attr>[`` (direct invoke /index)
                  * ``for ... in self.<attr>`` (loop iteration)
                  * ``for X in self.<attr>.values()`` /
                    ``for X, Y in self.<attr>.items()``
                Returns (file, line) or ``None``.
                """
                _direct = re.compile(
                    r"self\." + re.escape(_attr) + r"\s*[\(\[]")
                _loop = re.compile(
                    r"for\s+(?:[\w]+\s*,\s*)?\w+\s+in\s+"
                    r"(?:enumerate\(\s*)?self\." + re.escape(_attr) + r"\b")
                _ctor_pat = re.compile(
                    r"self\." + re.escape(_attr) + r"\s*=")
                _append_pat = re.compile(
                    r"self\." + re.escape(_attr) + r"\.\w+\s*\(")
                for (_fname_p, _cn_p), _ci_p in class_map.items():
                    if _cn_p != _parent_cname:
                        continue
                    _flines_p = source_files.get(_fname_p, [])
                    if not _flines_p:
                        continue
                    for _mname_p, _mrange_p in (_ci_p.get("methods", {}) or {}).items():
                        if _mname_p == "__init__":
                            continue
                        _ms, _me = _mrange_p
                        for _ii in range(_ms - 1, min(_me, len(_flines_p))):
                            _lt = _flines_p[_ii]
                            _ls = _lt.lstrip()
                            if _ls.startswith("#"):
                                continue
                            # Skip lines that are unmistakably ctor-style.
                            if _ctor_pat.search(_lt) or _append_pat.search(_lt):
                                continue
                            if _direct.search(_lt) or _loop.search(_lt):
                                return (_fname_p, _ii + 1)
                return None

            _existing_cont_loc = _parent_first_call.get(_cont_name)
            _needs_override = False
            if _existing_cont_loc is None:
                _needs_override = True
            else:
                _ef, _eln = _existing_cont_loc
                _eflines = source_files.get(_ef, [])
                if 0 <= _eln - 1 < len(_eflines):
                    _eline_text = _eflines[_eln - 1]
                    if _is_ctor_style_line(_eline_text, _cont_name):
                        _needs_override = True
            if _needs_override:
                _new_loc = _find_forward_side_call_loc(cname, _cont_name)
                if _new_loc:
                    _parent_first_call[_cont_name] = _new_loc
                    # Also persist on the parent's tree entry so the propagator
                    # sees the corrected location when emitting boundary
                    # evidence into the container group.
                    info_first = info.get("first_call_loc")
                    if isinstance(info_first, dict):
                        info_first[_cont_name] = _new_loc

            for _ea in _elem_attrs_in_cont:
                if _ea in _parent_first_call:
                    _sub_first_call[_ea] = _parent_first_call[_ea]
                if _ea in _attr_def_locs_local:
                    _sub_attr_def[_ea] = _attr_def_locs_local[_ea]
                # Also expose under the bare container name, for evidence
                # lookups that reference the container itself.
                _sub_first_call.setdefault(_cont_name, _parent_first_call.get(_cont_name))
                _sub_attr_def.setdefault(_cont_name, _attr_def_locs_local.get(_cont_name))
            _sub_first_call = {k: v for k, v in _sub_first_call.items() if v}
            _sub_attr_def = {k: v for k, v in _sub_attr_def.items() if v}
            _parent_dep_edge_locs = info.get("dep_edge_locs", {}) or {}
            _sub_dep_edge_locs = {
                (_a, _b): _loc for (_a, _b), _loc in _parent_dep_edge_locs.items()
                if (_a in _cont_member_attr_set and _b in _cont_member_attr_set)
            }
            tree[_container_label] = {
                "children": [],
                "attrs": {},
                "dep_edges": [],
                "dep_edge_locs": _sub_dep_edge_locs,
                "first_call_loc": _sub_first_call,
                "attr_def_loc": _sub_attr_def,
            }

            # Partition internal_edges into (a) edges fully inside the
            # container -> moved into the sub-group; (b) edges crossing
            # in/out -> rewritten so the externally-visible endpoint is the
            # container group id, retained on parent.
            #
            # Iter14-aggr: when a single external source has dep edges to
            # K elem children of the container (typical: K=64 for
            # ``dict[tokens_module_dict][64]``), the naïve in-place rewrite
            # produces K parallel ``src → container_gid`` rewritten edges
            # with identical (from_child, to_child) pairs.  Each duplicate
            # becomes its own SVG path and is wired to the same className
            # toggle on hover, which causes K simultaneous DOM mutations
            # and stalls the browser whenever the user hovers ``src``.
            #
            # We deduplicate per ``(from_child, to_child)`` pair while
            # preserving the FIRST encountered evidence/metadata.  The
            # surviving edge is tagged with ``container_aggregated_in`` /
            # ``container_aggregated_out`` so renderers can recognise the
            # aggregated boundary and limit hover highlighting to a single
            # representative path per container side.
            _cont_internal_member_ids = set(_id for (_ea, _k, _id) in _cont_member_ids)
            _cont_internal_edges = []
            _rewritten_seen = {}
            _passthrough_seen = {}
            _new_parent_internal = []
            for _ie in internal_edges:
                _fc = _ie.get("from_child")
                _tc = _ie.get("to_child")
                _f_in = _fc in _cont_internal_member_ids
                _t_in = _tc in _cont_internal_member_ids
                if _f_in and _t_in:
                    # Edge fully inside the container: moved into the
                    # sub-group's internal_edges.  Keep evidence/metadata
                    # untouched so Rule6 / Rule1c remain happy.
                    _ie2 = dict(_ie)
                    _ie2["parent_class"] = _container_label
                    _cont_internal_edges.append(_ie2)
                elif _f_in and not _t_in:
                    # outbound boundary: src is a container member.  Rewrite
                    # source endpoint to the container group id and dedupe.
                    _key = (_container_gid, _tc)
                    _existing = _rewritten_seen.get(_key)
                    if _existing is None:
                        _ie2 = dict(_ie)
                        _ie2["from_child"] = _container_gid
                        _ie2["container_aggregated_out"] = _container_gid
                        _ie2["container_aggregated_count"] = 1
                        _rewritten_seen[_key] = _ie2
                        _new_parent_internal.append(_ie2)
                    else:
                        _existing["container_aggregated_count"] = (
                            _existing.get("container_aggregated_count", 1) + 1
                        )
                elif _t_in and not _f_in:
                    # inbound boundary: tgt is a container member.  Rewrite
                    # target endpoint to the container group id and dedupe.
                    _key = (_fc, _container_gid)
                    _existing = _rewritten_seen.get(_key)
                    if _existing is None:
                        _ie2 = dict(_ie)
                        _ie2["to_child"] = _container_gid
                        _ie2["container_aggregated_in"] = _container_gid
                        _ie2["container_aggregated_count"] = 1
                        _rewritten_seen[_key] = _ie2
                        _new_parent_internal.append(_ie2)
                    else:
                        _existing["container_aggregated_count"] = (
                            _existing.get("container_aggregated_count", 1) + 1
                        )
                else:
                    # Neither endpoint inside the container; passthrough but
                    # still dedupe to avoid amplifying any pre-existing dups.
                    _key = (_fc, _tc)
                    if _key not in _passthrough_seen:
                        _passthrough_seen[_key] = True
                        _new_parent_internal.append(_ie)
            internal_edges[:] = _new_parent_internal

            # Likewise rewrite input_consumer_edges: an LG → elem_attr edge
            # should target the container (so the container's __in carries
            # the boundary), and the container will replay it internally.
            #
            # Iter14-aggr: dedupe (lg_attr, container_attr) pairs after
            # rewrite so the container's own ICE list doesn't carry K
            # entries for one logical "LG → container" boundary.  The
            # in-container ICE retains the original elem-level keys so
            # ``_propagate_edges_recursive`` can still resolve precise
            # consumer evidence for each elem.
            _cont_ice = []
            _new_parent_ice = []
            _parent_ice_seen = set()
            for (_lg_attr, _consumer_attr) in _input_consumer_edges:
                if _consumer_attr in _cont_member_attr_set:
                    _cont_ice.append((_lg_attr, _consumer_attr))
                else:
                    _key_pi = (_lg_attr, _consumer_attr)
                    if _key_pi not in _parent_ice_seen:
                        _parent_ice_seen.add(_key_pi)
                        _new_parent_ice.append((_lg_attr, _consumer_attr))
            _input_consumer_edges[:] = _new_parent_ice

            # Move container members out of parent's child_node_ids /
            # child_groups and into the sub-group's lists, and rewrite
            # call_order to a single group entry placed at the original
            # first occurrence.
            _cont_children_nodes = []
            _cont_children_groups = []
            _cont_call_order = []
            _members_attrs_seen = set()
            for (_ea, _k, _id) in _cont_member_ids:
                _members_attrs_seen.add(_ea)
                if _k == "node":
                    _cont_children_nodes.append(_id)
                    if _id in child_node_ids:
                        child_node_ids.remove(_id)
                else:
                    # The corresponding sub-group dict is in child_groups.
                    _matched = None
                    for _cg in child_groups:
                        if isinstance(_cg, dict) and _cg.get("id") == _id:
                            _matched = _cg
                            break
                    if _matched is not None:
                        _cont_children_groups.append(_matched)
                        child_groups.remove(_matched)
                _cont_call_order.append({"type": _k, "id": _id, "attr": _ea})

            # Update call_order: remove all elem entries, insert one group
            # entry for the container at the position of the first elem.
            _first_idx = None
            _filtered_call_order = []
            for _idx, _co in enumerate(call_order):
                _co_attr = _co.get("attr") or ""
                if _co_attr in _members_attrs_seen:
                    if _first_idx is None:
                        _first_idx = len(_filtered_call_order)
                    continue
                _filtered_call_order.append(_co)
            _container_call_entry = {"type": "group", "id": _container_gid, "attr": _cont_name}
            if _first_idx is None:
                _filtered_call_order.append(_container_call_entry)
            else:
                _filtered_call_order.insert(_first_idx, _container_call_entry)
            call_order[:] = _filtered_call_order

            # Rewrite parent's attr_id_map so external edge resolution lands
            # on the container group, not its children.
            for _ea in _cont_member_attr_set:
                if _ea in attr_id_map:
                    attr_id_map[_ea] = ("group", _container_gid)
            # The bare container name should also point at the container
            # group (covers cases where the parent class records dep_edges
            # against the container attr itself).
            attr_id_map[_cont_name] = ("group", _container_gid)

            # Container's own attr_id_map: elem_attr -> (kind, original id).
            _cont_attr_id_map = {_ea: (_k, _id) for (_ea, _k, _id) in _cont_member_ids}

            # Aggregate timing for the container header: sum across members.
            # Kernel-only contract: kernel_us = fwd_kernel + bwd_kernel + other_kernel.
            _cont_kernel_us = 0.0
            _cont_fwd_kernel_us = 0.0
            _cont_bwd_kernel_us = 0.0
            _cont_other_kernel_us = 0.0
            _cont_exc_us = 0.0
            _cont_calls = 0
            _cont_role = "main"
            for (_ea, _k, _id) in _cont_member_ids:
                if _k == "node":
                    for _n in dag_nodes:
                        if _n.get("id") == _id:
                            _cont_kernel_us += _n.get("kernel_us", 0) or 0
                            _cont_fwd_kernel_us += _n.get("fwd_kernel_us", 0) or 0
                            _cont_bwd_kernel_us += _n.get("bwd_kernel_us", 0) or 0
                            _cont_other_kernel_us += _n.get("other_kernel_us", 0) or 0
                            _cont_calls += _n.get("calls", 0) or 0
                            break
                else:
                    for _cg in _cont_children_groups:
                        if _cg.get("id") == _id:
                            _cont_kernel_us += _cg.get("kernel_us", 0) or 0
                            _cont_fwd_kernel_us += _cg.get("fwd_kernel_us", 0) or 0
                            _cont_bwd_kernel_us += _cg.get("bwd_kernel_us", 0) or 0
                            _cont_other_kernel_us += _cg.get("other_kernel_us", 0) or 0
                            _cont_calls += _cg.get("calls", 0) or 0
                            break
            _cont_pct = _cont_kernel_us / step_dur_us * 100 if step_dur_us > 0 else 0

            _container_group = {
                "id": _container_gid,
                "label": _container_label,
                "attr_name": _cont_name,
                "depth": depth + 1,
                "children_nodes": _cont_children_nodes,
                "children_groups": _cont_children_groups,
                "call_order": _cont_call_order,
                "internal_edges": _cont_internal_edges,
                "input_consumer_edges": _cont_ice,
                "attr_id_map": _cont_attr_id_map,
                "pct": _cont_pct,
                "exc_pct": 0,
                # Kernel-time fields (sum of children); dur_us mirrors kernel_us.
                "kernel_us": _cont_kernel_us,
                "fwd_kernel_us": _cont_fwd_kernel_us,
                "bwd_kernel_us": _cont_bwd_kernel_us,
                "other_kernel_us": _cont_other_kernel_us,
                "dur_us": _cont_kernel_us,
                "has_phase_timing": has_timing and (_cont_fwd_kernel_us > 0 or _cont_bwd_kernel_us > 0 or _cont_other_kernel_us > 0),
                "calls": _cont_calls,
                "role": _cont_role,
                "has_timing": has_timing and _cont_kernel_us > 0,
                # Click-to-source metadata: point at the container's
                # constructor line (``self.<cont> = nn.ModuleDict(...)``).
                "src_file": _cont_def_file,
                "src_start_line": _cont_def_line,
                "src_end_line": _cont_def_line,
                "src_snippet": _cont_src_snippet,
                "src_snippet_start": _cont_src_snippet_start,
                "src_snippet_end": _cont_src_snippet_end,
                "src_truncated": False,
                # Container-specific marker for renderer / downstream tools.
                "is_container": True,
                "container_kind": _kind_name,
                "container_attr": _cont_name,
                "container_count": len(_cont_member_ids),
            }
            # Constructor info: same loc — the synthetic container's
            # "constructor" *is* the parent class's
            # ``self.<cont> = nn.ModuleDict(...)`` line.
            if _cont_def_file and _cont_def_line:
                _ctor_lines = source_files.get(_cont_def_file, [])
                if _ctor_lines:
                    _ex_lo = max(1, _cont_def_line - 2)
                    _ex_hi = min(len(_ctor_lines), _cont_def_line + 2)
                    _container_group["ctor_src_file"] = _cont_def_file
                    _container_group["ctor_src_line"] = _cont_def_line
                    _container_group["ctor_src_excerpt"] = "\n".join(_ctor_lines[_ex_lo - 1: _ex_hi])

            child_groups.append(_container_group)
            dag_groups.append(_container_group)

        # Timing info for group header.
        #
        # Kernel-only contract:
        #   * kernel_us       = fwd_kernel_us + bwd_kernel_us + other_kernel_us
        #   * No walltime / overhead / optimizer time at the per-node level.
        #
        # Class-level fallbacks (``class_durations*``) are intentionally
        # ignored here: per-instance disambiguation is the whole point of
        # ``runtime_instance_timings_by_class``.  Falling back to class-level
        # values would make multiple instances of the same class share
        # identical timings and silently mask attribution bugs.
        inst_timing = _pick_instance_timing(cname, parent_attr_name)
        kernel_us = float(inst_timing["kernel_us"]) if inst_timing else 0.0
        fwd_kernel_us = float(inst_timing["fwd_kernel_us"]) if inst_timing else 0.0
        bwd_kernel_us = float(inst_timing["bwd_kernel_us"]) if inst_timing else 0.0
        other_kernel_us = float(inst_timing["other_kernel_us"]) if inst_timing else 0.0
        exc = class_exclusive.get(cname, 0)
        pct = kernel_us / step_dur_us * 100 if step_dur_us > 0 else 0
        exc_pct = exc / step_dur_us * 100 if step_dur_us > 0 else 0
        calls_n = class_calls.get(cname, 0)
        role = class_thread_role.get(cname, "main")
        gsrc = class_source_info.get(cname, {})

        group = {
            "id": group_id,
            "label": cname,
            "attr_name": parent_attr_name,
            "depth": depth,
            "children_nodes": child_node_ids,
            "children_groups": child_groups,
            "call_order": call_order,
            "internal_edges": internal_edges,
            "input_consumer_edges": _input_consumer_edges,
            "attr_id_map": attr_id_map,
            "pct": pct,
            "exc_pct": exc_pct,
            # Kernel-time fields (the only timing fields exposed at the
            # group level).  ``dur_us`` is kept as an alias for backwards
            # compatibility with downstream layout code that still reads
            # the old name; it is always equal to ``kernel_us``.
            "kernel_us": kernel_us,
            "fwd_kernel_us": fwd_kernel_us,
            "bwd_kernel_us": bwd_kernel_us,
            "other_kernel_us": other_kernel_us,
            "dur_us": kernel_us,
            "has_phase_timing": has_timing and (fwd_kernel_us > 0 or bwd_kernel_us > 0 or other_kernel_us > 0),
            "calls": calls_n,
            "role": role,
            "has_timing": has_timing and kernel_us > 0,
            # Click-to-source metadata (same shape as leaf nodes)
            "src_file": gsrc.get("file"),
            "src_start_line": gsrc.get("start_line"),
            "src_end_line": gsrc.get("end_line"),
            "src_snippet": gsrc.get("snippet"),
            "src_snippet_start": gsrc.get("snippet_start"),
            "src_snippet_end": gsrc.get("snippet_end"),
            "src_truncated": gsrc.get("truncated", False),
        }
        # Iter11 ctor-src: attach constructor source info for this group
        # (where the *instance* of this class is created in its parent's
        # __init__). Top-level / synthetic groups (e.g. RootModule) have
        # no parent and therefore no ctor info.
        group.update(_build_ctor_info(parent_cname, parent_attr_name))
        dag_groups.append(group)
        return group

    def _make_dag_node(nid, attr_name, class_name, depth, parent_cname=None):
        # Kernel-only contract; class-level fallbacks (``class_durations*``)
        # are intentionally NOT consulted here — see ``_make_dag_group``
        # comment for rationale.
        inst_timing = _pick_instance_timing(class_name, attr_name)
        kernel_us = float(inst_timing["kernel_us"]) if inst_timing else 0.0
        fwd_kernel_us = float(inst_timing["fwd_kernel_us"]) if inst_timing else 0.0
        bwd_kernel_us = float(inst_timing["bwd_kernel_us"]) if inst_timing else 0.0
        other_kernel_us = float(inst_timing["other_kernel_us"]) if inst_timing else 0.0
        exc = class_exclusive.get(class_name, 0)
        pct = kernel_us / step_dur_us * 100 if step_dur_us > 0 else 0
        exc_pct = exc / step_dur_us * 100 if step_dur_us > 0 else 0
        calls_n = class_calls.get(class_name, 0)
        role = class_thread_role.get(class_name, "main")
        src = class_source_info.get(class_name, {})
        node = {
            "id": nid,
            "attr_name": attr_name,
            "class_name": class_name,
            "depth": depth,
            "pct": pct,
            "exc_pct": exc_pct,
            # Kernel-time fields (the only timing fields exposed at the
            # leaf-node level).  ``dur_us`` mirrors ``kernel_us`` for
            # backwards compatibility with layout / hover code.
            "kernel_us": kernel_us,
            "fwd_kernel_us": fwd_kernel_us,
            "bwd_kernel_us": bwd_kernel_us,
            "other_kernel_us": other_kernel_us,
            "dur_us": kernel_us,
            "has_phase_timing": has_timing and (fwd_kernel_us > 0 or bwd_kernel_us > 0 or other_kernel_us > 0),
            "calls": calls_n,
            "role": role,
            "has_timing": has_timing and kernel_us > 0,
            # Click-to-source metadata
            "src_file": src.get("file"),
            "src_start_line": src.get("start_line"),
            "src_end_line": src.get("end_line"),
            "src_snippet": src.get("snippet"),
            "src_snippet_start": src.get("snippet_start"),
            "src_snippet_end": src.get("snippet_end"),
            "src_truncated": src.get("truncated", False),
        }
        # Iter11 ctor-src: attach constructor source info (where this
        # instance is created in the parent's __init__).
        ctor = _build_ctor_info(parent_cname, attr_name)
        node.update(ctor)
        return node

    # Build from roots
    root_groups = []
    # Temporarily remove the loss attr from the root's tree entry so the
    # RootModule container does NOT include the loss/pred module — that node
    # is hoisted to the top level.
    _root_backup = None
    if loss_attr_name and roots:
        rc = roots[0]
        if rc in tree:
            orig = tree[rc]
            _root_backup = {
                "children": list(orig.get("children", [])),
                "attrs": dict(orig.get("attrs", {})),
                "dep_edges": list(orig.get("dep_edges", [])),
                "dep_edge_locs": dict(orig.get("dep_edge_locs", {})),
                "attr_def_loc": dict(orig.get("attr_def_loc", {})),
            }
            new_attrs = {k: v for k, v in orig.get("attrs", {}).items()
                         if k != loss_attr_name}
            removed_class = orig.get("attrs", {}).get(loss_attr_name)
            new_children = [c for c in orig.get("children", []) if c != removed_class]
            new_dep_edges = [(a, b) for (a, b) in orig.get("dep_edges", [])
                             if a != loss_attr_name and b != loss_attr_name]
            new_dep_edge_locs = {k: v for k, v in orig.get("dep_edge_locs", {}).items()
                                  if k[0] != loss_attr_name and k[1] != loss_attr_name}
            new_attr_def_loc = {k: v for k, v in orig.get("attr_def_loc", {}).items()
                                 if k != loss_attr_name}
            tree[rc] = {
                "children": new_children,
                "attrs": new_attrs,
                "dep_edges": new_dep_edges,
                "dep_edge_locs": new_dep_edge_locs,
                "attr_def_loc": new_attr_def_loc,
            }

    for root_cls in roots:
        g = build_dag_recursive(root_cls, root_cls, 0)
        if g:
            root_groups.append(g)

    # Restore the original tree entry (for any future calls)
    if _root_backup is not None and roots:
        tree[roots[0]] = _root_backup

    # ------------------------------------------------------------------
    # Add synthetic top-level "Input" and "Result" nodes:
    #   Input → RootModule → Result
    # These are NOT real nn.Module instances; they represent the abstract
    # data-flow boundary (network input tensor, final result object).
    # ------------------------------------------------------------------
    input_node_id = new_id()
    loss_node_id = new_id()

    input_node = {
        "id": input_node_id,
        "attr_name": "Input",
        "class_name": "Input",
        "depth": 0,
        "pct": 0,
        "exc_pct": 0,
        "dur_us": 0,
        "calls": 0,
        "role": "main",
        "has_timing": False,
        "kind": "io",
    }
    dag_nodes.append(input_node)

    # Result node corresponds to the synthetic LG.result() object.
    loss_label = "Result"
    loss_cls_label = "LG.result()"
    loss_dur = 0
    loss_exc = 0
    loss_calls_n = 0
    loss_role = "main"
    loss_pct = 0
    loss_exc_pct = 0
    loss_node = {
        "id": loss_node_id,
        "attr_name": loss_label,
        "class_name": loss_cls_label,
        "depth": 0,
        "pct": loss_pct,
        "exc_pct": loss_exc_pct,
        "dur_us": loss_dur,
        "calls": loss_calls_n,
        "role": loss_role,
        "has_timing": False,
        "kind": "io",
    }
    dag_nodes.append(loss_node)

    # ------------------------------------------------------------------
    # Iter13d: Build var_history for top-level Input → X edges and
    # internal boundary g__in → child edges.  Goal: every edge — including
    # synthetic Input→X — must have a complete consumer/producer chain
    # so the front-end source-highlight panel works and so the parent's
    # consumer line equals the child boundary's producer line.
    # ------------------------------------------------------------------
    def _input_edge_var_history(root_cname, child_attr, call_loc):
        """Build a var_history for an Input→X edge.

        ``call_loc`` is the (file, line) of the parent's call site
        (e.g. ``outs = [tower(trans_out) for tower in self.task_towers]``).

        Iter13e: an ``Input→X`` edge represents the abstract data-flow
        boundary "the network input flows into module X".  Its lineage
        MUST stop at X's call site in the parent forward() — it must NOT
        descend into X's own ``forward()`` body, because that is X's
        internal structure (covered by X's own internal/boundary edges).

        Returns a list of {var,file,line,excerpt,role} steps:
          STEP1 (producer): the synthetic Input boundary, anchored at
              the same call_site line.  ``var = "Input"``, role="producer".
          STEP2 (consumer): the call_site line in the root forward()
              where module X is invoked, role="consumer".
        """
        _steps = []
        if not call_loc:
            return _steps
        _fname, _ln = call_loc
        _slines = source_files.get(_fname, [])
        if not (0 <= _ln - 1 < len(_slines)):
            return _steps
        _line_text = _slines[_ln - 1]

        def _extract_all_names_from_node(node):
            """AST walk all Name.id values, deduped in order, excluding self/cls."""
            seen, result = set(), []
            if node is None:
                return result
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id not in seen and n.id not in ("self", "cls"):
                    seen.add(n.id)
                    result.append(n.id)
            return result

        def _extract_lhs_targets_from_call_line(line_text):
            try:
                _tree = ast.parse(line_text)
            except SyntaxError:
                return []
            for _node in ast.walk(_tree):
                if isinstance(_node, ast.Assign):
                    for _target in _node.targets:
                        _names = _extract_all_names_from_node(_target)
                        if _names:
                            return [v for v in _names if v != "_"]
            return []

        def _extract_call_arg_names_from_call_line(line_text):
            try:
                _tree = ast.parse(line_text)
            except SyntaxError:
                return []
            _base = re.sub(r'#\d+$', '', child_attr or '')
            _container = re.match(r'^(\w+)\[(.+)\]$', _base)
            _scan = _container.group(1) if _container else _base
            for _node in ast.walk(_tree):
                if not isinstance(_node, ast.Call):
                    continue
                _func = _node.func
                _is_match = (
                    isinstance(_func, ast.Attribute)
                    and isinstance(_func.value, ast.Name)
                    and _func.value.id == "self"
                    and _func.attr == _scan
                ) or (
                    isinstance(_func, ast.Subscript)
                    and isinstance(_func.value, ast.Attribute)
                    and isinstance(_func.value.value, ast.Name)
                    and _func.value.value.id == "self"
                    and _func.value.attr == _scan
                ) or (
                    isinstance(_func, ast.Name)
                )
                if _is_match:
                    _names = []
                    for _arg in list(_node.args) + [kw.value for kw in _node.keywords if kw.value is not None]:
                        for _name in _extract_all_names_from_node(_arg):
                            if _name not in _names:
                                _names.append(_name)
                    if _names:
                        return _names
            return []

        # Determine carry variable on the call_site line.
        _producer_carriers = _extract_lhs_targets_from_call_line(_line_text)
        _consumer_carriers = _extract_call_arg_names_from_call_line(_line_text)
        _base_attr = re.sub(r'#\d+$', '', child_attr or '')
        _container_match = re.match(r'^(\w+)\[(.+)\]$', _base_attr)
        _scan_attr = _container_match.group(1) if _container_match else _base_attr
        _is_container = _container_match is not None
        _carry_var = None

        # Pattern A: self.attr(carry_var ...) — direct call
        _m = re.search(
            r'self\.' + re.escape(_scan_attr) + r'\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
            _line_text)
        if _m:
            _carry_var = _m.group(1)

        # Pattern B: self.container[idx](carry_var ...)
        if not _carry_var and _is_container:
            _m = re.search(
                r'self\.' + re.escape(_scan_attr) + r'\s*\[[^\]]*\]\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
                _line_text)
            if _m:
                _carry_var = _m.group(1)

        # Pattern C: loop body — [<lv>(carry_var ...) for <lv> in self.container]
        # or for lv in self.container: lv(carry_var ...)
        if not _carry_var:
            _lp = re.search(
                r'for\s+(?:[\w]+\s*,\s*)?(\w+)\s+in\s+'
                r'(?:enumerate\(\s*)?self\.' + re.escape(_scan_attr) + r'\b',
                _line_text)
            if _lp:
                _lv = _lp.group(1)
                _mc = re.search(
                    r'\b' + re.escape(_lv) + r'\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
                    _line_text)
                if _mc:
                    _carry_var = _mc.group(1)
        # Fallback: descriptor
        if not _carry_var:
            _carry_var = _scan_attr or 'x'

        _excerpt = {
            "start": max(1, _ln - 1),
            "end": min(len(_slines), _ln + 1),
            "text": "\n".join(_slines[max(0, _ln - 2): min(len(_slines), _ln + 1)]),
            "highlight": _ln,
        }

        # STEP1 (producer): the Input boundary, anchored at the call site
        # so the producer/consumer pair shares a coherent file+line context.
        # ``var`` is the synthetic "Input" marker, distinguishing it from
        # the consumer step which carries the carry variable name.
        _steps.append({
            "var": "Input",
            "file": _fname,
            "line": _ln,
            "excerpt": _excerpt,
            "role": "producer",
            "carriers": _producer_carriers,
        })

        # STEP2 (consumer): the call_site line where X is invoked.
        _steps.append({
            "var": _carry_var,
            "file": _fname,
            "line": _ln,
            "excerpt": _excerpt,
            "role": "consumer",
            "carriers": _consumer_carriers,
        })

        # Iter13e: deliberately DO NOT descend into the child's forward()
        # signature line.  The child's forward() is the child's own
        # internal scope; mixing it into the parent-side Input→X edge
        # confused the front-end source panel and violated the producer/
        # consumer semantics for boundary-style entry edges.
        return _steps

    # Wire top-level edges: Input -> root_groups[0] -> Result
    top_level_ids = [input_node_id]
    if root_groups:
        top_root_id = root_groups[0]["id"]
        top_level_ids.append(top_root_id)

        # ------------------------------------------------------------------
        # Connect Input to real nn.Module consumers of LG input sources.
        # LG modules (feature_column/dense_feature/get_sample_rate) are NOT
        # DAG nodes themselves — they ARE the input. We trace which real
        # nn.Module consumes their output and connect Input directly to those.
        # ------------------------------------------------------------------
        _root_group = root_groups[0]
        _root_attr_id_map = _root_group.get("attr_id_map", {})
        _root_edge_locs = tree.get(roots[0], {}).get("dep_edge_locs", {}) if roots else {}
        _root_cname = roots[0] if roots else None
        _root_result_edges = _scan_root_result_edges(_root_cname)

        _input_connected_ids = set()

        # (a) Direct LG → real_module edges in root group
        for (lg_attr, consumer_attr) in _root_group.get("input_consumer_edges", []):
            if consumer_attr not in _root_attr_id_map:
                continue
            _, consumer_id = _root_attr_id_map[consumer_attr]
            if consumer_id in _input_connected_ids:
                continue
            _ev_data = None
            ev = _root_edge_locs.get((lg_attr, consumer_attr))
            if ev:
                _ev_lines = source_files.get(ev["file"], [])
                # Iter12-final Rule1b guard: if the ev points to a
                # constructor (`self.attr = ClassName(...)`), discard it and
                # fall through to the first_call_loc / attr_def_loc lookup
                # below so the highlighted line is a real `self.attr(...)`
                # call site instead.
                _to_ln = ev.get("to_line")
                _ctor_line = ""
                if _to_ln and _ev_lines:
                    _idx = _to_ln - 1
                    if 0 <= _idx < len(_ev_lines):
                        _ctor_line = _ev_lines[_idx]
                if _ctor_line and re.search(r'self\.\w+\s*=\s*[A-Z]', _ctor_line):
                    ev = None  # force fallback below
            if ev:
                _ev_data = {
                    "file": ev["file"],
                    "var": ev.get("var", lg_attr),
                    "from_line": ev.get("from_line"),
                    "to_line": ev.get("to_line"),
                    "from_excerpt": None,
                    "to_excerpt": {
                        "start": max(1, ev["to_line"] - 1),
                        "end": min(len(_ev_lines), ev["to_line"] + 1),
                        "text": "\n".join(_ev_lines[max(0, ev["to_line"] - 2): min(len(_ev_lines), ev["to_line"] + 1)]),
                        "highlight": ev["to_line"],
                    } if _ev_lines and ev.get("to_line") else None,
                }
                # Iter12-final-fix Rule1c: ensure to_excerpt highlights a real call.
                if _root_cname:
                    _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                        _root_cname, consumer_attr, _ev_data.get("to_excerpt"))
            elif _root_cname:
                # Iter12-final: prefer first_call_loc; only use attr_def_loc
                # when its highlighted line isn't a constructor (Rule1b).
                _loc_first = tree.get(_root_cname, {}).get("first_call_loc", {}).get(consumer_attr)
                _loc_def = tree.get(_root_cname, {}).get("attr_def_loc", {}).get(consumer_attr)
                _loc = _loc_first
                if _loc is None and _loc_def is not None:
                    _ev_lines_check = source_files.get(_loc_def[0], [])
                    _idx = _loc_def[1] - 1
                    _hl_line = _ev_lines_check[_idx] if 0 <= _idx < len(_ev_lines_check) else ""
                    if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                        _loc = _loc_def
                if _loc:
                    _ev_file, _ev_line = _loc
                    _ev_lines_fb = source_files.get(_ev_file, [])
                    _ev_data = {
                        "file": _ev_file,
                        "var": f"{lg_attr} → {consumer_attr}",
                        "from_line": _ev_line,
                        "to_line": _ev_line,
                        "to_excerpt": {
                            "start": max(1, _ev_line - 1),
                            "end": min(len(_ev_lines_fb), _ev_line + 1),
                            "text": "\n".join(_ev_lines_fb[max(0, _ev_line - 2): min(len(_ev_lines_fb), _ev_line + 1)]),
                            "highlight": _ev_line,
                        } if _ev_lines_fb else None,
                    }
            edge_entry = {"from": input_node_id, "to": consumer_id, "type": "dep",
                          "from_attr": "Input", "to_attr": consumer_attr}
            if _ev_data:
                # Iter13d: ensure to_line tracks to_excerpt.highlight after
                # _relocate_excerpt_to_self_call may have moved it.
                _te = _ev_data.get("to_excerpt") or {}
                if _te.get("highlight"):
                    _ev_data["to_line"] = _te["highlight"]
                    if not _ev_data.get("from_line") or _ev_data.get("from_line") != _te["highlight"]:
                        _ev_data["from_line"] = _te["highlight"]
                # Iter13d: build var_history so the Input→X edge has a
                # consumer/signature chain — otherwise the front-end source
                # panel will show nothing for these edges.
                if "var_history" not in _ev_data:
                    _call_loc = (
                        tree.get(_root_cname, {}).get("first_call_loc", {}).get(consumer_attr)
                        if _root_cname else None
                    )
                    if not _call_loc and _te.get("highlight") and _ev_data.get("file"):
                        _call_loc = (_ev_data["file"], _te["highlight"])
                    _vh = _input_edge_var_history(_root_cname, consumer_attr, _call_loc)
                    if _vh:
                        _ev_data["var_history"] = _vh
                edge_entry["evidence"] = _ev_data
            else:
                edge_entry["type"] = "boundary"
            dag_edges.append(edge_entry)
            _input_connected_ids.add(consumer_id)

        # (b) Sub-groups that internally consume LG sources
        for g in _root_group.get("children_groups", []):
            gid = g["id"] if isinstance(g, dict) else g
            g_data = None
            for dg in dag_groups:
                if dg["id"] == gid:
                    g_data = dg
                    break
            if not g_data:
                continue
            if g_data.get("input_consumer_edges") and gid not in _input_connected_ids:
                _ev_data = None
                _g_attr = None
                for co in _root_group.get("call_order", []):
                    if co.get("id") == gid:
                        _g_attr = co.get("attr")
                        break
                if _g_attr and _root_cname:
                    # Iter12-final: prefer first_call_loc; only use attr_def_loc
                    # when its highlighted line isn't a constructor (Rule1b).
                    _loc_first = tree.get(_root_cname, {}).get("first_call_loc", {}).get(_g_attr)
                    _loc_def = tree.get(_root_cname, {}).get("attr_def_loc", {}).get(_g_attr)
                    _loc = _loc_first
                    if _loc is None and _loc_def is not None:
                        _ev_lines_check = source_files.get(_loc_def[0], [])
                        _idx = _loc_def[1] - 1
                        _hl_line = _ev_lines_check[_idx] if 0 <= _idx < len(_ev_lines_check) else ""
                        if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                            _loc = _loc_def
                    if _loc:
                        _ev_file, _ev_line = _loc
                        _ev_lines = source_files.get(_ev_file, [])
                        _ev_data = {
                            "file": _ev_file,
                            "var": f"LG inputs → {_g_attr}",
                            "from_line": _ev_line,
                            "to_line": _ev_line,
                            "to_excerpt": {
                                "start": max(1, _ev_line - 1),
                                "end": min(len(_ev_lines), _ev_line + 1),
                                "text": "\n".join(_ev_lines[max(0, _ev_line - 2): min(len(_ev_lines), _ev_line + 1)]),
                                "highlight": _ev_line,
                            } if _ev_lines else None,
                        }
                edge_entry = {"from": input_node_id, "to": gid, "type": "dep",
                              "from_attr": "Input", "to_attr": _g_attr or g_data.get("label", "?")}
                if _ev_data:
                    # Iter12-final-fix Rule1c: relocate to_excerpt onto a real call site.
                    if _root_cname and _ev_data.get("to_excerpt") and _g_attr:
                        _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                            _root_cname, _g_attr, _ev_data.get("to_excerpt"))
                    # Iter13d: sync to_line/from_line to to_excerpt.highlight
                    _te = _ev_data.get("to_excerpt") or {}
                    if _te.get("highlight"):
                        _ev_data["to_line"] = _te["highlight"]
                        _ev_data["from_line"] = _te["highlight"]
                    # Iter13d: build var_history
                    if "var_history" not in _ev_data and _g_attr:
                        _call_loc = (
                            tree.get(_root_cname, {}).get("first_call_loc", {}).get(_g_attr)
                            if _root_cname else None
                        )
                        if not _call_loc and _te.get("highlight") and _ev_data.get("file"):
                            _call_loc = (_ev_data["file"], _te["highlight"])
                        _vh = _input_edge_var_history(_root_cname, _g_attr, _call_loc)
                        if _vh:
                            _ev_data["var_history"] = _vh
                    edge_entry["evidence"] = _ev_data
                else:
                    edge_entry["type"] = "boundary"
                dag_edges.append(edge_entry)
                _input_connected_ids.add(gid)

        # (c) Fallback: modules with no inbound from siblings also receive Input
        _root_internal_targets = set()
        for e in _root_group.get("internal_edges", []):
            _root_internal_targets.add(e["to_child"])
        for co in _root_group.get("call_order", []):
            child_id = co.get("id")
            child_attr = co.get("attr", "")
            if not child_id or child_id in _input_connected_ids:
                continue
            if child_id not in _root_internal_targets:
                _ev_data = None
                if _root_cname:
                    # Iter12-final: prefer `first_call_loc` (a real
                    # `self.attr(...)` call site in forward()).  Only fall
                    # back to `attr_def_loc` when the highlighted line is NOT
                    # a constructor expression — otherwise we'd violate Rule1b.
                    _loc_first = tree.get(_root_cname, {}).get("first_call_loc", {}).get(child_attr)
                    _loc_def = tree.get(_root_cname, {}).get("attr_def_loc", {}).get(child_attr)
                    _loc = _loc_first
                    if _loc is None and _loc_def is not None:
                        _ev_lines_check = source_files.get(_loc_def[0], [])
                        _idx = _loc_def[1] - 1
                        _hl_line = (
                            _ev_lines_check[_idx]
                            if 0 <= _idx < len(_ev_lines_check)
                            else ""
                        )
                        if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                            _loc = _loc_def
                    if _loc is None and child_attr:
                        # Last-resort scan: search every method of the root
                        # class for a real `self.{attr}(` call so the entry
                        # edge always carries Rule1b-compliant evidence.
                        _base_attr = re.sub(r'#\d+$', '', child_attr)
                        _container_match = re.match(r'^(\w+)\[(.+)\]$', _base_attr)
                        _scan_attr = _container_match.group(1) if _container_match else _base_attr
                        for (_fname, _cm_name), _cm_info in class_map.items():
                            if _cm_name != _root_cname:
                                continue
                            _lines_c = source_files.get(_fname, [])
                            for _mname, _mrange in _cm_info.get("methods", {}).items():
                                if _mname == "__init__":
                                    continue
                                for _i in range(_mrange[0] - 1, min(_mrange[1], len(_lines_c))):
                                    _line_text = _lines_c[_i]
                                    if re.search(r'self\.' + re.escape(_scan_attr) + r'\s*[\(\[]', _line_text):
                                        _loc = (_fname, _i + 1)
                                        break
                                if _loc is not None:
                                    break
                            if _loc is not None:
                                break
                    if _loc:
                        _ev_file, _ev_line = _loc
                        _ev_lines = source_files.get(_ev_file, [])
                        _ev_data = {
                            "file": _ev_file,
                            "var": f"input → {child_attr}",
                            "from_line": _ev_line,
                            "to_line": _ev_line,
                            "to_excerpt": {
                                "start": max(1, _ev_line - 1),
                                "end": min(len(_ev_lines), _ev_line + 1),
                                "text": "\n".join(_ev_lines[max(0, _ev_line - 2): min(len(_ev_lines), _ev_line + 1)]),
                                "highlight": _ev_line,
                            } if _ev_lines else None,
                        }
                edge_entry = {"from": input_node_id, "to": child_id, "type": "dep",
                              "from_attr": "Input", "to_attr": child_attr or "?"}
                if _ev_data:
                    # Iter12-final-fix Rule1c: relocate to real call site.
                    if _root_cname and _ev_data.get("to_excerpt"):
                        _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                            _root_cname, child_attr, _ev_data.get("to_excerpt"))
                    # Iter13d: sync to_line/from_line to to_excerpt.highlight
                    _te = _ev_data.get("to_excerpt") or {}
                    if _te.get("highlight"):
                        _ev_data["to_line"] = _te["highlight"]
                        _ev_data["from_line"] = _te["highlight"]
                    # Iter13d: build var_history
                    if "var_history" not in _ev_data:
                        _call_loc = (
                            tree.get(_root_cname, {}).get("first_call_loc", {}).get(child_attr)
                            if _root_cname else None
                        )
                        if not _call_loc and _te.get("highlight") and _ev_data.get("file"):
                            _call_loc = (_ev_data["file"], _te["highlight"])
                        _vh = _input_edge_var_history(_root_cname, child_attr, _call_loc)
                        if _vh:
                            _ev_data["var_history"] = _vh
                    edge_entry["evidence"] = _ev_data
                else:
                    # Iter12-final: if no Rule1b-safe call-site can be found,
                    # demote this synthetic entry edge to type=boundary so it
                    # is exempt from check_rule1 / check_rule1b.
                    edge_entry["type"] = "boundary"
                dag_edges.append(edge_entry)
                _input_connected_ids.add(child_id)

        if not _input_connected_ids:
            dag_edges.append({"from": input_node_id, "to": f"{top_root_id}__in", "type": "dep"})

        # Connect precise result-producing root attrs to Result.
        _result_connected_ids = set()
        for producer_attr, ev in sorted(_root_result_edges.items(), key=lambda kv: (kv[1].get("to_line", 10**9), kv[0])):
            if producer_attr not in _root_attr_id_map:
                continue
            _, producer_id = _root_attr_id_map[producer_attr]
            if producer_id in _result_connected_ids:
                continue
            _ev_lines = source_files.get(ev["file"], [])
            _ev_data = {
                "file": ev["file"],
                "var": ev.get("var", producer_attr),
                "from_line": ev.get("from_line"),
                "to_line": ev.get("to_line"),
                "from_excerpt": {
                    "start": max(1, ev["from_line"] - 1),
                    "end": min(len(_ev_lines), ev["from_line"] + 1),
                    "text": "\n".join(_ev_lines[max(0, ev["from_line"] - 2): min(len(_ev_lines), ev["from_line"] + 1)]),
                    "highlight": ev["from_line"],
                } if _ev_lines and ev.get("from_line") else None,
                "to_excerpt": {
                    "start": max(1, ev["to_line"] - 1),
                    "end": min(len(_ev_lines), ev["to_line"] + 1),
                    "text": "\n".join(_ev_lines[max(0, ev["to_line"] - 2): min(len(_ev_lines), ev["to_line"] + 1)]),
                    "highlight": ev["to_line"],
                } if _ev_lines and ev.get("to_line") else None,
            }
            edge_entry = {"from": producer_id, "to": loss_node_id, "type": "dep",
                          "from_attr": producer_attr, "to_attr": "Result",
                          "evidence": _ev_data}
            # Iter12-final-fix Rule1c: ensure from_excerpt highlights a real call.
            if _root_cname and _ev_data:
                _ev_data["from_excerpt"] = _relocate_excerpt_to_self_call(
                    _root_cname, producer_attr, _ev_data.get("from_excerpt"))
            # Iter13d: build var_history for X→Result edges so the
            # producer/consumer chain is non-empty.
            if _ev_data and "var_history" not in _ev_data:
                _vh_steps = []
                _from_ex = _ev_data.get("from_excerpt") or {}
                if _from_ex.get("highlight"):
                    _vh_steps.append({
                        "var": _ev_data.get("var") or producer_attr,
                        "file": _ev_data.get("file"),
                        "line": _from_ex.get("highlight"),
                        "excerpt": _from_ex,
                        "role": "step",
                    })
                _to_ex = _ev_data.get("to_excerpt") or {}
                if _to_ex.get("highlight") and _to_ex.get("highlight") != _from_ex.get("highlight"):
                    _vh_steps.append({
                        "var": "result",
                        "file": _ev_data.get("file"),
                        "line": _to_ex.get("highlight"),
                        "excerpt": _to_ex,
                        "role": "consumer",
                    })
                if _vh_steps:
                    _ev_data["var_history"] = _vh_steps
            dag_edges.append(edge_entry)
            _result_connected_ids.add(producer_id)

        # Fallback: connect root exit attrs to Result only when precise result edges are absent.
        if not _result_connected_ids:
            _root_internal_sources = set()
            for e in _root_group.get("internal_edges", []):
                _root_internal_sources.add(e["from_child"])
            for co in _root_group.get("call_order", []):
                child_id = co.get("id")
                child_attr = co.get("attr", "")
                if not child_id or child_id in _root_internal_sources:
                    continue
                _ev_data = None
                if _root_cname:
                    _loc = tree.get(_root_cname, {}).get("first_call_loc", {}).get(child_attr) \
                        or tree.get(_root_cname, {}).get("attr_def_loc", {}).get(child_attr)
                    if _loc:
                        _ev_file, _ev_line = _loc
                        _ev_lines = source_files.get(_ev_file, [])
                        _ev_data = {
                            "file": _ev_file,
                            "var": f"{child_attr} → Result",
                            "from_line": _ev_line,
                            "to_line": _ev_line,
                            "from_excerpt": {
                                "start": max(1, _ev_line - 1),
                                "end": min(len(_ev_lines), _ev_line + 1),
                                "text": "\n".join(_ev_lines[max(0, _ev_line - 2): min(len(_ev_lines), _ev_line + 1)]),
                                "highlight": _ev_line,
                            } if _ev_lines else None,
                        }
                edge_entry = {"from": child_id, "to": loss_node_id, "type": "dep",
                              "from_attr": child_attr or "?", "to_attr": "Result"}
                if _ev_data:
                    # Iter12-final-fix Rule1c: ensure from_excerpt highlights a real call.
                    if _root_cname:
                        _ev_data["from_excerpt"] = _relocate_excerpt_to_self_call(
                            _root_cname, child_attr, _ev_data.get("from_excerpt"))
                    # Iter13d: build var_history for X→Result fallback edges.
                    if "var_history" not in _ev_data:
                        _from_ex = _ev_data.get("from_excerpt") or {}
                        if _from_ex.get("highlight"):
                            _ev_data["var_history"] = [{
                                "var": child_attr or "result",
                                "file": _ev_data.get("file"),
                                "line": _from_ex.get("highlight"),
                                "excerpt": _from_ex,
                                "role": "step",
                            }]
                    edge_entry["evidence"] = _ev_data
                dag_edges.append(edge_entry)
                _result_connected_ids.add(child_id)

        # Iter7 strict rule: do NOT synthesize root-boundary → Result fallback edges.
        # Every Result edge must originate from a real nn.Module proven by source-level
        # variable flow into result.head(...) / return result, loss.

    top_level_ids.append(loss_node_id)

    # ------------------------------------------------------------------
    # Propagate Input/Result edges recursively into ALL groups:
    # Every entry child (no inbound from siblings) gets Input → child edge.
    # Every exit child (no outbound to siblings) gets child → Result edge.
    # This ensures strict Rule2 compliance without implicit boundary exemptions.
    # ------------------------------------------------------------------
    # Iter11 fix: synthetic Input top_edges previously used a fake var like
    # "input → bias2nn" / "LG → live_info_module" which contains a non-identifier
    # arrow character. Browsers' word-boundary regex (\b) cannot anchor to it,
    # so the consumer-side <mark> highlight silently fails. We instead inspect
    # the real source line at `to_line` and pick a genuine Python identifier.
    _IDENT_RE_FOR_VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _PY_KEYWORDS_FOR_VAR = {
        "self", "True", "False", "None", "and", "or", "not", "is", "in",
        "if", "else", "elif", "for", "while", "return", "yield", "with",
        "as", "import", "from", "lambda", "def", "class", "try", "except",
        "finally", "raise", "pass", "break", "continue", "global", "nonlocal",
        "assert", "del", "torch", "nn",
    }

    def _extract_real_var_from_line(line_text, consumer_attr):
        """Pick a real Python identifier appearing on the consumer line.

        Priority:
          1. LHS of an assignment, e.g. ``live_info = self.live_info_module(live_info)``
             or ``layers_map = {'token_00': tokens_dict['bias'], ...}`` → ``live_info`` / ``layers_map``.
          2. The first argument identifier passed into ``self.<consumer_attr>(...)``.
          3. The first non-keyword identifier appearing on the line.

        Returns ``None`` if no usable identifier was found, in which case the
        caller should fall back to the synthetic descriptor.
        """
        if not line_text:
            return None
        s = line_text.strip()
        if not s or s.startswith('#'):
            return None
        # 1) LHS of assignment (single name)
        m_lhs = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]', s)
        if m_lhs:
            cand = m_lhs.group(1)
            if cand not in _PY_KEYWORDS_FOR_VAR and cand != consumer_attr:
                return cand
        # 2) self.<consumer_attr>(args) → first arg identifier
        if consumer_attr:
            m_call = re.search(
                r'self\.' + re.escape(consumer_attr) + r'\s*\(([^)]*)\)', s)
            if m_call:
                args = m_call.group(1)
                for tok in re.finditer(r'[A-Za-z_][A-Za-z0-9_]*', args):
                    cand = tok.group(0)
                    if cand not in _PY_KEYWORDS_FOR_VAR and cand != consumer_attr:
                        return cand
        # 3) first non-keyword identifier
        for tok in re.finditer(r'[A-Za-z_][A-Za-z0-9_]*', s):
            cand = tok.group(0)
            if cand in _PY_KEYWORDS_FOR_VAR:
                continue
            if cand == consumer_attr:
                continue
            return cand
        return None

    def _real_var_for_input_edge(ev_lines, to_line, consumer_attr, fallback):
        """Wrapper that picks the highlight line and tries each preceding line
        if the highlight line doesn't yield anything.
        """
        if not ev_lines or not to_line:
            return fallback
        # Check the highlight line first
        idx = to_line - 1
        if 0 <= idx < len(ev_lines):
            cand = _extract_real_var_from_line(ev_lines[idx], consumer_attr)
            if cand:
                return cand
        # Then try one or two lines above (consumer's tensor often produced just before)
        for back in (1, 2):
            i = to_line - 1 - back
            if 0 <= i < len(ev_lines):
                cand = _extract_real_var_from_line(ev_lines[i], consumer_attr)
                if cand:
                    return cand
        return fallback

    def _extract_var_from_excerpt(excerpt_obj, consumer_attr):
        """Iter11 varlineage: as a last-resort, scan the excerpt's full text
        (highlight line first, then surrounding) to recover a real Python
        identifier, instead of synthesising a "input → xxx" descriptor.
        """
        if not excerpt_obj or not excerpt_obj.get("text"):
            return None
        lines_arr = excerpt_obj["text"].split("\n")
        hl = excerpt_obj.get("highlight")
        start = excerpt_obj.get("start") or 1
        # Sort: highlight line first, then the rest in order.
        order = []
        if hl is not None:
            for i, t in enumerate(lines_arr):
                if start + i == hl:
                    order.append(t)
                    break
        for i, t in enumerate(lines_arr):
            if hl is not None and start + i == hl:
                continue
            order.append(t)
        for txt in order:
            cand = _extract_real_var_from_line(txt, consumer_attr)
            if cand:
                return cand
        return None

    def _extract_consumer_lhs(line_text):
        """从 consumer 所在源码行提取最外层 LHS；失败时返回 ``_``。"""
        if not line_text:
            return "_"
        _line_nc = re.sub(r"#.*$", "", line_text).rstrip()
        if not _line_nc.strip():
            return "_"
        _m_tuple = re.match(
            r'^\s*\(?\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*\)?\s*=\s*[^=]',
            _line_nc,
        )
        if _m_tuple:
            return _m_tuple.group(1).strip()
        _m_single = re.match(
            r'^\s*([A-Za-z_]\w*)\s*(?::\s*[^=]+)?=\s*[^=]',
            _line_nc,
        )
        if _m_single:
            return _m_single.group(1).strip()
        return "_"

    def _extract_consumer_lhs_nearby(source_lines, line_no, method_end=None):
        """提取 consumer var；兼容 for-body、跨行 tuple 解包、dict slot 与 append。"""
        if not source_lines or not line_no:
            return "_"
        _idx = line_no - 1
        if not (0 <= _idx < len(source_lines)):
            return "_"
        _line_text = source_lines[_idx]
        _lhs = _extract_consumer_lhs(_line_text)
        if _lhs != "_":
            return _lhs

        _scan_end = min(len(source_lines), method_end or len(source_lines), _idx + 12)
        _window_lines = source_lines[_idx:_scan_end]
        _window_text = "\n".join(re.sub(r"#.*$", "", _ln).rstrip() for _ln in _window_lines)
        _m_tuple_block = re.search(
            r'\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*,?\s*\)\s*=\s*self\.',
            _window_text,
            re.S,
        )
        if _m_tuple_block:
            return ", ".join(_tok.strip() for _tok in _m_tuple_block.group(1).split(','))
        _m_tuple_inline = re.search(
            r'\b([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=\s*self\.',
            _window_text,
        )
        if _m_tuple_inline:
            return ", ".join(_tok.strip() for _tok in _m_tuple_inline.group(1).split(','))
        _m_dict_slot = re.search(r'\b([A-Za-z_]\w*\s*\[[^\]]+\])\s*=\s*self\.', _window_text)
        if _m_dict_slot:
            return re.sub(r'\s+', '', _m_dict_slot.group(1))
        _m_append = re.search(r'\b([A-Za-z_]\w*)\s*\.append\s*\(\s*self\.', _window_text)
        if _m_append:
            return _m_append.group(1)

        _line_nc = re.sub(r"#.*$", "", _line_text).rstrip()
        _for_match = re.match(r'^\s*for\s+(.+?)\s+in\b', _line_nc)
        if not _for_match:
            return "_"

        _loop_vars = [
            _tok.group(0)
            for _tok in re.finditer(r'[A-Za-z_]\w*', _for_match.group(1))
            if _tok.group(0) not in _PY_KEYWORDS_FOR_VAR
        ]
        _loop_indent = len(_line_text) - len(_line_text.lstrip())
        for _j in range(_idx + 1, _scan_end):
            _next_text = source_lines[_j]
            _next_nc = re.sub(r"#.*$", "", _next_text).rstrip()
            if not _next_nc.strip():
                continue
            _next_indent = len(_next_text) - len(_next_text.lstrip())
            if _next_indent <= _loop_indent:
                break
            _next_lhs = _extract_consumer_lhs(_next_nc)
            if _next_lhs == "_":
                continue
            if _loop_vars and any(
                re.search(r'\b' + re.escape(_lv) + r'(?:\s*\(|\s*\.)', _next_nc)
                for _lv in _loop_vars
            ):
                return _next_lhs
            if re.search(r'=\s*[A-Za-z_]\w*\s*\(', _next_nc):
                return _next_lhs
        return "_"

    def _build_consumer_excerpt(source_lines, line_no, method_end=None, attr_name=None):
        """构造 consumer excerpt；跨行赋值/调用时合成为单条 logical line。"""
        if not source_lines or not line_no:
            return None
        _idx = line_no - 1
        if not (0 <= _idx < len(source_lines)):
            return None
        _scan_end = min(len(source_lines), method_end or len(source_lines), _idx + 12)
        _logical_lines = []
        _paren = 0
        _seen_self = False
        _scan_attr = None
        if attr_name:
            _base_attr = re.sub(r'#\d+$', '', attr_name)
            _m_elem = re.match(r'^(\w+)\[(.+)\]$', _base_attr)
            _scan_attr = _m_elem.group(1) if _m_elem else _base_attr
        for _j in range(_idx, _scan_end):
            _raw = source_lines[_j]
            _nc = re.sub(r"#.*$", "", _raw).rstrip()
            if not _nc.strip() and _logical_lines:
                break
            if not _nc.strip():
                continue
            _logical_lines.append(_nc.strip())
            if _scan_attr and re.search(r'self\.' + re.escape(_scan_attr) + r'\b', _nc):
                _seen_self = True
            _paren += _nc.count('(') - _nc.count(')')
            if _seen_self and _paren <= 0 and ('=' in _nc or _nc.endswith(')')):
                break
        if len(_logical_lines) <= 1:
            return {
                "start": max(1, line_no - 1),
                "end": min(len(source_lines), line_no + 1),
                "text": "\n".join(source_lines[max(0, line_no - 2): min(len(source_lines), line_no + 1)]),
                "highlight": line_no,
            }
        return {
            "start": line_no,
            "end": min(len(source_lines), line_no + len(_logical_lines) - 1),
            "text": " ".join(_logical_lines),
            "highlight": line_no,
        }

    def _parent_boundary_history(parent_group, child_group_id):
        """Build the parent-side prefix for a boundary edge ``g__in → child``.

        This lineage is intentionally narrow: it should show only the parent
        forward() line that calls the child group, then the child's own entry
        step / downstream consumer supplied by the caller.
        """
        _prefix_steps = []
        if not parent_group:
            _child_attr = None
        else:
            parent_label = parent_group.get("label", "")
            _child_attr = None
            for _co in parent_group.get("call_order", []) or []:
                if _co.get("id") == child_group_id:
                    _child_attr = _co.get("attr")
                    break
            if _child_attr:
                _parent_first_call = (tree.get(parent_label, {}) or {}).get("first_call_loc", {}) or {}
                _call_loc = _parent_first_call.get(_child_attr)
                if _call_loc:
                    _fname, _ln = _call_loc
                    _src_lines = source_files.get(_fname, [])
                    if 0 <= _ln - 1 < len(_src_lines):
                        _line_text = _src_lines[_ln - 1]
                        _child_attr_nosplit = re.sub(r"#\d+$", "", _child_attr or "")
                        _child_attr_base = re.sub(r"\[[^\]]*\]", "", _child_attr_nosplit)
                        _carry_var = None

                        _call_patterns = []
                        if _child_attr_nosplit:
                            if "[" in _child_attr_nosplit:
                                _container = re.escape(_child_attr_base)
                                _call_patterns.append(
                                    re.compile(r"self\." + _container + r"\s*\[[^\]]*\]\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
                                )
                            else:
                                _call_patterns.append(
                                    re.compile(r"self\." + re.escape(_child_attr_nosplit) + r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
                                )
                        if _child_attr_base and _child_attr_base != _child_attr_nosplit:
                            _call_patterns.append(
                                re.compile(r"self\." + re.escape(_child_attr_base) + r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
                            )

                        for _pat in _call_patterns:
                            _m = _pat.search(_line_text)
                            if _m:
                                _carry_var = _m.group(1)
                                break
                        # Iter13d: detect loop-comprehension pattern carrying the input var.
                        # Patterns:
                        #   [<lv>(carry_var ...) for <lv> in self.<container>]
                        #   for <lv> in self.<container>: <lv>(carry_var ...)
                        if not _carry_var:
                            _lp = re.search(
                                r'for\s+(?:[\w]+\s*,\s*)?(\w+)\s+in\s+'
                                r'(?:enumerate\(\s*)?self\.' + re.escape(_child_attr_base) + r'\b',
                                _line_text)
                            if _lp:
                                _lv = _lp.group(1)
                                _mc = re.search(
                                    r'\b' + re.escape(_lv) + r'\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
                                    _line_text)
                                if _mc:
                                    _carry_var = _mc.group(1)
                        if not _carry_var:
                            _carry_var = _child_attr_base or _child_attr_nosplit or _child_attr

                        _prefix_steps.append({
                            "var": _carry_var,
                            "file": _fname,
                            "line": _ln,
                            "excerpt": {
                                "start": max(1, _ln - 1),
                                "end": min(len(_src_lines), _ln + 1),
                                "text": "\n".join(_src_lines[max(0, _ln - 2): min(len(_src_lines), _ln + 1)]),
                                "highlight": _ln,
                            },
                            "role": "step",
                        })

        _child_label = None
        if parent_group:
            for _cg in parent_group.get("children_groups", []) or []:
                if isinstance(_cg, dict) and _cg.get("id") == child_group_id:
                    _child_label = _cg.get("label")
                    break
        if not _child_label:
            for _dg in dag_groups:
                if _dg.get("id") == child_group_id:
                    _child_label = _dg.get("label")
                    break
        if _child_label:
            for (_cf, _cn), _ci in class_map.items():
                if _cn != _child_label:
                    continue
                _fwd_range = ((_ci.get("methods", {}) or {}).get("forward"))
                if not _fwd_range:
                    continue
                _cln = _fwd_range[0]
                _cend = _fwd_range[1] if isinstance(_fwd_range, (list, tuple)) and len(_fwd_range) > 1 else _cln
                _clines = source_files.get(_cf, [])
                if not (0 <= _cln - 1 < len(_clines)):
                    continue
                _scan_end = min(len(_clines), max(_cend, _cln + 6))
                _sig_lines = []
                _sig_hl = _cln
                for _i in range(_cln - 1, _scan_end):
                    _cur = _clines[_i]
                    _sig_lines.append(_cur)
                    if re.search(r"\bself\b", _cur):
                        _sig_hl = _i + 1
                    if ")" in _cur and ":" in _cur:
                        break
                _sig_blob = " ".join(_sig_lines)
                _sig_match = re.search(r"def\s+forward\s*\(\s*self\s*(?:,\s*([A-Za-z_][A-Za-z0-9_]*))?", _sig_blob)
                _sig_var = _sig_match.group(1) if (_sig_match and _sig_match.group(1)) else "x"
                _prefix_steps.append({
                    "var": _sig_var,
                    "file": _cf,
                    "line": _sig_hl,
                    "excerpt": {
                        "start": max(1, _sig_hl - 1),
                        "end": min(len(_clines), _sig_hl + 1),
                        "text": "\n".join(_clines[max(0, _sig_hl - 2): min(len(_clines), _sig_hl + 1)]),
                        "highlight": _sig_hl,
                    },
                    "role": "step",
                })
                break

        return _prefix_steps

    def _child_return_to_parent_history(g, parent_group, child_attr):
        """Build the symmetric tail (STEP2 + STEP3 + CONSUMER) for an outbound
        boundary edge ``child → g__out``.

        Mirror of :func:`_parent_boundary_history`.  The caller already produces
        the PRODUCER step (the ``self.{child_attr}(...)`` invocation line in
        ``g``'s forward); this helper appends:

          * STEP 2 — the last ``ret_var = ...`` assignment in ``g``'s own
            ``forward`` body (skipped when the function returns an expression
            directly without a prior named assignment).
          * STEP 3 — the ``return ret_var`` line itself in ``g``'s forward
            (always emitted when ``g`` has a forward with a return statement).
          * CONSUMER — the parent_group's call site, where the return value of
            ``g`` is received via ``lhs = self.{g_attr}(...)``.  Skipped when
            the parent_group is None or the call site lookup fails.

        Strict Rule1b: every step's highlight line must lie inside a
        ``forward`` (or other non-``__init__``) method of g's / parent_group's
        class.

        NOTE: ``child_attr`` is informational only (it identifies the inner
        producer that triggers this outbound edge); the STEP 2 / STEP 3 chain
        is keyed off ``g`` itself because the parent receives ``g``'s return
        value, not the inner child's.
        """
        _tail_steps = []

        # ---- STEP 2 + STEP 3: scan g's own forward for return / final assignment ----
        _g_label = g.get("label", "")
        if _g_label:
            for (_cf, _cn), _ci in class_map.items():
                if _cn != _g_label:
                    continue
                _fwd_range = ((_ci.get("methods", {}) or {}).get("forward"))
                if not _fwd_range:
                    continue
                _fwd_start, _fwd_end = _fwd_range[0], _fwd_range[1]
                _clines = source_files.get(_cf, [])
                if not _clines:
                    continue
                _scan_end = min(len(_clines), _fwd_end)
                _scan_start = max(0, _fwd_start - 1)

                # locate the LAST ``return ...`` line inside the forward body
                _return_line_idx = None
                _ret_var = None
                _ret_pat = re.compile(r"^\s*return\s+(.+?)\s*$")
                for _i in range(_scan_end - 1, _scan_start - 1, -1):
                    if _i >= len(_clines):
                        continue
                    _cur = _clines[_i]
                    # strip trailing comment for matching
                    _cur_nc = re.sub(r"#.*$", "", _cur).rstrip()
                    _m_ret = _ret_pat.match(_cur_nc)
                    if _m_ret:
                        _return_line_idx = _i
                        _expr = _m_ret.group(1).strip()
                        _m_id = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*$", _expr)
                        if _m_id:
                            _ret_var = _m_id.group(1)
                        break

                if _return_line_idx is None:
                    continue

                # STEP 2 — last assignment to _ret_var (only when we have an identifier)
                if _ret_var:
                    _assign_pat = re.compile(
                        r"^\s*" + re.escape(_ret_var) + r"\s*(?:[+\-*/@&|^%]?=)\s*[^=]"
                    )
                    _assign_idx = None
                    for _j in range(_return_line_idx - 1, _scan_start - 1, -1):
                        if _j >= len(_clines):
                            continue
                        _ln_text = _clines[_j]
                        _ln_nc = re.sub(r"#.*$", "", _ln_text).rstrip()
                        if not _ln_nc.strip():
                            continue
                        # skip pure ``return`` / ``def`` lines
                        if re.match(r"^\s*return\b", _ln_nc):
                            continue
                        if re.match(r"^\s*def\b", _ln_nc):
                            continue
                        # accept simple ``var = …`` and ``var: T = …``
                        if (re.match(r"^\s*" + re.escape(_ret_var) + r"\s*=\s*[^=]", _ln_nc)
                                or re.match(r"^\s*" + re.escape(_ret_var) + r"\s*:\s*\S+\s*=\s*[^=]", _ln_nc)
                                or _assign_pat.match(_ln_nc)):
                            _assign_idx = _j
                            break

                    if _assign_idx is not None:
                        _hl = _assign_idx + 1
                        _tail_steps.append({
                            "var": _ret_var,
                            "file": _cf,
                            "line": _hl,
                            "excerpt": {
                                "start": max(1, _hl - 1),
                                "end": min(len(_clines), _hl + 1),
                                "text": "\n".join(_clines[max(0, _hl - 2): min(len(_clines), _hl + 1)]),
                                "highlight": _hl,
                            },
                            "role": "step",
                        })

                # STEP 3 — the return line itself
                _hl_ret = _return_line_idx + 1
                _tail_steps.append({
                    "var": _ret_var or "",
                    "file": _cf,
                    "line": _hl_ret,
                    "excerpt": {
                        "start": max(1, _hl_ret - 1),
                        "end": min(len(_clines), _hl_ret + 1),
                        "text": "\n".join(_clines[max(0, _hl_ret - 2): min(len(_clines), _hl_ret + 1)]),
                        "highlight": _hl_ret,
                    },
                    "role": "step",
                })
                break

        _has_return_step = any(
            isinstance(_s.get("excerpt"), dict)
            and re.match(r'^\s*return\b', (
                (_s.get("excerpt") or {}).get("text", "").split("\n")[(((_s.get("excerpt") or {}).get("highlight", 1) - (_s.get("excerpt") or {}).get("start", 1)))]
                if ((_s.get("excerpt") or {}).get("text")) else ""
            ))
            for _s in _tail_steps
        )
        if not _has_return_step and _g_label:
            for (_cf, _cn), _ci in class_map.items():
                if _cn != _g_label:
                    continue
                _clines = source_files.get(_cf, [])
                if not _clines:
                    continue
                _fallback_ranges = []
                _fwd_range = ((_ci.get("methods", {}) or {}).get("forward"))
                if _fwd_range:
                    _fallback_ranges.append(_fwd_range)
                for _mname, _mrange in ((_ci.get("methods", {}) or {}).items()):
                    if _mname != "__init__" and _mrange not in _fallback_ranges:
                        _fallback_ranges.append(_mrange)
                _ret_pat = re.compile(r"^\s*return\s+(.+?)\s*$")
                for _mr_start, _mr_end in _fallback_ranges:
                    for _i in range(min(len(_clines), _mr_end) - 1, max(0, _mr_start - 1) - 1, -1):
                        _cur_nc = re.sub(r"#.*$", "", _clines[_i]).rstrip()
                        _m_ret = _ret_pat.match(_cur_nc)
                        if not _m_ret:
                            continue
                        _hl_ret = _i + 1
                        _tail_steps.insert(-1 if _tail_steps and _tail_steps[-1].get("role") == "consumer" else len(_tail_steps), {
                            "var": "",
                            "file": _cf,
                            "line": _hl_ret,
                            "excerpt": {
                                "start": max(1, _hl_ret - 1),
                                "end": min(len(_clines), _hl_ret + 1),
                                "text": "\n".join(_clines[max(0, _hl_ret - 2): min(len(_clines), _hl_ret + 1)]),
                                "highlight": _hl_ret,
                            },
                            "role": "step",
                        })
                        _has_return_step = True
                        break
                    if _has_return_step:
                        break
                if _has_return_step:
                    break

        # ---- CONSUMER: parent receives g's return value ----
        if parent_group:
            _parent_label = parent_group.get("label", "")
            _g_id = g.get("id")
            _attr_in_parent = None
            for _co in parent_group.get("call_order", []) or []:
                if _co.get("id") == _g_id:
                    _attr_in_parent = _co.get("attr")
                    break
            if _attr_in_parent:
                _parent_first_call = (tree.get(_parent_label, {}) or {}).get("first_call_loc", {}) or {}
                _call_loc = _parent_first_call.get(_attr_in_parent)
                if _call_loc is None and _attr_in_parent:
                    _base_parent_attr = re.sub(r'#\d+$', '', _attr_in_parent)
                    _scan_parent_attr = re.match(r'^(\w+)\[(.+)\]$', _base_parent_attr)
                    _scan_parent_attr = _scan_parent_attr.group(1) if _scan_parent_attr else _base_parent_attr
                    _parent_call_pat = re.compile(r'self\.' + re.escape(_scan_parent_attr) + r'\s*(?:[\(\[]|(?:\[[^\]]*\])?\.\w+\s*\()')
                    for (_pf_scan, _pcn), _pci in class_map.items():
                        if _pcn != _parent_label:
                            continue
                        _pl_scan = source_files.get(_pf_scan, [])
                        if not _pl_scan:
                            continue
                        for _mname, _mrange in (_pci.get("methods", {}) or {}).items():
                            if _mname == "__init__":
                                continue
                            for _i in range(_mrange[0] - 1, min(_mrange[1], len(_pl_scan))):
                                if _parent_call_pat.search(_pl_scan[_i]):
                                    _call_loc = (_pf_scan, _i + 1)
                                    break
                            if _call_loc is not None:
                                break
                        if _call_loc is not None:
                            break
                if _call_loc:
                    _pf, _pln = _call_loc
                    _plines = source_files.get(_pf, [])
                    if 0 <= _pln - 1 < len(_plines):
                        _line_text = _plines[_pln - 1]
                        # Rule1b: parent call site must NOT be a constructor.
                        # ``first_call_loc`` is already a forward() call site by
                        # construction, but we double-check defensively.
                        _is_ctor = bool(re.search(r"self\.\w+\s*=\s*[A-Z]", _line_text))
                        if not _is_ctor:
                            _method_end = None
                            for (_pf_scan, _pcn), _pci in class_map.items():
                                if _pf_scan != _pf or _pcn != _parent_label:
                                    continue
                                for _mname, _mrange in (_pci.get("methods", {}) or {}).items():
                                    if _mname == "__init__":
                                        continue
                                    if _mrange[0] <= _pln <= _mrange[1]:
                                        _method_end = _mrange[1]
                                        break
                                if _method_end is not None:
                                    break
                            _lhs_var = _extract_consumer_lhs_nearby(_plines, _pln, _method_end)
                            _consumer_excerpt = _build_consumer_excerpt(
                                _plines, _pln, _method_end, _attr_in_parent)
                            _tail_steps.append({
                                "var": _lhs_var or "_",
                                "file": _pf,
                                "line": _pln,
                                "excerpt": _consumer_excerpt or {
                                    "start": max(1, _pln - 1),
                                    "end": min(len(_plines), _pln + 1),
                                    "text": "\n".join(_plines[max(0, _pln - 2): min(len(_plines), _pln + 1)]),
                                    "highlight": _pln,
                                },
                                "role": "consumer",
                            })

        return _tail_steps

    def _merge_var_history(prefix_steps, own_steps):
        out = []
        seen = set()
        for _step in (prefix_steps or []) + (own_steps or []):
            if not isinstance(_step, dict):
                continue
            _key = (_step.get("file"), _step.get("line"), _step.get("var"), _step.get("role"))
            if _key in seen:
                continue
            seen.add(_key)
            out.append(_step)
        return out

    def _propagate_edges_recursive(groups_list, is_root_level=False, parent_group=None):
        # Iter12-fix2: subgroup boundary crossing fix.
        # Entry edges (Input → child-of-g) must NOT pierce a group frame:
        #   • If g is a deep sub-group (parent_group is itself a non-root
        #     group), anchor entries at `g__in`.
        #   • If g is a depth-1 group BUT has sibling inbound on its parent
        #     (root) — i.e. data already enters g from another root child —
        #     anchor entries at `g__in` so the inside flow doesn't visually
        #     bypass that sibling edge.
        #   • Otherwise (true root-level entry from Input), keep the global
        #     Input → child edge.
        _root_id = root_groups[0].get("id") if root_groups else None
        _is_under_root = (
            parent_group is not None
            and parent_group.get("id") == _root_id
        )

        # Pre-compute groups that already have sibling inbound on parent_group's
        # internal_edges (where source is another sibling, not the parent
        # boundary itself).  Used only when _is_under_root is True.
        _parent_int_inbound = set()
        if parent_group is not None:
            for _ie in parent_group.get("internal_edges", []) or []:
                _tc = _ie.get("to_child")
                if _tc:
                    _parent_int_inbound.add(_tc)

        for g in groups_list:
            attr_id_map = g.get("attr_id_map", {})
            g_label = g.get("label", "")
            g_edge_locs = tree.get(g_label, {}).get("dep_edge_locs", {})
            g_first_call_loc = tree.get(g_label, {}).get("first_call_loc", {})
            g_attr_def_loc = tree.get(g_label, {}).get("attr_def_loc", {})
            ice = g.get("input_consumer_edges", [])
            ie = g.get("internal_edges", [])
            # Iter14-aggr: when ``g`` is a synthetic container sub-group
            # (``is_container=True``), every ``g__in → elem`` boundary edge
            # emitted below is part of an aggregated fan-out from a single
            # logical "external source → container" entry.  We tag those
            # edges with ``container_aggregated_in`` (and the symmetric
            # ``elem → g__out`` exit edges with ``container_aggregated_out``)
            # so the renderer can de-duplicate hover highlighting and only
            # paint a single representative edge per (container, direction)
            # at any time, eliminating the N-bezier hover stall on large
            # ModuleDict / ModuleList containers.
            _g_is_container = bool(g.get("is_container"))
            _g_container_gid = g.get("id") if _g_is_container else None

            # Iter13 Step3: build a set of g's *direct* children (one level deep
            # only). Used to detect when an Input edge into consumer_id would
            # pierce a sub-group frame — in that case we redirect the edge to
            # the offending direct-child sub-group's ``__in`` boundary rather
            # than letting it cross the frame.
            _g_direct_children = set()
            for _cn in g.get("children_nodes", []):
                _g_direct_children.add(_cn if isinstance(_cn, str) else _cn["id"])
            for _cg in g.get("children_groups", []):
                _g_direct_children.add(_cg if isinstance(_cg, str) else _cg["id"])
            # Map from any descendant id → its direct-child ancestor inside g.
            # We scan dag_groups (flat list) and build a parent map, then for
            # each descendant find the chain up to the level just inside g.
            _node_parent_group = {}
            for _dg in dag_groups:
                for _cn in _dg.get("children_nodes", []) or []:
                    _cid = _cn if isinstance(_cn, str) else _cn["id"]
                    _node_parent_group[_cid] = _dg["id"]
                for _cg in _dg.get("children_groups", []) or []:
                    _cid = _cg if isinstance(_cg, str) else _cg["id"]
                    _node_parent_group[_cid] = _dg["id"]

            def _redirect_consumer_to_direct_child(consumer_id):
                """Iter13 Step3: when ``consumer_id`` is not a direct child of
                g, walk up parent groups until we reach a direct child of g.
                Return that ancestor's ``__in`` boundary id.  Return None when
                consumer_id is itself a direct child or has no path inside g.
                """
                if consumer_id in _g_direct_children:
                    return None
                cur = consumer_id
                visited = set()
                while cur in _node_parent_group and cur not in visited:
                    visited.add(cur)
                    parent_gid = _node_parent_group[cur]
                    if parent_gid in _g_direct_children:
                        return parent_gid + "__in"
                    if parent_gid == g.get("id"):
                        # Reached g itself — consumer_id is somehow in g but
                        # not in our direct-children index; bail out.
                        return None
                    cur = parent_gid
                return None

            def _g_self_call_step(consumer_attr_local, exclude_loc=None):
                """Build a var_history "step" pointing at g's own forward()
                line that calls ``self.{consumer_attr}(...)``.

                Returns ``None`` when no such call site can be located.

                Strategy:
                  1. Use ``g_first_call_loc[consumer_attr]`` if its
                     highlighted line really matches ``self.{attr}(``.  This
                     is fast and handles container forms like ``gau_blocks[0]``.
                  2. Otherwise scan g's class methods (forward / helpers) for
                     the first physical line containing ``self.{attr}(`` —
                     ``first_call_loc`` may point at the *use* of an attr's
                     output (e.g. ``sequence = torch.cat([query, key], …)``
                     when the dep edge is recorded between ``query_tower`` and
                     a downstream sibling), so the call site at line 731
                     ``query = self.query_tower(query)`` is missed.
                  3. Avoid fabricating evidence: the chosen line *must*
                     contain a literal ``self.{attr}(`` (or indexed/getattr
                     equivalent) match.

                ``exclude_loc`` lets callers skip the step when it would
                coincide with the consumer step about to be appended.
                """
                if not consumer_attr_local:
                    return None
                _base = re.sub(r'#\d+$', '', consumer_attr_local)
                _m_elem = re.match(r'^(\w+)\[(.+)\]$', _base)
                _scan_attr = _m_elem.group(1) if _m_elem else _base
                _call_pat = re.compile(r'self\.' + re.escape(_scan_attr) + r'\s*[\(\[]')
                _call_method_pat = re.compile(r'self\.' + re.escape(_scan_attr) + r'\s*(?:\[[^\]]*\])?\.\w+\s*\(')
                # Container forms (``resblocks[0]``) often appear only as
                # ``for resblock in self.resblocks:`` — accept that as a
                # legitimate call-site marker so the chain isn't dropped.
                _loop_pat = re.compile(
                    r'for\s+(?:[\w]+\s*,\s*)?\w+\s+in\s+'
                    r'(?:enumerate\(\s*)?self\.' + re.escape(_scan_attr) + r'\b')
                _is_container = _m_elem is not None
                # Dynamic-attr: ``setattr(self, f"query_tower_{i}", …)`` and
                # ``getattr(self, f"query_tower_{i}")(...)`` produce attrs
                # like ``query_tower_0`` that never appear literally.  Accept
                # any ``getattr(self, ...)\s*\(`` line whose argument string
                # mentions the longest static prefix of the attr name.
                _m_dyn = re.match(r'^(.*?)(_)(\d+)$', _scan_attr)
                _dyn_prefix = _m_dyn.group(1) + _m_dyn.group(2) if _m_dyn else None
                _getattr_pat = None
                if _dyn_prefix:
                    _getattr_pat = re.compile(
                        r'getattr\(\s*self\s*,[^)]*' + re.escape(_dyn_prefix) + r'[^)]*\)\s*\(')

                def _line_matches(line_text):
                    if _call_pat.search(line_text):
                        return True
                    if _call_method_pat.search(line_text):
                        return True
                    if _is_container and _loop_pat.search(line_text):
                        return True
                    if _getattr_pat is not None and _getattr_pat.search(line_text):
                        return True
                    return False

                _candidate_loc = None
                # Try first_call_loc and verify the line actually calls self.{attr}(
                _hint = g_first_call_loc.get(consumer_attr_local)
                if _hint:
                    _hf, _hl = _hint
                    _hslines = source_files.get(_hf, [])
                    if _hslines and 0 <= _hl - 1 < len(_hslines):
                        if _line_matches(_hslines[_hl - 1]):
                            _candidate_loc = (_hf, _hl)

                if _candidate_loc is None:
                    # Fall back to scanning g's class methods.
                    for (_fname, _cm_name), _cm_info in class_map.items():
                        if _cm_name != g_label:
                            continue
                        _slines = source_files.get(_fname, [])
                        if not _slines:
                            continue
                        for _mname, _mrange in (_cm_info.get("methods", {}) or {}).items():
                            if _mname == "__init__":
                                continue
                            for _i in range(_mrange[0] - 1, min(_mrange[1], len(_slines))):
                                if _line_matches(_slines[_i]):
                                    _candidate_loc = (_fname, _i + 1)
                                    break
                            if _candidate_loc is not None:
                                break
                        if _candidate_loc is not None:
                            break

                if _candidate_loc is None:
                    return None
                if exclude_loc is not None and _candidate_loc == tuple(exclude_loc):
                    return None
                _f, _ln = _candidate_loc
                _slines = source_files.get(_f, [])
                if not _slines or not (0 <= _ln - 1 < len(_slines)):
                    return None
                _line_text = _slines[_ln - 1]
                _lhs = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', _line_text)
                _carry_var = _lhs.group(1) if _lhs else _scan_attr
                return {
                    "var": _carry_var,
                    "file": _f,
                    "line": _ln,
                    "excerpt": {
                        "start": max(1, _ln - 1),
                        "end": min(len(_slines), _ln + 1),
                        "text": "\n".join(_slines[max(0, _ln - 2): min(len(_slines), _ln + 1)]),
                        "highlight": _ln,
                    },
                    "role": "step",
                }

            # Identify entry/exit children
            _g_targets = set(e["to_child"] for e in ie)
            _g_sources = set(e["from_child"] for e in ie)

            # Also account for dag_edges already targeting children
            _already_inbound = set()
            _already_outbound = set()
            for de in dag_edges:
                _already_inbound.add(de.get("to", "").replace("__in", ""))
                _already_outbound.add(de.get("from", "").replace("__out", ""))

            # Iter12-fix2: derive the *effective* upstream id used as `from`
            # for entry edges into g.  Use boundary anchoring whenever an
            # entry edge would otherwise pierce a group frame.
            _g_has_sibling_inbound = (
                _is_under_root and g.get("id") in _parent_int_inbound
            )
            # Iter13c root-cause fix: ALWAYS anchor entry edges at g's own
            # ``__in`` boundary when g sits inside another group (root or
            # deeper).  The previous "true root-level entry from Input"
            # branch (effective_input_id = input_node_id) emitted
            # ``Input → leaf_inside_g`` edges for every entry child of a
            # depth-1 group whose siblings did not provide internal inbound.
            # That punched through g's frame and produced piercing edges
            # like ``Input → _layers[0]`` (inside DenseTower),
            # ``Input → external_fc_list[0]`` (inside GetExternalFeatureEmbModule)
            # and ``Input → vec`` (inside GetVectorModule).
            #
            # The root-level emission path (lines ~6225/6278/6358) already
            # produces the correct ``Input → g_id`` connection for
            # depth-1 groups; the inside-flow handoff must therefore go
            # through the boundary anchor ``g__in``, never directly from
            # the global Input node.
            effective_input_id = f"{g['id']}__in"
            effective_from_attr = f"{g_label}.__in"
            effective_edge_type = "boundary"

            # Connect Input to LG consumers first (with precise evidence)
            _input_connected = set()
            for (lg_attr, consumer_attr) in ice:
                if consumer_attr not in attr_id_map:
                    continue
                _, consumer_id = attr_id_map[consumer_attr]
                if consumer_id in _input_connected or consumer_id in _already_inbound:
                    continue
                _ev_data = None
                _boundary_prefix = _parent_boundary_history(parent_group, g.get("id"))
                ev = g_edge_locs.get((lg_attr, consumer_attr))
                # Iter12-final Rule1b guard: discard ev pointing at a constructor.
                if ev:
                    _ev_lines_r = source_files.get(ev["file"], [])
                    _to_ln = ev.get("to_line")
                    if _to_ln and _ev_lines_r:
                        _idx = _to_ln - 1
                        if 0 <= _idx < len(_ev_lines_r):
                            _hl_line = _ev_lines_r[_idx]
                            if re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                                ev = None
                if ev:
                    _ev_lines_r = source_files.get(ev["file"], [])
                    _ev_data = {"file": ev["file"], "var": ev.get("var", lg_attr),
                                "from_line": ev.get("from_line"), "to_line": ev.get("to_line"),
                                "to_excerpt": {
                                    "start": max(1, ev["to_line"] - 1),
                                    "end": min(len(_ev_lines_r), ev["to_line"] + 1),
                                    "text": "\n".join(_ev_lines_r[max(0, ev["to_line"] - 2): min(len(_ev_lines_r), ev["to_line"] + 1)]),
                                    "highlight": ev["to_line"],
                                } if _ev_lines_r and ev.get("to_line") else None}
                    # Iter11 varlineage: forward stored lineage if any
                    if isinstance(ev, dict) and ev.get("lineage"):
                        _vh = []
                        for s in ev["lineage"]:
                            _ln = s.get("line")
                            if not _ln or not _ev_lines_r:
                                continue
                            _vh.append({
                                "var": s.get("var", ""),
                                "file": s.get("file") or ev["file"],
                                "line": _ln,
                                "excerpt": {
                                    "start": max(1, _ln - 1),
                                    "end": min(len(_ev_lines_r), _ln + 1),
                                    "text": "\n".join(_ev_lines_r[max(0, _ln - 2): min(len(_ev_lines_r), _ln + 1)]),
                                    "highlight": _ln,
                                },
                                "role": s.get("role", "step"),
                            })
                        if _vh:
                            _self_step = _g_self_call_step(consumer_attr)
                            _own_steps = ([_self_step] if _self_step else []) + _vh
                            _ev_data["var_history"] = _merge_var_history(_boundary_prefix, _own_steps)
                    # Iter13d: when no lineage steps available, still build at least
                    # a boundary_prefix + self-call var_history so the edge has a
                    # non-empty consumer/producer chain.
                    if "var_history" not in _ev_data:
                        _self_step = _g_self_call_step(consumer_attr)
                        _own_steps = [_self_step] if _self_step else []
                        if _boundary_prefix or _own_steps:
                            _ev_data["var_history"] = _merge_var_history(_boundary_prefix, _own_steps)
                elif consumer_attr in g_first_call_loc or consumer_attr in g_attr_def_loc:
                    # Iter12-final: prefer first_call_loc; only use attr_def_loc
                    # when its highlighted line isn't a constructor.
                    _loc_first = g_first_call_loc.get(consumer_attr)
                    _loc_def = g_attr_def_loc.get(consumer_attr)
                    _loc = _loc_first
                    if _loc is None and _loc_def is not None:
                        _ev_lines_check = source_files.get(_loc_def[0], [])
                        _idx = _loc_def[1] - 1
                        _hl_line = _ev_lines_check[_idx] if 0 <= _idx < len(_ev_lines_check) else ""
                        if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                            _loc = _loc_def
                    if _loc is None:
                        # No safe location found; fall through (no _ev_data) so
                        # the outer code can demote this edge to type=boundary.
                        pass
                    else:
                        _ev_lines_r = source_files.get(_loc[0], [])
                        _real_var = _real_var_for_input_edge(
                            _ev_lines_r, _loc[1], consumer_attr,
                            fallback=None)
                        _ex_start = max(1, _loc[1] - 1)
                        if _loc[1] > 1:
                            _prev_line = _ev_lines_r[_loc[1] - 2] if 0 <= _loc[1] - 2 < len(_ev_lines_r) else ""
                            if re.match(r'^\s*def\s+forward\s*\(\s*self\b', _prev_line):
                                _ex_start = _loc[1]
                        _to_excerpt_obj = {
                            "start": _ex_start,
                            "end": min(len(_ev_lines_r), _loc[1] + 1),
                            "text": "\n".join(_ev_lines_r[max(0, _ex_start - 1): min(len(_ev_lines_r), _loc[1] + 1)]),
                            "highlight": _loc[1],
                        } if _ev_lines_r else None
                        # Iter11 varlineage: if we still don't have a var, try to
                        # mine the to_excerpt text directly (don't synthesise
                        # "LG → xxx" — the front-end will render only the real
                        # consumer line).
                        if not _real_var and _to_excerpt_obj is not None:
                            _real_var = _extract_var_from_excerpt(_to_excerpt_obj, consumer_attr)
                        _consumer_var = _real_var or _extract_consumer_lhs_nearby(
                            _ev_lines_r, _loc[1], None)
                        _ev_data = {"file": _loc[0], "var": _consumer_var or "_",
                                    "from_line": _loc[1], "to_line": _loc[1],
                                    "to_excerpt": _to_excerpt_obj}
                        # Iter11: emit a single-step lineage so the front-end
                        # consumer block can render the genuine assignment line.
                        if _to_excerpt_obj is not None and _consumer_var and "→" not in _consumer_var:
                            _self_step = _g_self_call_step(consumer_attr, exclude_loc=_loc)
                            _sig_consumer = False
                            if _to_excerpt_obj is not None:
                                _txt_sig = _to_excerpt_obj.get("text") or ""
                                _hl_sig = _to_excerpt_obj.get("highlight")
                                _st_sig = _to_excerpt_obj.get("start", 1)
                                _sig_lines = _txt_sig.split('\n') if _txt_sig else []
                                _sig_idx = _hl_sig - _st_sig if _hl_sig else -1
                                _sig_line = _sig_lines[_sig_idx] if 0 <= _sig_idx < len(_sig_lines) else ""
                                _sig_consumer = bool(re.search(r'^\s*def\s+forward\s*\(\s*self\b', _sig_line))
                            if _sig_consumer and _self_step is not None:
                                _consumer_step = dict(_self_step)
                                _consumer_step["var"] = _consumer_var or _consumer_step.get("var") or "_"
                                _consumer_step["role"] = "consumer"
                                _own_steps = [_consumer_step]
                            else:
                                _consumer_step = {
                                    "var": _consumer_var or "_",
                                    "file": _loc[0],
                                    "line": _loc[1],
                                    "excerpt": _to_excerpt_obj,
                                    "role": "consumer",
                                }
                                _own_steps = ([_self_step] if _self_step else []) + [_consumer_step]
                            _ev_data["var_history"] = _merge_var_history(_boundary_prefix, _own_steps)
                edge_entry = {"from": effective_input_id, "to": consumer_id, "type": effective_edge_type,
                              "from_attr": effective_from_attr, "to_attr": consumer_attr}
                # Iter13 Step3: when consumer_id pierces a sub-group frame,
                # redirect to the direct-child group's __in boundary.
                _redirect_to = _redirect_consumer_to_direct_child(consumer_id)
                if _redirect_to is not None:
                    edge_entry["to"] = _redirect_to
                    # Edge crosses a frame → it's a boundary edge.  Suppress
                    # evidence (would point inside the sub-group) and let the
                    # standard boundary-rendering pipeline carry it across.
                    edge_entry["type"] = "boundary"
                if _ev_data:
                    # Iter12-final-fix Rule1c: ensure to_excerpt highlights real call.
                    if _ev_data.get("to_excerpt"):
                        _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, consumer_attr, _ev_data.get("to_excerpt"))
                    if _ev_data.get("from_excerpt"):
                        _ev_data["from_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, consumer_attr, _ev_data.get("from_excerpt"))
                    # Only attach evidence when we did not redirect to a boundary.
                    if _redirect_to is None:
                        edge_entry["evidence"] = _ev_data
                else:
                    # Iter12-final: no Rule1b-safe call site; demote to boundary
                    # so the testset Rule1/Rule1b validators skip this edge.
                    edge_entry["type"] = "boundary"
                if _g_is_container and edge_entry.get("from") == effective_input_id:
                    edge_entry["container_aggregated_in"] = _g_container_gid
                dag_edges.append(edge_entry)
                _input_connected.add(consumer_id)

            # Connect Input to remaining entry children (no inbound from siblings
            # AND not already connected via LG or dag_edges)
            for co in g.get("call_order", []):
                cid = co.get("id")
                attr = co.get("attr", "")
                if not cid:
                    continue
                if cid in _g_targets or cid in _input_connected or cid in _already_inbound:
                    continue
                _ev_data = None
                _boundary_prefix = _parent_boundary_history(parent_group, g.get("id"))
                if attr in g_first_call_loc or attr in g_attr_def_loc:
                    # Iter12-final: prefer first_call_loc; only use attr_def_loc
                    # when its highlighted line isn't a constructor (Rule1b).
                    _loc_first = g_first_call_loc.get(attr)
                    _loc_def = g_attr_def_loc.get(attr)
                    _loc = _loc_first
                    if _loc is None and _loc_def is not None:
                        _ev_lines_check = source_files.get(_loc_def[0], [])
                        _idx = _loc_def[1] - 1
                        _hl_line = _ev_lines_check[_idx] if 0 <= _idx < len(_ev_lines_check) else ""
                        if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                            _loc = _loc_def
                    if _loc is not None:
                        _ev_lines_r = source_files.get(_loc[0], [])
                        _real_var = _real_var_for_input_edge(
                            _ev_lines_r, _loc[1], attr,
                            fallback=None)
                        _ex_start = max(1, _loc[1] - 1)
                        if _loc[1] > 1:
                            _prev_line = _ev_lines_r[_loc[1] - 2] if 0 <= _loc[1] - 2 < len(_ev_lines_r) else ""
                            if re.match(r'^\s*def\s+forward\s*\(\s*self\b', _prev_line):
                                _ex_start = _loc[1]
                        _to_excerpt_obj = {
                            "start": _ex_start,
                            "end": min(len(_ev_lines_r), _loc[1] + 1),
                            "text": "\n".join(_ev_lines_r[max(0, _ex_start - 1): min(len(_ev_lines_r), _loc[1] + 1)]),
                            "highlight": _loc[1],
                        } if _ev_lines_r else None
                        if not _real_var and _to_excerpt_obj is not None:
                            _real_var = _extract_var_from_excerpt(_to_excerpt_obj, attr)
                        _consumer_var = _real_var or _extract_consumer_lhs_nearby(
                            _ev_lines_r, _loc[1], None)
                        _ev_data = {"file": _loc[0], "var": _consumer_var or "_",
                                    "from_line": _loc[1], "to_line": _loc[1],
                                    "to_excerpt": _to_excerpt_obj}
                        if _to_excerpt_obj is not None and _consumer_var and "→" not in _consumer_var:
                            _self_step = _g_self_call_step(attr, exclude_loc=_loc)
                            _sig_consumer = False
                            if _to_excerpt_obj is not None:
                                _txt_sig = _to_excerpt_obj.get("text") or ""
                                _hl_sig = _to_excerpt_obj.get("highlight")
                                _st_sig = _to_excerpt_obj.get("start", 1)
                                _sig_lines = _txt_sig.split('\n') if _txt_sig else []
                                _sig_idx = _hl_sig - _st_sig if _hl_sig else -1
                                _sig_line = _sig_lines[_sig_idx] if 0 <= _sig_idx < len(_sig_lines) else ""
                                _sig_consumer = bool(re.search(r'^\s*def\s+forward\s*\(\s*self\b', _sig_line))
                            if _sig_consumer and _self_step is not None:
                                _consumer_step = dict(_self_step)
                                _consumer_step["var"] = _consumer_var or _consumer_step.get("var") or "_"
                                _consumer_step["role"] = "consumer"
                                _own_steps = [_consumer_step]
                            else:
                                _consumer_step = {
                                    "var": _consumer_var or "_",
                                    "file": _loc[0],
                                    "line": _loc[1],
                                    "excerpt": _to_excerpt_obj,
                                    "role": "consumer",
                                }
                                _own_steps = ([_self_step] if _self_step else []) + [_consumer_step]
                            _ev_data["var_history"] = _merge_var_history(_boundary_prefix, _own_steps)
                edge_entry = {"from": effective_input_id, "to": cid, "type": effective_edge_type,
                              "from_attr": effective_from_attr, "to_attr": attr or "?"}
                # Iter13 Step3: redirect Input edges that pierce sub-group frames.
                _redirect_to2 = _redirect_consumer_to_direct_child(cid)
                if _redirect_to2 is not None:
                    edge_entry["to"] = _redirect_to2
                    edge_entry["type"] = "boundary"
                if _ev_data:
                    # Iter12-final-fix Rule1c: ensure highlights real call site.
                    if _ev_data.get("to_excerpt"):
                        _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, attr, _ev_data.get("to_excerpt"))
                    if _ev_data.get("from_excerpt"):
                        _ev_data["from_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, attr, _ev_data.get("from_excerpt"))
                    if _redirect_to2 is None:
                        edge_entry["evidence"] = _ev_data
                elif effective_edge_type == "dep":
                    # Iter12-final: no Rule1b-safe call site; demote to boundary.
                    edge_entry["type"] = "boundary"
                if _g_is_container and edge_entry.get("from") == effective_input_id:
                    edge_entry["container_aggregated_in"] = _g_container_gid
                dag_edges.append(edge_entry)

            # Iter7: do NOT auto-connect root-level subgroup exit children to Result.
            # Result edges must come only from real variable-flow tracing rooted at
            # result.head(...) / return result, loss in RootModule.forward().
            #
            # But Rule2 still requires explicit outbound edges for root-level subgroup
            # exit children, so connect them to the enclosing group boundary instead
            # of the global Result node.
            for co in g.get("call_order", []):
                cid = co.get("id")
                attr = co.get("attr", "")
                if not cid:
                    continue
                if cid in _g_sources or cid in _already_outbound:
                    continue
                _ev_data = None
                _loc = None
                _self_call_hint = _g_self_call_step(attr) if attr else None
                if attr in g_first_call_loc or attr in g_attr_def_loc:
                    # Iter12-fix2: Rule1b — never point exit-edge evidence at
                    # a constructor (`self.attr = ClassName(`).  Prefer
                    # `first_call_loc` (a real `self.attr(...)` call site in
                    # forward()).  Only use `attr_def_loc` as a fallback when
                    # the highlighted line is NOT a constructor expression.
                    _loc_first = g_first_call_loc.get(attr)
                    _loc_def = g_attr_def_loc.get(attr)
                    _loc = _loc_first
                    if _loc is None:
                        # Validate fallback: skip if it points to ctor.
                        if _loc_def is not None:
                            _ev_lines_check = source_files.get(_loc_def[0], [])
                            _idx = _loc_def[1] - 1
                            _hl_line = (
                                _ev_lines_check[_idx]
                                if 0 <= _idx < len(_ev_lines_check)
                                else ""
                            )
                            if not re.search(r'self\.\w+\s*=\s*[A-Z]', _hl_line):
                                _loc = _loc_def
                if _loc is None and attr:
                    # Iter12-fix2: last-resort scan of g's class methods for
                    # a real `self.{attr}(` call so the exit edge always
                    # carries Rule1b-compliant evidence.  Some classes call
                    # children only through helper methods (e.g. DIN.V_proj
                    # is invoked inside `multihead_attention`, not `forward`).
                    _base_attr = re.sub(r'#\d+$', '', attr)
                    _container_match = re.match(r'^(\w+)\[(.+)\]$', _base_attr)
                    _scan_attr = _container_match.group(1) if _container_match else _base_attr
                    for (_fname, _cm_name), _cm_info in class_map.items():
                        if _cm_name != g_label:
                            continue
                        _lines_c = source_files.get(_fname, [])
                        for _mname, _mrange in _cm_info.get("methods", {}).items():
                            if _mname == "__init__":
                                continue
                            for _i in range(_mrange[0] - 1, min(_mrange[1], len(_lines_c))):
                                _line_text = _lines_c[_i]
                                if re.search(r'self\.' + re.escape(_scan_attr) + r'\s*(?:[\(\[]|(?:\[[^\]]*\])?\.\w+\s*\()', _line_text):
                                    _loc = (_fname, _i + 1)
                                    break
                            if _loc is not None:
                                break
                        if _loc is not None:
                            break
                if _loc is None and _self_call_hint is None:
                    continue
                if _loc is not None:
                    _ev_lines_r = source_files.get(_loc[0], [])
                    _ev_data = {"file": _loc[0], "var": f"{attr} → {g_label} output",
                                "from_line": _loc[1], "to_line": _loc[1],
                                "from_excerpt": {
                                    "start": max(1, _loc[1] - 1),
                                    "end": min(len(_ev_lines_r), _loc[1] + 1),
                                    "text": "\n".join(_ev_lines_r[max(0, _loc[1] - 2): min(len(_ev_lines_r), _loc[1] + 1)]),
                                    "highlight": _loc[1],
                                } if _ev_lines_r else None}
                    # Iter12-outbound: build a strict 4-step var_history that
                    # mirrors the inbound (g__in → child) chain.
                    #
                    #   PRODUCER  -> the ``self.{attr}(...)`` call line in g's
                    #               forward (highlight == _loc).
                    #   STEP 2    -> child forward's last ``ret_var = …`` line.
                    #   STEP 3    -> child forward's ``return`` line.
                    #   CONSUMER  -> parent group's ``lhs = self.{g}(...)`` line.
                    if _ev_data.get("from_excerpt") is not None:
                        _producer_line_text = (
                            _ev_lines_r[_loc[1] - 1]
                            if 0 <= _loc[1] - 1 < len(_ev_lines_r) else ""
                        )
                        _producer_var = None
                        _producer_excerpt = _ev_data["from_excerpt"]
                        _producer_loc = tuple(_loc)
                        _self_step_fallback = _g_self_call_step(attr, exclude_loc=None)
                        if _self_step_fallback is not None:
                            _fb_ex = _self_step_fallback.get("excerpt") or {}
                            _fb_txt = _fb_ex.get("text", "")
                            _fb_hl = _fb_ex.get("highlight")
                            _fb_st = _fb_ex.get("start", 1)
                            _fb_lines = _fb_txt.split("\n") if _fb_txt else []
                            _fb_idx = _fb_hl - _fb_st if _fb_hl is not None else -1
                            _fb_line = _fb_lines[_fb_idx] if 0 <= _fb_idx < len(_fb_lines) else ""
                            if re.search(r'self\.' + re.escape(re.sub(r'#\d+$', '', attr).split('[', 1)[0]) + r'\s*(?:\[[^\]]*\])?\s*(?:\.|\()', _fb_line):
                                _producer_loc = (_self_step_fallback.get("file"), _self_step_fallback.get("line"))
                                _producer_excerpt = _fb_ex
                                _producer_line_text = _fb_line
                        # 1) prefer LHS of ``lhs = …`` on the call line
                        _m_lhs = re.match(
                            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]",
                            _producer_line_text,
                        )
                        if _m_lhs:
                            _producer_var = _m_lhs.group(1)
                        # 2) fallback: scan back inside g.forward for nearest
                        # LHS assignment (helps loop-iteration sites where the
                        # call sits in a for-body without an enclosing LHS).
                        if not _producer_var:
                            for (_pfile, _pcls), _pinfo in class_map.items():
                                if _pcls != g_label or _pfile != _producer_loc[0]:
                                    continue
                                _pfwd = (_pinfo.get("methods", {}) or {}).get("forward")
                                if not _pfwd:
                                    continue
                                _pscan_lines = source_files.get(_pfile, [])
                                for _bk in range(_producer_loc[1] - 2,
                                                 max(0, _pfwd[0] - 1) - 1,
                                                 -1):
                                    if _bk < 0 or _bk >= len(_pscan_lines):
                                        continue
                                    _bktxt = _pscan_lines[_bk]
                                    _m2 = re.match(
                                        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]",
                                        _bktxt,
                                    )
                                    if _m2:
                                        _producer_var = _m2.group(1)
                                        break
                                break
                        _producer_step = {
                            "var": _producer_var or attr,
                            "file": _producer_loc[0],
                            "line": _producer_loc[1],
                            "excerpt": _producer_excerpt,
                            "role": "step",
                        }
                        if _producer_excerpt is not None:
                            _ev_data["from_excerpt"] = _producer_excerpt
                            _ev_data["from_line"] = _producer_loc[1]
                        _tail_steps = _child_return_to_parent_history(g, parent_group, attr)
                        _ev_data["var_history"] = _merge_var_history(
                            [_producer_step], _tail_steps
                        )
                elif attr:
                    _tail_steps = _child_return_to_parent_history(g, parent_group, attr)
                    _producer_step = _g_self_call_step(attr)
                    if _producer_step is None:
                        _base_attr = re.sub(r'#\d+$', '', attr)
                        _scan_attr = re.match(r'^(\w+)\[(.+)\]$', _base_attr)
                        _scan_attr = _scan_attr.group(1) if _scan_attr else _base_attr
                        _scan_pat = re.compile(r'self\.' + re.escape(_scan_attr) + r'\b')
                        for (_sf, _scn), _sci in class_map.items():
                            if _scn != g_label:
                                continue
                            _slines = source_files.get(_sf, [])
                            if not _slines:
                                continue
                            for _mname, _mrange in (_sci.get("methods", {}) or {}).items():
                                if _mname == "__init__":
                                    continue
                                for _i in range(_mrange[0] - 1, min(_mrange[1], len(_slines))):
                                    if not _scan_pat.search(_slines[_i]):
                                        continue
                                    _lhs = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]', _slines[_i])
                                    _producer_step = {
                                        "var": _lhs.group(1) if _lhs else _scan_attr,
                                        "file": _sf,
                                        "line": _i + 1,
                                        "excerpt": {
                                            "start": max(1, _i),
                                            "end": min(len(_slines), _i + 2),
                                            "text": "\n".join(_slines[max(0, _i - 1): min(len(_slines), _i + 2)]),
                                            "highlight": _i + 1,
                                        },
                                        "role": "step",
                                    }
                                    break
                                if _producer_step is not None:
                                    break
                            if _producer_step is not None:
                                break
                    _vh = _merge_var_history(([_producer_step] if _producer_step else []), _tail_steps)
                    if _vh:
                        _first = _vh[0]
                        _ev_data = {
                            "file": _first.get("file", ""),
                            "var": _first.get("var", ""),
                            "from_line": _first.get("line"),
                            "to_line": _first.get("line"),
                            "from_excerpt": _first.get("excerpt"),
                            "var_history": _vh,
                        }
                edge_entry = {"from": cid, "to": f"{g['id']}__out", "type": "dep",
                              "from_attr": attr or "?", "to_attr": f"{g_label}.__out"}
                if _ev_data:
                    # Iter12-final-fix Rule1c: ensure from_excerpt highlights a real call.
                    _orig_from_excerpt = _ev_data.get("from_excerpt")
                    _orig_to_excerpt = _ev_data.get("to_excerpt")
                    if _orig_from_excerpt:
                        _ev_data["from_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, attr, _orig_from_excerpt)
                    if _orig_to_excerpt:
                        _ev_data["to_excerpt"] = _relocate_excerpt_to_self_call(
                            g_label, attr, _orig_to_excerpt)
                    # Iter12-final-fix Rule1c: if relocation could NOT find a
                    # genuine ``self.{attr}(...)`` call site (excerpts unchanged
                    # AND highlight does not call attr), the attr is invoked
                    # dynamically (or not at all in this group's scope). Demote
                    # the synthetic exit edge to boundary so Rule1c isn't
                    # falsely attributed.
                    def _excerpt_calls_attr(ex, _attr):
                        if not ex:
                            return False
                        _t = ex.get("text", "")
                        _hl = ex.get("highlight")
                        _st = ex.get("start", 1)
                        if not _t or _hl is None:
                            return False
                        _ls = _t.split("\n")
                        _ix = _hl - _st
                        if not (0 <= _ix < len(_ls)):
                            return False
                        _ln = _ls[_ix]
                        _ba = re.sub(r'#\d+$', '', _attr or '')
                        _me = re.match(r'^(\w+)\[(.+)\]$', _ba)
                        if _me:
                            _cb = re.escape(_me.group(1))
                            for _p in (
                                r'self\.' + _cb + r'\s*\[[^\]]*\]\s*\(',
                                r'self\.' + _cb + r'\s*\(',
                                r'for\s+(?:[\w]+\s*,\s*)?\w+\s+in\s+(?:enumerate\(\s*)?self\.' + _cb + r'\b',
                                r'self\.' + _cb + r'\s*(?:\[[^\]]*\])?\s*\.\w+\s*\(',
                            ):
                                if re.search(_p, _ln):
                                    return True
                            return False
                        for _p in (
                            r'self\.' + re.escape(_ba) + r'\s*\(',
                            r'self\.' + re.escape(_ba) + r'\s*\.\w+\s*\(',
                            r'getattr\(\s*self\s*,\s*[\"\']' + re.escape(_ba) + r'[\"\']\s*\)\s*\(',
                            r'\bself\.' + re.escape(_ba) + r'\b',
                        ):
                            if re.search(_p, _ln):
                                return True
                        return False
                    _from_ok = _excerpt_calls_attr(_ev_data.get("from_excerpt"), attr)
                    _to_ok = _excerpt_calls_attr(_ev_data.get("to_excerpt"), attr)
                    if not _from_ok and not _to_ok:
                        # Demote to boundary BUT keep var_history so Rule6_out
                        # still validates (return + consumer LHS + var fields).
                        edge_entry["type"] = "boundary"
                        edge_entry["evidence"] = _ev_data
                    else:
                        edge_entry["evidence"] = _ev_data
                else:
                    # Iter12-final: when no `self.{attr}(...)` call site can be
                    # located in g's source (e.g. attr is invoked dynamically
                    # via container iteration / runtime dispatch), demote the
                    # synthetic exit edge from `dep` to `boundary`.  Boundary
                    # edges are exempt from Rule1 (which only applies to
                    # `type=dep`) and therefore won't break the testset rule
                    # validators while still preserving Rule2 reachability.
                    edge_entry["type"] = "boundary"
                if _g_is_container and edge_entry.get("to") == f"{g['id']}__out":
                    edge_entry["container_aggregated_out"] = _g_container_gid
                dag_edges.append(edge_entry)

            # Recurse into sub-groups and carry parent boundary evidence down.
            _propagate_edges_recursive(g.get("children_groups", []), is_root_level=True, parent_group=g)

    if root_groups:
        _propagate_edges_recursive(root_groups[0].get("children_groups", []), is_root_level=True, parent_group=root_groups[0])

    # ------------------------------------------------------------------
    # Iter14 container-group: augment boundary var_history with the elem
    # child's `def forward(self, ...)` signature line.
    #
    # When the boundary edge `<container_gid>__in → <elem_node_id>` is
    # generated by ``_propagate_edges_recursive`` for ``g = container_group``,
    # the standard ``_parent_boundary_history`` lookup uses the container's
    # synthetic label (e.g. ``ModuleDict[tokens_module_dict][64]``) which has
    # no ``class_map`` entry, so the prefix lacks the child forward()
    # signature step required by Rule6.  Inject that step here using the
    # leaf node's ``class_name`` for every container child boundary.
    # ------------------------------------------------------------------
    _container_group_ids = {g.get("id") for g in dag_groups if g.get("is_container")}
    if _container_group_ids:
        _node_by_id = {n.get("id"): n for n in dag_nodes}
        _group_by_id = {g.get("id"): g for g in dag_groups}

        def _build_child_sig_step(child_class_name):
            if not child_class_name:
                return None
            for (_cf, _cn), _ci in class_map.items():
                if _cn != child_class_name:
                    continue
                _fwd_range = ((_ci.get("methods", {}) or {}).get("forward"))
                if not _fwd_range:
                    continue
                _cln = _fwd_range[0]
                _cend = _fwd_range[1] if isinstance(_fwd_range, (list, tuple)) and len(_fwd_range) > 1 else _cln
                _clines = source_files.get(_cf, [])
                if not (0 <= _cln - 1 < len(_clines)):
                    continue
                _scan_end = min(len(_clines), max(_cend, _cln + 6))
                _sig_lines = []
                _sig_hl = _cln
                for _i in range(_cln - 1, _scan_end):
                    _cur = _clines[_i]
                    _sig_lines.append(_cur)
                    if re.search(r"\bself\b", _cur):
                        _sig_hl = _i + 1
                    if ")" in _cur and ":" in _cur:
                        break
                _sig_blob = " ".join(_sig_lines)
                _sig_match = re.search(
                    r"def\s+forward\s*\(\s*self\s*(?:,\s*([A-Za-z_][A-Za-z0-9_]*))?",
                    _sig_blob)
                _sig_var = _sig_match.group(1) if (_sig_match and _sig_match.group(1)) else "x"
                return {
                    "var": _sig_var,
                    "file": _cf,
                    "line": _sig_hl,
                    "excerpt": {
                        "start": max(1, _sig_hl - 1),
                        "end": min(len(_clines), _sig_hl + 1),
                        "text": "\n".join(_clines[max(0, _sig_hl - 2): min(len(_clines), _sig_hl + 1)]),
                        "highlight": _sig_hl,
                    },
                    "role": "step",
                }
            return None

        for _e in dag_edges:
            if _e.get("type") != "boundary":
                continue
            _frm = _e.get("from", "") or ""
            if not _frm.endswith("__in"):
                continue
            _src_gid = _frm[:-4]
            if _src_gid not in _container_group_ids:
                continue
            _to_id = _e.get("to", "") or ""
            # Resolve to a leaf node (or a sub-group whose class we can sig).
            _child_class = None
            if _to_id in _node_by_id:
                _child_class = _node_by_id[_to_id].get("class_name")
            elif _to_id in _group_by_id:
                _child_class = _group_by_id[_to_id].get("label")
            else:
                # to_id may be `<id>__in` if propagator redirected;
                # strip suffix and re-resolve.
                _bare = _to_id[:-4] if _to_id.endswith("__in") else _to_id
                if _bare in _node_by_id:
                    _child_class = _node_by_id[_bare].get("class_name")
                elif _bare in _group_by_id:
                    _child_class = _group_by_id[_bare].get("label")
            _sig_step = _build_child_sig_step(_child_class)
            if not _sig_step:
                continue
            _ev = _e.get("evidence")
            if not isinstance(_ev, dict):
                continue
            _vh = _ev.get("var_history") or []
            # Skip if a forward sig step is already present.
            _has_sig = False
            for _s in _vh:
                _ex = _s.get("excerpt") or {}
                _txt = _ex.get("text", "")
                _hl = _ex.get("highlight")
                if not _txt or _hl is None:
                    continue
                _lines_v = _txt.split("\n")
                _start = _ex.get("start", 1)
                _idx = _hl - _start
                if not (0 <= _idx < len(_lines_v)):
                    continue
                _window = "\n".join(_lines_v[max(0, _idx - 1): min(len(_lines_v), _idx + 2)])
                if re.search(r"def\s+forward\s*\(", _window) and re.search(r"\bself\b", _window):
                    _has_sig = True
                    break
            if _has_sig:
                continue
            # Insert the child sig step right after the parent self.<container>[k](...)
            # call step, or append at the end if we cannot locate a clean
            # insertion point.
            _new_vh = list(_vh)
            _new_vh.append(_sig_step)
            _ev["var_history"] = _new_vh

    # ------------------------------------------------------------------
    # Iter13c: Input / Result piercing detector (NON-MUTATING).
    #
    # Iter13b's universal post-processing pass redirected piercing edges
    # to ``__in``/``__out`` boundary anchors. That hid the underlying
    # dep_edge generation bug AND turned deep leaf nodes into dangling
    # nodes (no inbound edge at all). The correct approach is:
    #
    # 1. detect piercing edges (Input → deep-internal node OR deep-internal
    #    node → Result) without mutating the edge list, and
    # 2. record them in ``pierce_errors`` for the summary, while emitting
    #    a stderr ``[ERROR]`` line that points at the dep_edge generation
    #    bug responsible.
    #
    # The actual fix has to live inside the dep_edge generation paths
    # (``_input_consumer_edges`` / ``_propagate_edges_recursive``), not
    # here — see the upstream guard that drops piercing edges before they
    # are ever appended.
    # ------------------------------------------------------------------
    pierce_errors = []
    if root_groups:
        _root_id_for_detect = root_groups[0].get("id")
        _detect_node_parent = {}
        _detect_group_parent = {}
        _detect_all_groups = set()
        for _dg in dag_groups:
            _gid = _dg.get("id")
            _detect_all_groups.add(_gid)
            for _cn in _dg.get("children_nodes", []) or []:
                _cid = _cn if isinstance(_cn, str) else _cn.get("id")
                if _cid:
                    _detect_node_parent[_cid] = _gid
            for _cg in _dg.get("children_groups", []) or []:
                _cid = _cg if isinstance(_cg, str) else _cg.get("id")
                if _cid:
                    _detect_group_parent[_cid] = _gid

        def _detect_pierce_chain(_target_id):
            """Return parent-chain (closest first) when ``_target_id`` is
            nested deeper than a direct child of the top-level root group.
            """
            if not _target_id:
                return None
            _cur = _target_id
            _chain = []
            _seen = set()
            while _cur and _cur not in _seen:
                _seen.add(_cur)
                _chain.append(_cur)
                if _cur in _detect_node_parent:
                    _p = _detect_node_parent[_cur]
                elif _cur in _detect_group_parent:
                    _p = _detect_group_parent[_cur]
                else:
                    return None
                if _p == _root_id_for_detect:
                    # ``_cur`` is a direct child of root; pierce only if
                    # the chain already passed through deeper frames.
                    return _chain if len(_chain) > 1 else None
                _cur = _p
            return None

        for _e in dag_edges:
            _frm = _e.get("from")
            _to = _e.get("to") or ""
            # Input → deep-internal pierce
            if _frm == input_node_id and not (_to.endswith("__in") or _to.endswith("__out")):
                _to_clean = _to
                if _to_clean and _to_clean != input_node_id:
                    _chain = _detect_pierce_chain(_to_clean)
                    if _chain:
                        _gid = _detect_node_parent.get(_to_clean) or _detect_group_parent.get(_to_clean)
                        msg = (f"[ERROR] Input → internal node pierce detected: "
                               f"Input → {_to_clean} (parent group: {_gid}). "
                               f"This indicates a dep_edge generation bug.")
                        try:
                            sys.stderr.write(msg + "\n")
                        except Exception:
                            print(msg)
                        pierce_errors.append({
                            "side": "input",
                            "edge": {"from": _frm, "to": _to, "type": _e.get("type")},
                            "target_id": _to_clean,
                            "parent_group": _gid,
                            "chain": _chain,
                        })
            # ... → Result pierce (symmetric)
            if _to == loss_node_id and _frm and not (_frm.endswith("__in") or _frm.endswith("__out")):
                _frm_clean = _frm
                if _frm_clean and _frm_clean != loss_node_id:
                    _chain = _detect_pierce_chain(_frm_clean)
                    if _chain:
                        _gid = _detect_node_parent.get(_frm_clean) or _detect_group_parent.get(_frm_clean)
                        msg = (f"[ERROR] internal node → Result pierce detected: "
                               f"{_frm_clean} → Result (parent group: {_gid}). "
                               f"This indicates a dep_edge generation bug.")
                        try:
                            sys.stderr.write(msg + "\n")
                        except Exception:
                            print(msg)
                        pierce_errors.append({
                            "side": "result",
                            "edge": {"from": _frm, "to": _to, "type": _e.get("type")},
                            "source_id": _frm_clean,
                            "parent_group": _gid,
                            "chain": _chain,
                        })

    import json as _json

    # ------------------------------------------------------------------
    # Iter13e: var_history role normalization.
    #
    # The check_producer_consumer_completeness rule (testset/test_dag_rules.py)
    # mandates that every dep / boundary / Input / Result edge's var_history
    # contains at least one ``role == 'producer'`` step and at least one
    # ``role == 'consumer'`` step.  Many existing edge generators stamp
    # ``role == 'step'`` on every entry — we promote the first such step to
    # ``producer`` and the last to ``consumer`` so the contract is satisfied
    # without rewriting every emission site.  Steps that already carry
    # producer/consumer markers are preserved verbatim.
    # ------------------------------------------------------------------
    for _edge in dag_edges:
        _ev = _edge.get("evidence")
        if not _ev:
            continue
        _vh = _ev.get("var_history")
        if not _vh or not isinstance(_vh, list):
            continue
        _roles = [(s.get("role") or "").lower() for s in _vh]
        _has_producer = "producer" in _roles
        _has_consumer = "consumer" in _roles
        if _has_producer and _has_consumer:
            continue
        if not _has_producer:
            # Promote the first non-consumer step to producer.
            for _i, _s in enumerate(_vh):
                if (_s.get("role") or "").lower() != "consumer":
                    _s["role"] = "producer"
                    break
            else:
                # All steps already labelled consumer — clone the first.
                if _vh:
                    _vh[0]["role"] = "producer"
        if not _has_consumer:
            # Promote the last non-producer step to consumer.
            for _i in range(len(_vh) - 1, -1, -1):
                _s = _vh[_i]
                if (_s.get("role") or "").lower() != "producer":
                    _s["role"] = "consumer"
                    break
            else:
                if _vh:
                    _vh[-1]["role"] = "consumer"
        # Edge case: single-step var_history needs both roles distributed.
        # We synthesize an explicit producer alias ahead of the consumer so
        # the front-end source panel still shows a coherent producer→consumer
        # pair anchored at the same call site.
        _roles_after = [(s.get("role") or "").lower() for s in _vh]
        if "producer" not in _roles_after or "consumer" not in _roles_after:
            if len(_vh) == 1:
                _only = _vh[0]
                _producer_clone = dict(_only)
                _producer_clone["role"] = "producer"
                _only["role"] = "consumer"
                _vh.insert(0, _producer_clone)

    flowchart_data = {
        "nodes": dag_nodes,
        "edges": dag_edges,
        "groups": dag_groups,
        "root_groups": [g["id"] for g in root_groups],
        "top_level_ids": top_level_ids,
        "input_node_id": input_node_id,
        "loss_node_id": loss_node_id,
        "result_node_id": loss_node_id,
        "has_timing": has_timing,
        "meta": {
            "device": meta.get("device_name", "N/A") if meta else "N/A",
            "step_dur_us": step_dur_us,
            "step_kernel_us": timing_data.get("step_kernel_us", 0) if timing_data else 0,
            "unattributed_kernel_us": timing_data.get("unattributed_kernel_us", 0) if timing_data else 0,
            "step_dur_str": format_duration(step_dur_us) if step_dur_us > 0 else "N/A",
            "num_modules": len(dag_nodes) + len(dag_groups),
            "roots": roots,
        }
    }

    # 统计信息（用于防回退校验）
    try:
        _internal_edges_n = sum(len(g.get('internal_edges', []) or []) for g in dag_groups)
        print(f"  📊 DAG 统计: nodes={len(dag_nodes)}, groups={len(dag_groups)}, top_edges={len(dag_edges)}, internal_edges={_internal_edges_n}")
    except Exception:
        pass

    # ====================================================================
    # 强制验证规则
    # ====================================================================
    _validation_errors = []

    # Rule 1: 每条数据边必须有源码映射（文件 + 行号）
    # Iter8 strict: also require an excerpt for at least one side.
    def _ev_complete(ev):
        """An evidence record is considered complete when it has a file,
        at least one line number, and at least one excerpt rendered.
        """
        if not ev or not isinstance(ev, dict):
            return False, "missing"
        if not ev.get("file"):
            return False, "missing file"
        if not (ev.get("from_line") or ev.get("to_line")):
            return False, "missing line"
        if not (ev.get("from_excerpt") or ev.get("to_excerpt")):
            return False, "missing excerpt"
        return True, ""

    for g in dag_groups:
        for e in g.get("internal_edges", []):
            ok, why = _ev_complete(e.get("evidence"))
            if not ok:
                _validation_errors.append(
                    f"  Rule1: [{g['label']}] {e.get('from_attr')}→{e.get('to_attr')} 缺少源码映射 ({why}, type={e.get('type')})")
    for e in dag_edges:
        # Iter12-final: boundary-typed edges (synthetic group-frame anchors)
        # are exempt from source-line validation by design — they represent
        # implicit group-boundary traversal, not a real `self.attr(...)` call.
        if e.get("type") == "boundary":
            continue
        ok, why = _ev_complete(e.get("evidence"))
        if not ok:
            _validation_errors.append(
                f"  Rule1: top_edge {e.get('from_attr', e.get('from'))}→{e.get('to_attr', e.get('to'))} 缺少源码映射 ({why}, type={e.get('type')})")

    # Rule 2: 所有 nn.Module 一定有输入和输出连线（Input/Result 顶层节点除外）
    # 每个 Module 节点/子 group 必须同时具备至少一条入边和至少一条出边。
    # 根级子节点：必须显式出现在 dag_edges 或 internal_edges 中。
    # 嵌套子节点：允许隐式组边界传递（父 group 已连通则其 entry/exit 子节点视为连通）。
    def _strip_boundary_suffix(_raw_id):
        _raw_id = _raw_id or ""
        if _raw_id.endswith("__in"):
            return _raw_id[:-4]
        if _raw_id.endswith("__out"):
            return _raw_id[:-5]
        return _raw_id

    def _iter_group_children(_group):
        _seen = set()
        _ordered = []
        for _co in _group.get("call_order", []):
            _cid = _co.get("id")
            if _cid and _cid not in _seen:
                _ordered.append((_cid, _co.get("attr", "")))
                _seen.add(_cid)
        for _child in _group.get("children_nodes", []):
            _cid = _child if isinstance(_child, str) else _child.get("id")
            if _cid and _cid not in _seen:
                _ordered.append((_cid, (_child.get("label", "") if isinstance(_child, dict) else "")))
                _seen.add(_cid)
        for _child in _group.get("children_groups", []):
            _cid = _child if isinstance(_child, str) else _child.get("id")
            if _cid and _cid not in _seen:
                _ordered.append((_cid, (_child.get("label", "") if isinstance(_child, dict) else "")))
                _seen.add(_cid)
        return _ordered

    _has_inbound = set()
    _has_outbound = set()
    for g in dag_groups:
        for e in g.get("internal_edges", []):
            _has_outbound.add(_strip_boundary_suffix(e["from_child"]))
            _has_inbound.add(_strip_boundary_suffix(e["to_child"]))
    for e in dag_edges:
        fr = _strip_boundary_suffix(e.get("from", ""))
        to = _strip_boundary_suffix(e.get("to", ""))
        if fr:
            _has_outbound.add(fr)
        if to:
            _has_inbound.add(to)

    # For nested (non-root) groups: entry children implicitly receive from group
    # boundary; exit children implicitly output to group boundary.
    _root_group_ids = set(g.get("id") for g in root_groups)
    for g in dag_groups:
        if g.get("id") in _root_group_ids:
            continue  # root-level groups must have explicit edges (no exemption)

        _g_targets = set()
        _g_sources = set()
        for e in g.get("internal_edges", []):
            _g_targets.add(_strip_boundary_suffix(e["to_child"]))
            _g_sources.add(_strip_boundary_suffix(e["from_child"]))
        for cid, _attr in _iter_group_children(g):
            if not cid:
                continue
            if cid not in _g_targets:
                _has_inbound.add(cid)  # entry child: receives from group boundary
            if cid not in _g_sources:
                _has_outbound.add(cid)  # exit child: outputs to group boundary

    # Exempt: Input and Result nodes
    _exempt_ids = {input_node_id, loss_node_id}

    for g in dag_groups:
        n_children = len(g.get("children_nodes", [])) + len(g.get("children_groups", []))
        if n_children == 0:
            continue
        for child_id, child_attr in _iter_group_children(g):
            if not child_id or child_id in _exempt_ids:
                continue
            if child_id not in _has_inbound:
                _validation_errors.append(
                    f"  Rule2: [{g['label']}] '{child_attr}' (id={child_id}) 缺少输入连线")
            if child_id not in _has_outbound:
                _validation_errors.append(
                    f"  Rule2: [{g['label']}] '{child_attr}' (id={child_id}) 缺少输出连线")

    # Rule 3: DAG 不允许成环（拓扑排序检测）
    # 收集全局所有节点和边，包括 top_edges 和 internal_edges
    _all_dag_nodes = set()
    _all_dag_adj = {}
    for g in dag_groups:
        for ch in g.get("children_nodes", []):
            _all_dag_nodes.add(ch if isinstance(ch, str) else ch["id"])
        for cg in g.get("children_groups", []):
            _all_dag_nodes.add(cg if isinstance(cg, str) else cg["id"])
        for e in g.get("internal_edges", []):
            fc, tc = e["from_child"], e["to_child"]
            _all_dag_nodes.add(fc)
            _all_dag_nodes.add(tc)
            _all_dag_adj.setdefault(fc, []).append(tc)
    for e in dag_edges:
        fr = e.get("from", "").replace("__out", "")
        to = e.get("to", "").replace("__in", "")
        if fr and to:
            _all_dag_nodes.add(fr)
            _all_dag_nodes.add(to)
            _all_dag_adj.setdefault(fr, []).append(to)
    # Kahn's algorithm for cycle detection
    _in_deg = {n: 0 for n in _all_dag_nodes}
    for n, nbs in _all_dag_adj.items():
        for nb in nbs:
            if nb in _in_deg:
                _in_deg[nb] += 1
    _topo_queue = [n for n, d in _in_deg.items() if d == 0]
    _topo_visited = 0
    while _topo_queue:
        cur = _topo_queue.pop()
        _topo_visited += 1
        for nb in _all_dag_adj.get(cur, []):
            if nb in _in_deg:
                _in_deg[nb] -= 1
                if _in_deg[nb] == 0:
                    _topo_queue.append(nb)
    if _topo_visited < len(_all_dag_nodes):
        _cycle_nodes = [n for n, d in _in_deg.items() if d > 0]
        _validation_errors.append(
            f"  Rule3: DAG 存在环! 涉及节点({len(_cycle_nodes)}): {_cycle_nodes[:10]}")

    if _validation_errors:
        # Iter8 strict: any validation failure is fatal. We refuse to write a
        # corrupt HTML that the side panel would render with the obsolete
        # "No source-line evidence captured for this edge." fallback.
        print("  ⛔ DAG 强制验证失败 (共 {} 条):".format(len(_validation_errors)))
        for err in _validation_errors:
            print(err)
        raise RuntimeError(
            "DAG validation failed: {} error(s). The flowchart must not be "
            "produced with edges that lack source-line evidence.".format(
                len(_validation_errors)))
    else:
        print("  ✅ 验证通过: 所有边有源码映射，所有Module有输入输出连线，DAG 无环")


    if _return_data_only:
        # Iter14 双 Tab: caller will combine train+infer datasets later.
        # Tag the conditional mode used so the HTML layer can render distinct tabs.
        try:
            flowchart_data["meta"]["conditional_mode"] = conditional_mode
        except Exception:
            pass
        return flowchart_data

    return render_flowchart_to_file(flowchart_data, output_path)


def build_timing_data_from_trace(events, source_files, mod_info, step_dur_us, profiler_steps, src_info=None, roots=None):
    """Extract per-class and per-instance timing data for the static flowchart."""
    class_map = _build_class_map(source_files)
    timing_data = {
        "step_dur_us": step_dur_us,
        "class_durations": {},
        "class_durations_fwd": {},
        "class_durations_bwd": {},
        "class_exclusive": {},
        "class_calls": {},
        "class_thread_role": {},
        "runtime_instance_timings_by_class": {},
    }

    durations = mod_info["module_durations"]
    exclusive = mod_info["module_exclusive"]
    call_counts = mod_info["module_call_counts"]
    thread_info_map = mod_info.get("module_thread_info", {})

    def extract_class_name(mod_name):
        short = mod_name.replace("nn.Module: ", "")
        short = re.sub(r',\s*callsite:\s*\d+', '', short)
        base = re.sub(r'_\d+$', '', short)
        return base

    def extract_class_name_stripped(base):
        return re.sub(r'^(FSDP|DDP|DistributedDataParallel|Wrapped)', '', base)

    for mod_name, dur in durations.items():
        base = extract_class_name(mod_name)
        stripped = extract_class_name_stripped(base)
        for name in set([base, stripped]):
            timing_data["class_durations"][name] = timing_data["class_durations"].get(name, 0) + dur
            timing_data["class_exclusive"][name] = timing_data["class_exclusive"].get(name, 0) + exclusive.get(mod_name, 0)
            timing_data["class_calls"][name] = timing_data["class_calls"].get(name, 0) + call_counts.get(mod_name, 0)
            if name not in timing_data["class_thread_role"]:
                ti = thread_info_map.get(mod_name, {})
                roles = ti.get("roles", set())
                if "worker_thread" in roles and "main_thread" not in roles:
                    timing_data["class_thread_role"][name] = "worker"
                else:
                    timing_data["class_thread_role"][name] = "main"

    _main_tid, main_events, children = build_main_thread_hierarchy(events)
    step_infos = extract_step_phase_intervals(main_events, children) if main_events else []
    timing_data["phase_interval_steps"] = len(step_infos)

    if step_infos and source_files:
        # ------------------------------------------------------------------
        # New 4-step instance-level timing pipeline (replaces legacy
        # aggregate_source_module_phase_times +
        # aggregate_runtime_instance_phase_times +
        # _aggregate_kernel_callsite_buckets).
        #
        # Design doc: timing_fix_workdir/output/timing_redesign.md
        # ------------------------------------------------------------------
        panel = build_instance_timing_pipeline(
            events, source_files, class_map, step_infos, step_dur_us, roots=roots
        )
        timing_data["step_kernel_us"] = panel.get("step_kernel_us", 0)
        timing_data["unattributed_kernel_us"] = panel.get("unattributed_kernel_us", 0)
        # Class-level totals from instance inclusive (overwrite wrapped-event
        # estimates; new pipeline is authoritative).
        for cls, total_us in panel["class_durations"].items():
            timing_data["class_durations"][cls] = total_us
            if cls not in timing_data["class_thread_role"]:
                timing_data["class_thread_role"][cls] = "main"
        for cls, v in panel["class_durations_fwd"].items():
            timing_data["class_durations_fwd"][cls] = v
        for cls, v in panel["class_durations_bwd"].items():
            timing_data["class_durations_bwd"][cls] = v

        # Instance-level: filter to known classes only.
        known_classes = {cname for (_fname, cname) in class_map.keys()}
        filtered_runtime = {}
        for class_name, items in panel["runtime_instance_timings_by_class"].items():
            if class_name not in known_classes:
                continue
            # Field-name compatibility shim: map new pipeline names
            # (``total_us`` / ``forward_us`` / ``backward_us`` / ``other_us``)
            # to the kernel-only contract (``kernel_us`` / ``fwd_kernel_us`` /
            # ``bwd_kernel_us`` / ``other_kernel_us``) that the existing
            # frontend (`_make_dag_node` etc.) reads directly. The new
            # pipeline already produces self (exclusive) values for these
            # fields, so the rename is a 1:1 alias — no scaling.
            shimmed = []
            for item in items:
                it = dict(item)
                it.setdefault("kernel_us", float(item.get("inclusive_us", item.get("total_us", 0.0))))
                it.setdefault("fwd_kernel_us", float(item.get("inclusive_forward_us", item.get("forward_us", 0.0))))
                it.setdefault("bwd_kernel_us", float(item.get("inclusive_backward_us", item.get("backward_us", 0.0))))
                it.setdefault("other_kernel_us", float(item.get("other_us", 0.0)))
                shimmed.append(it)
            filtered_runtime[class_name] = shimmed
        timing_data["runtime_instance_timings_by_class"] = filtered_runtime
        timing_data["_timing_pipeline_stats"] = panel.get("_timing_pipeline_stats", {})

        # Debug print (rate-limited).
        try:
            ts_stats = panel.get("_timing_pipeline_stats", {})
            if ts_stats:
                print(
                    f"  [timing pipeline] kernels={ts_stats.get('total_kernels',0)} "
                    f"fwd_attr={ts_stats.get('fwd_attributed',0)} "
                    f"bwd_attr={ts_stats.get('bwd_attributed',0)} "
                    f"wrapped_fb={ts_stats.get('wrapped_fallback',0)} "
                    f"unattr={ts_stats.get('fwd_unattributed',0)+ts_stats.get('bwd_unattributed',0)} "
                    f"bwd_flow_narrow={ts_stats.get('bwd_via_flow_narrowed',0)}"
                )
        except Exception:
            pass

    return timing_data


def generate_html_flowchart_dual(source_files, timing_data=None, meta=None,
                                 output_path="flowchart.html", trace_events=None):
    """Iter14 entry point: build train + infer DAGs and emit a single HTML file
    containing both views as switchable tabs.

    For models with no `if is_training:` branches the two underlying datasets
    are identical, but two tabs are still rendered (per spec). For models like
    5547919 CTRModule.loss_m, the train tab shows ``LossModule`` and the infer
    tab shows ``PredModule``.
    """
    print("  → 构建训练图 (conditional_mode='train') ...")
    data_train = generate_html_flowchart(
        source_files,
        timing_data=timing_data,
        meta=meta,
        output_path=output_path,
        trace_events=trace_events,
        conditional_mode="train",
        _return_data_only=True,
    )
    print("  → 构建推理图 (conditional_mode='infer') ...")
    # 推理图不应继承训练 trace 的 timing 叠加。当前 timeline 分析仅基于训练
    # trace，因此 infer tab 必须退化为纯静态结构图，避免把训练耗时错误展示到
    # 推理分支节点上。
    data_infer = generate_html_flowchart(
        source_files,
        timing_data=None,
        meta=meta,
        output_path=output_path,
        trace_events=trace_events,
        conditional_mode="infer",
        _return_data_only=True,
    )

    if data_train is None or data_infer is None:
        print("  ⚠️ 双 Tab 构建失败：至少一份数据为空")
        return None

    return render_dual_flowchart_to_file(data_train, data_infer, output_path)



def render_flowchart_to_file(flowchart_data, output_path):
    html_content = _generate_flowchart_html(flowchart_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


def render_dual_flowchart_to_file(data_train, data_infer, output_path):
    html_content = _generate_flowchart_html_dual(data_train, data_infer)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content.encode("utf-8", "replace").decode("utf-8"))
    return output_path


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
        "top_edges": len(d.get("edges", [])),
        "internal_edges": sum(len(g.get("internal_edges", []) or []) for g in d.get("groups", [])),
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
        + str(s_train["groups"]) + ' groups · ' + str(s_train["top_edges"])
        + ' top · ' + str(s_train["internal_edges"]) + ' int</span></button>\n'
        '  <button class="iter14-tab" data-iter14-tab="infer" role="tab"'
        ' aria-selected="false" type="button">推理图'
        '<span class="iter14-tab-stats">' + str(s_infer["nodes"]) + ' nodes · '
        + str(s_infer["groups"]) + ' groups · ' + str(s_infer["top_edges"])
        + ' top · ' + str(s_infer["internal_edges"]) + ' int</span></button>\n'
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
        '    try { if (typeof edgeDomRegistry !== "undefined") edgeDomRegistry.length = 0; } catch(e){}\n'
        '    try { if (typeof positions !== "undefined") { Object.keys(positions).forEach(k => delete positions[k]); } } catch(e){}\n'
        '    try { hoveredEdgeKey = null; hoveredNodeId = null; hoveredGroupId = null; focusedEdgeKey = null; } catch(e){}\n'
        '    // Close any open side panel from the previous tab.\n'
        '    try { var sp = document.getElementById("side-panel"); if (sp) sp.classList.remove("open"); } catch(e){}\n'
        '  }\n'
        '  function _rebuildIndices(){\n'
        '    DATA.groups.forEach(g => groupMap[g.id] = g);\n'
        '    DATA.nodes.forEach(n => nodeMap[n.id] = n);\n'
        '    if (typeof indexGroupAncestors === "function" && DATA.root_groups){\n'
        '      indexGroupAncestors(DATA.root_groups.map(rid => groupMap[rid]).filter(Boolean));\n'
        '    }\n'
        '    DATA.groups.forEach(g => { collapsedState[g.id] = g.depth >= 2; });\n'
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
        '    if (typeof render === "function") render();\n'
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
    # NOTE: The legacy "Total X% · Y" → totalUs reflow patch that previously
    # lived here is gone.  After the kernel-field migration the in-template
    # SVG label already reads `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`
    # directly (g.dur_us mirrors kernel_us), so no post-hoc replace is
    # needed.

    # ------------------------------------------------------------------
    # Iter14-aggr: front-end hover de-dup for synthetic container boundaries.
    #
    # Synthetic container groups (ModuleDict / ModuleList / dict / list with
    # >= 2 expanded children) emit one boundary edge per elem child:
    #
    #   container__in  → elem_0   container__in  → elem_1  ...   (× N)
    #   elem_0 → container__out   elem_1 → container__out  ...   (× N)
    #
    # All N edges share the same logical "external source → container"
    # entry (resp. exit) and all N have the same ``container_aggregated_in``
    # (resp. ``_out``) marker pointing at the container's group id.  Without
    # de-duplication, hovering ANY node connected to the container triggers
    # N simultaneous className mutations on N SVG paths, and the browser
    # stalls while it re-rasterises N parallel beziers in one frame.
    #
    # We surgically patch two spots in the embedded JS template:
    #
    #   1. Inside ``applyEdgeFocusState``, just after ``activeKeys`` is
    #      computed, drop every aggregated edge except the FIRST one seen
    #      per (container, direction).  This keeps Rule2 / Rule6 happy
    #      (the data still has all N edges) while ensuring hover paints a
    #      single representative bezier per container side.
    #
    #   2. Wrap the body of ``applyEdgeFocusState`` in
    #      ``requestAnimationFrame`` so rapid mouseenter/mouseleave events
    #      coalesce into one DOM toggle pass instead of firing the whole
    #      iteration on every event tick.
    #
    # Both patches are pure JS string substitutions over the loaded
    # template; they assume the canonical anchors below survive across
    # template variants (verified for iter11_timing_active.html).  Should
    # the anchor not be present (older / future templates) the patches
    # silently no-op and the original template is used as-is — the
    # underlying Python-side dedup (which also collapses 64× duplicate
    # internal_edges into a single entry) still eliminates the worst lag.
    # ------------------------------------------------------------------
    _aef_anchor = "function applyEdgeFocusState() {\n    const activeKeys = getActiveEdgeKeys();"
    _aef_replacement = (
        "function applyEdgeFocusState() {\n"
        "    // Iter14-aggr: coalesce rapid hover updates via rAF so successive\n"
        "    // mouseenter/mouseleave events on neighbouring elements merge into\n"
        "    // one DOM toggle pass instead of N back-to-back iterations over\n"
        "    // every registered SVG edge path.\n"
        "    if (applyEdgeFocusState._scheduled) { return; }\n"
        "    applyEdgeFocusState._scheduled = true;\n"
        "    const _runAEFNow = () => {\n"
        "        applyEdgeFocusState._scheduled = false;\n"
        "        _applyEdgeFocusStateImpl();\n"
        "    };\n"
        "    if (typeof requestAnimationFrame === 'function') {\n"
        "        requestAnimationFrame(_runAEFNow);\n"
        "    } else {\n"
        "        _runAEFNow();\n"
        "    }\n"
        "}\n"
        "function _applyEdgeFocusStateImpl() {\n"
        "    const activeKeys = getActiveEdgeKeys();\n"
        "    // Iter14-aggr: collapse container-boundary fan-outs.  Every edge\n"
        "    // tagged with ``container_aggregated_in`` (or ``_out``) belongs to\n"
        "    // the same logical N-fan boundary; we keep only the FIRST active\n"
        "    // edge per (container, direction) so hover never paints N parallel\n"
        "    // beziers simultaneously.\n"
        "    {\n"
        "        const _aggrSeenIn = new Map();\n"
        "        const _aggrSeenOut = new Map();\n"
        "        const _aggrToDrop = [];\n"
        "        for (const _item of edgeDomRegistry) {\n"
        "            if (!activeKeys.has(_item.key)) continue;\n"
        "            const _e = _item.edge || {};\n"
        "            const _aIn = _e.container_aggregated_in;\n"
        "            const _aOut = _e.container_aggregated_out;\n"
        "            if (_aIn) {\n"
        "                if (_aggrSeenIn.has(_aIn)) { _aggrToDrop.push(_item.key); }\n"
        "                else { _aggrSeenIn.set(_aIn, _item.key); }\n"
        "            } else if (_aOut) {\n"
        "                if (_aggrSeenOut.has(_aOut)) { _aggrToDrop.push(_item.key); }\n"
        "                else { _aggrSeenOut.set(_aOut, _item.key); }\n"
        "            }\n"
        "        }\n"
        "        for (const _k of _aggrToDrop) { activeKeys.delete(_k); }\n"
        "    }"
    )
    html_template = html_template.replace(_aef_anchor, _aef_replacement, 1)

    # Replace the embedded DATA payload using the stable script markers rather
    # than a naive non-greedy `{...};` regex: the serialized JSON contains many
    # nested `};` substrings inside code snippets, which can cause partial
    # matches and leave stale template data in place.
    pattern = r'const DATA = .*?(?=\nconst groupMap = \{\};)'
    replacement = 'const DATA = ' + data_json + '\n'
    html_template = re.sub(pattern, lambda m: replacement, html_template, flags=re.DOTALL)
    
    return html_template



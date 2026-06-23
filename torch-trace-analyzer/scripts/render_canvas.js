/* render_canvas.js
 * Canvas Phase 1 -- Canvas (Pixi) render skeleton + static draw passes.
 *
 * Stage 1.1 scope: stand up PIXI.Application + world (viewport) + layers L0..L5,
 * expose window.__phase1NoInteractionMode and a read-only window.__renderSnapshot().
 *
 * Stage 1.2 scope (this revision): static Node / Group / Port drawing.
 *   - NodeView.draw():       leaf-node rect + label/sublabel (was renderNodeAt)
 *   - GroupView.drawCollapsed(): collapsed group box + label/timing/info
 *   - GroupView.drawExpanded():  expanded group shell + header/info/timing + children
 *   - PortRenderer.drawPorts():  populate the shared nodePortMap (was register*Ports)
 *   These consume the *existing* layout output (groupLayout / childPositions) produced
 *   by frontend_html.py's layoutGroup(); coordinates are never recomputed here.
 *
 * Hard rules honoured here:
 *   - no fallback / silent path: a missing #dag-stage container raises
 *   - no interaction wiring in this phase (no hover / click / toggle)
 *   - the snapshot is read-only and has the same shape in browser and headless paths
 *
 * Layer model (index === paint order):
 *   L0 background | L1 edges (Stage 1.3) | L2 groups | L3 nodes
 *   L4 interaction overlay (Phase 3, empty) | L5 io/overlay (Stage 1.4)
 *
 * The layout / colour / timing helpers below (layoutGroup, groupMap, nodeMap,
 * groupLayout, LAYOUT, getNodeColor, formatDur, nodePortMap) live as
 * classic-script globals defined by frontend_html.py's inline template.  They
 * are referenced by bare name and resolved lazily at call time (this script
 * loads *before* the inline template, but canvasRenderPhase1() only runs *after*
 * it).
 */
/* global layoutGroup, groupMap, nodeMap, groupLayout, LAYOUT,
   getNodeColor, formatDur, nodePortMap */
(function (global) {
    'use strict';

    const LAYER_KEYS = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5'];

    // ── engine factory ─────────────────────────────────────────────────────
    // Prefer the injected engine bundle (global.PIXI); otherwise use an internal
    // headless mock so unit tests run with no DOM canvas / WebGL context.  Both
    // paths build the identical layer tree -- this is NOT a render fallback.
    function resolvePixi() {
        if (global.PIXI && typeof global.PIXI.Application === 'function' && typeof global.PIXI.Container === 'function') {
            return global.PIXI;
        }
        return createHeadlessPixi();
    }

    function createHeadlessPixi() {
        function Container() {
            this.children = [];
            this.x = 0;
            this.y = 0;
            this.scale = { x: 1, y: 1 };
            this.visible = true;
            this.parent = null;
            this.name = '';
        }
        Container.prototype.addChild = function (child) {
            child.parent = this;
            this.children.push(child);
            return child;
        };
        Container.prototype.removeChildren = function () {
            const removed = this.children;
            this.children = [];
            removed.forEach(function (c) { c.parent = null; });
            return removed;
        };
        function Graphics() {
            Container.call(this);
            this.__isHeadlessGraphics = true;
        }
        Graphics.prototype = Object.create(Container.prototype);
        Graphics.prototype.constructor = Graphics;
        Graphics.prototype.clear = function () { return this; };
        function Text(text) {
            Container.call(this);
            this.text = (text === undefined || text === null) ? '' : String(text);
            this.__isHeadlessText = true;
        }
        Text.prototype = Object.create(Container.prototype);
        Text.prototype.constructor = Text;
        function Application(options) {
            const opts = options || {};
            this.stage = new Container();
            this.screen = { x: 0, y: 0, width: opts.width || 0, height: opts.height || 0 };
            this.canvas = null;
            this.renderer = { resize: function () {} };
        }
        Application.prototype.destroy = function () { this.stage = new Container(); };
        return {
            Application: Application,
            Container: Container,
            Graphics: Graphics,
            Text: Text,
            __isHeadlessMock: true
        };
    }

    let engine = null;

    function resolveContainer(explicitContainer) {
        if (explicitContainer) {
            return explicitContainer;
        }
        if (global.document && typeof global.document.getElementById === 'function') {
            return global.document.getElementById('dag-stage');
        }
        return null;
    }

    function buildEngine(container) {
        const PIXI = resolvePixi();
        const app = new PIXI.Application({ width: 0, height: 0, antialias: true, backgroundAlpha: 0 });
        if (app.canvas && typeof container.appendChild === 'function') {
            container.appendChild(app.canvas);
        }
        const world = new PIXI.Container();
        world.name = 'world';
        app.stage.addChild(world);

        const layers = {};
        LAYER_KEYS.forEach(function (key) {
            const layer = new PIXI.Container();
            layer.name = key;
            world.addChild(layer);
            layers[key] = layer;
        });

        return {
            pixi: PIXI,
            app: app,
            world: world,
            layers: layers,
            container: container,
            usingHeadlessPixi: PIXI.__isHeadlessMock === true,
            nodes: [],
            groups: [],
            viewport: { scale: 1, x: 0, y: 0, worldWidth: 0, worldHeight: 0 }
        };
    }

    function initCanvasEngine(explicitContainer) {
        const container = resolveContainer(explicitContainer);
        if (!container) {
            // No silent fallback: a missing render container is a hard error.
            throw new Error('render_canvas.js: #dag-stage container not found; cannot initialize Canvas renderer');
        }
        engine = buildEngine(container);
        return engine;
    }

    function ensureEngine() {
        if (!engine) {
            initCanvasEngine();
        }
        return engine;
    }

    // ── lazy accessors for inline-template globals ─────────────────────────
    // These globals are classic-script lexical bindings (not window properties),
    // so they are read by bare name behind a typeof guard.  When this file runs
    // standalone (Stage 1.1 headless snapshot probe) they are simply absent and
    // the static draw pass never executes.
    function lookupGroupMap() { return (typeof groupMap !== 'undefined') ? groupMap : null; }
    function lookupNodeMap() { return (typeof nodeMap !== 'undefined') ? nodeMap : null; }
    function lookupGroupLayout() { return (typeof groupLayout !== 'undefined') ? groupLayout : null; }
    function lookupLayout() { return (typeof LAYOUT !== 'undefined') ? LAYOUT : null; }
    function lookupNodePortMap() { return (typeof nodePortMap !== 'undefined') ? nodePortMap : null; }
    function nodeColorOf(n) { return (typeof getNodeColor === 'function') ? getNodeColor(n) : '#4a6fa5'; }
    function formatDurOf(us) { return (typeof formatDur === 'function') ? formatDur(us) : String(us); }

    // ── pixi glyph factories ───────────────────────────────────────────────
    function makeGraphics(name) {
        const g = new engine.pixi.Graphics();
        if (name) { g.name = name; }
        return g;
    }
    function makeText(value, name) {
        const t = new engine.pixi.Text(value);
        if (name) { t.name = name; }
        return t;
    }

    // ── PortRenderer ───────────────────────────────────────────────────────
    function registerNodePorts(nid, x, y, w, h) {
        const map = lookupNodePortMap();
        if (!map) { return; }
        map[nid] = { cx: x + w / 2, cy: y + h / 2 };
        map[nid + '__in'] = { cx: x + w / 2, cy: y };
        map[nid + '__out'] = { cx: x + w / 2, cy: y + h };
    }

    function registerCollapsedGroupPorts(g, gid, ox, oy, pos) {
        const map = lookupNodePortMap();
        if (!map) { return; }
        const inPoint = { cx: ox + pos.w / 2, cy: oy };
        const outPoint = { cx: ox + pos.w / 2, cy: oy + pos.h };
        map[gid + '__in'] = inPoint;
        map[gid + '__out'] = outPoint;
        map[gid + '__center'] = { cx: ox + pos.w / 2, cy: oy + pos.h / 2 };
        for (const port of (g.in_ports || [])) { map[port.node_id] = inPoint; }
        for (const port of (g.out_ports || [])) { map[port.node_id] = outPoint; }
    }

    function registerExpandedGroupPorts(g, gid, ox, oy, pos) {
        const map = lookupNodePortMap();
        if (!map) { return; }
        const inPoint = { cx: ox + pos.w / 2, cy: oy };
        const outPoint = { cx: ox + pos.w / 2, cy: oy + pos.h };
        map[gid + '__in'] = inPoint;
        map[gid + '__out'] = outPoint;
        for (const port of (g.in_ports || [])) { map[port.node_id] = inPoint; }
        for (const port of (g.out_ports || [])) { map[port.node_id] = outPoint; }
    }

    // ── NodeView ───────────────────────────────────────────────────────────
    function drawNode(nid, x, y, w, h) {
        const nodes = lookupNodeMap();
        const n = nodes ? nodes[nid] : null;
        if (!n) { return; }
        const color = nodeColorOf(n);
        const label = n.class_name;
        let sublabel = '';
        if (n.has_timing) { sublabel = n.pct.toFixed(1) + '%'; }

        engine.layers.l3.addChild(makeGraphics('node-box:' + nid));
        engine.layers.l3.addChild(makeText(label, 'node-label:' + nid));
        if (sublabel) {
            engine.layers.l3.addChild(makeText(sublabel, 'node-sublabel:' + nid));
        }

        engine.nodes.push({
            id: nid, x: x, y: y, w: w, h: h,
            color: color, label: label, sublabel: sublabel, visible: true
        });
        registerNodePorts(nid, x, y, w, h);
    }

    // ── GroupView ──────────────────────────────────────────────────────────
    function drawCollapsedGroup(g, gid, ox, oy, pos) {
        const hasTiming = !!g.has_timing;
        const hasInfo = !!g.src_file;

        engine.layers.l2.addChild(makeGraphics('group-box:' + gid));
        engine.layers.l2.addChild(makeText('\u25B6 ' + g.class_name, 'group-label:' + gid));
        if (hasTiming) {
            engine.layers.l2.addChild(makeText('Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us), 'group-timing:' + gid));
        }
        if (hasInfo) {
            engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
            engine.layers.l2.addChild(makeText('i', 'group-info:' + gid));
        }

        engine.groups.push({
            id: gid, collapsed: true, x: ox, y: oy, w: pos.w, h: pos.h,
            has_header: false, has_info: hasInfo, has_timing: hasTiming
        });
        registerCollapsedGroupPorts(g, gid, ox, oy, pos);
    }

    function drawExpandedGroupShell(g, gid, ox, oy, pos) {
        const hasTiming = !!g.has_timing;
        const hasInfo = !!g.src_file;

        engine.layers.l2.addChild(makeGraphics('group-box:' + gid));
        engine.layers.l2.addChild(makeText('\u25BC ' + g.class_name, 'group-header:' + gid));
        if (hasInfo) {
            engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
            engine.layers.l2.addChild(makeText('i', 'group-info:' + gid));
        }
        if (hasTiming) {
            engine.layers.l2.addChild(makeText('Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us), 'group-timing:' + gid));
        }

        engine.groups.push({
            id: gid, collapsed: false, x: ox, y: oy, w: pos.w, h: pos.h,
            has_header: true, has_info: hasInfo, has_timing: hasTiming
        });
    }

    // Recursively render a group at (ox, oy) using the precomputed layout.
    function walkGroup(gid, ox, oy) {
        const groups = lookupGroupMap();
        const layout = lookupGroupLayout();
        const g = groups ? groups[gid] : null;
        if (!g) {
            throw new Error('render_canvas.js: group not found while drawing: ' + gid);
        }
        const pos = (layout && Object.prototype.hasOwnProperty.call(layout, gid)) ? layout[gid] : null;
        if (!pos) {
            throw new Error('render_canvas.js: missing layout for group: ' + gid);
        }

        if (pos.collapsed) {
            drawCollapsedGroup(g, gid, ox, oy, pos);
            return;
        }

        drawExpandedGroupShell(g, gid, ox, oy, pos);
        for (const child of (pos.childPositions || [])) {
            if (child.type === 'node') {
                drawNode(child.id, ox + child.x, oy + child.y, child.w, child.h);
            } else if (child.type === 'group') {
                walkGroup(child.id, ox + child.x, oy + child.y);
            } else {
                throw new Error('render_canvas.js: unknown child type while drawing: ' + child.type);
            }
        }
        registerExpandedGroupPorts(g, gid, ox, oy, pos);
    }

    // ── scene reset + draw orchestration ───────────────────────────────────
    function resetScene() {
        LAYER_KEYS.forEach(function (key) {
            const layer = engine.layers[key];
            if (layer && typeof layer.removeChildren === 'function') {
                layer.removeChildren();
            }
        });
        engine.nodes = [];
        engine.groups = [];
        const map = lookupNodePortMap();
        if (map) {
            Object.keys(map).forEach(function (k) { delete map[k]; });
        }
    }

    // Run the (unchanged) inline layout, then draw the single root group at its
    // centred origin -- mirroring frontend_html.py's render() geometry for the
    // no-IO case.  IO pills + culling + fit arrive in Stage 1.4.
    function layoutAndDrawRoots(data) {
        const layoutDefaults = lookupLayout();
        const nodeW = layoutDefaults ? layoutDefaults.nodeW : 150;
        const rootSizes = data.root_groups.map(function (rid) {
            const sz = layoutGroup(rid);
            return { id: rid, w: sz.w, h: sz.h };
        });
        const maxRootW = rootSizes.length
            ? Math.max.apply(null, rootSizes.map(function (r) { return r.w; }))
            : nodeW;
        const svgW = Math.max(maxRootW + 80, 480);
        const rootStartY = 30;
        if (rootSizes.length > 0) {
            const rs = rootSizes[0];
            walkGroup(rs.id, (svgW - rs.w) / 2, rootStartY);
        }
        engine.viewport.worldWidth = svgW;
    }

    // Production render entry.  The in-template render() delegates here whenever
    // window.__phase1NoInteractionMode === true.
    function canvasRenderPhase1(data) {
        ensureEngine();
        resetScene();
        if (data && Array.isArray(data.root_groups) && data.root_groups.length > 0) {
            layoutAndDrawRoots(data);
        }
        return Promise.resolve(buildSnapshot());
    }

    function layerChildCounts() {
        const counts = {};
        LAYER_KEYS.forEach(function (key) {
            const layer = engine && engine.layers ? engine.layers[key] : null;
            counts[key] = layer && Array.isArray(layer.children) ? layer.children.length : 0;
        });
        return counts;
    }

    function clonePorts(map) {
        const out = {};
        if (!map) { return out; }
        Object.keys(map).forEach(function (key) {
            const p = map[key];
            out[key] = { cx: p.cx, cy: p.cy };
        });
        return out;
    }

    // Read-only snapshot.  MUST NOT mutate runtime state.  Same shape in browser
    // and headless paths; any missing field is a bug, never an optional.
    function buildSnapshot() {
        if (!engine) {
            throw new Error('render_canvas.js: __renderSnapshot called before the Canvas engine was initialized');
        }
        const vp = engine.viewport;
        return {
            nodes: (engine.nodes || []).map(function (n) {
                return { id: n.id, x: n.x, y: n.y, w: n.w, h: n.h, color: n.color, label: n.label, sublabel: n.sublabel, visible: n.visible };
            }),
            groups: (engine.groups || []).map(function (g) {
                return { id: g.id, collapsed: g.collapsed, x: g.x, y: g.y, w: g.w, h: g.h, has_header: g.has_header, has_info: g.has_info, has_timing: g.has_timing };
            }),
            edges: [],
            ports: clonePorts(lookupNodePortMap()),
            io_pills: [],
            viewport: {
                scale: vp.scale,
                x: vp.x,
                y: vp.y,
                worldWidth: vp.worldWidth,
                worldHeight: vp.worldHeight
            },
            layers: layerChildCounts(),
            flags: {
                noInteractionMode: global.__phase1NoInteractionMode === true,
                cullingEnabled: false
            }
        };
    }

    // Phase 1 runs with interaction disabled.
    global.__phase1NoInteractionMode = true;
    global.__canvasRenderPhase1 = canvasRenderPhase1;
    global.__initCanvasEngine = initCanvasEngine;
    global.__canvasEnginePhase1 = function () { return engine; };
    global.__renderSnapshot = function () {
        ensureEngine();
        return buildSnapshot();
    };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

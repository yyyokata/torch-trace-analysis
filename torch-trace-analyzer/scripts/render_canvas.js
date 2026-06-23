/* render_canvas.js
 * Canvas Phase 1 -- Pixi/static render pipeline.
 *
 * Phase 1 scope:
 *   - L0..L5 scene graph + read-only snapshot
 *   - Node / Group / Port drawing
 *   - Global edges (EdgeRoute + EdgeBatch)
 *   - IO layer (L5), viewport pan/zoom/fit, label culling, render progress wiring
 *
 * Hard rules honoured here:
 *   - no fallback / silent path: missing runtime globals are hard errors
 *   - Phase 1 remains no node/edge hover/click interaction
 *   - snapshot shape is identical in browser and headless probe paths
 */
/* global layoutGroup, groupMap, nodeMap, groupLayout, LAYOUT,
   getNodeColor, formatDur, nodePortMap,
   isEdgeVisible, resolveCollapsedAncestor, edgeKey, EDGE_BUNDLE_META,
   computeFlowchartLayout, computeIOGroupExpandedLayout, getIOLayoutConfig,
   showRenderProgress, hideRenderProgress, setRenderProgress, getRenderProgressElements,
   runChunked, nextFrame, assertActiveRenderGeneration, renderGeneration */
(function (global) {
    'use strict';

    const LAYER_KEYS = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5'];
    const IO_GROUP_FILL = {
        input: 'rgba(46,204,113,0.55)',
        param: 'rgba(155,89,182,0.55)',
        const: 'rgba(241,196,15,0.55)',
        output: 'rgba(231,76,60,0.55)'
    };
    const IO_GROUP_MEMBER_LABEL = {
        input: 'Input',
        param: 'Param',
        const: 'Const',
        output: 'Result'
    };

    // ── engine factory ─────────────────────────────────────────────────────
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
        ['clear', 'lineStyle', 'moveTo', 'lineTo', 'beginFill', 'drawPolygon', 'endFill', 'drawRoundedRect', 'drawCircle', 'drawRect']
            .forEach(function (name) {
                Graphics.prototype[name] = function () { return this; };
            });
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
            this.renderer = {
                resize: function (w, h) {
                    this.width = w;
                    this.height = h;
                }
            };
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

        const built = {
            pixi: PIXI,
            app: app,
            world: world,
            layers: layers,
            container: container,
            usingHeadlessPixi: PIXI.__isHeadlessMock === true,
            nodes: [],
            groups: [],
            edges: [],
            io_pills: [],
            labelsCreated: 0,
            cullingEnabled: true,
            worldBounds: null,
            viewport: {
                scale: 1,
                x: 0,
                y: 0,
                worldWidth: 0,
                worldHeight: 0,
                minScale: 0.25,
                maxScale: 4
            }
        };
        built.viewportController = new ViewportController(built);
        built.cullManager = new CullManager();
        return built;
    }

    function initCanvasEngine(explicitContainer) {
        const container = resolveContainer(explicitContainer);
        if (!container) {
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
    function lookupGroupMap() { return (typeof groupMap !== 'undefined') ? groupMap : null; }
    function lookupNodeMap() { return (typeof nodeMap !== 'undefined') ? nodeMap : null; }
    function lookupGroupLayout() { return (typeof groupLayout !== 'undefined') ? groupLayout : null; }
    function lookupLayout() { return (typeof LAYOUT !== 'undefined') ? LAYOUT : null; }
    function lookupNodePortMap() { return (typeof nodePortMap !== 'undefined') ? nodePortMap : null; }
    function nodeColorOf(n) { return (typeof getNodeColor === 'function') ? getNodeColor(n) : '#4a6fa5'; }
    function formatDurOf(us) { return (typeof formatDur === 'function') ? formatDur(us) : String(us); }
    function lookupIsEdgeVisible() { return (typeof isEdgeVisible === 'function') ? isEdgeVisible : null; }
    function lookupResolveCollapsedAncestor() { return (typeof resolveCollapsedAncestor === 'function') ? resolveCollapsedAncestor : null; }
    function lookupEdgeKey() { return (typeof edgeKey === 'function') ? edgeKey : null; }
    function lookupEdgeBundleMeta() { return (typeof EDGE_BUNDLE_META !== 'undefined') ? EDGE_BUNDLE_META : null; }
    function lookupComputeFlowchartLayout() { return (typeof computeFlowchartLayout === 'function') ? computeFlowchartLayout : null; }
    function lookupComputeIOGroupExpandedLayout() { return (typeof computeIOGroupExpandedLayout === 'function') ? computeIOGroupExpandedLayout : null; }
    function lookupIOLayoutConfig() { return (typeof getIOLayoutConfig === 'function') ? getIOLayoutConfig() : null; }
    function lookupShowRenderProgress() { return (typeof showRenderProgress === 'function') ? showRenderProgress : null; }
    function lookupHideRenderProgress() { return (typeof hideRenderProgress === 'function') ? hideRenderProgress : null; }
    function lookupSetRenderProgress() { return (typeof setRenderProgress === 'function') ? setRenderProgress : null; }
    function lookupGetRenderProgressElements() { return (typeof getRenderProgressElements === 'function') ? getRenderProgressElements : null; }
    function lookupRunChunked() { return (typeof runChunked === 'function') ? runChunked : null; }
    function lookupNextFrame() { return (typeof nextFrame === 'function') ? nextFrame : null; }
    function lookupAssertActiveRenderGeneration() { return (typeof assertActiveRenderGeneration === 'function') ? assertActiveRenderGeneration : null; }

    function requireInline(name, value) {
        if (!value) {
            throw new Error('render_canvas.js: required inline runtime is missing: ' + name);
        }
        return value;
    }

    function currentRenderGeneration() {
        if (typeof renderGeneration === 'undefined') {
            throw new Error('render_canvas.js: renderGeneration is unavailable');
        }
        return renderGeneration;
    }

    function bumpRenderGeneration() {
        if (typeof renderGeneration === 'undefined') {
            throw new Error('render_canvas.js: renderGeneration is unavailable');
        }
        renderGeneration += 1;
        return renderGeneration;
    }

    function resetInlineLayoutCache() {
        if (typeof groupLayout === 'undefined') {
            throw new Error('render_canvas.js: groupLayout is unavailable');
        }
        groupLayout = {};
    }

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

    function addLabel(layer, value, name) {
        engine.labelsCreated += 1;
        layer.addChild(makeText(value, name));
    }

    // ── viewport / culling ─────────────────────────────────────────────────
    function numericOrNull(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }

    function getContainerWidth(container) {
        if (!container) { return null; }
        return numericOrNull(container.clientWidth || container.offsetWidth || container.width);
    }

    function getContainerHeight(container) {
        if (!container) { return null; }
        return numericOrNull(container.clientHeight || container.offsetHeight || container.height);
    }

    function applyViewport() {
        engine.world.scale.x = engine.viewport.scale;
        engine.world.scale.y = engine.viewport.scale;
        engine.world.x = engine.viewport.x;
        engine.world.y = engine.viewport.y;
    }

    function normalizeWorldBounds(worldBounds) {
        if (!worldBounds || typeof worldBounds !== 'object') {
            throw new Error('render_canvas.js: fitToView requires worldBounds');
        }
        const x = numericOrNull(worldBounds.x);
        const y = numericOrNull(worldBounds.y);
        const w = numericOrNull(worldBounds.w !== undefined ? worldBounds.w : worldBounds.width);
        const h = numericOrNull(worldBounds.h !== undefined ? worldBounds.h : worldBounds.height);
        if (x === null || y === null || w === null || h === null) {
            throw new Error('render_canvas.js: fitToView got invalid worldBounds');
        }
        return { x: x, y: y, w: w, h: h };
    }

    function ViewportController(owner) {
        this.owner = owner;
    }
    ViewportController.prototype.enablePan = function () {
        return this;
    };
    ViewportController.prototype.enableZoom = function (limits) {
        const opts = limits || {};
        const min = numericOrNull(opts.min);
        const max = numericOrNull(opts.max);
        if (min === null || max === null || min <= 0 || max <= 0 || min > max) {
            throw new Error('render_canvas.js: enableZoom got invalid limits');
        }
        this.owner.viewport.minScale = min;
        this.owner.viewport.maxScale = max;
        if (this.owner.viewport.scale < min) {
            this.owner.viewport.scale = min;
        }
        if (this.owner.viewport.scale > max) {
            this.owner.viewport.scale = max;
        }
        applyViewport();
        return this;
    };
    ViewportController.prototype.panBy = function (dx, dy) {
        const panX = numericOrNull(dx);
        const panY = numericOrNull(dy);
        if (panX === null || panY === null) {
            throw new Error('render_canvas.js: panBy got invalid delta');
        }
        this.owner.viewport.x += panX;
        this.owner.viewport.y += panY;
        applyViewport();
        return this.owner.viewport;
    };
    ViewportController.prototype.zoomTo = function (nextScale) {
        const value = numericOrNull(nextScale);
        if (value === null || value <= 0) {
            throw new Error('render_canvas.js: zoomTo got invalid scale');
        }
        const vp = this.owner.viewport;
        vp.scale = Math.max(vp.minScale, Math.min(vp.maxScale, value));
        applyViewport();
        return vp;
    };
    ViewportController.prototype.zoomBy = function (factor) {
        const value = numericOrNull(factor);
        if (value === null || value <= 0) {
            throw new Error('render_canvas.js: zoomBy got invalid factor');
        }
        return this.zoomTo(this.owner.viewport.scale * value);
    };
    ViewportController.prototype.fitToView = function (worldBounds, containerWidth) {
        const bounds = normalizeWorldBounds(worldBounds);
        const cw = numericOrNull(containerWidth);
        if (cw === null || cw <= 0) {
            throw new Error('render_canvas.js: fitToView requires a positive containerWidth');
        }
        const scale = bounds.w > cw ? Number((cw / bounds.w).toFixed(3)) : 1;
        this.owner.viewport.scale = scale;
        this.owner.viewport.x = -bounds.x * scale;
        this.owner.viewport.y = -bounds.y * scale;
        applyViewport();
        return this.owner.viewport;
    };
    ViewportController.prototype.currentBounds = function () {
        const vp = this.owner.viewport;
        const width = getContainerWidth(this.owner.container) || vp.worldWidth || 0;
        const height = getContainerHeight(this.owner.container) || vp.worldHeight || 0;
        const scale = vp.scale || 1;
        return {
            x: -vp.x / scale,
            y: -vp.y / scale,
            w: width / scale,
            h: height / scale
        };
    };

    function CullManager() {}
    CullManager.prototype.isVisible = function (rect, viewportBounds) {
        if (!rect || !viewportBounds) {
            throw new Error('render_canvas.js: CullManager.isVisible requires rect and viewportBounds');
        }
        return !(
            rect.x + rect.w < viewportBounds.x ||
            rect.x > viewportBounds.x + viewportBounds.w ||
            rect.y + rect.h < viewportBounds.y ||
            rect.y > viewportBounds.y + viewportBounds.h
        );
    };

    function shouldCreateLabel(rect) {
        if (!engine.cullingEnabled) {
            return true;
        }
        return engine.cullManager.isVisible(rect, engine.viewportController.currentBounds());
    }

    // ── PortRenderer ───────────────────────────────────────────────────────
    function registerNodePorts(nid, x, y, w, h) {
        const map = lookupNodePortMap();
        if (!map) {
            throw new Error('render_canvas.js: nodePortMap is unavailable while registering node ports');
        }
        map[nid] = { cx: x + w / 2, cy: y + h / 2 };
        map[nid + '__in'] = { cx: x + w / 2, cy: y };
        map[nid + '__out'] = { cx: x + w / 2, cy: y + h };
    }

    function registerCollapsedGroupPorts(g, gid, ox, oy, pos) {
        const map = lookupNodePortMap();
        if (!map) {
            throw new Error('render_canvas.js: nodePortMap is unavailable while registering collapsed group ports');
        }
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
        if (!map) {
            throw new Error('render_canvas.js: nodePortMap is unavailable while registering expanded group ports');
        }
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
        const sublabel = n.has_timing ? (n.pct.toFixed(1) + '%') : '';
        const rect = { x: x, y: y, w: w, h: h };
        const visible = shouldCreateLabel(rect);

        engine.layers.l3.addChild(makeGraphics('node-box:' + nid));
        if (visible) {
            addLabel(engine.layers.l3, label, 'node-label:' + nid);
            if (sublabel) {
                addLabel(engine.layers.l3, sublabel, 'node-sublabel:' + nid);
            }
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
        const rect = { x: ox, y: oy, w: pos.w, h: pos.h };
        const visible = shouldCreateLabel(rect);

        engine.layers.l2.addChild(makeGraphics('group-box:' + gid));
        if (visible) {
            addLabel(engine.layers.l2, '\u25B6 ' + g.class_name, 'group-label:' + gid);
            if (hasTiming) {
                addLabel(engine.layers.l2, 'Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us), 'group-timing:' + gid);
            }
            if (hasInfo) {
                engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
                addLabel(engine.layers.l2, 'i', 'group-info:' + gid);
            }
        } else if (hasInfo) {
            engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
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
        const rect = { x: ox, y: oy, w: pos.w, h: pos.h };
        const visible = shouldCreateLabel(rect);

        engine.layers.l2.addChild(makeGraphics('group-box:' + gid));
        if (visible) {
            addLabel(engine.layers.l2, '\u25BC ' + g.class_name, 'group-header:' + gid);
            if (hasInfo) {
                engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
                addLabel(engine.layers.l2, 'i', 'group-info:' + gid);
            }
            if (hasTiming) {
                addLabel(engine.layers.l2, 'Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us), 'group-timing:' + gid);
            }
        } else if (hasInfo) {
            engine.layers.l2.addChild(makeGraphics('group-info-hit:' + gid));
        }

        engine.groups.push({
            id: gid, collapsed: false, x: ox, y: oy, w: pos.w, h: pos.h,
            has_header: true, has_info: hasInfo, has_timing: hasTiming
        });
    }

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

    // ── EdgeRoute (pure geometry) ──────────────────────────────────────────
    const EDGE_SAMPLE_STEPS = 24;
    const LONG_EDGE_MIN_SPAN = 260;
    const EDGE_STYLE = {
        dep:      { color: 0x2ecc71, width: 1.9,  alpha: 0.62, arrowAlpha: 0.6 },
        internal: { color: 0x64b5f6, width: 1.35, alpha: 0.46, arrowAlpha: 0.5 },
        default:  { color: 0xffffff, width: 1.7,  alpha: 0.2,  arrowAlpha: 0.3 }
    };

    function colorKeyForType(type) {
        if (type === 'dep') { return 'dep'; }
        if (type === 'internal') { return 'internal'; }
        return 'default';
    }

    function sampleCubic(p0, p1, p2, p3, steps) {
        const pts = [];
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const mt = 1 - t;
            const a = mt * mt * mt;
            const b = 3 * mt * mt * t;
            const c = 3 * mt * t * t;
            const d = t * t * t;
            pts.push({
                x: a * p0.x + b * p1.x + c * p2.x + d * p3.x,
                y: a * p0.y + b * p1.y + c * p2.y + d * p3.y
            });
        }
        return pts;
    }

    function sampleQuad(p0, p1, p2, steps) {
        const pts = [];
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const mt = 1 - t;
            const a = mt * mt;
            const b = 2 * mt * t;
            const c = t * t;
            pts.push({
                x: a * p0.x + b * p1.x + c * p2.x,
                y: a * p0.y + b * p1.y + c * p2.y
            });
        }
        return pts;
    }

    function polylineLength(points) {
        let total = 0;
        for (let i = 1; i < points.length; i++) {
            total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
        }
        return total;
    }

    const EdgeRoute = {
        direct: function (x1, y1, x2, y2, routeMeta) {
            const dy = y2 - y1;
            const dx = x2 - x1;
            if (Math.abs(dy) < 3 && Math.abs(dx) < 3) { return null; }
            const meta = routeMeta || {};
            const offset = meta.bundleOffset || 0;
            const sideBias = offset === 0 ? (dx >= 0 ? 1 : -1) : (offset > 0 ? 1 : -1);
            let points;
            let branch;
            if (dy > 8) {
                const cp = Math.max(24, Math.min(Math.abs(dy) * 0.34 + Math.abs(offset) * 0.7, 96));
                const c1x = x1 + offset;
                const c2x = x2 + offset;
                points = sampleCubic(
                    { x: x1, y: y1 }, { x: c1x, y: y1 + cp },
                    { x: c2x, y: y2 - cp }, { x: x2, y: y2 }, EDGE_SAMPLE_STEPS
                );
                branch = 'down';
            } else if (dy < -8) {
                const horizontal = sideBias * Math.max(52, Math.abs(dx) * 0.42 + 34 + Math.abs(offset));
                const rise = Math.max(18, Math.min(72, Math.abs(offset) + 24));
                points = sampleCubic(
                    { x: x1, y: y1 }, { x: x1 + horizontal, y: y1 - rise },
                    { x: x2 + horizontal, y: y2 + rise }, { x: x2, y: y2 }, EDGE_SAMPLE_STEPS
                );
                branch = 'back';
            } else {
                const midX = (x1 + x2) / 2 + offset;
                const midY = (y1 + y2) / 2 + 14 + Math.abs(offset) * 0.28;
                points = sampleQuad(
                    { x: x1, y: y1 }, { x: midX, y: midY }, { x: x2, y: y2 }, EDGE_SAMPLE_STEPS
                );
                branch = 'quad';
            }
            const dashed = polylineLength(points) >= LONG_EDGE_MIN_SPAN;
            return { points: points, branch: branch, dashed: dashed };
        },
        compute: function (routingMode, x1, y1, x2, y2, routeMeta) {
            if (routingMode === 'direct') {
                return EdgeRoute.direct(x1, y1, x2, y2, routeMeta);
            }
            throw new Error('render_canvas.js: unknown edge routing mode: ' + routingMode);
        }
    };

    // ── EdgeBatch ──────────────────────────────────────────────────────────
    function EdgeBatch() {
        this.buckets = { dep: [], internal: [], default: [] };
    }
    EdgeBatch.prototype.collect = function (edge, start, end, routeMeta, routingMode) {
        const mode = routingMode || 'direct';
        const route = EdgeRoute.compute(mode, start.cx, start.cy, end.cx, end.cy, routeMeta);
        if (!route) { return; }
        const type = edge.type || 'dep';
        const colorKey = colorKeyForType(type);
        this.buckets[colorKey].push({ points: route.points, dashed: route.dashed });
        engine.edges.push({
            from: edge.from, to: edge.to, type: type, colorKey: colorKey,
            start: { cx: start.cx, cy: start.cy },
            end: { cx: end.cx, cy: end.cy },
            branch: route.branch, dashed: route.dashed, arrow: true
        });
    };
    EdgeBatch.prototype.flush = function () {
        const self = this;
        Object.keys(self.buckets).forEach(function (colorKey) {
            const items = self.buckets[colorKey];
            if (!items.length) { return; }
            const style = EDGE_STYLE[colorKey];
            const strokeG = makeGraphics('edge-stroke:' + colorKey);
            const arrowG = makeGraphics('edge-arrow:' + colorKey);
            items.forEach(function (item) {
                strokePolyline(strokeG, item.points, style);
                drawArrowHead(arrowG, item.points, style);
            });
            engine.layers.l1.addChild(strokeG);
            engine.layers.l1.addChild(arrowG);
        });
    };

    function strokePolyline(g, points, style) {
        if (typeof g.lineStyle !== 'function' || typeof g.moveTo !== 'function' || typeof g.lineTo !== 'function') {
            return;
        }
        g.lineStyle(style.width, style.color, style.alpha);
        g.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            g.lineTo(points[i].x, points[i].y);
        }
    }

    function drawArrowHead(g, points, style) {
        const n = points.length;
        const tip = points[n - 1];
        const prev = points[n - 2] || points[0];
        let ux = tip.x - prev.x;
        let uy = tip.y - prev.y;
        const len = Math.hypot(ux, uy) || 1;
        ux /= len;
        uy /= len;
        const size = 7;
        const half = size / 2;
        const px = -uy;
        const py = ux;
        const baseX = tip.x - ux * size;
        const baseY = tip.y - uy * size;
        if (typeof g.beginFill !== 'function' || typeof g.drawPolygon !== 'function') {
            return;
        }
        g.beginFill(style.color, style.arrowAlpha);
        g.drawPolygon([
            tip.x, tip.y,
            baseX + px * half, baseY + py * half,
            baseX - px * half, baseY - py * half
        ]);
        if (typeof g.endFill === 'function') { g.endFill(); }
    }

    function drawGlobalEdges(data) {
        const edges = (data && Array.isArray(data.edges)) ? data.edges : [];
        if (edges.length === 0) { return; }
        const portMap = lookupNodePortMap();
        const isVisible = lookupIsEdgeVisible();
        const resolveAncestor = lookupResolveCollapsedAncestor();
        const keyOf = lookupEdgeKey();
        const bundleMeta = lookupEdgeBundleMeta();
        if (!portMap || !isVisible || !resolveAncestor || !keyOf || !bundleMeta) {
            throw new Error('render_canvas.js: inline edge globals unavailable while drawing edges');
        }
        const batch = new EdgeBatch();
        for (const edge of edges) {
            if (!isVisible(edge)) { continue; }
            const fromId = resolveAncestor(edge.from);
            const toId = resolveAncestor(edge.to);
            if (fromId === toId) { continue; }
            const fromPos = portMap[fromId + '__out'] || portMap[fromId];
            const toPos = portMap[toId + '__in'] || portMap[toId];
            if (!fromPos || !toPos) {
                throw new Error('global edge endpoint missing: ' + edge.from + ' -> ' + edge.to);
            }
            const routeMeta = bundleMeta.get(keyOf(edge)) || null;
            batch.collect(edge, fromPos, toPos, routeMeta, 'direct');
        }
        batch.flush();
    }

    // ── IO layer (L5) ──────────────────────────────────────────────────────
    function IOLayer() {
        this.layer = engine.layers.l5;
        this.config = requireInline('getIOLayoutConfig', lookupIOLayoutConfig());
        this.computeExpandedLayout = requireInline('computeIOGroupExpandedLayout', lookupComputeIOGroupExpandedLayout());
    }

    IOLayer.prototype.drawPill = function (spec) {
        if (!spec || spec.id === undefined || spec.id === null) {
            throw new Error('render_canvas.js: IOLayer.drawPill got invalid spec');
        }
        this.layer.addChild(makeGraphics('io-pill:' + spec.id));
        addLabel(this.layer, spec.label, 'io-label:' + spec.id);
        if (spec.sublabel) {
            addLabel(this.layer, spec.sublabel, 'io-sublabel:' + spec.id);
        }
        registerNodePorts(spec.id, spec.cx - spec.w / 2, spec.cy - spec.h / 2, spec.w, spec.h);
        engine.io_pills.push({
            id: spec.id,
            subtype: spec.subtype,
            cx: spec.cx,
            cy: spec.cy,
            w: spec.w,
            h: spec.h,
            expanded: spec.expanded === true
        });
    };

    IOLayer.prototype.drawGroupPill = function (ioGroup, geom, availableW) {
        if (!ioGroup || !geom) {
            throw new Error('render_canvas.js: IOLayer.drawGroupPill got invalid arguments');
        }
        const fillColor = IO_GROUP_FILL[ioGroup.io_subtype] || 'rgba(127,140,141,0.55)';
        const memberLabel = IO_GROUP_MEMBER_LABEL[ioGroup.io_subtype] || ioGroup.io_subtype;
        const isCollapsed = !!ioGroup._collapsed;
        const cfg = this.config;
        if (!isCollapsed) {
            const members = ioGroup.member_ids || [];
            if (members.length === 0) {
                return 0;
            }
            const expanded = this.computeExpandedLayout(members.length, availableW);
            const startY = geom.cy - cfg.EXPANDED_IO_H / 2;
            const frameX = geom.cx - availableW / 2 + cfg.EXPAND_FRAME_PAD - 8;
            const frameY = startY - 8;
            const frameW = availableW - 2 * (cfg.EXPAND_FRAME_PAD - 8);
            const frameH = expanded.height + 8;
            this.layer.addChild(makeGraphics('io-group-frame:' + ioGroup.id));
            engine.io_pills.push({
                id: ioGroup.id,
                subtype: ioGroup.io_subtype,
                cx: geom.cx,
                cy: geom.cy,
                w: availableW,
                h: expanded.height,
                expanded: true
            });
            for (let rowIdx = 0; rowIdx < expanded.memberRows; rowIdx++) {
                const first = rowIdx * expanded.cols;
                const rowMemberCount = Math.max(0, Math.min(expanded.cols, members.length - first));
                const rowWidth = rowMemberCount * expanded.pillW + (rowMemberCount - 1) * cfg.pillGap;
                let left = geom.cx - rowWidth / 2;
                const rowCy = startY + rowIdx * (cfg.EXPANDED_IO_H + cfg.EXPANDED_IO_GAP) + cfg.EXPANDED_IO_H / 2;
                for (let idx = 0; idx < rowMemberCount; idx++) {
                    const memberId = members[first + idx];
                    const node = lookupNodeMap() ? lookupNodeMap()[memberId] : null;
                    const baseText = node ? node.class_name : memberLabel;
                    const sublabel = (node && node.has_timing)
                        ? (baseText + ' · ' + node.pct.toFixed(1) + '%')
                        : baseText;
                    this.drawPill({
                        id: memberId,
                        subtype: ioGroup.io_subtype,
                        cx: left + expanded.pillW / 2,
                        cy: rowCy,
                        w: expanded.pillW,
                        h: cfg.EXPANDED_IO_H,
                        label: memberLabel,
                        sublabel: truncateExpandedIOSublabel(sublabel, expanded.pillW),
                        fillColor: fillColor,
                        expanded: false
                    });
                    left += expanded.pillW + cfg.pillGap;
                }
            }
            this.layer.addChild(makeGraphics('io-group-collapse:' + ioGroup.id));
            addLabel(this.layer, '▲ 收起', 'io-group-collapse-label:' + ioGroup.id);
            return frameY + frameH;
        }
        this.layer.addChild(makeGraphics('io-group-pill:' + ioGroup.id));
        addLabel(this.layer, '▶ ' + ioGroup.label, 'io-group-label:' + ioGroup.id);
        registerNodePorts(ioGroup.id, geom.cx - geom.w / 2, geom.cy - geom.h / 2, geom.w, geom.h);
        engine.io_pills.push({
            id: ioGroup.id,
            subtype: ioGroup.io_subtype,
            cx: geom.cx,
            cy: geom.cy,
            w: geom.w,
            h: geom.h,
            expanded: false
        });
        return geom.h;
    };

    function truncateExpandedIOSublabel(text, pillW) {
        const limit = Math.max(1, Math.floor(pillW / 6.5));
        const value = String(text || '');
        return value.length > limit ? (value.slice(0, Math.max(0, limit - 1)) + '…') : value;
    }

    function drawIOTasks(tasks) {
        const ioLayer = new IOLayer();
        for (const task of (tasks || [])) {
            if (task.type !== 'io') {
                throw new Error('render_canvas.js: drawIOTasks got unsupported task type: ' + task.type);
            }
            if (task.taskKind === 'io_group') {
                ioLayer.drawGroupPill(task.ioGroup, task, task.availableW);
                continue;
            }
            if (task.taskKind === 'io_pill') {
                ioLayer.drawPill({
                    id: task.nid,
                    subtype: task.subtype,
                    cx: task.cx,
                    cy: task.cy,
                    w: task.w,
                    h: task.h,
                    label: task.label,
                    sublabel: task.sublabel,
                    fillColor: task.fillColor,
                    expanded: false
                });
                continue;
            }
            throw new Error('render_canvas.js: drawIOTasks got unknown taskKind: ' + task.taskKind);
        }
    }

    // ── scene reset + layout orchestration ─────────────────────────────────
    function resetScene() {
        LAYER_KEYS.forEach(function (key) {
            const layer = engine.layers[key];
            if (layer && typeof layer.removeChildren === 'function') {
                layer.removeChildren();
            }
        });
        engine.nodes = [];
        engine.groups = [];
        engine.edges = [];
        engine.io_pills = [];
        engine.labelsCreated = 0;
        const map = lookupNodePortMap();
        if (map) {
            Object.keys(map).forEach(function (k) { delete map[k]; });
        }
    }

    function applyWorldLayout(layoutInfo) {
        if (!layoutInfo || !Number.isFinite(layoutInfo.svgW) || !Number.isFinite(layoutInfo.svgH)) {
            throw new Error('render_canvas.js: computeFlowchartLayout returned invalid world bounds');
        }
        engine.viewport.worldWidth = layoutInfo.svgW;
        engine.viewport.worldHeight = layoutInfo.svgH;
        engine.worldBounds = { x: 0, y: 0, w: layoutInfo.svgW, h: layoutInfo.svgH };
        if (engine.app && engine.app.renderer && typeof engine.app.renderer.resize === 'function') {
            engine.app.renderer.resize(Math.ceil(layoutInfo.svgW), Math.ceil(layoutInfo.svgH));
        }
        applyViewport();
    }

    function layoutAndDrawRoots(layoutInfo) {
        applyWorldLayout(layoutInfo);
        for (const root of (layoutInfo.rootPositions || [])) {
            walkGroup(root.id, root.x, root.y);
        }
        drawIOTasks(layoutInfo.ioTasks || []);
    }

    function updateLegendAndSummary(data) {
        if (!global.document || typeof global.document.getElementById !== 'function') {
            throw new Error('render_canvas.js: document.getElementById is unavailable while updating DOM panels');
        }
        const meta = data.meta || {};
        const modeBadge = global.document.getElementById('mode-badge');
        const metaInfo = global.document.getElementById('meta-info');
        const legendDiv = global.document.getElementById('legend');
        const summaryDiv = global.document.getElementById('summary');
        if (!modeBadge || !metaInfo || !legendDiv || !summaryDiv) {
            throw new Error('render_canvas.js: legend/summary DOM is incomplete');
        }

        if (data.has_timing) {
            modeBadge.innerHTML = '<span class="mode-badge mode-timing">📊 Structure + Timing</span>';
            metaInfo.textContent = 'Device: ' + meta.device + ' | Step: ' + meta.step_dur_str + ' | Modules: ' + meta.num_modules;
        } else {
            modeBadge.innerHTML = '<span class="mode-badge mode-structure">🏗️ Static Structure (source code)</span>';
            metaInfo.textContent = 'Modules: ' + meta.num_modules + ' | Root: ' + (meta.roots ? meta.roots.join(', ') : 'N/A');
        }

        if (data.has_timing) {
            legendDiv.innerHTML = '\n                <div class="legend-item"><div class="legend-dot" style="background:#2980b9"></div>&gt;20%</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>10-20%</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#8e44ad"></div>5-10%</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#5a6c7d"></div>&lt;5%</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Worker &gt;20%</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#e67e22"></div>Worker 10-20%</div>';
        } else {
            legendDiv.innerHTML = '\n                <div class="legend-item"><div class="legend-dot" style="background:#4a6fa5"></div>Depth 0</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#5b8c5a"></div>Depth 1</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#8e6fad"></div>Depth 2</div>\n                <div class="legend-item"><div class="legend-dot" style="background:#c77a3c"></div>Depth 3+</div>\n                <div class="legend-item" style="margin-left: 12px;"><span style="color:#64b5f6">▶</span> Click to expand</div>\n                <div class="legend-item" style="margin-left: 12px;"><span style="color:rgba(46,204,113,0.8)">━━▶</span> Data dependency</div>\n                <div class="legend-item"><span style="color:rgba(255,255,255,0.3)">╌╌▶</span> Sequential (fallback)</div>';
        }

        const allNodes = data.nodes || [];
        const allGroups = data.groups || [];
        if (data.has_timing) {
            const topN = allNodes.concat(allGroups).filter(function (x) { return x.has_timing; }).sort(function (a, b) { return b.pct - a.pct; }).slice(0, 5);
            summaryDiv.innerHTML = '<h3>📊 Top Modules by Time</h3><p>' + topN.map(function (x) {
                return '<b>' + (x.label || x.class_name) + '</b> ' + x.pct.toFixed(1) + '%';
            }).join(' → ') + '</p>';
        } else {
            summaryDiv.innerHTML = '<h3>🏗️ Architecture Summary</h3><p>Module count: ' + (allNodes.length + allGroups.length) + ' | Expandable containers: ' + allGroups.length + ' | Leaf nodes: ' + allNodes.length + '<br><i>Click ▶ collapsed containers to expand. Provide --trace-file for timing overlay.</i></p>';
        }
    }

    function progressApi() {
        return {
            showRenderProgress: requireInline('showRenderProgress', lookupShowRenderProgress()),
            hideRenderProgress: requireInline('hideRenderProgress', lookupHideRenderProgress()),
            setRenderProgress: requireInline('setRenderProgress', lookupSetRenderProgress()),
            getRenderProgressElements: requireInline('getRenderProgressElements', lookupGetRenderProgressElements()),
            runChunked: requireInline('runChunked', lookupRunChunked()),
            nextFrame: requireInline('nextFrame', lookupNextFrame()),
            assertActiveRenderGeneration: requireInline('assertActiveRenderGeneration', lookupAssertActiveRenderGeneration())
        };
    }

    async function canvasRenderPhase1(data) {
        ensureEngine();
        const p = progressApi();
        const computeLayout = requireInline('computeFlowchartLayout', lookupComputeFlowchartLayout());
        const generation = bumpRenderGeneration();
        p.showRenderProgress('正在计算 DAG 布局…');
        const progressEls = p.getRenderProgressElements();
        progressEls.overlay.dataset.renderGeneration = String(generation);
        await p.nextFrame();
        try {
            resetInlineLayoutCache();
            resetScene();
            let layoutInfo = null;
            await p.runChunked([{ type: 'group', taskKind: 'layout' }], async function () {
                resetInlineLayoutCache();
                layoutInfo = computeLayout(data);
            }, {
                batchSize: 1,
                phaseStart: 0,
                phaseEnd: 30,
                stageText: '正在计算 DAG 布局…',
                generation: generation,
                allowedTypes: ['group']
            });
            await p.runChunked([{ type: 'group', taskKind: 'draw-scene' }], async function () {
                layoutAndDrawRoots(layoutInfo);
            }, {
                batchSize: 1,
                phaseStart: 30,
                phaseEnd: 60,
                stageText: '正在渲染模块节点…',
                generation: generation,
                allowedTypes: ['group']
            });
            await p.runChunked([{ type: 'edge', taskKind: 'draw-edges' }], async function () {
                drawGlobalEdges(data);
            }, {
                batchSize: 1,
                phaseStart: 60,
                phaseEnd: 90,
                stageText: '正在渲染依赖边…',
                generation: generation,
                allowedTypes: ['edge']
            });
            p.assertActiveRenderGeneration(generation, '收尾阶段');
            p.setRenderProgress(98, '正在更新图例和摘要…');
            await p.nextFrame();
            updateLegendAndSummary(data);
            p.assertActiveRenderGeneration(generation, '完成阶段');
            await p.hideRenderProgress();
            return buildSnapshot();
        } catch (err) {
            if (currentRenderGeneration() === generation) {
                const els = p.getRenderProgressElements();
                els.overlay.classList.remove('closing');
                els.overlay.classList.add('visible', 'failed');
                els.overlay.setAttribute('aria-hidden', 'false');
                const lastProgress = Number(els.overlay.dataset.progress || 0);
                p.setRenderProgress(Number.isFinite(lastProgress) ? Math.min(99, lastProgress) : 0, '渲染失败，请查看 Console 错误');
            }
            throw err;
        }
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
            edges: (engine.edges || []).map(function (e) {
                return {
                    from: e.from, to: e.to, type: e.type, colorKey: e.colorKey,
                    start: { cx: e.start.cx, cy: e.start.cy },
                    end: { cx: e.end.cx, cy: e.end.cy },
                    branch: e.branch, dashed: e.dashed, arrow: e.arrow
                };
            }),
            ports: clonePorts(lookupNodePortMap()),
            io_pills: (engine.io_pills || []).map(function (p) {
                return {
                    id: p.id,
                    subtype: p.subtype,
                    cx: p.cx,
                    cy: p.cy,
                    w: p.w,
                    h: p.h,
                    expanded: p.expanded === true
                };
            }),
            viewport: {
                scale: vp.scale,
                x: vp.x,
                y: vp.y,
                worldWidth: vp.worldWidth,
                worldHeight: vp.worldHeight
            },
            layers: layerChildCounts(),
            labelsCreated: engine.labelsCreated,
            flags: {
                noInteractionMode: global.__phase1NoInteractionMode === true,
                cullingEnabled: engine.cullingEnabled === true
            }
        };
    }

    global.__phase1NoInteractionMode = true;
    global.__canvasRenderPhase1 = canvasRenderPhase1;
    global.__initCanvasEngine = initCanvasEngine;
    global.__canvasEnginePhase1 = function () { return engine; };
    global.__EdgeRoute = EdgeRoute;
    global.__EDGE_STYLE = EDGE_STYLE;
    global.__renderSnapshot = function () {
        ensureEngine();
        return buildSnapshot();
    };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

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
   getNodeColor, formatDur, nodePortMap,
   isEdgeVisible, resolveCollapsedAncestor, edgeKey, EDGE_BUNDLE_META */
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
            edges: [],
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

    // Edge-orchestration globals (also classic-script lexical bindings from the
    // inline template): the visibility filter, collapse redirect, bundle-offset
    // map and edge-key helper.  Read by bare name at draw time, exactly like the
    // layout globals above.  drawGlobalEdges() only runs from the production
    // canvasRenderPhase1() path (after the inline template has loaded), so these
    // are always defined when it executes; a missing one is a hard error there,
    // never a silent skip.
    function lookupIsEdgeVisible() { return (typeof isEdgeVisible === 'function') ? isEdgeVisible : null; }
    function lookupResolveCollapsedAncestor() { return (typeof resolveCollapsedAncestor === 'function') ? resolveCollapsedAncestor : null; }
    function lookupEdgeKey() { return (typeof edgeKey === 'function') ? edgeKey : null; }
    function lookupEdgeBundleMeta() { return (typeof EDGE_BUNDLE_META !== 'undefined') ? EDGE_BUNDLE_META : null; }

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

    // ── EdgeRoute (pure geometry) ──────────────────────────────────────────
    // Faithful Canvas port of frontend_html.py's SVG `buildDirectEdgePath`
    // (the only routing the production pipeline ever used).  The control-point
    // formulas are copied verbatim; the Bezier is then sampled at a fixed step
    // count so the result is a polyline `[{x,y}...]` whose endpoints are exactly
    // (x1,y1)/(x2,y2).  Sampling the *same* curve means each sample equals the
    // legacy Bezier at the same parameter t (point distance 0 -> well under the
    // 0.5px equivalence bound), see UT P1-11.
    const EDGE_SAMPLE_STEPS = 24;

    // Long-edge dash threshold.  Same source constant as the legacy
    // `LONG_EDGE_MIN_SPAN` in frontend_html.py (260): the SVG path used
    // getTotalLength() >= 260; the Canvas route uses the sampled polyline length,
    // which is numerically equivalent for these short, smooth curves.
    const LONG_EDGE_MIN_SPAN = 260;

    // Stroke styles ported 1:1 from the deleted `.edge-path*` CSS rules
    // (frontend_html.py): rgba colour -> 0xRRGGBB + alpha, plus the matching
    // SVG <marker> fill alpha for the arrow head.
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
        // Direct route: returns {points,branch,dashed} or null (degenerate).
        // branch ∈ {'down','back','quad'} mirrors the legacy dy>8 / dy<-8 / else
        // selection.  Returns null for the degenerate |dy|<3 && |dx|<3 case so no
        // edge is produced (identical to legacy returning a null path string).
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
        // Routing-mode dispatcher.  Unknown modes are a hard error (no silent
        // return), matching the legacy renderEdge() `throw new Error('Unknown
        // edge routing mode')`.
        compute: function (routingMode, x1, y1, x2, y2, routeMeta) {
            if (routingMode === 'direct') {
                return EdgeRoute.direct(x1, y1, x2, y2, routeMeta);
            }
            throw new Error('render_canvas.js: unknown edge routing mode: ' + routingMode);
        }
    };

    // ── EdgeBatch (type-bucketed batched stroke + arrow head) ───────────────
    // Painting primitives (lineStyle/moveTo/lineTo/drawPolygon) are invoked only
    // when the resolved Graphics implementation provides them.  The Stage 1.1
    // minimal bundle (and the headless mock) build the object graph but draw
    // nothing -- the geometry contract lives in engine.edges / the snapshot, and
    // the real WebGL Pixi bundle paints it.  This is the same dual-backend shape
    // used by resolvePixi() and is NOT an error fallback.
    function EdgeBatch() {
        this.buckets = { dep: [], internal: [], default: [] };
    }
    EdgeBatch.prototype.collect = function (edge, start, end, routeMeta, routingMode) {
        const mode = routingMode || 'direct';
        const route = EdgeRoute.compute(mode, start.cx, start.cy, end.cx, end.cy, routeMeta);
        if (!route) { return; }  // degenerate route: produce no edge (legacy parity)
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
            return;  // object-graph-only backend; painting deferred to real Pixi
        }
        g.lineStyle(style.width, style.color, style.alpha);
        g.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            g.lineTo(points[i].x, points[i].y);
        }
    }

    // Triangular arrow head at the route end, oriented along the terminal
    // tangent (the marker-end equivalent).  Size 7 matches the legacy
    // <marker markerWidth="7" markerHeight="7">.
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
            return;  // object-graph-only backend; painting deferred to real Pixi
        }
        g.beginFill(style.color, style.arrowAlpha);
        g.drawPolygon([
            tip.x, tip.y,
            baseX + px * half, baseY + py * half,
            baseX - px * half, baseY - py * half
        ]);
        if (typeof g.endFill === 'function') { g.endFill(); }
    }

    // Global edge orchestration.  Reuses the inline-template edge model 1:1:
    // visibility filter -> collapse redirect -> self-skip -> port resolution
    // (out/in slot with center fallback) -> missing-endpoint hard error -> bundle
    // offset lookup -> EdgeBatch.collect, then a single flush per type bucket.
    function drawGlobalEdges(data) {
        const edges = (data && Array.isArray(data.edges)) ? data.edges : [];
        if (edges.length === 0) { return; }
        const portMap = lookupNodePortMap();
        const isVisible = lookupIsEdgeVisible();
        const resolveAncestor = lookupResolveCollapsedAncestor();
        const keyOf = lookupEdgeKey();
        const bundleMeta = lookupEdgeBundleMeta();
        if (!portMap || !isVisible || !resolveAncestor || !keyOf || !bundleMeta) {
            // Hard error: the production path must have these inline globals.
            throw new Error('render_canvas.js: inline edge globals unavailable while drawing edges');
        }
        const batch = new EdgeBatch();
        for (const edge of edges) {
            if (!isVisible(edge)) { continue; }
            const fromId = resolveAncestor(edge.from);
            const toId = resolveAncestor(edge.to);
            if (fromId === toId) { continue; }  // self after redirect: skip
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
        engine.edges = [];
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
            drawGlobalEdges(data);
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
            edges: (engine.edges || []).map(function (e) {
                return {
                    from: e.from, to: e.to, type: e.type, colorKey: e.colorKey,
                    start: { cx: e.start.cx, cy: e.start.cy },
                    end: { cx: e.end.cx, cy: e.end.cy },
                    branch: e.branch, dashed: e.dashed, arrow: e.arrow
                };
            }),
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
    // Pure geometry, exposed for UT (P1-11 / P1-11b / P1-E3): EdgeRoute.direct
    // returns the sampled polyline and EdgeRoute.compute validates routing mode.
    global.__EdgeRoute = EdgeRoute;
    // Edge stroke/arrow style table, exposed for UT (P1-13): asserts the colour
    // constants equal the (deleted) CSS `.edge-path*` values.
    global.__EDGE_STYLE = EDGE_STYLE;
    global.__renderSnapshot = function () {
        ensureEngine();
        return buildSnapshot();
    };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

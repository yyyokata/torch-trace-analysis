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

    // Stage 1.5: Text styles for the v8 Text builder.  Plain style objects are
    // accepted directly by both real PixiJS v8 Text and the headless mock.
    const TEXT_STYLE = {
        nodeTitle:   { fontFamily: 'Menlo, Consolas, monospace', fontSize: 12, fontWeight: '600', fill: 0xffffff },
        nodeSub:     { fontFamily: 'Menlo, Consolas, monospace', fontSize: 10, fill: 0xcfe3ff },
        groupHeader: { fontFamily: 'Menlo, Consolas, monospace', fontSize: 13, fontWeight: '700', fill: 0xffffff },
        groupTiming: { fontFamily: 'Menlo, Consolas, monospace', fontSize: 10, fill: 0xffe08a },
        info:        { fontFamily: 'Menlo, Consolas, monospace', fontSize: 11, fontWeight: '700', fill: 0xffffff },
        ioTitle:     { fontFamily: 'Menlo, Consolas, monospace', fontSize: 11, fontWeight: '600', fill: 0xffffff },
        ioSub:       { fontFamily: 'Menlo, Consolas, monospace', fontSize: 9, fill: 0xeafff0 }
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
        ['clear', 'roundRect', 'rect', 'circle', 'ellipse', 'poly', 'fill', 'stroke',
         'setStrokeStyle', 'setFillStyle', 'moveTo', 'lineTo', 'closePath', 'beginPath']
            .forEach(function (name) {
                Graphics.prototype[name] = function () { return this; };
            });
        function Text(arg) {
            Container.call(this);
            const opts = (arg && typeof arg === 'object') ? arg : { text: arg };
            this.text = (opts.text === undefined || opts.text === null) ? '' : String(opts.text);
            this.style = opts.style || {};
            this.resolution = opts.resolution || 1;
            this.anchor = {
                x: 0,
                y: 0,
                set: function (ax, ay) {
                    this.x = ax;
                    this.y = (ay === undefined ? ax : ay);
                }
            };
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
        // PixiJS v8 init() is async; the headless mock mirrors the contract so the
        // renderer's `ensureStageMounted()` await works identically in node + browser.
        Application.prototype.init = function (options) {
            const opts = options || {};
            this.screen = { x: 0, y: 0, width: opts.width || 0, height: opts.height || 0 };
            return Promise.resolve(this);
        };
        Application.prototype.render = function () { return this; };
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
        // PixiJS v8 splits construction (sync) from `app.init()` (async, sets up
        // the WebGL renderer + canvas).  We only build the detached scene graph
        // here so the synchronous `__renderSnapshot()` skeleton probe keeps working
        // headless; `ensureStageMounted()` performs the async init lazily on the
        // first real render and appends the canvas to the container.
        const app = new PIXI.Application();
        const world = new PIXI.Container();
        world.name = 'world';
        if (app.stage && typeof app.stage.addChild === 'function') {
            app.stage.addChild(world);
        }

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
            initialized: false,
            initPromise: null,
            hasRenderedOnce: false,
            lastKnownContainerW: null,
            lastKnownContainerH: null,
            nodes: [],
            groups: [],
            edges: [],
            io_pills: [],
            labelsCreated: 0,
            labels: [],
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

    // Stage 1.5: perform the async PixiJS v8 `app.init()` lazily on the first real
    // render.  Idempotent via `initPromise`; the headless mock resolves instantly.
    async function ensureStageMounted() {
        const eng = ensureEngine();
        if (eng.initialized) { return eng; }
        if (!eng.initPromise) {
            eng.initPromise = (async function () {
                if (typeof eng.app.init === 'function') {
                    await eng.app.init({
                        backgroundAlpha: 0,
                        antialias: true,
                        autoDensity: true,
                        resolution: (global.devicePixelRatio || 1),
                        preference: 'webgl',
                        width: Math.max(1, getContainerWidth(eng.container) || 1280),
                        height: Math.max(1, getContainerHeight(eng.container) || 720)
                    });
                }
                if (eng.app.canvas && typeof eng.container.appendChild === 'function') {
                    eng.container.appendChild(eng.app.canvas);
                    // PixiJS autoDensity writes canvas.style.width = physicalPx + 'px' as inline
                    // style, which overrides CSS class rules (e.g. max-width: 100%).  Force
                    // width: 100% here so the canvas never inflates the container.
                    if (eng.app.canvas) {
                        eng.app.canvas.style.width = '100%';
                        eng.app.canvas.style.height = '';
                    }
                }
                // `init()` may (re)create app.stage; (re-)attach the world graph.
                if (eng.app.stage && typeof eng.app.stage.addChild === 'function') {
                    eng.app.stage.addChild(eng.world);
                }
                eng.initialized = true;
            })();
        }
        await eng.initPromise;
        return eng;
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

    function makeText(value, name, style) {
        const t = new engine.pixi.Text({
            text: (value === undefined || value === null) ? '' : String(value),
            style: style || TEXT_STYLE.nodeSub,
            resolution: (global.devicePixelRatio || 1)
        });
        if (name) { t.name = name; }
        return t;
    }

    function addLabel(layer, value, name, x, y, style, anchor) {
        const ax = (anchor && anchor.ax !== undefined) ? anchor.ax : 0;
        const ay = (anchor && anchor.ay !== undefined) ? anchor.ay : 0;
        const px = numericOrNull(x);
        const py = numericOrNull(y);
        if (px === null || py === null) {
            throw new Error('render_canvas.js: addLabel requires numeric x/y for ' + name);
        }
        const t = makeText(value, name, style);
        if (t.anchor && typeof t.anchor.set === 'function') {
            t.anchor.set(ax, ay);
        } else {
            t.anchor = { x: ax, y: ay };
        }
        t.x = px;
        t.y = py;
        layer.addChild(t);
        engine.labelsCreated += 1;
        engine.labels.push({ name: name, x: px, y: py });
        return t;
    }

    // Stage 1.5: shared rounded-box painter on the v8 Graphics builder.  `fill` /
    // `stroke` accept a CSS color string or a numeric color, matching v8.
    function fillStrokeBox(g, x, y, w, h, opts) {
        const o = opts || {};
        const r = (o.radius !== undefined) ? o.radius : 8;
        g.roundRect(x, y, w, h, r);
        if (o.fill !== undefined && o.fill !== null) {
            g.fill({ color: o.fill, alpha: (o.fillAlpha !== undefined ? o.fillAlpha : 1) });
        }
        if (o.stroke !== undefined && o.stroke !== null) {
            g.stroke({ color: o.stroke, width: (o.strokeWidth || 1), alpha: (o.strokeAlpha !== undefined ? o.strokeAlpha : 1) });
        }
        return g;
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

    function resolveContainerSize(context) {
        // Width is the canvas-usable width.  Two failure modes must be guarded:
        //   1. The #dag-stage reserves a vertical-scrollbar gutter (scrollbar-gutter:
        //      stable), so its clientWidth is the real drawable width MINUS the
        //      scrollbar (~15px on Windows classic scrollbars).  Laying out against
        //      the wider parent (#dag-container, no scrollbar) over-sizes the world
        //      and clips its right edge under the scrollbar.
        //   2. PixiJS autoDensity transiently writes canvas.style.width = physicalPx,
        //      which can INFLATE #dag-stage.clientWidth before reflow settles.
        // Taking the min of the stage and its parent satisfies both: normally the
        // stage (scrollbar-excluded) is the smaller, correct value; if the stage is
        // momentarily inflated, the parent's stable layout width wins instead.
        const stageW = getContainerWidth(engine.container);
        const parentW = (engine.container && engine.container.parentElement)
            ? getContainerWidth(engine.container.parentElement)
            : null;
        let cw = null;
        if (stageW !== null && stageW > 0 && parentW !== null && parentW > 0) {
            cw = Math.min(stageW, parentW);
        } else if (parentW !== null && parentW > 0) {
            cw = parentW;
        } else if (stageW !== null && stageW > 0) {
            cw = stageW;
        }
        let ch = getContainerHeight(engine.container);
        if (cw !== null && cw > 0 && ch !== null && ch > 0) {
            engine.lastKnownContainerW = cw;
            engine.lastKnownContainerH = ch;
            return { w: cw, h: ch };
        }
        if (engine.lastKnownContainerW !== null && engine.lastKnownContainerW > 0 &&
                engine.lastKnownContainerH !== null && engine.lastKnownContainerH > 0) {
            return { w: engine.lastKnownContainerW, h: engine.lastKnownContainerH };
        }
        throw new Error('render_canvas.js: ' + context + ' requires positive container dimensions');
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
    ViewportController.prototype.fitToView = function (worldBounds, containerWidth, containerHeight, options) {
        const bounds = normalizeWorldBounds(worldBounds);
        const cw = numericOrNull(containerWidth);
        const ch = numericOrNull(containerHeight);
        if (cw === null || cw <= 0) {
            throw new Error('render_canvas.js: fitToView requires a positive containerWidth');
        }
        if (ch === null || ch <= 0) {
            throw new Error('render_canvas.js: fitToView requires a positive containerHeight');
        }
        const opts = options || {};
        const padding = (numericOrNull(opts.padding) !== null && opts.padding >= 0) ? Number(opts.padding) : 40;
        const vp = this.owner.viewport;
        const availW = Math.max(1, cw - 2 * padding);
        // Width-only fit (matches the legacy SVG semantics): scale so the graph
        // width fills the available container width, then clamp into the viewport
        // zoom range.  Height is NOT constrained — when the scaled graph is taller
        // than the viewport the overflow is reached by vertical scrolling / panning.
        let scale = availW / bounds.w;
        // opts.maxScale may override vp.maxScale for auto-fit paths that must not upscale content.
        const effectiveMax = (options && options.maxScale !== undefined) ? options.maxScale : vp.maxScale;
        scale = Math.max(vp.minScale, Math.min(effectiveMax, scale));
        scale = Number(scale.toFixed(3));
        vp.scale = scale;
        // Center horizontally; top-align vertically so the graph starts at the top
        // padding and grows downward into the scrollable overflow.
        vp.x = (cw - bounds.w * scale) / 2 - bounds.x * scale;
        vp.y = padding - bounds.y * scale;
        applyViewport();
        return vp;
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
        // Truncate the title to the node box width (12px monospace ≈ 7.2px/char,
        // 8px total side padding) so long class names no longer overflow the box.
        const label = truncateLabel(n.class_name, maxCharsForWidth(w, 7.2, 8));
        const sublabel = n.has_timing ? (n.pct.toFixed(1) + '%') : '';
        const rect = { x: x, y: y, w: w, h: h };
        const visible = shouldCreateLabel(rect);

        const box = makeGraphics('node-box:' + nid);
        fillStrokeBox(box, x, y, w, h, {
            radius: 7, fill: color, fillAlpha: 0.95,
            stroke: 0xffffff, strokeAlpha: 0.14, strokeWidth: 1
        });
        engine.layers.l3.addChild(box);
        if (visible) {
            const cx = x + w / 2;
            const cy = y + h / 2;
            addLabel(engine.layers.l3, label, 'node-label:' + nid,
                cx, sublabel ? cy - 7 : cy, TEXT_STYLE.nodeTitle, { ax: 0.5, ay: 0.5 });
            if (sublabel) {
                addLabel(engine.layers.l3, sublabel, 'node-sublabel:' + nid,
                    cx, cy + 8, TEXT_STYLE.nodeSub, { ax: 0.5, ay: 0.5 });
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

        const groupColor = nodeColorOf(g);
        const box = makeGraphics('group-box:' + gid);
        fillStrokeBox(box, ox, oy, pos.w, pos.h, {
            radius: 8, fill: groupColor, fillAlpha: 0.22,
            stroke: groupColor, strokeAlpha: 0.85, strokeWidth: 1.5
        });
        engine.layers.l2.addChild(box);
        if (visible) {
            // Header text grows rightward from ox+10; reserve room on the right
            // for the info badge.  13px monospace ≈ 7.8px/char; the leading
            // arrow glyph + space cost 2 chars and must not be truncated.
            const headerChars = maxCharsForWidth(pos.w - 10 - (hasInfo ? 26 : 10), 7.8, 0) - 2;
            addLabel(engine.layers.l2, '\u25B6 ' + truncateLabel(g.class_name, headerChars), 'group-label:' + gid,
                ox + 10, oy + 15, TEXT_STYLE.groupHeader, { ax: 0, ay: 0.5 });
            if (hasTiming) {
                addLabel(engine.layers.l2, 'Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us),
                    'group-timing:' + gid, ox + 10, oy + 32, TEXT_STYLE.groupTiming, { ax: 0, ay: 0.5 });
            }
            if (hasInfo) {
                const info = makeGraphics('group-info-hit:' + gid);
                info.circle(ox + pos.w - 13, oy + 13, 8).fill({ color: 0x000000, alpha: 0.35 }).stroke({ color: 0xffffff, width: 1, alpha: 0.6 });
                engine.layers.l2.addChild(info);
                addLabel(engine.layers.l2, 'i', 'group-info:' + gid,
                    ox + pos.w - 13, oy + 13, TEXT_STYLE.info, { ax: 0.5, ay: 0.5 });
            }
        } else if (hasInfo) {
            const info = makeGraphics('group-info-hit:' + gid);
            info.circle(ox + pos.w - 13, oy + 13, 8).fill({ color: 0x000000, alpha: 0.35 });
            engine.layers.l2.addChild(info);
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

        const groupColor = nodeColorOf(g);
        const box = makeGraphics('group-box:' + gid);
        fillStrokeBox(box, ox, oy, pos.w, pos.h, {
            radius: 8, fill: groupColor, fillAlpha: 0.08,
            stroke: groupColor, strokeAlpha: 0.7, strokeWidth: 1.5
        });
        // header bar so the expanded container title stays legible
        box.roundRect(ox, oy, pos.w, 26, 8).fill({ color: groupColor, alpha: 0.35 });
        engine.layers.l2.addChild(box);
        if (visible) {
            // Expanded header shares the bar with the right-aligned info badge
            // and timing text; reserve room for whichever are present so the
            // truncated title never collides with them.
            const rightReserve = (hasInfo ? 26 : 10) + (hasTiming ? 130 : 0);
            const headerChars = maxCharsForWidth(pos.w - 10 - rightReserve, 7.8, 0) - 2;
            addLabel(engine.layers.l2, '\u25BC ' + truncateLabel(g.class_name, headerChars), 'group-header:' + gid,
                ox + 10, oy + 13, TEXT_STYLE.groupHeader, { ax: 0, ay: 0.5 });
            if (hasInfo) {
                const info = makeGraphics('group-info-hit:' + gid);
                info.circle(ox + pos.w - 13, oy + 13, 8).fill({ color: 0x000000, alpha: 0.35 }).stroke({ color: 0xffffff, width: 1, alpha: 0.6 });
                engine.layers.l2.addChild(info);
                addLabel(engine.layers.l2, 'i', 'group-info:' + gid,
                    ox + pos.w - 13, oy + 13, TEXT_STYLE.info, { ax: 0.5, ay: 0.5 });
            }
            if (hasTiming) {
                addLabel(engine.layers.l2, 'Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us),
                    'group-timing:' + gid, ox + pos.w - 26, oy + 13, TEXT_STYLE.groupTiming, { ax: 1, ay: 0.5 });
            }
        } else if (hasInfo) {
            const info = makeGraphics('group-info-hit:' + gid);
            info.circle(ox + pos.w - 13, oy + 13, 8).fill({ color: 0x000000, alpha: 0.35 });
            engine.layers.l2.addChild(info);
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
        if (!points || points.length < 2) { return; }
        g.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            g.lineTo(points[i].x, points[i].y);
        }
        g.stroke({ width: style.width, color: style.color, alpha: style.alpha });
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
        g.poly([
            tip.x, tip.y,
            baseX + px * half, baseY + py * half,
            baseX - px * half, baseY - py * half
        ]);
        g.fill({ color: style.color, alpha: style.arrowAlpha });
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
        const pillX = spec.cx - spec.w / 2;
        const pillY = spec.cy - spec.h / 2;
        const pillFill = spec.fillColor || IO_GROUP_FILL[spec.subtype] || 'rgba(127,140,141,0.55)';
        const pill = makeGraphics('io-pill:' + spec.id);
        fillStrokeBox(pill, pillX, pillY, spec.w, spec.h, {
            radius: Math.min(spec.h / 2, 12), fill: pillFill, fillAlpha: 1,
            stroke: 0xffffff, strokeAlpha: 0.2, strokeWidth: 1
        });
        this.layer.addChild(pill);
        addLabel(this.layer, spec.label, 'io-label:' + spec.id,
            spec.cx, spec.sublabel ? spec.cy - 6 : spec.cy, TEXT_STYLE.ioTitle, { ax: 0.5, ay: 0.5 });
        if (spec.sublabel) {
            addLabel(this.layer, spec.sublabel, 'io-sublabel:' + spec.id,
                spec.cx, spec.cy + 7, TEXT_STYLE.ioSub, { ax: 0.5, ay: 0.5 });
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
            const frame = makeGraphics('io-group-frame:' + ioGroup.id);
            fillStrokeBox(frame, frameX, frameY, frameW, frameH, {
                radius: 10, fill: fillColor, fillAlpha: 0.12,
                stroke: fillColor, strokeAlpha: 0.7, strokeWidth: 1.2
            });
            this.layer.addChild(frame);
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
            const collapseCx = geom.cx;
            const collapseCy = frameY + frameH - 4;
            const collapseBtn = makeGraphics('io-group-collapse:' + ioGroup.id);
            fillStrokeBox(collapseBtn, collapseCx - 34, collapseCy - 9, 68, 18, {
                radius: 9, fill: 0x000000, fillAlpha: 0.35, stroke: 0xffffff, strokeAlpha: 0.4, strokeWidth: 1
            });
            this.layer.addChild(collapseBtn);
            addLabel(this.layer, '\u25B2 \u6536\u8D77', 'io-group-collapse-label:' + ioGroup.id,
                collapseCx, collapseCy, TEXT_STYLE.ioSub, { ax: 0.5, ay: 0.5 });
            return frameY + frameH;
        }
        const groupPillX = geom.cx - geom.w / 2;
        const groupPillY = geom.cy - geom.h / 2;
        const groupPill = makeGraphics('io-group-pill:' + ioGroup.id);
        fillStrokeBox(groupPill, groupPillX, groupPillY, geom.w, geom.h, {
            radius: Math.min(geom.h / 2, 12), fill: fillColor, fillAlpha: 1,
            stroke: 0xffffff, strokeAlpha: 0.2, strokeWidth: 1
        });
        this.layer.addChild(groupPill);
        addLabel(this.layer, '\u25B6 ' + ioGroup.label, 'io-group-label:' + ioGroup.id,
            geom.cx, geom.cy, TEXT_STYLE.ioTitle, { ax: 0.5, ay: 0.5 });
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

    // Estimate how many monospace glyphs fit in `boxW` px after subtracting
    // `padding`.  charW is the per-glyph advance in px for the relevant font
    // size (~7.2px for the 12px node title, ~7.8px for the 13px group header).
    function maxCharsForWidth(boxW, charW, padding) {
        const usable = boxW - (padding || 0);
        return Math.max(1, Math.floor(usable / (charW || 7.2)));
    }

    // Hard-truncate `text` to `maxChars` glyphs, appending an ellipsis when it
    // overflows.  Box width / layout are unaffected — only the rendered string
    // is shortened so long labels no longer spill past their node/group box.
    function truncateLabel(text, maxChars) {
        const value = String(text || '');
        if (!(maxChars >= 1)) { return ''; }
        return value.length > maxChars ? (value.slice(0, Math.max(0, maxChars - 1)) + '…') : value;
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
        engine.labels = [];
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
        // The canvas/renderer must stay sized to the visible viewport (container),
        // NOT to the full world.  The world (engine.world) is scaled + translated by
        // the viewport so the whole DAG fits inside this fixed canvas.  Sizing the
        // renderer to svgW x svgH blows the canvas up to the entire graph and pushes
        // everything off-screen, which defeats the first-screen auto-fit.
        const containerSize = resolveContainerSize('applyWorldLayout');
        if (engine.app && engine.app.renderer && typeof engine.app.renderer.resize === 'function') {
            engine.app.renderer.resize(Math.ceil(containerSize.w), Math.ceil(containerSize.h));
        }
        if (engine.app && engine.app.canvas) {
            engine.app.canvas.style.width = '100%';
            engine.app.canvas.style.height = '';
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

    // Stage 1.5: fit the freshly-laid-out world into the visible container.
    // Required on the first render and after Expand/Collapse-All re-layouts.
    function performAutoFit() {
        if (!engine.worldBounds) {
            throw new Error('render_canvas.js: auto-fit requires worldBounds');
        }
        const containerSize = resolveContainerSize('auto-fit');
        const cw = containerSize.w;
        const ch = containerSize.h;
        const FIT_PADDING = 40;
        const vp = engine.viewportController.fitToView(engine.worldBounds, cw, ch, { padding: FIT_PADDING, maxScale: 1.0 });
        // Width-only fit: the canvas width fills the container so there is no
        // horizontal scroll.  The height is grown to the full scaled content so a
        // graph taller than the viewport overflows into the .dag-stage's vertical
        // scroll (matches the legacy SVG semantics) instead of being squeezed.
        const contentHeight = Math.ceil(engine.worldBounds.h * vp.scale + 2 * FIT_PADDING);
        const canvasHeight = Math.max(Math.ceil(ch), contentHeight);
        if (engine.app && engine.app.renderer && typeof engine.app.renderer.resize === 'function') {
            engine.app.renderer.resize(Math.ceil(cw), canvasHeight);
        }
        if (engine.app && engine.app.canvas) {
            engine.app.canvas.style.width = '100%';
            engine.app.canvas.style.height = '';
        }
        // The width-only fit top-aligns the world at y = padding inside a canvas
        // that was just grown to the full (tall) content height.  Growing the
        // canvas turns the #dag-stage into a vertical scroll container; on some
        // browsers (Chrome scroll anchoring) the stage is left scrolled away from
        // the top when its content resizes, which hides the top-of-graph
        // Input/Const nodes on first load.  Explicitly snap the stage back to the
        // top so the freshly top-aligned fit is actually what the user sees.
        if (engine.container && typeof engine.container.scrollTop !== 'undefined') {
            engine.container.scrollTop = 0;
        }
        return vp;
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

    function requestAnimationFramePromise(fn) {
        return new Promise(function (resolve) {
            const raf = (global.requestAnimationFrame || function (cb) { return setTimeout(cb, 0); });
            raf(function () {
                resolve(fn());
            });
        });
    }

    async function canvasRenderPhase1(data, renderOpts) {
        ensureEngine();
        await ensureStageMounted();
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
                layoutInfo = computeLayout(data, resolveContainerSize('layout').w);
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
            const wantAutoFit = (!engine.hasRenderedOnce) || (renderOpts && renderOpts.autoFit === true);
            if (wantAutoFit) {
                if (engine.app && engine.app.canvas) {
                    engine.app.canvas.style.width = '100%';
                    engine.app.canvas.style.height = '';
                }
                await requestAnimationFramePromise(function () {
                    return performAutoFit();
                });
            }
            engine.hasRenderedOnce = true;
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
            labels: (engine.labels || []).map(function (l) { return { name: l.name, x: l.x, y: l.y }; }),
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

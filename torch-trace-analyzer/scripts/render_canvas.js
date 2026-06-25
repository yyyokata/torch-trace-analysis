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
   getNodeColor, formatDur,
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
        // Phase 2 step 3: headless Pixi mock gains a minimal EventEmitter so the
        // node-side render probe can simulate click / dblclick / pointerdown on
        // group hit boxes (real PixiJS v8 wires the same API through Federated
        // Events).  Headless ``emit()`` is *synchronous*; tests rely on this to
        // schedule click vs. dblclick disambiguation timers deterministically.
        Container.prototype.on = function (name, fn) {
            if (typeof fn !== 'function') {
                throw new Error('render_canvas.js: headless Container.on requires a function listener for "' + name + '"');
            }
            if (!this.__listeners) { this.__listeners = {}; }
            if (!this.__listeners[name]) { this.__listeners[name] = []; }
            this.__listeners[name].push(fn);
            return this;
        };
        Container.prototype.off = function (name, fn) {
            if (!this.__listeners || !this.__listeners[name]) { return this; }
            if (typeof fn !== 'function') {
                this.__listeners[name] = [];
                return this;
            }
            this.__listeners[name] = this.__listeners[name].filter(function (l) { return l !== fn; });
            return this;
        };
        Container.prototype.emit = function (name, evt) {
            if (!this.__listeners || !this.__listeners[name]) { return this; }
            const listeners = this.__listeners[name].slice();
            for (let i = 0; i < listeners.length; i++) {
                listeners[i](evt);
            }
            return this;
        };
        function Graphics() {
            Container.call(this);
            this.__isHeadlessGraphics = true;
            this.__drawOps = [];
        }
        Graphics.prototype = Object.create(Container.prototype);
        Graphics.prototype.constructor = Graphics;
        ['clear', 'roundRect', 'rect', 'circle', 'ellipse', 'poly', 'fill', 'stroke',
         'setStrokeStyle', 'setFillStyle', 'moveTo', 'lineTo', 'closePath', 'beginPath']
            .forEach(function (name) {
                Graphics.prototype[name] = (name === 'clear')
                    ? function () { this.__drawOps = []; return this; }
                    : function () { this.__drawOps.push(name); return this; };
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
            // Live io pill geometry index keyed by io node/group id.  This is the
            // io half of the coordinate truth source: ``drawIOTasks`` populates it
            // (top-left box form) while drawing l5 so edge routing can derive io
            // endpoints from the same live geometry as the pill draw.  Cleared
            // wherever ``io_pills`` is reset.
            ioPillById: new Map(),
            // Phase 2 step 1: object pools persist across renders so toggle /
            // expand-all / focus paths can patch existing views instead of
            // running resetScene() and rebuilding every Pixi DisplayObject.
            // ``visible*`` sets describe the *current* frame and are updated by
            // ``diffAndPatch()``; pool entries themselves are never destroyed
            // here, only set ``visible=false``.
            nodePool: new Map(),
            groupPool: new Map(),
            edgePool: new Map(),
            visibleNodeIds: new Set(),
            visibleGroupIds: new Set(),
            visibleEdgeKeys: new Set(),
            labelsCreated: 0,
            labels: [],
            cullingEnabled: true,
            worldBounds: null,
            contentBounds: null,
            viewport: {
                scale: 1,
                x: 0,
                y: 0,
                worldWidth: 0,
                worldHeight: 0,
                minScale: 0.25,
                maxScale: 4
            },
            // Phase 2 step 3: ``groupBoxes`` indexes the live group hit target
            // (Graphics box) by gid so the inline runtime can attach click /
            // dblclick handlers to whichever path drew it.  Both the legacy
            // ``walkGroup`` path and the pool-driven ``createGroupView`` path
            // register their box here.  ``selectedGroupId`` tracks the
            // currently focused group for panel display.
            groupBoxes: new Map(),
            selectedGroupId: null,
            // Phase 2 step 4: ``dataRef`` / ``collapsedStateRef`` are the inline
            // runtime references the incremental render path needs in order to
            // (re)compute the visible scene without going back through the full
            // ``canvasRenderPhase1`` pipeline.  The inline runtime installs them
            // exactly once via ``__canvasSetIncrementalContext`` — never mutated
            // afterwards.  ``rendererResizeCallCount`` is a diagnostic counter
            // used by the phase-2 step-4 regression tests to confirm the toggle /
            // Expand-All / Collapse-All path never re-runs ``renderer.resize()``
            // (only the initial render / auto-fit path may bump it).
            dataRef: null,
            collapsedStateRef: null,
            rendererResizeCallCount: 0,
            // Phase 2 step 4: positive diagnostic counter bumped once per
            // ``invokeIncrementalRender`` (toggle / Expand-All / Collapse-All).
            // The regression tests assert this advances while
            // ``rendererResizeCallCount`` stays put — proving the pool-first
            // incremental path ran instead of a full re-render.
            incrementalRenderCount: 0
        };
        // Phase 2 step 3: ``engine.onGroupToggle`` / ``engine.onGroupSelect``
        // are the *engine-side* interaction hooks fired by ``bindGroupBox()``.
        // The inline runtime wires their behaviour by installing
        // ``window.__canvasOnGroupToggle`` / ``window.__canvasOnGroupSelect``
        // — render_canvas.js forwards to them at *call time* (not script load)
        // so wire ordering does not matter as long as the globals exist by the
        // first user click.  No silent fallback: missing globals throw hard.
        built.onGroupToggle = function (gid) {
            if (typeof global.__canvasOnGroupToggle !== 'function') {
                throw new Error('render_canvas.js: window.__canvasOnGroupToggle is not wired by inline runtime');
            }
            global.__canvasOnGroupToggle(gid);
        };
        built.onGroupSelect = function (gid) {
            if (typeof global.__canvasOnGroupSelect !== 'function') {
                throw new Error('render_canvas.js: window.__canvasOnGroupSelect is not wired by inline runtime');
            }
            global.__canvasOnGroupSelect(gid);
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

    function ensureLiveContainer(eng) {
        const liveContainer = resolveContainer();
        if (!liveContainer) {
            throw new Error('render_canvas.js: #dag-stage container not found while mounting canvas');
        }
        eng.container = liveContainer;
        return liveContainer;
    }

    function attachInitializedCanvasToCurrentContainer(eng) {
        const liveContainer = ensureLiveContainer(eng);
        if (!eng.initialized) {
            throw new Error('render_canvas.js: cannot attach canvas before Pixi application initialization completes');
        }
        if (!eng.app) {
            throw new Error('render_canvas.js: engine.app is missing after Pixi application initialization');
        }
        const canvas = eng.app.canvas;
        if (!canvas) {
            if (eng.usingHeadlessPixi) {
                return;
            }
            throw new Error('render_canvas.js: Pixi application did not expose canvas after initialization');
        }
        if (canvas.parentNode !== liveContainer) {
            if (typeof liveContainer.appendChild !== 'function') {
                throw new Error('render_canvas.js: live #dag-stage container cannot accept canvas append');
            }
            liveContainer.appendChild(canvas);
        }
        if (canvas.style) {
            canvas.style.width = '100%';
        }
    }

    // Stage 1.5: perform the async PixiJS v8 `app.init()` lazily on the first real
    // render.  Idempotent via `initPromise`; the headless mock resolves instantly.
    async function ensureStageMounted() {
        const eng = ensureEngine();
        if (eng.initialized) {
            attachInitializedCanvasToCurrentContainer(eng);
            return eng;
        }
        if (!eng.initPromise) {
            eng.initPromise = (async function () {
                ensureLiveContainer(eng);
                if (typeof eng.app.init === 'function') {
                    await eng.app.init({
                        backgroundAlpha: 0,
                        antialias: true,
                        autoDensity: false,
                        resolution: 1,
                        preference: 'webgl',
                        width: Math.max(1, getContainerWidth(eng.container) || 1280),
                        height: Math.max(1, getContainerHeight(eng.container) || 720)
                    });
                }
                eng.initialized = true;
                attachInitializedCanvasToCurrentContainer(eng);
                // `init()` may (re)create app.stage; (re-)attach the world graph.
                if (eng.app.stage && typeof eng.app.stage.addChild === 'function') {
                    eng.app.stage.addChild(eng.world);
                }
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
    function nodeColorOf(n) { return (typeof getNodeColor === 'function') ? getNodeColor(n) : '#4a6fa5'; }
    function formatDurOf(us) { return (typeof formatDur === 'function') ? formatDur(us) : String(us); }
    function lookupIsEdgeVisible() { return (typeof isEdgeVisible === 'function') ? isEdgeVisible : null; }
    function lookupResolveCollapsedAncestor() { return (typeof resolveCollapsedAncestor === 'function') ? resolveCollapsedAncestor : null; }
    function lookupEdgeKey() { return (typeof edgeKey === 'function') ? edgeKey : null; }
    function lookupCollapsedState() { return (typeof collapsedState !== 'undefined') ? collapsedState : null; }
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

    // Pool-first scene-label bookkeeping.  The pool view text glyphs are created
    // lazily inside the patch functions (not via addLabel, since they are parented
    // to the recyclable view root, not a layer).  This records a label entry in
    // the same ``engine.labels`` / ``engine.labelsCreated`` model the legacy draw
    // path used, so the snapshot label introspection (count, names, coords) and
    // the culling regression tests keep working with absolute world coordinates.
    function registerSceneLabel(name, absX, absY) {
        const x = numericOrNull(absX);
        const y = numericOrNull(absY);
        if (x === null || y === null) {
            throw new Error('render_canvas.js: registerSceneLabel requires numeric x/y for ' + name);
        }
        engine.labels.push({ name: name, x: x, y: y });
        engine.labelsCreated += 1;
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
        // Width tracks the actual visible stage width (or its stable parent when the
        // stage is transiently inflated by autoDensity).  Height is different after
        // the CSS switch to page-level scrolling: #dag-stage no longer has a fixed
        // viewport height, so clientHeight may be 0 while the browser viewport is
        // still perfectly valid for width-only auto-fit. In that case we must
        // explicitly use window.innerHeight; if neither source yields a positive
        // height we still hard-fail.
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
        if (ch === null || ch <= 0) {
            ch = numericOrNull(global.innerHeight);
        }
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

    function expandBounds(acc, rect) {
        const bounds = normalizeWorldBounds(rect);
        if (!acc) {
            return { x: bounds.x, y: bounds.y, w: bounds.w, h: bounds.h };
        }
        const minX = Math.min(acc.x, bounds.x);
        const minY = Math.min(acc.y, bounds.y);
        const maxX = Math.max(acc.x + acc.w, bounds.x + bounds.w);
        const maxY = Math.max(acc.y + acc.h, bounds.y + bounds.h);
        return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
    }

    function computeRenderableContentBounds() {
        let bounds = null;
        // Pool-first: the rendered content lives in the object pools (the legacy
        // engine.nodes / engine.groups arrays are no longer populated).  Walk the
        // currently-visible group + node views and take their snapshot boxes
        // (absolute world coordinates) as the content extent.
        engine.groupPool.forEach(function (view) {
            if (view.visible !== true || !view.snapshot) { return; }
            const s = view.snapshot;
            bounds = expandBounds(bounds, { x: s.x, y: s.y, w: s.w, h: s.h });
        });
        engine.nodePool.forEach(function (view) {
            if (view.visible !== true || !view.snapshot) { return; }
            const s = view.snapshot;
            bounds = expandBounds(bounds, { x: s.x, y: s.y, w: s.w, h: s.h });
        });
        (engine.io_pills || []).forEach(function (pill) {
            bounds = expandBounds(bounds, {
                x: pill.cx - pill.w / 2,
                y: pill.cy - pill.h / 2,
                w: pill.w,
                h: pill.h
            });
        });
        if (!bounds) {
            throw new Error('render_canvas.js: auto-fit requires rendered content bounds');
        }
        return bounds;
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
        if (!engine.hasRenderedOnce) {
            return true;
        }
        return engine.cullManager.isVisible(rect, engine.viewportController.currentBounds());
    }

    // ── GroupView ──────────────────────────────────────────────────────────
    // Phase 2 step 3: bind click / right-click / pointerdown handlers to a
    // group hit box.  Idempotent — the second call on the same box is a no-op.
    // Click handlers are wired to ``engine.onGroupSelect`` (single click, show
    // panel) and ``engine.onGroupToggle`` (double click, collapse/expand)
    // which forward to the inline-runtime hooks.
    //
    // PixiJS v8 does not reliably dispatch ``dblclick`` for Graphics hit boxes,
    // so we simulate a double click by watching two ``click`` events arrive
    // within a 200ms window.  The first click arms the delayed
    // ``onGroupSelect`` timer; the second click clears that timer, resets the
    // timer slot, toggles the group immediately, and returns so a double click
    // never opens the panel.
    //
    // pointerdown with button===2 is the right-mouse path: two within 250ms
    // is reserved as the Semantic Zoom entry (Step 5).  Step 3 only defines
    // the recognition handler and the body is a placeholder.  ``rightclick``
    // is intercepted purely to suppress the browser context menu (Pixi v8
    // emits ``rightclick`` synthetically on right-mouse-up).
    function bindGroupBox(gid, box) {
        if (!box) {
            throw new Error('render_canvas.js: bindGroupBox missing box for group ' + gid);
        }
        if (box.__phase2EventsBound) { return; }
        box.__phase2EventsBound = true;
        box.eventMode = 'static';
        const clickDelayMs = 200;
        const rightDblIntervalMs = 250;
        const state = { clickTimer: null, lastClickTime: 0, rightLastDown: 0 };
        box.on('click', function (e) {
            if (e && typeof e.stopPropagation === 'function') { e.stopPropagation(); }
            const now = (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
            const isDoubleClick = state.lastClickTime > 0 && (now - state.lastClickTime) <= clickDelayMs;
            state.lastClickTime = now;
            if (isDoubleClick) {
                if (state.clickTimer !== null) {
                    clearTimeout(state.clickTimer);
                    state.clickTimer = null;
                }
                engine.onGroupToggle(gid);
                return;
            }
            if (state.clickTimer !== null) {
                clearTimeout(state.clickTimer);
                state.clickTimer = null;
            }
            state.clickTimer = setTimeout(function () {
                state.clickTimer = null;
                engine.onGroupSelect(gid);
            }, clickDelayMs);
        });
        box.on('rightclick', function (e) {
            if (e && typeof e.preventDefault === 'function') { e.preventDefault(); }
        });
        box.on('pointerdown', function (e) {
            if (!e || e.button !== 2) { return; }
            const now = (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
            if (now - state.rightLastDown < rightDblIntervalMs) {
                state.rightLastDown = 0;
                // TODO Step5: semantic zoom
            } else {
                state.rightLastDown = now;
            }
        });
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

    // ── port derivation (coordinate truth source) ──────────────────────────
    // Edge endpoints are derived on demand from the LIVE box of whatever drew
    // the endpoint (a pool node/group view, or an io pill).  These formulas are
    // byte-identical to the deleted nodePortMap registration:
    //   * out-port  = bottom-edge mid-point  (was ``id__out``)
    //   * in-port   = top-edge mid-point     (was ``id__in``)
    //   * center    = box centre             (was ``id__center`` / bare ``id``)
    function outPortOf(box) { return { cx: box.x + box.w / 2, cy: box.y + box.h }; }
    function inPortOf(box) { return { cx: box.x + box.w / 2, cy: box.y }; }
    function centerPortOf(box) { return { cx: box.x + box.w / 2, cy: box.y + box.h / 2 }; }

    // Resolve an id to its LIVE {x,y,w,h} box from the object pools (visible
    // views only) or the io pill index.  Returns null when no live box owns the
    // id; callers are responsible for turning that into a hard error so a
    // missing endpoint never silently degrades.
    function boxForId(id) {
        const sid = String(id);
        const nv = engine.nodePool.get(sid);
        if (nv && nv.visible === true) { return { x: nv.x, y: nv.y, w: nv.w, h: nv.h }; }
        const gv = engine.groupPool.get(sid);
        if (gv && gv.visible === true) { return { x: gv.x, y: gv.y, w: gv.w, h: gv.h }; }
        const pill = engine.ioPillById.get(sid);
        if (pill) { return { x: pill.x, y: pill.y, w: pill.w, h: pill.h }; }
        return null;
    }

    // Boundary-port index: a group's ``in_ports`` / ``out_ports`` declare
    // *boundary node ids* that are never drawn as standalone nodes (they have no
    // box of their own).  The deleted nodePortMap redirected every such id onto
    // the owning group's in/out port; we rebuild that mapping (id -> {groupId,
    // kind}) from DATA once per dataRef so endpoint resolution can do the same
    // redirect against the LIVE group box.  This is the exact set the legacy
    // ``registerCollapsedGroupPorts`` / ``registerExpandedGroupPorts`` walked.
    function ensurePortNodeIndex() {
        if (engine.portNodeIndex && engine.portNodeIndexData === engine.dataRef) {
            return engine.portNodeIndex;
        }
        const idx = new Map();
        const data = engine.dataRef;
        if (data) {
            (data.groups || []).forEach(function (g) {
                (g.in_ports || []).forEach(function (p) {
                    if (p && p.node_id !== null && p.node_id !== undefined) {
                        idx.set(String(p.node_id), { groupId: String(g.id), kind: 'in' });
                    }
                });
                (g.out_ports || []).forEach(function (p) {
                    if (p && p.node_id !== null && p.node_id !== undefined) {
                        idx.set(String(p.node_id), { groupId: String(g.id), kind: 'out' });
                    }
                });
            });
        }
        engine.portNodeIndex = idx;
        engine.portNodeIndexData = engine.dataRef;
        return idx;
    }

    // Resolve a (already collapsed-ancestor-redirected) endpoint id to its draw
    // port {cx,cy} using ``boxResolver`` for the LIVE/fresh box lookup.
    //   * a boundary port node id  -> its owning group's in/out port (the legacy
    //     bare ``nodePortMap[port.node_id]`` redirect; kind decides in vs out,
    //     independent of the edge's src/dst role);
    //   * any other id (node / group / collapsed box / io pill) -> the box's
    //     out-port when it is the edge source, in-port when it is the dest (the
    //     legacy ``id__out`` / ``id__in`` keys).
    // Returns null when no live box backs the id so callers raise a hard error.
    function portPointForEndpoint(resolvedId, role, boxResolver) {
        const sid = String(resolvedId);
        const portInfo = ensurePortNodeIndex().get(sid);
        if (portInfo) {
            const gbox = boxResolver(portInfo.groupId);
            if (!gbox) { return null; }
            return portInfo.kind === 'in' ? inPortOf(gbox) : outPortOf(gbox);
        }
        const box = boxResolver(sid);
        if (!box) { return null; }
        return role === 'src' ? outPortOf(box) : inPortOf(box);
    }


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
        engine.ioPillById.set(String(spec.id), {
            x: spec.cx - spec.w / 2, y: spec.cy - spec.h / 2, w: spec.w, h: spec.h
        });
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
            engine.ioPillById.set(String(ioGroup.id), {
                x: geom.cx - availableW / 2, y: geom.cy - expanded.height / 2,
                w: availableW, h: expanded.height
            });
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
        engine.ioPillById.set(String(ioGroup.id), {
            x: geom.cx - geom.w / 2, y: geom.cy - geom.h / 2, w: geom.w, h: geom.h
        });
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

    // ── Phase 2 step 1: object pool + diffAndPatch ─────────────────────────
    // The factories below build the long-lived NodeView / GroupView / EdgeView
    // entries that ``diffAndPatch()`` populates and patches.  Each factory
    // attaches its root container to the appropriate scene-graph layer once at
    // create time so subsequent renders only flip ``visible`` and patch
    // snapshot fields — they never re-add to the layer or re-create Pixi
    // DisplayObjects.
    //
    // Required snapshot fields are listed for each kind below; callers must
    // hand in fully-populated snapshots and ``diffAndPatch()`` will throw if
    // any field is missing (no silent default fill, no fallback).
    const NODE_SNAPSHOT_FIELDS  = ['x', 'y', 'w', 'h', 'label', 'sublabel', 'fill', 'stroke', 'alpha'];
    const GROUP_SNAPSHOT_FIELDS = ['x', 'y', 'w', 'h', 'label', 'collapsed', 'fill', 'alpha', 'hasInfo', 'hasTiming', 'timingText'];
    const EDGE_SNAPSHOT_FIELDS  = ['srcId', 'dstId', 'stroke', 'strokeWidth', 'alpha', 'dashed', 'arrowAlpha'];

    function requireSnapshotFields(kind, id, snapshot, fields) {
        if (!snapshot || typeof snapshot !== 'object') {
            throw new Error('render_canvas.js: diffAndPatch missing ' + kind + ' snapshot for ' + id);
        }
        for (let i = 0; i < fields.length; i++) {
            const f = fields[i];
            if (!Object.prototype.hasOwnProperty.call(snapshot, f)) {
                throw new Error('render_canvas.js: diffAndPatch ' + kind + ' snapshot missing field "' + f + '" for ' + id);
            }
        }
    }

    function createNodeView(id) {
        const root = new engine.pixi.Container();
        root.name = 'node-view:' + id;
        root.visible = false;
        const box = makeGraphics('node-box:' + id);
        root.addChild(box);
        engine.layers.l3.addChild(root);
        return {
            id: String(id),
            root: root,
            box: box,
            titleText: null,
            subText: null,
            visible: false,
            snapshot: null
        };
    }

    function createGroupView(id) {
        const root = new engine.pixi.Container();
        root.name = 'group-view:' + id;
        root.visible = false;
        const box = makeGraphics('group-box:' + id);
        // Phase 2 step 3 wires the click / dblclick / right-dblclick handlers
        // onto this hit target; ``bindGroupBox`` flips ``eventMode='static'``
        // and idempotently registers the listeners.
        box.eventMode = 'static';
        root.addChild(box);
        engine.layers.l2.addChild(root);
        // Index this pool entry's box so the inline runtime / tests can find
        // it via ``engine.groupBoxes`` once ``diffAndPatch`` reactivates it.
        engine.groupBoxes.set(String(id), box);
        bindGroupBox(id, box);
        return {
            id: String(id),
            root: root,
            box: box,
            headerText: null,
            timingTextNode: null,
            infoGfx: null,
            infoText: null,
            visible: false,
            interactionEnabled: false,
            snapshot: null
        };
    }

    function createEdgeView(key) {
        const root = new engine.pixi.Container();
        root.name = 'edge-view:' + key;
        root.visible = false;
        const path = makeGraphics('edge-path:' + key);
        root.addChild(path);
        engine.layers.l1.addChild(root);
        return {
            key: String(key),
            root: root,
            path: path,
            arrow: null,
            visible: false,
            snapshot: null
        };
    }

    function patchNodeView(view, snapshot) {
        requireSnapshotFields('node', view.id, snapshot, NODE_SNAPSHOT_FIELDS);
        view.root.x = snapshot.x;
        view.root.y = snapshot.y;
        // Mirror the absolute layout box onto the view so edge routing can derive
        // this node's live ports (out/in/center) without a global port dictionary.
        view.x = snapshot.x;
        view.y = snapshot.y;
        view.w = snapshot.w;
        view.h = snapshot.h;
        view.root.alpha = snapshot.alpha;
        view.box.clear();
        fillStrokeBox(view.box, 0, 0, snapshot.w, snapshot.h, {
            radius: 7, fill: snapshot.fill, fillAlpha: 0.95,
            stroke: snapshot.stroke, strokeAlpha: 0.14, strokeWidth: 1
        });
        // title + sublabel are culled (the box itself is always drawn).  On the
        // first render shouldCreateLabel() returns true unconditionally; once the
        // graph has rendered once, off-screen labels are skipped (and any existing
        // recycled glyph is hidden) so labelsCreated tracks only on-screen text.
        const showLabels = shouldCreateLabel({ x: snapshot.x, y: snapshot.y, w: snapshot.w, h: snapshot.h });
        if (showLabels) {
            // title (lazily created, parented to view.root)
            if (!view.titleText) {
                view.titleText = makeText(snapshot.label, 'node-label:' + view.id, TEXT_STYLE.nodeTitle);
                view.root.addChild(view.titleText);
            }
            view.titleText.text = snapshot.label;
            view.titleText.anchor.set(0.5, 0.5);
            view.titleText.x = snapshot.w / 2;
            view.titleText.y = snapshot.sublabel ? (snapshot.h / 2 - 7) : (snapshot.h / 2);
            view.titleText.visible = true;
            registerSceneLabel('node-label:' + view.id,
                snapshot.x + snapshot.w / 2,
                snapshot.y + (snapshot.sublabel ? (snapshot.h / 2 - 7) : (snapshot.h / 2)));
            // sublabel
            if (snapshot.sublabel) {
                if (!view.subText) {
                    view.subText = makeText(snapshot.sublabel, 'node-sublabel:' + view.id, TEXT_STYLE.nodeSub);
                    view.root.addChild(view.subText);
                }
                view.subText.text = snapshot.sublabel;
                view.subText.anchor.set(0.5, 0.5);
                view.subText.x = snapshot.w / 2;
                view.subText.y = snapshot.h / 2 + 8;
                view.subText.visible = true;
                registerSceneLabel('node-sublabel:' + view.id,
                    snapshot.x + snapshot.w / 2, snapshot.y + snapshot.h / 2 + 8);
            } else if (view.subText) {
                view.subText.visible = false;
            }
        } else {
            if (view.titleText) { view.titleText.visible = false; }
            if (view.subText) { view.subText.visible = false; }
        }
        view.snapshot = snapshot;
    }

    function patchGroupView(view, snapshot) {
        requireSnapshotFields('group', view.id, snapshot, GROUP_SNAPSHOT_FIELDS);
        view.root.x = snapshot.x;
        view.root.y = snapshot.y;
        // Mirror the absolute layout box (collapsed OR expanded — snapshot.x/y/w/h
        // is already the current-state real box) so edge routing can derive this
        // group's live ports without a global port dictionary.
        view.x = snapshot.x;
        view.y = snapshot.y;
        view.w = snapshot.w;
        view.h = snapshot.h;
        view.root.alpha = snapshot.alpha;
        view.box.clear();
        const gc = snapshot.fill;
        if (snapshot.collapsed) {
            fillStrokeBox(view.box, 0, 0, snapshot.w, snapshot.h, {
                radius: 8, fill: gc, fillAlpha: 0.22, stroke: gc, strokeAlpha: 0.85, strokeWidth: 1.5
            });
        } else {
            fillStrokeBox(view.box, 0, 0, snapshot.w, snapshot.h, {
                radius: 8, fill: gc, fillAlpha: 0.08, stroke: gc, strokeAlpha: 0.7, strokeWidth: 1.5
            });
            view.box.roundRect(0, 0, snapshot.w, 26, 8).fill({ color: gc, alpha: 0.35 }); // header bar
        }
        // Header / info-'i' / timing glyphs are culled (the box + the info circle
        // graphic are always drawn).  Mirrors the legacy draw path which always
        // painted the container chrome but only created the text labels for
        // on-screen groups.
        const showLabels = shouldCreateLabel({ x: snapshot.x, y: snapshot.y, w: snapshot.w, h: snapshot.h });
        // header text (lazily created, parented to view.root)
        const arrow = snapshot.collapsed ? '\u25B6 ' : '\u25BC ';
        const rightReserve = snapshot.collapsed
            ? (snapshot.hasInfo ? 26 : 10)
            : ((snapshot.hasInfo ? 26 : 10) + (snapshot.hasTiming ? 130 : 0));
        const headerChars = maxCharsForWidth(snapshot.w - 10 - rightReserve, 7.8, 0) - 2;
        if (showLabels) {
            if (!view.headerText) {
                view.headerText = makeText('', 'group-header:' + view.id, TEXT_STYLE.groupHeader);
                view.root.addChild(view.headerText);
            }
            view.headerText.text = arrow + truncateLabel(snapshot.label, headerChars);
            view.headerText.anchor.set(0, 0.5);
            view.headerText.x = 10;
            view.headerText.y = snapshot.collapsed ? 15 : 13;
            view.headerText.visible = true;
            registerSceneLabel('group-header:' + view.id,
                snapshot.x + 10, snapshot.y + (snapshot.collapsed ? 15 : 13));
        } else if (view.headerText) {
            view.headerText.visible = false;
        }
        // info badge: the circle graphic is always drawn when hasInfo; the 'i'
        // text glyph is culled like the other labels.
        if (snapshot.hasInfo) {
            if (!view.infoGfx) {
                view.infoGfx = makeGraphics('group-info-hit:' + view.id);
                view.root.addChild(view.infoGfx);
            }
            view.infoGfx.clear();
            view.infoGfx.circle(snapshot.w - 13, 13, 8).fill({ color: 0x000000, alpha: 0.35 }).stroke({ color: 0xffffff, width: 1, alpha: 0.6 });
            view.infoGfx.visible = true;
            if (showLabels) {
                if (!view.infoText) {
                    view.infoText = makeText('i', 'group-info:' + view.id, TEXT_STYLE.info);
                    view.root.addChild(view.infoText);
                }
                view.infoText.anchor.set(0.5, 0.5);
                view.infoText.x = snapshot.w - 13;
                view.infoText.y = 13;
                view.infoText.visible = true;
                registerSceneLabel('group-info:' + view.id, snapshot.x + snapshot.w - 13, snapshot.y + 13);
            } else if (view.infoText) {
                view.infoText.visible = false;
            }
        } else {
            if (view.infoGfx) { view.infoGfx.visible = false; }
            if (view.infoText) { view.infoText.visible = false; }
        }
        // timing text (lazily created, parented to view.root)
        if (snapshot.hasTiming && showLabels) {
            if (!view.timingTextNode) {
                view.timingTextNode = makeText('', 'group-timing:' + view.id, TEXT_STYLE.groupTiming);
                view.root.addChild(view.timingTextNode);
            }
            view.timingTextNode.text = snapshot.timingText;
            let timingX;
            let timingY;
            if (snapshot.collapsed) {
                view.timingTextNode.anchor.set(0, 0.5);
                view.timingTextNode.x = 10;
                view.timingTextNode.y = 32;
                timingX = snapshot.x + 10;
                timingY = snapshot.y + 32;
            } else {
                view.timingTextNode.anchor.set(1, 0.5);
                view.timingTextNode.x = snapshot.w - 26;
                view.timingTextNode.y = 13;
                timingX = snapshot.x + snapshot.w - 26;
                timingY = snapshot.y + 13;
            }
            view.timingTextNode.visible = true;
            registerSceneLabel('group-timing:' + view.id, timingX, timingY);
        } else if (view.timingTextNode) {
            view.timingTextNode.visible = false;
        }
        view.snapshot = snapshot;
    }

    function patchEdgeView(view, snapshot) {
        requireSnapshotFields('edge', view.key, snapshot, EDGE_SNAPSHOT_FIELDS);
        view.root.alpha = snapshot.alpha;
        view.path.clear();
        // Derive both endpoints from the LIVE boxes of whatever drew them.  By the
        // time patchEdges runs, patchGroups + patchNodes (and drawIOTasks before
        // that) have already populated the pool views / io pill index for this
        // frame, so a missing box is a hard error — never a silent skip.  A
        // boundary port-node endpoint is redirected onto its owning group's
        // in/out port (see portPointForEndpoint).
        const fromPort = portPointForEndpoint(snapshot.srcId, 'src', boxForId);
        if (!fromPort) {
            throw new Error('patchEdgeView: src view missing from pools/io: ' + snapshot.srcId);
        }
        const toPort = portPointForEndpoint(snapshot.dstId, 'dst', boxForId);
        if (!toPort) {
            throw new Error('patchEdgeView: dst view missing from pools/io: ' + snapshot.dstId);
        }
        const route = EdgeRoute.compute('direct', fromPort.cx, fromPort.cy, toPort.cx, toPort.cy, snapshot.routeMeta || null);
        // route is null only for a degenerate span; computeVisibleScene already
        // drops such edges, so this is defensive: clear-only, never draw garbage.
        if (route) {
            const style = {
                color: snapshot.stroke,
                width: snapshot.strokeWidth,
                alpha: snapshot.alpha,
                arrowAlpha: snapshot.arrowAlpha,
                dashed: snapshot.dashed
            };
            strokePolyline(view.path, route.points, style);
            drawArrowHead(view.path, route.points, style);
        }
        view.snapshot = snapshot;
    }

    function getSnapshotEntry(map, id) {
        if (!map || typeof map !== 'object') {
            throw new Error('render_canvas.js: diffAndPatch missing snapshot map for id ' + id);
        }
        const key = String(id);
        if (Object.prototype.hasOwnProperty.call(map, key)) { return map[key]; }
        if (Object.prototype.hasOwnProperty.call(map, id)) { return map[id]; }
        throw new Error('render_canvas.js: diffAndPatch missing snapshot entry for id ' + id);
    }

    function patchGroups(prevIds, nextIds, snapshotMap) {
        const prevSet = (prevIds instanceof Set) ? prevIds : new Set(prevIds || []);
        const nextSet = (nextIds instanceof Set) ? nextIds : new Set(nextIds || []);
        nextSet.forEach(function (gid) {
            const key = String(gid);
            let view = engine.groupPool.get(key);
            if (!view) {
                view = createGroupView(gid);
                engine.groupPool.set(key, view);
            }
            view.root.visible = true;
            view.visible = true;
            view.box.eventMode = 'static';
            view.interactionEnabled = true;
            patchGroupView(view, getSnapshotEntry(snapshotMap, gid));
        });
        prevSet.forEach(function (gid) {
            if (nextSet.has(gid)) { return; }
            const key = String(gid);
            const view = engine.groupPool.get(key);
            if (!view) {
                throw new Error('render_canvas.js: diffAndPatch recycle missing groupView in pool: ' + gid);
            }
            view.root.visible = false;
            view.visible = false;
            view.box.eventMode = 'none';
            view.interactionEnabled = false;
        });
    }

    function patchNodes(prevIds, nextIds, snapshotMap) {
        const prevSet = (prevIds instanceof Set) ? prevIds : new Set(prevIds || []);
        const nextSet = (nextIds instanceof Set) ? nextIds : new Set(nextIds || []);
        nextSet.forEach(function (nid) {
            const key = String(nid);
            let view = engine.nodePool.get(key);
            if (!view) {
                view = createNodeView(nid);
                engine.nodePool.set(key, view);
            }
            view.root.visible = true;
            view.visible = true;
            patchNodeView(view, getSnapshotEntry(snapshotMap, nid));
        });
        prevSet.forEach(function (nid) {
            if (nextSet.has(nid)) { return; }
            const key = String(nid);
            const view = engine.nodePool.get(key);
            if (!view) {
                throw new Error('render_canvas.js: diffAndPatch recycle missing nodeView in pool: ' + nid);
            }
            view.root.visible = false;
            view.visible = false;
        });
    }

    function patchEdges(prevKeys, nextKeys, snapshotMap) {
        const prevSet = (prevKeys instanceof Set) ? prevKeys : new Set(prevKeys || []);
        const nextSet = (nextKeys instanceof Set) ? nextKeys : new Set(nextKeys || []);
        nextSet.forEach(function (ek) {
            const key = String(ek);
            let view = engine.edgePool.get(key);
            if (!view) {
                view = createEdgeView(ek);
                engine.edgePool.set(key, view);
            }
            view.root.visible = true;
            view.visible = true;
            patchEdgeView(view, getSnapshotEntry(snapshotMap, ek));
        });
        prevSet.forEach(function (ek) {
            if (nextSet.has(ek)) { return; }
            const key = String(ek);
            const view = engine.edgePool.get(key);
            if (!view) {
                throw new Error('render_canvas.js: diffAndPatch recycle missing edgeView in pool: ' + ek);
            }
            view.root.visible = false;
            view.visible = false;
        });
    }

    function diffAndPatch(prevVisible, nextVisible) {
        if (!prevVisible || typeof prevVisible !== 'object') {
            throw new Error('render_canvas.js: diffAndPatch requires prevVisible');
        }
        if (!nextVisible || typeof nextVisible !== 'object') {
            throw new Error('render_canvas.js: diffAndPatch requires nextVisible');
        }
        const required = ['nodeIds', 'groupIds', 'edgeKeys', 'nodeSnapshots', 'groupSnapshots', 'edgeSnapshots'];
        for (let i = 0; i < required.length; i++) {
            const f = required[i];
            if (!Object.prototype.hasOwnProperty.call(nextVisible, f)) {
                throw new Error('render_canvas.js: diffAndPatch nextVisible missing field "' + f + '"');
            }
        }
        // Pre-validate every snapshot **before** mutating any pool so a
        // malformed frame leaves the existing pool state untouched (no
        // half-built views, no fabricated default fields).
        (nextVisible.groupIds || []).forEach(function (gid) {
            requireSnapshotFields('group', gid, getSnapshotEntry(nextVisible.groupSnapshots, gid), GROUP_SNAPSHOT_FIELDS);
        });
        (nextVisible.nodeIds || []).forEach(function (nid) {
            requireSnapshotFields('node', nid, getSnapshotEntry(nextVisible.nodeSnapshots, nid), NODE_SNAPSHOT_FIELDS);
        });
        (nextVisible.edgeKeys || []).forEach(function (ek) {
            requireSnapshotFields('edge', ek, getSnapshotEntry(nextVisible.edgeSnapshots, ek), EDGE_SNAPSHOT_FIELDS);
        });
        patchGroups(prevVisible.groupIds, nextVisible.groupIds, nextVisible.groupSnapshots);
        patchNodes(prevVisible.nodeIds, nextVisible.nodeIds, nextVisible.nodeSnapshots);
        patchEdges(prevVisible.edgeKeys, nextVisible.edgeKeys, nextVisible.edgeSnapshots);
        engine.visibleGroupIds = new Set((nextVisible.groupIds || []).map(String));
        engine.visibleNodeIds  = new Set((nextVisible.nodeIds  || []).map(String));
        engine.visibleEdgeKeys = new Set((nextVisible.edgeKeys || []).map(String));
    }

    // ── Phase 2 step 4: incremental render path ────────────────────────────
    //
    // ``setIncrementalContext`` is the *one-time* hand-off the inline runtime
    // performs at startup so render_canvas.js holds live references to ``DATA``
    // (the deserialised flowchart payload) and ``collapsedState`` (the inline
    // mutable map keyed by group id).  The references are read-only from the
    // engine's perspective; only the inline runtime mutates ``collapsedState``
    // (via ``__canvasOnGroupToggle``, ``expandAll`` and ``collapseAll``), and
    // ``DATA`` itself is immutable for the lifetime of the page.
    //
    // ``computeVisibleScene`` walks ``data.root_groups`` top-down and returns a
    // ``VisibilityFrame`` (``{ groupIds, nodeIds, edgeKeys, *Snapshots }``) that
    // ``diffAndPatch()`` can consume directly.  Callers may pass a precomputed
    // layout bundle from ``computeLayoutMeta`` so the incremental render path can
    // re-use the exact reflow it already applied to world bounds and canvas size.
    //
    // ``invokeIncrementalRender`` is the engine-side hook: it captures the
    // current visible sets as ``prev``, recomputes layout metadata + world
    // bounds from ``DATA + collapsedState``, applies the resized viewport world,
    // computes the next frame, patches the pool, and then auto-fits the new
    // content.  Incremental render is therefore allowed to call
    // ``renderer.resize()`` when the re-layout changes the canvas dimensions.
    function setIncrementalContext(ctx) {
        ensureEngine();
        if (!ctx || typeof ctx !== 'object') {
            throw new Error('render_canvas.js: __canvasSetIncrementalContext requires an object');
        }
        if (!ctx.data || typeof ctx.data !== 'object') {
            throw new Error('render_canvas.js: __canvasSetIncrementalContext: ctx.data is required');
        }
        if (!ctx.collapsedState || typeof ctx.collapsedState !== 'object') {
            throw new Error('render_canvas.js: __canvasSetIncrementalContext: ctx.collapsedState is required');
        }
        engine.dataRef = ctx.data;
        engine.collapsedStateRef = ctx.collapsedState;
    }

    function requireFiniteXYWH(kind, id, meta) {
        if (!meta) {
            throw new Error('computeVisibleScene: ' + kind + ' ' + id + ' has no layout meta in groupLayout');
        }
        if (!Number.isFinite(meta.x) || !Number.isFinite(meta.y) ||
            !Number.isFinite(meta.w) || !Number.isFinite(meta.h)) {
            throw new Error('computeVisibleScene: ' + kind + ' ' + id + ' has incomplete layout meta {x,y,w,h}');
        }
    }

    function makeIncrGroupSnapshot(g, collapsed, groupMeta) {
        // groupMeta carries the *real* absolute layout box (x/y/w/h) for the
        // current collapsed state — never fabricate (0,0).  A missing/partial
        // meta is a hard error (no silent fallback): it means computeVisibleScene
        // walked a group the freshly-computed layout never positioned.
        requireFiniteXYWH('group', g.id, groupMeta);
        const groupColor = nodeColorOf(g);
        const hasTiming = !!g.has_timing;
        const hasInfo = !!g.src_file;
        return {
            id: g.id,
            x: groupMeta.x, y: groupMeta.y, w: groupMeta.w, h: groupMeta.h,
            label: String(g.class_name),
            collapsed: collapsed[g.id] === true,
            fill: groupColor,
            alpha: 1.0,
            hasInfo: hasInfo,
            hasTiming: hasTiming,
            timingText: hasTiming ? ('Kernel ' + g.pct.toFixed(1) + '% \u00B7 ' + formatDurOf(g.dur_us)) : ''
        };
    }

    function makeIncrNodeSnapshot(n, nodeMeta) {
        // nodeMeta carries the *real* absolute layout box (x/y/w/h) for the
        // current collapsed state.  Missing/partial meta is a hard error.
        requireFiniteXYWH('node', n.id, nodeMeta);
        return {
            id: n.id,
            x: nodeMeta.x, y: nodeMeta.y, w: nodeMeta.w, h: nodeMeta.h,
            label: truncateLabel(n.class_name, maxCharsForWidth(nodeMeta.w, 7.2, 8)),
            sublabel: n.has_timing ? (n.pct.toFixed(1) + '%') : '',
            fill: nodeColorOf(n),
            stroke: 0xffffff,
            alpha: 1.0
        };
    }

    function makeIncrEdgeSnapshot(e, edgeMeta) {
        // The edge snapshot no longer carries pre-baked geometry (points/start/
        // end).  It persists only the resolved endpoint *ids* (``srcId`` /
        // ``dstId``) plus the routing offset (``routeMeta``) and style fields;
        // patchEdgeView re-derives the polyline from the LIVE endpoint boxes at
        // patch time.  All draw fields are carried truthfully — no fallback.
        if (edgeMeta.srcId === undefined || edgeMeta.srcId === null ||
            edgeMeta.dstId === undefined || edgeMeta.dstId === null) {
            throw new Error('computeVisibleScene: edge ' +
                (e && e.from !== undefined ? e.from : '?') + '->' + (e && e.to !== undefined ? e.to : '?') +
                ' has no resolved src/dst id');
        }
        return {
            srcId: edgeMeta.srcId,
            dstId: edgeMeta.dstId,
            routeMeta: edgeMeta.routeMeta || null,
            stroke: edgeMeta.stroke,
            strokeWidth: edgeMeta.strokeWidth,
            alpha: edgeMeta.alpha,
            dashed: edgeMeta.dashed,
            arrowAlpha: edgeMeta.arrowAlpha,
            // Legacy edge-record fields consumed by rebuildLegacyArraysFromPool
            // (from/to are the *original* endpoint ids; start/end are recomputed
            // there from the live boxes addressed by srcId/dstId).
            from: e.from,
            to: e.to,
            type: edgeMeta.type,
            colorKey: edgeMeta.colorKey,
            branch: edgeMeta.branch,
            arrow: edgeMeta.arrow
        };
    }

    // Recompute the flowchart layout for the *current* ``collapsedState`` and
    // flatten it into per-id absolute-coordinate meta maps.  This is the
    // coordinate source for the incremental snapshots.
    //
    // Why not index ``engine.nodes`` / ``engine.groups`` (as the obvious "reuse
    // the cache" idea suggests)?  Two source-verified reasons:
    //   1. The walkGroup pass only pushes the *load-time visible* set, so every
    //      node hidden inside an initially-collapsed group (default rule:
    //      ``depth >= 2``) is absent — Expand-All would have nothing to read and
    //      would throw for thousands of nodes.
    //   2. Coordinates shift on every collapse/expand (the DAG re-flows), so the
    //      load-time cache is stale the moment ``collapsedState`` changes.
    // Recomputing the pure layout (no draw / no resize / no resetScene) yields
    // correct, complete coordinates for whatever the next visible set will be.
    function computeLayoutMeta(data) {
        const computeLayout = requireInline('computeFlowchartLayout', lookupComputeFlowchartLayout());
        resetInlineLayoutCache();
        const layoutInfo = computeLayout(data, resolveContainerSize('computeVisibleScene').w);
        const layoutMap = lookupGroupLayout();
        if (!layoutMap) {
            throw new Error('render_canvas.js: computeVisibleScene requires groupLayout after computeFlowchartLayout');
        }
        const nodeMeta = new Map();
        const groupMeta = new Map();
        function walk(gid, ox, oy) {
            const pos = layoutMap[gid];
            if (!pos) {
                throw new Error('render_canvas.js: computeVisibleScene missing layout for group ' + gid);
            }
            groupMeta.set(String(gid), { x: ox, y: oy, w: pos.w, h: pos.h, collapsed: pos.collapsed === true });
            if (pos.collapsed) { return; }
            (pos.childPositions || []).forEach(function (child) {
                const cx = ox + child.x;
                const cy = oy + child.y;
                if (child.type === 'node') {
                    nodeMeta.set(String(child.id), { x: cx, y: cy, w: child.w, h: child.h });
                } else if (child.type === 'group') {
                    walk(child.id, cx, cy);
                } else {
                    throw new Error('render_canvas.js: computeVisibleScene unknown layout child type: ' + child.type);
                }
            });
        }
        (layoutInfo.rootPositions || []).forEach(function (root) {
            walk(root.id, root.x, root.y);
        });
        return { layoutInfo: layoutInfo, nodeMeta: nodeMeta, groupMeta: groupMeta };
    }

    function computeVisibleScene(layoutMeta) {
        ensureEngine();
        const data = engine.dataRef;
        if (!data) {
            throw new Error('render_canvas.js: computeVisibleScene called before __canvasSetIncrementalContext');
        }
        const collapsed = engine.collapsedStateRef;
        if (!collapsed) {
            throw new Error('render_canvas.js: computeVisibleScene requires collapsedStateRef to be wired');
        }
        const groupById = new Map();
        (data.groups || []).forEach(function (g) { groupById.set(String(g.id), g); });
        const nodeById = new Map();
        (data.nodes || []).forEach(function (n) { nodeById.set(String(n.id), n); });

        // Real, current-state coordinates for every node/group the visible walk
        // can possibly reach (see computeLayoutMeta's rationale).
        const metaBundle = layoutMeta || computeLayoutMeta(data);
        const nodeMetaById = metaBundle.nodeMeta;
        const groupMetaById = metaBundle.groupMeta;

        const groupIds = [];
        const nodeIds = [];
        const edgeKeys = [];
        const groupSnapshots = {};
        const nodeSnapshots = {};
        const edgeSnapshots = {};
        const visibleGroupSet = new Set();
        const visibleNodeSet = new Set();

        // Recursive top-down walk that mirrors the legacy ``walkGroup`` order and
        // records the visible group/node set plus their incremental snapshots.
        // It no longer registers any ports: edge endpoints are derived on demand
        // from the LIVE boxes (pool views + io pills) at patch time, so the walk
        // only needs to decide which groups/nodes are visible and stamp their
        // fresh-layout geometry into the snapshots.  A collapsed group stops the
        // recursion (its children are hidden); an expanded group recurses into
        // its child nodes and groups.
        function walk(gidRaw) {
            const gid = String(gidRaw);
            if (visibleGroupSet.has(gid)) { return; }
            const g = groupById.get(gid);
            if (!g) {
                throw new Error('render_canvas.js: computeVisibleScene found root_groups/children entry with no matching group: ' + gid);
            }
            const gmeta = groupMetaById.get(gid);
            requireFiniteXYWH('group', g.id, gmeta);
            visibleGroupSet.add(gid);
            groupIds.push(gid);
            groupSnapshots[gid] = makeIncrGroupSnapshot(g, collapsed, gmeta);
            if (collapsed[g.id] === true) {
                return;
            }
            (g.children_nodes || []).forEach(function (cnid) {
                const snid = String(cnid);
                if (visibleNodeSet.has(snid)) { return; }
                const node = nodeById.get(snid);
                if (!node) {
                    throw new Error('render_canvas.js: computeVisibleScene found children_nodes entry with no matching node: ' + snid);
                }
                const nmeta = nodeMetaById.get(snid);
                requireFiniteXYWH('node', node.id, nmeta);
                visibleNodeSet.add(snid);
                nodeIds.push(snid);
                nodeSnapshots[snid] = makeIncrNodeSnapshot(node, nmeta);
            });
            (g.children_group_ids || []).forEach(function (cgid) {
                walk(cgid);
            });
        }
        (data.root_groups || []).forEach(function (rid) {
            walk(rid);
        });

        // Global edges (ported from the legacy ``drawGlobalEdges`` path): each
        // endpoint is redirected to its outermost collapsed ancestor, and the
        // edge is dropped if both endpoints collapse onto the same box.  The
        // endpoint *ids* (``srcId`` / ``dstId``) are persisted on the snapshot;
        // the polyline is re-derived from the LIVE endpoint boxes in
        // patchEdgeView (and again in rebuildLegacyArraysFromPool), so there is
        // no ``nodePortMap`` dead-coord dictionary any more.
        //
        // The degenerate-span drop and the bundle routing offset still have to be
        // decided here (they govern which edges enter the visible set).  Both are
        // computed against the *fresh* per-id geometry: nodes/groups from the
        // freshly-recomputed layout metas, io pills from the already-live
        // ``engine.ioPillById`` index (drawIOTasks runs before computeVisibleScene
        // in invokeIncrementalRender).  This fresh geometry is byte-identical to
        // what the pool views hold after diffAndPatch, so the drop decision here
        // and the polyline drawn in patchEdgeView never disagree.
        function metaBoxForId(id) {
            const sid = String(id);
            const nm = nodeMetaById.get(sid);
            if (nm) { return nm; }
            const gm = groupMetaById.get(sid);
            if (gm) { return gm; }
            const pill = engine.ioPillById.get(sid);
            if (pill) { return { x: pill.x, y: pill.y, w: pill.w, h: pill.h }; }
            return null;
        }
        const isVisible = lookupIsEdgeVisible();
        const resolveAncestor = lookupResolveCollapsedAncestor();
        const edgeKeyFn = lookupEdgeKey();
        if (!isVisible || !resolveAncestor || !edgeKeyFn) {
            throw new Error('render_canvas.js: inline edge globals unavailable while routing global edges');
        }
        const bundleMeta = lookupEdgeBundleMeta();
        const seenEdgeKeys = new Set();
        (data.edges || []).forEach(function (e) {
            if (!isVisible(e)) { return; }
            const fromId = String(resolveAncestor(e.from));
            const toId = String(resolveAncestor(e.to));
            if (fromId === toId) { return; }
            const fromPort = portPointForEndpoint(fromId, 'src', metaBoxForId);
            const toPort = portPointForEndpoint(toId, 'dst', metaBoxForId);
            if (!fromPort || !toPort) {
                throw new Error('global edge endpoint missing: ' + e.from + ' -> ' + e.to);
            }
            const routeMeta = bundleMeta ? (bundleMeta.get(edgeKeyFn(e)) || null) : null;
            const route = EdgeRoute.compute('direct', fromPort.cx, fromPort.cy, toPort.cx, toPort.cy, routeMeta);
            // EdgeRoute.direct returns null for a degenerate span (|dy|<3 &&
            // |dx|<3); such edges are not drawable and are dropped (legacy parity).
            if (!route) { return; }
            const key = String(edgeKeyFn(e));
            if (!key || seenEdgeKeys.has(key)) { return; }
            seenEdgeKeys.add(key);
            const type = e.type || 'dep';
            const colorKey = colorKeyForType(type);
            const st = EDGE_STYLE[colorKey];
            edgeKeys.push(key);
            edgeSnapshots[key] = makeIncrEdgeSnapshot(e, {
                srcId: fromId,
                dstId: toId,
                routeMeta: routeMeta,
                dashed: route.dashed,
                stroke: st.color,
                strokeWidth: st.width,
                alpha: st.alpha,
                arrowAlpha: st.arrowAlpha,
                type: type,
                colorKey: colorKey,
                branch: route.branch,
                arrow: true
            });
        });

        return {
            groupIds: groupIds,
            nodeIds: nodeIds,
            edgeKeys: edgeKeys,
            groupSnapshots: groupSnapshots,
            nodeSnapshots: nodeSnapshots,
            edgeSnapshots: edgeSnapshots
        };
    }

    function invokeIncrementalRender() {
        ensureEngine();
        engine.incrementalRenderCount = (engine.incrementalRenderCount || 0) + 1;
        const prev = {
            groupIds: engine.visibleGroupIds,
            nodeIds: engine.visibleNodeIds,
            edgeKeys: engine.visibleEdgeKeys
        };
        const layoutMeta = computeLayoutMeta(engine.dataRef);
        applyWorldLayout(layoutMeta.layoutInfo);
        // Draw the IO pills BEFORE computeVisibleScene/diffAndPatch so the
        // ``engine.ioPillById`` index is live when edge routing (both the
        // degenerate-span drop in computeVisibleScene and patchEdgeView's polyline
        // derivation) resolves io-anchored endpoints from it.  io_pills /
        // ioPillById are rebuilt from scratch each frame.
        engine.layers.l5.removeChildren();
        engine.io_pills = [];
        engine.ioPillById.clear();
        drawIOTasks(layoutMeta.layoutInfo.ioTasks || []);
        const next = computeVisibleScene(layoutMeta);
        diffAndPatch(prev, next);
        performAutoFit();
        // Phase 2 step 4 invariant: this path still stays pool-first — it
        // reflows layout, refreshes canvas size, patches visible objects in
        // place, and then re-applies auto-fit without ever resetting the scene.
    }

    function applyWorldLayout(layoutInfo) {
        if (!layoutInfo || !Number.isFinite(layoutInfo.svgW) || !Number.isFinite(layoutInfo.svgH)) {
            throw new Error('render_canvas.js: computeFlowchartLayout returned invalid world bounds');
        }
        engine.viewport.worldWidth = layoutInfo.svgW;
        engine.viewport.worldHeight = layoutInfo.svgH;
        engine.worldBounds = { x: 0, y: 0, w: layoutInfo.svgW, h: layoutInfo.svgH };
        if (engine.app && engine.app.canvas) {
            engine.app.canvas.style.width = '100%';
        }
        applyViewport();
    }

    // Stage 1.5: fit the freshly-laid-out world into the visible container.
    // Required on the first render and after Expand/Collapse-All re-layouts.
    function performAutoFit(options) {
        const snapToTop = (options && options.snapToTop === true);
        if (!engine.worldBounds) {
            throw new Error('render_canvas.js: auto-fit requires worldBounds');
        }
        const fitBounds = computeRenderableContentBounds();
        engine.contentBounds = fitBounds;
        const containerSize = resolveContainerSize('auto-fit');
        const cw = containerSize.w;
        const ch = containerSize.h;
        const FIT_PADDING = 40;
        const vp = engine.viewportController.fitToView(fitBounds, cw, ch, { padding: FIT_PADDING, maxScale: 1.0 });
        // Width-only fit: the canvas width fills the container so there is no
        // horizontal scroll.  The height is grown to the full scaled content so a
        // graph taller than the viewport overflows into the .dag-stage's vertical
        // scroll (matches the legacy SVG semantics) instead of being squeezed.
        const contentHeight = Math.ceil(fitBounds.h * vp.scale + 2 * FIT_PADDING);
        const canvasHeight = Math.max(Math.ceil(ch), contentHeight);
        if (engine.app && engine.app.renderer && typeof engine.app.renderer.resize === 'function') {
            engine.rendererResizeCallCount = (engine.rendererResizeCallCount || 0) + 1;
            engine.app.renderer.resize(Math.ceil(cw), canvasHeight);
        }
        if (engine.app && engine.app.canvas) {
            engine.app.canvas.style.width = '100%';
        }
        // Apply the computed viewport transform to the world container.
        // renderer.resize() may reset internal transforms; applyViewport() must
        // be called AFTER resize so engine.world.x/y/scale reflect the fit result.
        applyViewport();
        // The width-only fit top-aligns the world at y = padding inside a canvas
        // that was just grown to the full (tall) content height.  Growing the
        // canvas turns the #dag-stage into a vertical scroll container; on some
        // browsers (Chrome scroll anchoring) the stage is left scrolled away from
        // the top when its content resizes, which hides the top-of-graph
        // Input/Const nodes on first load.  Explicitly snap the stage back to the
        // top so the freshly top-aligned fit is actually what the user sees.
        if (snapToTop) {
            if (engine.container && typeof engine.container.scrollTop !== 'undefined') {
                engine.container.scrollTop = 0;
            }
            // The top IO pills are laid out correctly (cy≈50), but page chrome above the
            // canvas can still leave them below the initial viewport on high-DPR setups.
            // Scroll to the document Y of the top-most IO pill (rather than merely page
            // top) so Input/Const are visible on first paint regardless of header height.
            if (global.window && global.document && typeof global.window.scrollTo === 'function') {
                const _win = global.window;
                const _doc = global.document;
                const snapToTopInput = function () {
                    const canvas = engine.app && engine.app.canvas;
                    if (!canvas || typeof canvas.getBoundingClientRect !== 'function') {
                        _win.scrollTo(0, 0);
                        return;
                    }
                    let currentY = numericOrNull(_win.scrollY);
                    if (currentY === null && _doc.documentElement) {
                        currentY = numericOrNull(_doc.documentElement.scrollTop);
                    }
                    if (currentY === null && _doc.body) {
                        currentY = numericOrNull(_doc.body.scrollTop);
                    }
                    if (currentY === null) {
                        currentY = 0;
                    }
                    let topPill = null;
                    (engine.io_pills || []).forEach(function (pill) {
                        if (!pill || !Number.isFinite(pill.cy)) {
                            return;
                        }
                        if (!topPill || pill.cy < topPill.cy) {
                            topPill = pill;
                        }
                    });
                    const rect = canvas.getBoundingClientRect();
                    const margin = 24;
                    let revealY = rect.top;
                    if (topPill && engine.viewport && Number.isFinite(engine.viewport.scale) && Number.isFinite(engine.viewport.y)) {
                        revealY += engine.viewport.y + topPill.cy * engine.viewport.scale;
                    }
                    const targetY = Math.max(0, Math.round(currentY + revealY - margin));
                    _win.scrollTo(0, targetY);
                };
                snapToTopInput();
                if (typeof _win.requestAnimationFrame === 'function') {
                    _win.requestAnimationFrame(function () {
                        _win.requestAnimationFrame(function () {
                            snapToTopInput();
                        });
                    });
                }
            }
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
        // Pool-first initial render needs computeVisibleScene() context. The
        // inline runtime normally installs it via __canvasSetIncrementalContext
        // before the first render, but headless probes call canvasRenderPhase1
        // directly with fresh ``data`` (and mutate the inline ``collapsedState``
        // global) without going through that hook. Establish the context here
        // from the render's own ``data`` argument + the live collapsedState so
        // the initial scene walk always has a valid, current reference.
        if (data && typeof data === 'object') {
            engine.dataRef = data;
            const liveCollapsed = lookupCollapsedState();
            if (liveCollapsed && typeof liveCollapsed === 'object') {
                engine.collapsedStateRef = liveCollapsed;
            }
        }
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
            // Phase 2 step 4: the initial render is now *pool-first* — there is
            // no resetScene() / legacy walkGroup draw anymore.  The node/group/
            // edge pools live on l3/l2/l1 (their roots are added once at create
            // time and reused), so we must NOT removeChildren() those layers or
            // we would orphan every pool root.  We only clear the IO layer (l5),
            // which is redrawn from scratch by drawIOTasks below, and reset the
            // legacy bookkeeping arrays so buildSnapshot()/auto-fit start clean.
            engine.layers.l5.removeChildren();
            engine.io_pills = [];
            engine.ioPillById.clear();
            engine.nodes = [];
            engine.groups = [];
            engine.edges = [];
            engine.labels = [];
            engine.labelsCreated = 0;
            engine.contentBounds = null;
            let layoutInfo = null;
            if (!await p.runChunked([{ type: 'group', taskKind: 'layout' }], async function () {
                resetInlineLayoutCache();
                layoutInfo = computeLayout(data, resolveContainerSize('layout').w);
            }, {
                batchSize: 1,
                phaseStart: 0,
                phaseEnd: 30,
                stageText: '正在计算 DAG 布局…',
                generation: generation,
                allowedTypes: ['group']
            })) {
                await p.hideRenderProgress();
                return;
            }
            if (!await p.runChunked([{ type: 'group', taskKind: 'draw-scene' }], async function () {
                applyWorldLayout(layoutInfo);
                drawIOTasks(layoutInfo.ioTasks || []);
            }, {
                batchSize: 1,
                phaseStart: 30,
                phaseEnd: 60,
                stageText: '正在渲染模块节点…',
                generation: generation,
                allowedTypes: ['group']
            })) {
                await p.hideRenderProgress();
                return;
            }
            if (!await p.runChunked([{ type: 'edge', taskKind: 'draw-edges' }], async function () {
                // Pool-first scene fill: diff the freshly computed visible scene
                // against the current visible sets and patch the pools.  On the
                // first render engine.visible* are empty Sets, so this is a full
                // create+patch; on a re-render it recycles unchanged views.
                const prev = {
                    groupIds: engine.visibleGroupIds,
                    nodeIds: engine.visibleNodeIds,
                    edgeKeys: engine.visibleEdgeKeys
                };
                const next = computeVisibleScene();
                diffAndPatch(prev, next);
            }, {
                batchSize: 1,
                phaseStart: 60,
                phaseEnd: 90,
                stageText: '正在渲染依赖边…',
                generation: generation,
                allowedTypes: ['edge']
            })) {
                await p.hideRenderProgress();
                return;
            }
            if (!p.assertActiveRenderGeneration(generation, '收尾阶段')) {
                await p.hideRenderProgress();
                return;
            }
            const wantAutoFit = (!engine.hasRenderedOnce) || (renderOpts && renderOpts.autoFit === true);
            if (wantAutoFit) {
                if (engine.app && engine.app.canvas) {
                    engine.app.canvas.style.width = '100%';
                }
                await requestAnimationFramePromise(function () {
                    return performAutoFit({ snapToTop: true });
                });
            }
            engine.hasRenderedOnce = true;
            p.setRenderProgress(98, '正在更新图例和摘要…');
            await p.nextFrame();
            updateLegendAndSummary(data);
            if (!p.assertActiveRenderGeneration(generation, '完成阶段')) {
                await p.hideRenderProgress();
                return;
            }
            // Phase 2 step 4: the object pool is now seeded by the pool-first
            // initial render itself (the ``draw-edges`` chunk above ran
            // ``diffAndPatch(prev, computeVisibleScene())`` with real,
            // current-state coordinates from ``computeLayoutMeta``).  Subsequent
            // toggles / Expand-All / Collapse-All route through
            // ``invokeIncrementalRender`` and diff against the previous visible
            // sets.  ``engine.dataRef`` / ``collapsedStateRef`` are installed by
            // the inline runtime via ``__canvasSetIncrementalContext`` before the
            // first render, so ``computeVisibleScene`` always has its context.
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

    // Reconstruct the legacy ``snapshot.ports`` map purely from the LIVE boxes
    // (pool node/group views, io pills) and the boundary-port index.  This is a
    // read-only introspection view computed on demand at snapshot time — it is
    // NOT the deleted ``nodePortMap`` dead-coord dictionary and is never read by
    // edge routing (edges resolve their endpoints through portPointForEndpoint).
    // Its key layout is byte-identical to the former registration:
    //   * node / io pill id  -> bare(center), id__in(top-mid), id__out(bottom-mid)
    //   * group id           -> gid__in, gid__out, plus gid__center when collapsed
    //   * boundary port node -> bare id == owning group's in/out port
    function buildPortsFromLive() {
        const ports = {};
        engine.nodePool.forEach(function (view) {
            if (view.visible !== true) { return; }
            const id = String(view.id);
            const box = { x: view.x, y: view.y, w: view.w, h: view.h };
            ports[id] = centerPortOf(box);
            ports[id + '__in'] = inPortOf(box);
            ports[id + '__out'] = outPortOf(box);
        });
        engine.groupPool.forEach(function (view) {
            if (view.visible !== true) { return; }
            const id = String(view.id);
            const box = { x: view.x, y: view.y, w: view.w, h: view.h };
            ports[id + '__in'] = inPortOf(box);
            ports[id + '__out'] = outPortOf(box);
            if (view.snapshot && view.snapshot.collapsed === true) {
                ports[id + '__center'] = centerPortOf(box);
            }
        });
        engine.ioPillById.forEach(function (pill, id) {
            const sid = String(id);
            const box = { x: pill.x, y: pill.y, w: pill.w, h: pill.h };
            ports[sid] = centerPortOf(box);
            ports[sid + '__in'] = inPortOf(box);
            ports[sid + '__out'] = outPortOf(box);
        });
        // Boundary port nodes redirect to their (visible) owning group's in/out
        // port — exactly the legacy bare ``nodePortMap[port.node_id]`` override.
        ensurePortNodeIndex().forEach(function (info, portNodeId) {
            const gbox = boxForId(info.groupId);
            if (!gbox) { return; }
            ports[String(portNodeId)] = info.kind === 'in' ? inPortOf(gbox) : outPortOf(gbox);
        });
        return ports;
    }

    function poolSnapshot() {
        const groups = [];
        engine.groupPool.forEach(function (view) {
            groups.push({
                id: view.id,
                visible: view.visible === true,
                interactionEnabled: view.interactionEnabled === true,
                eventMode: (view.box && view.box.eventMode) ? view.box.eventMode : null,
                rootVisible: view.root.visible === true,
                snapshot: view.snapshot
            });
        });
        const nodes = [];
        engine.nodePool.forEach(function (view) {
            nodes.push({
                id: view.id,
                visible: view.visible === true,
                rootVisible: view.root.visible === true,
                snapshot: view.snapshot
            });
        });
        const edges = [];
        engine.edgePool.forEach(function (view) {
            edges.push({
                key: view.key,
                visible: view.visible === true,
                rootVisible: view.root.visible === true,
                snapshot: view.snapshot
            });
        });
        return {
            groups: groups,
            nodes: nodes,
            edges: edges,
            visible: {
                groupIds: Array.from(engine.visibleGroupIds),
                nodeIds: Array.from(engine.visibleNodeIds),
                edgeKeys: Array.from(engine.visibleEdgeKeys)
            }
        };
    }

    // Pool-first reconstruction of the legacy ``engine.nodes`` / ``engine.groups``
    // / ``engine.edges`` arrays.  The initial render no longer walks the scene and
    // pushes into these arrays directly; instead the object pools hold the live
    // views.  This rebuilds the legacy-shaped arrays from the currently-visible
    // pool views so existing introspection (``buildSnapshot`` and the few tests
    // that read ``engine.groups`` directly) keeps working.  It is idempotent —
    // it overwrites the arrays from the pool every call.
    function rebuildLegacyArraysFromPool() {
        const nodes = [];
        engine.nodePool.forEach(function (view) {
            if (view.visible !== true || !view.snapshot) { return; }
            const s = view.snapshot;
            nodes.push({
                id: s.id, x: s.x, y: s.y, w: s.w, h: s.h,
                color: s.fill, label: s.label, sublabel: s.sublabel, visible: true
            });
        });
        const groups = [];
        engine.groupPool.forEach(function (view) {
            if (view.visible !== true || !view.snapshot) { return; }
            const s = view.snapshot;
            groups.push({
                id: s.id, collapsed: s.collapsed, x: s.x, y: s.y, w: s.w, h: s.h,
                has_header: !s.collapsed, has_info: s.hasInfo, has_timing: s.hasTiming
            });
        });
        const edges = [];
        engine.edgePool.forEach(function (view) {
            if (view.visible !== true || !view.snapshot) { return; }
            const s = view.snapshot;
            // Re-derive the endpoints from the LIVE boxes addressed by srcId /
            // dstId (pool views + io pills, with boundary port nodes redirected
            // onto their group) — the snapshot no longer carries baked start/end
            // coordinates.  A missing live endpoint is a hard error.
            const fromPort = portPointForEndpoint(s.srcId, 'src', boxForId);
            const toPort = portPointForEndpoint(s.dstId, 'dst', boxForId);
            if (!fromPort || !toPort) {
                throw new Error('render_canvas.js: rebuildLegacyArraysFromPool edge endpoint missing from live pools/io: ' + s.srcId + ' -> ' + s.dstId);
            }
            edges.push({
                from: s.from, to: s.to, type: s.type, colorKey: s.colorKey,
                start: { cx: fromPort.cx, cy: fromPort.cy },
                end: { cx: toPort.cx, cy: toPort.cy },
                branch: s.branch, dashed: s.dashed, arrow: s.arrow
            });
        });
        engine.nodes = nodes;
        engine.groups = groups;
        engine.edges = edges;
    }

    function buildSnapshot() {
        if (!engine) {
            throw new Error('render_canvas.js: __renderSnapshot called before the Canvas engine was initialized');
        }
        // Rebuild the legacy arrays from the live pools so the snapshot below
        // (and direct engine.nodes/groups/edges access) reflect the current scene.
        rebuildLegacyArraysFromPool();
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
            ports: buildPortsFromLive(),
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
            // Phase 2 step 1: pool + visible-set introspection.  Tests assert
            // that the three pools persist across diffAndPatch() calls and that
            // recycled entries stay in the pool with ``visible=false``.
            pool: poolSnapshot(),
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
    global.__canvasDiffAndPatch = function (prevVisible, nextVisible) {
        ensureEngine();
        diffAndPatch(prevVisible, nextVisible);
    };
    // Phase 2 step 4: incremental render entry points.  ``setIncrementalContext``
    // is called once by the inline runtime at startup; the inline ``invokeIncrementalRender``
    // / ``expandAll`` / ``collapseAll`` helpers route through ``invokeIncrementalRender``
    // here, which never touches ``resetScene()`` / ``renderer.resize()``.
    global.__canvasSetIncrementalContext = function (ctx) {
        setIncrementalContext(ctx);
    };
    global.__canvasComputeVisibleScene = function () {
        ensureEngine();
        return computeVisibleScene();
    };
    global.__canvasInvokeIncrementalRender = function () {
        ensureEngine();
        invokeIncrementalRender();
    };
    // Phase 2 step 3: ``__canvasGetGroupBox`` returns the live group hit box
    // (Graphics) for a given gid, regardless of whether the legacy walkGroup
    // path or the pool path drew it.  Tests use this to simulate click /
    // dblclick on the actual on-screen box; the inline runtime does not need
    // it (the wiring runs through ``engine.onGroupToggle`` / ``onGroupSelect``).
    global.__canvasGetGroupBox = function (gid) {
        ensureEngine();
        return engine.groupBoxes.get(String(gid)) || null;
    };
    global.__EdgeRoute = EdgeRoute;
    global.__EDGE_STYLE = EDGE_STYLE;
    global.__renderSnapshot = function () {
        ensureEngine();
        return buildSnapshot();
    };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

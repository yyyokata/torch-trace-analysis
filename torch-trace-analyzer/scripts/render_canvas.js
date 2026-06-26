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
        // Mirror PixiJS v8 ``DisplayObject.destroy``: detach from the parent and
        // (with ``{children:true}``) recursively destroy children.  The headless
        // mock needs this so the dataset-switch stale-pool teardown (problem 4)
        // runs identically under node and the real WebGL renderer.
        Container.prototype.destroy = function (opts) {
            const destroyChildren = !!(opts && opts.children);
            if (this.parent && Array.isArray(this.parent.children)) {
                this.parent.children = this.parent.children.filter((c) => c !== this);
            }
            this.parent = null;
            if (destroyChildren) {
                const kids = this.children.slice();
                kids.forEach(function (c) { if (c && typeof c.destroy === 'function') { c.destroy(opts); } });
            }
            this.children = [];
            this.destroyed = true;
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
                _lastResizeW: null,
                _lastResizeH: null,
                resize: function (w, h) {
                    this.width = w;
                    this.height = h;
                    this._lastResizeW = w;
                    this._lastResizeH = h;
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
            // Problem 4: the payload the pools were last built from.  A render
            // whose ``data`` differs from this is a dataset switch and triggers
            // the stale-pool teardown in canvasRenderPhase1.
            renderedDataRef: null,
            collapsedStateRef: null,
            rendererResizeCallCount: 0,
            // Phase 2 step 4: positive diagnostic counter bumped once per
            // ``invokeIncrementalRender`` (toggle / Expand-All / Collapse-All).
            // The regression tests assert this advances while
            // ``rendererResizeCallCount`` stays put — proving the pool-first
            // incremental path ran instead of a full re-render.
            incrementalRenderCount: 0,
            isIncrementalPatching: false
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
        // Leaf node single click → show the source/evidence panel.  Forwarded to
        // the inline-runtime ``window.__canvasOnNodeSelect`` at call time (same
        // late-binding contract as ``onGroupSelect``).  No silent fallback: a
        // missing global throws hard.
        built.onNodeSelect = function (nid) {
            if (typeof global.__canvasOnNodeSelect !== 'function') {
                throw new Error('render_canvas.js: window.__canvasOnNodeSelect is not wired by inline runtime');
            }
            global.__canvasOnNodeSelect(nid);
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
    // Phase 2 step 5 — Semantic Zoom: ``focusStack`` is the inline runtime's
    // global focus stack (declared in frontend_html.py).  Lazy lookup mirrors
    // the collapsedState pattern so headless probes that drive
    // ``canvasRenderPhase1`` directly (without calling
    // ``__canvasSetIncrementalContext``) still find a valid reference.
    function lookupFocusStack() { return (typeof focusStack !== 'undefined') ? focusStack : null; }
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

    function eventButton(e) {
        if (!e) { return null; }
        if (typeof e.button === 'number') { return e.button; }
        return null;
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
            const x = numericOrNull(pill.x);
            const y = numericOrNull(pill.y);
            const w = numericOrNull(pill.w);
            const h = numericOrNull(pill.h);
            if (x === null || y === null || w === null || h === null) {
                throw new Error('render_canvas.js: io pill bounds must expose finite x/y/w/h');
            }
            bounds = expandBounds(bounds, { x: x, y: y, w: w, h: h });
        });
        if (!bounds) {
            throw new Error('render_canvas.js: auto-fit requires rendered content bounds');
        }
        return bounds;
    }

    // Step 5 fit (problem 2/4): bounds of the focus subtree only.  ``idSet`` is
    // ``engine.lastFocusSubtreeIds`` — the drilled-in group/node ids EXCLUDING
    // the one-hop boundary.  Returns null when the set is empty or none of its
    // members are currently rendered, so the caller falls back to the full
    // renderable bounds.  IO pills are intentionally excluded: they sit on a
    // group's edge and the 40px fit padding already covers them.
    function computeFocusSubtreeBounds(idSet) {
        if (!idSet || typeof idSet.forEach !== 'function') { return null; }
        let bounds = null;
        idSet.forEach(function (id) {
            const key = String(id);
            const gv = engine.groupPool.get(key);
            if (gv && gv.visible === true && gv.snapshot) {
                const s = gv.snapshot;
                bounds = expandBounds(bounds, { x: s.x, y: s.y, w: s.w, h: s.h });
                return;
            }
            const nv = engine.nodePool.get(key);
            if (nv && nv.visible === true && nv.snapshot) {
                const s = nv.snapshot;
                bounds = expandBounds(bounds, { x: s.x, y: s.y, w: s.w, h: s.h });
            }
        });
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
        if (engine.isIncrementalPatching === true) {
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
    function suppressPointerDefault(e) {
        if (e && typeof e.stopPropagation === 'function') { e.stopPropagation(); }
        if (e && e.nativeEvent && typeof e.nativeEvent.preventDefault === 'function') { e.nativeEvent.preventDefault(); }
        if (e && typeof e.preventDefault === 'function') { e.preventDefault(); }
    }

    function bindPointerGestures(target, handlers) {
        if (!target) {
            throw new Error('render_canvas.js: bindPointerGestures missing target');
        }
        if (!handlers || typeof handlers !== 'object') {
            throw new Error('render_canvas.js: bindPointerGestures requires handlers object');
        }
        const leftClick = handlers.leftClick;
        const leftDblClick = handlers.leftDblClick;
        const rightClick = handlers.rightClick;
        const rightDblClick = handlers.rightDblClick;
        if (leftClick !== null && leftClick !== undefined && typeof leftClick !== 'function') {
            throw new Error('render_canvas.js: bindPointerGestures leftClick must be function/null');
        }
        if (leftDblClick !== null && leftDblClick !== undefined && typeof leftDblClick !== 'function') {
            throw new Error('render_canvas.js: bindPointerGestures leftDblClick must be function/null');
        }
        if (rightClick !== null && rightClick !== undefined && typeof rightClick !== 'function') {
            throw new Error('render_canvas.js: bindPointerGestures rightClick must be function/null');
        }
        if (rightDblClick !== null && rightDblClick !== undefined && typeof rightDblClick !== 'function') {
            throw new Error('render_canvas.js: bindPointerGestures rightDblClick must be function/null');
        }
        const leftDelay = numericOrNull(handlers.leftDblClickDelayMs);
        const leftDblClickDelayMs = (leftDelay !== null && leftDelay > 0) ? leftDelay : 200;
        const rightDelay = numericOrNull(handlers.rightDblClickDelayMs);
        const rightDblClickDelayMs = (rightDelay !== null && rightDelay > 0) ? rightDelay : 250;
        target.eventMode = 'static';
        target.cursor = 'pointer';
        const state = { leftClickTimer: null, leftLastClickTime: 0, rightLastDown: 0, lastRightClickAt: 0 };
        target.on('click', function (e) {
            const btn = eventButton(e);
            if (btn !== null && btn !== 0) { return; }
            // Pixi v8 fires an extra `click` (button=0) on right-mouse-up right after
            // the `rightclick` event.  Guard against it: if a rightclick fired within
            // the last 120ms, this click is a Pixi artefact — skip it.
            const nowMs = (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
            if (state.lastRightClickAt > 0 && (nowMs - state.lastRightClickAt) < 120) { return; }
            suppressPointerDefault(e);
            const now = (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
            const isDoubleClick = state.leftLastClickTime > 0 && (now - state.leftLastClickTime) <= leftDblClickDelayMs;
            state.leftLastClickTime = now;
            if (isDoubleClick) {
                if (state.leftClickTimer !== null) {
                    clearTimeout(state.leftClickTimer);
                    state.leftClickTimer = null;
                }
                if (typeof leftDblClick === 'function') { leftDblClick(e); }
                return;
            }
            if (state.leftClickTimer !== null) {
                clearTimeout(state.leftClickTimer);
                state.leftClickTimer = null;
            }
            state.leftClickTimer = setTimeout(function () {
                state.leftClickTimer = null;
                if (typeof leftClick === 'function') { leftClick(e); }
            }, leftDblClickDelayMs);
        });
        target.on('rightclick', function (e) {
            suppressPointerDefault(e);
            const now = (typeof Date !== 'undefined' && typeof Date.now === 'function') ? Date.now() : 0;
            // Always stamp lastRightClickAt so the click-guard above can suppress the
            // artefact `click(button=0)` that Pixi v8 fires on right-mouse-up.
            state.lastRightClickAt = now;
            if (now - state.rightLastDown < rightDblClickDelayMs) {
                state.rightLastDown = 0;
                if (typeof rightDblClick === 'function') { rightDblClick(e); }
            } else {
                state.rightLastDown = now;
                if (typeof rightClick === 'function') { rightClick(e); }
            }
        });
        return state;
    }

    function bindGroupBox(gid, box) {
        if (!box) {
            throw new Error('render_canvas.js: bindGroupBox missing box for group ' + gid);
        }
        if (box.__phase2EventsBound) { return; }
        box.__phase2EventsBound = true;
        bindPointerGestures(box, {
            // Left single click → show the source/evidence panel.
            // Left double click → collapse/expand the group.
            // Right click carries NO business logic any more: Semantic Zoom is
            // entered via the data-panel "Focus" button and exited via ESC /
            // breadcrumb / the stage-level contextmenu guard.  The ``rightclick``
            // handler inside bindPointerGestures still fires to suppress the
            // browser context menu and to swallow the artefact left-``click``
            // Pixi v8 emits on right-mouse-up — but it invokes no callback here.
            leftClick: function (e) { void e; engine.onGroupSelect(gid); },
            leftDblClick: function (e) { void e; engine.onGroupToggle(gid); },
            rightClick: null,
            rightDblClick: null
        });
    }

    // Phase 2: bind a *single* left-click handler to a leaf node hit box so a
    // click opens the source/evidence panel.  Idempotent (second call on the
    // same box is a no-op).  Nodes have no double-click / right-click gesture —
    // only ``onNodeSelect``.  Mirrors ``bindGroupBox`` but with just leftClick.
    function bindNodeBox(nid, box) {
        if (!box) {
            throw new Error('render_canvas.js: bindNodeBox missing box for node ' + nid);
        }
        if (box.__phase2NodeEventsBound) { return; }
        box.__phase2NodeEventsBound = true;
        bindPointerGestures(box, {
            leftClick: function (e) { void e; engine.onNodeSelect(nid); },
            leftDblClick: null,
            rightClick: null,
            rightDblClick: null
        });
    }

    function toggleIOGroup(ioGroupId) {
        ensureEngine();
        if (!engine.dataRef || !Array.isArray(engine.dataRef.io_groups)) {
            throw new Error('render_canvas.js: IO group toggle requires dataRef.io_groups');
        }
        const sid = String(ioGroupId);
        const found = engine.dataRef.io_groups.some(function (g) { return g && String(g.id) === sid; });
        if (!found) {
            throw new Error('render_canvas.js: unknown IO group id ' + ioGroupId);
        }
        if (!engine.collapsedStateRef || typeof engine.collapsedStateRef !== 'object') {
            throw new Error('render_canvas.js: IO group toggle requires collapsedStateRef');
        }
        engine.collapsedStateRef[ioGroupId] = !engine.collapsedStateRef[ioGroupId];
        invokeIncrementalRender();
    }

    function bindIOGroupToggle(ioGroupId, target) {
        if (!target) {
            throw new Error('render_canvas.js: bindIOGroupToggle missing target for io group ' + ioGroupId);
        }
        if (target.__ioGroupToggleBound) { return; }
        target.__ioGroupToggleBound = true;
        bindPointerGestures(target, {
            leftClick: null,
            leftDblClick: function () {
                toggleIOGroup(ioGroupId);
            },
            rightClick: null,
            rightDblClick: null
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
    function outPortOf(box) { return { cx: box.x + box.w / 2, cy: box.y + box.h }; }
    function inPortOf(box) { return { cx: box.x + box.w / 2, cy: box.y }; }

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

    // ── Semantic-Zoom boundary anchor resolution (problem 1) ───────────────
    // ``buildBoundaryAnchorCtx`` precomputes the id → record lookups
    // ``resolveBoundaryAnchor`` needs.  Building them once per call site (instead
    // of per edge) keeps the focus boundary classification O(E) rather than
    // O(E·(G+N+IO)).  All four lookups are keyed by *string* ids.
    function buildBoundaryAnchorCtx(data) {
        if (!data || typeof data !== 'object') {
            throw new Error('render_canvas.js: buildBoundaryAnchorCtx requires data');
        }
        const groupById = new Map();
        (data.groups || []).forEach(function (g) { groupById.set(String(g.id), g); });
        const nodeById = new Map();
        (data.nodes || []).forEach(function (n) { nodeById.set(String(n.id), n); });
        const ioById = new Map();
        const ioByMember = new Map();
        (data.io_groups || []).forEach(function (ig) {
            ioById.set(String(ig.id), ig);
            (ig.member_ids || []).forEach(function (mid) { ioByMember.set(String(mid), ig); });
        });
        const resolveAncestor = lookupResolveCollapsedAncestor();
        if (typeof resolveAncestor !== 'function') {
            throw new Error('render_canvas.js: resolveBoundaryAnchor requires resolveCollapsedAncestor');
        }
        return {
            groupById: groupById,
            nodeById: nodeById,
            ioById: ioById,
            ioByMember: ioByMember,
            portIndex: ensurePortNodeIndex(),
            resolveAncestor: resolveAncestor
        };
    }

    // Resolve a raw cross-boundary endpoint id to a stable *anchor* describing
    // how it should be drawn in focus mode.  Order matters and is the single
    // source of truth shared by ``augmentFocusBoundaryMeta`` (which lays the
    // boundary card / io pill out) and ``computeVisibleScene.addBoundary`` (which
    // admits it to the visible set):
    //   1. resolveCollapsedAncestor(rawId) → resolved id
    //   2. resolved is a Group in/out port node id → its owning group (kind=group)
    //   3. resolved is an io_group id            → that io_group  (kind=io)
    //   4. resolved is an io_group member node id → its owning io_group (kind=io)
    //   5. resolved is a data.groups id          → the group      (kind=group)
    //   6. resolved is a data.nodes id           → the node       (kind=node)
    //   7. otherwise → hard error (fail-fast, no silent drop)
    function resolveBoundaryAnchor(rawId, ctx) {
        const resolved = String(ctx.resolveAncestor(rawId));
        const portInfo = ctx.portIndex.get(resolved);
        if (portInfo) {
            return { kind: 'group', id: String(portInfo.groupId) };
        }
        if (ctx.ioById.has(resolved)) {
            return { kind: 'io', id: resolved, ioGroup: ctx.ioById.get(resolved) };
        }
        const ownerIo = ctx.ioByMember.get(resolved);
        if (ownerIo) {
            return { kind: 'io', id: String(ownerIo.id), ioGroup: ownerIo };
        }
        if (ctx.groupById.has(resolved)) {
            return { kind: 'group', id: resolved };
        }
        if (ctx.nodeById.has(resolved)) {
            return { kind: 'node', id: resolved };
        }
        throw new Error('render_canvas.js: resolveBoundaryAnchor could not resolve boundary id ' + rawId + ' (resolved=' + resolved + ')');
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
            x: spec.cx - spec.w / 2,
            y: spec.cy - spec.h / 2,
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
            bindIOGroupToggle(ioGroup.id, frame);
            engine.ioPillById.set(String(ioGroup.id), {
                x: frameX, y: frameY, w: frameW, h: frameH
            });
            engine.io_pills.push({
                id: ioGroup.id,
                subtype: ioGroup.io_subtype,
                x: frameX,
                y: frameY,
                cx: frameX + frameW / 2,
                cy: frameY + frameH / 2,
                w: frameW,
                h: frameH,
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
        bindIOGroupToggle(ioGroup.id, groupPill);
        addLabel(this.layer, '\u25B6 ' + ioGroup.label, 'io-group-label:' + ioGroup.id,
            geom.cx, geom.cy, TEXT_STYLE.ioTitle, { ax: 0.5, ay: 0.5 });
        engine.ioPillById.set(String(ioGroup.id), {
            x: geom.cx - geom.w / 2, y: geom.cy - geom.h / 2, w: geom.w, h: geom.h
        });
        engine.io_pills.push({
            id: ioGroup.id,
            subtype: ioGroup.io_subtype,
            x: geom.cx - geom.w / 2,
            y: geom.cy - geom.h / 2,
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
    const GROUP_SNAPSHOT_FIELDS = ['x', 'y', 'w', 'h', 'label', 'collapsed', 'fill', 'alpha', 'hasTiming', 'timingText'];
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
        // Phase 2: leaf nodes are clickable — a left click opens the
        // source/evidence panel via ``engine.onNodeSelect``.  ``bindNodeBox``
        // flips ``eventMode='static'`` and idempotently registers the listener.
        box.eventMode = 'static';
        root.addChild(box);
        engine.layers.l3.addChild(root);
        bindNodeBox(id, box);
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
        // Header / timing glyphs are culled (the box chrome is always drawn).
        // Mirrors the legacy draw path which always painted the container chrome
        // but only created the text labels for on-screen groups.
        const showLabels = shouldCreateLabel({ x: snapshot.x, y: snapshot.y, w: snapshot.w, h: snapshot.h });
        // header text (lazily created, parented to view.root)
        const arrow = snapshot.collapsed ? '\u25B6 ' : '\u25BC ';
        const rightReserve = snapshot.collapsed
            ? 10
            : (10 + (snapshot.hasTiming ? 130 : 0));
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

    function _assertDiffInputs(prevVisible, nextVisible) {
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
    }

    function _commitVisibleSets(nextVisible) {
        engine.visibleGroupIds = new Set((nextVisible.groupIds || []).map(String));
        engine.visibleNodeIds  = new Set((nextVisible.nodeIds  || []).map(String));
        engine.visibleEdgeKeys = new Set((nextVisible.edgeKeys || []).map(String));
    }

    function diffAndPatch(prevVisible, nextVisible) {
        _assertDiffInputs(prevVisible, nextVisible);
        patchGroups(prevVisible.groupIds, nextVisible.groupIds, nextVisible.groupSnapshots);
        patchNodes(prevVisible.nodeIds, nextVisible.nodeIds, nextVisible.nodeSnapshots);
        patchEdges(prevVisible.edgeKeys, nextVisible.edgeKeys, nextVisible.edgeSnapshots);
        _commitVisibleSets(nextVisible);
    }

    // Frame-chunked variant of ``diffAndPatch`` (problem 4): the dataset-switch /
    // initial render path creates *all* new Graphics in one go, which makes
    // PixiJS flush the entire geometry buffer to the GPU in a single tick
    // (~1 s ``bufferSubData`` for the 5698781 model — confirmed by the Chrome
    // trace).  Yielding a frame between patchGroups → patchNodes → patchEdges
    // lets PixiJS upload each slice in its own tick, spreading the GPU cost over
    // three frames instead of one.  Snapshot validation still happens up front so
    // a malformed frame aborts before any pool mutation.  ``p`` is the progress
    // API (its ``nextFrame`` yields to the event loop / rAF).
    async function diffAndPatchChunked(prevVisible, nextVisible, p) {
        if (!p || typeof p.nextFrame !== 'function') {
            throw new Error('render_canvas.js: diffAndPatchChunked requires a progress api with nextFrame()');
        }
        _assertDiffInputs(prevVisible, nextVisible);
        patchGroups(prevVisible.groupIds, nextVisible.groupIds, nextVisible.groupSnapshots);
        await p.nextFrame();
        patchNodes(prevVisible.nodeIds, nextVisible.nodeIds, nextVisible.nodeSnapshots);
        await p.nextFrame();
        patchEdges(prevVisible.edgeKeys, nextVisible.edgeKeys, nextVisible.edgeSnapshots);
        _commitVisibleSets(nextVisible);
    }

    // Destroy a single pool entry: tear down its Pixi display object (freeing the
    // GPU geometry/texture it holds) and drop it from the pool Map and the live
    // visible-id set so the next diffAndPatch never tries to recycle a destroyed
    // view.  ``view.root.destroy({children:true})`` releases the whole subtree.
    function destroyPoolEntry(pool, key, visibleSet) {
        const view = pool.get(key);
        if (!view) { return false; }
        if (!view.root || typeof view.root.destroy !== 'function') {
            throw new Error('render_canvas.js: destroyPoolEntry pool view ' + key + ' has no destroyable root');
        }
        view.root.destroy({ children: true });
        pool.delete(key);
        if (visibleSet instanceof Set) { visibleSet.delete(String(key)); }
        return true;
    }

    // Dataset-switch teardown (problem 4): when the page swaps to a different
    // dataset (e.g. Training → Inference) the new graph's node/group/edge ids
    // barely overlap the old one's, so the object pools would otherwise keep
    // every stale view alive forever — holding their VRAM geometry and inflating
    // the pool the diff has to walk.  Destroy every pool view whose key is absent
    // from the new dataset and prune it from the live visible-id sets.  Returns
    // the number of views destroyed (used by tests / diagnostics).
    function destroyStalePoolViews(data) {
        if (!data || typeof data !== 'object') {
            throw new Error('render_canvas.js: destroyStalePoolViews requires data');
        }
        const validGroupIds = new Set((data.groups || []).map(function (g) { return String(g.id); }));
        const validNodeIds = new Set((data.nodes || []).map(function (n) { return String(n.id); }));
        const edgeKeyFn = lookupEdgeKey();
        if (typeof edgeKeyFn !== 'function') {
            throw new Error('render_canvas.js: destroyStalePoolViews requires the inline edgeKey() helper');
        }
        const validEdgeKeys = new Set();
        (data.edges || []).forEach(function (e) { validEdgeKeys.add(String(edgeKeyFn(e))); });
        let destroyed = 0;
        Array.from(engine.groupPool.keys()).forEach(function (key) {
            if (!validGroupIds.has(String(key))) {
                if (destroyPoolEntry(engine.groupPool, key, engine.visibleGroupIds)) { destroyed++; }
            }
        });
        Array.from(engine.nodePool.keys()).forEach(function (key) {
            if (!validNodeIds.has(String(key))) {
                if (destroyPoolEntry(engine.nodePool, key, engine.visibleNodeIds)) { destroyed++; }
            }
        });
        Array.from(engine.edgePool.keys()).forEach(function (key) {
            if (!validEdgeKeys.has(String(key))) {
                if (destroyPoolEntry(engine.edgePool, key, engine.visibleEdgeKeys)) { destroyed++; }
            }
        });
        return destroyed;
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
        // Step 5: ``focusStack`` is the inline runtime's array of focus root ids
        // (see design/frontend_canvas_phase2.md §6.3).  It is a HARD requirement
        // — Step 5 has shipped and there is no global-mode-only path any more.
        if (!Array.isArray(ctx.focusStack)) {
            throw new Error('render_canvas.js: __canvasSetIncrementalContext: ctx.focusStack must be an array');
        }
        engine.dataRef = ctx.data;
        engine.collapsedStateRef = ctx.collapsedState;
        engine.focusStackRef = ctx.focusStack;
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
        return {
            id: g.id,
            x: groupMeta.x, y: groupMeta.y, w: groupMeta.w, h: groupMeta.h,
            label: String(g.class_name),
            collapsed: collapsed[g.id] === true,
            fill: groupColor,
            alpha: 1.0,
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
        const focusStack = engine.focusStackRef;
        const focusActive = Array.isArray(focusStack) && focusStack.length > 0;
        const focusRootId = focusActive ? String(focusStack[focusStack.length - 1]) : null;
        resetInlineLayoutCache();
        const containerW = resolveContainerSize('computeVisibleScene').w;
        // Step 5 — Semantic Zoom subgraph layout (problem 3): in focus mode we do
        // NOT reuse the full-graph coordinates (which would leave the drilled-in
        // subtree at its original far-away position and fling the one-hop boundary
        // off-screen).  Instead we re-run the layout engine on a reduced graph
        // rooted at the focus group alone, producing compact origin-based coords,
        // then place the one-hop boundary cards in flanking rows (see
        // ``augmentFocusBoundaryMeta``).
        const layoutData = focusActive
            ? { root_groups: [focusRootId], io_groups: [], input_node_ids: [], param_node_ids: [], const_node_ids: [], output_node_ids: [] }
            : data;
        const layoutInfo = computeLayout(layoutData, containerW);
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
        if (focusActive) {
            augmentFocusBoundaryMeta(data, focusRootId, layoutInfo, nodeMeta, groupMeta);
        }
        return { layoutInfo: layoutInfo, nodeMeta: nodeMeta, groupMeta: groupMeta };
    }

    // ``augmentFocusBoundaryMeta`` positions the one-hop boundary cards around the
    // freshly re-laid-out focus subtree.  ``nodeMeta`` / ``groupMeta`` already hold
    // the compact subtree boxes (only the focus subtree was laid out).  We classify
    // every visible cross-boundary edge into IN-neighbours (feed the subtree, placed
    // in a row above) and OUT-neighbours (consume from the subtree, placed in a row
    // below), give each a card-sized box, and grow ``layoutInfo.svgW/svgH`` so the
    // world bounds (and hence the auto-fit) cover the whole local closure.
    //
    // Problem 1: cross-boundary endpoints that resolve to a model IO group
    // (input / param / const / output) are now drawn too.  They are placed in the
    // SAME flanking IN / OUT rows as the regular boundary cards, and each emits an
    // ``io_pill`` draw task appended to ``layoutInfo.ioTasks`` so ``drawIOTasks``
    // (which runs before ``computeVisibleScene``) registers the pill in
    // ``engine.ioPillById`` — giving ``addBoundary`` / edge routing a live box.
    function augmentFocusBoundaryMeta(data, focusRootId, layoutInfo, nodeMeta, groupMeta) {
        const portIndex = ensurePortNodeIndex();
        const subtreeIds = new Set();
        groupMeta.forEach(function (_v, k) { subtreeIds.add(k); });
        nodeMeta.forEach(function (_v, k) { subtreeIds.add(k); });
        function insideFocus(id) {
            const sid = String(id);
            if (subtreeIds.has(sid)) { return true; }
            const portInfo = portIndex.get(sid);
            return !!(portInfo && subtreeIds.has(String(portInfo.groupId)));
        }
        const resolveAncestor = lookupResolveCollapsedAncestor();
        const isVisible = lookupIsEdgeVisible();
        if (typeof resolveAncestor !== 'function' || typeof isVisible !== 'function') {
            throw new Error('render_canvas.js: augmentFocusBoundaryMeta requires resolveCollapsedAncestor + isEdgeVisible');
        }
        const boundaryCtx = buildBoundaryAnchorCtx(data);
        const inNeighbors = [];
        const outNeighbors = [];
        const seen = new Set();
        function classify(rawId, isSource) {
            const anchor = resolveBoundaryAnchor(rawId, boundaryCtx);
            // ``kind === 'io'`` keys on the io_group id; group / node anchors key
            // on the resolved card id (and must still sit outside the subtree).
            if (anchor.kind !== 'io' && insideFocus(anchor.id)) { return; }
            if (seen.has(anchor.id)) { return; }
            seen.add(anchor.id);
            const entry = (anchor.kind === 'io')
                ? { kind: 'io', id: anchor.id, ioGroup: anchor.ioGroup }
                : { kind: anchor.kind, id: anchor.id };
            if (isSource) { inNeighbors.push(entry); } else { outNeighbors.push(entry); }
        }
        (data.edges || []).forEach(function (e) {
            if (!isVisible(e)) { return; }
            const fromId = String(resolveAncestor(e.from));
            const toId = String(resolveAncestor(e.to));
            if (fromId === toId) { return; }
            const fromIn = insideFocus(fromId);
            const toIn = insideFocus(toId);
            if (fromIn === toIn) { return; }
            if (!fromIn) { classify(e.from, true); }   // outside source feeds the subtree
            if (!toIn) { classify(e.to, false); }      // outside dest consumes from the subtree
        });
        if (inNeighbors.length === 0 && outNeighbors.length === 0) { return; }

        const GAP = 60;
        const COL_GAP = 24;
        function boxFor(entry) {
            if (entry.kind === 'group') { return { w: LAYOUT.nodeW + 20, h: LAYOUT.nodeH + 8 }; }
            // node + io boundary entries both use a node-sized card/pill.
            return { w: LAYOUT.nodeW, h: LAYOUT.nodeH };
        }
        if (!Array.isArray(layoutInfo.ioTasks)) { layoutInfo.ioTasks = []; }
        function emitIoTask(entry, cx, cy, w, h) {
            const ig = entry.ioGroup;
            const subtype = ig.io_subtype;
            layoutInfo.ioTasks.push({
                type: 'io',
                taskKind: 'io_pill',
                nid: ig.id,
                subtype: subtype,
                cx: cx,
                cy: cy,
                w: w,
                h: h,
                label: ig.label || IO_GROUP_MEMBER_LABEL[subtype] || String(subtype),
                sublabel: '',
                fillColor: IO_GROUP_FILL[subtype] || 'rgba(127,140,141,0.55)'
            });
        }
        // Subtree bounding box (pre-offset).
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        function expand(m) {
            minX = Math.min(minX, m.x); minY = Math.min(minY, m.y);
            maxX = Math.max(maxX, m.x + m.w); maxY = Math.max(maxY, m.y + m.h);
        }
        groupMeta.forEach(expand);
        nodeMeta.forEach(expand);
        const inH = inNeighbors.length ? Math.max.apply(null, inNeighbors.map(function (e) { return boxFor(e).h; })) : 0;
        const topPad = inNeighbors.length ? (inH + GAP) : 0;
        // Shift the subtree down to make room for the IN row.
        if (topPad > 0) {
            groupMeta.forEach(function (m) { m.y += topPad; });
            nodeMeta.forEach(function (m) { m.y += topPad; });
            minY += topPad; maxY += topPad;
        }
        const centerX = (minX + maxX) / 2;
        function placeRow(neighbors, rowY) {
            const boxes = neighbors.map(boxFor);
            let total = 0;
            boxes.forEach(function (b, i) { total += b.w + (i > 0 ? COL_GAP : 0); });
            let x = centerX - total / 2;
            let rowMaxBottom = rowY;
            neighbors.forEach(function (entry, i) {
                const b = boxes[i];
                if (entry.kind === 'group') {
                    groupMeta.set(entry.id, { x: x, y: rowY, w: b.w, h: b.h, collapsed: true });
                } else if (entry.kind === 'node') {
                    nodeMeta.set(entry.id, { x: x, y: rowY, w: b.w, h: b.h });
                } else {
                    // io boundary: emit an io_pill draw task (cx/cy are centre).
                    emitIoTask(entry, x + b.w / 2, rowY + b.h / 2, b.w, b.h);
                }
                rowMaxBottom = Math.max(rowMaxBottom, rowY + b.h);
                x += b.w + COL_GAP;
            });
            return { left: centerX - total / 2, right: centerX - total / 2 + total, bottom: rowMaxBottom };
        }
        let worldMinX = minX, worldMaxX = maxX, worldMaxY = maxY;
        if (inNeighbors.length) {
            const r = placeRow(inNeighbors, (topPad - inH) / 2);
            worldMinX = Math.min(worldMinX, r.left); worldMaxX = Math.max(worldMaxX, r.right);
        }
        if (outNeighbors.length) {
            const r = placeRow(outNeighbors, maxY + GAP);
            worldMinX = Math.min(worldMinX, r.left); worldMaxX = Math.max(worldMaxX, r.right);
            worldMaxY = Math.max(worldMaxY, r.bottom);
        }
        // Grow the reported world so applyWorldLayout / auto-fit cover the boundary
        // rows.  computeFlowchartLayout anchors content at x>=0; if a wide boundary
        // row would spill left of 0 we keep svgW big enough to contain it.
        layoutInfo.svgW = Math.max(layoutInfo.svgW, worldMaxX, worldMaxX - Math.min(0, worldMinX));
        layoutInfo.svgH = Math.max(layoutInfo.svgH, worldMaxY + 30);
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
        // Step 5 — Semantic Zoom: ``focusStackRef`` is the inline runtime's
        // focus stack.  Empty = full graph (legacy behaviour).  Non-empty =
        // focus on top-of-stack group; only the focus subtree is walked, and
        // edges crossing the focus boundary are augmented with the one-hop
        // outside endpoint (a leaf node, a collapsed-ancestor group, or an IO
        // pill) so users see the "local semantic closure" defined in
        // design/frontend_canvas_phase2.md §6.
        const focusStack = engine.focusStackRef;
        if (!Array.isArray(focusStack)) {
            throw new Error('render_canvas.js: computeVisibleScene requires focusStackRef to be wired');
        }
        const focusActive = focusStack.length > 0;
        const focusRootId = focusActive ? String(focusStack[focusStack.length - 1]) : null;
        const groupById = new Map();
        (data.groups || []).forEach(function (g) { groupById.set(String(g.id), g); });
        const nodeById = new Map();
        (data.nodes || []).forEach(function (n) { nodeById.set(String(n.id), n); });

        if (focusActive && !groupById.has(focusRootId)) {
            throw new Error('render_canvas.js: computeVisibleScene focus root not found in data.groups: ' + focusRootId);
        }

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
        if (focusActive) {
            // Focus mode: only descend from the focus root.  The inline runtime
            // is responsible for ensuring the focus root + all its descendants
            // are expanded BEFORE the render starts (see ``cascadeExpand`` in
            // frontend_html.py).  If the focus root itself is still flagged as
            // collapsed in ``collapsedState`` that is a hard wiring bug — the
            // walk would emit just the collapsed card and produce an empty
            // focus, which we surface as an error rather than silently render
            // nothing.
            if (collapsed[focusRootId] === true) {
                throw new Error('render_canvas.js: focus root ' + focusRootId + ' is still collapsed; inline cascadeExpand failed to flip collapsedState before render');
            }
            walk(focusRootId);
        } else {
            (data.root_groups || []).forEach(function (rid) {
                walk(rid);
            });
        }
        // Snapshot the focus-subtree membership BEFORE boundary augmentation.
        // The edge loop must classify endpoints against the original subtree
        // (one hop = "outside the focus subtree, not yet boundary"); without
        // this snapshot, a two-hop chain ``inside → boundary1 → boundary2``
        // would silently promote ``boundary2`` to one-hop visible.
        const focusSubtreeGroupSet = new Set(visibleGroupSet);
        const focusSubtreeNodeSet = new Set(visibleNodeSet);

        // Step 5 fit (problem 2/4): remember the focus subtree (the drilled-in
        // closure, EXCLUDING the one-hop boundary that is augmented below) so
        // ``performAutoFit`` can zoom to the subgraph itself.  Without this the
        // fit would use the boundary-inflated visible set and the drill-down
        // would not visibly magnify (and a far boundary would stretch the
        // frame).  Cleared to null on the full-graph path so exit restores the
        // normal full-graph fit.
        if (focusActive) {
            const focusFitIds = new Set();
            focusSubtreeGroupSet.forEach(function (gid) { focusFitIds.add(gid); });
            focusSubtreeNodeSet.forEach(function (nid) { focusFitIds.add(nid); });
            engine.lastFocusSubtreeIds = focusFitIds;
        } else {
            engine.lastFocusSubtreeIds = null;
        }

        // Shared boundary-anchor classifier (problem 1).  Built once per scene in
        // focus mode and reused by every ``addBoundary`` call so the admission and
        // the layout (augmentFocusBoundaryMeta) agree id-for-id.
        const boundaryCtx = focusActive ? buildBoundaryAnchorCtx(data) : null;

        // Boundary helper (Step 5 / problem 1): admit a one-hop *outside* endpoint
        // to the focus frame so users still see where the focus root connects to
        // the rest of the graph.  The classification is delegated to the shared
        // ``resolveBoundaryAnchor`` so this admission path and
        // ``augmentFocusBoundaryMeta`` (which lays the boundary card / io pill out)
        // can never disagree on what an id resolves to.  The anchor kind is one of:
        //   - 'group' : a collapsed parent-group boundary card (also covers a
        //               Group in/out port node id → its owning group)
        //   - 'node'  : a leaf-node boundary card
        //   - 'io'    : a model IO group (input/param/const/output) drawn as a pill
        // Anything that resolves to none of these is a hard wiring bug and raises
        // (no silent drop).  Returns the *route id* the edge must use for its
        // endpoint box lookup (the owning group for ports, the io_group id for io
        // anchors, the resolved card id otherwise) so the caller routes and
        // snapshots against the exact box admitted here.  Boundary entries are
        // *non-recursive* — we never expand inside them.
        function addBoundary(rawId) {
            if (!boundaryCtx) {
                throw new Error('render_canvas.js: addBoundary called outside focus mode (boundaryCtx unset)');
            }
            const anchor = resolveBoundaryAnchor(rawId, boundaryCtx);
            const sid = anchor.id;
            if (anchor.kind === 'group') {
                if (!(visibleGroupSet.has(sid) || visibleNodeSet.has(sid))) {
                    const groupRecord = groupById.get(sid);
                    if (!groupRecord) {
                        throw new Error('render_canvas.js: boundary group ' + sid + ' missing from data.groups');
                    }
                    const gmeta = groupMetaById.get(sid);
                    if (!gmeta) {
                        throw new Error('render_canvas.js: boundary group ' + sid + ' has no layout meta (augmentFocusBoundaryMeta did not place it)');
                    }
                    visibleGroupSet.add(sid);
                    groupIds.push(sid);
                    // Step 5 (problem 3): the boundary card is re-laid-out adjacent
                    // to the focus subtree, so include it in the auto-fit set — the
                    // whole local closure should fit the viewport together.
                    if (engine.lastFocusSubtreeIds) { engine.lastFocusSubtreeIds.add(sid); }
                    // Boundary groups always render as collapsed cards (we never
                    // recurse into them in focus mode), regardless of the user's
                    // global expand state.
                    const boundaryCollapsed = Object.assign({}, collapsed);
                    boundaryCollapsed[sid] = true;
                    groupSnapshots[sid] = makeIncrGroupSnapshot(groupRecord, boundaryCollapsed, gmeta);
                }
                return sid;
            }
            if (anchor.kind === 'node') {
                if (!(visibleNodeSet.has(sid) || visibleGroupSet.has(sid))) {
                    const nodeRecord = nodeById.get(sid);
                    if (!nodeRecord) {
                        throw new Error('render_canvas.js: boundary node ' + sid + ' missing from data.nodes');
                    }
                    const nmeta = nodeMetaById.get(sid);
                    if (!nmeta) {
                        throw new Error('render_canvas.js: boundary node ' + sid + ' has no layout meta (augmentFocusBoundaryMeta did not place it)');
                    }
                    visibleNodeSet.add(sid);
                    nodeIds.push(sid);
                    if (engine.lastFocusSubtreeIds) { engine.lastFocusSubtreeIds.add(sid); }
                    nodeSnapshots[sid] = makeIncrNodeSnapshot(nodeRecord, nmeta);
                }
                return sid;
            }
            // anchor.kind === 'io': the pill box must already be live —
            // augmentFocusBoundaryMeta emitted an io_pill task that drawIOTasks
            // registered in engine.ioPillById before computeVisibleScene ran.
            if (!engine.ioPillById.get(sid)) {
                throw new Error('render_canvas.js: boundary io group ' + sid + ' has no live pill box (augmentFocusBoundaryMeta/drawIOTasks did not register it)');
            }
            if (engine.lastFocusSubtreeIds) { engine.lastFocusSubtreeIds.add(sid); }
            return sid;
        }

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
        function isIoPill(id) {
            return engine.ioPillById.get(String(id)) ? true : false;
        }
        void isIoPill; // retained for diagnostics; addBoundary now fail-fasts on io anchors
        // Step 5 (problem 5 — ReturnVal edge): an endpoint counts as "inside the
        // focus subtree" when it is a focus-subtree node/group directly, OR when
        // it is a boundary-port node id (Group in/out_port, e.g. a ReturnVal
        // output port) whose owning group is inside the focus subtree.  The
        // ReturnVal port id is never a member of data.nodes/groups, so the bare
        // ``focusSubtreeNodeSet/GroupSet`` membership check misses it; without
        // this the focus group's own return edge would be treated as fully
        // outside and dropped (after resolveCollapsedAncestor stops redirecting
        // it onto the collapsed parent card).  ``portPointForEndpoint`` then
        // anchors the port to the (expanded) owning group's real out-port box.
        function isInsideFocus(id) {
            const sid = String(id);
            if (focusSubtreeNodeSet.has(sid) || focusSubtreeGroupSet.has(sid)) { return true; }
            const portInfo = ensurePortNodeIndex().get(sid);
            return !!(portInfo && focusSubtreeGroupSet.has(portInfo.groupId));
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
            // ── Step 5 focus filter ────────────────────────────────────────
            // In focus mode the visible edge set is the union of:
            //   - both endpoints inside the focus subtree (purely-internal),
            //   - exactly one endpoint inside (boundary edge → admit the outside
            //     endpoint as a non-recursive boundary anchor; ``addBoundary``
            //     returns the *route id* of the live box it admitted — the owning
            //     group for a port, the io_group id for an IO anchor, the resolved
            //     card id otherwise).
            // Edges with neither endpoint inside the focus subtree are dropped —
            // they are not part of the local semantic closure.  An outside
            // endpoint that resolves to nothing drawable raises in addBoundary
            // (no silent skip).
            let routeFromId = fromId;
            let routeToId = toId;
            if (focusActive) {
                const fromInside = isInsideFocus(fromId);
                const toInside = isInsideFocus(toId);
                if (!fromInside && !toInside) { return; }
                if (!fromInside) { routeFromId = addBoundary(e.from); }
                if (!toInside) { routeToId = addBoundary(e.to); }
            }
            const fromPort = portPointForEndpoint(routeFromId, 'src', metaBoxForId);
            const toPort = portPointForEndpoint(routeToId, 'dst', metaBoxForId);
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
                srcId: routeFromId,
                dstId: routeToId,
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
        engine.isIncrementalPatching = true;
        try {
            diffAndPatch(prev, next);
        } finally {
            engine.isIncrementalPatching = false;
        }
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
        // Step 5 fit (problem 2/4): when focused, zoom to the drilled-in subtree
        // (excluding the one-hop boundary) and allow upscaling up to
        // ``FOCUS_MAX_SCALE`` so the subgraph actually fills the viewport width
        // — the previous code fit the boundary-inflated visible set with a 1.0
        // cap, so the drill-down never visibly magnified.  On the full-graph
        // path (including focus exit, where ``lastFocusSubtreeIds`` is null)
        // use the full renderable bounds with the normal 1.0 cap so the graph
        // is never enlarged past native size.
        const FOCUS_MAX_SCALE = 2.0;
        const focusActive = Array.isArray(engine.focusStackRef) && engine.focusStackRef.length > 0;
        let fitBounds = null;
        let fitMaxScale = 1.0;
        if (focusActive) {
            const subtreeBounds = computeFocusSubtreeBounds(engine.lastFocusSubtreeIds);
            if (subtreeBounds) {
                fitBounds = subtreeBounds;
                fitMaxScale = FOCUS_MAX_SCALE;
            }
        }
        if (!fitBounds) {
            fitBounds = computeRenderableContentBounds();
        }
        engine.contentBounds = fitBounds;
        const containerSize = resolveContainerSize('auto-fit');
        const cw = containerSize.w;
        const viewportH = (typeof global !== 'undefined' && global && typeof global.innerHeight === 'number' && global.innerHeight > 0)
            ? global.innerHeight
            : null;
        const ch = (viewportH !== null) ? viewportH : containerSize.h;
        const FIT_PADDING = 40;
        const vp = engine.viewportController.fitToView(fitBounds, cw, ch, { padding: FIT_PADDING, maxScale: fitMaxScale });
        // Width-only fit: the canvas width fills the container so there is no
        // horizontal scroll.  The height is grown to the full scaled content so a
        // graph taller than the viewport overflows into the .dag-stage's vertical
        // scroll (matches the legacy SVG semantics) instead of being squeezed.
        const contentHeight = Math.ceil(fitBounds.h * vp.scale + 2 * FIT_PADDING);
        const canvasHeight = contentHeight;
        if (engine.app && engine.app.renderer && typeof engine.app.renderer.resize === 'function') {
            engine.rendererResizeCallCount = (engine.rendererResizeCallCount || 0) + 1;
            engine.app.renderer.resize(Math.ceil(cw), Math.ceil(canvasHeight));
        }
        if (engine.app && engine.app.canvas) {
            engine.app.canvas.style.width = '100%';
            engine.app.canvas.style.height = canvasHeight + 'px';
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
        // Phase 2 step 5 — the unified topbar now carries a *static* inline
        // legend plus the breadcrumb nav; the standalone legend div, mode badge
        // and meta line were removed from the template.  Only the bottom
        // architecture / timing summary is still populated here.
        const summaryDiv = global.document.getElementById('summary');
        if (!summaryDiv) {
            throw new Error('render_canvas.js: summary DOM is missing');
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
        let datasetSwitched = false;
        // Pool-first initial render needs computeVisibleScene() context. The
        // inline runtime normally installs it via __canvasSetIncrementalContext
        // before the first render, but headless probes call canvasRenderPhase1
        // directly with fresh ``data`` (and mutate the inline ``collapsedState``
        // global) without going through that hook. Establish the context here
        // from the render's own ``data`` argument + the live collapsedState so
        // the initial scene walk always has a valid, current reference.
        if (data && typeof data === 'object') {
            // Problem 4: detect a dataset switch (a *different* payload object than
            // the one the pools were last built from) so the draw-edges chunk can
            // destroy the now-stale pool views before diffing the new scene.
            datasetSwitched = engine.renderedDataRef != null && engine.renderedDataRef !== data;
            engine.dataRef = data;
            const liveCollapsed = lookupCollapsedState();
            if (liveCollapsed && typeof liveCollapsed === 'object') {
                engine.collapsedStateRef = liveCollapsed;
            }
            // Phase 2 step 5 — same lazy fallback for the focus stack.
            // Headless probes mutate the inline ``focusStack`` global directly
            // (or skip the inline runtime entirely); either way we must hand
            // computeVisibleScene a non-null Array reference.
            const liveFocus = lookupFocusStack();
            if (Array.isArray(liveFocus)) {
                engine.focusStackRef = liveFocus;
            } else if (!Array.isArray(engine.focusStackRef)) {
                engine.focusStackRef = [];
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
                //
                // Problem 4: on a dataset switch, destroy the stale pool views
                // FIRST (freeing their VRAM and pruning them from the visible-id
                // sets) so the diff below never tries to recycle a view that
                // belongs to the previous graph, then patch the new scene over
                // three frames so PixiJS uploads the geometry in three GPU ticks
                // instead of one ~1s ``bufferSubData`` flush.
                if (datasetSwitched) {
                    destroyStalePoolViews(data);
                }
                const prev = {
                    groupIds: engine.visibleGroupIds,
                    nodeIds: engine.visibleNodeIds,
                    edgeKeys: engine.visibleEdgeKeys
                };
                const next = computeVisibleScene();
                await diffAndPatchChunked(prev, next, p);
                engine.renderedDataRef = data;
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
                has_header: !s.collapsed, has_timing: s.hasTiming
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
                return { id: g.id, collapsed: g.collapsed, x: g.x, y: g.y, w: g.w, h: g.h, has_header: g.has_header, has_timing: g.has_timing };
            }),
            edges: (engine.edges || []).map(function (e) {
                return {
                    from: e.from, to: e.to, type: e.type, colorKey: e.colorKey,
                    start: { cx: e.start.cx, cy: e.start.cy },
                    end: { cx: e.end.cx, cy: e.end.cy },
                    branch: e.branch, dashed: e.dashed, arrow: e.arrow
                };
            }),
            io_pills: (engine.io_pills || []).map(function (p) {
                return {
                    id: p.id,
                    subtype: p.subtype,
                    x: p.x,
                    y: p.y,
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

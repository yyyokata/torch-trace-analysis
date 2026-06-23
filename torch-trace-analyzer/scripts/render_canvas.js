/* render_canvas.js
 * Canvas Phase 1 / Stage 1.1 -- Canvas (Pixi) render skeleton.
 *
 * Stage 1.1 scope (strictly skeleton, draws no real glyphs):
 *   - build PIXI.Application + world (viewport) + layers L0..L5
 *   - expose window.__phase1NoInteractionMode === true
 *   - expose a read-only window.__renderSnapshot() returning layer / flag info
 *   - provide a headless mock path so Python UTs can run without a real canvas
 *
 * Hard rules honoured here:
 *   - no fallback / silent path: a missing #dag-stage container raises
 *   - no interaction wiring in this stage (no hover / click / toggle)
 *
 * Later stages (1.2 .. 1.4) extend the snapshot with nodes/groups/edges/io and
 * add viewport pan/zoom/fit + culling.  Stage 1.1 only stands up the layer tree.
 */
(function (global) {
    'use strict';

    // L0 background .. L5 overlay; L4 is the (currently empty) interaction
    // overlay layer that Phase 3 will populate.  Order matters: index === paint
    // order.
    const LAYER_KEYS = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5'];

    // Resolve a PIXI-like factory.  Prefer the injected engine bundle
    // (global.PIXI); otherwise fall back to an internal headless mock so unit
    // tests run with no DOM canvas / WebGL context.  This is NOT a render
    // fallback -- both paths build the identical layer tree, they only differ in
    // whether a real <canvas> is attached.
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
            return removed;
        };
        function Application(options) {
            const opts = options || {};
            this.stage = new Container();
            this.screen = { x: 0, y: 0, width: opts.width || 0, height: opts.height || 0 };
            this.canvas = null;
            this.renderer = { resize: function () {} };
        }
        Application.prototype.destroy = function () { this.stage = new Container(); };
        return { Application: Application, Container: Container, __isHeadlessMock: true };
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

    // Production render entry.  The legacy in-template SVG render() delegates
    // here whenever window.__phase1NoInteractionMode === true.  Stage 1.1 builds
    // the skeleton only and resolves with the current snapshot.
    function canvasRenderPhase1(/* data */) {
        ensureEngine();
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

    // Read-only snapshot.  MUST NOT mutate runtime state.  Same shape in browser
    // and headless paths; any missing field is a bug, never an optional.
    function buildSnapshot() {
        if (!engine) {
            throw new Error('render_canvas.js: __renderSnapshot called before the Canvas engine was initialized');
        }
        const vp = engine.viewport;
        return {
            nodes: [],
            groups: [],
            edges: [],
            ports: {},
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

    // Stage 1.1 runs with interaction disabled.
    global.__phase1NoInteractionMode = true;
    global.__canvasRenderPhase1 = canvasRenderPhase1;
    global.__initCanvasEngine = initCanvasEngine;
    global.__canvasEnginePhase1 = function () { return engine; };
    global.__renderSnapshot = function () {
        ensureEngine();
        return buildSnapshot();
    };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

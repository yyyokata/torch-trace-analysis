/* pixi_engine_bundle.js
 * Canvas Phase 1 / Stage 1.1 engine bundle.
 *
 * This is a minimal, self-contained PIXI-compatible scene-graph runtime that
 * exposes a `window.PIXI` global with just enough surface for the Stage 1.1
 * Canvas skeleton (Application / Container plus the Graphics / Text / Rectangle
 * stubs the later stages will flesh out).  It intentionally draws nothing: the
 * real WebGL Pixi bundle will replace this file in a later iteration, but the
 * public API (PIXI.Application, PIXI.Container, addChild, removeChildren, ...)
 * is kept identical so render_canvas.js does not need to change.
 *
 * No real canvas painting happens here; it only builds the object graph so the
 * renderer can attach layers and report a render snapshot.
 */
(function (root) {
    'use strict';

    function Container() {
        this.children = [];
        this.x = 0;
        this.y = 0;
        this.scale = { x: 1, y: 1 };
        this.visible = true;
        this.parent = null;
        this.name = '';
        this.__isPixiContainer = true;
    }
    Container.prototype.addChild = function (child) {
        if (!child) {
            throw new Error('PIXI.Container.addChild called without a child');
        }
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
    Container.prototype.removeChild = function (child) {
        this.children = this.children.filter(function (c) { return c !== child; });
        if (child) { child.parent = null; }
        return child;
    };

    function Graphics() {
        Container.call(this);
        this.__isPixiGraphics = true;
    }
    Graphics.prototype = Object.create(Container.prototype);
    Graphics.prototype.constructor = Graphics;
    Graphics.prototype.clear = function () { return this; };

    function Text(text) {
        Container.call(this);
        this.text = (text === undefined || text === null) ? '' : String(text);
        this.__isPixiText = true;
    }
    Text.prototype = Object.create(Container.prototype);
    Text.prototype.constructor = Text;

    function Rectangle(x, y, width, height) {
        this.x = x || 0;
        this.y = y || 0;
        this.width = width || 0;
        this.height = height || 0;
    }

    function Application(options) {
        const opts = options || {};
        this.stage = new Container();
        this.screen = new Rectangle(0, 0, opts.width || 0, opts.height || 0);
        this.renderer = { resize: function (w, h) { this.width = w; this.height = h; } };
        this.canvas = null;
        const doc = root && root.document;
        if (doc && typeof doc.createElement === 'function') {
            this.canvas = doc.createElement('canvas');
        }
        this.__isPixiApplication = true;
    }
    Application.prototype.destroy = function () {
        this.stage = new Container();
        this.canvas = null;
    };

    const PIXI = {
        Application: Application,
        Container: Container,
        Graphics: Graphics,
        Text: Text,
        Rectangle: Rectangle,
        VERSION: 'phase1-stage1.1-bundle',
        __isMinimalBundle: true
    };

    root.PIXI = PIXI;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = PIXI;
    }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));

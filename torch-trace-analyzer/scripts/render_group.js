/* global svg, groupMap, groupLayout, nodePortMap, LAYOUT, registerNodeDom, bindGroupHover, toggleGroup, appendGroupInfoButton, formatDur, getGroupBorderColor, isEdgeVisible, renderEdge, renderNodeAt */
// render_group.js
// Depends on closure variables from the main template:
//   svg, groupMap, groupLayout, nodePortMap, LAYOUT,
//   registerNodeDom, bindGroupHover, toggleGroup,
//   appendGroupInfoButton, formatDur, getGroupBorderColor,
//   isEdgeVisible, renderEdge, renderNodeAt, renderGroupAt

const RENDER_GROUP_SVG_NS = 'http://www.w3.org/2000/svg';
const RENDER_GROUP_EXPORTS = {};

function getGroupRenderContext(gid, ox, oy) {
    const g = groupMap[gid];
    if (!g) {
        throw new Error(`renderGroupAt group not found: ${gid}`);
    }
    const pos = Object.prototype.hasOwnProperty.call(groupLayout, gid) ? groupLayout[gid] : null;
    return { gid, g, pos, ox, oy };
}

function renderCollapsedGroupBox(ctx) {
    const rect = document.createElementNS(RENDER_GROUP_SVG_NS, 'rect');
    rect.setAttribute('x', ctx.ox);
    rect.setAttribute('y', ctx.oy);
    rect.setAttribute('width', ctx.pos.w);
    rect.setAttribute('height', ctx.pos.h);
    rect.setAttribute('class', 'group-box collapsed');
    rect.setAttribute('style', `stroke: ${getGroupBorderColor(ctx.g)}`);
    rect.dataset.gid = ctx.gid;
    rect.addEventListener('dblclick', () => toggleGroup(ctx.gid));
    bindGroupHover(rect, ctx.gid);
    svg.appendChild(rect);
    registerNodeDom(ctx.gid, rect);
    return rect;
}

function renderCollapsedGroupLabel(ctx) {
    const labelText = `▶ ${ctx.g.class_name || ctx.g.label}`;
    const labelEl = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    labelEl.setAttribute('x', ctx.ox + ctx.pos.w / 2);
    labelEl.setAttribute('y', ctx.oy + ctx.pos.h / 2 - 2);
    labelEl.setAttribute('text-anchor', 'middle');
    labelEl.setAttribute('class', 'group-label');
    labelEl.textContent = labelText;
    bindGroupHover(labelEl, ctx.gid);
    svg.appendChild(labelEl);
    return { labelText, labelEl };
}

function renderCollapsedGroupTiming(ctx) {
    if (!ctx.g.has_timing) return null;
    const timingEl = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    timingEl.setAttribute('x', ctx.ox + ctx.pos.w / 2);
    timingEl.setAttribute('y', ctx.oy + ctx.pos.h / 2 + 12);
    timingEl.setAttribute('text-anchor', 'middle');
    timingEl.setAttribute('class', 'group-timing');
    timingEl.textContent = `Kernel ${ctx.g.pct.toFixed(1)}% · ${formatDur(ctx.g.dur_us)}`;
    bindGroupHover(timingEl, ctx.gid);
    svg.appendChild(timingEl);
    return timingEl;
}

function renderCollapsedGroupInfoButton(ctx) {
    if (!ctx.g.src_file) return;
    appendGroupInfoButton(
        ctx.g,
        ctx.ox + ctx.pos.w - 13,
        ctx.oy + 10,
        `View ${ctx.g.label} class definition (${ctx.g.src_file}:${ctx.g.src_start_line}-${ctx.g.src_end_line})`
    );
}

function registerCollapsedGroupPorts(ctx) {
    nodePortMap[ctx.gid + '__in'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy };
    nodePortMap[ctx.gid + '__out'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h };
    nodePortMap[ctx.gid + '__center'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h / 2 };
}

function renderExpandedGroupBox(ctx) {
    const rect = document.createElementNS(RENDER_GROUP_SVG_NS, 'rect');
    rect.setAttribute('x', ctx.ox);
    rect.setAttribute('y', ctx.oy);
    rect.setAttribute('width', ctx.pos.w);
    rect.setAttribute('height', ctx.pos.h);
    rect.setAttribute('class', 'group-box');
    rect.setAttribute('style', `stroke: ${getGroupBorderColor(ctx.g)}`);
    rect.dataset.gid = ctx.gid;
    rect.addEventListener('dblclick', (e) => {
        if (e.target === rect) toggleGroup(ctx.gid);
    });
    bindGroupHover(rect, ctx.gid);
    svg.appendChild(rect);
    registerNodeDom(ctx.gid, rect);
    return rect;
}

function renderExpandedGroupHeaderLabel(ctx) {
    const headerText = `▼ ${ctx.g.class_name || ctx.g.label}`;
    const labelEl = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    labelEl.setAttribute('x', ctx.ox + 12);
    labelEl.setAttribute('y', ctx.oy + 18);
    labelEl.setAttribute('class', 'group-label');
    labelEl.setAttribute('style', 'cursor: pointer;');
    labelEl.textContent = headerText;
    labelEl.addEventListener('dblclick', () => toggleGroup(ctx.gid));
    bindGroupHover(labelEl, ctx.gid);
    svg.appendChild(labelEl);
    return { headerText, labelEl };
}

function renderExpandedGroupInfoButton(ctx, headerText) {
    if (!ctx.g.src_file) return;
    const bx = ctx.ox + ctx.pos.w - 20;
    appendGroupInfoButton(
        ctx.g,
        bx,
        ctx.oy + 15,
        `View ${ctx.g.label} class definition (${ctx.g.src_file}:${ctx.g.src_start_line}-${ctx.g.src_end_line})`
    );
}

function renderExpandedGroupTiming(ctx) {
    if (!ctx.g.has_timing) return null;
    const timingEl = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    timingEl.setAttribute('x', ctx.ox + ctx.pos.w - 10);
    timingEl.setAttribute('y', ctx.oy + 18);
    timingEl.setAttribute('text-anchor', 'end');
    timingEl.setAttribute('class', 'group-timing');
    timingEl.textContent = `Kernel ${ctx.g.pct.toFixed(1)}% · ${formatDur(ctx.g.dur_us)}`;
    bindGroupHover(timingEl, ctx.gid);
    svg.appendChild(timingEl);
    return timingEl;
}

function renderGroupChildren(ctx) {
    for (const child of (ctx.pos.childPositions || [])) {
        if (child.type === 'node') {
            renderNodeAt(child.id, ctx.ox + child.x, ctx.oy + child.y, child.w, child.h);
        } else {
            renderGroupAt(child.id, ctx.ox + child.x, ctx.oy + child.y);
        }
    }
}

function buildInternalEdgeRoutingContext(ctx) {
    const childPositions = ctx.pos.childPositions || [];
    if (childPositions.length === 0) {
        return {
            childAbsRect: {},
            groupLeft: null,
            groupRight: null,
            childLeftEdge: null,
            childRightEdge: null,
            skipLaneCounter: {},
        };
    }

    const childAbsRect = {};
    let cMinX = Infinity;
    let cMaxX = -Infinity;
    for (const child of childPositions) {
        const ax = ctx.ox + child.x;
        childAbsRect[child.id] = {
            x: ax,
            y: ctx.oy + child.y,
            w: child.w,
            h: child.h,
            rank: child.rank,
        };
        if (ax < cMinX) cMinX = ax;
        if (ax + child.w > cMaxX) cMaxX = ax + child.w;
    }
    return {
        childAbsRect,
        groupLeft: Math.max(ctx.ox + 6, cMinX - LAYOUT.laneGutter),
        groupRight: Math.min(ctx.ox + ctx.pos.w - 6, cMaxX + LAYOUT.laneGutter),
        childLeftEdge: cMinX,
        childRightEdge: cMaxX,
        skipLaneCounter: { left: 0, right: 0 },
    };
}

function resolveEdgeEndpointRect(ed, which, routeCtx, ctx) {
    const portKind = which === 'from' ? ed.from_port : ed.to_port;
    if (portKind === 'in') {
        const pt = nodePortMap[ctx.gid + '__in'];
        if (!pt) {
            throw new Error(`resolveEdgeEndpointRect: missing in-port for group ${ctx.gid}`);
        }
        return { x: pt.cx, y: pt.cy, w: 0, h: 0, rank: -1 };
    }
    if (portKind === 'out') {
        const pt = nodePortMap[ctx.gid + '__out'];
        if (!pt) {
            throw new Error(`resolveEdgeEndpointRect: missing out-port for group ${ctx.gid}`);
        }
        return { x: pt.cx, y: pt.cy, w: 0, h: 0, rank: -1 };
    }
    const childKey = which === 'from' ? ed.from_child : ed.to_child;
    const rect = routeCtx.childAbsRect[childKey];
    if (!rect) {
        return null;
    }
    return rect;
}

function renderGroupInternalEdges(ctx, routeCtx) {
    if (!(ctx.g.internal_edges && ctx.pos.rowLayouts)) return;
    for (const ed of ctx.g.internal_edges) {
        const fr = resolveEdgeEndpointRect(ed, 'from', routeCtx, ctx);
        const to = resolveEdgeEndpointRect(ed, 'to', routeCtx, ctx);
        if (!fr || !to) {
            console.warn('[renderGroupInternalEdges] missing child rect for edge:', ed);
            continue;
        }
        const routedEdge = {
            __internal: true,
            __gid: ctx.gid,
            from: ed.from_port ? (ctx.gid + '__' + ed.from_port) : ed.from_child,
            to: ed.to_port ? (ctx.gid + '__' + ed.to_port) : ed.to_child,
            type: ed.type || 'internal',
            from_attr: ed.from_attr,
            to_attr: ed.to_attr,
            parent_class: ed.parent_class || ctx.g.label,
            evidence: ed.evidence,
        };
        if (!isEdgeVisible(routedEdge)) continue;
        renderEdge({
            routingMode: 'intra_group',
            fr,
            to,
            type: ed.type,
            edgeData: routedEdge,
            routeCtx: {
                groupLeft: routeCtx.groupLeft,
                groupRight: routeCtx.groupRight,
                childLeftEdge: routeCtx.childLeftEdge,
                childRightEdge: routeCtx.childRightEdge,
                skipLaneCounter: routeCtx.skipLaneCounter,
            },
        });
    }
}

function registerExpandedGroupPorts(ctx) {
    nodePortMap[ctx.gid + '__in'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy };
    nodePortMap[ctx.gid + '__out'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h };
}

function renderGroupAt(gid, ox, oy) {
    const ctx = RENDER_GROUP_EXPORTS.getGroupRenderContext(gid, ox, oy);
    if (!ctx.pos) return;
    if (ctx.pos.collapsed) {
        RENDER_GROUP_EXPORTS.renderCollapsedGroupBox(ctx);
        RENDER_GROUP_EXPORTS.renderCollapsedGroupLabel(ctx);
        RENDER_GROUP_EXPORTS.renderCollapsedGroupTiming(ctx);
        RENDER_GROUP_EXPORTS.renderCollapsedGroupInfoButton(ctx);
        RENDER_GROUP_EXPORTS.registerCollapsedGroupPorts(ctx);
        return;
    }
    RENDER_GROUP_EXPORTS.renderExpandedGroupBox(ctx);
    const { headerText } = RENDER_GROUP_EXPORTS.renderExpandedGroupHeaderLabel(ctx);
    RENDER_GROUP_EXPORTS.renderExpandedGroupInfoButton(ctx, headerText);
    RENDER_GROUP_EXPORTS.renderExpandedGroupTiming(ctx);
    RENDER_GROUP_EXPORTS.renderGroupChildren(ctx);
    const routeCtx = RENDER_GROUP_EXPORTS.buildInternalEdgeRoutingContext(ctx);
    // Register this group's own __in/__out ports BEFORE rendering internal edges,
    // because port-routed internal edges (from_port/to_port) resolve against them.
    RENDER_GROUP_EXPORTS.registerExpandedGroupPorts(ctx);
    RENDER_GROUP_EXPORTS.renderGroupInternalEdges(ctx, routeCtx);
}

Object.assign(RENDER_GROUP_EXPORTS, {
    getGroupRenderContext,
    renderCollapsedGroupBox,
    renderCollapsedGroupLabel,
    renderCollapsedGroupTiming,
    renderCollapsedGroupInfoButton,
    registerCollapsedGroupPorts,
    renderExpandedGroupBox,
    renderExpandedGroupHeaderLabel,
    renderExpandedGroupInfoButton,
    renderExpandedGroupTiming,
    renderGroupChildren,
    buildInternalEdgeRoutingContext,
    resolveEdgeEndpointRect,
    renderGroupInternalEdges,
    registerExpandedGroupPorts,
    renderGroupAt,
});

if (typeof globalThis !== 'undefined') {
    Object.assign(globalThis, RENDER_GROUP_EXPORTS);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RENDER_GROUP_EXPORTS;
}

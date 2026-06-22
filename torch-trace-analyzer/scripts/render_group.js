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
    const labelText = `▶ ${ctx.g.class_name}`;
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
    const inPoint = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy };
    const outPoint = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h };
    nodePortMap[ctx.gid + '__in'] = inPoint;
    nodePortMap[ctx.gid + '__out'] = outPoint;
    nodePortMap[ctx.gid + '__center'] = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h / 2 };
    for (const port of (ctx.g.in_ports || [])) {
        nodePortMap[port.node_id] = inPoint;
    }
    for (const port of (ctx.g.out_ports || [])) {
        nodePortMap[port.node_id] = outPoint;
    }
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
    const headerText = `▼ ${ctx.g.class_name}`;
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

function renderExpandedGroupInfoButton(ctx, _headerText) {
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

function registerExpandedGroupPorts(ctx) {
    const inPoint = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy };
    const outPoint = { cx: ctx.ox + ctx.pos.w / 2, cy: ctx.oy + ctx.pos.h };
    nodePortMap[ctx.gid + '__in'] = inPoint;
    nodePortMap[ctx.gid + '__out'] = outPoint;
    for (const port of (ctx.g.in_ports || [])) {
        nodePortMap[port.node_id] = inPoint;
    }
    for (const port of (ctx.g.out_ports || [])) {
        nodePortMap[port.node_id] = outPoint;
    }
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
    RENDER_GROUP_EXPORTS.registerExpandedGroupPorts(ctx);
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
    registerExpandedGroupPorts,
    renderGroupAt,
});

if (typeof globalThis !== 'undefined') {
    Object.assign(globalThis, RENDER_GROUP_EXPORTS);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RENDER_GROUP_EXPORTS;
}

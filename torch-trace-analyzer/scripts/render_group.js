/* global groupMap, groupLayout, nodePortMap, svg, LAYOUT, bindGroupHover, registerNodeDom, toggleGroup, showSourcePanel, getGroupBorderColor, formatDur, isEdgeVisible, renderEdge, renderNodeAt */
/*
 * render_group.js
 *
 * Closure dependencies injected by frontend_html.py inline <script> scope:
 * - state: groupMap, groupLayout, nodePortMap, svg, LAYOUT
 * - DOM helpers: bindGroupHover, registerNodeDom, toggleGroup, showSourcePanel
 * - style/data helpers: getGroupBorderColor, formatDur, isEdgeVisible, renderEdge
 * - renderers: renderNodeAt, renderGroupAt (recursive self-call)
 */
const RENDER_GROUP_SVG_NS = 'http://www.w3.org/2000/svg';

function appendGroupInfoButton(g, cx, cy, titleText = '') {
    if (!g || !g.src_file) return null;
    const hit = document.createElementNS(RENDER_GROUP_SVG_NS, 'circle');
    hit.setAttribute('cx', cx);
    hit.setAttribute('cy', cy - 1);
    hit.setAttribute('r', 9);
    hit.setAttribute('class', 'group-info-hit');
    hit.addEventListener('click', (e) => {
        e.stopPropagation();
        showSourcePanel(g);
    });
    svg.appendChild(hit);

    const info = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    info.setAttribute('x', cx);
    info.setAttribute('y', cy - 1);
    info.setAttribute('class', 'group-toggle group-info-text');
    info.textContent = 'i';
    if (titleText) {
        const titleEl = document.createElementNS(RENDER_GROUP_SVG_NS, 'title');
        titleEl.textContent = titleText;
        info.appendChild(titleEl);
    }
    svg.appendChild(info);
    return { hit, info };
}

function registerGroupPorts(gid, ox, oy, pos, includeCenter = false) {
    nodePortMap[gid + '__in'] = { cx: ox + pos.w / 2, cy: oy };
    nodePortMap[gid + '__out'] = { cx: ox + pos.w / 2, cy: oy + pos.h };
    if (includeCenter) {
        nodePortMap[gid + '__center'] = { cx: ox + pos.w / 2, cy: oy + pos.h / 2 };
    }
    return includeCenter ? nodePortMap[gid + '__center'] : null;
}

function renderCollapsedGroupBox(gid, g, pos, ox, oy) {
    const rect = document.createElementNS(RENDER_GROUP_SVG_NS, 'rect');
    rect.setAttribute('x', ox);
    rect.setAttribute('y', oy);
    rect.setAttribute('width', pos.w);
    rect.setAttribute('height', pos.h);
    rect.setAttribute('class', 'group-box collapsed');
    rect.setAttribute('style', `stroke: ${getGroupBorderColor(g)}`);
    rect.dataset.gid = gid;
    rect.addEventListener('dblclick', () => toggleGroup(gid));
    bindGroupHover(rect, gid);
    svg.appendChild(rect);
    registerNodeDom(gid, rect);
    return rect;
}

function renderCollapsedGroupLabel(gid, g, pos, ox, oy) {
    const label = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    label.setAttribute('x', ox + pos.w / 2);
    label.setAttribute('y', oy + pos.h / 2 - 2);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'group-label');
    label.textContent = `▶ ${g.label}`;
    bindGroupHover(label, gid);
    svg.appendChild(label);
    return label;
}

function renderCollapsedGroupTiming(gid, g, pos, ox, oy) {
    if (!g.has_timing) return null;
    const timing = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    timing.setAttribute('x', ox + pos.w / 2);
    timing.setAttribute('y', oy + pos.h / 2 + 12);
    timing.setAttribute('text-anchor', 'middle');
    timing.setAttribute('class', 'group-timing');
    timing.textContent = `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`;
    bindGroupHover(timing, gid);
    svg.appendChild(timing);
    return timing;
}

function renderCollapsedGroupInfoButton(g, pos, ox, oy) {
    if (!g.src_file) return null;
    return appendGroupInfoButton(
        g,
        ox + pos.w - 13,
        oy + 10,
        `View ${g.label} class definition (${g.src_file}:${g.src_start_line}-${g.src_end_line})`
    );
}

function renderCollapsedGroup(gid, g, pos, ox, oy) {
    const rect = renderCollapsedGroupBox(gid, g, pos, ox, oy);
    const label = renderCollapsedGroupLabel(gid, g, pos, ox, oy);
    const timing = renderCollapsedGroupTiming(gid, g, pos, ox, oy);
    const info = renderCollapsedGroupInfoButton(g, pos, ox, oy);
    registerGroupPorts(gid, ox, oy, pos, true);
    return { rect, label, timing, info };
}

function renderExpandedGroupBox(gid, g, pos, ox, oy) {
    const rect = document.createElementNS(RENDER_GROUP_SVG_NS, 'rect');
    rect.setAttribute('x', ox);
    rect.setAttribute('y', oy);
    rect.setAttribute('width', pos.w);
    rect.setAttribute('height', pos.h);
    rect.setAttribute('class', 'group-box');
    rect.setAttribute('style', `stroke: ${getGroupBorderColor(g)}`);
    rect.dataset.gid = gid;
    rect.addEventListener('dblclick', (e) => {
        if (e.target === rect) toggleGroup(gid);
    });
    bindGroupHover(rect, gid);
    svg.appendChild(rect);
    registerNodeDom(gid, rect);
    return rect;
}

function renderExpandedGroupHeader(gid, g, ox, oy) {
    const headerText = `▼ ${g.label}`;
    const label = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    label.setAttribute('x', ox + 12);
    label.setAttribute('y', oy + 18);
    label.setAttribute('class', 'group-label');
    label.setAttribute('style', 'cursor: pointer;');
    label.textContent = headerText;
    label.addEventListener('dblclick', () => toggleGroup(gid));
    bindGroupHover(label, gid);
    svg.appendChild(label);
    if (g.src_file) {
        const labelLen = (headerText.length * 6.6) + 8;
        appendGroupInfoButton(
            g,
            ox + 12 + labelLen + 6,
            oy + 15,
            `View ${g.label} class definition (${g.src_file}:${g.src_start_line}-${g.src_end_line})`
        );
    }
    return { headerText, label };
}

function renderExpandedGroupTiming(gid, g, pos, ox, oy) {
    if (!g.has_timing) return null;
    const timing = document.createElementNS(RENDER_GROUP_SVG_NS, 'text');
    timing.setAttribute('x', ox + pos.w - 10);
    timing.setAttribute('y', oy + 18);
    timing.setAttribute('text-anchor', 'end');
    timing.setAttribute('class', 'group-timing');
    timing.textContent = `Kernel ${g.pct.toFixed(1)}% · ${formatDur(g.dur_us)}`;
    bindGroupHover(timing, gid);
    svg.appendChild(timing);
    return timing;
}

function renderExpandedGroupChildren(childPositions, ox, oy) {
    for (const child of (childPositions || [])) {
        if (child.type === 'node') {
            renderNodeAt(child.id, ox + child.x, oy + child.y, child.w, child.h);
        } else {
            renderGroupAt(child.id, ox + child.x, oy + child.y);
        }
    }
    return (childPositions || []).length;
}

function renderGroupInternalEdges(gid, g, pos, childPositions, ox, oy) {
    if (!(g.internal_edges && pos.rowLayouts)) return 0;
    const childAbsRect = {};
    let cMinX = Infinity;
    let cMaxX = -Infinity;
    for (const child of childPositions) {
        const ax = ox + child.x;
        childAbsRect[child.id] = {
            x: ax,
            y: oy + child.y,
            w: child.w,
            h: child.h,
            rank: child.rank,
        };
        if (ax < cMinX) cMinX = ax;
        if (ax + child.w > cMaxX) cMaxX = ax + child.w;
    }
    const groupRight = Math.min(ox + pos.w - 6, cMaxX + LAYOUT.laneGutter);
    const groupLeft = Math.max(ox + 6, cMinX - LAYOUT.laneGutter);
    const childRightEdge = cMaxX;
    const childLeftEdge = cMinX;
    const skipLaneCounter = { left: 0, right: 0 };
    let renderedCount = 0;
    for (const ed of g.internal_edges) {
        const fr = childAbsRect[ed.from_child];
        const to = childAbsRect[ed.to_child];
        if (!fr || !to) {
            console.warn('[renderGroupInternalEdges] missing child rect, skip edge', {
                gid,
                edge: ed,
                hasFrom: Boolean(fr),
                hasTo: Boolean(to),
            });
            continue;
        }
        const routedEdge = {
            __internal: true,
            __gid: gid,
            from: ed.from_child,
            to: ed.to_child,
            type: ed.type || 'internal',
            from_attr: ed.from_attr,
            to_attr: ed.to_attr,
            parent_class: ed.parent_class || g.label,
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
                groupLeft,
                groupRight,
                childLeftEdge,
                childRightEdge,
                skipLaneCounter,
            },
        });
        renderedCount += 1;
    }
    return renderedCount;
}

function renderExpandedGroup(gid, g, pos, ox, oy) {
    const rect = renderExpandedGroupBox(gid, g, pos, ox, oy);
    const header = renderExpandedGroupHeader(gid, g, ox, oy);
    const timing = renderExpandedGroupTiming(gid, g, pos, ox, oy);
    const childPositions = pos.childPositions || [];
    const childCount = renderExpandedGroupChildren(childPositions, ox, oy);
    const edgeCount = renderGroupInternalEdges(gid, g, pos, childPositions, ox, oy);
    registerGroupPorts(gid, ox, oy, pos, false);
    return { rect, header, timing, childCount, edgeCount };
}

function renderGroupAt(gid, ox, oy) {
    const g = groupMap[gid];
    const pos = groupLayout[gid];
    if (!pos) return;
    if (pos.collapsed) {
        renderCollapsedGroup(gid, g, pos, ox, oy);
        return;
    }
    renderExpandedGroup(gid, g, pos, ox, oy);
}

const RENDER_GROUP_EXPORTS = {
    appendGroupInfoButton,
    registerGroupPorts,
    renderCollapsedGroupBox,
    renderCollapsedGroupLabel,
    renderCollapsedGroupTiming,
    renderCollapsedGroupInfoButton,
    renderCollapsedGroup,
    renderExpandedGroupBox,
    renderExpandedGroupHeader,
    renderExpandedGroupTiming,
    renderExpandedGroupChildren,
    renderGroupInternalEdges,
    renderExpandedGroup,
    renderGroupAt,
};

if (typeof globalThis !== 'undefined') {
    Object.assign(globalThis, RENDER_GROUP_EXPORTS);
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RENDER_GROUP_EXPORTS;
}

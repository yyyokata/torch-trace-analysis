#!/usr/bin/env python3

import io
import os
import re
import tokenize


def _strip_inline_comment(code_line: str) -> str:
    """剥离行内 # 注释（尽量不影响字符串中的 #）。

    目的：用于括号计数/静态解析时避免注释中的 " :( " 等非平衡括号破坏 open_count。
    """
    # tokenize.untokenize 会保留大部分原始空白/结构，比简单 split('#') 更安全
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code_line).readline))
        toks = [t for t in toks if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        # 兜底：不处理引号场景，仅用于防止解析直接崩
        return code_line.split('#', 1)[0]


def _join_logical_lines(raw_lines, base_lineno):
    """Join multi-line statements by tracking open brackets/parens.

    Returns list of (start_lineno, joined_text) tuples (start_lineno is the
    absolute line number in the source file of the FIRST physical line that
    contributed to the joined logical line).
    """
    logical = []
    buf = ""
    buf_start = None
    open_count = 0
    for offset, line in enumerate(raw_lines):
        phys_lineno = base_lineno + offset
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if not buf:
                logical.append((phys_lineno, line))
            continue
        if not buf:
            buf_start = phys_lineno
        stripped_nc = _strip_inline_comment(stripped).strip()
        if not stripped_nc:
            continue
        buf += (" " if buf else "") + stripped_nc
        open_count += stripped_nc.count('(') + stripped_nc.count('[') + stripped_nc.count('{')
        open_count -= stripped_nc.count(')') + stripped_nc.count(']') + stripped_nc.count('}')
        if open_count <= 0:
            logical.append((buf_start, buf))
            buf = ""
            buf_start = None
            open_count = 0
    if buf:
        logical.append((buf_start, buf))
    return logical


def format_duration(us):
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    elif us >= 1e3:
        return f"{us/1e3:.3f} ms"
    else:
        return f"{us:.2f} us"


def pct_str(part, total):
    return f"{part / total * 100:.2f}%" if total > 0 else "N/A"


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


PARALLEL_WRAPPER_CLASSES = ("FullyShardedDataParallel", "DistributedDataParallel", "CheckpointWrapper", "OffloadWrapper")
PARALLEL_WRAPPER_PREFIXES = ("FSDP", "DDP")


def _frame_get(frame, idx, name, default=None):
    if isinstance(frame, dict):
        return frame.get(name, default)
    if hasattr(frame, name):
        return getattr(frame, name)
    try:
        return frame[idx]
    except (TypeError, IndexError, KeyError):
        return default


def _frame_with_func(frame, new_func):
    if isinstance(frame, dict):
        copied = dict(frame)
        copied["func"] = new_func
        return copied
    if hasattr(frame, "_replace"):
        try:
            return frame._replace(func=new_func)
        except (TypeError, ValueError):
            pass
    if hasattr(frame, "__dataclass_fields__"):
        try:
            from dataclasses import replace
            return replace(frame, func=new_func)
        except (TypeError, ValueError):
            pass
    try:
        return (frame[0], frame[1], new_func)
    except (TypeError, IndexError):
        return frame


def _is_user_source_frame(frame, source_files):
    file_path = str(_frame_get(frame, 0, "file", "") or "").replace("\\", "/")
    fname = os.path.basename(file_path)
    if fname in (source_files or {}):
        return True
    for src_name in (source_files or {}):
        norm_src = str(src_name or "").replace("\\", "/")
        src_base = os.path.basename(norm_src)
        if file_path == norm_src or file_path.endswith("/" + norm_src):
            return True
        if src_base and (file_path == src_base or file_path.endswith("/" + src_base)):
            return True
    return False


def _extract_frame_class_name(func):
    text = str(func or "")
    if text.startswith("nn.Module:"):
        short = text.replace("nn.Module: ", "").replace("nn.Module:", "")
        short = re.sub(r',\s*callsite:\s*\d+', '', short).strip()
        return re.sub(r'_\d+$', '', short)
    m = re.match(r'([A-Za-z_]\w*)\.', text)
    if m:
        return m.group(1)
    return text


def _normalize_parallel_wrapper_frame(frame):
    func = str(_frame_get(frame, 2, "func", "") or "")
    cls = _extract_frame_class_name(func)
    if cls in PARALLEL_WRAPPER_CLASSES:
        return None
    new_func = func
    for prefix in PARALLEL_WRAPPER_PREFIXES:
        if func.startswith("nn.Module: " + prefix):
            new_func = "nn.Module: " + func[len("nn.Module: " + prefix):]
            break
        if func.startswith("nn.Module:" + prefix):
            new_func = "nn.Module:" + func[len("nn.Module:" + prefix):]
            break
        if func.startswith(prefix):
            new_func = func[len(prefix):]
            break
    return _frame_with_func(frame, new_func) if new_func != func else frame


def _dedup_consecutive_frames(frames):
    deduped = []
    last_key = None
    for frame in frames or []:
        key = (_frame_get(frame, 0, "file"), _frame_get(frame, 1, "line"), _frame_get(frame, 2, "func"))
        if key == last_key:
            continue
        deduped.append(frame)
        last_key = key
    return deduped

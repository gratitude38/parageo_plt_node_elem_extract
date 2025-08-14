
import io
import os
import re
import tempfile
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import h5py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Helpers & Utilities
# -------------------------

def bytes_to_str(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x

def decode_array(arr: np.ndarray) -> np.ndarray:
    """Decode byte arrays to strings where needed."""
    if isinstance(arr, np.ndarray) and (arr.dtype.kind in ["S", "O"]):
        try:
            return np.vectorize(bytes_to_str)(arr)
        except Exception:
            return arr.astype(str)
    return arr

def parse_int_list(user_text: str) -> List[int]:
    """
    Parse list like: "1,2,5-10, 15"
    Returns a sorted list of unique integers.
    """
    s = user_text.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    nums = set()
    for p in parts:
        if "-" in p:
            try:
                a, b = p.split("-", 1)
                a_i, b_i = int(a), int(b)
                lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                nums.update(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                nums.add(int(p))
            except ValueError:
                continue
    return sorted(nums)

def safe_read_dataset(dset: h5py.Dataset, max_preview: int = 1000) -> np.ndarray:
    """Read with a cap for preview; for full downloads read directly later."""
    data = dset[()]
    if isinstance(data, np.ndarray) and data.size > max_preview:
        if data.ndim == 1:
            data = data[:max_preview]
        else:
            sl = [slice(None)] * data.ndim
            sl[0] = slice(0, max_preview)
            data = data[tuple(sl)]
    return decode_array(np.array(data))

def list_groups(h: h5py.Group) -> List[str]:
    return [k for k, v in h.items() if isinstance(v, h5py.Group)]

def list_datasets(h: h5py.Group) -> List[str]:
    return [k for k, v in h.items() if isinstance(v, h5py.Dataset)]

def match_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    """Return the first key that matches any of the regex patterns (case-insensitive, full match)."""
    for pat in patterns:
        for k in keys:
            if re.fullmatch(pat, k, flags=re.IGNORECASE):
                return k
    return None

def extract_step_from_filename(name: str) -> Optional[int]:
    """Expect *_NNN.ext at the end OR ...StepXX_NNN.ext -> extract trailing NNN."""
    m = re.search(r"_(\d{1,})\.[^.]+$", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None

def get_dump_group(f: h5py.File, preferred_step: Optional[int]) -> Optional[str]:
    """Find Dump_XXX group with preference for preferred_step."""
    dumps = [k for k in f.keys() if k.startswith("Dump_")]
    if not dumps:
        return None
    if preferred_step is not None:
        cand = f"Dump_{preferred_step:03d}"
        if cand in f:
            return cand
    if len(dumps) == 1:
        return dumps[0]
    steps = []
    for d in dumps:
        m = re.match(r"Dump_(\d+)$", d)
        steps.append((int(m.group(1)) if m else -1, d))
    steps.sort()
    return steps[-1][1]

def path_join(*parts):
    """Join HDF5-style paths (no leading/trailing //)."""
    return "/".join([parts[0].strip("/")] + [q.strip("/") for q in parts[1:]])

def invert_mapping(numbers: np.ndarray) -> Dict[int, int]:
    """Index is internal ID; value is actual number. Return actual_number -> internal_id."""
    inv = {}
    for idx, val in enumerate(np.array(numbers).reshape(-1).tolist()):
        try:
            inv[int(val)] = int(idx)
        except Exception:
            pass
    return inv

def classify_variables(g: h5py.Group, node_count: int, elem_count: int, exclude_keys: List[str]) -> Tuple[List[str], List[str]]:
    """Return (nodal_vars, element_vars) among numeric datasets (exclude_keys ignored)."""
    nodal, elem = [], []
    for k, v in g.items():
        if not isinstance(v, h5py.Dataset):
            continue
        if k in exclude_keys:
            continue
        # numeric-ish only
        if v.dtype.kind not in ("i", "u", "f"):
            try:
                np.array(v[()], dtype=float)
            except Exception:
                continue
        shape = v.shape
        if len(shape) == 0:
            continue
        n0 = shape[0]
        if n0 == node_count:
            nodal.append(k)
        elif n0 == elem_count:
            elem.append(k)
    return sorted(nodal), sorted(elem)

def get_dump_subgroups(f: h5py.File, dump_group: str) -> List[str]:
    """Return direct subgroups under Dump (generic; not hardcoded)."""
    return [k for k, v in f[dump_group].items() if isinstance(v, h5py.Group)]

# -------------------------
# Plot Styling (FT-like)
# -------------------------

def ft_style(fig: go.Figure, x_title: str, y_title_left: str, y_title_right: Optional[str] = None):
    """Apply an FT-like minimalist style to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#fff1e5",
        plot_bgcolor="#fff1e5",
        font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#262a33"),
        margin=dict(l=60, r=60 if y_title_right else 40, t=30, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        colorway=["#0f5499", "#d35400", "#6a4c93", "#00876c", "#e83f6f", "#7a5195", "#ef8354", "#2f4b7c"],
    )
    grid_color = "#d9d6ce"
    axis_color = "#262a33"
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=x_title)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left, secondary_y=False)
    if y_title_right:
        fig.update_yaxes(title_text=y_title_right, secondary_y=True)

# -------------------------
# Streamlit App State / Layout
# -------------------------

st.set_page_config(page_title="HDF5 .plt Viewer (Plotly, Minimal)", layout="wide")

if "files_meta" not in st.session_state:
    st.session_state.files_meta = {}  # step -> {"name": str, "path": str}
if "active_step_index" not in st.session_state:
    st.session_state.active_step_index = 0

# -------------------------
# Sidebar: Drag & Drop + Parts
# -------------------------

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Drop `.plt` files",
        type=["plt", "h5", "hdf5"],
        accept_multiple_files=True,
        help="Each file is one time step result; suffix like `_002.plt` orders steps.",
        key="uploader",
    )

    # Ingest uploads and force a refresh even if filenames are the same
    if uploads is not None and len(uploads) > 0:
        new_meta = {}
        for uf in uploads:
            suffix_step = extract_step_from_filename(uf.name)
            tmpdir = tempfile.mkdtemp(prefix="plt_")
            dst = os.path.join(tmpdir, uf.name)
            with open(dst, "wb") as out:
                out.write(uf.getbuffer())
            try:
                with h5py.File(dst, "r") as f:
                    dump_group = get_dump_group(f, suffix_step)
                    internal_step = None
                    if dump_group and "Step" in f[dump_group]:
                        try:
                            internal_step = int(np.array(f[dump_group]["Step"][()]).item())
                        except Exception:
                            internal_step = None
                    step = suffix_step if suffix_step is not None else internal_step
                    if step is None and dump_group:
                        m = re.match(r"Dump_(\d+)$", dump_group)
                        step = int(m.group(1)) if m else None
            except Exception as e:
                st.error(f"Failed to read {uf.name}: {e}")
                continue
            if step is None:
                step = max(st.session_state.files_meta.keys(), default=-1) + 1
            new_meta[step] = {"name": uf.name, "path": dst}
        if new_meta:
            st.session_state.files_meta.update(new_meta)
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.header("View")
    part_choice = st.radio("Available parts", ["Dump", "Equations", "Materials"], index=0, help="Only one section shown at a time.")

# If no files loaded, stop here
if not st.session_state.files_meta:
    st.info("No files loaded yet. Use the sidebar to drop .plt files.")
    st.stop()

# -------------------------
# Step Navigation (Top Bar)
# -------------------------

steps_sorted = sorted(st.session_state.files_meta.keys())
cols = st.columns([1,1,3,1,1])
with cols[0]:
    if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.active_step_index <= 0):
        st.session_state.active_step_index = max(0, st.session_state.active_step_index - 1)
with cols[1]:
    if st.button("Next ▶", use_container_width=True, disabled=st.session_state.active_step_index >= len(steps_sorted)-1):
        st.session_state.active_step_index = min(len(steps_sorted)-1, st.session_state.active_step_index + 1)
with cols[2]:
    active_step = steps_sorted[st.session_state.active_step_index]
    st.markdown(f"### Active time step: **{active_step}**")
with cols[3]:
    selected_step = st.selectbox("Jump to step", steps_sorted, index=st.session_state.active_step_index, key="jump")
    if selected_step != active_step:
        st.session_state.active_step_index = steps_sorted.index(selected_step)
        active_step = selected_step
with cols[4]:
    if st.button("Clear files", help="Forget uploaded files"):
        st.session_state.files_meta = {}
        st.session_state.active_step_index = 0
        st.rerun()

meta = st.session_state.files_meta[active_step]
st.caption(f"File: {meta['name']}")

# -------------------------
# Summary helpers
# -------------------------

@st.cache_data(show_spinner=False)
def summarize_file(path: str, preferred_step: Optional[int]) -> Dict[str, Any]:
    info = {"path": path, "preferred_step": preferred_step}
    with h5py.File(path, "r") as f:
        dump_group = get_dump_group(f, preferred_step)
        info["dump_group"] = dump_group
        time = None; step = None; reset = None
        if dump_group:
            g = f[dump_group]
            time = float(np.array(g["Time"][()]).item()) if "Time" in g else None
            step = int(np.array(g["Step"][()]).item()) if "Step" in g else preferred_step
            reset = int(np.array(g["Reset_time_stage"][()]).item()) if "Reset_time_stage" in g else None
        info["time"] = time; info["step"] = step; info["reset_time_stage"] = reset

        info["has_equations"] = "Equations" in f
        info["has_materials"] = "Materials" in f
        info["dump_subgroups"] = get_dump_subgroups(f, dump_group) if dump_group else []
        info["equations_items"] = list(f["Equations"].keys()) if "Equations" in f else []
        info["materials_items"] = list(f["Materials"].keys()) if "Materials" in f else []
    return info

summary = summarize_file(meta["path"], active_step)

with st.sidebar:
    st.caption(f"Time: {summary.get('time')}, Step: {summary.get('step')}, Reset: {summary.get('reset_time_stage')}")

if part_choice == "Equations" and not summary["has_equations"]:
    st.warning("No 'Equations' in this file. Showing Dump instead.")
    part_choice = "Dump"
if part_choice == "Materials" and not summary["has_materials"]:
    st.warning("No 'Materials' in this file. Showing Dump instead.")
    part_choice = "Dump"
if part_choice == "Dump" and not summary["dump_group"]:
    st.warning("No 'Dump_*' group found in this file.")
    st.stop()

# -------------------------
# EQUATIONS (minimal, side-by-side)
# -------------------------

def render_equations(path: str):
    with h5py.File(path, "r") as f:
        g = f["Equations"]
        items = list(g.keys())
        c1, c2 = st.columns([1,2])
        with c1:
            sel = st.selectbox("Items", items, key="eq_item")
        with c2:
            obj = g[sel]
            records = []
            if isinstance(obj, h5py.Group):
                for k, v in obj.items():
                    if isinstance(v, h5py.Dataset):
                        data = v[()]
                        if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                            val = data.item() if np.isscalar(data) else np.array(data).flatten()[0]
                            records.append({"name": k, "value": bytes_to_str(val)})
            else:
                data = obj[()]
                if np.isscalar(data):
                    records.append({"name": sel, "value": bytes_to_str(data.item())})
                elif isinstance(data, np.ndarray) and data.size == 1:
                    records.append({"name": sel, "value": bytes_to_str(np.array(data).flatten()[0])})
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True)
            else:
                st.info("No scalar values found for this selection.")

# -------------------------
# MATERIALS (three side-by-side)
# -------------------------

def render_materials(path: str):
    with h5py.File(path, "r") as f:
        g = f["Materials"]
        items = list(g.keys())
        c1, c2, c3 = st.columns([1,1,1.6])
        with c1:
            sel = st.selectbox("Items", items, key="mat_item")
        names, values = [], []
        obj = g[sel]
        if isinstance(obj, h5py.Group):
            for k, v in obj.items():
                if isinstance(v, h5py.Dataset):
                    data = v[()]
                    if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                        val = data.item() if np.isscalar(data) else np.array(data).flatten()[0]
                        names.append(k); values.append(bytes_to_str(val))
        else:
            data = obj[()]
            if np.isscalar(data):
                names.append(sel); values.append(bytes_to_str(data.item()))
            elif isinstance(data, np.ndarray) and data.size == 1:
                names.append(sel); values.append(bytes_to_str(np.array(data).flatten()[0]))
        with c2:
            st.write("**Names**")
            st.dataframe(pd.DataFrame({"name": names}), hide_index=True, use_container_width=True)
        with c3:
            st.write("**Values**")
            st.dataframe(pd.DataFrame({"value": values}), hide_index=True, use_container_width=True)

# -------------------------
# DUMP (generic, minimal)
# -------------------------

def get_mappings_for_group(g: h5py.Group) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    keys = list(g.keys())
    topo_key = match_key(keys, [r"Topology"])
    node_map_key = match_key(keys, [r"Contact\s*Node\s*number", r"Node\s*Numbers"])
    elem_map_key = match_key(keys, [r"Contact\s*Element\s*number", r"Element\s*Numbers", r"Element\s*numbers"])
    return topo_key, node_map_key, elem_map_key

def classify_in_group(g: h5py.Group):
    topo_key, node_map_key, elem_map_key = get_mappings_for_group(g)
    if not (topo_key and node_map_key and elem_map_key):
        return None
    topology = np.array(g[topo_key])
    node_numbers = np.array(g[node_map_key]).astype(int).reshape(-1)
    elem_numbers = np.array(g[elem_map_key]).astype(int).reshape(-1)
    node_inv = invert_mapping(node_numbers)
    elem_inv = invert_mapping(elem_numbers)
    num_elems = topology.shape[0]
    num_nodes = node_numbers.shape[0]
    exclude = [topo_key, node_map_key, elem_map_key]
    nodal_vars, elem_vars = classify_variables(g, num_nodes, num_elems, exclude)
    return {
        "node_inv": node_inv, "elem_inv": elem_inv,
        "num_nodes": num_nodes, "num_elems": num_elems,
        "nodal_vars": nodal_vars, "elem_vars": elem_vars
    }

def render_dump(path: str, dump_group: str, dump_subgroups: List[str]):
    with h5py.File(path, "r") as f:
        st.subheader("Dump")
        group_name = st.selectbox("Group inside Dump", dump_subgroups, key="dump_group_choice")
        g = f[path_join(dump_group, group_name)]
        ctx = classify_in_group(g)
        if not ctx:
            st.error("Could not find Topology / Node Numbers / Element Numbers in this group.")
            return

        mode = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="dump_var_type")
        var_options = ctx["nodal_vars"] if mode == "Nodal" else ctx["elem_vars"]
        vars_pick = st.multiselect("Variables to plot/table", var_options, key="dump_vars_pick")

        numbers_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs)", placeholder="e.g., 1,2,3-10", key="dump_numbers")
        numbers_list = parse_int_list(numbers_text)

        sec_choice = None
        if len(vars_pick) >= 2:
            sec_choice = st.selectbox("Secondary y-axis (optional)", ["None"] + vars_pick, index=0, key="dump_secondary")
            if sec_choice == "None":
                sec_choice = None

        if not vars_pick or not numbers_list:
            st.info("Select one or more variables and enter a list of numbers to plot/table.")
            return

        inv = ctx["node_inv"] if mode == "Nodal" else ctx["elem_inv"]
        ids = [inv.get(n) for n in numbers_list]
        missing = [n for n, i in zip(numbers_list, ids) if i is None]
        ids = [i for i in ids if i is not None]
        if missing:
            st.warning(f"Numbers not found and skipped: {missing}")

        data = {"entity_number": [n for n, i in zip(numbers_list, ids + [None]*(len(numbers_list)-len(ids))) if i is not None]}
        for var in vars_pick:
            arr = np.array(g[var])
            if arr.ndim == 1:
                vals = [arr[i] if i is not None and i < arr.shape[0] else np.nan for i in ids]
            else:
                vals = [arr[i, 0] if i is not None and i < arr.shape[0] else np.nan for i in ids]
            vals = [float(np.array(v).item()) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in vals]
            data[var] = vals

        df = pd.DataFrame(data)

        if sec_choice:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for var in vars_pick:
                if var == sec_choice:
                    continue
                fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[var], mode="lines+markers", name=var), secondary_y=False)
            fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[sec_choice], mode="lines+markers", name=f"{sec_choice} (right)"), secondary_y=True)
            ft_style(fig, x_title=f"{mode} number", y_title_left="Primary", y_title_right="Secondary")
        else:
            fig = go.Figure()
            for var in vars_pick:
                fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[var], mode="lines+markers", name=var))
            ft_style(fig, x_title=f"{mode} number", y_title_left="Value")

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# -------------------------
# Render chosen part
# -------------------------

summary = summarize_file(meta["path"], active_step)

with st.sidebar:
    st.caption(f"Time: {summary.get('time')}, Step: {summary.get('step')}, Reset: {summary.get('reset_time_stage')}")

if part_choice == "Equations" and not summary["has_equations"]:
    st.warning("No 'Equations' in this file. Showing Dump instead.")
    part_choice = "Dump"
if part_choice == "Materials" and not summary["has_materials"]:
    st.warning("No 'Materials' in this file. Showing Dump instead.")
    part_choice = "Dump"
if part_choice == "Dump" and not summary["dump_group"]:
    st.warning("No 'Dump_*' group found in this file.")
    st.stop()

if part_choice == "Equations":
    render_equations(meta["path"])
elif part_choice == "Materials":
    render_materials(meta["path"])
else:
    if not summary["dump_subgroups"]:
        st.info("No subgroups under Dump.")
    else:
        render_dump(meta["path"], summary["dump_group"], summary["dump_subgroups"])

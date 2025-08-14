

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

# =========================
# App config
# =========================
st.set_page_config(page_title="HDF5 .plt Viewer (FEM, Plotly)", layout="wide")

# =========================
# Session init
# =========================
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "files" not in st.session_state:
    # list of entries: {"id": str, "name": str, "path": str, "step": int|None, "dump_group": str|None, "time": float|None}
    st.session_state.files = []
if "active_index" not in st.session_state:
    st.session_state.active_index = 0
if "flash_msg" not in st.session_state:
    st.session_state.flash_msg = None

# Persistent UI selections
for k, v in {
    "part_choice": "Dump",
    "dump_container": None,
    "dump_subgroup": None,
    "dump_mode": "Nodal",
    "dump_vars": [],
    "dump_numbers": "",
    "dump_secondary": "None",
}.items():
    st.session_state.setdefault(k, v)

# =========================
# Helpers
# =========================

def bytes_to_str(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x

def parse_int_list(user_text: str) -> List[int]:
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

def get_dump_group(f: h5py.File, preferred_step: Optional[int]) -> Optional[str]:
    """Return Dump_XXX group; prefer preferred_step when present."""
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

def extract_step_from_filename(name: str) -> Optional[int]:
    m = re.search(r"_(\d{1,})\.[^.]+$", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None

def list_groups(h: h5py.Group) -> List[str]:
    return [k for k, v in h.items() if isinstance(v, h5py.Group)]

def path_join(*parts):
    return "/".join([parts[0].strip("/")] + [q.strip("/") for q in parts[1:]])

def invert_mapping(numbers: np.ndarray) -> Dict[int, int]:
    inv = {}
    flat = np.array(numbers).reshape(-1).tolist()
    for idx, val in enumerate(flat):
        try:
            inv[int(val)] = int(idx)
        except Exception:
            pass
    return inv

def find_mapping_keys(g: h5py.Group, container_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (topology_key, node_map_key, elem_map_key) with rules:
       - If container is 'Contact' (case-insensitive), use 'Contact Node number' & 'Contact Element number'
       - Else, use 'Node Numbers' & 'Element Numbers'.
       Topology is optional.
    """
    keys = list(g.keys())
    topo_key = "Topology" if "Topology" in keys else None
    node_key = None
    elem_key = None
    if container_name.lower() == "contact":
        if "Contact Node number" in keys: node_key = "Contact Node number"
        if "Contact Element number" in keys: elem_key = "Contact Element number"
    else:
        if "Node Numbers" in keys: node_key = "Node Numbers"
        if "Element Numbers" in keys: elem_key = "Element Numbers"
    # mild fallback: case-insensitive
    if node_key is None:
        for k in keys:
            if re.fullmatch(r"(?i)contact\s*node\s*number", k) and container_name.lower()=="contact":
                node_key = k; break
            if re.fullmatch(r"(?i)node\s*numbers", k) and container_name.lower()!="contact":
                node_key = k; break
    if elem_key is None:
        for k in keys:
            if re.fullmatch(r"(?i)contact\s*element\s*number", k) and container_name.lower()=="contact":
                elem_key = k; break
            if re.fullmatch(r"(?i)element\s*numbers", k) and container_name.lower()!="contact":
                elem_key = k; break
    return topo_key, node_key, elem_key

def classify_variables(g: h5py.Group, node_count: int, elem_count: int, exclude: List[str]) -> Tuple[List[str], List[str]]:
    nodal, elem = [], []
    for k, v in g.items():
        if not isinstance(v, h5py.Dataset): continue
        if k in exclude: continue
        shape = v.shape
        if len(shape)==0: continue
        n0 = shape[0]
        if n0 == node_count: nodal.append(k)
        elif n0 == elem_count: elem.append(k)
    return sorted(nodal), sorted(elem)

def ft_style(fig: go.Figure, x_title: str, y_title_left: str, y_title_right: Optional[str] = None):
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
    if y_title_right is None:
        # Single y-axis figure
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left)
    else:
        # Dual y-axis (requires subplots with secondary_y)
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left, secondary_y=False)
        fig.update_yaxes(title_text=y_title_right, secondary_y=True)

# =========================
# File analysis
# =========================

def analyze_file(dst_path: str, orig_name: str) -> Dict[str, Any]:
    suffix_step = extract_step_from_filename(orig_name)
    dump_group = None
    tval = None
    step_val = suffix_step
    try:
        with h5py.File(dst_path, "r") as f:
            dump_group = get_dump_group(f, suffix_step)
            if dump_group and "Time" in f[dump_group]:
                try:
                    tval = float(np.array(f[dump_group]["Time"][()]).item())
                except Exception:
                    tval = None
            if dump_group and "Step" in f[dump_group]:
                try:
                    step_val = int(np.array(f[dump_group]["Step"][()]).item())
                except Exception:
                    pass
    except Exception:
        pass
    return {"name": orig_name, "path": dst_path, "step": step_val, "dump_group": dump_group, "time": tval}

# =========================
# Sidebar: uploader + part toggle
# =========================

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Drop `.plt` files",
        type=["plt", "h5", "hdf5"],
        accept_multiple_files=True,
        help="Each file is one time step; filename suffix like `_002.plt` is used for ordering.",
        key=f"uploader_{st.session_state.upload_key}",
    )

    if uploads:
        added_ids = []
        for uf in uploads:
            tmpdir = tempfile.mkdtemp(prefix="plt_")
            dst = os.path.join(tmpdir, uf.name)
            with open(dst, "wb") as out:
                out.write(uf.read())
            meta = analyze_file(dst, uf.name)

            # Build a unique id; allow duplicates even if same step and same filename (suffix later)
            base_id = f"{meta['step'] or 'NA'}::{meta['name']}"
            unique_id = base_id
            i = 1
            existing_ids = {e["id"] for e in st.session_state.files}
            while unique_id in existing_ids:
                i += 1
                unique_id = f"{base_id} ({i})"
            meta["id"] = unique_id
            st.session_state.files.append(meta)
            added_ids.append(unique_id)

        # focus the last added file
        if added_ids:
            st.session_state.active_id = added_ids[-1]
        st.session_state.flash_msg = f"Files loaded: {added_ids}"
        st.session_state.upload_key += 1
        st.rerun()

    # flash after rerun
    if st.session_state.flash_msg:
        st.success(st.session_state.flash_msg)
        st.session_state.flash_msg = None

    st.divider()
    st.header("View")
    st.session_state.part_choice = st.radio(
        "Available parts",
        ["Dump", "Equations", "Materials"],
        index=0,
        help="Only one section shown at a time.",
        key="part_choice_radio",
    )

# Guard if nothing loaded
if not st.session_state.files:
    st.info("No files loaded yet. Use the sidebar to drop `.plt` files.")
    st.stop()

# Sort files by (step, name) for navigation
def sort_key(entry):
    s = entry["step"]
    return (999999 if s is None else int(s), entry["name"])

files_sorted = sorted(st.session_state.files, key=sort_key)

# Compute active index preferring active_id if set
if st.session_state.active_id is not None:
    try:
        st.session_state.active_index = next(i for i, e in enumerate(files_sorted) if e.get("id") == st.session_state.active_id)
    except StopIteration:
        st.session_state.active_index = min(max(0, st.session_state.active_index), len(files_sorted)-1)
else:
    st.session_state.active_index = min(max(0, st.session_state.active_index), len(files_sorted)-1)
active = files_sorted[st.session_state.active_index]

# =========================
# Top bar: step/file navigation
# =========================

cols = st.columns([1,1,4,2,1])
with cols[0]:
    if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.active_index <= 0):
        st.session_state.active_index = max(0, st.session_state.active_index - 1)
with cols[1]:
    if st.button("Next ▶", use_container_width=True, disabled=st.session_state.active_index >= len(files_sorted)-1):
        st.session_state.active_index = min(len(files_sorted)-1, st.session_state.active_index + 1)
with cols[2]:
    label = f"Step {active['step'] if active['step'] is not None else 'NA'} — {active['name']}"
    st.markdown(f"### {label}")
with cols[3]:
    # drop-down of all files
    labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']} ({e['id']})" for e in files_sorted]
    sel = st.selectbox("Jump to file", options=list(range(len(files_sorted))), format_func=lambda i: labels[i], index=st.session_state.active_index, key="jump_file")
    if sel != st.session_state.active_index:
        st.session_state.active_index = sel
        st.session_state.active_id = files_sorted[sel].get("id")
with cols[4]:
    if st.button("Clear files", help="Forget uploaded files"):
        st.session_state.files = []
        st.session_state.active_index = 0
        st.session_state.active_id = None
        st.session_state.upload_key += 1
        st.rerun()

st.caption(f"Dump group: {active.get('dump_group')} | Time: {active.get('time')} | File: {active.get('name')}")

# =========================
# Equations (side-by-side, scalars only)
# =========================

def render_equations(entry: Dict[str, Any]):
    path = entry["path"]
    try:
        with h5py.File(path, "r") as f:
            if "Equations" not in f:
                st.info("No 'Equations' group in this file."); return
            g = f["Equations"]
            items = list(g.keys())
            if not items:
                st.info("No items under 'Equations'."); return
            c1, c2 = st.columns([1,2])
            with c1:
                sel = st.selectbox("Items", items, key="eq_item")
            with c2:
                obj = g[sel]
                rows = []
                if isinstance(obj, h5py.Group):
                    for k, v in obj.items():
                        if isinstance(v, h5py.Dataset):
                            data = v[()]
                            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                                val = data.item() if np.isscalar(data) else np.array(data).flatten()[0]
                                rows.append({"name": k, "value": bytes_to_str(val)})
                else:
                    data = obj[()]
                    if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                        val = data.item() if np.isscalar(data) else np.array(data).flatten()[0]
                        rows.append({"name": sel, "value": bytes_to_str(val)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No scalar values in this selection.")
    except Exception as e:
        st.error(f"Equations read error: {e}")

# =========================
# Materials (three columns; scalars only)
# =========================

def render_materials(entry: Dict[str, Any]):
    path = entry["path"]
    try:
        with h5py.File(path, "r") as f:
            if "Materials" not in f:
                st.info("No 'Materials' group in this file."); return
            g = f["Materials"]
            items = list(g.keys())
            if not items:
                st.info("No items under 'Materials'."); return
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
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                    val = data.item() if np.isscalar(data) else np.array(data).flatten()[0]
                    names.append(sel); values.append(bytes_to_str(val))
            with c2:
                st.write("**Names**")
                st.dataframe(pd.DataFrame({"name": names}), hide_index=True, use_container_width=True)
            with c3:
                st.write("**Values**")
                st.dataframe(pd.DataFrame({"value": values}), hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Materials read error: {e}")

# =========================
# Dump (two-level containers -> subgroups)
# =========================

def render_dump(entry: Dict[str, Any]):
    path = entry["path"]
    dump_group = entry["dump_group"]
    if not dump_group:
        st.info("This file has no Dump_* group."); return
    try:
        with h5py.File(path, "r") as f:
            root = f[dump_group]
            containers = [k for k, v in root.items() if isinstance(v, h5py.Group)]
            if not containers:
                st.info("Dump group has no containers."); return

            # Container select (remember prior selection)
            if st.session_state.dump_container not in containers:
                st.session_state.dump_container = containers[0]
            container = st.selectbox("Container", containers, index=containers.index(st.session_state.dump_container), key="dump_container_select")
            st.session_state.dump_container = container

            # Subgroup select
            subgroups = list_groups(root[container])
            if not subgroups:
                st.info("Selected container has no subgroups."); return
            if st.session_state.dump_subgroup not in subgroups:
                st.session_state.dump_subgroup = subgroups[0]
            subgroup = st.selectbox("Subgroup", subgroups, index=subgroups.index(st.session_state.dump_subgroup), key="dump_subgroup_select")
            st.session_state.dump_subgroup = subgroup

            g = root[container][subgroup]

            # Mapping keys based on container
            topo_key, node_key, elem_key = find_mapping_keys(g, container_name=container)
            if not (node_key and elem_key):
                st.error("Required mapping datasets not found (Node/Element Numbers)."); return
            node_numbers = np.array(g[node_key]).astype(int).reshape(-1)
            elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
            node_inv = invert_mapping(node_numbers)
            elem_inv = invert_mapping(elem_numbers)

            # Mode (explicit)
            mode = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, index=0 if st.session_state.dump_mode=="Nodal" else 1, key="dump_mode_radio")
            st.session_state.dump_mode = mode
            num_nodes, num_elems = node_numbers.shape[0], elem_numbers.shape[0]

            # Classify variables
            exclude = [k for k in [topo_key, node_key, elem_key] if k]
            nodal_vars, elem_vars = classify_variables(g, node_count=num_nodes, elem_count=num_elems, exclude=exclude)
            var_options = nodal_vars if mode=="Nodal" else elem_vars
            if not var_options:
                st.info(f"No {mode.lower()} variables detected in this subgroup."); return

            # Variables to plot (remember selection)
            default_vars = [v for v in st.session_state.dump_vars if v in var_options]
            vars_pick = st.multiselect("Variables", var_options, default=default_vars, key="dump_vars_select")
            st.session_state.dump_vars = vars_pick

            # Per-variable component selectors
            var_components = {}
            for var in vars_pick:
                arr = np.array(g[var])
                comp_max = arr.shape[1]-1 if (arr.ndim >= 2) else 0
                if comp_max > 0:
                    key = f"comp::{container}::{subgroup}::{var}"
                    comp_default = st.session_state.get(key, 0)
                    comp_idx = st.number_input(f"{var} • component", min_value=0, max_value=comp_max, value=min(comp_default, comp_max), step=1, key=key)
                    var_components[var] = int(comp_idx)
                    st.session_state[key] = int(comp_idx)
                else:
                    var_components[var] = None

            # Numbers input (remember)
            numbers_default = st.session_state.dump_numbers
            numbers_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs)", value=numbers_default, placeholder="e.g., 1,2,3-10", key="dump_numbers_input")
            st.session_state.dump_numbers = numbers_text
            numbers_list = parse_int_list(numbers_text)

            # Secondary Y selection
            sec_options = ["None"] + vars_pick
            sec_default = st.session_state.dump_secondary if st.session_state.dump_secondary in sec_options else "None"
            sec_choice = st.selectbox("Secondary y-axis (optional)", sec_options, index=sec_options.index(sec_default), key="dump_secondary_select")
            st.session_state.dump_secondary = sec_choice

            if not vars_pick or not numbers_list:
                st.info("Choose at least one variable and enter a list of numbers to plot.")
                return

            # Map actual numbers -> internal IDs
            inv = node_inv if mode=="Nodal" else elem_inv
            ids = [inv.get(n) for n in numbers_list]
            missing = [n for n, i in zip(numbers_list, ids) if i is None]
            ids = [i for i in ids if i is not None]
            used_numbers = [n for n in numbers_list if inv.get(n) is not None]
            if missing:
                st.warning(f"Numbers not found and dropped: {missing}")

            # Build DataFrame with columns: entity_number + each var (label with component if present)
            df = pd.DataFrame({"entity_number": used_numbers})
            for var in vars_pick:
                arr = np.array(g[var])
                if arr.ndim == 1:
                    vals = [arr[i] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                    label = var
                else:
                    comp = var_components[var] or 0
                    vals = [arr[i, comp] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                    label = f"{var} [comp {comp}]"
                # coerce to float for plotting/table
                vals = [float(np.array(v).item()) if (isinstance(v, (np.generic,)) or np.isscalar(v)) else float(v) for v in vals]
                df[label] = vals

            # Plot
            if sec_choice != "None":
                # Map sec label (consider comp suffix)
                sec_label = None
                for col in df.columns[1:]:
                    if col.startswith(sec_choice):
                        sec_label = col; break
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for col in df.columns[1:]:
                    if col == sec_label:
                        continue
                    fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[col], mode="lines+markers", name=col), secondary_y=False)
                if sec_label is not None:
                    fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[sec_label], mode="lines+markers", name=f"{sec_label} (right)"), secondary_y=True)
                ft_style(fig, x_title=f"{mode} number", y_title_left="Primary", y_title_right="Secondary")
            else:
                fig = go.Figure()
                for col in df.columns[1:]:
                    fig.add_trace(go.Scattergl(x=df["entity_number"], y=df[col], mode="lines+markers", name=col))
                ft_style(fig, x_title=f"{mode} number", y_title_left="Value")

            st.plotly_chart(fig, use_container_width=True)

            # Exports
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button("Download plot (HTML)", data=html_bytes, file_name="plot.html", mime="text/html")
            try:
                png_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
                st.download_button("Download plot (PNG)", data=png_bytes, file_name="plot.png", mime="image/png")
            except Exception as e:
                st.caption("PNG export unavailable (kaleido not installed in this runtime).")

            # Table
            st.dataframe(df, use_container_width=True)
            # CSV export for table (kept as a convenience)
            st.download_button("Download table (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="table.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Dump read error: {e}")

# =========================
# Render chosen part
# =========================

if st.session_state.part_choice == "Equations":
    render_equations(active)
elif st.session_state.part_choice == "Materials":
    render_materials(active)
else:
    render_dump(active)


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
    st.session_state.files = []  # entries: {"id","name","path","step","dump_group","time"}
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
    """Return (topology_key, node_map_key, elem_map_key).
       - If container is 'Contact' (case-insensitive): 'Contact Node number' & 'Contact Element number'
       - Else: 'Node Numbers' & 'Element Numbers'.
       Topology is optional (unused).
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
    """Apply an FT-like style. Only use 'secondary_y' when the figure actually has it."""
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
        # Single y-axis
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left)
    else:
        # Dual y-axis (make_subplots required)
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
# Sidebar: uploader + view + dump navigators and remove buttons
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
        prev_count = len(st.session_state.files)
        added_ids = []
        for uf in uploads:
            tmpdir = tempfile.mkdtemp(prefix="plt_")
            dst = os.path.join(tmpdir, uf.name)
            with open(dst, "wb") as out:
                out.write(uf.read())
            meta = analyze_file(dst, uf.name)

            # Unique id; allow duplicates even if same step+filename by suffixing
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

        # If these are the first files, start from the smallest step
        if prev_count == 0 and st.session_state.files:
            smallest = sorted(st.session_state.files, key=lambda e: (999999 if e["step"] is None else int(e["step"]), e["name"]))[0]
            st.session_state.active_id = smallest["id"]
        elif added_ids:
            # Keep current selection; ensure active_id is valid
            if st.session_state.active_id is None:
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
        index=["Dump","Equations","Materials"].index(st.session_state.part_choice) if st.session_state.get("part_choice") else 0,
        help="Only one section shown at a time.",
        key="part_choice",
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

cols = st.columns([1,1,4,2])
with cols[0]:
    if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.active_index <= 0, key="prev_btn"):
        st.session_state.active_index = max(0, st.session_state.active_index - 1)
        st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
        st.rerun()
with cols[1]:
    if st.button("Next ▶", use_container_width=True, disabled=st.session_state.active_index >= len(files_sorted)-1, key="next_btn"):
        st.session_state.active_index = min(len(files_sorted)-1, st.session_state.active_index + 1)
        st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
        st.rerun()
with cols[2]:
    label = f"Step {active['step'] if active['step'] is not None else 'NA'} — {active['name']}"
    st.markdown(f"### {label}")
with cols[3]:
    labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']} ({e['id']})" for e in files_sorted]
    sel = st.selectbox("Jump to file", options=list(range(len(files_sorted))), format_func=lambda i: labels[i], index=st.session_state.active_index, key="jump_file")
    if sel != st.session_state.active_index:
        st.session_state.active_index = sel
        st.session_state.active_id = files_sorted[sel].get("id")
        st.rerun()

st.caption(f"Dump group: {active.get('dump_group')} | Time: {active.get('time')} | File: {active.get('name')}")

# =========================
# Sidebar: remove buttons + Dump navigator (Container/Subgroup)
# =========================

with st.sidebar:
    st.divider()
    st.subheader("Remove")
    colra, colrc = st.columns(2)
    with colra:
        if st.button("Remove all files", type="secondary", use_container_width=True, key="btn_rm_all"):
            st.session_state.files = []
            st.session_state.active_index = 0
            st.session_state.active_id = None
            st.session_state.upload_key += 1
            st.rerun()
    with colrc:
        if st.button("Remove current file", type="secondary", use_container_width=True, key="btn_rm_curr"):
            if st.session_state.files:
                curr_id = st.session_state.active_id or files_sorted[st.session_state.active_index]["id"]
                st.session_state.files = [x for x in st.session_state.files if x["id"] != curr_id]
                st.session_state.active_id = None
                st.rerun()

# Dump navigation in sidebar
def render_dump_sidebar(entry: Dict[str, Any]) -> bool:
    path = entry["path"]
    dump_group = entry["dump_group"]
    with h5py.File(path, "r") as f:
        if not dump_group or dump_group not in f:
            st.sidebar.info("This file has no Dump_* group.")
            return False
        root = f[dump_group]
        containers = [k for k, v in root.items() if isinstance(v, h5py.Group)]
        if not containers:
            st.sidebar.info("Dump group has no containers.")
            return False

        # Container select
        if st.session_state.dump_container not in containers:
            st.session_state.dump_container = containers[0]
        st.sidebar.selectbox("Container", containers, index=containers.index(st.session_state.dump_container), key="dump_container")

        # Subgroup select
        subgroups = list_groups(root[st.session_state.dump_container])
        if not subgroups:
            st.sidebar.info("Selected container has no subgroups.")
            return False
        if st.session_state.dump_subgroup not in subgroups:
            st.session_state.dump_subgroup = subgroups[0]
        st.sidebar.selectbox("Subgroup", subgroups, index=subgroups.index(st.session_state.dump_subgroup), key="dump_subgroup")

    return True

# =========================
# Equations (now shows vectors & scalars)
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
                if isinstance(obj, h5py.Group):
                    # pick a dataset inside
                    ds_names = [k for k, v in obj.items() if isinstance(v, h5py.Dataset)]
                    if not ds_names:
                        st.info("No datasets inside this group."); return
                    ds_sel = st.selectbox("Dataset", ds_names, key=f"eq_ds::{sel}")
                    dset = obj[ds_sel]
                    data = np.array(dset)
                    if data.ndim == 0:
                        st.table(pd.DataFrame([{"name": ds_sel, "value": bytes_to_str(data.item())}]))
                    elif data.ndim == 1:
                        df = pd.DataFrame({"index": np.arange(data.shape[0]), "value": data})
                        st.dataframe(df, use_container_width=True)
                    else:
                        # flatten last dims into columns if reasonable
                        cols = data.shape[1] if data.ndim == 2 else int(np.prod(data.shape[1:]))
                        flat = data.reshape(data.shape[0], cols)
                        colnames = [f"c{i}" for i in range(flat.shape[1])]
                        df = pd.DataFrame(flat, columns=colnames)
                        st.dataframe(df, use_container_width=True)
                else:
                    data = np.array(obj)
                    if data.ndim == 0:
                        st.table(pd.DataFrame([{"name": sel, "value": bytes_to_str(data.item())}]))
                    elif data.ndim == 1:
                        df = pd.DataFrame({"index": np.arange(data.shape[0]), "value": data})
                        st.dataframe(df, use_container_width=True)
                    else:
                        cols = data.shape[1] if data.ndim == 2 else int(np.prod(data.shape[1:]))
                        flat = data.reshape(data.shape[0], cols)
                        colnames = [f"c{i}" for i in range(flat.shape[1])]
                        df = pd.DataFrame(flat, columns=colnames)
                        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Equations read error: {e}")

# =========================
# Materials (combine name & value into one table)
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
            c1, c2 = st.columns([1,2])
            with c1:
                sel = st.selectbox("Items", items, key="mat_item")
            with c2:
                rows = []
                obj = g[sel]
                if isinstance(obj, h5py.Group):
                    for k, v in obj.items():
                        if isinstance(v, h5py.Dataset):
                            data = np.array(v)
                            if data.ndim == 0 or data.size == 1:
                                val = data.item() if data.ndim == 0 else data.flatten()[0]
                                rows.append({"name": k, "value": bytes_to_str(val)})
                            else:
                                # show shape summary and head
                                flat = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(-1, 1)
                                head = flat[: min(20, flat.shape[0])]
                                rows.append({"name": k, "value": f"array{tuple(data.shape)} (showing first {head.shape[0]} rows)"})
                else:
                    data = np.array(obj)
                    if data.ndim == 0 or data.size == 1:
                        val = data.item() if data.ndim == 0 else data.flatten()[0]
                        rows.append({"name": sel, "value": bytes_to_str(val)})
                    else:
                        flat = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(-1, 1)
                        head = flat[: min(20, flat.shape[0])]
                        rows.append({"name": sel, "value": f"array{tuple(data.shape)} (showing first {head.shape[0]} rows)"})
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No values to show.")
    except Exception as e:
        st.error(f"Materials read error: {e}")

# =========================
# Dump (container/subgroup in sidebar; plotting in main pane)
# =========================

def render_dump_main(entry: Dict[str, Any]):
    path = entry["path"]
    dump_group = entry["dump_group"]
    if not dump_group:
        st.info("This file has no Dump_* group.")
        return
    try:
        with h5py.File(path, "r") as f:
            root = f[dump_group]
            container = st.session_state.dump_container
            subgroup = st.session_state.dump_subgroup
            if container is None or subgroup is None or container not in root or subgroup not in root[container]:
                st.info("Pick a Container and Subgroup in the sidebar.")
                return
            g = root[container][subgroup]

            # Mapping keys based on container
            topo_key, node_key, elem_key = find_mapping_keys(g, container_name=container)
            if not (node_key and elem_key):
                st.error("Required mapping datasets not found (Node/Element Numbers).")
                return
            node_numbers = np.array(g[node_key]).astype(int).reshape(-1)
            elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
            node_inv = invert_mapping(node_numbers)
            elem_inv = invert_mapping(elem_numbers)

            # Controls row (Variables + Numbers + Secondary) - persistent but no manual reassignment
            mode = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="dump_mode")
            num_nodes, num_elems = node_numbers.shape[0], elem_numbers.shape[0]
            exclude = [k for k in [topo_key, node_key, elem_key] if k]
            nodal_vars, elem_vars = classify_variables(g, node_count=num_nodes, elem_count=num_elems, exclude=exclude)
            var_options = nodal_vars if mode=="Nodal" else elem_vars

            col_vars, col_nums, col_sec = st.columns([2,2,1])
            with col_vars:
                vars_pick = st.multiselect("Variables", var_options, key="dump_vars")
            with col_nums:
                label_numbers = "Enter nodal NUMBERS (not IDs)" if mode=="Nodal" else "Enter element NUMBERS (not IDs)"
                numbers_text = st.text_input(label_numbers, value=st.session_state.get("dump_numbers",""), placeholder="e.g., 1,2,3-10", key="dump_numbers")
                numbers_list = parse_int_list(numbers_text)
            with col_sec:
                sec_options = ["None"] + vars_pick
                if st.session_state.get("dump_secondary") not in sec_options:
                    st.session_state.dump_secondary = "None"
                sec_choice = st.selectbox("Secondary y-axis", sec_options, index=sec_options.index(st.session_state.dump_secondary), key="dump_secondary")

            if not vars_pick or not numbers_list:
                st.info("Choose at least one variable and enter a list of numbers to plot.")
                return

            # Components row (one number_input per vector variable)
            vec_vars = []
            for var in vars_pick:
                arr = np.array(g[var])
                if arr.ndim >= 2 and arr.shape[1] > 1:
                    vec_vars.append((var, arr.shape[1]-1))
            var_components = {v: 0 for v in vars_pick}
            if vec_vars:
                st.markdown("**Components**")
                comp_cols = st.columns(len(vec_vars))
                for (var, comp_max), col in zip(vec_vars, comp_cols):
                    with col:
                        key = f"comp::{container}::{subgroup}::{var}"
                        default = int(st.session_state.get(key, 0))
                        default = max(0, min(default, comp_max))
                        try:
                            comp_idx = st.number_input(f"{var}", min_value=0, max_value=comp_max, value=default, step=1, key=key)
                        except Exception:
                            comp_idx = st.number_input(f"{var}", min_value=0, max_value=comp_max, value=default, step=1, key=key+"::v2")
                        var_components[var] = int(comp_idx)

            # Map actual numbers -> internal IDs
            inv = node_inv if mode=="Nodal" else elem_inv
            ids = [inv.get(n) for n in numbers_list]
            missing = [n for n, i in zip(numbers_list, ids) if i is None]
            ids = [i for i in ids if i is not None]
            used_numbers = [n for n in numbers_list if inv.get(n) is not None]
            if missing:
                st.warning(f"Numbers not found and dropped: {missing}")

            # Build DataFrame with columns: Node/Element Number + each var (label with comp if present)
            first_col_name = "Node Number" if mode=="Nodal" else "Element Number"
            df = pd.DataFrame({first_col_name: used_numbers})
            labels_map = {}
            for var in vars_pick:
                arr = np.array(g[var])
                if arr.ndim == 1:
                    vals = [arr[i] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                    label = var
                else:
                    comp = var_components.get(var, 0)
                    vals = [arr[i, comp] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                    label = f"{var} [comp {comp}]"
                clean = []
                for v in vals:
                    try:
                        clean.append(float(np.array(v).item()))
                    except Exception:
                        clean.append(np.nan)
                df[label] = clean
                labels_map[var] = label

            # Plot
            if st.session_state.dump_secondary != "None":
                sec_var = st.session_state.dump_secondary
                sec_label = labels_map.get(sec_var)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                prim_labels = []
                for var in vars_pick:
                    col = labels_map[var]
                    if col == sec_label:
                        continue
                    prim_labels.append(col)
                    fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col), secondary_y=False)
                if sec_label is not None:
                    fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[sec_label], mode="lines+markers", name=f"{sec_label} (right)"), secondary_y=True)
                y_left_title = ", ".join(prim_labels) if prim_labels else "Value"
                y_right_title = sec_label or "Secondary"
                ft_style(fig, x_title=first_col_name, y_title_left=y_left_title, y_title_right=y_right_title)
            else:
                fig = go.Figure()
                prim_labels = []
                for var in vars_pick:
                    col = labels_map[var]
                    prim_labels.append(col)
                    fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col))
                y_left_title = ", ".join(prim_labels) if prim_labels else "Value"
                ft_style(fig, x_title=first_col_name, y_title_left=y_left_title)

            st.plotly_chart(fig, use_container_width=True)

            # Exports
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button("Download plot (HTML)", data=html_bytes, file_name="plot.html", mime="text/html")
            try:
                png_bytes = fig.to_image(format="png", scale=2)  # needs kaleido
                st.download_button("Download plot (PNG)", data=png_bytes, file_name="plot.png", mime="image/png")
            except Exception:
                st.caption("PNG export unavailable (kaleido not installed in this runtime).")

            # Table
            st.dataframe(df, use_container_width=True)
            st.download_button("Download table (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="table.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Dump read error: {e}")

# =========================
# Render chosen part
# =========================

# Dump: put container/subgroup in sidebar, plotting in main
def render_dump(entry: Dict[str, Any]):
    if render_dump_sidebar(entry=entry):
        render_dump_main(entry=entry)

if st.session_state.part_choice == "Equations":
    render_equations(active)
elif st.session_state.part_choice == "Materials":
    render_materials(active)
else:
    render_dump(active)

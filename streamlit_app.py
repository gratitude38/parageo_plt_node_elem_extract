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

st.set_page_config(page_title="HDF5 .plt Viewer (FEM, Plotly)", layout="wide")

# -----------------------------
# Session defaults
# -----------------------------
def ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

ss_default("upload_key", 0)
ss_default("active_id", None)
ss_default("files", [])                 # list of {id,name,path,step,dump_group,time}
ss_default("active_index", 0)
ss_default("flash_msg", None)

# Global intent (locked unless user changes)
ss_default("desired_container", None)          # string
ss_default("desired_subgroup_map", {})         # {container -> subgroup}

# Section choice
ss_default("part_choice_radio", "Dump")

# Persistent store per logical group "container::subgroup" (normalized)
# store[group]    = {"mode": "Nodal"/"Element", "vars":[...], "nums":"1,2,3", "sec":"var/None", "comp":{var:int}}
# store_ts[group] = {"mode": "Nodal"/"Element", "var":"...", "nums":"1,2,3", "comp":{var:int}}
ss_default("store", {})
ss_default("store_ts", {})

# Which group is currently bound to the stable widgets
ss_default("current_group_norm", None)

# -----------------------------
# Helpers
# -----------------------------
def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def bytes_to_str(x):
    if isinstance(x, (bytes, bytearray, np.bytes_, np.void)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x

def parse_int_list(user_text: str) -> List[int]:
    s = (user_text or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    nums = set()
    for p in parts:
        if not p:
            continue
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
    if node_key is None:
        for k in keys:
            if container_name.lower() == "contact" and re.fullmatch(r"(?i)contact\s*node\s*number", k):
                node_key = k; break
            if container_name.lower() != "contact" and re.fullmatch(r"(?i)node\s*numbers", k):
                node_key = k; break
    if elem_key is None:
        for k in keys:
            if container_name.lower() == "contact" and re.fullmatch(r"(?i)contact\s*element\s*number", k):
                elem_key = k; break
            if container_name.lower() != "contact" and re.fullmatch(r"(?i)element\s*numbers", k):
                elem_key = k; break
    return topo_key, node_key, elem_key

def classify_variables(g: h5py.Group, node_count: int, elem_count: int, exclude: List[str]) -> Tuple[List[str], List[str]]:
    nodal, elem = [], []
    for k, v in g.items():
        if not isinstance(v, h5py.Dataset):
            continue
        if k in exclude:
            continue
        shp = v.shape
        if len(shp) == 0:
            continue
        n0 = shp[0]
        if n0 == node_count:
            nodal.append(k)
        elif n0 == elem_count:
            elem.append(k)
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
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left)
    else:
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor=axis_color, ticks="outside", title_text=y_title_left, secondary_y=False)
        fig.update_yaxes(title_text=y_title_right, secondary_y=True)

def axis_title_from_labels(labels: List[str], fallback: str = "Value") -> str:
    """Reduce legend labels into a single tidy axis title.
    - Strips suffixes like ' (right)', ' [comp k]', and ' @ Node ...'
    - If multiple base names, use the first; if none, fallback.
    """
    if not labels:
        return fallback
    cleaned = []
    for s in labels:
        if not s: 
            continue
        x = re.sub(r"\s*\(right\)$", "", s)
        x = re.sub(r"\s*\[comp.*?\]$", "", x)
        x = re.sub(r"\s*@\s*Node.*$", "", x)
        x = re.sub(r"\s*@\s*Elem.*$", "", x)
        cleaned.append(x.strip())
    if not cleaned:
        return fallback
    # If all the same, use that; otherwise use the first base name
    base = cleaned[0]
    return base

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

def key_to_str(k) -> str:
    if isinstance(k, (bytes, np.bytes_)):
        try:
            return k.decode("utf-8")
        except Exception:
            return str(k)
    return str(k)

def ci_match(name: Optional[str], options: List[str]) -> Optional[str]:
    if not name:
        return None
    low = name.lower()
    for o in options:
        if o.lower() == low:
            return o
    return None

# -----------------------------
# Sidebar: upload + housekeeping
# -----------------------------
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
        if st.session_state.files:
            files_sorted_tmp = sorted(st.session_state.files, key=lambda e: (999999 if e["step"] is None else int(e["step"]), e["name"]))
            st.session_state.active_id = files_sorted_tmp[0]["id"]
        st.session_state.flash_msg = f"Files loaded: {added_ids}"
        st.session_state.upload_key += 1
        st.rerun()

    if st.session_state.flash_msg:
        st.success(st.session_state.flash_msg)
        st.session_state.flash_msg = None

    if st.session_state.files:
        st.subheader("Sources")
        c1, c2 = st.columns(2)
        if c1.button("Remove current file"):
            if st.session_state.active_id is not None:
                st.session_state.files = [x for x in st.session_state.files if x["id"] != st.session_state.active_id]
                if st.session_state.files:
                    files_sorted_tmp = sorted(st.session_state.files, key=lambda e: (999999 if e["step"] is None else int(e["step"]), e["name"]))
                    st.session_state.active_id = files_sorted_tmp[0]["id"]
                else:
                    st.session_state.active_id = None
            st.rerun()
        if c2.button("Remove all files"):
            st.session_state.files = []
            st.session_state.active_id = None
            st.session_state.upload_key += 1
            st.rerun()

    st.divider()
    st.header("View")
    st.radio(
        "Available parts",
        ["Dump", "Equations", "Materials"],
        key="part_choice_radio",
        index=["Dump","Equations","Materials"].index(st.session_state.get("part_choice_radio","Dump")),
        help="Only one section shown at a time.",
    )

# -----------------------------
# Active file selection
# -----------------------------

if not st.session_state.files:
    st.info("No files loaded yet. Use the sidebar to drop `.plt` files.")
    st.stop()

def sort_key(entry):
    s = entry["step"]
    return (999999 if s is None else int(s), entry["name"])

files_sorted = sorted(st.session_state.files, key=sort_key)

if st.session_state.active_id is not None:
    try:
        st.session_state.active_index = next(i for i, e in enumerate(files_sorted) if e.get("id") == st.session_state.active_id)
    except StopIteration:
        st.session_state.active_index = 0
else:
    st.subheader("Spatial distribution (for a fixed time)")
    st.session_state.active_index = 0
    st.session_state.active_id = files_sorted[0]["id"]

active = files_sorted[st.session_state.active_index]

# Top bar
cols = st.columns([1,1,5,3])
with cols[0]:
    if st.button("◀ Prev", use_container_width=True, key="prev_btn", disabled=st.session_state.active_index <= 0):
        st.session_state.active_index = max(0, st.session_state.active_index - 1)
        st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
        st.rerun()
with cols[1]:
    if st.button("Next ▶", use_container_width=True, key="next_btn", disabled=st.session_state.active_index >= len(files_sorted)-1):
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

# -----------------------------
# Equations
# -----------------------------
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
                if "eq_item" not in st.session_state:
                    st.session_state["eq_item"] = items[0]
                idx = items.index(st.session_state["eq_item"]) if st.session_state["eq_item"] in items else 0
                sel = st.selectbox("Items", items, index=idx, key="eq_item")
            with c2:
                obj = g[sel]
                if isinstance(obj, h5py.Group):
                    cols = {}
                    max_len = 0
                    for k, v in obj.items():
                        if not isinstance(v, h5py.Dataset): continue
                        arr = np.array(v[()])
                        if arr.ndim == 0:
                            cols[k] = [bytes_to_str(arr.item())]; max_len = max(max_len, 1)
                        elif arr.ndim == 1:
                            cols[k] = arr.tolist(); max_len = max(max_len, len(cols[k]))
                        elif arr.ndim == 2:
                            for j in range(arr.shape[1]):
                                key = f"{k}[{j}]"
                                cols[key] = arr[:, j].tolist(); max_len = max(max_len, len(cols[key]))
                        else:
                            cols[k] = [f"array(shape={arr.shape})"]; max_len = max(max_len, 1)
                    for k, vals in cols.items():
                        if len(vals) < max_len:
                            cols[k] = vals + [None]*(max_len - len(vals))
                    st.dataframe(pd.DataFrame(cols), use_container_width=True)
                else:
                    arr = np.array(obj[()])
                    if arr.ndim == 0:
                        st.dataframe(pd.DataFrame({"value":[bytes_to_str(arr.item())]}), use_container_width=True, hide_index=True)
                    elif arr.ndim == 1:
                        st.dataframe(pd.DataFrame({"value":arr}), use_container_width=True)
                    elif arr.ndim == 2:
                        df = pd.DataFrame(arr, columns=[f"comp_{i}" for i in range(arr.shape[1])])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(f"Dataset with shape {arr.shape} not directly shown.")
    except Exception as e:
        st.error(f"Equations read error: {e}")

# -----------------------------
# Materials
# -----------------------------
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
                if "mat_item" not in st.session_state:
                    st.session_state["mat_item"] = items[0]
                idx = items.index(st.session_state["mat_item"]) if st.session_state["mat_item"] in items else 0
                sel = st.selectbox("Items", items, index=idx, key="mat_item")
            rows = []
            obj = g[sel]
            if isinstance(obj, h5py.Group):
                for k, v in obj.items():
                    if isinstance(v, h5py.Dataset):
                        arr = np.array(v[()])
                        if arr.ndim == 0:
                            rows.append({"name": k, "value": bytes_to_str(arr.item())})
                        elif arr.ndim == 1 and arr.size <= 50:
                            rows.append({"name": k, "value": arr.tolist()})
                        else:
                            rows.append({"name": k, "value": f"array(shape={arr.shape})"})
            else:
                arr = np.array(obj[()])
                if arr.ndim == 0:
                    rows.append({"name": sel, "value": bytes_to_str(arr.item())})
                elif arr.ndim == 1 and arr.size <= 50:
                    rows.append({"name": sel, "value": arr.tolist()})
                else:
                    rows.append({"name": sel, "value": f"array(shape={arr.shape})"})
            with c2:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Materials read error: {e}")

# -----------------------------
# Dump: Sidebar (lock selections)
# -----------------------------
def render_dump_sidebar(entry: Dict[str, Any]) -> bool:
    path = entry["path"]
    dump_group = entry["dump_group"]
    try:
        with h5py.File(path, "r") as f:
            if not dump_group or dump_group not in f:
                st.sidebar.info("This file has no Dump_* group.")
                return False
            root = f[dump_group]
            containers = [key_to_str(k) for k, v in root.items() if isinstance(v, h5py.Group)]
            if not containers:
                st.sidebar.info("Dump group has no containers.")
                return False

            desired_container = st.session_state.get("desired_container")
            if desired_container is None:
                st.session_state["desired_container"] = containers[0]
                desired_container = containers[0]

            matched_container = ci_match(desired_container, containers)
            if matched_container is None:
                st.sidebar.warning(f"Container '{desired_container}' not in this file.")
                pick_key = "container_pick_tmp"
                if pick_key not in st.session_state:
                    st.session_state[pick_key] = containers[0]
                idx = containers.index(st.session_state[pick_key]) if st.session_state[pick_key] in containers else 0
                st.sidebar.selectbox("Choose available container", containers, index=idx, key=pick_key)
                if st.sidebar.button("Use this container"):
                    st.session_state["desired_container"] = st.session_state[pick_key]
                    st.rerun()
                return False
            else:
                sel_key = "container_selector_current"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = matched_container
                idx = containers.index(st.session_state[sel_key]) if st.session_state[sel_key] in containers else containers.index(matched_container)
                sel = st.sidebar.selectbox("Container", containers, index=idx, key=sel_key)
                if sel != desired_container:
                    st.session_state["desired_container"] = sel

            container = st.session_state["desired_container"]
            grp = root[container]
            subgroups = [key_to_str(k) for k, v in grp.items() if isinstance(v, h5py.Group)]
            if not subgroups:
                st.sidebar.info("Selected container has no subgroups.")
                return False

            desired_subgroup_map = st.session_state.get("desired_subgroup_map", {})
            desired_subgroup = desired_subgroup_map.get(container)
            if desired_subgroup is None:
                desired_subgroup = subgroups[0]
                desired_subgroup_map[container] = desired_subgroup
                st.session_state["desired_subgroup_map"] = desired_subgroup_map

            matched_subgroup = ci_match(desired_subgroup, subgroups)
            if matched_subgroup is None:
                pick_key = f"subgroup_pick_tmp::{container}"
                if pick_key not in st.session_state:
                    st.session_state[pick_key] = subgroups[0]
                idx = subgroups.index(st.session_state[pick_key]) if st.session_state[pick_key] in subgroups else 0
                st.sidebar.selectbox("Choose available subgroup", subgroups, index=idx, key=pick_key)
                if st.sidebar.button("Use this subgroup"):
                    desired_subgroup_map = st.session_state.get("desired_subgroup_map", {})
                    desired_subgroup_map[container] = st.session_state[pick_key]
                    st.session_state["desired_subgroup_map"] = desired_subgroup_map
                    st.rerun()
                return False
            else:
                sel_key = f"subgroup_selector_current::{container}"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = matched_subgroup
                idx = subgroups.index(st.session_state[sel_key]) if st.session_state[sel_key] in subgroups else subgroups.index(matched_subgroup)
                sel = st.sidebar.selectbox("Subgroup", subgroups, index=idx, key=sel_key)
                if sel != desired_subgroup:
                    desired_subgroup_map = st.session_state.get("desired_subgroup_map", {})
                    desired_subgroup_map[container] = sel
                    st.session_state["desired_subgroup_map"] = desired_subgroup_map

            return True
    except Exception as e:
        st.sidebar.error(f"Dump navigation error: {e}")
        return False

# -----------------------------
# Helpers for Temporal Evolution
# -----------------------------
def union_variables_across_files(container: str, subgroup: str, mode: str) -> Tuple[List[str], Dict[str, int]]:
    """Return (union variable names, max component index per var) for the given group across all files."""
    names = set()
    comp_max: Dict[str, int] = {}
    for entry in files_sorted:
        path, dump_group = entry["path"], entry["dump_group"]
        if not dump_group:
            continue
        try:
            with h5py.File(path, "r") as f:
                root = f[dump_group]
                if container not in root or subgroup not in root[container]:
                    continue
                g = root[container][subgroup]
                topo_key, node_key, elem_key = find_mapping_keys(g, container)
                if not (node_key and elem_key):
                    continue
                node_numbers = np.array(g[node_key]).reshape(-1)
                elem_numbers = np.array(g[elem_key]).reshape(-1)
                exclude = [k for k in [topo_key, node_key, elem_key] if k]
                nodal_vars, elem_vars = classify_variables(
                    g, node_count=node_numbers.shape[0], elem_count=elem_numbers.shape[0], exclude=exclude
                )
                vars_here = nodal_vars if mode == "Nodal" else elem_vars
                names.update(vars_here)
                for var in vars_here:
                    ds = g[var]
                    if len(ds.shape) >= 2 and ds.shape[1] > 1:
                        comp_max[var] = max(comp_max.get(var, 0), ds.shape[1] - 1)
        except Exception:
            continue
    return sorted(names), comp_max

def build_timeseries_dataframe(container: str, subgroup: str, mode: str, var: str, comp: int, numbers: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract time series across steps for a single variable and multiple entity numbers."""
    rows = []
    notes = []
    for entry in files_sorted:
        path, dump_group, t = entry["path"], entry["dump_group"], entry.get("time")
        if t is None:
            # fallback if missing time in meta
            t = np.nan
        if not dump_group:
            rows.append((t, {n: np.nan for n in numbers}))
            continue
        try:
            with h5py.File(path, "r") as f:
                root = f[dump_group]
                if container not in root or subgroup not in root[container]:
                    rows.append((t, {n: np.nan for n in numbers})); continue
                g = root[container][subgroup]
                topo_key, node_key, elem_key = find_mapping_keys(g, container)
                if not (node_key and elem_key) or var not in g:
                    rows.append((t, {n: np.nan for n in numbers})); continue
                node_numbers = np.array(g[node_key]).astype(int).reshape(-1)
                elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
                inv = invert_mapping(node_numbers if mode == "Nodal" else elem_numbers)
                ds = g[var]
                if len(ds.shape) == 1:
                    vals = []
                    for n in numbers:
                        idx = inv.get(n)
                        vals.append(float(ds[idx]) if idx is not None and idx < ds.shape[0] else np.nan)
                else:
                    c = min(comp, ds.shape[1]-1)
                    vals = []
                    for n in numbers:
                        idx = inv.get(n)
                        vals.append(float(ds[idx, c]) if idx is not None and idx < ds.shape[0] else np.nan)
                rows.append((t, dict(zip(numbers, vals))))
        except Exception as e:
            notes.append(str(e))
            rows.append((t, {n: np.nan for n in numbers}))
    # assemble DF
    times = [r[0] for r in rows]
    data = {("Node Number" if mode=="Nodal" else "Element Number"): numbers}
    # transpose to columns per entity
    df = pd.DataFrame({"Time": times})
    for n in numbers:
        df[f"{'Node' if mode=='Nodal' else 'Elem'} {n}"] = [r[1].get(n, np.nan) for r in rows]
    return df, notes

# -----------------------------
# Dump: Main (per-step + temporal)
# -----------------------------
def render_dump_main(entry: Dict[str, Any]):
    path = entry["path"]
    dump_group = entry["dump_group"]
    if not dump_group:
        st.info("This file has no Dump_* group.")
        return
    try:
        with h5py.File(path, "r") as f:
            root = f[dump_group]
            container = st.session_state.get("desired_container")
            if container is None or container not in root:
                st.info("Pick a valid Container in the sidebar."); return
            subgroup_map = st.session_state.get("desired_subgroup_map", {})
            subgroup = subgroup_map.get(container)
            if subgroup is None or subgroup not in root[container]:
                st.info("Pick a valid Subgroup in the sidebar."); return

            g = root[container][subgroup]

            topo_key, node_key, elem_key = find_mapping_keys(g, container_name=container)
            if not (node_key and elem_key):
                st.error("Required mapping datasets not found (Node/Element Numbers).")
                return
            node_numbers = np.array(g[node_key]).astype(int).reshape(-1)
            elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
            node_inv = invert_mapping(node_numbers)
            elem_inv = invert_mapping(elem_numbers)

            num_nodes, num_elems = node_numbers.shape[0], elem_numbers.shape[0]
            exclude = [k for k in [topo_key, node_key, elem_key] if k]
            nodal_vars, elem_vars = classify_variables(g, node_count=num_nodes, elem_count=num_elems, exclude=exclude)

            # ---- normalized group key
            group_key_norm = f"{norm_name(container)}::{norm_name(subgroup)}"
            store: Dict[str, Dict[str, Any]] = st.session_state["store"]

            # ---- ALWAYS seed widgets if missing (per-step view)
            ss_default("mode_widget", "Nodal")
            ss_default("vars_widget", [])
            ss_default("nums_widget", "")
            ss_default("sec_widget",  "None")

            # ---- Load from store when group changes
            if st.session_state["current_group_norm"] != group_key_norm:
                rec = store.get(group_key_norm, {"mode":"Nodal","vars":[],"nums":"","sec":"None","comp":{}})
                st.session_state["mode_widget"] = rec["mode"]
                st.session_state["vars_widget"] = list(rec["vars"])
                st.session_state["nums_widget"] = rec["nums"]
                st.session_state["sec_widget"]  = rec["sec"]
                st.session_state["current_group_norm"] = group_key_norm

            # Mode
            st.radio("Variable type", ["Nodal","Element"], key="mode_widget", horizontal=True)
            mode = st.session_state["mode_widget"]
            var_options_avail = nodal_vars if mode == "Nodal" else elem_vars

            # Variables (union to keep selections even if missing this step)
            selected_names = list(st.session_state["vars_widget"])
            union_options = sorted(set(var_options_avail).union(selected_names))
            def labelize(n: str) -> str:
                return f"{n} (missing in this step)" if n not in var_options_avail else n

            c_vars, c_nums, c_sec = st.columns([2,2,1])
            with c_vars:
                st.multiselect("Variables", options=union_options, key="vars_widget", format_func=labelize)
                sel_vars = list(st.session_state["vars_widget"])
            with c_nums:
                label_numbers = "Enter nodal NUMBERS (not IDs)" if mode=="Nodal" else "Enter element NUMBERS (not IDs)"
                st.text_input(label_numbers, key="nums_widget", placeholder="e.g., 1,2,3-10")
                numbers_list = parse_int_list(st.session_state["nums_widget"])
            with c_sec:
                sec_pool = ["None"] + (sel_vars if sel_vars else [])
                if st.session_state["sec_widget"] not in sec_pool:
                    st.session_state["sec_widget"] = "None"
                st.selectbox("Secondary y-axis", sec_pool, key="sec_widget")

            if not sel_vars or not numbers_list:
                st.info("Choose at least one variable and enter a list of numbers to plot.")
                # Save current state
                store[group_key_norm] = {
                    "mode": st.session_state["mode_widget"],
                    "vars": list(st.session_state["vars_widget"]),
                    "nums": st.session_state["nums_widget"],
                    "sec":  st.session_state["sec_widget"],
                    "comp": store.get(group_key_norm, {}).get("comp", {}),
                }
                st.session_state["store"] = store
            else:
                # Components (only for present & vector-valued)
                rec_comp = store.get(group_key_norm, {}).get("comp", {})
                var_components = {}
                vecs = []
                for var in sel_vars:
                    if var in g:
                        arr = np.array(g[var])
                        if arr.ndim >= 2 and arr.shape[1] > 1:
                            vecs.append((var, arr.shape[1]-1))
                if vecs:
                    st.markdown("**Components**")
                    comp_cols = st.columns(len(vecs))
                else:
                    comp_cols = []
                for (var, comp_max), col in zip(vecs, comp_cols):
                    with col:
                        ck = f"comp_widget::{var}"
                        if ck not in st.session_state:
                            st.session_state[ck] = int(rec_comp.get(var, 0))
                        st.session_state[ck] = max(0, min(int(st.session_state[ck]), comp_max))
                        st.number_input(f"{var}", min_value=0, max_value=comp_max, key=ck, step=1)
                        var_components[var] = int(st.session_state[ck])

                # Map numbers -> internal IDs
                inv = node_inv if mode == "Nodal" else elem_inv
                ids = [inv.get(n) for n in numbers_list]
                missing_nums = [n for n, i in zip(numbers_list, ids) if i is None]
                ids = [i for i in ids if i is not None]
                used_numbers = [n for n in numbers_list if inv.get(n) is not None]
                if missing_nums:
                    st.warning(f"Numbers not found and dropped: {missing_nums}")

                if len(ids) == 0 or len(used_numbers) == 0:
                    fig = go.Figure()
                    fig.add_annotation(text="No valid numbers in this step for the current selection",
                                       xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    ft_style(fig, x_title=("Node Number" if mode=="Nodal" else "Element Number"), y_title_left="Value")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pd.DataFrame({"requested": numbers_list, "found": [n in used_numbers for n in numbers_list]}),
                                 use_container_width=True)
                else:
                    # Build DF
                    missing_vars = [v for v in sel_vars if v not in g]
                    if missing_vars:
                        st.info(f"Variables missing in this step: {missing_vars}")

                    first_col_name = "Node Number" if mode=="Nodal" else "Element Number"
                    df = pd.DataFrame({first_col_name: used_numbers})
                    labels_map = {}
                    for var in sel_vars:
                        if var not in g:
                            continue
                        arr = np.array(g[var])
                        if arr.ndim == 1:
                            vals = [arr[i] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                            label = var
                        else:
                            comp = var_components.get(var, store.get(group_key_norm, {}).get("comp", {}).get(var, 0))
                            vals = [arr[i, comp] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                            label = f"{var} [comp {comp}]"
                        vals = [float(np.array(v).item()) if v is not None and np.isfinite(v) else np.nan for v in vals]
                        df[label] = vals
                        labels_map[var] = label

                    # Plot
                    sec_choice = st.session_state["sec_widget"]
                    sec_present = (sec_choice != "None") and (sec_choice in labels_map)

                    if sec_choice != "None" and not sec_present:
                        st.info(f"Secondary axis variable '{sec_choice}' is missing in this step and will be ignored.")
        
                    if sec_present:
                        sec_label = labels_map.get(sec_choice)
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        prim_labels = []
                        for var in sel_vars:
                            if var not in labels_map:  # missing
                                continue
                            col = labels_map[var]
                            if col == sec_label: 
                                continue
                            prim_labels.append(col)
                            fig.add_trace(
                                go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col),
                                secondary_y=False
                            )
                        fig.add_trace(
                            go.Scattergl(x=df[first_col_name], y=df[sec_label], mode="lines+markers", name=f"{sec_label} (right)"),
                            secondary_y=True
                        )
                        y_left_title  = axis_title_from_labels(prim_labels, fallback="Value")
                        y_right_title = axis_title_from_labels([sec_label], fallback="Secondary")
                        ft_style(fig, x_title=first_col_name, y_title_left=y_left_title, y_title_right=y_right_title)
                    else:
                        fig = go.Figure()
                        prim_labels = []
                        for var in sel_vars:
                            if var not in labels_map:  # missing
                                continue
                            col = labels_map[var]
                            prim_labels.append(col)
                            fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col))
                        y_left_title = axis_title_from_labels(prim_labels, fallback="Value")
                        ft_style(fig, x_title=first_col_name, y_title_left=y_left_title)

                    st.plotly_chart(fig, use_container_width=True)
                    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    st.download_button("Download plot (HTML)", data=html_bytes, file_name="plot.html", mime="text/html")
                    try:
                        png_bytes = fig.to_image(format="png", scale=2)  # needs kaleido
                        st.download_button("Download plot (PNG)", data=png_bytes, file_name="plot.png", mime="image/png")
                    except Exception:
                        st.caption("PNG export unavailable (kaleido not installed in this runtime).")
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Download table (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="table.csv", mime="text/csv")

                # Save per-step state
                store[group_key_norm] = {
                    "mode": st.session_state["mode_widget"],
                    "vars": list(st.session_state["vars_widget"]),
                    "nums": st.session_state["nums_widget"],
                    "sec":  st.session_state["sec_widget"],
                    "comp": {**store.get(group_key_norm, {}).get("comp", {}),
                             **{k.split('::')[-1]: int(st.session_state[k]) for k in st.session_state.keys() if k.startswith("comp_widget::")}},
                }
                st.session_state["store"] = store

            # -----------------------------
            # TEMPORAL EVOLUTION (across steps)
            # -----------------------------
            st.divider()
            st.subheader("Temporal evolution (across steps)")
            store_ts: Dict[str, Dict[str, Any]] = st.session_state["store_ts"]
            rec_ts = store_ts.get(group_key_norm, {"mode":"Nodal","var":"", "nums":"", "comp":{}})

            # Always seed TS widgets if missing
            ss_default("ts_mode_widget", rec_ts["mode"])
            ss_default("ts_var_widget",  rec_ts["var"])
            ss_default("ts_nums_widget", rec_ts["nums"])

            # Union of variables across all files for this group+mode
            ts_mode = st.radio("Variable type", ["Nodal","Element"], key="ts_mode_widget", horizontal=True)
            union_vars, comp_max_map = union_variables_across_files(container, subgroup, ts_mode)
            if not union_vars:
                st.info("No variables available across steps for this selection.")
                # Save and return
                store_ts[group_key_norm] = {
                    "mode": st.session_state["ts_mode_widget"],
                    "var":  st.session_state["ts_var_widget"],
                    "nums": st.session_state["ts_nums_widget"],
                    "comp": store_ts.get(group_key_norm, {}).get("comp", {}),
                }
                st.session_state["store_ts"] = store_ts
                return

            # If current ts_var not in union, set to first available
            if st.session_state["ts_var_widget"] not in union_vars:
                st.session_state["ts_var_widget"] = union_vars[0]

            # Inline controls
            c_ts1, c_ts2 = st.columns([2,2])
            with c_ts1:
                st.selectbox("Variable (single)", options=union_vars, key="ts_var_widget")
            with c_ts2:
                label_numbers = "Enter nodal NUMBERS (not IDs) for time series" if ts_mode=="Nodal" else "Enter element NUMBERS (not IDs) for time series"
                st.text_input(label_numbers, key="ts_nums_widget", placeholder="e.g., 10,12,15-18")

            ts_numbers = parse_int_list(st.session_state["ts_nums_widget"])

            # Component selector if needed
            sel_var = st.session_state["ts_var_widget"]
            comp_max = comp_max_map.get(sel_var, 0)
            if comp_max > 0:
                ck = f"ts_comp_widget::{sel_var}"
                if ck not in st.session_state:
                    st.session_state[ck] = int(store_ts.get(group_key_norm, {}).get("comp", {}).get(sel_var, 0))
                st.number_input(f"{sel_var} • component", min_value=0, max_value=comp_max, key=ck, step=1)
                ts_comp = int(st.session_state[ck])
            else:
                ts_comp = 0

            if not ts_numbers:
                st.info("Enter at least one number to plot time series.")
            else:
                # Build time series DF
                df_ts, notes = build_timeseries_dataframe(container, subgroup, ts_mode, sel_var, ts_comp, ts_numbers)
                # Plot
                fig_ts = go.Figure()
                for n in ts_numbers:
                    col = f"{'Node' if ts_mode=='Nodal' else 'Elem'} {n}"
                    fig_ts.add_trace(go.Scattergl(x=df_ts["Time"], y=df_ts[col], mode="lines+markers", name=col))
                y_title = sel_var if comp_max == 0 else f"{sel_var} [comp {ts_comp}]"
                ft_style(fig_ts, x_title="Time", y_title_left=y_title)
                st.plotly_chart(fig_ts, use_container_width=True)
                html_bytes = fig_ts.to_html(include_plotlyjs="cdn").encode("utf-8")
                st.download_button("Download time series (HTML)", data=html_bytes, file_name="timeseries.html", mime="text/html")
                try:
                    png_bytes = fig_ts.to_image(format="png", scale=2)
                    st.download_button("Download time series (PNG)", data=png_bytes, file_name="timeseries.png", mime="image/png")
                except Exception:
                    st.caption("PNG export unavailable (kaleido not installed in this runtime).")
                st.dataframe(df_ts, use_container_width=True)
                st.download_button("Download time series (CSV)", data=df_ts.to_csv(index=False).encode("utf-8"), file_name="timeseries.csv", mime="text/csv")
                if notes:
                    st.caption(f"Notes: {set(notes)}")

            # Save TS state
            store_ts[group_key_norm] = {
                "mode": st.session_state["ts_mode_widget"],
                "var":  st.session_state["ts_var_widget"],
                "nums": st.session_state["ts_nums_widget"],
                "comp": {**store_ts.get(group_key_norm, {}).get("comp", {}),
                         **({sel_var: ts_comp} if sel_var else {})},
            }
            st.session_state["store_ts"] = store_ts

    except Exception as e:
        st.error(f"Dump read error: {e}")

# -----------------------------
# Render
# -----------------------------
if st.session_state.part_choice_radio == "Equations":
    render_equations(active)
elif st.session_state.part_choice_radio == "Materials":
    render_materials(active)
else:
    if render_dump_sidebar(active):
        render_dump_main(active)

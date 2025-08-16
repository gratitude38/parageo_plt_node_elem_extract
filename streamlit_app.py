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

st.set_page_config(page_title="ParaGeo .plt visualiser", layout="wide")

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
# store[group] = {"mode": "Nodal"/"Element", "vars":[...], "nums":"1,2,3", "sec":"var/None", "comp":{var:int}}
ss_default("store", {})
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
    lname = (container_name or "").lower()

    if lname == "contact":
        if "Contact Node number" in keys: node_key = "Contact Node number"
        if "Contact Element number" in keys: elem_key = "Contact Element number"
        if node_key is None:
            for k in keys:
                if re.fullmatch(r"(?i)contact\s*node\s*number", k):
                    node_key = k; break
        if elem_key is None:
            for k in keys:
                if re.fullmatch(r"(?i)contact\s*element\s*number", k):
                    elem_key = k; break

    elif lname == "wells":
        # Wells: nodes live in Well_nodes; element numbers may be absent
        if "Well_nodes" in keys:
            node_key = "Well_nodes"
        else:
            for k in keys:
                if re.fullmatch(r"(?i)well[_\s]*nodes?", k):
                    node_key = k; break
        # try a normal element mapping if it exists; otherwise leave None
        if "Element Numbers" in keys:
            elem_key = "Element Numbers"
        else:
            for k in keys:
                if re.fullmatch(r"(?i)element\s*numbers", k):
                    elem_key = k; break

    else:
        if "Node Numbers" in keys: node_key = "Node Numbers"
        if "Element Numbers" in keys: elem_key = "Element Numbers"
        if node_key is None:
            for k in keys:
                if re.fullmatch(r"(?i)node\s*numbers", k):
                    node_key = k; break
        if elem_key is None:
            for k in keys:
                if re.fullmatch(r"(?i)element\s*numbers", k):
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
        type=["plt"],
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
    st.session_state.active_index = 0
    st.session_state.active_id = files_sorted[0]["id"]

active = files_sorted[st.session_state.active_index]

# Top bar
# cols = st.columns([1,1,5,3])
# with cols[0]:
#     if st.button("◀ Prev", use_container_width=True, key="prev_btn", disabled=st.session_state.active_index <= 0):
#         st.session_state.active_index = max(0, st.session_state.active_index - 1)
#         st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
#         st.rerun()
# with cols[1]:
#     if st.button("Next ▶", use_container_width=True, key="next_btn", disabled=st.session_state.active_index >= len(files_sorted)-1):
#         st.session_state.active_index = min(len(files_sorted)-1, st.session_state.active_index + 1)
#         st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
#         st.rerun()
# with cols[2]:
#     label = f"Step {active['step'] if active['step'] is not None else 'NA'} — {active['name']}"
#     #st.markdown(f"### {label}")
# with cols[3]:
#     #labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']} ({e['id']})" for e in files_sorted]
#     labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']}" for e in files_sorted]
#     sel = st.selectbox("Jump to file", options=list(range(len(files_sorted))), format_func=lambda i: labels[i], index=st.session_state.active_index, key="jump_file")
#     if sel != st.session_state.active_index:
#         st.session_state.active_index = sel
#         st.session_state.active_id = files_sorted[sel].get("id")
#         st.rerun()
# 
# st.caption(f"Dump group: {active.get('dump_group')} | Step {active['step'] if active['step'] is not None else 'NA'} | Time: {active.get('time')} | File: {active.get('name')}")



cols = st.columns([8,3])
with cols[0]:
    st.header("ParaGeo *.plt* visualiser")
with cols[1]:
    #labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']} ({e['id']})" for e in files_sorted]
    labels = [f"Step {e['step'] if e['step'] is not None else 'NA'} — {e['name']}" for e in files_sorted]
    sel = st.selectbox("Jump to file", options=list(range(len(files_sorted))), format_func=lambda i: labels[i], index=st.session_state.active_index, key="jump_file")
    if sel != st.session_state.active_index:
        st.session_state.active_index = sel
        st.session_state.active_id = files_sorted[sel].get("id")
        st.rerun()

cols = st.columns([8,1.5,1.5])
with cols[0]:
    st.caption(f"Dump group: {active.get('dump_group')} | Step {active['step'] if active['step'] is not None else 'NA'} | Time: {active.get('time')} | File: {active.get('name')}")
with cols[1]:
    if st.button("◀ Prev", use_container_width=True, key="prev_btn", disabled=st.session_state.active_index <= 0):
        st.session_state.active_index = max(0, st.session_state.active_index - 1)
        st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
        st.rerun()
with cols[2]:
    if st.button("Next ▶", use_container_width=True, key="next_btn", disabled=st.session_state.active_index >= len(files_sorted)-1):
        st.session_state.active_index = min(len(files_sorted)-1, st.session_state.active_index + 1)
        st.session_state.active_id = files_sorted[st.session_state.active_index].get("id")
        st.rerun()

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

def clean_axis_title(label: str, fallback: str = "Value") -> str:
    """Strip node/element numbers and suffixes from a legend label.
    Examples:
      'Contact Normal Effective stress 392'           -> 'Contact Normal Effective stress'
      'Deviatoric Stress [comp 1] @ Node 271 (right)' -> 'Deviatoric Stress'
    """
    if not label:
        return fallback
    s = label
    s = re.sub(r"\s*\(right\)$", "", s)          # drop ' (right)'
    s = re.sub(r"\s*\[comp[^\]]*\]\s*", "", s)   # drop ' [comp k]'
    s = re.sub(r"\s*@\s*(Node|Elem).*?$", "", s) # drop '@ Node 123' / '@ Elem 45'
    s = re.sub(r"\s+(Node|Elem)\s*\d+\s*$", "", s)  # drop 'Node 123' or 'Elem 45' at end
    s = re.sub(r"\s*\d+\s*$", "", s)             # drop trailing bare numbers
    return s.strip() or fallback

# -----------------------------
# Dump: Main (stable widgets + per-group store)
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

            #topo_key, node_key, elem_key = find_mapping_keys(g, container_name=container)
            #if not (node_key and elem_key):
            #    st.error("Required mapping datasets not found (Node/Element Numbers).")
            #    return

            topo_key, node_key, elem_key = find_mapping_keys(g, container_name=container)
            # Node mapping must exist
            if not node_key or node_key not in g:
                st.error("Required node mapping not found.")
                return
            node_numbers = np.array(g[node_key]).astype(int).reshape(-1)

            # Element mapping: use dataset if present; else synthesize for Wells
            if elem_key and elem_key in g:
                elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
            else:
                if (container or "").lower() == "wells":
                    # n = int(node_numbers.shape[0])
                    # n nodes → n-1 elements: numbers 1..(n-1)
                    elem_numbers = np.arange(1, int(node_numbers.shape[0]), dtype=int)
                else:
                    st.error("Element mapping missing.")
                    return
                
            #node_numbers = np.array(g[node_key]).astype(int).reshape(-1)
            #elem_numbers = np.array(g[elem_key]).astype(int).reshape(-1)
            node_inv = invert_mapping(node_numbers)
            elem_inv = invert_mapping(elem_numbers)

            num_nodes, num_elems = node_numbers.shape[0], elem_numbers.shape[0]
            exclude = [k for k in [topo_key, node_key, elem_key] if k]
            nodal_vars, elem_vars = classify_variables(g, node_count=num_nodes, elem_count=num_elems, exclude=exclude)

            # ---- normalized group key
            group_key_norm = f"{norm_name(container)}::{norm_name(subgroup)}"
            store: Dict[str, Dict[str, Any]] = st.session_state["store"]
            rec = store.get(group_key_norm, {"mode":"Nodal","vars":[],"nums":"","sec":"None","comp":{}})

            # ---- ALWAYS seed widgets if missing
            ss_default("mode_widget", rec["mode"])
            ss_default("vars_widget", list(rec["vars"]))
            ss_default("nums_widget", rec["nums"])
            ss_default("sec_widget",  rec["sec"])
            ss_default(f"time_toggle::{group_key_norm}", False)
            ss_default(f"time_xaxis::{group_key_norm}", "Time")   # "Time" or "Step"

            # ---- Load from store when group changes
            if st.session_state["current_group_norm"] != group_key_norm:
                st.session_state["mode_widget"] = rec["mode"]
                st.session_state["vars_widget"] = list(rec["vars"])
                st.session_state["nums_widget"] = rec["nums"]
                st.session_state["sec_widget"]  = rec["sec"]
                st.session_state[f"time_toggle::{group_key_norm}"] = False
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

            # ---- NEW: time-series controls
            c_ts1, c_ts2 = st.columns([1,1])
            with c_ts1:
                st.checkbox("Plot over time (across steps)", key=f"time_toggle::{group_key_norm}")
            with c_ts2:
                st.selectbox("X-axis", ["Time", "Step"], key=f"time_xaxis::{group_key_norm}")

            plot_over_time = st.session_state[f"time_toggle::{group_key_norm}"]
            x_axis_choice = st.session_state[f"time_xaxis::{group_key_norm}"]

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
                return

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

            # ---------- Branch: current-step vs time-series ----------
            if not plot_over_time:
                # ===== current-step (existing behaviour) =====
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
                    return

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
                        comp = var_components.get(var, rec_comp.get(var, 0))
                        vals = [arr[i, comp] if i is not None and i < arr.shape[0] else np.nan for i in ids]
                        label = f"{var} [comp {comp}]"
                    vals = [float(np.array(v).item()) if v is not None and np.isfinite(v) else np.nan for v in vals]
                    df[label] = vals
                    labels_map[var] = label

                sec_choice = st.session_state["sec_widget"]
                sec_present = (sec_choice != "None") and (sec_choice in labels_map)
                if sec_choice != "None" and not sec_present:
                    st.info(f"Secondary axis variable '{sec_choice}' is missing in this step and will be ignored.")

                if sec_present:
                    sec_label = labels_map[sec_choice]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    prim_labels = []
                    for var in sel_vars:
                        if var not in labels_map:
                            continue
                        col = labels_map[var]
                        if col == sec_label: 
                            continue
                        prim_labels.append(col)
                        fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col), secondary_y=False)
                    fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[sec_label], mode="lines+markers", name=f"{sec_label} (right)"), secondary_y=True)
                    #y_left_title = ", ".join(prim_labels) if prim_labels else "Value"
                    y_left_title = clean_axis_title(prim_labels[0]) if prim_labels else "Value"
                    #y_right_title = sec_label or "Secondary"
                    y_right_title = clean_axis_title(sec_label)
                    ft_style(fig, x_title=first_col_name, y_title_left=y_left_title, y_title_right=y_right_title)
                else:
                    fig = go.Figure()
                    prim_labels = []
                    for var in sel_vars:
                        if var not in labels_map:
                            continue
                        col = labels_map[var]
                        prim_labels.append(col)
                        fig.add_trace(go.Scattergl(x=df[first_col_name], y=df[col], mode="lines+markers", name=col))
                    #y_left_title = ", ".join(prim_labels) if prim_labels else "Value"
                    y_left_title = clean_axis_title(prim_labels[0]) if prim_labels else "Value"
                    ft_style(fig, x_title=first_col_name, y_title_left=y_left_title)

                st.plotly_chart(fig, use_container_width=True)
                html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                st.download_button("Download plot (HTML)", data=html_bytes, file_name="plot.html", mime="text/html")
                try:
                    png_bytes = fig.to_image(format="png", scale=2)
                    st.download_button("Download plot (PNG)", data=png_bytes, file_name="plot.png", mime="image/png")
                except Exception:
                    st.caption("PNG export unavailable (kaleido not installed in this runtime).")
                st.dataframe(df, use_container_width=True)
                st.download_button("Download table (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="table.csv", mime="text/csv")

            else:
                # ===== time-series across steps =====
                mode_label = "Node" if mode == "Nodal" else "Element"
                # Pre-build all trace labels (var x number)
                def trace_label(var: str, n: int, comp_map: Dict[str,int]) -> str:
                    if var in g and np.array(g[var]).ndim >= 2 and np.array(g[var]).shape[1] > 1:
                        cidx = comp_map.get(var, rec_comp.get(var, 0))
                        #return f"{var} [comp {cidx}] @ {mode_label} {n}"
                        return f"{var} [comp {cidx}] {n}"
                    elif var in g and np.array(g[var]).ndim == 1:
                        #return f"{var} @ {mode_label} {n}"
                        return f"{var} {n}"
                    else:
                        # unknown shape in this step → still keep a generic label
                        # return f"{var} @ {mode_label} {n}"
                        return f"{var} {n}"

                all_labels = [trace_label(v, n, var_components) for v in sel_vars for n in numbers_list]
                x_vals: List[float] = []
                rows: List[Dict[str, float]] = []

                # Iterate all files (ordered)
                for ent in files_sorted:
                    try:
                        with h5py.File(ent["path"], "r") as f2:
                            dg = ent.get("dump_group")
                            if not dg or dg not in f2: 
                                continue
                            root2 = f2[dg]
                            if container not in root2: 
                                continue
                            if subgroup not in root2[container]: 
                                continue
                            g2 = root2[container][subgroup]

                            # x-axis value
                            x_val = None
                            if x_axis_choice == "Time" and "Time" in f2[dg]:
                                try:
                                    x_val = float(np.array(f2[dg]["Time"][()]).item())
                                except Exception:
                                    x_val = None
                            if x_val is None:
                                # fallback to Step or index
                                if "Step" in f2[dg]:
                                    try:
                                        x_val = float(np.array(f2[dg]["Step"][()]).item())
                                    except Exception:
                                        x_val = None
                            if x_val is None:
                                # ultimate fallback = file order index
                                x_val = float(ent.get("step") or 0)
                            # build row initialized with NaNs
                            row = {lab: np.nan for lab in all_labels}

                            # mapping per file
                            #topo2, node_key2, elem_key2 = find_mapping_keys(g2, container_name=container)
                            #if not (node_key2 and elem_key2):
                            #    x_vals.append(x_val); rows.append(row); continue
                            #node_nums2 = np.array(g2[node_key2]).astype(int).reshape(-1)
                            #elem_nums2 = np.array(g2[elem_key2]).astype(int).reshape(-1)
                            #inv2 = invert_mapping(node_nums2 if mode=="Nodal" else elem_nums2)


                            topo2, node_key2, elem_key2 = find_mapping_keys(g2, container_name=container)
                            # Need nodes to proceed
                            if not node_key2 or node_key2 not in g2:
                                x_vals.append(x_val); rows.append(row); continue
                            node_nums2 = np.array(g2[node_key2]).astype(int).reshape(-1)
                            
                            # Elements: dataset if present; else synthesize for Wells
                            if elem_key2 and elem_key2 in g2:
                                elem_nums2 = np.array(g2[elem_key2]).astype(int).reshape(-1)
                            elif (container or "").lower() == "wells":
                                # n2 = int(node_nums2.shape[0])
                                # n nodes → n-1 elements in this step
                                elem_nums2 = np.arange(1, int(node_nums2.shape[0]), dtype=int)
                            else:
                                if mode == "Element":
                                    x_vals.append(x_val); rows.append(row); continue
                                elem_nums2 = np.array([], dtype=int)
                            inv2 = invert_mapping(node_nums2 if mode == "Nodal" else elem_nums2)                            

                            # for each var/number get value
                            for var in sel_vars:
                                if var not in g2: 
                                    continue
                                arr = np.array(g2[var])
                                # pick component if needed (clamped)
                                if arr.ndim >= 2 and arr.shape[1] > 1:
                                    comp_idx = int(var_components.get(var, rec_comp.get(var, 0)))
                                    comp_idx = max(0, min(comp_idx, arr.shape[1]-1))
                                else:
                                    comp_idx = None
                                for n in numbers_list:
                                    idx = inv2.get(n)
                                    if idx is None or idx >= arr.shape[0]:
                                        continue
                                    if comp_idx is None:
                                        val = arr[idx]
                                    else:
                                        val = arr[idx, comp_idx]
                                    try:
                                        val = float(np.array(val).item())
                                    except Exception:
                                        val = np.nan
                                    lab = trace_label(var, n, {var: comp_idx} if comp_idx is not None else {})
                                    row[lab] = val
                            x_vals.append(x_val)
                            rows.append(row)
                    except Exception:
                        # ignore file-level errors and keep going
                        continue

                if not rows:
                    st.info("No compatible data found across the loaded steps for the current selection.")
                    return

                # Build DataFrame
                x_name = x_axis_choice
                df_ts = pd.DataFrame(rows)
                df_ts.insert(0, x_name, x_vals)

                # Plot
                sec_choice = st.session_state["sec_widget"]
                # variable names present in label start; detect which labels belong to sec variable
                def is_sec_label(lbl: str) -> bool:
                    return sec_choice != "None" and lbl.startswith(sec_choice + " ")
                data_cols = [c for c in df_ts.columns if c != x_name]

                if sec_choice != "None" and not any(is_sec_label(c) for c in data_cols):
                    st.info(f"Secondary axis variable '{sec_choice}' is missing in these steps and will be ignored.")

                # Build figure
                if sec_choice != "None" and any(is_sec_label(c) for c in data_cols):
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    prim_labels = []
                    for c in data_cols:
                        if is_sec_label(c): 
                            continue
                        prim_labels.append(c)
                        fig.add_trace(go.Scattergl(x=df_ts[x_name], y=df_ts[c], mode="lines+markers", name=c), secondary_y=False)
                    # secondary traces
                    for c in data_cols:
                        if is_sec_label(c):
                            fig.add_trace(go.Scattergl(x=df_ts[x_name], y=df_ts[c], mode="lines+markers", name=f"{c} (right)"), secondary_y=True)
                    #y_left = ", ".join(prim_labels) if prim_labels else "Value"
                    #y_right = sec_choice
                    y_left  = clean_axis_title(prim_labels[0]) if prim_labels else "Value"
                    y_right = clean_axis_title(sec_choice)
                    ft_style(fig, x_title=x_name, y_title_left=y_left, y_title_right=y_right)
                else:
                    fig = go.Figure()
                    for c in data_cols:
                        fig.add_trace(go.Scattergl(x=df_ts[x_name], y=df_ts[c], mode="lines+markers", name=c))
                    ft_style(fig, x_title=x_name, y_title_left="Value")

                st.plotly_chart(fig, use_container_width=True)
                # Exports
                html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                st.download_button("Download plot (HTML)", data=html_bytes, file_name="time_series.html", mime="text/html")
                try:
                    png_bytes = fig.to_image(format="png", scale=2)
                    st.download_button("Download plot (PNG)", data=png_bytes, file_name="time_series.png", mime="image/png")
                except Exception:
                    st.caption("PNG export unavailable (kaleido not installed in this runtime).")
                # Table
                st.dataframe(df_ts, use_container_width=True)
                st.download_button("Download table (CSV)", data=df_ts.to_csv(index=False).encode("utf-8"), file_name="time_series.csv", mime="text/csv")

            # Save persistent state
            store[group_key_norm] = {
                "mode": st.session_state["mode_widget"],
                "vars": list(st.session_state["vars_widget"]),
                "nums": st.session_state["nums_widget"],
                "sec":  st.session_state["sec_widget"],
                "comp": {**rec_comp, **{v: var_components.get(v, rec_comp.get(v, 0)) for v in sel_vars}},
            }
            st.session_state["store"] = store

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

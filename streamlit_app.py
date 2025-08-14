
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
import zipfile

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
        # Trim along first dimension for preview
        if data.ndim == 1:
            data = data[:max_preview]
        else:
            sl = [slice(None)] * data.ndim
            sl[0] = slice(0, max_preview)
            data = data[tuple(sl)]
    return decode_array(np.array(data))

def list_groups(h: h5py.Group) -> List[str]:
    return [k for k, v in h.items() if isinstance(v, h5py.Group)]

def match_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    """Return the first key that matches any of the regex patterns (case-insensitive, full match)."""
    for pat in patterns:
        for k in keys:
            if re.fullmatch(pat, k, flags=re.IGNORECASE):
                return k
    return None

def extract_step_from_filename(name: str) -> Optional[int]:
    """
    Expect *_NNN.ext at the end OR ...StepXX_NNN.ext -> extract trailing NNN.
    """
    m = re.search(r"_(\d{1,})\.[^.]+$", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None

def get_dump_group(f: h5py.File, preferred_step: Optional[int]) -> Optional[str]:
    """
    Find Dump_XXX group. Prefer the one matching preferred_step; otherwise return the only Dump_* group or the first one.
    """
    dumps = [k for k in f.keys() if k.startswith("Dump_")]
    if not dumps:
        return None
    if preferred_step is not None:
        cand = f"Dump_{preferred_step:03d}"
        if cand in f:
            return cand
    if len(dumps) == 1:
        return dumps[0]
    # Fallback: pick the numerically largest step
    steps = []
    for d in dumps:
        m = re.match(r"Dump_(\d+)$", d)
        steps.append((int(m.group(1)) if m else -1, d))
    steps.sort()
    return steps[-1][1]

def classify_variables(g: h5py.Group, node_count: int, elem_count: int, exclude_keys: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (nodal_vars, element_vars, unknown_vars) among numeric datasets (exclude_keys ignored)."""
    nodal, elem, unknown = [], [], []
    for k, v in g.items():
        if not isinstance(v, h5py.Dataset):
            continue
        if k in exclude_keys:
            continue
        # numeric-ish only
        if v.dtype.kind not in ("i", "u", "f"):
            try:
                # attempt to treat as numeric if convertible
                np.array(v[()], dtype=float)
            except Exception:
                continue
        shape = v.shape
        if len(shape) == 0:
            unknown.append(k)  # scalar -> unknown
            continue
        n0 = shape[0]
        if n0 == node_count:
            nodal.append(k)
        elif n0 == elem_count:
            elem.append(k)
        else:
            unknown.append(k)
    return sorted(nodal), sorted(elem), sorted(unknown)

def invert_mapping(numbers: np.ndarray) -> Dict[int, int]:
    """
    Given an array where index is internal ID and value is actual number,
    return dict: actual_number -> internal_id
    """
    inv = {}
    for idx, val in enumerate(np.array(numbers).reshape(-1).tolist()):
        try:
            inv[int(val)] = int(idx)
        except Exception:
            pass
    return inv

def path_join(*parts):
    """Join HDF5-style paths (no leading/trailing //)."""
    return "/".join([parts[0].strip("/") ] + [q.strip("/") for q in parts[1:]])

def line_fig(x, series: dict, x_label: str, y_label: str, use_gl: bool = True):
    fig = go.Figure()
    for name, y in series.items():
        Trace = go.Scattergl if use_gl else go.Scatter
        fig.add_trace(Trace(x=x, y=y, mode="lines+markers", name=name))
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig

# -------------------------
# App State
# -------------------------

st.set_page_config(page_title="HDF5 .plt Viewer (FEM)", layout="wide")

if "files_meta" not in st.session_state:
    st.session_state.files_meta = {}  # step -> {"name": str, "path": str}
if "active_step_index" not in st.session_state:
    st.session_state.active_step_index = 0

st.title("HDF5-Structured FEM Viewer (Plotly)")
st.caption("Drag & drop multiple `.plt` files (each is a time step). Explore Dump / Equations / Materials, extract nodal/element variables, build cross-time overlays, and export bundles.")

# -------------------------
# File Uploader (Drag & Drop)
# -------------------------

uploads = st.file_uploader(
    "Drag & drop one or more `.plt` files",
    type=["plt", "h5", "hdf5"],
    accept_multiple_files=True,
    help="Each file is one time step result; filename suffix like `_002.plt` is used for ordering."
)

if uploads:
    # Persist uploads to temp files so h5py can open them
    new_meta = {}
    for uf in uploads:
        suffix_step = extract_step_from_filename(uf.name)
        # Save to a temp file with the same name for readability
        tmpdir = tempfile.mkdtemp(prefix="plt_")
        dst = os.path.join(tmpdir, uf.name)
        with open(dst, "wb") as out:
            out.write(uf.getbuffer())
        try:
            with h5py.File(dst, "r") as f:
                dump_group = get_dump_group(f, suffix_step)
                # double-check Step inside dump if available
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
            st.warning(f"Could not determine time step for {uf.name}; assigning step order by upload.")
        new_meta[step if step is not None else len(new_meta)] = {"name": uf.name, "path": dst}

    # Merge and sort by step
    st.session_state.files_meta.update(new_meta)
    steps_sorted = sorted(st.session_state.files_meta.keys())
    st.session_state.active_step_index = 0 if st.session_state.active_step_index >= len(steps_sorted) else st.session_state.active_step_index

# If no files loaded, show quick help
if not st.session_state.files_meta:
    st.info("No files loaded yet. Drag & drop `.plt` files above to begin.")
    st.stop()

# -------------------------
# Step Navigation
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
        st.experimental_rerun()

meta = st.session_state.files_meta[active_step]
st.caption(f"File: {meta['name']}")

# -------------------------
# Summaries & Indices
# -------------------------

@st.cache_data(show_spinner=False)
def read_file_summary(path: str, preferred_step: Optional[int]) -> Dict[str, Any]:
    info = {"path": path, "preferred_step": preferred_step}
    with h5py.File(path, "r") as f:
        dump_group = get_dump_group(f, preferred_step)
        info["dump_group"] = dump_group
        # basic metadata
        time = None; step = None; reset = None
        if dump_group:
            g = f[dump_group]
            time = float(np.array(g["Time"][()]).item()) if "Time" in g else None
            step = int(np.array(g["Step"][()]).item()) if "Step" in g else preferred_step
            reset = int(np.array(g["Reset_time_stage"][()]).item()) if "Reset_time_stage" in g else None
        info["time"] = time; info["step"] = step; info["reset_time_stage"] = reset

        # presence of other parts
        info["has_equations"] = "Equations" in f
        info["has_materials"] = "Materials" in f
        info["has_contact"] = bool(dump_group and "Contact" in f[dump_group])
        info["has_group_results"] = bool(dump_group and "group_results" in f[dump_group])

        # enumerate subgroups for selection
        info["contact_subgroups"] = []
        info["group_results_subgroups"] = []
        if info["has_contact"]:
            info["contact_subgroups"] = list_groups(f[path_join(dump_group, "Contact")])
        if info["has_group_results"]:
            info["group_results_subgroups"] = list_groups(f[path_join(dump_group, "group_results")])

        # Build quick indices for Equations & Materials
        eq_index = {}
        if info["has_equations"]:
            g = f["Equations"]
            eq_index = {k: ("group" if isinstance(v, h5py.Group) else "dataset") for k, v in g.items()}
        mt_index = {}
        if info["has_materials"]:
            g = f["Materials"]
            mt_index = {k: ("group" if isinstance(v, h5py.Group) else "dataset") for k, v in g.items()}
        info["equations_index"] = eq_index
        info["materials_index"] = mt_index

    return info

@st.cache_data(show_spinner=False)
def collect_cross_time_overview(files_meta: Dict[int, Dict[str,str]]) -> Dict[str, Any]:
    """Gather per-step: time, contact fault names, group_results names. Used to populate overlay choices."""
    overview = {"steps": [], "times": {}, "contact_names": set(), "group_results_names": set(), "contact_by_step": {}, "gr_by_step": {}}
    for step in sorted(files_meta.keys()):
        path = files_meta[step]["path"]
        with h5py.File(path, "r") as f:
            dump_group = get_dump_group(f, step)
            if not dump_group:
                continue
            g = f[dump_group]
            t = float(np.array(g["Time"][()]).item()) if "Time" in g else None
            overview["steps"].append(step)
            overview["times"][step] = t
            # Contact faults
            if "Contact" in g:
                names = list_groups(g["Contact"])
                overview["contact_by_step"][step] = names
                overview["contact_names"].update(names)
            # group_results
            if "group_results" in g:
                names = list_groups(g["group_results"])
                overview["gr_by_step"][step] = names
                overview["group_results_names"].update(names)
    return overview

summary = read_file_summary(meta["path"], active_step)
overview = collect_cross_time_overview(st.session_state.files_meta)

# -------------------------
# Sidebar Overview
# -------------------------

with st.sidebar:
    st.header("Overview")
    st.write(f"**Time:** {summary.get('time')}")
    st.write(f"**Step:** {summary.get('step')}")
    st.write(f"**Reset_time_stage:** {summary.get('reset_time_stage')}")
    st.divider()
    st.write("**Available parts**")
    st.checkbox("Dump", value=True, key="show_dump")
    st.checkbox("Equations", value=summary["has_equations"], key="show_equations", disabled=not summary["has_equations"])
    st.checkbox("Materials", value=summary["has_materials"], key="show_materials", disabled=not summary["has_materials"])

# -------------------------
# Equations & Materials
# -------------------------

def render_keyvalue_table(g: h5py.Group, title: str):
    st.subheader(title)
    ds_rows = []
    for k, v in g.items():
        if isinstance(v, h5py.Dataset):
            data = v[()]
            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1 and data.shape == ()):
                ds_rows.append({"name": k, "value": bytes_to_str(data.item() if np.isscalar(data) else data.tolist())})
    if ds_rows:
        st.dataframe(pd.DataFrame(ds_rows))
    # Explorer
    with st.expander("Browse all entries"):
        sel = st.selectbox("Select item", list(g.keys()))
        obj = g[sel]
        if isinstance(obj, h5py.Dataset):
            arr = safe_read_dataset(obj, max_preview=3000)
            df = pd.DataFrame(arr) if isinstance(arr, np.ndarray) and arr.ndim <= 2 else None
            st.write(f"Shape: {obj.shape} | dtype: {obj.dtype} | attrs: {len(obj.attrs)}")
            if df is not None and df.size > 0:
                st.dataframe(df.head(1000))
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{sel}.csv", mime="text/csv")
            else:
                st.text(repr(arr)[:2000])
        else:
            st.write("Group:")
            st.write(list(obj.keys())[:200])

if st.session_state.get("show_equations") and summary["has_equations"]:
    with st.expander("Equations", expanded=False):
        with h5py.File(meta["path"], "r") as f:
            render_keyvalue_table(f["Equations"], "Equations")

if st.session_state.get("show_materials") and summary["has_materials"]:
    with st.expander("Materials", expanded=False):
        with h5py.File(meta["path"], "r") as f:
            render_keyvalue_table(f["Materials"], "Materials")

# -------------------------
# DUMP (FEM) — Contact & group_results
# -------------------------

def get_mappings_for_group(g: h5py.Group, is_contact: bool) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    keys = list(g.keys())
    topo_key = match_key(keys, [r"Topology"])
    if is_contact:
        node_map_key = match_key(keys, [r"Contact\s*Node\s*number", r"Node\s*Numbers"])
        elem_map_key = match_key(keys, [r"Contact\s*Element\s*number", r"Element\s*Numbers"])
    else:
        node_map_key = match_key(keys, [r"Node\s*Numbers"])
        elem_map_key = match_key(keys, [r"Element\s*Numbers", r"Element\s*numbers"])
    return topo_key, node_map_key, elem_map_key

def classify_in_group(g: h5py.Group, is_contact: bool):
    topo_key, node_map_key, elem_map_key = get_mappings_for_group(g, is_contact)
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
    nodal_vars, elem_vars, unk_vars = classify_variables(g, num_nodes, num_elems, exclude)
    return {
        "topo_key": topo_key, "node_map_key": node_map_key, "elem_map_key": elem_map_key,
        "topology": topology, "node_numbers": node_numbers, "elem_numbers": elem_numbers,
        "node_inv": node_inv, "elem_inv": elem_inv, "num_elems": num_elems, "num_nodes": num_nodes,
        "nodal_vars": nodal_vars, "elem_vars": elem_vars, "unk_vars": unk_vars
    }

if st.session_state.get("show_dump"):
    st.header("Dump")
    if not summary["dump_group"]:
        st.warning("No Dump_* group found.")
    else:
        dump_gname = summary["dump_group"]
        with h5py.File(meta["path"], "r") as f:
            dump_g = f[dump_gname]
            st.write(f"Using `{dump_gname}`")

            tabs = st.tabs(["Contact", "Group Results", "Cross-Time & Export"])

            # -----------------
            # Contact tab
            # -----------------
            with tabs[0]:
                if not summary["has_contact"]:
                    st.info("No `Contact` found under Dump.")
                else:
                    faults = summary["contact_subgroups"]
                    fault = st.selectbox("Choose a discrete fault", faults, help="Each subfolder under Dump/Contact is a fault.")
                    fg = dump_g["Contact"][fault]

                    ctx = classify_in_group(fg, is_contact=True)
                    if not ctx:
                        st.error("Could not find Topology / Node Numbers / Element Numbers datasets in this fault.")
                    else:
                        st.markdown(f"**Elements:** {ctx['num_elems']} | **Nodes:** {ctx['num_nodes']}")
                        st.dataframe(pd.DataFrame(ctx["topology"]).head(20), use_container_width=True)

                        st.markdown("**Variable discovery**")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.write(f"Nodal vars: {len(ctx['nodal_vars'])}")
                        with c2: st.write(f"Element vars: {len(ctx['elem_vars'])}")
                        with c3: 
                            if len(ctx["unk_vars"])>0:
                                st.write(f"Unknown vars: {len(ctx['unk_vars'])}")
                                with st.expander("See unknowns"):
                                    st.write(ctx["unk_vars"])

                        mode = st.radio("Pick variable type", ["Nodal", "Element"], horizontal=True, key="ct_mode_contact")
                        var_list = ctx["nodal_vars"] if mode == "Nodal" else ctx["elem_vars"]
                        if not var_list:
                            st.info(f"No {mode.lower()} variables detected here.")
                        else:
                            var = st.selectbox("Variable", var_list, key="ct_var_contact")
                            arr = np.array(fg[var])
                            comp_index = None
                            if arr.ndim >= 2 and arr.shape[1] <= 10:
                                comp_index = st.number_input("Component index (column)", min_value=0, max_value=arr.shape[1]-1, value=0, step=1, key="ct_comp_contact")
                            id_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs) to extract", value="", placeholder="e.g., 1,2,3-10", key="ct_ids_contact")
                            id_list = parse_int_list(id_text)
                            if id_list:
                                inv = ctx["node_inv"] if mode == "Nodal" else ctx["elem_inv"]
                                ids = [inv.get(n) for n in id_list]
                                missing = [n for n, i in zip(id_list, ids) if i is None]
                                ids = [i for i in ids if i is not None]
                                if missing:
                                    st.warning(f"Numbers not found and skipped: {missing}")
                                values = []
                                for i in ids:
                                    try:
                                        if arr.ndim == 1:
                                            values.append(arr[i])
                                        elif arr.ndim >= 2:
                                            values.append(arr[i, comp_index if comp_index is not None else 0])
                                    except Exception:
                                        values.append(np.nan)
                                df = pd.DataFrame({
                                    f"{mode}_number": [n for n, i in zip(id_list, ids + [None]*(len(id_list)-len(ids))) if i is not None],
                                    f"{mode}_id": ids,
                                    "value": [float(v) if np.isscalar(v) else float(np.array(v).item()) for v in values]
                                })
                                st.subheader("Preview")
                                st.dataframe(df, use_container_width=True)

                                fig = line_fig(
                                    x=df[f"{mode}_number"],
                                    series={var if comp_index is None else f"{var} [comp {comp_index}]": df["value"]},
                                    x_label=f"{mode} number",
                                    y_label=var if comp_index is None else f"{var} [comp {comp_index}]",
                                    use_gl=True,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.download_button(
                                    "Download plot (HTML)",
                                    fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                                    file_name="plot.html",
                                    mime="text/html",
                                )

                                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{fault}_{mode.lower()}_{var}.csv", mime="text/csv")

                            with st.expander("See raw array head"):
                                preview = safe_read_dataset(fg[var], max_preview=1000)
                                if isinstance(preview, np.ndarray) and preview.ndim <= 2:
                                    st.dataframe(pd.DataFrame(preview).head(20))
                                else:
                                    st.text(repr(preview)[:2000])

            # -----------------
            # group_results tab
            # -----------------
            with tabs[1]:
                if not summary["has_group_results"]:
                    st.info("No `group_results` found under Dump.")
                else:
                    groups = summary["group_results_subgroups"]
                    group_name = st.selectbox("Choose a group (e.g., Reservoir)", groups, key="gr_group_select")
                    gg = dump_g["group_results"][group_name]

                    ctx = classify_in_group(gg, is_contact=False)
                    if not ctx:
                        st.error("Could not find Topology / Node Numbers / Element Numbers datasets in this group.")
                    else:
                        st.markdown(f"**Elements:** {ctx['num_elems']} | **Nodes:** {ctx['num_nodes']}")
                        st.dataframe(pd.DataFrame(ctx['topology']).head(20), use_container_width=True)

                        st.markdown("**Variable discovery**")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.write(f"Nodal vars: {len(ctx['nodal_vars'])}")
                        with c2: st.write(f"Element vars: {len(ctx['elem_vars'])}")
                        with c3: 
                            if len(ctx["unk_vars"])>0:
                                st.write(f"Unknown vars: {len(ctx['unk_vars'])}")
                                with st.expander("See unknowns"):
                                    st.write(ctx["unk_vars"])

                        mode = st.radio("Pick variable type", ["Nodal", "Element"], horizontal=True, key="gr_mode")
                        var_list = ctx["nodal_vars"] if mode == "Nodal" else ctx["elem_vars"]
                        if not var_list:
                            st.info(f"No {mode.lower()} variables detected here.")
                        else:
                            var = st.selectbox("Variable", var_list, key="gr_var")
                            arr = np.array(gg[var])
                            comp_index = None
                            if arr.ndim >= 2 and arr.shape[1] <= 10:
                                comp_index = st.number_input("Component index (column)", min_value=0, max_value=arr.shape[1]-1, value=0, step=1, key="gr_comp")

                            id_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs) to extract", value="", placeholder="e.g., 1,2,3-10", key="gr_ids")
                            id_list = parse_int_list(id_text)
                            if id_list:
                                inv = ctx["node_inv"] if mode == "Nodal" else ctx["elem_inv"]
                                ids = [inv.get(n) for n in id_list]
                                missing = [n for n, i in zip(id_list, ids) if i is None]
                                ids = [i for i in ids if i is not None]
                                if missing:
                                    st.warning(f"Numbers not found and skipped: {missing}")
                                values = []
                                for i in ids:
                                    try:
                                        if arr.ndim == 1:
                                            values.append(arr[i])
                                        elif arr.ndim >= 2:
                                            values.append(arr[i, comp_index if comp_index is not None else 0])
                                    except Exception:
                                        values.append(np.nan)
                                df = pd.DataFrame({
                                    f"{mode}_number": [n for n, i in zip(id_list, ids + [None]*(len(id_list)-len(ids))) if i is not None],
                                    f"{mode}_id": ids,
                                    "value": [float(v) if np.isscalar(v) else float(np.array(v).item()) for v in values]
                                })
                                st.subheader("Preview")
                                st.dataframe(df, use_container_width=True)

                                fig = line_fig(
                                    x=df[f"{mode}_number"],
                                    series={var if comp_index is None else f"{var} [comp {comp_index}]": df["value"]},
                                    x_label=f"{mode} number",
                                    y_label=var if comp_index is None else f"{var} [comp {comp_index}]",
                                    use_gl=True,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.download_button(
                                    "Download plot (HTML)",
                                    fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                                    file_name="plot.html",
                                    mime="text/html",
                                )

                                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{group_name}_{mode.lower()}_{var}.csv", mime="text/csv")

                            with st.expander("See raw array head"):
                                preview = safe_read_dataset(gg[var], max_preview=1000)
                                if isinstance(preview, np.ndarray) and preview.ndim <= 2:
                                    st.dataframe(pd.DataFrame(preview).head(20))
                                else:
                                    st.text(repr(preview)[:2000])

            # -----------------
            # Cross-Time & Export tab
            # -----------------
            with tabs[2]:
                st.subheader("Cross-Time Overlays")
                overlay_scope = st.radio("Data source", ["Contact faults", "Group results"], horizontal=True)

                if overlay_scope == "Contact faults":
                    # Choose fault from union over steps
                    all_faults = sorted(overview["contact_names"])
                    if not all_faults:
                        st.info("No Contact faults found across the loaded files.")
                    else:
                        fault_sel = st.selectbox("Fault", all_faults, key="ct_fault_sel")
                        mode = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="ct_over_mode")
                        # Variable list from active step (fallback if empty later)
                        try:
                            fg = dump_g["Contact"][fault_sel]
                            ctx_act = classify_in_group(fg, is_contact=True)
                            base_vars = ctx_act["nodal_vars"] if mode == "Nodal" else ctx_act["elem_vars"]
                        except Exception:
                            base_vars = []
                        var = st.selectbox("Variable", base_vars, key="ct_over_var")
                        comp_index = st.number_input("Component index (column, if applicable)", min_value=0, value=0, step=1, key="ct_over_comp")
                        id_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs)", value="", placeholder="e.g., 1,2,3-10", key="ct_over_ids")
                        id_list = parse_int_list(id_text)
                        x_axis = st.radio("X-axis", ["Time", "Step"], horizontal=True, key="ct_over_x")

                        if st.button("Build overlay", key="ct_over_build"):
                            rows = []
                            missing_steps = []
                            for step in overview["steps"]:
                                fmeta = st.session_state.files_meta.get(step)
                                if not fmeta: 
                                    continue
                                path = fmeta["path"]
                                try:
                                    with h5py.File(path, "r") as ff:
                                        dg = get_dump_group(ff, step)
                                        if not dg or "Contact" not in ff[dg] or fault_sel not in ff[dg]["Contact"]:
                                            missing_steps.append(step); continue
                                        g = ff[dg]["Contact"][fault_sel]
                                        ctx = classify_in_group(g, is_contact=True)
                                        if not ctx: 
                                            missing_steps.append(step); continue
                                        var_ds = g.get(var, None)
                                        if var_ds is None:
                                            missing_steps.append(step); continue
                                        arr = np.array(var_ds)
                                        inv = ctx["node_inv"] if mode == "Nodal" else ctx["elem_inv"]
                                        ids = [inv.get(n) for n in id_list] if id_list else []
                                        t = float(np.array(ff[dg]["Time"][()]).item()) if "Time" in ff[dg] else None
                                        for n, idx in zip(id_list, ids):
                                            if idx is None: 
                                                continue
                                            try:
                                                val = arr[idx] if arr.ndim == 1 else arr[idx, min(comp_index, arr.shape[1]-1)]
                                            except Exception:
                                                val = np.nan
                                            rows.append({"step": step, "time": t, "entity_number": n, "value": float(np.array(val).item()) if not np.isnan(val) else np.nan})
                                except Exception:
                                    missing_steps.append(step)
                            if not rows:
                                st.warning("No data found for the selection.")
                            else:
                                df = pd.DataFrame(rows).sort_values("step")
                                st.dataframe(df, use_container_width=True)
                                # Plot overlay: one line per entity_number
                                x_col = "time" if x_axis == "Time" else "step"
                                fig = go.Figure()
                                for n, sub in df.groupby("entity_number"):
                                    fig.add_trace(go.Scattergl(x=sub[x_col], y=sub["value"], mode="lines+markers", name=f"{mode} {n}"))
                                fig.update_layout(hovermode="x unified", xaxis_title=x_axis, yaxis_title=var, margin=dict(l=40, r=10, t=30, b=40))
                                st.plotly_chart(fig, use_container_width=True)
                                st.download_button(
                                    "Download overlay (HTML)",
                                    fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                                    file_name="overlay.html",
                                    mime="text/html",
                                )
                                st.download_button("Download overlay CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"overlay_contact_{fault_sel}_{mode.lower()}_{var}.csv", mime="text/csv")
                                if missing_steps:
                                    st.caption(f"Skipped steps without data: {sorted(set(missing_steps))}")

                else:
                    # Group results
                    all_groups = sorted(overview["group_results_names"])
                    if not all_groups:
                        st.info("No group_results groups found across the loaded files.")
                    else:
                        group_sel = st.selectbox("Group", all_groups, key="ct_group_sel")
                        mode = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="ct_over_mode_gr")
                        # Variable list from active step
                        try:
                            gg = dump_g["group_results"][group_sel]
                            ctx_act = classify_in_group(gg, is_contact=False)
                            base_vars = ctx_act["nodal_vars"] if mode == "Nodal" else ctx_act["elem_vars"]
                        except Exception:
                            base_vars = []
                        var = st.selectbox("Variable", base_vars, key="ct_over_var_gr")
                        comp_index = st.number_input("Component index (column, if applicable)", min_value=0, value=0, step=1, key="ct_over_comp_gr")
                        id_text = st.text_input(f"Enter {mode.lower()} NUMBERS (not IDs)", value="", placeholder="e.g., 1,2,3-10", key="ct_over_ids_gr")
                        id_list = parse_int_list(id_text)
                        x_axis = st.radio("X-axis", ["Time", "Step"], horizontal=True, key="ct_over_x_gr")

                        if st.button("Build overlay", key="ct_over_build_gr"):
                            rows = []
                            missing_steps = []
                            for step in overview["steps"]:
                                fmeta = st.session_state.files_meta.get(step)
                                if not fmeta: 
                                    continue
                                path = fmeta["path"]
                                try:
                                    with h5py.File(path, "r") as ff:
                                        dg = get_dump_group(ff, step)
                                        if not dg or "group_results" not in ff[dg] or group_sel not in ff[dg]["group_results"]:
                                            missing_steps.append(step); continue
                                        g = ff[dg]["group_results"][group_sel]
                                        ctx = classify_in_group(g, is_contact=False)
                                        if not ctx: 
                                            missing_steps.append(step); continue
                                        var_ds = g.get(var, None)
                                        if var_ds is None:
                                            missing_steps.append(step); continue
                                        arr = np.array(var_ds)
                                        inv = ctx["node_inv"] if mode == "Nodal" else ctx["elem_inv"]
                                        ids = [inv.get(n) for n in id_list] if id_list else []
                                        t = float(np.array(ff[dg]["Time"][()]).item()) if "Time" in ff[dg] else None
                                        for n, idx in zip(id_list, ids):
                                            if idx is None: 
                                                continue
                                            try:
                                                val = arr[idx] if arr.ndim == 1 else arr[idx, min(comp_index, arr.shape[1]-1)]
                                            except Exception:
                                                val = np.nan
                                            rows.append({"step": step, "time": t, "entity_number": n, "value": float(np.array(val).item()) if not np.isnan(val) else np.nan})
                                except Exception:
                                    missing_steps.append(step)
                            if not rows:
                                st.warning("No data found for the selection.")
                            else:
                                df = pd.DataFrame(rows).sort_values("step")
                                st.dataframe(df, use_container_width=True)
                                fig = go.Figure()
                                x_col = "time" if x_axis == "Time" else "step"
                                for n, sub in df.groupby("entity_number"):
                                    fig.add_trace(go.Scattergl(x=sub[x_col], y=sub["value"], mode="lines+markers", name=f"{mode} {n}"))
                                fig.update_layout(hovermode="x unified", xaxis_title=x_axis, yaxis_title=var, margin=dict(l=40, r=10, t=30, b=40))
                                st.plotly_chart(fig, use_container_width=True)
                                st.download_button(
                                    "Download overlay (HTML)",
                                    fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                                    file_name="overlay.html",
                                    mime="text/html",
                                )
                                st.download_button("Download overlay CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"overlay_group_{group_sel}_{mode.lower()}_{var}.csv", mime="text/csv")
                                if missing_steps:
                                    st.caption(f"Skipped steps without data: {sorted(set(missing_steps))}")

                st.divider()
                st.subheader("Export Bundles (ZIP)")
                st.caption("Bundle one or more variables across selected time steps into CSV files (one CSV per variable).")

                steps_choice = st.multiselect("Select steps", overview["steps"], default=overview["steps"], key="bundle_steps")
                exp_scope = st.radio("Bundle source", ["Contact faults", "Group results"], horizontal=True, key="bundle_scope")

                if exp_scope == "Contact faults":
                    all_faults = sorted(overview["contact_names"])
                    if not all_faults:
                        st.info("No Contact faults found across the loaded files.")
                    else:
                        fault_sel_b = st.selectbox("Fault", all_faults, key="bundle_fault")
                        mode_b = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="bundle_mode_contact")
                        # Variables list from active step if possible
                        try:
                            fg = dump_g["Contact"][fault_sel_b]
                            ctx_act = classify_in_group(fg, is_contact=True)
                            vars_list = ctx_act["nodal_vars"] if mode_b == "Nodal" else ctx_act["elem_vars"]
                        except Exception:
                            vars_list = []
                        vars_pick = st.multiselect("Variables to export", vars_list, key="bundle_vars_contact")
                        comp_index_b = st.number_input("Component index (column, if applicable)", min_value=0, value=0, step=1, key="bundle_comp_contact")
                        id_text_b = st.text_input(f"Enter {mode_b.lower()} NUMBERS (not IDs)", value="", placeholder="e.g., 1,2,3-10", key="bundle_ids_contact")
                        id_list_b = parse_int_list(id_text_b)

                        if st.button("Build ZIP", key="bundle_build_contact") and vars_pick and steps_choice:
                            mem = io.BytesIO()
                            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                for var in vars_pick:
                                    rows = []
                                    for step in steps_choice:
                                        fmeta = st.session_state.files_meta.get(step)
                                        if not fmeta: 
                                            continue
                                        path = fmeta["path"]
                                        try:
                                            with h5py.File(path, "r") as ff:
                                                dg = get_dump_group(ff, step)
                                                if not dg or "Contact" not in ff[dg] or fault_sel_b not in ff[dg]["Contact"]:
                                                    continue
                                                g = ff[dg]["Contact"][fault_sel_b]
                                                ctx = classify_in_group(g, is_contact=True)
                                                if not ctx or var not in g: 
                                                    continue
                                                arr = np.array(g[var])
                                                inv = ctx["node_inv"] if mode_b == "Nodal" else ctx["elem_inv"]
                                                ids = [inv.get(n) for n in id_list_b] if id_list_b else []
                                                t = float(np.array(ff[dg]["Time"][()]).item()) if "Time" in ff[dg] else None
                                                for n, idx in zip(id_list_b, ids):
                                                    if idx is None: continue
                                                    try:
                                                        val = arr[idx] if arr.ndim == 1 else arr[idx, min(comp_index_b, arr.shape[1]-1)]
                                                    except Exception:
                                                        val = np.nan
                                                    rows.append({"step": step, "time": t, "entity_number": n, "value": float(np.array(val).item()) if not np.isnan(val) else np.nan})
                                        except Exception:
                                            continue
                                    if rows:
                                        dfv = pd.DataFrame(rows).sort_values("step")
                                        zf.writestr(f"{fault_sel_b}_{mode_b.lower()}_{var}.csv", dfv.to_csv(index=False))
                            mem.seek(0)
                            st.download_button("Download ZIP", mem, file_name=f"bundle_contact_{fault_sel_b}_{mode_b.lower()}.zip", mime="application/zip")

                else:
                    # group_results
                    all_groups = sorted(overview["group_results_names"])
                    if not all_groups:
                        st.info("No group_results groups found across the loaded files.")
                    else:
                        group_sel_b = st.selectbox("Group", all_groups, key="bundle_group")
                        mode_b = st.radio("Variable type", ["Nodal", "Element"], horizontal=True, key="bundle_mode_gr")
                        # Variables list from active step
                        try:
                            gg = dump_g["group_results"][group_sel_b]
                            ctx_act = classify_in_group(gg, is_contact=False)
                            vars_list = ctx_act["nodal_vars"] if mode_b == "Nodal" else ctx_act["elem_vars"]
                        except Exception:
                            vars_list = []
                        vars_pick = st.multiselect("Variables to export", vars_list, key="bundle_vars_gr")
                        comp_index_b = st.number_input("Component index (column, if applicable)", min_value=0, value=0, step=1, key="bundle_comp_gr")
                        id_text_b = st.text_input(f"Enter {mode_b.lower()} NUMBERS (not IDs)", value="", placeholder="e.g., 1,2,3-10", key="bundle_ids_gr")
                        id_list_b = parse_int_list(id_text_b)

                        if st.button("Build ZIP", key="bundle_build_gr") and vars_pick and steps_choice:
                            mem = io.BytesIO()
                            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                for var in vars_pick:
                                    rows = []
                                    for step in steps_choice:
                                        fmeta = st.session_state.files_meta.get(step)
                                        if not fmeta: 
                                            continue
                                        path = fmeta["path"]
                                        try:
                                            with h5py.File(path, "r") as ff:
                                                dg = get_dump_group(ff, step)
                                                if not dg or "group_results" not in ff[dg] or group_sel_b not in ff[dg]["group_results"]:
                                                    continue
                                                g = ff[dg]["group_results"][group_sel_b]
                                                ctx = classify_in_group(g, is_contact=False)
                                                if not ctx or var not in g: 
                                                    continue
                                                arr = np.array(g[var])
                                                inv = ctx["node_inv"] if mode_b == "Nodal" else ctx["elem_inv"]
                                                ids = [inv.get(n) for n in id_list_b] if id_list_b else []
                                                t = float(np.array(ff[dg]["Time"][()]).item()) if "Time" in ff[dg] else None
                                                for n, idx in zip(id_list_b, ids):
                                                    if idx is None: continue
                                                    try:
                                                        val = arr[idx] if arr.ndim == 1 else arr[idx, min(comp_index_b, arr.shape[1]-1)]
                                                    except Exception:
                                                        val = np.nan
                                                    rows.append({"step": step, "time": t, "entity_number": n, "value": float(np.array(val).item()) if not np.isnan(val) else np.nan})
                                        except Exception:
                                            continue
                                    if rows:
                                        dfv = pd.DataFrame(rows).sort_values("step")
                                        zf.writestr(f"{group_sel_b}_{mode_b.lower()}_{var}.csv", dfv.to_csv(index=False))
                            mem.seek(0)
                            st.download_button("Download ZIP", mem, file_name=f"bundle_group_{group_sel_b}_{mode_b.lower()}.zip", mime="application/zip")

# -------------------------
# Footer
# -------------------------
st.caption("Tip: Use the left/right buttons at the top to switch time steps. Variables, overlays, and exports update automatically.")

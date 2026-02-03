# NextBillion.ai — Multi-Driver Routing Testbench (Streamlit)
# Deploy on Streamlit Community Cloud. Set secret NEXTBILLION_API_KEY in Streamlit Secrets.
#
# Key features:
# - Stops editor with optional time windows
# - Driver/vehicle generator (default 25 for testing)
# - Direct multi-vehicle Route Optimization (VRP) with Open vs Round-trip toggle
# - Fair before/after comparison (ensures same trip type)
# - Optional: raw Clustering API runner (paste JSON body)
# - Lightweight backend logging (Cloud logs + in-app session logs)

import os
import time
import json
import uuid
import random
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium


# =========================
# CONFIG
# =========================
NB_BASE = "https://api.nextbillion.io"
UA = {"User-Agent": "NextBillion-Routing-Testbench/1.0"}

st.set_page_config(page_title="NextBillion.ai — Routing Testbench", layout="wide")

DEFAULT_API_KEY = st.secrets.get("NEXTBILLION_API_KEY", os.getenv("NEXTBILLION_API_KEY", ""))
API_KEY_UI = st.sidebar.text_input("NextBillion API key", type="password", value=DEFAULT_API_KEY, help="Prefer Streamlit Secrets: NEXTBILLION_API_KEY")


# =========================
# LIGHTWEIGHT LOGGING (perf-safe)
# =========================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mask_secret(s: str, keep_last: int = 4) -> str:
    s = "" if s is None else str(s)
    if not s:
        return s
    if len(s) <= keep_last:
        return "*" * len(s)
    return "*" * (len(s) - keep_last) + s[-keep_last:]


def _init_logger() -> logging.Logger:
    if "run_id" not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())[:8]
    if "audit_events" not in st.session_state:
        st.session_state.audit_events = []
    st.session_state.setdefault("audit_enabled", False)     # verbose events
    st.session_state.setdefault("audit_state_diff", False)  # expensive
    st.session_state.setdefault("audit_console_all", False) # cloud logs spam
    st.session_state.setdefault("audit_max_events", 600)

    logger = logging.getLogger("nb_audit")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = _init_logger()


def audit(event: str, level: str = "INFO", force_console: bool = False, **fields):
    """Logs to in-app buffer always; optionally to console (Cloud logs)."""
    payload = {"ts": _utc_now_iso(), "run_id": st.session_state.get("run_id", ""), "event": event, **fields}
    st.session_state.audit_events.append(payload)
    if len(st.session_state.audit_events) > int(st.session_state.audit_max_events):
        st.session_state.audit_events = st.session_state.audit_events[-int(st.session_state.audit_max_events):]

    to_console = force_console or st.session_state.get("audit_console_all", False) or event.startswith("api_") or level == "ERROR"
    if to_console:
        msg = json.dumps(payload, ensure_ascii=False)
        getattr(LOG, level.lower(), LOG.info)(msg)


def audit_state_changes():
    if not st.session_state.get("audit_state_diff", False):
        return
    snap_key = "_prev_state_snapshot"
    prev = st.session_state.get(snap_key, {})
    now = {}
    # Only snapshot simple keys to avoid large diffs
    for k, v in st.session_state.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            now[k] = v
    # Diff
    changed = {k: {"from": prev.get(k), "to": now.get(k)} for k in now.keys() if prev.get(k) != now.get(k)}
    if changed:
        audit("state_change", changes=changed)
    st.session_state[snap_key] = now


audit_state_changes()


# =========================
# HTTP HELPERS
# =========================
def _req(method: str, path: str, *, params: Optional[dict] = None, body: Optional[dict] = None, timeout: int = 45) -> requests.Response:
    if not API_KEY_UI:
        raise RuntimeError("Missing API key. Set NEXTBILLION_API_KEY in Streamlit Secrets or paste in sidebar.")
    url = f"{NB_BASE}{path}"
    params = dict(params or {})
    params["key"] = API_KEY_UI

    safe_params = dict(params)
    safe_params["key"] = _mask_secret(API_KEY_UI)

    t0 = time.time()
    audit("api_request", method=method.upper(), url=url, params=safe_params, has_body=bool(body), force_console=True)
    r = requests.request(method.upper(), url, params=params, json=body, headers=UA, timeout=timeout)
    dt_ms = int((time.time() - t0) * 1000)
    audit("api_response", method=method.upper(), url=str(r.url), status=r.status_code, latency_ms=dt_ms, bytes=len(r.content or b""), force_console=True)
    return r


def nb_geocode(query: str, country: str = "", limit: int = 5) -> dict:
    params = {"q": query, "limit": limit}
    if country:
        params["country"] = country
    r = _req("GET", "/geocode", params=params)
    return r.json()


def nb_directions(coords: List[Tuple[float, float]], mode: str = "car", option: str = "fast", avoid: str = "") -> dict:
    # Directions API expects "origin" and "destination" and optionally "waypoints".
    origin = f"{coords[0][1]},{coords[0][0]}"       # lng,lat
    destination = f"{coords[-1][1]},{coords[-1][0]}"
    params = {"origin": origin, "destination": destination, "mode": mode, "option": option}
    if len(coords) > 2:
        # waypoints: lng,lat|lng,lat
        params["waypoints"] = "|".join([f"{lng},{lat}" for (lat, lng) in coords[1:-1]])
    if avoid:
        params["avoid"] = avoid
    r = _req("GET", "/directions", params=params)
    return r.json()


def nb_optimize_create(payload: dict) -> dict:
    # Route Optimization API (task)
    r = _req("POST", "/optimization/v2", body=payload)
    return r.json()


def nb_optimize_result(task_id: str) -> dict:
    r = _req("GET", "/optimization/v2/result", params={"id": task_id})
    return r.json()


def nb_cluster_create(payload: dict) -> dict:
    # Clustering endpoint per docs: POST /clustering?key=... citeturn1view0
    r = _req("POST", "/clustering", body=payload)
    return r.json()


def nb_cluster_result(task_id: str) -> dict:
    # GET /clustering/result?id=...&key=... citeturn1view0
    r = _req("GET", "/clustering/result", params={"id": task_id})
    return r.json()


# =========================
# POLYLINE DECODER (polyline6 by default)
# =========================
def decode_polyline(polyline_str: str, precision: int = 6) -> List[Tuple[float, float]]:
    """Decodes a Google-style encoded polyline string."""
    if not polyline_str:
        return []
    index, lat, lng = 0, 0, 0
    coordinates = []
    factor = 10 ** precision

    while index < len(polyline_str):
        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / factor, lng / factor))
    return coordinates


# =========================
# DEFAULT DATA
# =========================
def default_stops_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stop_id": "S1", "address": "Jaipur Junction Railway Station", "lat": 26.9196, "lng": 75.7873, "tw_start": "09:00", "tw_end": "18:00"},
            {"stop_id": "S2", "address": "Mansarovar, Jaipur", "lat": 26.8531, "lng": 75.7644, "tw_start": "10:00", "tw_end": "16:00"},
            {"stop_id": "S3", "address": "Vaishali Nagar, Jaipur", "lat": 26.9066, "lng": 75.7385, "tw_start": "09:30", "tw_end": "17:30"},
            {"stop_id": "S4", "address": "Malviya Nagar, Jaipur", "lat": 26.8538, "lng": 75.8120, "tw_start": "11:00", "tw_end": "18:00"},
        ]
    )


def default_depots_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"depot_id": "D1", "name": "Default Depot", "lat": 26.9124, "lng": 75.7873},
        ]
    )


def hhmm_to_seconds(s: str) -> int:
    """Convert HH:MM to seconds from midnight."""
    try:
        hh, mm = s.strip().split(":")
        return int(hh) * 3600 + int(mm) * 60
    except Exception:
        return 0


def seconds_to_hhmm(secs: int) -> str:
    secs = max(0, int(secs))
    hh = (secs // 3600) % 24
    mm = (secs % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


# =========================
# SESSION INIT
# =========================
if "stops_df" not in st.session_state:
    st.session_state.stops_df = default_stops_df()
if "depots_df" not in st.session_state:
    st.session_state.depots_df = default_depots_df()

st.title("NextBillion.ai — Multi‑Driver Routing Testbench")
st.caption("Build stops → configure 25 drivers → optimize (time windows supported) → compare directions before/after → export JSONs.")

# Sidebar controls
with st.sidebar:
    st.subheader("Run settings")
    mode = st.selectbox("Planner mode", ["Direct VRP (25 drivers)", "Single route (1 driver)"], index=0)

    return_to_start = st.checkbox("Round trip (return to depot)", value=True,
                                  help="When enabled, vehicle end_index is same as start_index (depot). When disabled, routes are open. The Route Optimization API supports start_index and end_index. citeturn1view1")

    objective = st.selectbox("Optimization objective (travel_cost)", ["duration", "distance"], index=0,
                             help="Controls what the solver tries to minimize (time vs distance).")

    # Directions settings
    st.subheader("Directions settings")
    directions_mode = st.selectbox("Directions mode", ["car", "truck"], index=0)
    directions_option = st.selectbox("Directions option", ["fast", "flexible"], index=0)
    avoid = st.text_input("Avoid (comma-separated)", value="", help="Examples: tolls,highways. citeturn0search2")

    # Debug toggles
    st.subheader("Debug")
    st.session_state.audit_enabled = st.toggle("Enable verbose audit (slower)", value=st.session_state.audit_enabled)
    st.session_state.audit_state_diff = st.toggle("Track widget state diffs (slow)", value=st.session_state.audit_state_diff)
    st.session_state.audit_console_all = st.toggle("Log ALL events to Cloud logs (slow)", value=st.session_state.audit_console_all)


tabs = st.tabs(["Stops", "Depots / Drivers", "Optimize", "Results", "Clustering (raw)", "Logs"])

# -------------------------
# Stops tab
# -------------------------
with tabs[0]:
    st.subheader("Stops")
    st.write("Edit stops here. For time windows, use HH:MM (same-day).")
    df = st.session_state.stops_df.copy()

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key="stops_editor",
        column_config={
            "stop_id": st.column_config.TextColumn(required=True),
            "address": st.column_config.TextColumn(),
            "lat": st.column_config.NumberColumn(format="%.6f"),
            "lng": st.column_config.NumberColumn(format="%.6f"),
            "tw_start": st.column_config.TextColumn(help="HH:MM"),
            "tw_end": st.column_config.TextColumn(help="HH:MM"),
        },
    )
    st.session_state.stops_df = edited

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("Reset sample stops", width="content"):
            st.session_state.stops_df = default_stops_df()
            audit("button_clicked", widget="reset_sample_stops")
            st.rerun()

    with colB:
        country = st.text_input("Country filter (optional)", value="IN", help="Helps geocode precision.")
    with colC:
        st.write("Geocode helper: enter an address and click search.")
        q = st.text_input("Geocode query", value="", key="geocode_query")
        if st.button("Geocode lookup", width="content"):
            audit("button_clicked", widget="geocode_lookup", query=q)
            try:
                res = nb_geocode(q, country=country.strip(), limit=5)
                st.session_state.last_geocode_json = res
                st.json(res)
            except Exception as e:
                audit("error", where="geocode_lookup", msg=str(e), level="ERROR")
                st.error(str(e))

    if st.session_state.get("last_geocode_json"):
        st.download_button(
            "Download last Geocode JSON",
            data=json.dumps(st.session_state.last_geocode_json, ensure_ascii=False),
            file_name="geocode_last.json",
            mime="application/json",
            width="content",
        )

# -------------------------
# Depots / Drivers tab
# -------------------------
with tabs[1]:
    st.subheader("Depots & Driver setup")

    depots_df = st.session_state.depots_df.copy()
    st.write("Add one or more candidate depots. For multi-depot testing, add several.")
    depots_edit = st.data_editor(
        depots_df,
        num_rows="dynamic",
        width="stretch",
        key="depots_editor",
        column_config={
            "depot_id": st.column_config.TextColumn(required=True),
            "name": st.column_config.TextColumn(),
            "lat": st.column_config.NumberColumn(format="%.6f"),
            "lng": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.session_state.depots_df = depots_edit

    st.divider()
    st.subheader("Driver / Vehicle generator (testing)")
    driver_count = st.number_input("Number of drivers (vehicles)", min_value=1, max_value=200, value=25, step=1)
    shift_start = st.text_input("Shift start (HH:MM)", value="09:00")
    shift_end = st.text_input("Shift end (HH:MM)", value="18:00")

    depot_mode = st.selectbox(
        "Depot mode",
        ["Single depot for all drivers (use first depot)", "Random depot per driver (from list)", "Open route: start at first job (no depot)"],
        index=0,
        help="Route Optimization supports specifying vehicle start/end locations via start_index/end_index (location list indices). citeturn1view1",
    )

    st.info(
        "Note on 'depot can change': the solver won’t invent depot locations — it can only use the depot candidates you provide. "
        "So 'dynamic depot' = choose from your depot list (random or per driver), or use open routes."
    )

# -------------------------
# Optimize tab
# -------------------------
def build_vrp_payload(
    stops: pd.DataFrame,
    depots: pd.DataFrame,
    *,
    mode: str,
    driver_count: int,
    depot_mode: str,
    return_to_start: bool,
    objective: str,
    shift_start: str,
    shift_end: str,
) -> Tuple[dict, List[Tuple[float, float]], Dict[int, str]]:
    """
    Returns:
    - payload for /optimization/v2
    - locations list (lat,lng) in index order
    - location_index -> label mapping for UI
    """
    # Build locations list: [depots..., stops...]
    locs: List[Tuple[float, float]] = []
    loc_labels: Dict[int, str] = {}

    # Depots first
    for _, d in depots.iterrows():
        loc_labels[len(locs)] = f"DEPOT:{d.get('depot_id','')}"
        locs.append((float(d["lat"]), float(d["lng"])))

    depot_count = len(locs)

    # Stops next
    for _, s in stops.iterrows():
        loc_labels[len(locs)] = f"STOP:{s.get('stop_id','')}"
        locs.append((float(s["lat"]), float(s["lng"])))

    # Jobs reference stop locations by index
    jobs = []
    for i, s in enumerate(stops.itertuples(index=False), start=0):
        location_index = depot_count + i
        tw_start = hhmm_to_seconds(getattr(s, "tw_start", "") or "00:00")
        tw_end = hhmm_to_seconds(getattr(s, "tw_end", "") or "23:59")
        # Route Optimization API jobs can include time_windows; schema described in docs. citeturn1view1
        jobs.append(
            {
                "id": int(i + 1),  # must be non-zero positive int in many NB APIs
                "location_index": int(location_index),
                "service": 0,  # no service time (your requirement)
                "time_windows": [[int(tw_start), int(tw_end)]],
            }
        )

    # Vehicles
    vehicles = []
    ss = hhmm_to_seconds(shift_start)
    se = hhmm_to_seconds(shift_end)
    if se <= ss:
        se = ss + 8 * 3600

    def pick_start_index(v_i: int) -> Optional[int]:
        if depot_mode.startswith("Open route"):
            return None
        if depot_count <= 0:
            return None
        if depot_mode.startswith("Random depot"):
            return int(v_i % depot_count)
        return 0

    for v in range(int(driver_count)):
        start_idx = pick_start_index(v)
        # end_idx logic: if open route -> omit; else round trip vs open
        if start_idx is None:
            # open route with no explicit depot
            veh = {"id": int(v + 1)}
        else:
            end_idx = int(start_idx) if return_to_start else int(start_idx)  # for open routes with depot, we still set end_index below
            # If not returning to depot, leave end_index unspecified to allow solver to end at last job.
            veh = {"id": int(v + 1), "start_index": int(start_idx)}
            if return_to_start:
                veh["end_index"] = int(start_idx)
        # Shift window (vehicle availability)
        veh["time_window"] = [int(ss), int(se)]
        vehicles.append(veh)

    payload = {
        "description": f"Routing testbench | {mode} | drivers={driver_count}",
        "locations": {"id": 1, "location": [[lng, lat] for (lat, lng) in locs]},  # NB uses [lng,lat] in many APIs
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {"objective": {"travel_cost": objective}},
    }

    return payload, locs, loc_labels


def parse_vrp_routes(opt_result: dict) -> List[dict]:
    """
    Attempts to parse routes from 'result' which may be JSON string.
    """
    if not opt_result:
        return []
    r = opt_result.get("result")
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except Exception:
            return []
    if not isinstance(r, dict):
        return []
    return r.get("routes", []) or []


def route_steps_to_location_indices(route: dict) -> List[int]:
    steps = route.get("steps", []) or []
    out = []
    for s in steps:
        # Steps contain either location_index or embedded location index. Existing NextBillion examples include location_index.
        li = s.get("location_index")
        if li is None and "job" in s:
            # sometimes job steps have job id; not enough to map without lookup
            continue
        if li is not None:
            out.append(int(li))
    # Deduplicate consecutive duplicates
    cleaned = []
    for x in out:
        if not cleaned or cleaned[-1] != x:
            cleaned.append(x)
    return cleaned


with tabs[2]:
    st.subheader("Optimize")
    stops = st.session_state.stops_df.copy()
    depots = st.session_state.depots_df.copy()

    if len(stops) < 2:
        st.warning("Add at least 2 stops to optimize.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Create optimization job", width="stretch"):
                audit("button_clicked", widget="create_optimization_job")
                try:
                    payload, locs, loc_labels = build_vrp_payload(
                        stops,
                        depots,
                        mode=mode,
                        driver_count=1 if mode.startswith("Single") else int(driver_count),
                        depot_mode=depot_mode,
                        return_to_start=return_to_start,
                        objective=objective,
                        shift_start=shift_start,
                        shift_end=shift_end,
                    )
                    st.session_state.last_opt_create_payload = payload
                    res = nb_optimize_create(payload)
                    st.session_state.last_opt_create_json = res
                    task_id = res.get("id") or res.get("task_id")
                    st.session_state.last_opt_task_id = task_id
                    st.success(f"Optimization job created: {task_id}")
                    st.json(res)
                except Exception as e:
                    audit("error", where="create_optimization_job", msg=str(e), level="ERROR")
                    st.error(str(e))

        with col2:
            if st.button("Fetch optimization result", width="stretch"):
                audit("button_clicked", widget="fetch_optimization_result")
                try:
                    task_id = st.session_state.get("last_opt_task_id")
                    if not task_id:
                        st.error("No task id yet. Create job first.")
                    else:
                        res = nb_optimize_result(task_id)
                        st.session_state.last_opt_result_json = res
                        st.json(res)
                except Exception as e:
                    audit("error", where="fetch_optimization_result", msg=str(e), level="ERROR")
                    st.error(str(e))

        # Downloads
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.get("last_opt_create_payload"):
                st.download_button(
                    "Download last Optimization Create Payload",
                    data=json.dumps(st.session_state.last_opt_create_payload, ensure_ascii=False),
                    file_name="opt_create_last.json",
                    mime="application/json",
                    width="content",
                )
        with c2:
            if st.session_state.get("last_opt_result_json"):
                st.download_button(
                    "Download last Optimization Result JSON",
                    data=json.dumps(st.session_state.last_opt_result_json, ensure_ascii=False),
                    file_name="opt_result_last.json",
                    mime="application/json",
                    width="content",
                )

# -------------------------
# Results tab (map + fair compare)
# -------------------------
def compute_directions_metrics(d: dict) -> Tuple[float, float]:
    """Return (distance_m, duration_s) for first route."""
    if not d or d.get("status") != "Ok":
        return (0.0, 0.0)
    r0 = (d.get("routes") or [{}])[0]
    return float(r0.get("distance", 0.0)), float(r0.get("duration", 0.0))


def build_baseline_order(stops: pd.DataFrame, depots: pd.DataFrame, return_to_start: bool, depot_mode: str) -> List[Tuple[float, float]]:
    coords = [(float(x["lat"]), float(x["lng"])) for _, x in stops.iterrows()]
    if depot_mode.startswith("Open route"):
        # no explicit depot; baseline is just stops in given order
        if return_to_start:
            # round trip requested but no depot: close at first stop
            coords = coords + [coords[0]]
        return coords

    # Use first depot as baseline depot
    if len(depots) == 0:
        if return_to_start:
            coords = coords + [coords[0]]
        return coords

    depot = (float(depots.iloc[0]["lat"]), float(depots.iloc[0]["lng"]))
    if return_to_start:
        return [depot] + coords + [depot]
    else:
        # open route: start at depot, end at last stop
        return [depot] + coords


def folium_map_for_route(coords: List[Tuple[float, float]], geometry: str = "", labels: Optional[List[str]] = None) -> folium.Map:
    if not coords:
        return folium.Map(location=[0, 0], zoom_start=2)

    m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=12)
    # markers
    for i, (lat, lng) in enumerate(coords):
        label = labels[i] if labels and i < len(labels) else f"{i}"
        folium.Marker(
            [lat, lng],
            tooltip=label,
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 12px; color: white; background: #017dff;
                            border-radius: 999px; width: 24px; height: 24px;
                            display:flex; align-items:center; justify-content:center;
                            border:2px solid white; box-shadow:0 1px 3px rgba(0,0,0,.35);">
                    {i}
                </div>
            """),
        ).add_to(m)

    # polyline from geometry (preferred) or straight lines
    pts = decode_polyline(geometry, precision=6) if geometry else coords
    if pts and len(pts) >= 2:
        pl = folium.PolyLine(pts, weight=5, opacity=0.8)
        pl.add_to(m)
        try:
            PolyLineTextPath(pl, "➤", repeat=True, offset=7, attributes={"font-size": "16", "fill": "black"}).add_to(m)
        except Exception:
            pass
    return m


with tabs[3]:
    st.subheader("Results & Comparison")

    stops = st.session_state.stops_df.copy()
    depots = st.session_state.depots_df.copy()

    if len(stops) < 2:
        st.warning("Need at least 2 stops.")
    else:
        # Baseline
        if st.button("Run baseline Directions (fair compare)", width="stretch"):
            audit("button_clicked", widget="baseline_directions")
            try:
                base_coords = build_baseline_order(stops, depots, return_to_start=return_to_start, depot_mode=depot_mode)
                d0 = nb_directions(base_coords, mode=directions_mode, option=directions_option, avoid=avoid.strip())
                st.session_state.directions_before = d0
                st.success("Baseline directions computed.")
            except Exception as e:
                audit("error", where="baseline_directions", msg=str(e), level="ERROR")
                st.error(str(e))

        # Optimized routes -> directions
        st.write("Compute directions for optimized routes. For performance, you can limit how many driver routes to fetch.")
        max_routes = st.number_input("Max driver routes to fetch directions for", min_value=1, max_value=25, value=5, step=1)

        if st.button("Run Directions for optimized solution", width="stretch"):
            audit("button_clicked", widget="optimized_directions")
            try:
                opt = st.session_state.get("last_opt_result_json")
                if not opt:
                    st.error("Fetch optimization result first (Optimize tab).")
                else:
                    routes = parse_vrp_routes(opt)
                    if not routes:
                        st.error("No routes found in optimization result.")
                    else:
                        # Build map from location indices to coordinates
                        # Rebuild locations mapping from latest payload
                        payload = st.session_state.get("last_opt_create_payload")
                        if not payload:
                            st.error("Missing last optimization payload. Re-run optimization create.")
                        else:
                            loc_arr = payload.get("locations", {}).get("location", [])
                            # loc_arr is [[lng,lat], ...]
                            locs = [(float(latlng[1]), float(latlng[0])) for latlng in loc_arr]

                            dirs_by_vehicle = {}
                            for r in routes[: int(max_routes)]:
                                vehicle_id = r.get("vehicle")
                                idxs = route_steps_to_location_indices(r)
                                if len(idxs) < 2:
                                    continue
                                coords = [locs[i] for i in idxs]
                                d = nb_directions(coords, mode=directions_mode, option=directions_option, avoid=avoid.strip())
                                dirs_by_vehicle[str(vehicle_id)] = {"indices": idxs, "directions": d}
                            st.session_state.directions_after_by_vehicle = dirs_by_vehicle
                            st.success(f"Computed directions for {len(dirs_by_vehicle)} route(s).")
            except Exception as e:
                audit("error", where="optimized_directions", msg=str(e), level="ERROR")
                st.error(str(e))

        # Show metrics + maps
        colL, colR = st.columns(2)

        with colL:
            st.markdown("### Baseline (Before)")
            d0 = st.session_state.get("directions_before")
            if d0:
                dist_m, dur_s = compute_directions_metrics(d0)
                st.metric("Distance", f"{dist_m/1000:.2f} km")
                st.metric("Duration", f"{dur_s/60:.1f} min")
                geom = (d0.get("routes") or [{}])[0].get("geometry", "")
                # coords used for baseline map
                base_coords = build_baseline_order(stops, depots, return_to_start=return_to_start, depot_mode=depot_mode)
                m = folium_map_for_route(base_coords, geometry=geom, labels=["B"] * len(base_coords))
                st_folium(m, height=420, width="100%")
                st.download_button("Download baseline directions JSON", data=json.dumps(d0, ensure_ascii=False), file_name="directions_before.json", mime="application/json", width="content")
            else:
                st.info("Run baseline directions first.")

        with colR:
            st.markdown("### Optimized (After)")
            dirs = st.session_state.get("directions_after_by_vehicle", {})
            if dirs:
                # Aggregate metrics across fetched routes
                total_d_m = 0.0
                total_t_s = 0.0
                rows = []
                for vid, pack in dirs.items():
                    d = pack["directions"]
                    dm, ts = compute_directions_metrics(d)
                    total_d_m += dm
                    total_t_s += ts
                    rows.append({"vehicle": vid, "distance_km": dm/1000, "duration_min": ts/60})
                st.metric("Total distance (fetched routes)", f"{total_d_m/1000:.2f} km")
                st.metric("Total duration (fetched routes)", f"{total_t_s/60:.1f} min")
                st.dataframe(pd.DataFrame(rows).sort_values("vehicle"), width="stretch")

                # Show one route map (select)
                sel = st.selectbox("View route for vehicle", options=[r["vehicle"] for r in rows], index=0)
                chosen = dirs[str(sel)]
                idxs = chosen["indices"]
                payload = st.session_state.get("last_opt_create_payload")
                loc_arr = payload.get("locations", {}).get("location", [])
                locs = [(float(latlng[1]), float(latlng[0])) for latlng in loc_arr]
                coords = [locs[i] for i in idxs]
                geom = (chosen["directions"].get("routes") or [{}])[0].get("geometry", "")
                m = folium_map_for_route(coords, geometry=geom, labels=[str(i) for i in idxs])
                st_folium(m, height=420, width="100%")

                st.download_button(
                    "Download optimized directions bundle (JSON)",
                    data=json.dumps(dirs, ensure_ascii=False),
                    file_name="directions_after_by_vehicle.json",
                    mime="application/json",
                    width="content",
                )
            else:
                st.info("Run 'Directions for optimized solution' to populate per-vehicle routes.")

# -------------------------
# Clustering tab (raw runner)
# -------------------------
with tabs[4]:
    st.subheader("Clustering API (raw)")
    st.write(
        "This tab lets you run Clustering API without guessing request schema. "
        "Paste your JSON request body, submit, then fetch result by ID. "
        "Endpoints: POST /clustering and GET /clustering/result per docs. citeturn1view0"
    )
    example = {
        "description": "My clustering test",
        "routing": {"mode": "car", "option": "fast"},
        "jobs": [
            {"id": 1, "location": [75.7873, 26.9124]},
            {"id": 2, "location": [75.7644, 26.8531]},
        ],
        "clustering": {"max_cluster_radius": 5000},
    }
    body_text = st.text_area("Clustering request body (JSON)", value=json.dumps(example, indent=2), height=220)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Create clustering job", width="stretch"):
            audit("button_clicked", widget="cluster_create")
            try:
                payload = json.loads(body_text)
                st.session_state.last_cluster_payload = payload
                res = nb_cluster_create(payload)
                st.session_state.last_cluster_create_json = res
                st.session_state.last_cluster_task_id = res.get("id")
                st.success(f"Clustering job created: {st.session_state.last_cluster_task_id}")
                st.json(res)
            except Exception as e:
                audit("error", where="cluster_create", msg=str(e), level="ERROR")
                st.error(str(e))

    with colB:
        if st.button("Fetch clustering result", width="stretch"):
            audit("button_clicked", widget="cluster_result")
            try:
                cid = st.session_state.get("last_cluster_task_id")
                if not cid:
                    st.error("No clustering job id yet.")
                else:
                    res = nb_cluster_result(cid)
                    st.session_state.last_cluster_result_json = res
                    st.json(res)
            except Exception as e:
                audit("error", where="cluster_result", msg=str(e), level="ERROR")
                st.error(str(e))

    if st.session_state.get("last_cluster_result_json"):
        st.download_button(
            "Download last Clustering Result JSON",
            data=json.dumps(st.session_state.last_cluster_result_json, ensure_ascii=False),
            file_name="clustering_result_last.json",
            mime="application/json",
            width="content",
        )

# -------------------------
# Logs tab
# -------------------------
with tabs[5]:
    st.subheader("Session logs")
    st.caption("These are session-scoped logs. Backend logs are also visible in Streamlit Community Cloud → Manage app → Cloud logs.")
    last_n = st.number_input("Show last N events", min_value=10, max_value=600, value=80, step=10)
    events = st.session_state.get("audit_events", [])[-int(last_n):]
    st.code("\n".join(json.dumps(e, ensure_ascii=False) for e in events), language="json")
    st.download_button(
        "Download session logs (JSONL)",
        data="\n".join(json.dumps(e, ensure_ascii=False) for e in st.session_state.get("audit_events", [])),
        file_name=f"audit_{st.session_state.get('run_id','')}.jsonl",
        mime="application/json",
        width="content",
    )

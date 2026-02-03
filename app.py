# -*- coding: utf-8 -*-
"""
NextBillion Routing Testbench (Streamlit)
- Max 200 stops
- Editable driver count
- Baseline: greedy assignment + nearest-neighbor sequencing
- Guided clustering (UI-generated payload + optional manual tweak before submit)
- Separate steps with clear "next step" guidance
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

import warnings

# Silence noisy pandas FutureWarning triggered internally by st.data_editor when adding empty/all-NA rows.
warnings.filterwarnings(
    "ignore",
    message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

# Optional map components
try:
    import folium
    from folium.plugins import PolyLineTextPath
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


# =========================
# Config & constants
# =========================

NB_BASE = "https://api.nextbillion.io"
MAX_STOPS = 200
DEFAULT_DRIVERS = 25

st.set_page_config(page_title="NextBillion Routing Testbench", layout="wide")

# =========================
# Utilities
# =========================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace('+00:00','Z')

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in km."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))

def hhmm_to_seconds(hhmm: str) -> Optional[int]:
    """Convert HH:MM to seconds from midnight; returns None if blank/invalid."""
    if hhmm is None:
        return None
    s = str(hhmm).strip()
    if s == "":
        return None
    try:
        parts = s.split(":")
        if len(parts) != 2:
            return None
        h = int(parts[0]); m = int(parts[1])
        if h < 0 or h > 23 or m < 0 or m > 59:
            return None
        return h * 3600 + m * 60
    except Exception:
        return None

def seconds_to_hhmm(sec: Optional[int]) -> str:
    if sec is None:
        return ""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"

def init_state():
    ss = st.session_state
    if "run_id" not in ss:
        ss.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if "seed" not in ss:
        ss.seed = 42
    if "stops_df" not in ss:
        ss.stops_df = pd.DataFrame(columns=["stop_id", "address", "lat", "lng", "tw_start", "tw_end"])
    if "depots_df" not in ss:
        ss.depots_df = pd.DataFrame(
            [{"depot_id": "D1", "name": "Depot 1", "lat": 26.9124, "lng": 75.7873}]
        )
    if "driver_cfg" not in ss:
        ss.driver_cfg = {
            "driver_count": DEFAULT_DRIVERS,
            "shift_start": "09:00",
            "shift_end": "18:00",
            "depot_mode": "single_depot",  # single_depot | multi_depot_random | open_route
            "route_type": "open",          # open | round_trip
        }
    if "routing_cfg" not in ss:
        ss.routing_cfg = {
            "mode": "car",
            "option": "fast",  # for navigation/directions where applicable
            "avoid": "",
        }
    if "objective" not in ss:
        ss.objective = "duration"  # duration | distance
    if "planner_mode" not in ss:
        ss.planner_mode = "Direct VRP"  # Direct VRP | Cluster-first
    if "validated" not in ss:
        ss.validated = False

    # Clustering state
    if "cluster_task_id" not in ss:
        ss.cluster_task_id = ""
    if "cluster_create_payload" not in ss:
        ss.cluster_create_payload = {}
    if "cluster_result" not in ss:
        ss.cluster_result = {}

    # Optimization state
    if "opt_task_id" not in ss:
        ss.opt_task_id = ""
    if "opt_create_payload" not in ss:
        ss.opt_create_payload = {}
    if "opt_result" not in ss:
        ss.opt_result = {}

    # Baseline state
    if "baseline_assignment" not in ss:
        ss.baseline_assignment = {}  # vehicle_id -> list[stop_id]
    if "baseline_routes" not in ss:
        ss.baseline_routes = {}      # vehicle_id -> ordered stop_id list

    # Directions/Navigation state
    if "dir_optimized" not in ss:
        ss.dir_optimized = {}        # vehicle_id -> directions json
    if "dir_baseline" not in ss:
        ss.dir_baseline = {}         # vehicle_id -> directions json

    # Logs
    if "log_rows" not in ss:
        ss.log_rows = []  # list[dict]

def log_event(level: str, msg: str, **fields):
    row = {"ts": now_iso(), "level": level, "msg": msg}
    row.update(fields)
    st.session_state.log_rows.append(row)
    # Keep log size bounded for performance
    if len(st.session_state.log_rows) > 2000:
        st.session_state.log_rows = st.session_state.log_rows[-1500:]

def nb_key() -> str:
    # Prefer Streamlit secrets; fallback to environment variable; fallback to empty
    return st.secrets.get("NEXTBILLION_API_KEY", os.getenv("NEXTBILLION_API_KEY", ""))

def http_get(url: str, params: Optional[dict] = None, timeout: int = 60) -> dict:
    t0 = time.time()
    try:
        r = requests.get(url, params=params, timeout=timeout)
        dt = round((time.time() - t0) * 1000)
        log_event("INFO", "GET", url=url, status=r.status_code, ms=dt)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        dt = round((time.time() - t0) * 1000)
        body = ""
        try:
            body = r.text[:2000]  # type: ignore
        except Exception:
            pass
        log_event("ERROR", "GET failed", url=url, ms=dt, error=str(e), body=body)
        raise

def http_post(url: str, payload: dict, params: Optional[dict] = None, timeout: int = 120) -> dict:
    t0 = time.time()
    r = None
    try:
        r = requests.post(url, params=params, json=payload, timeout=timeout)
        dt = round((time.time() - t0) * 1000)
        log_event("INFO", "POST", url=url, status=r.status_code, ms=dt)

        # If the API returns a structured error, surface it in logs/UI.
        if r.status_code >= 400:
            body = (r.text or "")[:5000]
            log_event("ERROR", "POST failed", url=url, status=r.status_code, ms=dt, body=body)
            raise RuntimeError(f"HTTP {r.status_code} from {url}: {body}")

        return r.json()
    except requests.RequestException as e:
        dt = round((time.time() - t0) * 1000)
        body = ""
        status = None
        try:
            if r is not None:
                status = r.status_code
                body = (r.text or "")[:5000]
        except Exception:
            pass
        log_event("ERROR", "POST exception", url=url, status=status, ms=dt, error=str(e), body=body)
        raise


def next_step_banner():
    ss = st.session_state
    steps = []
    steps.append(("Validated", ss.validated))
    if ss.planner_mode == "Cluster-first":
        steps.append(("Cluster created", bool(ss.cluster_task_id)))
        steps.append(("Cluster fetched", bool(ss.cluster_result)))
    steps.append(("Optimization created", bool(ss.opt_task_id)))
    steps.append(("Optimization fetched", bool(ss.opt_result)))
    steps.append(("Baseline built", bool(ss.baseline_routes)))
    steps.append(("Directions (opt) ready", bool(ss.dir_optimized)))
    steps.append(("Directions (baseline) ready", bool(ss.dir_baseline)))

    # Determine next suggestion
    if not ss.validated:
        suggestion = "Next: Validate Inputs"
    elif ss.planner_mode == "Cluster-first" and not ss.cluster_task_id:
        suggestion = "Next: Create Clustering Job"
    elif ss.planner_mode == "Cluster-first" and ss.cluster_task_id and not ss.cluster_result:
        suggestion = "Next: Fetch Clustering Result"
    elif not ss.opt_task_id:
        suggestion = "Next: Create Optimization Job"
    elif ss.opt_task_id and not ss.opt_result:
        suggestion = "Next: Fetch Optimization Result"
    elif not ss.baseline_routes:
        suggestion = "Next: Build Baseline (Greedy + Nearest-Neighbor)"
    elif not ss.dir_optimized:
        suggestion = "Next: Run Directions for Optimized Routes"
    elif not ss.dir_baseline:
        suggestion = "Next: Run Directions for Baseline Routes"
    else:
        suggestion = "Next: Review & Export"

    cols = st.columns([3, 2])
    with cols[0]:
        badges = "  ".join([f"✅ {name}" if ok else f"⬜ {name}" for name, ok in steps])
        st.markdown(badges)
    with cols[1]:
        st.info(suggestion)

def validate_inputs() -> Tuple[bool, List[str]]:
    ss = st.session_state
    errs = []

    stops = ss.stops_df.copy()
    if len(stops) == 0:
        errs.append("No stops loaded.")
    if len(stops) > MAX_STOPS:
        errs.append(f"Stop limit exceeded: {len(stops)} > {MAX_STOPS}")

    for col in ["lat", "lng"]:
        if col not in stops.columns:
            errs.append(f"Stops missing column '{col}'.")
        else:
            bad = stops[col].isna().sum()
            if bad:
                errs.append(f"Stops have {bad} missing values in '{col}'.")

    if "stop_id" not in stops.columns:
        errs.append("Stops missing 'stop_id'.")
    else:
        if stops["stop_id"].astype(str).str.strip().eq("").any():
            errs.append("Some stops have empty stop_id.")
        if stops["stop_id"].duplicated().any():
            errs.append("Duplicate stop_id values found.")

    # Depots
    depots = ss.depots_df.copy()
    if depots.empty and ss.driver_cfg["depot_mode"] != "open_route":
        errs.append("No depots provided (required unless using open routes).")
    else:
        for col in ["lat", "lng"]:
            if col not in depots.columns:
                errs.append(f"Depots missing column '{col}'.")
            else:
                if depots[col].isna().sum():
                    errs.append(f"Some depots have missing {col}.")

    # Drivers
    try:
        n = int(ss.driver_cfg["driver_count"])
        if n <= 0:
            errs.append("Driver count must be > 0.")
    except Exception:
        errs.append("Driver count must be an integer.")

    # Time windows
    if "tw_start" in stops.columns and "tw_end" in stops.columns:
        for i, row in stops.iterrows():
            s = hhmm_to_seconds(row.get("tw_start", ""))
            e = hhmm_to_seconds(row.get("tw_end", ""))
            if (s is None) != (e is None):
                errs.append(f"Stop {row.get('stop_id')} has only one side of time window filled.")
                break
            if s is not None and e is not None and s >= e:
                errs.append(f"Stop {row.get('stop_id')} has invalid time window (start >= end).")
                break

    ok = len(errs) == 0
    return ok, errs

def generate_stops(seed: int, n: int, preset: str, center: Tuple[float, float], jitter_m: int,
                   tw_pct: int, tw_preset: str) -> pd.DataFrame:
    rng = random.Random(seed)
    lat0, lng0 = center

    def jitter_point(lat, lng, meters):
        # Rough conversion: 1 deg lat ~ 111km, lng scaled by cos(lat)
        dlat = (rng.uniform(-meters, meters) / 1000.0) / 111.0
        dlng = (rng.uniform(-meters, meters) / 1000.0) / (111.0 * max(0.2, math.cos(math.radians(lat))))
        return lat + dlat, lng + dlng

    points = []
    if preset == "Dense":
        for i in range(n):
            lat, lng = jitter_point(lat0, lng0, jitter_m)
            points.append((lat, lng))
    elif preset == "3 pockets":
        pockets = [jitter_point(lat0, lng0, jitter_m * 4) for _ in range(3)]
        for i in range(n):
            base = pockets[i % 3]
            lat, lng = jitter_point(base[0], base[1], jitter_m)
            points.append((lat, lng))
    elif preset == "Mixed":
        for i in range(n):
            m = jitter_m if rng.random() < 0.7 else jitter_m * 8
            lat, lng = jitter_point(lat0, lng0, m)
            points.append((lat, lng))
    else:  # Long tail
        for i in range(n):
            m = jitter_m if rng.random() < 0.8 else jitter_m * 15
            lat, lng = jitter_point(lat0, lng0, m)
            points.append((lat, lng))

    rows = []
    for i, (lat, lng) in enumerate(points, start=1):
        tw_start = ""
        tw_end = ""
        if rng.randint(1, 100) <= tw_pct:
            if tw_preset == "Loose":
                tw_start, tw_end = "09:00", "18:00"
            elif tw_preset == "Waves":
                wave = (i % 3)
                if wave == 0:
                    tw_start, tw_end = "09:00", "12:00"
                elif wave == 1:
                    tw_start, tw_end = "11:00", "14:00"
                else:
                    tw_start, tw_end = "13:00", "18:00"
            else:  # Tight
                # Tight random window of 60-120 minutes between 09:00 and 18:00
                start = rng.randint(9 * 60, 16 * 60)
                dur = rng.randint(60, 120)
                end = min(start + dur, 18 * 60)
                tw_start = f"{start//60:02d}:{start%60:02d}"
                tw_end = f"{end//60:02d}:{end%60:02d}"

        rows.append({
            "stop_id": f"S{i}",
            "address": "",
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "tw_start": tw_start,
            "tw_end": tw_end,
        })

    return pd.DataFrame(rows)

def assign_depots_to_vehicles(dep_df: pd.DataFrame, driver_count: int, depot_mode: str, seed: int) -> List[Optional[Tuple[float, float]]]:
    rng = random.Random(seed)
    depots = [(float(r["lat"]), float(r["lng"])) for _, r in dep_df.iterrows()] if not dep_df.empty else []
    starts: List[Optional[Tuple[float, float]]] = []
    if depot_mode == "open_route":
        return [None] * driver_count
    if not depots:
        return [None] * driver_count
    if depot_mode == "single_depot":
        return [depots[0]] * driver_count
    # multi_depot_random
    for _ in range(driver_count):
        starts.append(rng.choice(depots))
    return starts

def greedy_assign_stops_to_vehicles(
    stops_df: pd.DataFrame,
    vehicle_starts: List[Optional[Tuple[float, float]]],
    seed: int,
) -> Dict[str, List[str]]:
    """
    Greedy assignment: iterate stops in a seeded order and assign each stop to the vehicle
    that yields the smallest incremental straight-line distance from that vehicle's last point.
    Then we sequence within each vehicle using nearest neighbor.
    """
    rng = random.Random(seed)
    stops = stops_df[["stop_id", "lat", "lng"]].copy()
    stops["lat"] = stops["lat"].astype(float)
    stops["lng"] = stops["lng"].astype(float)

    stop_rows = stops.to_dict("records")
    rng.shuffle(stop_rows)

    vehicle_last = []
    for s in vehicle_starts:
        if s is None:
            vehicle_last.append(None)
        else:
            vehicle_last.append(s)

    assigned: Dict[str, List[str]] = {f"V{i+1}": [] for i in range(len(vehicle_starts))}
    last_point: Dict[str, Optional[Tuple[float, float]]] = {f"V{i+1}": vehicle_last[i] for i in range(len(vehicle_starts))}

    for row in stop_rows:
        sid = str(row["stop_id"])
        p = (float(row["lat"]), float(row["lng"]))
        best_v = None
        best_cost = None

        for i in range(len(vehicle_starts)):
            vid = f"V{i+1}"
            lp = last_point[vid]
            if lp is None:
                # if open route and no last point, cost is 0 (vehicle can start here)
                cost = 0.0
            else:
                cost = haversine_km(lp, p)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_v = vid

        assigned[best_v].append(sid)  # type: ignore
        # update last point for that vehicle
        last_point[best_v] = p  # type: ignore

    return assigned

def nearest_neighbor_order(points: List[Tuple[str, Tuple[float, float]]], start: Optional[Tuple[float, float]]) -> List[str]:
    """Nearest-neighbor sequencing for a list of (stop_id, (lat,lng))."""
    remaining = points.copy()
    ordered: List[str] = []
    cur = start
    while remaining:
        if cur is None:
            # open route: pick any first (deterministic: smallest id)
            remaining.sort(key=lambda x: x[0])
            sid, p = remaining.pop(0)
            ordered.append(sid)
            cur = p
            continue
        # pick nearest
        best_idx = 0
        best_d = None
        for i, (sid, p) in enumerate(remaining):
            d = haversine_km(cur, p)
            if best_d is None or d < best_d:
                best_d = d
                best_idx = i
        sid, p = remaining.pop(best_idx)
        ordered.append(sid)
        cur = p
    return ordered

def build_baseline_routes(stops_df: pd.DataFrame, vehicle_starts: List[Optional[Tuple[float, float]]], seed: int) -> Dict[str, List[str]]:
    assigned = greedy_assign_stops_to_vehicles(stops_df, vehicle_starts, seed)
    pts_by_id = {str(r["stop_id"]): (float(r["lat"]), float(r["lng"])) for _, r in stops_df.iterrows()}
    routes: Dict[str, List[str]] = {}
    for i, start in enumerate(vehicle_starts):
        vid = f"V{i+1}"
        pts = [(sid, pts_by_id[sid]) for sid in assigned.get(vid, []) if sid in pts_by_id]
        routes[vid] = nearest_neighbor_order(pts, start=start)
    return routes

def build_opt_payload(stops_df: pd.DataFrame, depots_df: pd.DataFrame, cfg: dict, objective: str, seed: int) -> dict:
    """
    Build a Route Optimization payload for:
      POST https://api.nextbillion.io/optimization/v2?key=...
      GET  https://api.nextbillion.io/optimization/v2/result?id=...&key=...

    Key rules from docs:
      - `locations` is an ordered list of coordinate strings ("lat,lng"); `start_index`/`end_index` refer to indices in that list.
      - Every vehicle must have a start location configured (via `start_index` or a depot workflow).
      - For round trip: set `start_index` == `end_index`.
    """
    driver_count = int(cfg["driver_count"])
    depot_mode = cfg["depot_mode"]
    route_type = cfg["route_type"]

    rnd = random.Random(int(seed))

    # ---------- Locations array ----------
    # Convention: depots first, then stops
    locations: List[str] = []
    depot_coords: List[Tuple[float, float]] = []

    # If depot mode expects depots but none are provided, create a synthetic depot.
    # This prevents 400 errors caused by missing vehicle start locations.
    if depot_mode != "open_route" and depots_df.empty:
        if not stops_df.empty:
            lat0 = float(stops_df["lat"].astype(float).mean())
            lng0 = float(stops_df["lng"].astype(float).mean())
        else:
            lat0 = float(cfg.get("default_depot_lat", 0.0) or 0.0)
            lng0 = float(cfg.get("default_depot_lng", 0.0) or 0.0)
        depot_coords.append((lat0, lng0))
        locations.append(f"{lat0},{lng0}")
    else:
        if depot_mode != "open_route" and not depots_df.empty:
            for _, r in depots_df.iterrows():
                lat0, lng0 = float(r["lat"]), float(r["lng"])
                depot_coords.append((lat0, lng0))
                locations.append(f"{lat0},{lng0}")

    stop_index_offset = len(locations)

    for _, r in stops_df.iterrows():
        locations.append(f"{float(r['lat'])},{float(r['lng'])}")

    # ---------- Jobs ----------
    jobs = []
    for idx, r in stops_df.reset_index(drop=True).iterrows():
        job = {
            "id": int(idx + 1),  # jobs ids should be unique positive integers in most examples
            "location_index": stop_index_offset + idx,
        }
        tws = hhmm_to_seconds(r.get("tw_start", ""))
        twe = hhmm_to_seconds(r.get("tw_end", ""))
        if tws is not None and twe is not None and twe >= tws:
            job["time_windows"] = [{"start": tws, "end": twe}]
        jobs.append(job)

    # ---------- Vehicles ----------
    shift_start = hhmm_to_seconds(cfg.get("shift_start", "09:00")) or 9 * 3600
    shift_end = hhmm_to_seconds(cfg.get("shift_end", "18:00")) or 18 * 3600

    vehicles = []
    for i in range(driver_count):
        v = {
            "id": int(i + 1),
            "time_window": {"start": shift_start, "end": shift_end},
        }

        # Ensure every vehicle has a start_index.
        if depot_mode == "open_route":
            # Open route: choose a start among stops (or fall back to index 0 if no stops)
            if len(stops_df) > 0:
                start_idx = stop_index_offset + (i % len(stops_df))
            else:
                start_idx = 0
            v["start_index"] = int(start_idx)
            if route_type == "round_trip":
                v["end_index"] = int(start_idx)
        else:
            # Depot-based: assign a depot index
            if depot_coords:
                depot_idx = i % len(depot_coords)
                v["start_index"] = int(depot_idx)
                if route_type == "round_trip":
                    v["end_index"] = int(depot_idx)
            else:
                # Extremely defensive fallback (should not happen due to synthetic depot creation)
                v["start_index"] = 0
                if route_type == "round_trip":
                    v["end_index"] = 0

        vehicles.append(v)

    options: Dict[str, Any] = {"objective": {"travel_cost": objective}}
    payload = {
        "locations": locations,
        "jobs": jobs,
        "vehicles": vehicles,
        "options": options,
    }
    return payload

def parse_opt_routes(opt_result: dict) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract per-vehicle route geometry as list of coordinates, if available.
    If not available, extract ordered job indices and reconstruct using locations list if present.
    This function is defensive because response schema may vary by configuration.
    """
    routes: Dict[str, List[Tuple[float, float]]] = {}
    # Try common structure: result -> routes list
    res = opt_result.get("result") or opt_result
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            return routes

    if isinstance(res, dict):
        route_list = res.get("routes") or res.get("route") or res.get("solutions")
        if isinstance(route_list, list):
            for r in route_list:
                vid = str(r.get("vehicle_id") or r.get("vehicle") or r.get("id") or "")
                steps = r.get("steps") or r.get("activities") or r.get("stops")
                coords: List[Tuple[float, float]] = []
                if isinstance(steps, list):
                    for s in steps:
                        loc = s.get("location") or s.get("lat_lng") or s.get("coordinate")
                        if isinstance(loc, (list, tuple)) and len(loc) == 2:
                            coords.append((float(loc[0]), float(loc[1])))
                        elif isinstance(loc, str) and "," in loc:
                            a, b = loc.split(",", 1)
                            coords.append((float(a), float(b)))
                if vid:
                    routes[vid] = coords
    return routes

def build_waypoint_string(coords: List[Tuple[float, float]]) -> str:
    return "|".join([f"{lat},{lng}" for (lat, lng) in coords])

def call_navigation_route(coords: List[Tuple[float, float]], routing_cfg: dict) -> dict:
    """
    Uses Navigation API (Fast) because it supports POST and large waypoint counts; docs recommend POST for >50 (up to 200). 
    """
    if len(coords) < 2:
        return {"status": "Error", "message": "Need at least origin and destination."}

    origin = f"{coords[0][0]},{coords[0][1]}"
    destination = f"{coords[-1][0]},{coords[-1][1]}"
    waypoints = coords[1:-1]
    key = nb_key()
    if not key:
        raise RuntimeError("Missing NextBillion API key. Set NEXTBILLION_API_KEY or Streamlit secrets.")

    params = {"key": key}
    mode = routing_cfg.get("mode", "car")
    avoid = routing_cfg.get("avoid", "").strip()
    option = routing_cfg.get("option", "fast").strip().lower()

    # Use /navigation/json for fast; flexible uses /navigation?option=flexible. We'll keep fast by default.
    if option == "flexible":
        url = f"{NB_BASE}/navigation"
        params["option"] = "flexible"
    else:
        url = f"{NB_BASE}/navigation/json"

    payload: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
    }
    if waypoints:
        payload["waypoints"] = build_waypoint_string(waypoints)
    if avoid:
        payload["avoid"] = avoid

    return http_post(url, payload=payload, params=params, timeout=120)

def compute_kpis_from_nav(nav_json: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract distance (meters) and duration (seconds) from Navigation/Directions-like response.
    """
    try:
        routes = nav_json.get("routes", [])
        if routes:
            r0 = routes[0]
            # Common keys in routing APIs
            dist = r0.get("distance") or r0.get("distance_meters")
            dur = r0.get("duration") or r0.get("duration_seconds")
            if isinstance(dist, (int, float)) and isinstance(dur, (int, float)):
                return float(dist), float(dur)
    except Exception:
        pass
    return None, None

def to_map(stops_df: pd.DataFrame, depots_df: pd.DataFrame, coords: List[Tuple[float, float]], title: str):
    if folium is None or st_folium is None:
        st.warning("Map libraries not installed. Install folium + streamlit-folium to view maps.")
        return

    if coords:
        center = coords[0]
    elif not depots_df.empty:
        center = (float(depots_df.iloc[0]["lat"]), float(depots_df.iloc[0]["lng"]))
    elif not stops_df.empty:
        center = (float(stops_df.iloc[0]["lat"]), float(stops_df.iloc[0]["lng"]))
    else:
        center = (0.0, 0.0)

    m = folium.Map(location=center, zoom_start=12, control_scale=True)
    # Depots
    for _, d in depots_df.iterrows():
        folium.Marker(
            (float(d["lat"]), float(d["lng"])),
            tooltip=f"Depot {d.get('depot_id','')}",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(m)

    # Stops
    for _, s in stops_df.iterrows():
        folium.CircleMarker(
            (float(s["lat"]), float(s["lng"])),
            radius=4,
            tooltip=str(s.get("stop_id","")),
            fill=True,
        ).add_to(m)

    # Route polyline
    if coords:
        folium.PolyLine(coords, weight=4).add_to(m)

    st.markdown(f"**{title}**")
    st_folium(m, width="stretch", height=520)  # streamlit-folium supports width like this in recent versions


# =========================
# Sidebar controls
# =========================

def sidebar_controls():
    ss = st.session_state
    st.sidebar.header("Run Control Panel")

    with st.sidebar.expander("Run identity", expanded=True):
        ss.run_id = st.text_input("Run ID", value=ss.run_id)
        c1, c2 = st.columns([1, 1])
        with c1:
            ss.seed = st.number_input("Seed", value=int(ss.seed), step=1)
        with c2:
            if st.button("Random seed", use_container_width=True):
                ss.seed = random.randint(1, 1_000_000)
                log_event("INFO", "Seed randomized", seed=int(ss.seed))

    with st.sidebar.expander("Planner mode", expanded=True):
        ss.planner_mode = st.radio("Mode", ["Direct VRP", "Cluster-first"], index=0 if ss.planner_mode == "Direct VRP" else 1)
        st.caption(f"Stops loaded: {len(ss.stops_df)} / {MAX_STOPS}")

    with st.sidebar.expander("Objectives & routing", expanded=True):
        ss.objective = st.selectbox("Optimization objective", ["duration", "distance"], index=0 if ss.objective == "duration" else 1)

        # Driver & depot behavior
        ss.driver_cfg["driver_count"] = int(st.number_input("Driver count", min_value=1, max_value=200, value=int(ss.driver_cfg["driver_count"]), step=1))
        ss.driver_cfg["shift_start"] = st.text_input("Shift start (HH:MM)", value=ss.driver_cfg["shift_start"])
        ss.driver_cfg["shift_end"] = st.text_input("Shift end (HH:MM)", value=ss.driver_cfg["shift_end"])
        ss.driver_cfg["depot_mode"] = st.selectbox("Depot mode", ["single_depot", "multi_depot_random", "open_route"],
                                                  index=["single_depot","multi_depot_random","open_route"].index(ss.driver_cfg["depot_mode"]))
        ss.driver_cfg["route_type"] = st.selectbox("Route type", ["open", "round_trip"], index=0 if ss.driver_cfg["route_type"] == "open" else 1)

        ss.routing_cfg["mode"] = st.selectbox("Directions/Navi mode", ["car", "truck"], index=0 if ss.routing_cfg["mode"] == "car" else 1)
        ss.routing_cfg["option"] = st.selectbox("Navigation option", ["fast", "flexible"], index=0 if ss.routing_cfg["option"] == "fast" else 1)
        ss.routing_cfg["avoid"] = st.text_input("Avoid (e.g., highway|toll)", value=ss.routing_cfg.get("avoid", ""))

    with st.sidebar.expander("Actions", expanded=True):
        if st.button("Validate Inputs", type="primary", use_container_width=True):
            ok, errs = validate_inputs()
            ss.validated = ok
            if ok:
                st.success("Validated ✅")
                log_event("INFO", "Validation passed", stops=len(ss.stops_df), drivers=int(ss.driver_cfg["driver_count"]))
            else:
                st.error("Validation failed")
                for e in errs:
                    st.warning(e)
                log_event("ERROR", "Validation failed", errors=errs)

        if ss.planner_mode == "Cluster-first":
            if st.button("Create Clustering Job", use_container_width=True, disabled=not ss.validated):
                create_clustering_job()

            if st.button("Fetch Clustering Result", use_container_width=True, disabled=not bool(ss.cluster_task_id)):
                fetch_clustering_result()

        if st.button("Create Optimization Job", use_container_width=True, disabled=not ss.validated):
            create_optimization_job()

        if st.button("Fetch Optimization Result", use_container_width=True, disabled=not bool(ss.opt_task_id)):
            fetch_optimization_result()

        if st.button("Build Baseline (Greedy + NN)", use_container_width=True, disabled=not ss.validated):
            build_baseline()

        if st.button("Run Directions for Optimized", use_container_width=True, disabled=not bool(ss.opt_result)):
            run_directions_for_optimized()

        if st.button("Run Directions for Baseline", use_container_width=True, disabled=not bool(ss.baseline_routes)):
            run_directions_for_baseline()

        if st.button("Export Run Pack", use_container_width=True):
            export_run_pack()

# =========================
# Clustering: guided + editable payload
# =========================

def default_cluster_payload(stops_df: pd.DataFrame, depots_df: pd.DataFrame, k: int, max_radius_km: float, max_jobs: int, routing_cfg: dict) -> dict:
    """
    Because the docs don't render the schema directly in HTML here,
    we generate a best-effort payload and allow the user to tweak it before submission.
    """
    # Locations: depots then stops, similar to optimization pattern
    locations: List[str] = []
    if not depots_df.empty:
        for _, r in depots_df.iterrows():
            locations.append(f"{float(r['lat'])},{float(r['lng'])}")
    stop_offset = len(locations)
    for _, r in stops_df.iterrows():
        locations.append(f"{float(r['lat'])},{float(r['lng'])}")

    jobs = []
    for idx, r in stops_df.reset_index(drop=True).iterrows():
        jobs.append({
            "id": int(idx + 1),
            "location_index": stop_offset + idx,
        })

    payload: Dict[str, Any] = {
        "description": f"Clusters_{k}_{now_iso()}",
        "locations": locations,
        "jobs": jobs,
        "routing": {
            "mode": routing_cfg.get("mode", "car"),
            "option": routing_cfg.get("option", "fast"),
        },
        "k": int(k),
    }
    if max_radius_km > 0:
        payload["max_cluster_radius"] = int(max_radius_km * 1000)  # meters (best-effort)
    if max_jobs > 0:
        payload["max_jobs_in_cluster"] = int(max_jobs)
    return payload

def create_clustering_job():
    ss = st.session_state
    key = nb_key()
    if not key:
        st.error("Missing NextBillion API key. Set NEXTBILLION_API_KEY (or Streamlit secrets).")
        return

    # Pull guided settings from session (stored in ss.cluster_ui dict if set)
    cluster_ui = ss.get("cluster_ui", {})
    k = int(cluster_ui.get("k", ss.driver_cfg["driver_count"]))
    max_radius_km = float(cluster_ui.get("max_radius_km", 0))
    max_jobs = int(cluster_ui.get("max_jobs", 0))

    payload = default_cluster_payload(ss.stops_df, ss.depots_df, k, max_radius_km, max_jobs, ss.routing_cfg)
    ss.cluster_create_payload = payload

    url = f"{NB_BASE}/clustering"
    params = {"key": key}
    try:
        resp = http_post(url, payload=payload, params=params, timeout=120)
        ss.cluster_task_id = resp.get("id", "") or resp.get("request_id", "")
        st.success(f"Clustering job created: {ss.cluster_task_id}")
        log_event("INFO", "Clustering job created", cluster_id=ss.cluster_task_id)
    except Exception as e:
        st.exception(e)

def fetch_clustering_result():
    ss = st.session_state
    key = nb_key()
    if not key:
        st.error("Missing NextBillion API key. Set NEXTBILLION_API_KEY (or Streamlit secrets).")
        return
    if not ss.cluster_task_id:
        st.warning("No clustering task ID.")
        return

    url = f"{NB_BASE}/clustering/result"
    params = {"id": ss.cluster_task_id, "key": key}
    try:
        resp = http_get(url, params=params, timeout=120)
        ss.cluster_result = resp
        st.success("Clustering result fetched.")
        log_event("INFO", "Clustering result fetched", cluster_id=ss.cluster_task_id)
    except Exception as e:
        st.exception(e)

# =========================
# Optimization
# =========================

def create_optimization_job():
    ss = st.session_state
    key = nb_key()
    if not key:
        st.error("Missing NextBillion API key. Set NEXTBILLION_API_KEY (or Streamlit secrets).")
        return

    payload = build_opt_payload(ss.stops_df, ss.depots_df, ss.driver_cfg, ss.objective, int(ss.seed))
    ss.opt_create_payload = payload

    url = f"{NB_BASE}/optimization/v2"
    params = {"key": key}
    try:
        resp = http_post(url, payload=payload, params=params, timeout=120)
        ss.opt_task_id = resp.get("id", "") or resp.get("request_id", "")
        st.success(f"Optimization job created: {ss.opt_task_id}")
        log_event("INFO", "Optimization job created", opt_id=ss.opt_task_id)
    except Exception as e:
        st.exception(e)

def fetch_optimization_result():
    ss = st.session_state
    key = nb_key()
    if not key:
        st.error("Missing NextBillion API key. Set NEXTBILLION_API_KEY (or Streamlit secrets).")
        return
    if not ss.opt_task_id:
        st.warning("No optimization task ID.")
        return

    url = f"{NB_BASE}/optimization/v2/result"
    params = {"id": ss.opt_task_id, "key": key}
    try:
        resp = http_get(url, params=params, timeout=180)
        ss.opt_result = resp
        st.success("Optimization result fetched.")
        log_event("INFO", "Optimization result fetched", opt_id=ss.opt_task_id)
    except Exception as e:
        st.exception(e)

# =========================
# Baseline
# =========================

def build_baseline():
    ss = st.session_state
    starts = assign_depots_to_vehicles(ss.depots_df, int(ss.driver_cfg["driver_count"]), ss.driver_cfg["depot_mode"], int(ss.seed))
    routes = build_baseline_routes(ss.stops_df, starts, int(ss.seed))
    ss.baseline_routes = routes
    # For transparency, keep assignment too
    ss.baseline_assignment = greedy_assign_stops_to_vehicles(ss.stops_df, starts, int(ss.seed))
    st.success("Baseline built (Greedy assignment + nearest-neighbor sequencing).")
    log_event("INFO", "Baseline built", vehicles=len(routes))

# =========================
# Directions/Navigation per route
# =========================

def route_coords_for_vehicle(vehicle_id: str, ordered_stop_ids: List[str], starts: Optional[Tuple[float, float]], route_type: str) -> List[Tuple[float, float]]:
    pts = {str(r["stop_id"]): (float(r["lat"]), float(r["lng"])) for _, r in st.session_state.stops_df.iterrows()}
    coords: List[Tuple[float, float]] = []
    if starts is not None:
        coords.append(starts)
    coords.extend([pts[sid] for sid in ordered_stop_ids if sid in pts])
    if route_type == "round_trip" and starts is not None:
        coords.append(starts)
    return coords

def run_directions_for_baseline():
    ss = st.session_state
    starts = assign_depots_to_vehicles(ss.depots_df, int(ss.driver_cfg["driver_count"]), ss.driver_cfg["depot_mode"], int(ss.seed))
    out = {}
    for i in range(int(ss.driver_cfg["driver_count"])):
        vid = f"V{i+1}"
        route = ss.baseline_routes.get(vid, [])
        coords = route_coords_for_vehicle(vid, route, starts[i], ss.driver_cfg["route_type"])
        if len(coords) < 2:
            continue
        try:
            nav = call_navigation_route(coords, ss.routing_cfg)
            out[vid] = nav
        except Exception as e:
            out[vid] = {"status": "Error", "message": str(e)}
    ss.dir_baseline = out
    st.success("Directions/Navigation computed for baseline routes.")
    log_event("INFO", "Baseline navigation done", vehicles=len(out))

def run_directions_for_optimized():
    """
    Defensive approach: if the optimizer result contains explicit waypoint coordinates per vehicle, use them.
    Otherwise, we try to reconstruct ordered job location indices from the response if present.
    """
    ss = st.session_state
    if not ss.opt_result:
        st.warning("No optimization result.")
        return
    out = {}

    # Best-effort parse: we look for route steps that contain location indices, then map to payload locations.
    payload_locations = ss.opt_create_payload.get("locations", [])
    loc_coords: List[Tuple[float, float]] = []
    for s in payload_locations:
        try:
            a, b = str(s).split(",", 1)
            loc_coords.append((float(a), float(b)))
        except Exception:
            loc_coords.append((0.0, 0.0))

    res = ss.opt_result.get("result") or ss.opt_result
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            res = {}

    routes = []
    if isinstance(res, dict) and isinstance(res.get("routes"), list):
        routes = res["routes"]
    elif isinstance(res, dict) and isinstance(res.get("solutions"), list):
        routes = res["solutions"]

    for r in routes:
        vid = str(r.get("vehicle_id") or r.get("vehicle") or r.get("id") or "")
        steps = r.get("steps") or r.get("activities") or r.get("stops") or []
        coords: List[Tuple[float, float]] = []
        for s in steps:
            # Prefer explicit coordinates
            loc = s.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) == 2:
                coords.append((float(loc[0]), float(loc[1])))
                continue
            if isinstance(loc, str) and "," in loc:
                a, b = loc.split(",", 1)
                coords.append((float(a), float(b)))
                continue
            # Try location_index
            li = s.get("location_index")
            if isinstance(li, int) and 0 <= li < len(loc_coords):
                coords.append(loc_coords[li])

        if len(coords) >= 2 and vid:
            try:
                nav = call_navigation_route(coords, ss.routing_cfg)
                out[vid] = nav
            except Exception as e:
                out[vid] = {"status": "Error", "message": str(e)}

    ss.dir_optimized = out
    st.success("Directions/Navigation computed for optimized routes (best-effort).")
    log_event("INFO", "Optimized navigation done", vehicles=len(out))

# =========================
# Export
# =========================

def export_run_pack():
    ss = st.session_state
    pack = {
        "run_id": ss.run_id,
        "seed": int(ss.seed),
        "planner_mode": ss.planner_mode,
        "objective": ss.objective,
        "driver_cfg": ss.driver_cfg,
        "routing_cfg": ss.routing_cfg,
        "created_at": now_iso(),
        "stops": ss.stops_df.to_dict("records"),
        "depots": ss.depots_df.to_dict("records"),
        "cluster_task_id": ss.cluster_task_id,
        "cluster_create_payload": ss.cluster_create_payload,
        "cluster_result": ss.cluster_result,
        "opt_task_id": ss.opt_task_id,
        "opt_create_payload": ss.opt_create_payload,
        "opt_result": ss.opt_result,
        "baseline_assignment": ss.baseline_assignment,
        "baseline_routes": ss.baseline_routes,
        "dir_baseline": ss.dir_baseline,
        "dir_optimized": ss.dir_optimized,
        "logs": ss.log_rows[-500:],  # last 500
    }
    data = safe_json_dumps(pack).encode("utf-8")
    st.download_button(
        "Download Run Pack (JSON)",
        data=data,
        file_name=f"{ss.run_id}_run_pack.json",
        mime="application/json",
        width="content",
    )
    log_event("INFO", "Run pack prepared", bytes=len(data))

# =========================
# Pages
# =========================

def page_stops():
    ss = st.session_state
    st.subheader("Stops")

    with st.expander("Generate stops (seeded)", expanded=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            n = st.number_input("Stop count", min_value=1, max_value=MAX_STOPS, value=min(50, MAX_STOPS), step=1)
        with c2:
            preset = st.selectbox("Preset", ["Dense", "3 pockets", "Mixed", "Long tail"])
        with c3:
            jitter_m = st.number_input("Jitter (meters)", min_value=50, max_value=5000, value=800, step=50)
        with c4:
            tw_pct = st.slider("% with time windows", min_value=0, max_value=100, value=40, step=5)

        c5, c6 = st.columns([1, 1])
        with c5:
            tw_preset = st.selectbox("Time window preset", ["Loose", "Waves", "Tight"])
        with c6:
            center_lat = st.number_input("Center lat", value=float(ss.depots_df.iloc[0]["lat"]) if not ss.depots_df.empty else 0.0, format="%.6f")
            center_lng = st.number_input("Center lng", value=float(ss.depots_df.iloc[0]["lng"]) if not ss.depots_df.empty else 0.0, format="%.6f")

        if st.button("Generate & replace stops", width="stretch"):
            df = generate_stops(int(ss.seed), int(n), preset, (float(center_lat), float(center_lng)), int(jitter_m), int(tw_pct), tw_preset)
            ss.stops_df = df
            ss.validated = False
            ss.baseline_routes = {}
            ss.opt_task_id = ""
            ss.opt_result = {}
            ss.dir_baseline = {}
            ss.dir_optimized = {}
            st.success(f"Generated {len(df)} stops.")
            log_event("INFO", "Stops generated", count=len(df), preset=preset)

    st.caption("Edit stops below. Required: stop_id, lat, lng. Optional: tw_start/tw_end (HH:MM).")
    ss.stops_df = st.data_editor(
        ss.stops_df,
        num_rows="dynamic",
        width="stretch",
        key="stops_editor",
    )

    with st.expander("Import stops CSV", expanded=False):
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                df = pd.read_csv(up)
                # Normalize columns
                col_map = {c.lower(): c for c in df.columns}
                # Accept common variants
                def pick(*names):
                    for n in names:
                        if n in col_map:
                            return col_map[n]
                    return None

                stop_id_col = pick("stop_id", "id", "stopid")
                lat_col = pick("lat", "latitude")
                lng_col = pick("lng", "lon", "longitude")
                addr_col = pick("address")
                tws_col = pick("tw_start", "time_window_start", "window_start")
                twe_col = pick("tw_end", "time_window_end", "window_end")

                out = pd.DataFrame()
                out["stop_id"] = df[stop_id_col] if stop_id_col else [f"S{i+1}" for i in range(len(df))]
                out["address"] = df[addr_col] if addr_col else ""
                out["lat"] = df[lat_col] if lat_col else None
                out["lng"] = df[lng_col] if lng_col else None
                out["tw_start"] = df[tws_col] if tws_col else ""
                out["tw_end"] = df[twe_col] if twe_col else ""
                ss.stops_df = out
                ss.validated = False
                st.success(f"Imported {len(out)} stops.")
                log_event("INFO", "Stops imported", count=len(out))
            except Exception as e:
                st.exception(e)

def page_drivers_depots():
    ss = st.session_state
    st.subheader("Drivers & Depots")

    st.markdown("### Depots")
    ss.depots_df = st.data_editor(
        ss.depots_df,
        num_rows="dynamic",
        width="stretch",
        key="depots_editor",
    )

    st.markdown("### Driver settings")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        ss.driver_cfg["driver_count"] = int(st.number_input("Driver count", min_value=1, max_value=200, value=int(ss.driver_cfg["driver_count"]), step=1, key="driver_count_main"))
    with c2:
        ss.driver_cfg["shift_start"] = st.text_input("Shift start (HH:MM)", value=ss.driver_cfg["shift_start"], key="shift_start_main")
    with c3:
        ss.driver_cfg["shift_end"] = st.text_input("Shift end (HH:MM)", value=ss.driver_cfg["shift_end"], key="shift_end_main")
    with c4:
        ss.driver_cfg["depot_mode"] = st.selectbox("Depot mode", ["single_depot", "multi_depot_random", "open_route"],
                                                  index=["single_depot","multi_depot_random","open_route"].index(ss.driver_cfg["depot_mode"]),
                                                  key="depot_mode_main")

    ss.driver_cfg["route_type"] = st.selectbox("Route type", ["open", "round_trip"], index=0 if ss.driver_cfg["route_type"] == "open" else 1, key="route_type_main")

    st.info("Tip: For round trips, vehicles return to their start depot. For open routes, vehicles end at the last stop.")

def page_clustering():
    ss = st.session_state
    st.subheader("Clustering (guided)")

    if ss.planner_mode != "Cluster-first":
        st.warning("Switch Planner Mode to 'Cluster-first' in the sidebar to use clustering.")
        return

    st.markdown("### Clustering settings")
    if "cluster_ui" not in ss:
        ss.cluster_ui = {"k": int(ss.driver_cfg["driver_count"]), "max_radius_km": 0.0, "max_jobs": 0}

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ss.cluster_ui["k"] = int(st.number_input("k clusters", min_value=1, max_value=200, value=int(ss.cluster_ui["k"]), step=1))
    with c2:
        ss.cluster_ui["max_radius_km"] = float(st.number_input("Max cluster radius (km) [optional]", min_value=0.0, max_value=200.0, value=float(ss.cluster_ui["max_radius_km"]), step=1.0))
    with c3:
        ss.cluster_ui["max_jobs"] = int(st.number_input("Max jobs per cluster [optional]", min_value=0, max_value=MAX_STOPS, value=int(ss.cluster_ui["max_jobs"]), step=1))

    st.markdown("### Generated clustering payload (editable)")
    payload = default_cluster_payload(ss.stops_df, ss.depots_df, int(ss.cluster_ui["k"]), float(ss.cluster_ui["max_radius_km"]), int(ss.cluster_ui["max_jobs"]), ss.routing_cfg)
    txt = st.text_area("Clustering JSON", value=safe_json_dumps(payload), height=260)
    try:
        edited = json.loads(txt)
        ss.cluster_create_payload = edited
        st.success("Clustering payload JSON is valid.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return

    c4, c5 = st.columns([1, 1])
    with c4:
        if st.button("Create clustering job (using edited payload)", width="stretch", disabled=not ss.validated):
            key = nb_key()
            if not key:
                st.error("Missing API key.")
            else:
                url = f"{NB_BASE}/clustering"
                params = {"key": key}
                try:
                    resp = http_post(url, payload=ss.cluster_create_payload, params=params, timeout=120)
                    ss.cluster_task_id = resp.get("id", "") or resp.get("request_id", "")
                    st.success(f"Clustering job created: {ss.cluster_task_id}")
                except Exception as ex:
                    st.exception(ex)
    with c5:
        if st.button("Fetch clustering result", width="stretch", disabled=not bool(ss.cluster_task_id)):
            fetch_clustering_result()

    if ss.cluster_result:
        st.markdown("### Clustering result (raw)")
        st.code(safe_json_dumps(ss.cluster_result), language="json")

def page_optimization():
    ss = st.session_state
    st.subheader("Optimization")

    st.markdown("### Optimization payload (generated)")
    payload = build_opt_payload(ss.stops_df, ss.depots_df, ss.driver_cfg, ss.objective, int(ss.seed))
    st.code(safe_json_dumps(payload), language="json")
    ss.opt_create_payload = payload

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Create optimization job", width="stretch", disabled=not ss.validated):
            create_optimization_job()
    with c2:
        if st.button("Fetch optimization result", width="stretch", disabled=not bool(ss.opt_task_id)):
            fetch_optimization_result()

    if ss.opt_task_id:
        st.info(f"Current Optimization Task ID: {ss.opt_task_id}")

    if ss.opt_result:
        st.markdown("### Optimization result (raw)")
        st.code(safe_json_dumps(ss.opt_result)[:8000], language="json")

def compute_totals_from_dir(dir_by_vehicle: Dict[str, dict]) -> Tuple[float, float]:
    total_dist = 0.0
    total_dur = 0.0
    for _, j in dir_by_vehicle.items():
        dist, dur = compute_kpis_from_nav(j)
        if dist is not None:
            total_dist += dist
        if dur is not None:
            total_dur += dur
    return total_dist, total_dur

def page_review_export():
    ss = st.session_state
    st.subheader("Review & Export")

    # SUMMARY FIRST
    st.markdown("## Summary (Baseline vs Optimized)")

    b_dist, b_dur = compute_totals_from_dir(ss.dir_baseline) if ss.dir_baseline else (0.0, 0.0)
    o_dist, o_dur = compute_totals_from_dir(ss.dir_optimized) if ss.dir_optimized else (0.0, 0.0)

    def pct(a, b):
        if a == 0:
            return None
        return (b - a) / a * 100.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline distance (m)", f"{b_dist:,.0f}")
    col2.metric("Optimized distance (m)", f"{o_dist:,.0f}", delta=f"{(pct(b_dist, o_dist) or 0):.2f}%" if b_dist else None)
    col3.metric("Baseline duration (s)", f"{b_dur:,.0f}")
    col4.metric("Optimized duration (s)", f"{o_dur:,.0f}", delta=f"{(pct(b_dur, o_dur) or 0):.2f}%" if b_dur else None)

    st.caption("Note: Distances/durations come from Navigation API validation. If you haven't run Directions yet, these will be 0.")

    st.markdown("## Maps (select a vehicle)")
    vids = sorted(set(list(ss.dir_baseline.keys()) + list(ss.dir_optimized.keys())))
    if not vids:
        st.info("Run Directions for baseline and/or optimized routes to view maps.")
        return

    vid = st.selectbox("Vehicle", vids)
    show = st.radio("Show", ["Optimized", "Baseline"], horizontal=True)
    data = ss.dir_optimized.get(vid) if show == "Optimized" else ss.dir_baseline.get(vid)

    # Try to extract polyline points if present; otherwise plot stops only
    coords: List[Tuple[float, float]] = []
    try:
        routes = data.get("routes", [])
        if routes:
            geom = routes[0].get("geometry") or routes[0].get("overview_polyline")
            # We won't decode polyline here to keep dependencies low.
            # We'll just show stops + depot; route polyline requires decode libs.
    except Exception:
        pass

    # If no polyline decode, show points in order (origin->...->destination) from request if possible.
    # This is why we store routes in baseline/opt payloads; for baseline we can reconstruct.
    if show == "Baseline" and ss.baseline_routes:
        starts = assign_depots_to_vehicles(ss.depots_df, int(ss.driver_cfg["driver_count"]), ss.driver_cfg["depot_mode"], int(ss.seed))
        idx = int(vid.replace("V", "")) - 1
        route = ss.baseline_routes.get(vid, [])
        coords = route_coords_for_vehicle(vid, route, starts[idx] if 0 <= idx < len(starts) else None, ss.driver_cfg["route_type"])

    if folium is not None and st_folium is not None:
        to_map(ss.stops_df, ss.depots_df, coords, title=f"{show} route: {vid}")
    else:
        st.json(data)

    st.markdown("## Export")
    export_run_pack()

def page_logs():
    ss = st.session_state
    st.subheader("Logs")

    df = pd.DataFrame(ss.log_rows)
    if df.empty:
        st.info("No logs yet.")
        return
    st.dataframe(df.tail(200), width="stretch", height=400)

    st.download_button(
        "Download logs (jsonl)",
        data="\n".join(json.dumps(r, ensure_ascii=False) for r in ss.log_rows).encode("utf-8"),
        file_name=f"{ss.run_id}_logs.jsonl",
        mime="application/json",
        width="content",
    )

# =========================
# Main
# =========================

def main():
    init_state()
    sidebar_controls()

    st.title("NextBillion Routing Testbench (200-stop QA)")
    next_step_banner()

    tabs = st.tabs(["Stops", "Drivers & Depots", "Clustering", "Optimization", "Review & Export", "Logs"])

    with tabs[0]:
        page_stops()
    with tabs[1]:
        page_drivers_depots()
    with tabs[2]:
        page_clustering()
    with tabs[3]:
        page_optimization()
    with tabs[4]:
        page_review_export()
    with tabs[5]:
        page_logs()

if __name__ == "__main__":
    main()

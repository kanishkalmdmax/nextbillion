# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# FIXES:
# - VRP v2 payload shape (locations.location array) + validation (fixes recurring 400)  (docs show locations as an object w/ location array)
# - Distance Matrix endpoint uses /distancematrix/json first (fixes 404s)
# - Directions distance/time not stuck at 0 (robust parsing)
# - VRP result polling (no-cache fetch button)
# - Debug panel shows exact request/response (URL, params, JSON, raw text)
# - Polyline decoding made crash-safe (no IndexError)

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# -----------------------------
# CONFIG
# -----------------------------
NB_API_KEY = "a08a2b15af0f432c8e438403bc2b00e3"
NB_BASE = "https://api.nextbillion.io"

st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")
YESNO = ["No", "Yes"]

# -----------------------------
# HELPERS
# -----------------------------
def _now_unix() -> int:
    return int(time.time())

def unix_to_human(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)

def normalize_latlng(lat: Any, lng: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        lat_f = float(lat)
        lng_f = float(lng)
        if math.isnan(lat_f) or math.isnan(lng_f):
            return None, None
        if not (-90 <= lat_f <= 90 and -180 <= lng_f <= 180):
            return None, None
        return lat_f, lng_f
    except Exception:
        return None, None

def latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"

def unique_key(prefix: str) -> str:
    st.session_state["_key_seq"] = st.session_state.get("_key_seq", 0) + 1
    return f"{prefix}_{st.session_state['_key_seq']}"

def safe_json_loads(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if not s or s == "{}":
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        st.error(f"Invalid JSON override: {e}")
        return None

def deep_merge(a: Any, b: Any) -> Any:
    """Merge b into a (dicts only), returning new merged object."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return b

# -----------------------------
# STATE
# -----------------------------
def ensure_state():
    ss = st.session_state
    if "center" not in ss:
        ss.center = {"lat": 28.6139, "lng": 77.2090, "country": "IND"}
    if "clicked_pin" not in ss:
        ss.clicked_pin = None
    if "stops_df" not in ss:
        ss.stops_df = pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    if "mapsig" not in ss:
        ss.mapsig = {"geocode": 0, "places": 0, "route_before": 0, "route_after": 0, "matrix": 0, "snap": 0, "iso": 0}
    if "route_before" not in ss:
        ss.route_before = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    if "route_after" not in ss:
        ss.route_after = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    if "vrp" not in ss:
        ss.vrp = {"create": None, "result": None, "order": None, "job_id": None}
    if "last_json" not in ss:
        ss.last_json = {}
    if "last_http" not in ss:
        ss.last_http = {}  # last request/response debug

ensure_state()
ss = st.session_state

def set_center(lat: float, lng: float, country: str | None = None):
    ss.center["lat"] = float(lat)
    ss.center["lng"] = float(lng)
    if country:
        ss.center["country"] = country

def add_stops(rows: List[Dict[str, Any]], source: str):
    df = ss.stops_df.copy()
    for r in rows:
        label = r.get("label") or r.get("name") or f"Stop {len(df)+1}"
        addr = r.get("address") or r.get("name") or ""
        lat = r.get("lat")
        lng = r.get("lng")
        lat_f, lng_f = normalize_latlng(lat, lng)
        if lat_f is None:
            continue
        df = pd.concat(
            [df, pd.DataFrame([{"label": str(label), "address": str(addr), "lat": lat_f, "lng": lng_f, "source": source}])],
            ignore_index=True,
        )
    ss.stops_df = df.reset_index(drop=True)

def replace_stops(df_new: pd.DataFrame):
    ss.stops_df = df_new.reset_index(drop=True)

def clear_stops():
    ss.stops_df = pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    ss.route_before = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.route_after = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.vrp = {"create": None, "result": None, "order": None, "job_id": None}
    ss.last_json = {}
    ss.last_http = {}

# -----------------------------
# HTTP (debuggable)
# -----------------------------
def _http(method: str, path: str, params: Dict[str, Any] | None = None, body: Dict[str, Any] | None = None, timeout: int = 60, nocache: bool = False) -> Tuple[int, Any, str]:
    """
    Returns (status_code, parsed_json_or_text, raw_text)
    Stores request/response in ss.last_http for debugging.
    """
    url = NB_BASE + path
    params = params or {}
    headers = {"accept": "application/json"}

    # Avoid caching on "result fetch" etc by adding a _ts param if requested
    if nocache:
        params = dict(params)
        params["_ts"] = int(time.time() * 1000)

    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        else:
            headers["Content-Type"] = "application/json"
            r = requests.post(url, params=params, json=body or {}, headers=headers, timeout=timeout)

        raw = r.text or ""
        parsed: Any
        try:
            parsed = r.json()
        except Exception:
            parsed = raw

        ss.last_http = {
            "method": method.upper(),
            "url": url,
            "params": params,
            "body": body,
            "status": r.status_code,
            "response_headers": dict(r.headers),
            "raw_text": raw[:20000],  # prevent huge UI payloads
            "parsed": parsed,
        }
        return r.status_code, parsed, raw
    except Exception as e:
        ss.last_http = {"error": str(e), "method": method.upper(), "url": url, "params": params, "body": body}
        return 0, {"error": str(e)}, str(e)

def nb_get(path: str, params: Dict[str, Any], *, nocache: bool = False) -> Tuple[int, Any]:
    stt, parsed, _ = _http("GET", path, params=params, body=None, nocache=nocache)
    return stt, parsed

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any], *, nocache: bool = False) -> Tuple[int, Any]:
    stt, parsed, _ = _http("POST", path, params=params, body=body, nocache=nocache)
    return stt, parsed

# -----------------------------
# PARSERS
# -----------------------------
def extract_geocode_candidates(resp: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(name: str, lat: Any, lng: Any, raw: Any):
        lat_f, lng_f = normalize_latlng(lat, lng)
        if lat_f is None:
            return
        out.append({"name": name, "lat": lat_f, "lng": lng_f, "raw": raw})

    if isinstance(resp, list):
        for it in resp:
            if isinstance(it, dict):
                name = it.get("display_name") or it.get("name") or it.get("label") or "Result"
                add(name, it.get("lat") or it.get("latitude"), it.get("lon") or it.get("lng") or it.get("longitude"), it)
        return out

    if isinstance(resp, dict):
        if "items" in resp and isinstance(resp["items"], list):
            for it in resp["items"]:
                if isinstance(it, dict):
                    pos = it.get("position") or {}
                    name = it.get("title") or it.get("name") or it.get("label") or "Result"
                    add(name, pos.get("lat") or it.get("lat"), pos.get("lng") or it.get("lng"), it)
            return out

        if "results" in resp and isinstance(resp["results"], list):
            for it in resp["results"]:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("formatted") or it.get("label") or "Result"
                    add(name, it.get("lat"), it.get("lng"), it)
            return out

        if "features" in resp and isinstance(resp["features"], list):
            for f in resp["features"]:
                if not isinstance(f, dict):
                    continue
                props = f.get("properties") or {}
                geom = f.get("geometry") or {}
                coords = geom.get("coordinates")
                if isinstance(coords, list) and len(coords) >= 2:
                    lng, lat = coords[0], coords[1]
                    name = props.get("label") or props.get("name") or props.get("display_name") or "Result"
                    add(name, lat, lng, f)
            return out

    return out

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    if not isinstance(polyline_str, str):
        return []
    polyline_str = polyline_str.strip()
    if not polyline_str:
        return []

    coordinates: List[Tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    factor = 10 ** precision
    length = len(polyline_str)

    try:
        while index < length:
            result = 1
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(polyline_str[index]) - 63
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1F:
                    break
            delta_lat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += delta_lat

            result = 1
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(polyline_str[index]) - 63
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1F:
                    break
            delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += delta_lng

            coordinates.append((lat / factor, lng / factor))
    except Exception:
        return coordinates

    return coordinates

def extract_route_geometry(resp: Any) -> List[Tuple[float, float]]:
    if not isinstance(resp, dict):
        return []

    # GeoJSON
    if resp.get("type") == "FeatureCollection" and "features" in resp:
        for f in resp["features"]:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates") or []
                pts = []
                for c in coords:
                    if isinstance(c, list) and len(c) >= 2:
                        lng, lat = c[0], c[1]
                        lat_f, lng_f = normalize_latlng(lat, lng)
                        if lat_f is not None:
                            pts.append((lat_f, lng_f))
                if pts:
                    return pts

    routes = resp.get("routes") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    r0 = routes[0] if isinstance(routes[0], dict) else {}
    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")

    # polyline string
    if isinstance(poly, str) and poly.strip():
        pts5 = decode_polyline(poly, precision=5)
        pts6 = decode_polyline(poly, precision=6)
        # choose whichever looks more plausible
        pts = pts6 if len(pts6) > len(pts5) else pts5
        if len(pts) >= 2:
            return pts

    # list geometry
    geom = r0.get("geometry")
    if isinstance(geom, list):
        pts = []
        for p in geom:
            if isinstance(p, dict):
                lat_f, lng_f = normalize_latlng(p.get("lat"), p.get("lng"))
                if lat_f is not None:
                    pts.append((lat_f, lng_f))
            elif isinstance(p, list) and len(p) >= 2:
                lat_f, lng_f = normalize_latlng(p[0], p[1])
                if lat_f is not None:
                    pts.append((lat_f, lng_f))
        if len(pts) >= 2:
            return pts

    return []

def parse_distance_duration_from_directions(resp: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (distance_m, duration_s) if present.
    Handles numeric or nested formats.
    """
    if not isinstance(resp, dict):
        return None, None
    routes = resp.get("routes") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes[0], dict):
        return None, None
    r0 = routes[0]

    def _num(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                # common shapes: {"value": 1234}
                if "value" in v and isinstance(v["value"], (int, float, str)):
                    return float(v["value"])
            if isinstance(v, str) and v.strip():
                return float(v)
        except Exception:
            return None
        return None

    dist = _num(r0.get("distance"))
    dur = _num(r0.get("duration"))

    # some APIs: distanceMeters / durationSeconds
    if dist is None:
        dist = _num(r0.get("distanceMeters") or r0.get("distance_meters"))
    if dur is None:
        dur = _num(r0.get("durationSeconds") or r0.get("duration_seconds"))

    return dist, dur

# -----------------------------
# MAP RENDERING
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    clicked: Optional[Tuple[float, float]] = None,
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")

    if clicked is not None:
        folium.Marker(
            location=[clicked[0], clicked[1]],
            popup="Pinned center",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    for i, row in stops.iterrows():
        lat = row.get("lat")
        lng = row.get("lng")
        if pd.isna(lat) or pd.isna(lng):
            continue
        label = row.get("label") or f"Stop {i+1}"
        addr = row.get("address") or ""
        folium.Marker(
            location=[float(lat), float(lng)],
            tooltip=f"{label}: {addr}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:#e11d48;color:white;border-radius:999px;
                    width:26px;height:26px;display:flex;align-items:center;justify-content:center;
                    font-size:12px;font-weight:700;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,.35);
                ">{i+1}</div>
                """
            ),
        ).add_to(m)

    if route_pts and len(route_pts) >= 2:
        folium.PolyLine(route_pts, weight=5, opacity=0.8).add_to(m)
        try:
            pl = folium.PolyLine(route_pts, weight=0, opacity=0)
            pl.add_to(m)
            PolyLineTextPath(pl, "  ➤  ", repeat=True, offset=7, attributes={"font-size": "14", "fill": "black"}).add_to(m)
        except Exception:
            pass

        lats = [p[0] for p in route_pts]
        lngs = [p[1] for p in route_pts]
        m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    return m

def render_map(m: folium.Map, key: str) -> Dict[str, Any]:
    return st_folium(m, height=520, width="stretch", key=key)

# -----------------------------
# API BUILDERS
# -----------------------------
def build_global_route_params(ui: Dict[str, Any]) -> Dict[str, Any]:
    p = {"key": NB_API_KEY}
    if ui.get("language"):
        p["language"] = ui["language"]
    if ui.get("departure_time"):
        p["departure_time"] = int(ui["departure_time"])
    if ui.get("mode"):
        p["mode"] = ui["mode"]

    avoid_list = ui.get("avoid") or []
    if avoid_list:
        p["avoid"] = ",".join(avoid_list)

    if ui.get("traffic") is not None:
        p["traffic"] = "true" if ui["traffic"] else "false"
    if ui.get("alternatives") is not None:
        p["alternatives"] = "true" if ui["alternatives"] else "false"
    if ui.get("overview"):
        p["overview"] = ui["overview"]
    return p

def geocode_forward(query: str, country: str, language: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "q": query, "country": country, "language": language}
    stt, data = nb_get("/geocode", params=params)
    if stt == 404:
        stt, data = nb_get("/geocode/v1", params=params)
    return stt, data

def geocode_reverse(lat: float, lng: float, language: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "lat": lat, "lng": lng, "language": language}
    stt, data = nb_get("/reverse", params=params)
    if stt == 404:
        stt, data = nb_get("/reverse/v1", params=params)
    return stt, data

def distance_matrix(origins: List[str], destinations: List[str], params_extra: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Prefer /distancematrix/json (documented) then fallback to /distancematrix/v2.
    """
    params = dict(params_extra)
    params["origins"] = "|".join(origins)
    params["destinations"] = "|".join(destinations)

    stt, data = nb_get("/distancematrix/json", params=params)
    if stt == 404:
        stt, data = nb_get("/distancematrix/v2", params=params)
    return stt, data

def directions_multi_stop(order_latlng: List[str], params_extra: Dict[str, Any], override_params: Dict[str, Any] | None = None) -> Tuple[int, Any]:
    if len(order_latlng) < 2:
        return 0, {"error": "Need at least 2 points"}

    origin = order_latlng[0]
    dest = order_latlng[-1]
    waypoints = order_latlng[1:-1]

    params = dict(params_extra)
    params["origin"] = origin
    params["destination"] = dest
    if waypoints:
        params["waypoints"] = "|".join(waypoints)

    if override_params:
        params = deep_merge(params, override_params)

    # Prefer /directions/json then fallback
    stt, data = nb_get("/directions/json", params=params)
    if stt == 404:
        stt, data = nb_get("/directions/v2", params=params)
    if stt == 404:
        stt, data = nb_get("/navigation/v2", params=params)
    return stt, data

# -----------------------------
# VRP v2 (FIXED PAYLOAD SHAPE)
# -----------------------------
def build_vrp_payload_v2(stops: pd.DataFrame, objective: str, routing_mode: str, traffic_ts: int) -> Dict[str, Any]:
    """
    Correct VRP v2 schema (fixes recurring 400):
      locations is an OBJECT:
        "locations": { "id": 0, "location": ["lat,lng", ...] }
    NOT:
        "locations": [ {id, location}, ... ]
    """
    loc_list: List[str] = []
    for r in stops.reset_index(drop=True).itertuples():
        loc_list.append(latlng_str(float(r.lat), float(r.lng)))

    payload = {
        "locations": {
            "id": 0,
            "location": loc_list
        },
        "jobs": [{"id": int(i), "location_index": int(i)} for i in range(len(loc_list))],
        "vehicles": [{"id": "vehicle_1", "start_index": 0, "end_index": 0}],
        "options": {
            "objective": {"travel_cost": objective},
            "routing": {"mode": routing_mode, "traffic_timestamp": int(traffic_ts)}
        }
    }
    return payload

def validate_vrp_payload_v2(body: Dict[str, Any]) -> Tuple[bool, str]:
    locs = body.get("locations")
    if not isinstance(locs, dict):
        return False, "VRP payload invalid: 'locations' must be an object with field 'location' as an array."
    loc_arr = locs.get("location")
    if not isinstance(loc_arr, list) or not loc_arr or not all(isinstance(x, str) and "," in x for x in loc_arr):
        return False, "VRP payload invalid: locations.location must be a non-empty array of 'lat,lng' strings."
    return True, "OK"

def vrp_create_v2(body: Dict[str, Any]) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY}
    stt, data = nb_post("/optimization/v2", params=params, body=body)
    ss.last_json["vrp_create_payload"] = body
    ss.last_json["vrp_create_response"] = data
    return stt, data

def vrp_result_v2(job_id: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "id": job_id}
    # IMPORTANT: result fetch must not be cached (job becomes ready later)
    stt, data = nb_get("/optimization/v2/result", params=params, nocache=True)
    ss.last_json["vrp_result_response"] = data
    return stt, data

def parse_vrp_order(resp: Any) -> Optional[List[int]]:
    if not isinstance(resp, dict):
        return None

    # common nesting: result.routes[0].steps / activities
    routes = resp.get("routes") or resp.get("result", {}).get("routes") or resp.get("solution", {}).get("routes")
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list) or not isinstance(routes[0], dict):
        return None

    r0 = routes[0]
    steps = r0.get("steps") or r0.get("activities") or r0.get("visits") or []
    order: List[int] = []

    for s in steps:
        if not isinstance(s, dict):
            continue
        li = s.get("location_index")
        if li is None:
            li = s.get("locationIndex")
        if li is not None:
            try:
                order.append(int(li))
            except Exception:
                pass

    if order:
        seen = set()
        out = []
        for x in order:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # Sometimes order is in "sequence" or "jobs"
    seq = r0.get("sequence")
    if isinstance(seq, list) and seq:
        try:
            return [int(x) for x in seq]
        except Exception:
            pass

    return None

# -----------------------------
# SNAP + ISO (best-effort; debug-first)
# -----------------------------
def snap_to_road(points: List[Tuple[float, float]]) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY}
    body = {"points": [{"lat": p[0], "lng": p[1]} for p in points]}
    ss.last_json["snap_payload"] = body

    # try common endpoints
    stt, data = nb_post("/snapToRoad/v2", params=params, body=body)
    if stt == 404:
        stt, data = nb_post("/snaptoroad/v2", params=params, body=body)
    if stt == 404:
        stt, data = nb_post("/snapToRoad", params=params, body=body)

    ss.last_json["snap_response"] = data
    return stt, data

def isochrone(center_lat: float, center_lng: float, iso_type: str, iso_val: int, iso_mode: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY}
    body = {
        "center": {"lat": float(center_lat), "lng": float(center_lng)},
        "type": iso_type,
        "value": int(iso_val),
        "mode": iso_mode,
    }
    ss.last_json["iso_payload"] = body
    stt, data = nb_post("/isochrone/v2", params=params, body=body)
    if stt == 404:
        stt, data = nb_post("/isochrone", params=params, body=body)
    ss.last_json["iso_response"] = data
    return stt, data

# -----------------------------
# UI
# -----------------------------
st.title("NextBillion.ai — Visual API Tester")
st.caption(f"Stops loaded: {len(ss.stops_df)}")

with st.expander("Global route options (Directions / Matrix / Optimize)", expanded=False):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="g_mode")
    with c2:
        language = st.text_input("Language", value="en-US", key="g_lang")
    with c3:
        traffic = st.selectbox("Traffic", YESNO, index=1, key="g_traffic")
    with c4:
        alternatives = st.selectbox("Alternatives", YESNO, index=0, key="g_alts")

    avoid_options = ["toll", "highway", "ferry", "indoor", "unpaved", "tunnel", "sharp_turn", "u_turn", "service_road"]
    avoid = st.multiselect("Avoid", avoid_options, default=[], key="g_avoid")

    dep_col1, dep_col2 = st.columns([1, 1])
    with dep_col1:
        dep_unix = st.number_input("Departure time (unix seconds)", min_value=0, value=_now_unix(), step=60, key="g_dep_unix")
    with dep_col2:
        st.write("Human readable")
        st.code(unix_to_human(int(dep_unix)))

    overview = st.selectbox("Overview", ["full", "simplified", "false"], index=0, key="g_overview")

global_ui = {
    "mode": mode,
    "language": language,
    "traffic": (traffic == "Yes"),
    "alternatives": (alternatives == "Yes"),
    "avoid": avoid,
    "departure_time": int(dep_unix),
    "overview": overview,
}
GLOBAL_PARAMS = build_global_route_params(global_ui)

with st.sidebar:
    st.header("Config")
    st.text_input("NextBillion API Key", value=NB_API_KEY, type="password", key="sb_key", disabled=True)

    st.divider()
    st.subheader("Stops (paste / edit)")

    input_mode = st.radio("How to input stops?", ["Addresses (one per line)", "Lat/Lng (one per line: lat,lng)"], index=0, key="sb_input_mode")
    paste = st.text_area("Paste lines (you can paste 20+)", height=220, key="sb_paste")

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Add / Append Stops", key="sb_add"):
            lines = [x.strip() for x in (paste or "").splitlines() if x.strip()]
            if input_mode.startswith("Addresses"):
                df = ss.stops_df.copy()
                for i, addr in enumerate(lines):
                    df = pd.concat([df, pd.DataFrame([{
                        "label": f"Stop {len(df)+1}",
                        "address": addr,
                        "lat": None,
                        "lng": None,
                        "source": "Pasted address",
                    }])], ignore_index=True)
                ss.stops_df = df.reset_index(drop=True)
            else:
                rows = []
                for i, ll in enumerate(lines):
                    parts = [p.strip() for p in ll.split(",")]
                    if len(parts) >= 2:
                        lat_f, lng_f = normalize_latlng(parts[0], parts[1])
                        if lat_f is not None:
                            rows.append({"label": f"Stop {len(ss.stops_df)+i+1}", "address": "", "lat": lat_f, "lng": lng_f})
                add_stops(rows, "Pasted lat/lng")

    with colB:
        if st.button("🗑️ Clear Stops", key="sb_clear"):
            clear_stops()

    st.caption("Tip: Geocode once → reuse across all tabs (saves API calls).")

tabs = st.tabs(["Geocode & Map", "Places (Search + Generate Stops)", "Route + Optimize (Before vs After)", "Distance Matrix (NxN)", "Snap-to-Road + Isochrone", "Debug (Last API Call)"])

# -----------------------------
# TAB 1: GEOCODE & MAP
# -----------------------------
with tabs[0]:
    st.subheader("Geocode your stops and show them on the map")
    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        country = st.text_input("Country filter (3-letter)", value=ss.center["country"], key="geo_country")
    with gcol2:
        if st.button("🌍 Set country filter", key="geo_set_country"):
            ss.center["country"] = country.strip().upper()[:3] or ss.center["country"]

    df = ss.stops_df.copy()

    if st.button("🧭 Geocode all missing coordinates", key="geo_geocode_btn", use_container_width=True):
        missing = df[df["lat"].isna() | df["lng"].isna()].copy()
        last_resp = None
        for idx, row in missing.iterrows():
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            status, resp = geocode_forward(addr, country=ss.center["country"], language=language)
            last_resp = resp
            candidates = extract_geocode_candidates(resp)
            if candidates:
                top = candidates[0]
                df.loc[idx, "lat"] = top["lat"]
                df.loc[idx, "lng"] = top["lng"]
                df.loc[idx, "source"] = f"Geocoded ({status})"
            else:
                df.loc[idx, "source"] = f"Geocode failed ({status})"

        replace_stops(df)
        ss.last_json["geocode_response_sample"] = last_resp
        ss.mapsig["geocode"] += 1

    st.dataframe(ss.stops_df, use_container_width=True, height=220)

    m = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=ss.stops_df, clicked=ss.clicked_pin)
    map_ret = render_map(m, key=f"map_geocode_{ss.mapsig['geocode']}")

    if map_ret and map_ret.get("last_clicked"):
        lc = map_ret["last_clicked"]
        lat_f, lng_f = normalize_latlng(lc.get("lat"), lc.get("lng"))
        if lat_f is not None:
            ss.clicked_pin = (lat_f, lng_f)
            set_center(lat_f, lng_f)
            ss.mapsig["geocode"] += 1

# -----------------------------
# TAB 2: PLACES + GENERATE STOPS
# -----------------------------
with tabs[1]:
    st.subheader("Search region/city → set center → generate 20+ random stops OR add POIs as stops")
    rcol1, rcol2 = st.columns([3, 1])
    with rcol1:
        region_q = st.text_input("Region/City/State/Country", value="", key="pl_region_q")
    with rcol2:
        pl_country = st.text_input("Country filter (3-letter)", value=ss.center["country"], key="pl_country")

    if st.button("Search Region", key="pl_search_region"):
        status, resp = geocode_forward(region_q, country=pl_country.strip().upper()[:3], language=language)
        ss.last_json["region_search_response"] = resp
        ss._region_candidates = extract_geocode_candidates(resp)
    candidates = ss.get("_region_candidates", [])

    if candidates:
        labels = [f"{c['name']}  ({c['lat']:.5f},{c['lng']:.5f})" for c in candidates[:20]]
        pick = st.selectbox("Pick a region result", labels, index=0, key="pl_pick_region")
        picked = candidates[labels.index(pick)]
        if st.button("Use picked region as center", key="pl_use_region_center"):
            set_center(picked["lat"], picked["lng"], country=pl_country.strip().upper()[:3] or ss.center["country"])
            ss.mapsig["places"] += 1
            st.success(f"Center set to: {picked['name']}")

    st.caption(f"Current center: {ss.center['lat']:.5f},{ss.center['lng']:.5f} | Country: {ss.center['country']}")

    m2 = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=ss.stops_df, clicked=ss.clicked_pin)
    ret2 = render_map(m2, key=f"map_places_{ss.mapsig['places']}")
    if ret2 and ret2.get("last_clicked"):
        lc = ret2["last_clicked"]
        lat_f, lng_f = normalize_latlng(lc.get("lat"), lc.get("lng"))
        if lat_f is not None:
            ss.clicked_pin = (lat_f, lng_f)
            set_center(lat_f, lng_f)
            ss.mapsig["places"] += 1

    st.markdown("### Generate random stops (no keyword needed)")
    g1, g2, g3, g4 = st.columns([1, 1, 1, 1])
    with g1:
        n_rand = st.number_input("How many stops?", min_value=2, max_value=200, value=20, step=1, key="pl_n_rand")
    with g2:
        radius_m = st.number_input("Radius (m)", min_value=200, max_value=200000, value=5000, step=100, key="pl_radius")
    with g3:
        resolve_addr = st.selectbox("Resolve to addresses (costs API)", YESNO, index=0, key="pl_resolve")
    with g4:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="pl_seed")

    def gen_random_points(center_lat: float, center_lng: float, n: int, radius_m: int, seed: int) -> List[Tuple[float, float]]:
        random.seed(seed)
        pts = []
        for _ in range(n):
            r = radius_m * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            dlat = (r * math.cos(theta)) / 111_320.0
            dlng = (r * math.sin(theta)) / (111_320.0 * math.cos(math.radians(center_lat)) + 1e-9)
            pts.append((center_lat + dlat, center_lng + dlng))
        return pts

    if st.button("🎲 Generate random stops around center", key="pl_gen_random", use_container_width=True):
        pts = gen_random_points(float(ss.center["lat"]), float(ss.center["lng"]), int(n_rand), int(radius_m), int(seed))
        rows = [{"label": f"Rand {i}", "address": "", "lat": la, "lng": ln} for i, (la, ln) in enumerate(pts, start=1)]
        add_stops(rows, "Random around center")

        if resolve_addr == "Yes":
            df = ss.stops_df.copy()
            for idx, row in df[df["source"].str.contains("Random", na=False)].iterrows():
                if str(row.get("address") or "").strip():
                    continue
                status, rresp = geocode_reverse(float(row["lat"]), float(row["lng"]), language=language)
                cand = extract_geocode_candidates(rresp)
                if cand:
                    df.loc[idx, "address"] = cand[0]["name"]
                    df.loc[idx, "source"] = f"Reverse-geocode ({status})"
            replace_stops(df)

        ss.mapsig["places"] += 1

# -----------------------------
# TAB 3: ROUTE + OPTIMIZE
# -----------------------------
with tabs[2]:
    st.subheader("Compute route (Before) → run optimization → recompute route (After) + compare")

    if len(ss.stops_df) < 2:
        st.info("Add at least 2 stops first.")
    else:
        stops_ll = [latlng_str(float(r.lat), float(r.lng)) for r in ss.stops_df.itertuples()]

        left, right = st.columns([1.15, 1.0])

        with left:
            st.markdown("### Step 1 — Directions (Before)")
            dir_override_text = st.text_area("Optional: Directions query override (JSON). Leave {} for none.", value="{}", height=90, key="dir_before_override")

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", use_container_width=True):
                override = safe_json_loads(dir_override_text)
                if override is None:
                    st.stop()

                stt_d, resp_d = directions_multi_stop(stops_ll, params_extra=GLOBAL_PARAMS, override_params=override)
                ss.route_before["resp"] = resp_d
                ss.last_json["directions_before"] = resp_d

                dist_m, dur_s = parse_distance_duration_from_directions(resp_d)
                ss.route_before["distance_m"] = dist_m
                ss.route_before["duration_s"] = dur_s

                if dist_m is None or dur_s is None:
                    # fallback via matrix sum
                    stt_m, data_m = distance_matrix(stops_ll[:-1], stops_ll[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_before"] = data_m
                    dist_total = 0.0
                    dur_total = 0.0
                    if isinstance(data_m, dict):
                        rows = data_m.get("rows") or []
                        for row in rows:
                            elems = (row or {}).get("elements") or []
                            if elems:
                                e0 = elems[0]
                                dist_total += float((e0.get("distance") or {}).get("value") or 0)
                                dur_total += float((e0.get("duration") or {}).get("value") or 0)
                    ss.route_before["distance_m"] = dist_total
                    ss.route_before["duration_s"] = dur_total

                ss.route_before["geometry"] = extract_route_geometry(resp_d)
                ss.mapsig["route_before"] += 1

            before_geom = ss.route_before.get("geometry") or []
            m_before = make_map(
                center=(float(ss.center["lat"]), float(ss.center["lng"])),
                stops=ss.stops_df,
                route_pts=before_geom if before_geom else None,
                clicked=ss.clicked_pin,
                zoom=12,
            )
            render_map(m_before, key=f"map_route_before_{ss.mapsig['route_before']}")

            bdist = ss.route_before.get("distance_m")
            bdur = ss.route_before.get("duration_s")
            if bdist is not None:
                st.metric("Before distance (km)", f"{bdist/1000:.2f}")
            if bdur is not None:
                st.metric("Before duration (min)", f"{bdur/60:.1f}")

        with right:
            st.markdown("### Step 2 — Optimization (VRP v2)")
            objective = st.selectbox("Optimization objective", ["distance", "duration"], index=0, key="vrp_obj")

            vrp_override_text = st.text_area(
                "Optional: Custom VRP request body override (JSON). Leave {} to use generated payload.",
                value="{}",
                height=120,
                key="vrp_override",
            )

            routing_mode = st.selectbox("VRP routing.mode", ["car", "truck", "bike", "walk"], index=0, key="vrp_routing_mode")
            traffic_ts = st.number_input("VRP routing.traffic_timestamp (unix)", min_value=0, value=_now_unix(), step=60, key="vrp_traffic_ts")

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", use_container_width=True):
                override = safe_json_loads(vrp_override_text)
                if override is None:
                    st.stop()

                base = build_vrp_payload_v2(ss.stops_df, objective=objective, routing_mode=routing_mode, traffic_ts=int(traffic_ts))
                body = deep_merge(base, override or {})

                ok, msg = validate_vrp_payload_v2(body)
                ss.last_json["vrp_validated_payload"] = body
                if not ok:
                    st.error(msg)
                    st.json(body)
                    st.stop()

                stt_c, data_c = vrp_create_v2(body)
                if stt_c != 200:
                    st.error(f"VRP create failed: HTTP {stt_c}")
                    st.json(data_c)
                else:
                    ss.vrp["create"] = data_c
                    job_id = data_c.get("id") or data_c.get("job_id") or data_c.get("jobId")
                    ss.vrp["job_id"] = job_id
                    st.success(f"Optimization job created: {job_id}")

            if st.button("🔄 Fetch VRP result again", key="vrp_fetch_btn", use_container_width=True):
                job_id = ss.vrp.get("job_id")
                if not job_id:
                    st.warning("No job id yet. Run optimization first.")
                else:
                    stt_r, rr = vrp_result_v2(str(job_id))
                    ss.vrp["result"] = rr
                    if stt_r != 200:
                        st.error(f"VRP result fetch failed: HTTP {stt_r}")
                        st.json(rr)
                    else:
                        order = parse_vrp_order(rr)
                        ss.vrp["order"] = order
                        if not order:
                            st.warning("Result fetched, but order not found yet (or schema differs). Try again in a few seconds.")
                        else:
                            st.success(f"Order parsed with {len(order)} stops.")

            st.markdown("### Step 3 — Recompute Directions (After)")
            dir_after_override_text = st.text_area("Optional: Directions (After) query override (JSON). Leave {} for none.", value="{}", height=90, key="dir_after_override")

            df_after = ss.get("_optimized_stops", None)
            order = ss.vrp.get("order")

            if order:
                df2 = ss.stops_df.copy().reset_index(drop=True)
                valid = [i for i in order if 0 <= i < len(df2)]
                missing = [i for i in range(len(df2)) if i not in valid]
                final_order = valid + missing
                df_after = df2.iloc[final_order].reset_index(drop=True)
                ss._optimized_stops = df_after
            else:
                if df_after is None:
                    df_after = ss.stops_df.copy()

            if st.button("🧭 Compute route (After)", key="rt_after_btn", use_container_width=True):
                override = safe_json_loads(dir_after_override_text)
                if override is None:
                    st.stop()

                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]
                stt_d2, resp_d2 = directions_multi_stop(ll_after, params_extra=GLOBAL_PARAMS, override_params=override)
                ss.route_after["resp"] = resp_d2
                ss.last_json["directions_after"] = resp_d2

                dist_m2, dur_s2 = parse_distance_duration_from_directions(resp_d2)
                ss.route_after["distance_m"] = dist_m2
                ss.route_after["duration_s"] = dur_s2

                if dist_m2 is None or dur_s2 is None:
                    stt_m2, data_m2 = distance_matrix(ll_after[:-1], ll_after[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_after"] = data_m2
                    dist_total2 = 0.0
                    dur_total2 = 0.0
                    if isinstance(data_m2, dict):
                        rows = data_m2.get("rows") or []
                        for row in rows:
                            elems = (row or {}).get("elements") or []
                            if elems:
                                e0 = elems[0]
                                dist_total2 += float((e0.get("distance") or {}).get("value") or 0)
                                dur_total2 += float((e0.get("duration") or {}).get("value") or 0)
                    ss.route_after["distance_m"] = dist_total2
                    ss.route_after["duration_s"] = dur_total2

                ss.route_after["geometry"] = extract_route_geometry(resp_d2)
                ss.mapsig["route_after"] += 1

            after_geom = ss.route_after.get("geometry") or []
            m_after = make_map(
                center=(float(ss.center["lat"]), float(ss.center["lng"])),
                stops=df_after,
                route_pts=after_geom if after_geom else None,
                clicked=ss.clicked_pin,
                zoom=12,
            )
            render_map(m_after, key=f"map_route_after_{ss.mapsig['route_after']}")

            adist = ss.route_after.get("distance_m")
            adur = ss.route_after.get("duration_s")
            if adist is not None:
                st.metric("After distance (km)", f"{adist/1000:.2f}")
            if adur is not None:
                st.metric("After duration (min)", f"{adur/60:.1f}")

            if ss.route_before.get("distance_m") and ss.route_after.get("distance_m"):
                b = float(ss.route_before["distance_m"])
                a = float(ss.route_after["distance_m"])
                saved = b - a
                pct = (saved / b * 100.0) if b > 0 else 0.0
                st.success(f"Distance saved: {saved/1000:.2f} km ({pct:.1f}%)")

            if ss.route_before.get("duration_s") and ss.route_after.get("duration_s"):
                b = float(ss.route_before["duration_s"])
                a = float(ss.route_after["duration_s"])
                saved = b - a
                pct = (saved / b * 100.0) if b > 0 else 0.0
                st.success(f"Time saved: {saved/60:.1f} min ({pct:.1f}%)")

# -----------------------------
# TAB 4: MATRIX
# -----------------------------
with tabs[3]:
    st.subheader("Distance Matrix (NxN) — up to 20+ points")

    if len(ss.stops_df) < 2:
        st.info("Add at least 2 stops first.")
    else:
        max_n = max(2, len(ss.stops_df))
        if max_n == 2:
            n = 2
            st.caption("Only 2 stops available.")
        else:
            n = st.slider("How many stops to include in NxN?", min_value=2, max_value=max_n, value=min(20, max_n), step=1, key="mx_n")

        use_first_n = st.selectbox("Use", ["First N stops", "All stops"], index=0, key="mx_use")
        df_use = ss.stops_df.copy()
        if use_first_n == "First N stops":
            df_use = df_use.iloc[:n].reset_index(drop=True)

        ll = [latlng_str(float(r.lat), float(r.lng)) for r in df_use.itertuples()]

        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", use_container_width=True):
            stt, data = distance_matrix(ll, ll, params_extra=GLOBAL_PARAMS)
            ss.last_json["matrix_nxn"] = data
            ss.mapsig["matrix"] += 1
            if stt != 200:
                st.error(f"Matrix failed: HTTP {stt}")
                st.json(data)
            else:
                st.success("Matrix computed.")

        m4 = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=df_use, clicked=ss.clicked_pin, zoom=12)
        render_map(m4, key=f"map_matrix_{ss.mapsig['matrix']}")

        data = ss.last_json.get("matrix_nxn")
        if isinstance(data, dict) and data.get("rows"):
            rows = data["rows"]
            dist_km = []
            dur_min = []
            for r in rows:
                elems = r.get("elements") or []
                dist_km.append([((e.get("distance") or {}).get("value") or 0) / 1000 for e in elems])
                dur_min.append([((e.get("duration") or {}).get("value") or 0) / 60 for e in elems])

            st.markdown("### Distance (km)")
            st.dataframe(pd.DataFrame(dist_km).round(2), use_container_width=True, height=240)
            st.markdown("### Duration (min)")
            st.dataframe(pd.DataFrame(dur_min).round(1), use_container_width=True, height=240)

# -----------------------------
# TAB 5: SNAP + ISO
# -----------------------------
with tabs[4]:
    st.subheader("Snap-to-Road + Isochrone (debug-first; runs only on click)")

    s1, s2 = st.columns([1, 1])

    with s1:
        st.markdown("### Snap-to-Road (best-effort)")
        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use Directions (After) geometry"], index=0, key="snap_src")
        geom = ss.route_before.get("geometry") if src.startswith("Use Directions (Before)") else ss.route_after.get("geometry")
        coords = geom or []

        max_pts = max(2, len(coords)) if coords else 2
        if max_pts <= 2:
            n_path = 2
            st.caption("Not enough geometry points yet. Compute a route first.")
        else:
            n_path = st.slider("N geometry points to send (first N)", min_value=2, max_value=max_pts, value=min(200, max_pts), step=1, key="snap_n")

        if st.button("🧷 Snap to Road", key="snap_btn", use_container_width=True):
            pts = coords[: int(n_path)] if coords else []
            if len(pts) < 2:
                st.warning("Need route geometry (2+ points). Compute directions first.")
            else:
                stt, data = snap_to_road(pts)
                ss.mapsig["snap"] += 1
                if stt != 200:
                    st.error(f"Snap-to-road failed: HTTP {stt}")
                    st.json(data)
                else:
                    st.success("Snap-to-road response received.")
                    st.json(data)

        m5 = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=ss.stops_df, route_pts=coords[: int(n_path)] if coords else None, clicked=ss.clicked_pin, zoom=12)
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (best-effort)")
        iso_type = st.selectbox("Type", ["time", "distance"], index=0, key="iso_type")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")
        iso_val = st.number_input("Value (seconds if time, meters if distance)", min_value=60, max_value=2000000, value=1800, step=60, key="iso_val")

        if st.button("🟦 Compute Isochrone", key="iso_btn", use_container_width=True):
            stt, data = isochrone(float(ss.center["lat"]), float(ss.center["lng"]), iso_type, int(iso_val), iso_mode)
            ss.mapsig["iso"] += 1
            if stt != 200:
                st.error(f"Isochrone failed: HTTP {stt}")
                st.json(data)
            else:
                st.success("Isochrone response received.")
                st.json(data)

        m6 = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=ss.stops_df, clicked=ss.clicked_pin, zoom=12)
        render_map(m6, key=f"map_iso_{ss.mapsig['iso']}")

# -----------------------------
# TAB 6: DEBUG
# -----------------------------
with tabs[5]:
    st.subheader("Debug — Last API Call (exact request + exact response)")
    if not ss.last_http:
        st.info("No API calls made yet in this session.")
    else:
        st.json(ss.last_http)

    st.markdown("### Saved JSON bundles")
    st.json(ss.last_json)

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import polyline  # pip install polyline


NB_BASE = "https://api.nextbillion.io"


# -----------------------------
# Session State Helpers
# -----------------------------
def init_state():
    st.session_state.setdefault("run_logs", [])
    st.session_state.setdefault("last_vrp_job_id", None)
    st.session_state.setdefault("last_vrp_result", None)
    st.session_state.setdefault("last_route_latlng", None)  # list[(lat,lng)]
    st.session_state.setdefault("api_key", "")


def add_log(entry: Dict[str, Any]) -> None:
    st.session_state.run_logs.append(entry)


def download_logs_button():
    payload = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "logs": st.session_state.run_logs,
    }
    st.download_button(
        "Download ALL Run Logs (JSON)",
        data=json.dumps(payload, indent=2),
        file_name=f"nextbillion_run_logs_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True,
    )


# -----------------------------
# Request Wrapper (logs everything)
# -----------------------------
def nb_request(
    method: str,
    path: str,
    *,
    key: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    if not params:
        params = {}
    params = dict(params)
    params["key"] = key  # NextBillion uses key= in query string

    url = f"{NB_BASE}{path}"
    started = time.time()

    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json_body,
            timeout=timeout,
        )
        elapsed_ms = int((time.time() - started) * 1000)

        # Try JSON, fallback to text
        try:
            parsed = resp.json()
        except Exception:
            parsed = None

        log_entry = {
            "ts_utc": datetime.utcnow().isoformat() + "Z",
            "method": method.upper(),
            "url": url,
            "params": params,
            "body": json_body,
            "status": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "response_headers": dict(resp.headers),
            "parsed": parsed,
            "raw_text": resp.text if parsed is None else None,
        }
        add_log(log_entry)

        # Return consistent structure
        return {
            "ok": resp.ok,
            "status": resp.status_code,
            "data": parsed if parsed is not None else resp.text,
        }

    except Exception as e:
        elapsed_ms = int((time.time() - started) * 1000)
        add_log(
            {
                "ts_utc": datetime.utcnow().isoformat() + "Z",
                "method": method.upper(),
                "url": url,
                "params": params,
                "body": json_body,
                "status": None,
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            }
        )
        return {"ok": False, "status": None, "data": {"error": str(e)}}


# -----------------------------
# Geometry Helpers
# -----------------------------
def decode_route_polyline_to_latlng(encoded: str) -> List[Tuple[float, float]]:
    """
    NextBillion returns encoded polyline. polyline.decode gives [(lat, lng), ...].
    This format is what Folium/Leaflet expects for PolyLine.
    """
    if not encoded:
        return []
    return polyline.decode(encoded)


# -----------------------------
# API Calls (Correct Endpoints)
# -----------------------------
def get_isochrone(
    key: str,
    lat: float,
    lng: float,
    minutes: int = 30,
    mode: str = "car",
    polygons: bool = True,
) -> Dict[str, Any]:
    # Isochrone is GET /isochrone/json with coordinates=lat,lng (per docs) :contentReference[oaicite:3]{index=3}
    params = {
        "coordinates": f"{lat},{lng}",
        "contours_minutes": str(minutes),
        "polygons": "true" if polygons else "false",
        "mode": mode,
    }
    return nb_request("GET", "/isochrone/json", key=key, params=params)


def snap_to_roads(
    key: str,
    points_latlng: List[Tuple[float, float]],
) -> Dict[str, Any]:
    # Snap is POST /snapToRoads/json :contentReference[oaicite:4]{index=4}
    body = {
        "points": [{"lat": lat, "lng": lng} for lat, lng in points_latlng]
    }
    return nb_request("POST", "/snapToRoads/json", key=key, json_body=body)


def create_vrp_job(key: str, vrp_payload: Dict[str, Any]) -> Dict[str, Any]:
    # Create job: POST /optimization/v2?key=... :contentReference[oaicite:5]{index=5}
    return nb_request("POST", "/optimization/v2", key=key, json_body=vrp_payload)


def fetch_vrp_result(key: str, job_id: str) -> Dict[str, Any]:
    # Fetch result: GET /optimization/v2/result?id=...&key=... :contentReference[oaicite:6]{index=6}
    return nb_request("GET", "/optimization/v2/result", key=key, params={"id": job_id})


def poll_vrp_result(
    key: str,
    job_id: str,
    *,
    max_attempts: int = 10,
    sleep_s: float = 1.0,
) -> Dict[str, Any]:
    """
    Handles the "Job created, but result not ready yet" situation.
    We retry a few times, logging every attempt.
    """
    last = None
    for attempt in range(1, max_attempts + 1):
        last = fetch_vrp_result(key, job_id)
        if last.get("ok") and isinstance(last.get("data"), dict):
            # Heuristic: if result exists, we're done
            if "result" in last["data"] and last["data"]["result"]:
                return last
        time.sleep(sleep_s)
    return last or {"ok": False, "status": None, "data": {"error": "No response"}}


# -----------------------------
# UI: Run Logs Panel
# -----------------------------
def render_logs_panel():
    st.subheader("Run Logs (for debugging)")
    st.caption("Every API action is captured here. Download and paste the JSON to share a full run.")

    col1, col2 = st.columns([1, 1])
    with col1:
        download_logs_button()
    with col2:
        if st.button("Clear logs", use_container_width=True):
            st.session_state.run_logs = []
            st.rerun()

    with st.expander(f"View latest logs ({len(st.session_state.run_logs)})", expanded=False):
        st.json(st.session_state.run_logs[-5:] if st.session_state.run_logs else [])


# -----------------------------
# Example: Route Optimization Screen (minimal + correct)
# -----------------------------
def route_optimization_screen():
    st.header("Route Optimization (VRP)")

    key = st.session_state.api_key.strip()
    if not key:
        st.warning("Add your NextBillion API key in the sidebar.")
        return

    st.write("Paste your VRP payload (JSON) below, then create + fetch results.")

    vrp_text = st.text_area(
        "VRP Payload JSON",
        height=250,
        value=json.dumps(
            {
                "locations": {"id": 0, "location": ["-35.616515,-65.299796"]},
                "jobs": [{"id": 0, "location_index": 0}],
                "vehicles": [{"id": "vehicle_1", "start_index": 0, "end_index": 0}],
            },
            indent=2,
        ),
    )

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("Create VRP Job", use_container_width=True):
            try:
                payload = json.loads(vrp_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                return

            resp = create_vrp_job(key, payload)
            st.json(resp["data"])

            if resp["ok"] and isinstance(resp["data"], dict) and resp["data"].get("id"):
                st.session_state.last_vrp_job_id = resp["data"]["id"]
                st.success(f"Job created: {st.session_state.last_vrp_job_id}")

    with colB:
        if st.button("Fetch VRP Result (once)", use_container_width=True):
            job_id = st.session_state.last_vrp_job_id
            if not job_id:
                st.warning("No job_id yet. Create a job first.")
            else:
                res = fetch_vrp_result(key, job_id)
                st.session_state.last_vrp_result = res["data"]
                st.json(res["data"])

    with colC:
        if st.button("Fetch VRP Result (retry)", use_container_width=True):
            job_id = st.session_state.last_vrp_job_id
            if not job_id:
                st.warning("No job_id yet. Create a job first.")
            else:
                res = poll_vrp_result(key, job_id, max_attempts=12, sleep_s=1.0)
                st.session_state.last_vrp_result = res["data"]
                st.json(res["data"])

    st.divider()

    st.subheader("Route Geometry (fixes the ‘straight line’ issue)")
    st.caption("We decode the route geometry polyline from optimization result and store lat/lng pairs.")

    if st.session_state.last_vrp_result and isinstance(st.session_state.last_vrp_result, dict):
        data = st.session_state.last_vrp_result
        try:
            routes = data["result"]["routes"]
            if routes and "geometry" in routes[0]:
                encoded = routes[0]["geometry"]
                latlng = decode_route_polyline_to_latlng(encoded)
                st.session_state.last_route_latlng = latlng
                st.success(f"Decoded {len(latlng)} points (lat,lng). This is what you should plot on the map.")
                st.write(latlng[:5], " ...")
            else:
                st.info("No geometry found in result yet.")
        except Exception as e:
            st.error(f"Could not read geometry: {e}")

    st.divider()
    render_logs_panel()


# -----------------------------
# Sidebar + App Router
# -----------------------------
def main():
    st.set_page_config(page_title="NextBillion Debuggable Console", layout="wide")
    init_state()

    with st.sidebar:
        st.title("Settings")
        st.session_state.api_key = st.text_input("NextBillion API Key", value=st.session_state.api_key, type="password")

        st.caption("Tip: After any run, download the full logs JSON and share it.")

        if st.session_state.last_vrp_job_id:
            st.info(f"Last VRP job_id:\n{st.session_state.last_vrp_job_id}")

    route_optimization_screen()


if __name__ == "__main__":
    main()


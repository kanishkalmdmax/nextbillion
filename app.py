# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# Fixes:
# - Correct endpoints/payloads for Directions, Matrix, VRP v2, Snap-to-Roads, Isochrone
# - VRP v2 "locations" MUST be a list of "lat,lng" strings (not an object/dict)
# - Distance/time now read from Directions response (fallback to matrix)
# - Safer polyline decoding (prevents IndexError)
# - GeoJSON polygon rendering for Isochrone
# - JSON override boxes for all major calls
#
# Docs references:
# - Directions Fast API: /directions/json  (GET)  https://api.nextbillion.io/directions/json?key=...
# - Distance Matrix: /distancematrix/json (GET)  https://api.nextbillion.io/distancematrix/json?key=...
# - Optimization VRP v2: /optimization/v2 (POST) https://api.nextbillion.io/optimization/v2?key=...
# - SnapToRoads: /snapToRoads/json (GET/POST) https://api.nextbillion.io/snapToRoads/json?key=...
# - Isochrone: /isochrone/json (GET) https://api.nextbillion.io/isochrone/json?key=...

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# -----------------------------
# CONFIG
# -----------------------------
NB_API_KEY = "a08a2b15af0f432c8e438403bc2b00e3"  # embedded as requested
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
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

# -----------------------------
# HTTP (button-only + caching)
# -----------------------------
@dataclass(frozen=True)
class ReqSig:
    method: str
    path: str
    params_json: str
    body_json: str

def _sig(method: str, path: str, params: Dict[str, Any] | None, body: Any | None) -> ReqSig:
    pj = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    bj = json.dumps(body or {}, sort_keys=True, separators=(",", ":"))
    return ReqSig(method=method.upper(), path=path, params_json=pj, body_json=bj)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_request(sig: ReqSig) -> Tuple[int, Union[Dict[str, Any], List[Any], str]]:
    method = sig.method
    path = sig.path
    params = json.loads(sig.params_json)
    body = json.loads(sig.body_json)

    url = NB_BASE + path
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=60)
        else:
            r = requests.post(url, params=params, json=body, timeout=60)

        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, {"error": str(e)}

def nb_get(path: str, params: Dict[str, Any]) -> Tuple[int, Any]:
    sig = _sig("GET", path, params=params, body=None)
    return _cached_request(sig)

def nb_post(path: str, params: Dict[str, Any], body: Any) -> Tuple[int, Any]:
    sig = _sig("POST", path, params=params, body=body)
    return _cached_request(sig)

# -----------------------------
# PARSERS (robust)
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

def extract_places_items(resp: Any) -> List[Dict[str, Any]]:
    return extract_geocode_candidates(resp)

def decode_polyline(polyline_str: Any, precision: int = 5) -> List[Tuple[float, float]]:
    # Safe decoder: never throws IndexError; returns [] on malformed polyline
    if not isinstance(polyline_str, str):
        return []
    s = polyline_str.strip()
    if not s:
        return []

    index = 0
    lat = 0
    lng = 0
    coordinates: List[Tuple[float, float]] = []
    factor = 10 ** precision
    length = len(s)

    try:
        while index < length:
            # lat
            result = 1
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(s[index]) - 63
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1F:
                    break
            delta_lat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += delta_lat

            # lng
            result = 1
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(s[index]) - 63
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1F:
                    break
            delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += delta_lng

            coordinates.append((lat / factor, lng / factor))
        return coordinates
    except Exception:
        return []

def extract_route_geometry(resp: Any) -> List[Tuple[float, float]]:
    # Directions API usually returns routes[0].geometry (encoded polyline)
    if not isinstance(resp, dict):
        return []

    # GeoJSON line
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
    if not routes or not isinstance(routes, list) or not isinstance(routes[0], dict):
        return []

    r0 = routes[0]
    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")
    if isinstance(poly, str) and poly.strip():
        # Some deployments use polyline6; try both and pick the longer
        pts5 = decode_polyline(poly, precision=5)
        pts6 = decode_polyline(poly, precision=6)
        if pts6 and len(pts6) >= len(pts5):
            return pts6
        return pts5

    return []

def extract_directions_totals(resp: Any) -> Tuple[Optional[float], Optional[float]]:
    # Returns (distance_m, duration_s) from Directions response if present
    if not isinstance(resp, dict):
        return None, None
    routes = resp.get("routes") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes[0], dict):
        return None, None
    r0 = routes[0]
    dist = r0.get("distance")
    dur = r0.get("duration")
    try:
        dist_m = float(dist) if dist is not None else None
    except Exception:
        dist_m = None
    try:
        dur_s = float(dur) if dur is not None else None
    except Exception:
        dur_s = None
    return dist_m, dur_s

def extract_isochrone_polygons(resp: Any) -> List[List[Tuple[float, float]]]:
    # Isochrone /isochrone/json often returns GeoJSON polygons in "features"
    # Return list of rings as [(lat,lng), ...]
    out: List[List[Tuple[float, float]]] = []
    if not isinstance(resp, dict):
        return out

    if resp.get("type") == "FeatureCollection" and isinstance(resp.get("features"), list):
        for f in resp["features"]:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates") or []
                if coords and isinstance(coords, list) and isinstance(coords[0], list):
                    ring = []
                    for c in coords[0]:
                        if isinstance(c, list) and len(c) >= 2:
                            lng, lat = c[0], c[1]
                            lat_f, lng_f = normalize_latlng(lat, lng)
                            if lat_f is not None:
                                ring.append((lat_f, lng_f))
                    if ring:
                        out.append(ring)
            elif geom.get("type") == "MultiPolygon":
                polys = geom.get("coordinates") or []
                for p in polys:
                    if not p or not isinstance(p, list) or not isinstance(p[0], list):
                        continue
                    ring = []
                    for c in p[0]:
                        if isinstance(c, list) and len(c) >= 2:
                            lng, lat = c[0], c[1]
                            lat_f, lng_f = normalize_latlng(lat, lng)
                            if lat_f is not None:
                                ring.append((lat_f, lng_f))
                    if ring:
                        out.append(ring)
    return out

# -----------------------------
# MAP RENDERING (stable)
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    polygons: Optional[List[List[Tuple[float, float]]]] = None,
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

    if route_pts:
        folium.PolyLine(route_pts, weight=5, opacity=0.8).add_to(m)
        try:
            pl = folium.PolyLine(route_pts, weight=0, opacity=0)
            pl.add_to(m)
            PolyLineTextPath(pl, "  ➤  ", repeat=True, offset=7, attributes={"font-size": "14", "fill": "black"}).add_to(m)
        except Exception:
            pass

    if polygons:
        for ring in polygons:
            folium.Polygon(locations=ring, weight=3, opacity=0.9, fill=True, fill_opacity=0.2).add_to(m)

    # Fit bounds
    bounds_pts: List[Tuple[float, float]] = []
    if route_pts:
        bounds_pts.extend(route_pts)
    if polygons:
        for ring in polygons:
            bounds_pts.extend(ring)
    if not bounds_pts:
        # fit to stops
        for _, row in stops.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lng")):
                continue
            bounds_pts.append((float(row["lat"]), float(row["lng"])))

    if bounds_pts:
        lats = [p[0] for p in bounds_pts]
        lngs = [p[1] for p in bounds_pts]
        m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    return m

def render_map(m: folium.Map, key: str) -> Dict[str, Any]:
    return st_folium(m, height=520, width="stretch", key=key)

# -----------------------------
# STATE
# -----------------------------
def ensure_state():
    ss = st.session_state
    if "center" not in ss:
        ss.center = {"lat": 28.6139, "lng": 77.2090, "country": "IND"}  # default Delhi
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

def set_center(lat: float, lng: float, country: str | None = None):
    ss = st.session_state
    ss.center["lat"] = float(lat)
    ss.center["lng"] = float(lng)
    if country:
        ss.center["country"] = country

def add_stops(rows: List[Dict[str, Any]], source: str):
    ss = st.session_state
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
            [df, pd.DataFrame([{
                "label": str(label),
                "address": str(addr),
                "lat": lat_f,
                "lng": lng_f,
                "source": source,
            }])],
            ignore_index=True,
        )
    ss.stops_df = df.reset_index(drop=True)

def replace_stops(df_new: pd.DataFrame):
    st.session_state.stops_df = df_new.reset_index(drop=True)

def clear_stops():
    st.session_state.stops_df = pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    st.session_state.route_before = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    st.session_state.route_after = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    st.session_state.vrp = {"create": None, "result": None, "order": None, "job_id": None}

# -----------------------------
# API CALL BUILDERS (correct endpoints)
# -----------------------------
def build_global_route_params(ui: Dict[str, Any]) -> Dict[str, Any]:
    p: Dict[str, Any] = {"key": NB_API_KEY}

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
    status, data = nb_get("/geocode", params=params)
    if status == 404:
        status, data = nb_get("/geocode/v1", params=params)
    return status, data

def geocode_reverse(lat: float, lng: float, language: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "lat": lat, "lng": lng, "language": language}
    status, data = nb_get("/reverse", params=params)
    if status == 404:
        status, data = nb_get("/reverse/v1", params=params)
    return status, data

def distance_matrix(origins: List[str], destinations: List[str], params_extra: Dict[str, Any], custom_override: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # Correct endpoint: /distancematrix/json (GET)
    params = dict(params_extra)
    params["origins"] = "|".join(origins)
    params["destinations"] = "|".join(destinations)

    # optional "option=fast|flexible" etc can be set by override
    if custom_override:
        params.update(custom_override)

    status, data = nb_get("/distancematrix/json", params=params)
    return status, data

def directions_multi_stop(order_latlng: List[str], params_extra: Dict[str, Any], custom_override: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # Correct endpoint: /directions/json (GET)
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

    if custom_override:
        params.update(custom_override)

    status, data = nb_get("/directions/json", params=params)
    return status, data

def vrp_create_v2(stops: pd.DataFrame, objective: str, global_mode: str, traffic_ts: Optional[int], override_body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    """
    Correct VRP v2 payload (per docs):
      - locations: ["lat,lng", ...]
      - jobs: [{"id":0,"location_index":0}, ...]
      - vehicles: [{"id":"vehicle_1","start_index":0,"end_index":0}]
      - options.objective: {"travel_cost":"distance"|"duration"}
    """
    if override_body:
        body = override_body
    else:
        if len(stops) < 2:
            return 0, {"error": "Need at least 2 stops"}

        locs = [latlng_str(float(r["lat"]), float(r["lng"])) for _, r in stops.reset_index(drop=True).iterrows()]
        jobs = [{"id": int(i), "location_index": int(i)} for i in range(len(locs))]
        vehicles = [{"id": "vehicle_1", "start_index": 0, "end_index": 0}]

        body: Dict[str, Any] = {
            "locations": locs,
            "jobs": jobs,
            "vehicles": vehicles,
            "options": {
                "objective": {"travel_cost": objective},
                "routing": {"mode": global_mode},
            },
        }
        if traffic_ts is not None:
            body["options"]["routing"]["traffic_timestamp"] = int(traffic_ts)

    params = {"key": NB_API_KEY}
    status, data = nb_post("/optimization/v2", params=params, body=body)

    st.session_state.last_json["vrp_create_payload"] = body
    st.session_state.last_json["vrp_create_response"] = data
    return status, data

def vrp_result_v2(job_id: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "id": job_id}
    status, data = nb_get("/optimization/v2/result", params=params)
    st.session_state.last_json["vrp_result_response"] = data
    return status, data

def parse_vrp_order(resp: Any) -> Optional[List[int]]:
    if not isinstance(resp, dict):
        return None

    # common containers
    routes = resp.get("routes") or resp.get("result", {}).get("routes") or resp.get("solution", {}).get("routes")
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list) or not isinstance(routes[0], dict):
        return None

    r0 = routes[0]

    # Steps/activities often contain location_index
    steps = r0.get("steps") or r0.get("activities") or r0.get("stops") or []
    order: List[int] = []
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict):
                continue
            li = s.get("location_index")
            if li is None:
                li = s.get("locationIndex")
            if li is None and isinstance(s.get("location"), dict):
                li = s["location"].get("index")
            if li is not None:
                try:
                    order.append(int(li))
                except Exception:
                    pass

    # De-dup preserving order
    if order:
        seen = set()
        out = []
        for x in order:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return None

def snap_to_roads(path_points: List[str], mode: str, geometry: str, override_body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # /snapToRoads/json supports GET (path param) or POST (body has "path")
    params = {"key": NB_API_KEY}
    if override_body is not None:
        status, data = nb_post("/snapToRoads/json", params=params, body=override_body)
        st.session_state.last_json["snap_payload"] = override_body
        st.session_state.last_json["snap_response"] = data
        return status, data

    path_str = "|".join(path_points)
    # prefer GET for <100 points
    get_params = {"key": NB_API_KEY, "path": path_str, "mode": mode, "geometry": geometry}
    status, data = nb_get("/snapToRoads/json", params=get_params)
    st.session_state.last_json["snap_params"] = get_params
    st.session_state.last_json["snap_response"] = data
    return status, data

def isochrone(center_latlng: str, mode: str, contours_minutes: Optional[str], contours_meters: Optional[str], polygons: bool, override_params: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # /isochrone/json is GET with parameters
    params: Dict[str, Any] = {"key": NB_API_KEY, "coordinates": center_latlng, "mode": mode, "polygons": "true" if polygons else "false"}
    if contours_minutes:
        params["contours_minutes"] = contours_minutes
    if contours_meters:
        params["contours_meters"] = contours_meters
    if override_params:
        params.update(override_params)

    status, data = nb_get("/isochrone/json", params=params)
    st.session_state.last_json["iso_params"] = params
    st.session_state.last_json["iso_response"] = data
    return status, data

# -----------------------------
# UI
# -----------------------------
ensure_state()
ss = st.session_state

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

# Stops editor sidebar
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
            rows = []
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

tabs = st.tabs([
    "Geocode & Map",
    "Places (Search + Generate Stops)",
    "Route + Optimize (Before vs After)",
    "Distance Matrix (NxN)",
    "Snap-to-Road + Isochrone",
])

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

    if st.button("🧭 Geocode all missing coordinates (cached)", key="geo_geocode_btn", use_container_width=True):
        missing = df[df["lat"].isna() | df["lng"].isna()].copy()
        resp_last = None
        for idx, row in missing.iterrows():
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            status, resp = geocode_forward(addr, country=ss.center["country"], language=language)
            resp_last = resp
            candidates = extract_geocode_candidates(resp)
            if candidates:
                top = candidates[0]
                df.loc[idx, "lat"] = top["lat"]
                df.loc[idx, "lng"] = top["lng"]
                df.loc[idx, "source"] = f"Geocoded ({status})"
            else:
                df.loc[idx, "source"] = f"Geocode failed ({status})"

        replace_stops(df)
        ss.last_json["geocode_response_sample"] = resp_last
        ss.mapsig["geocode"] += 1

    st.dataframe(ss.stops_df, use_container_width=True, height=220)

    base_center = (float(ss.center["lat"]), float(ss.center["lng"]))
    m = make_map(center=base_center, stops=ss.stops_df, clicked=ss.clicked_pin)
    map_ret = render_map(m, key=f"map_geocode_{ss.mapsig['geocode']}")

    if map_ret and map_ret.get("last_clicked"):
        lc = map_ret["last_clicked"]
        lat_f, lng_f = normalize_latlng(lc.get("lat"), lc.get("lng"))
        if lat_f is not None:
            ss.clicked_pin = (lat_f, lng_f)
            set_center(lat_f, lng_f)
            ss.mapsig["geocode"] += 1

    with st.expander("Download JSON (debug)", expanded=False):
        payload = {"stops": ss.stops_df.fillna("").to_dict(orient="records")}
        st.download_button(
            "Download Stops JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name="stops.json",
            mime="application/json",
            key="dl_stops_json",
        )

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
        n_rand = st.number_input("How many stops?", min_value=2, max_value=60, value=20, step=1, key="pl_n_rand")
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

    st.divider()
    st.markdown("### Optional: POI keyword search (best-effort) and add results as stops")

    kw = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="", key="pl_kw")
    kw_radius = st.slider("Search radius (m)", min_value=500, max_value=50000, value=5000, step=500, key="pl_kw_radius")
    kw_max = st.slider("Max results", min_value=1, max_value=50, value=10, step=1, key="pl_kw_max")

    if st.button("Search Places (POIs)", key="pl_search_pois", use_container_width=True):
        q = f"{kw}".strip()
        if not q:
            st.warning("Enter a keyword (or use Random Stops above).")
        else:
            params = {
                "key": NB_API_KEY,
                "q": q,
                "country": ss.center["country"],
                "language": language,
                "at": latlng_str(float(ss.center["lat"]), float(ss.center["lng"])),
                "radius": int(kw_radius),
                "limit": int(kw_max),
            }
            status, resp = nb_get("/geocode", params=params)
            if status == 404:
                status, resp = nb_get("/geocode/v1", params=params)
            ss.last_json["places_response"] = resp
            st.success(f"Places response: HTTP {status}")

            items = extract_places_items(resp)
            if not items:
                st.warning("No items found (or schema differed). Try increasing radius or changing keyword.")
            else:
                add_rows = [{"label": f"POI {i}", "address": it["name"], "lat": it["lat"], "lng": it["lng"]} for i, it in enumerate(items[: int(kw_max)], start=1)]
                add_stops(add_rows, f"POI search: {kw}")
                ss.mapsig["places"] += 1

    with st.expander("Download JSON (debug)", expanded=False):
        st.download_button(
            "Download Places JSON",
            data=json.dumps(ss.last_json.get("places_response", {}), indent=2).encode("utf-8"),
            file_name="places.json",
            mime="application/json",
            key="dl_places_json",
        )

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

            directions_override_text = st.text_area(
                "Optional: Directions query override (JSON). Leave {} for none.",
                value="{}",
                height=90,
                key="dir_before_override",
            )
            dir_override = safe_json_loads(directions_override_text) or {}

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", use_container_width=True):
                stt, resp = directions_multi_stop(stops_ll, params_extra=GLOBAL_PARAMS, custom_override=dir_override)
                ss.route_before["resp"] = resp
                ss.last_json["directions_before"] = resp

                # totals from directions first
                dist_m, dur_s = extract_directions_totals(resp)
                ss.route_before["distance_m"] = dist_m
                ss.route_before["duration_s"] = dur_s

                ss.route_before["geometry"] = extract_route_geometry(resp)
                ss.mapsig["route_before"] += 1

                if stt != 200:
                    st.error(f"Directions failed: HTTP {stt}")
                    st.json(resp)

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

            st.caption("Optional: Custom VRP request body override (JSON). Leave {} to use generated payload.")
            vrp_override_text = st.text_area("VRP override JSON", value="{}", height=140, key="vrp_override")
            vrp_override = safe_json_loads(vrp_override_text)
            if vrp_override_text.strip() and vrp_override is None:
                st.warning("VRP override JSON is invalid. Using generated payload.")

            traffic_ts = int(dep_unix) if (traffic == "Yes") else None

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", use_container_width=True):
                status_c, data_c = vrp_create_v2(
                    ss.stops_df,
                    objective=objective,
                    global_mode=mode,
                    traffic_ts=traffic_ts,
                    override_body=vrp_override if isinstance(vrp_override, dict) else None,
                )

                ss.vrp["create"] = data_c
                if status_c != 200:
                    st.error(f"VRP create failed: HTTP {status_c}")
                    st.json(data_c)
                else:
                    job_id = None
                    if isinstance(data_c, dict):
                        job_id = data_c.get("id") or data_c.get("job_id") or data_c.get("jobId")
                    ss.vrp["job_id"] = job_id
                    st.success(f"VRP job created. job_id={job_id}")

            if st.button("🔁 Fetch VRP result again", key="vrp_fetch_again", use_container_width=True):
                job_id = ss.vrp.get("job_id")
                if not job_id:
                    st.warning("No job_id available yet. Run optimization first.")
                else:
                    result = None
                    last = None
                    # poll for up to ~45s
                    for _ in range(15):
                        stt, rr = vrp_result_v2(str(job_id))
                        last = rr
                        if stt == 200 and isinstance(rr, dict):
                            # Some responses include a "status" field; if present, wait until not "processing"
                            st_status = (rr.get("status") or rr.get("result", {}).get("status") or "").lower()
                            if st_status and ("process" in st_status or "running" in st_status or "queued" in st_status):
                                time.sleep(3)
                                continue
                            result = rr
                            break
                        time.sleep(3)

                    ss.vrp["result"] = result or last
                    if result is None:
                        st.warning("VRP result not ready yet (or schema differs). Showing latest response:")
                        st.json(last)
                    else:
                        order = parse_vrp_order(result)
                        ss.vrp["order"] = order
                        if order:
                            df2 = ss.stops_df.copy().reset_index(drop=True)
                            valid = [i for i in order if 0 <= i < len(df2)]
                            missing = [i for i in range(len(df2)) if i not in valid]
                            final_order = valid + missing
                            ss._optimized_stops = df2.iloc[final_order].reset_index(drop=True)
                            st.success("Optimization order parsed and applied.")
                            ss.mapsig["route_after"] += 1
                        else:
                            st.warning("Optimization result did not include an order we could parse. Check VRP result JSON below.")
                            st.json(result)

            st.markdown("### Step 3 — Recompute Directions (After)")

            df_after = ss.get("_optimized_stops", None)
            if df_after is None:
                st.info("Run optimization to generate an optimized order.")
                df_after = ss.stops_df.copy()

            directions_after_override_text = st.text_area(
                "Optional: Directions (After) query override (JSON). Leave {} for none.",
                value="{}",
                height=90,
                key="dir_after_override",
            )
            dir_after_override = safe_json_loads(directions_after_override_text) or {}

            if st.button("🧭 Compute route (After)", key="rt_after_btn", use_container_width=True):
                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]
                stt2, resp2 = directions_multi_stop(ll_after, params_extra=GLOBAL_PARAMS, custom_override=dir_after_override)
                ss.route_after["resp"] = resp2
                ss.last_json["directions_after"] = resp2

                dist_m2, dur_s2 = extract_directions_totals(resp2)
                ss.route_after["distance_m"] = dist_m2
                ss.route_after["duration_s"] = dur_s2

                ss.route_after["geometry"] = extract_route_geometry(resp2)
                ss.mapsig["route_after"] += 1

                if stt2 != 200:
                    st.error(f"Directions (After) failed: HTTP {stt2}")
                    st.json(resp2)

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

        with st.expander("Download JSON (debug)", expanded=False):
            bundle = {
                "directions_before": ss.last_json.get("directions_before"),
                "directions_after": ss.last_json.get("directions_after"),
                "vrp_create_payload": ss.last_json.get("vrp_create_payload"),
                "vrp_create_response": ss.last_json.get("vrp_create_response"),
                "vrp_result_response": ss.last_json.get("vrp_result_response"),
            }
            st.download_button(
                "Download Route+Optimize JSON bundle",
                data=json.dumps(bundle, indent=2).encode("utf-8"),
                file_name="route_optimize_bundle.json",
                mime="application/json",
                key="dl_bundle",
            )

# -----------------------------
# TAB 4: DISTANCE MATRIX (NxN)
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

        st.caption("Optional: Matrix query override (JSON). Example: {\"mode\":\"truck\",\"option\":\"flexible\"}")
        mx_override_text = st.text_area("Matrix override JSON", value="{}", height=90, key="mx_override")
        mx_override = safe_json_loads(mx_override_text) or {}

        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", use_container_width=True):
            status, data = distance_matrix(ll, ll, params_extra=GLOBAL_PARAMS, custom_override=mx_override)
            ss.last_json["matrix_nxn"] = data
            ss.mapsig["matrix"] += 1
            if status != 200:
                st.error(f"Matrix failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Matrix computed (cached).")

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

        with st.expander("Download JSON (debug)", expanded=False):
            st.download_button(
                "Download Matrix JSON",
                data=json.dumps(ss.last_json.get("matrix_nxn", {}), indent=2).encode("utf-8"),
                file_name="distance_matrix.json",
                mime="application/json",
                key="dl_matrix_json",
            )

# -----------------------------
# TAB 5: SNAP-TO-ROAD + ISOCHRONE
# -----------------------------
with tabs[4]:
    st.subheader("Snap-to-Road + Isochrone (stable UI; runs only on button click)")

    s1, s2 = st.columns([1, 1])

    with s1:
        st.markdown("### Snap-to-Roads (Snap To Roads API)")

        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use Directions (After) geometry"], index=0, key="snap_src")
        geom = ss.route_before.get("geometry") if src.startswith("Use Directions (Before)") else ss.route_after.get("geometry")
        coords = geom or []

        max_pts = len(coords) if coords else 0
        if max_pts < 2:
            st.caption("Not enough geometry points yet. Compute a route first.")
            n_path = 0
        else:
            n_path = st.slider("N geometry points to send (first N)", min_value=2, max_value=min(200, max_pts), value=min(100, max_pts), step=1, key="snap_n")

        snap_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="snap_mode")
        snap_geometry = st.selectbox("Geometry format", ["polyline", "geojson"], index=0, key="snap_geom_fmt")

        st.caption("Optional: Snap POST override body (JSON). Leave {} to use GET with path param.")
        snap_override_text = st.text_area("Snap override JSON", value="{}", height=120, key="snap_override")
        snap_override = safe_json_loads(snap_override_text)
        if snap_override_text.strip() and snap_override is None:
            st.warning("Snap override JSON is invalid. Using generated request.")

        if st.button("🧷 Snap to Roads", key="snap_btn", use_container_width=True):
            if isinstance(snap_override, dict) and snap_override != {}:
                status, data = snap_to_roads([], mode=snap_mode, geometry=snap_geometry, override_body=snap_override)
            else:
                path_points = [latlng_str(p[0], p[1]) for p in (coords[: int(n_path)] if n_path else [])]
                status, data = snap_to_roads(path_points, mode=snap_mode, geometry=snap_geometry, override_body=None)

            ss.mapsig["snap"] += 1
            if status != 200:
                st.error(f"Snap-to-roads failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Snap-to-roads computed.")

        # Render snapped geometry if present
        snap_resp = ss.last_json.get("snap_response")
        snap_route_pts = None
        if isinstance(snap_resp, dict):
            g = snap_resp.get("geometry")
            if isinstance(g, str) and g.strip():
                snap_route_pts = decode_polyline(g, precision=5) or decode_polyline(g, precision=6)
            elif isinstance(g, dict) and g.get("type") == "FeatureCollection":
                # draw via extract_route_geometry
                snap_route_pts = extract_route_geometry(g)

        m5 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            route_pts=snap_route_pts if snap_route_pts else (coords[: int(n_path)] if (coords and n_path) else None),
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (Isochrone API)")

        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")
        iso_kind = st.selectbox("Contour type", ["minutes", "meters"], index=0, key="iso_kind")

        # Higher limits + optional custom string list (e.g., "5,10,15")
        if iso_kind == "minutes":
            iso_val = st.number_input("Single contour minutes (quick)", min_value=1, max_value=6000, value=30, step=1, key="iso_val_min")
            custom_list = st.text_input("Or custom contours_minutes list (comma-separated)", value="", key="iso_custom_min")
        else:
            iso_val = st.number_input("Single contour meters (quick)", min_value=100, max_value=5_000_000, value=5000, step=100, key="iso_val_m")
            custom_list = st.text_input("Or custom contours_meters list (comma-separated)", value="", key="iso_custom_m")

        polygons = st.selectbox("Return polygons", YESNO, index=1, key="iso_polys")

        st.caption("Optional: Isochrone query override (JSON). Example: {\"denoise\":1,\"generalize\":50}")
        iso_override_text = st.text_area("Isochrone override JSON", value="{}", height=100, key="iso_override")
        iso_override = safe_json_loads(iso_override_text) or {}

        if st.button("🟦 Compute Isochrone", key="iso_btn", use_container_width=True):
            center_ll = latlng_str(float(ss.center["lat"]), float(ss.center["lng"]))
            if iso_kind == "minutes":
                cm = custom_list.strip() if custom_list.strip() else str(int(iso_val))
                status, data = isochrone(center_ll, mode=iso_mode, contours_minutes=cm, contours_meters=None, polygons=(polygons == "Yes"), override_params=iso_override)
            else:
                cm = custom_list.strip() if custom_list.strip() else str(int(iso_val))
                status, data = isochrone(center_ll, mode=iso_mode, contours_minutes=None, contours_meters=cm, polygons=(polygons == "Yes"), override_params=iso_override)

            ss.mapsig["iso"] += 1
            if status != 200:
                st.error(f"Isochrone failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Isochrone computed.")

        iso_resp = ss.last_json.get("iso_response")
        polys = extract_isochrone_polygons(iso_resp)

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            polygons=polys if polys else None,
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m6, key=f"map_iso_{ss.mapsig['iso']}")

    with st.expander("Download JSON (debug)", expanded=False):
        bundle = {
            "snap_params": ss.last_json.get("snap_params"),
            "snap_payload": ss.last_json.get("snap_payload"),
            "snap_response": ss.last_json.get("snap_response"),
            "iso_params": ss.last_json.get("iso_params"),
            "iso_response": ss.last_json.get("iso_response"),
        }
        st.download_button(
            "Download Snap+Iso JSON",
            data=json.dumps(bundle, indent=2).encode("utf-8"),
            file_name="snap_iso.json",
            mime="application/json",
            key="dl_snap_iso",
        )

# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# Fixes: disappearing maps, duplicate element IDs, slider min=max crash, stable “click pin”, numbered stops,
#        distance/duration totals = 0 bug, VRP v2 payload format (locations object), safer polyline decode,
#        directions chunking, route outline + arrows, isochrone outline render, snap-to-road payload sizing.
#
# Run: streamlit run app.py

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

# -----------------------------
# HTTP (button-only + caching)
# -----------------------------
@dataclass(frozen=True)
class ReqSig:
    method: str
    path: str
    params_json: str
    body_json: str

def _sig(method: str, path: str, params: Dict[str, Any] | None, body: Dict[str, Any] | None) -> ReqSig:
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
            r = requests.get(url, params=params, timeout=90)
        else:
            r = requests.post(url, params=params, json=body, timeout=180)

        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, {"error": str(e)}

def nb_get(path: str, params: Dict[str, Any]) -> Tuple[int, Any]:
    return _cached_request(_sig("GET", path, params=params, body=None))

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any]) -> Tuple[int, Any]:
    return _cached_request(_sig("POST", path, params=params, body=body))

# -----------------------------
# PARSERS (robust)
# -----------------------------
def extract_geocode_candidates(resp: Any) -> List[Dict[str, Any]]:
    """
    Try multiple shapes:
    - list[dict] with lat/lon
    - {"items": [...]}
    - {"features":[{"geometry":{"coordinates":[lng,lat]}, "properties":{...}}]}
    - {"results":[...]}
    """
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

def safe_decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    """
    Safe Google encoded polyline decoder.
    Prevents IndexError when API returns unexpected strings.
    """
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
            # --- decode lat
            result = 0
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(s[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            # --- decode lng
            result = 0
            shift = 0
            while True:
                if index >= length:
                    return coordinates
                b = ord(s[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            coordinates.append((lat / factor, lng / factor))
    except Exception:
        return coordinates

    return coordinates

def extract_route_geometry(resp: Any) -> List[Tuple[float, float]]:
    """
    Pull route geometry from various response shapes.
    Tries polyline6 and polyline5 and GeoJSON.
    """
    if not isinstance(resp, dict):
        return []

    # GeoJSON
    if resp.get("type") == "FeatureCollection" and "features" in resp:
        for f in resp["features"]:
            geom = (f or {}).get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            if gtype == "LineString" and isinstance(coords, list):
                pts = []
                for c in coords:
                    if isinstance(c, list) and len(c) >= 2:
                        lng, lat = c[0], c[1]
                        lat_f, lng_f = normalize_latlng(lat, lng)
                        if lat_f is not None:
                            pts.append((lat_f, lng_f))
                if pts:
                    return pts

    routes = resp.get("routes") or resp.get("route") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    r0 = routes[0] if isinstance(routes[0], dict) else {}

    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")
    if isinstance(poly, str) and poly.strip():
        # try both: NB sometimes uses polyline6
        pts6 = safe_decode_polyline(poly, precision=6)
        pts5 = safe_decode_polyline(poly, precision=5)
        if pts6 and (len(pts6) >= len(pts5)):
            return pts6
        return pts5

    # legs/steps
    legs = r0.get("legs") or []
    if isinstance(legs, dict):
        legs = [legs]

    merged: List[Tuple[float, float]] = []
    for leg in legs:
        steps = (leg or {}).get("steps") or []
        for stp in steps:
            g = (stp or {}).get("geometry")
            if isinstance(g, str) and g.strip():
                pts6 = safe_decode_polyline(g, precision=6)
                pts5 = safe_decode_polyline(g, precision=5)
                seg = pts6 if pts6 and len(pts6) >= len(pts5) else pts5
                if seg:
                    if merged and seg and merged[-1] == seg[0]:
                        merged.extend(seg[1:])
                    else:
                        merged.extend(seg)
    return merged

def extract_iso_geometry(resp: Any) -> List[List[Tuple[float, float]]]:
    """
    Returns list of rings/lines: each is [(lat,lng),...]
    Supports FeatureCollection, Polygon/MultiPolygon/LineString.
    """
    out: List[List[Tuple[float, float]]] = []
    if not isinstance(resp, dict):
        return out

    feats = resp.get("features")
    if not isinstance(feats, list):
        return out

    for f in feats:
        geom = (f or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        def to_latlng_list(line_coords):
            pts = []
            for c in line_coords:
                if isinstance(c, list) and len(c) >= 2:
                    lng, lat = c[0], c[1]
                    lat_f, lng_f = normalize_latlng(lat, lng)
                    if lat_f is not None:
                        pts.append((lat_f, lng_f))
            return pts

        if gtype == "LineString" and isinstance(coords, list):
            pts = to_latlng_list(coords)
            if pts:
                out.append(pts)

        elif gtype == "Polygon" and isinstance(coords, list):
            # coords = [ring1, ring2...]
            for ring in coords:
                if isinstance(ring, list):
                    pts = to_latlng_list(ring)
                    if pts:
                        out.append(pts)

        elif gtype == "MultiPolygon" and isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list):
                    for ring in poly:
                        if isinstance(ring, list):
                            pts = to_latlng_list(ring)
                            if pts:
                                out.append(pts)

    return out

# -----------------------------
# MAP RENDERING (stable)
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    clicked: Optional[Tuple[float, float]] = None,
    iso_shapes: Optional[List[List[Tuple[float, float]]]] = None,
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")

    # Click marker
    if clicked is not None:
        folium.Marker(
            location=[clicked[0], clicked[1]],
            popup="Pinned point",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    # Stops (numbered)
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

    # Isochrone shapes
    if iso_shapes:
        for pts in iso_shapes:
            if len(pts) < 2:
                continue
            closed = (pts[0] == pts[-1])
            if closed and len(pts) >= 4:
                folium.Polygon(pts, weight=3, opacity=0.9, fill=True, fill_opacity=0.15).add_to(m)
            else:
                folium.PolyLine(pts, weight=3, opacity=0.9).add_to(m)

    # Route polyline + arrow
    if route_pts and len(route_pts) >= 2:
        folium.PolyLine(route_pts, weight=5, opacity=0.85).add_to(m)
        try:
            pl = folium.PolyLine(route_pts, weight=0, opacity=0)
            pl.add_to(m)
            PolyLineTextPath(
                pl, "  ➤  ", repeat=True, offset=7,
                attributes={"font-size": "14", "fill": "black"}
            ).add_to(m)
        except Exception:
            pass

        # fit bounds
        lats = [p[0] for p in route_pts]
        lngs = [p[1] for p in route_pts]
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
    if "_optimized_stops" not in ss:
        ss._optimized_stops = None
    if "_region_candidates" not in ss:
        ss._region_candidates = []

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
    st.session_state._optimized_stops = None

# -----------------------------
# API BUILDERS
# -----------------------------
def build_global_route_params(ui: Dict[str, Any]) -> Dict[str, Any]:
    p = {"key": NB_API_KEY}
    if ui.get("language"):
        p["language"] = ui["language"]
    if ui.get("departure_time") is not None:
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
    if ui.get("overview") is not None:
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

def distance_matrix(origins: List[str], destinations: List[str], params_extra: Dict[str, Any]) -> Tuple[int, Any]:
    params = dict(params_extra)
    params["origins"] = "|".join(origins)
    params["destinations"] = "|".join(destinations)
    return nb_get("/distancematrix/v2", params=params)

def sum_pair_matrix(diagonal_matrix_resp: Any) -> Tuple[float, float]:
    """
    For pair totals we send origins[i] -> destinations[i]
    Matrix returns rows[i].elements[j] for all j; we need the diagonal element [i][i].
    """
    dist_total = 0.0
    dur_total = 0.0

    if not isinstance(diagonal_matrix_resp, dict):
        return dist_total, dur_total

    rows = diagonal_matrix_resp.get("rows") or []
    for i, row in enumerate(rows):
        elems = (row or {}).get("elements") or []
        if i < len(elems):
            e = elems[i]  # <-- FIXED (was elems[0])
            dist_total += float((e.get("distance") or {}).get("value") or 0)
            dur_total += float((e.get("duration") or {}).get("value") or 0)

    return dist_total, dur_total

def directions_multi_stop(order_latlng: List[str], params_extra: Dict[str, Any]) -> Tuple[int, Any]:
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

    # Try common endpoints
    status, data = nb_get("/directions/v2", params=params)
    if status == 404:
        status, data = nb_get("/navigation/v2", params=params)
    if status == 404:
        status, data = nb_get("/directions", params=params)
    return status, data

def directions_multi_stop_chunked(order_latlng: List[str], params_extra: Dict[str, Any], max_points_per_call: int = 25) -> Tuple[int, Dict[str, Any]]:
    """
    Directions APIs often cap waypoints. For long lists, chunk into overlapping segments and merge geometry.
    Returns: { "responses": [...], "merged": <best first response>, "merged_geometry":[...] }
    """
    if len(order_latlng) <= max_points_per_call:
        stt, resp = directions_multi_stop(order_latlng, params_extra)
        return stt, {"responses": [resp], "merged": resp, "merged_geometry": extract_route_geometry(resp)}

    responses = []
    merged_geom: List[Tuple[float, float]] = []
    final_status = 0

    step = max_points_per_call - 1  # overlap last point
    start = 0
    while start < len(order_latlng) - 1:
        end = min(start + max_points_per_call, len(order_latlng))
        chunk = order_latlng[start:end]
        stt, resp = directions_multi_stop(chunk, params_extra)
        responses.append(resp)
        final_status = stt

        if stt == 200 and isinstance(resp, dict):
            seg = extract_route_geometry(resp)
            if seg:
                if merged_geom and seg and merged_geom[-1] == seg[0]:
                    merged_geom.extend(seg[1:])
                else:
                    merged_geom.extend(seg)

        if end >= len(order_latlng):
            break
        start = end - 1

    merged = responses[0] if responses else {}
    return final_status, {"responses": responses, "merged": merged, "merged_geometry": merged_geom}

def vrp_create_v2(stops: pd.DataFrame, objective: str) -> Tuple[int, Any, Dict[str, Any]]:
    """
    Correct VRP v2 payload format:
      locations: { "id": <int>, "location": ["lat,lng", ...] }
      jobs: [{id:<int>, location_index:<int>}, ...]
      vehicles: [{id:"vehicle_1", start_index:0, end_index:0}]
    """
    if len(stops) < 2:
        return 0, {"error": "Need at least 2 stops"}, {}

    loc_list = [latlng_str(float(r["lat"]), float(r["lng"])) for _, r in stops.reset_index(drop=True).iterrows()]

    body = {
        "locations": {
            "id": 1,
            "location": loc_list
        },
        "jobs": [{"id": int(i + 1), "location_index": int(i)} for i in range(len(loc_list))],
        "vehicles": [{"id": "vehicle_1", "start_index": 0, "end_index": 0}],
        "options": {"objective": {"travel_cost": objective}},  # distance | duration
    }

    params = {"key": NB_API_KEY}
    status, data = nb_post("/optimization/v2", params=params, body=body)
    return status, data, body

def vrp_result_v2(job_id: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "id": job_id}
    return nb_get("/optimization/v2/result", params=params)

def parse_vrp_order(resp: Any) -> Optional[List[int]]:
    """
    Flexible order extraction:
    - result.routes[0].steps[*].location_index
    - result.routes[0].activities[*].location_index
    """
    if not isinstance(resp, dict):
        return None

    result = resp.get("result") or {}
    routes = result.get("routes") or resp.get("routes")
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list):
        return None

    r0 = routes[0] if isinstance(routes[0], dict) else {}
    steps = r0.get("steps") or r0.get("activities") or []

    order: List[int] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        li = s.get("location_index", s.get("locationIndex"))
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
    return None

# -----------------------------
# Snap-to-Road + Isochrone
# -----------------------------
def snap_to_road(points: List[Tuple[float, float]]) -> Tuple[int, Any, Dict[str, Any]]:
    """
    Endpoint varies by deployment; we try common ones.
    We keep payload small to avoid 413.
    """
    body = {"points": [{"lat": p[0], "lng": p[1]} for p in points]}
    params = {"key": NB_API_KEY}

    for ep in ["/snapToRoad/v2", "/snaptoroad/v2", "/snapToRoad", "/snaptoroad"]:
        stt, resp = nb_post(ep, params=params, body=body)
        if stt != 404:
            return stt, resp, body
    return 404, {"error": "Snap endpoint not found"}, body

def isochrone_v2(center_lat: float, center_lng: float, iso_type: str, iso_val: int, mode: str) -> Tuple[int, Any, Dict[str, Any]]:
    params = {"key": NB_API_KEY}
    body = {
        "center": {"lat": float(center_lat), "lng": float(center_lng)},
        "type": iso_type,
        "value": int(iso_val),
        "mode": mode,
    }
    # try v2 then fallback
    stt, resp = nb_post("/isochrone/v2", params=params, body=body)
    if stt == 404:
        stt, resp = nb_post("/isochrone", params=params, body=body)
    return stt, resp, body

# -----------------------------
# UI
# -----------------------------
ensure_state()
ss = st.session_state

st.title("NextBillion.ai — Visual API Tester")
st.caption(f"Stops loaded: {len(ss.stops_df)}")

# Global route options
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

    avoid_options = [
        "toll", "highway", "ferry", "indoor", "unpaved", "tunnel",
        "sharp_turn", "u_turn", "service_road"
    ]
    avoid = st.multiselect("Avoid", avoid_options, default=[], key="g_avoid")

    dep_col1, dep_col2 = st.columns([1, 1])
    with dep_col1:
        dep_unix = st.number_input(
            "Departure time (unix seconds)",
            min_value=0,
            value=_now_unix(),
            step=60,
            key="g_dep_unix",
        )
    with dep_col2:
        st.write("Human readable")
        st.code(unix_to_human(int(dep_unix)))

    overview = st.selectbox("Overview", ["full", "simplified", "false"], index=0, key="g_overview")

    # Optional: advanced/custom query params
    custom_params_text = st.text_area(
        "Custom query params (JSON) — merged into request params",
        value="{}",
        height=90,
        key="g_custom_params",
        help='Example: {"units":"metric","someFlag":"true"}'
    )
    try:
        custom_params = json.loads(custom_params_text or "{}")
        if not isinstance(custom_params, dict):
            custom_params = {}
    except Exception:
        custom_params = {}
        st.warning("Custom params JSON invalid — ignoring.")

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
GLOBAL_PARAMS.update(custom_params)

# Stops editor sidebar
with st.sidebar:
    st.header("Config")
    st.text_input("NextBillion API Key", value=NB_API_KEY, type="password", key="sb_key", disabled=True)

    st.divider()
    st.subheader("Stops (paste / edit)")

    input_mode = st.radio(
        "How to input stops?",
        ["Addresses (one per line)", "Lat/Lng (one per line: lat,lng)"],
        index=0,
        key="sb_input_mode",
    )
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
                        "source": "Pasted address"
                    }])], ignore_index=True)
                ss.stops_df = df.reset_index(drop=True)
            else:
                rows = []
                for ll in lines:
                    parts = [p.strip() for p in ll.split(",")]
                    if len(parts) >= 2:
                        lat_f, lng_f = normalize_latlng(parts[0], parts[1])
                        if lat_f is not None:
                            rows.append({"label": f"Stop {len(ss.stops_df)+len(rows)+1}", "address": "", "lat": lat_f, "lng": lng_f})
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

    if st.button("🧭 Geocode all missing coordinates (cached)", key="geo_geocode_btn", width="stretch"):
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

    st.dataframe(ss.stops_df, width="stretch", height=220)

    m = make_map(
        center=(float(ss.center["lat"]), float(ss.center["lng"])),
        stops=ss.stops_df,
        clicked=ss.clicked_pin,
    )
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

    m2 = make_map(
        center=(float(ss.center["lat"]), float(ss.center["lng"])),
        stops=ss.stops_df,
        clicked=ss.clicked_pin
    )
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

    if st.button("🎲 Generate random stops around center", key="pl_gen_random", width="stretch"):
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
    kw_radius = st.slider("Search radius (m)", min_value=500, max_value=200000, value=5000, step=500, key="pl_kw_radius")
    kw_max = st.slider("Max results", min_value=1, max_value=50, value=10, step=1, key="pl_kw_max")

    if st.button("Search Places (POIs)", key="pl_search_pois", width="stretch"):
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
                add_rows = [{
                    "label": f"POI {i}",
                    "address": it["name"],
                    "lat": it["lat"],
                    "lng": it["lng"],
                } for i, it in enumerate(items[: int(kw_max)], start=1)]
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

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", width="stretch"):
                origins = stops_ll[:-1]
                dests = stops_ll[1:]
                status_m, data_m = distance_matrix(origins, dests, params_extra=GLOBAL_PARAMS)
                ss.last_json["matrix_pairs_before_raw"] = data_m

                dist_total, dur_total = sum_pair_matrix(data_m)
                ss.route_before["distance_m"] = dist_total
                ss.route_before["duration_s"] = dur_total

                status_d, pack = directions_multi_stop_chunked(stops_ll, params_extra=GLOBAL_PARAMS, max_points_per_call=25)
                ss.last_json["directions_before_raw"] = pack
                merged = pack["merged"] if isinstance(pack, dict) else pack
                ss.route_before["resp"] = merged
                ss.route_before["geometry"] = pack.get("merged_geometry", []) if isinstance(pack, dict) else extract_route_geometry(merged)

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

            # Optional: custom VRP JSON override (advanced)
            vrp_override = st.text_area(
                "Optional: Custom VRP request body override (JSON). Leave {} to use generated payload.",
                value="{}",
                height=120,
                key="vrp_override",
            )
            try:
                vrp_override_obj = json.loads(vrp_override or "{}")
                if not isinstance(vrp_override_obj, dict):
                    vrp_override_obj = {}
            except Exception:
                vrp_override_obj = {}
                st.warning("VRP override JSON invalid — ignoring.")

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", width="stretch"):
                status_c, data_c, payload = vrp_create_v2(ss.stops_df, objective=objective)

                # apply override if provided
                if vrp_override_obj:
                    payload = vrp_override_obj
                    status_c, data_c = nb_post("/optimization/v2", params={"key": NB_API_KEY}, body=payload)

                ss.last_json["vrp_create_payload"] = payload
                ss.last_json["vrp_create_response"] = data_c

                if status_c != 200:
                    st.error(f"VRP create failed: HTTP {status_c}")
                    st.json(data_c)
                else:
                    ss.vrp["create"] = data_c
                    job_id = data_c.get("id") or data_c.get("job_id") or data_c.get("jobId")
                    ss.vrp["job_id"] = job_id

                    if not job_id:
                        st.warning("VRP create succeeded, but no job id found in response.")
                    else:
                        # Poll result (longer, because NB can take time)
                        result = None
                        last = None
                        for _ in range(30):
                            stt, rr = vrp_result_v2(str(job_id))
                            last = rr
                            if stt == 200 and isinstance(rr, dict):
                                # if result exists and has routes, break
                                r = rr.get("result") or {}
                                if (r.get("routes") or rr.get("routes")):
                                    result = rr
                                    break
                            time.sleep(1.0)

                        ss.vrp["result"] = result or last
                        ss.last_json["vrp_result_response"] = ss.vrp["result"]

                        order = parse_vrp_order(ss.vrp["result"])
                        ss.vrp["order"] = order

                        if order:
                            df2 = ss.stops_df.copy().reset_index(drop=True)
                            valid = [i for i in order if 0 <= i < len(df2)]
                            missing = [i for i in range(len(df2)) if i not in valid]
                            final_order = valid + missing
                            ss._optimized_stops = df2.iloc[final_order].reset_index(drop=True)
                            st.success("Optimization order computed.")
                            ss.mapsig["route_after"] += 1
                        else:
                            st.warning("Optimization result did not include an order yet. Try again using 'Fetch result again'.")

            if ss.vrp.get("job_id"):
                if st.button("🔁 Fetch VRP result again", key="vrp_fetch_again", width="stretch"):
                    stt, rr = vrp_result_v2(str(ss.vrp["job_id"]))
                    ss.vrp["result"] = rr
                    ss.last_json["vrp_result_response"] = rr
                    order = parse_vrp_order(rr)
                    ss.vrp["order"] = order
                    if order:
                        df2 = ss.stops_df.copy().reset_index(drop=True)
                        valid = [i for i in order if 0 <= i < len(df2)]
                        missing = [i for i in range(len(df2)) if i not in valid]
                        final_order = valid + missing
                        ss._optimized_stops = df2.iloc[final_order].reset_index(drop=True)
                        st.success("Optimization order computed.")
                        ss.mapsig["route_after"] += 1
                    else:
                        st.warning("Still no order in result.")

            st.markdown("### Step 3 — Recompute Directions (After)")

            df_after = ss._optimized_stops if ss._optimized_stops is not None else ss.stops_df.copy()

            if st.button("🧭 Compute route (After)", key="rt_after_btn", width="stretch"):
                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]

                status_m2, data_m2 = distance_matrix(ll_after[:-1], ll_after[1:], params_extra=GLOBAL_PARAMS)
                ss.last_json["matrix_pairs_after_raw"] = data_m2

                dist_total2, dur_total2 = sum_pair_matrix(data_m2)
                ss.route_after["distance_m"] = dist_total2
                ss.route_after["duration_s"] = dur_total2

                status_d2, pack2 = directions_multi_stop_chunked(ll_after, params_extra=GLOBAL_PARAMS, max_points_per_call=25)
                ss.last_json["directions_after_raw"] = pack2
                merged2 = pack2["merged"] if isinstance(pack2, dict) else pack2
                ss.route_after["resp"] = merged2
                ss.route_after["geometry"] = pack2.get("merged_geometry", []) if isinstance(pack2, dict) else extract_route_geometry(merged2)

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

            # Comparison
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
                "matrix_pairs_before_raw": ss.last_json.get("matrix_pairs_before_raw"),
                "matrix_pairs_after_raw": ss.last_json.get("matrix_pairs_after_raw"),
                "directions_before_raw": ss.last_json.get("directions_before_raw"),
                "directions_after_raw": ss.last_json.get("directions_after_raw"),
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
    st.subheader("Distance Matrix (NxN) — up to 200 points (API limits may apply)")

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
        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", width="stretch"):
            status, data = distance_matrix(ll, ll, params_extra=GLOBAL_PARAMS)
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
            st.dataframe(pd.DataFrame(dist_km).round(2), width="stretch", height=240)
            st.markdown("### Duration (min)")
            st.dataframe(pd.DataFrame(dur_min).round(1), width="stretch", height=240)

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
        st.markdown("### Snap-to-Road (best-effort)")

        src = st.selectbox(
            "Path source",
            ["Use Directions (Before) geometry", "Use Directions (After) geometry"],
            index=0,
            key="snap_src"
        )
        geom = ss.route_before.get("geometry") if src.startswith("Use Directions (Before)") else ss.route_after.get("geometry")
        coords = geom or []

        if not coords or len(coords) < 2:
            st.caption("Not enough geometry points yet. Compute a route first.")
            sample_n = 0
        else:
            # Hard cap to avoid 413
            hard_cap = min(300, len(coords))
            sample_n = st.slider(
                "How many geometry points to send (sampled to avoid 413)",
                min_value=2,
                max_value=hard_cap,
                value=min(120, hard_cap),
                step=1,
                key="snap_n",
            )

        # Sampling factor
        if coords and sample_n >= 2:
            step = max(1, int(len(coords) / sample_n))
            sampled = coords[::step]
            sampled = sampled[:sample_n]
        else:
            sampled = []

        if st.button("🧷 Snap to Road", key="snap_btn", width="stretch"):
            stt, resp, payload = snap_to_road(sampled)
            ss.last_json["snap"] = {"status": stt, "response": resp, "payload": payload}
            ss.mapsig["snap"] += 1

            if stt != 200:
                st.error(f"Snap-to-road failed: HTTP {stt}")
                st.write(resp)
            else:
                st.success("Snap-to-road computed (cached).")

        m5 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            route_pts=sampled if sampled else None,
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (best-effort)")

        iso_type = st.selectbox("Type", ["time", "distance"], index=0, key="iso_type")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")

        # increased max + optional custom
        if iso_type == "time":
            iso_val = st.number_input(
                "Value (seconds)",
                min_value=60,
                max_value=86400,  # 24h
                value=1800,
                step=60,
                key="iso_val_time",
            )
        else:
            iso_val = st.number_input(
                "Value (meters)",
                min_value=100,
                max_value=500000,  # 500 km
                value=10000,
                step=100,
                key="iso_val_dist",
            )

        if st.button("🟦 Compute Isochrone", key="iso_btn", width="stretch"):
            stt, resp, payload = isochrone_v2(float(ss.center["lat"]), float(ss.center["lng"]), iso_type, int(iso_val), iso_mode)
            ss.last_json["iso"] = {"status": stt, "response": resp, "payload": payload}
            ss.mapsig["iso"] += 1

            if stt != 200:
                st.error(f"Isochrone failed: HTTP {stt}")
                st.write(resp)
            else:
                st.success("Isochrone computed (cached).")

        iso_resp = (ss.last_json.get("iso") or {}).get("response")
        iso_shapes = extract_iso_geometry(iso_resp) if iso_resp else None

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            clicked=ss.clicked_pin,
            iso_shapes=iso_shapes,
            zoom=12
        )
        render_map(m6, key=f"map_iso_{ss.mapsig['iso']}")

    with st.expander("Download JSON (debug)", expanded=False):
        bundle = {"snap": ss.last_json.get("snap"), "iso": ss.last_json.get("iso")}
        st.download_button(
            "Download Snap+Iso JSON",
            data=json.dumps(bundle, indent=2).encode("utf-8"),
            file_name="snap_iso.json",
            mime="application/json",
            key="dl_snap_iso",
        )

# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# FIXES (this version):
# - Remove polyline dependency (fix ModuleNotFoundError)
# - Make PolyLineTextPath optional (fix folium.plugins import errors)
# - Replace use_container_width -> width="stretch" (Streamlit deprecation)
# - Fix Isochrone: use GET /isochrone/json with coordinates + contours_* (per docs)
# - Render isochrone GeoJSON on map
# - Places: Reverse geocode writes to address column reliably (dtype safe)
# - Snap-to-road: defensive parsing and UI-safe response display

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# Optional: some folium versions do not include this plugin
try:
    from folium.plugins import PolyLineTextPath  # type: ignore
except Exception:
    PolyLineTextPath = None  # arrows will be disabled

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


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def choose_latlng(lat: Any, lng: Any, ref: Tuple[float, float] | None = None) -> Tuple[Optional[float], Optional[float], bool]:
    """Return (lat, lng, swapped) choosing between (lat,lng) and (lng,lat).

    Many APIs return coordinates as [lng, lat]. This helper picks the more plausible
    pair using bounds checks and (optionally) proximity to a reference point.
    """
    lat1, lng1 = normalize_latlng(lat, lng)
    lat2, lng2 = normalize_latlng(lng, lat)

    v1 = lat1 is not None and lng1 is not None
    v2 = lat2 is not None and lng2 is not None

    if not v1 and not v2:
        return None, None, False
    if v1 and not v2:
        return lat1, lng1, False
    if not v1 and v2:
        return lat2, lng2, True

    if ref is not None:
        rlat, rlng = float(ref[0]), float(ref[1])
        d1 = haversine_km(lat1, lng1, rlat, rlng)
        d2 = haversine_km(lat2, lng2, rlat, rlng)
        if d2 + 1e-6 < d1:
            return lat2, lng2, True

    return lat1, lng1, False

def sanitize_stops_df(df: pd.DataFrame, ref: Tuple[float, float] | None = None) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return df, 0
    out = df.copy()
    swapped = 0
    for idx, row in out.iterrows():
        lat = row.get("lat")
        lng = row.get("lng")
        if pd.isna(lat) or pd.isna(lng):
            continue
        lat2, lng2, did_swap = choose_latlng(lat, lng, ref=ref)
        if lat2 is None:
            continue
        if did_swap:
            swapped += 1
            out.at[idx, "lat"] = lat2
            out.at[idx, "lng"] = lng2
    if "address" in out.columns:
        out["address"] = out["address"].astype("string")
    return out.reset_index(drop=True), swapped

def maybe_fix_session_stops(ref: Tuple[float, float] | None = None) -> int:
    df2, nsw = sanitize_stops_df(ss.stops_df, ref=ref)
    if nsw:
        ss.stops_df = df2
    return nsw

def fix_points_order(pts: List[Tuple[float, float]], ref: Tuple[float, float] | None = None) -> List[Tuple[float, float]]:
    if not pts or ref is None:
        return pts
    rlat, rlng = float(ref[0]), float(ref[1])

    def median_dist(points):
        ds = [haversine_km(p[0], p[1], rlat, rlng) for p in points[: min(len(points), 200)]]
        ds.sort()
        return ds[len(ds)//2] if ds else 0.0

    d1 = median_dist(pts)
    swapped_pts = [(p[1], p[0]) for p in pts]
    d2 = median_dist(swapped_pts)
    if d1 > 800 and d2 + 1e-6 < d1:
        return swapped_pts
    return pts


def latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"

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
    if "_region_candidates" not in ss:
        ss._region_candidates = []
    if "_optimized_stops" not in ss:
        ss._optimized_stops = None
    if "iso_geojson" not in ss:
        ss.iso_geojson = None

ensure_state()
ss = st.session_state

def set_center(lat: float, lng: float, country: str | None = None):
    ss.center["lat"] = float(lat)
    ss.center["lng"] = float(lng)
    if country:
        ss.center["country"] = country

def add_stops(rows: List[Dict[str, Any]], source: str):
    df = ss.stops_df.copy()
    # Ensure address column stays string-like
    if "address" in df.columns:
        df["address"] = df["address"].astype("string")

    for r in rows:
        label = r.get("label") or r.get("name") or f"Stop {len(df)+1}"
        addr = r.get("address") or r.get("name") or ""
        lat = r.get("lat")
        lng = r.get("lng")
        lat_f, lng_f, _sw = choose_latlng(lat, lng, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)
        if lat_f is None:
            continue

        df = pd.concat(
            [df, pd.DataFrame([{
                "label": str(label),
                "address": str(addr),
                "lat": lat_f,
                "lng": lng_f,
                "source": source
            }])],
            ignore_index=True,
        )

    ss.stops_df = df.reset_index(drop=True)

def replace_stops(df_new: pd.DataFrame):
    # keep types stable + fix lat/lng order (using current center as reference)
    df = df_new.copy()
    ref = None
    try:
        ref = (float(ss.center["lat"]), float(ss.center["lng"]))
    except Exception:
        ref = None
    df, _ = sanitize_stops_df(df, ref=ref)
    ss.stops_df = df.reset_index(drop=True)


def clear_stops():
    ss.stops_df = pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    ss.route_before = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.route_after = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.vrp = {"create": None, "result": None, "order": None, "job_id": None}
    ss.last_json = {}
    ss.last_http = {}
    ss.iso_geojson = None

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
        try:
            parsed: Any = r.json()
        except Exception:
            parsed = raw

        ss.last_http = {
            "method": method.upper(),
            "url": url,
            "params": params,
            "body": body,
            "status": r.status_code,
            "response_headers": dict(r.headers),
            "raw_text": raw[:20000],
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
        lat_f, lng_f, _sw = choose_latlng(lat, lng, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)
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
                    name = it.get("label") or (it.get("address") or {}).get("label") or it.get("title") or it.get("name") or "Result"
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

    # GeoJSON FeatureCollection
    if resp.get("type") == "FeatureCollection" and "features" in resp:
        for f in resp["features"]:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates") or []
                pts = []
                for c in coords:
                    if isinstance(c, list) and len(c) >= 2:
                        lng, lat = c[0], c[1]
                        lat_f, lng_f, _sw = choose_latlng(lat, lng, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)
                        if lat_f is not None:
                            pts.append((lat_f, lng_f))
                if pts:
                    return fix_points_order(pts, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)

    routes = resp.get("routes") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    r0 = routes[0] if isinstance(routes[0], dict) else {}
    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")

    if isinstance(poly, str) and poly.strip():
        pts5 = decode_polyline(poly, precision=5)
        pts6 = decode_polyline(poly, precision=6)
        pts = pts6 if len(pts6) > len(pts5) else pts5
        if len(pts) >= 2:
            return fix_points_order(pts, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)

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
            return fix_points_order(pts, ref=(float(ss.center['lat']), float(ss.center['lng'])) if 'center' in ss else None)

    return []

def parse_distance_duration_from_directions(resp: Any) -> Tuple[Optional[float], Optional[float]]:
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
            if isinstance(v, dict) and "value" in v:
                return float(v["value"])
            if isinstance(v, str) and v.strip():
                return float(v)
        except Exception:
            return None
        return None

    dist = _num(r0.get("distance")) or _num(r0.get("distanceMeters") or r0.get("distance_meters"))
    dur = _num(r0.get("duration")) or _num(r0.get("durationSeconds") or r0.get("duration_seconds"))
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
    geojson: Any = None,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")

    # Isochrone / GeoJSON overlay
    if geojson is not None:
        try:
            folium.GeoJson(geojson, name="isochrone").add_to(m)
        except Exception:
            pass

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

        # Direction arrows (only if plugin exists)
        if PolyLineTextPath is not None:
            try:
                pl = folium.PolyLine(route_pts, weight=0, opacity=0)
                pl.add_to(m)
                PolyLineTextPath(pl, "  ➤  ", repeat=True, offset=7, attributes={"font-size": "14", "fill": "black"}).add_to(m)
            except Exception:
                pass

        try:
            lats = [p[0] for p in route_pts]
            lngs = [p[1] for p in route_pts]
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
        except Exception:
            pass

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
    # Reverse Geocode API: /revgeocode?at=lat,lng&key=...
    params = {"key": NB_API_KEY, "at": f"{float(lat):.6f},{float(lng):.6f}", "language": language}
    stt, data = nb_get("/revgeocode", params=params)
    return stt, data


def distance_matrix(origins: List[str], destinations: List[str], params_extra: Dict[str, Any]) -> Tuple[int, Any]:
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

    stt, data = nb_get("/directions/json", params=params)
    if stt == 404:
        stt, data = nb_get("/directions/v2", params=params)
    if stt == 404:
        stt, data = nb_get("/navigation/v2", params=params)
    return stt, data

# -----------------------------
# VRP v2 (payload shape correct)
# -----------------------------
def build_vrp_payload_v2(stops: pd.DataFrame, objective: str, routing_mode: str, traffic_ts: int) -> Dict[str, Any]:
    loc_list: List[str] = []
    for r in stops.reset_index(drop=True).itertuples():
        loc_list.append(latlng_str(float(r.lat), float(r.lng)))

    payload = {
        "locations": {"id": 0, "location": loc_list},
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
    stt, data = nb_get("/optimization/v2/result", params=params, nocache=True)
    ss.last_json["vrp_result_response"] = data
    return stt, data

def parse_vrp_order(resp: Any) -> Optional[List[int]]:
    if not isinstance(resp, dict):
        return None

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

    seq = r0.get("sequence")
    if isinstance(seq, list) and seq:
        try:
            return [int(x) for x in seq]
        except Exception:
            pass

    return None

# -----------------------------
# SNAP + ISO
# -----------------------------
def snap_to_road(points: List[Tuple[float, float]]) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY}
    body = {"points": [{"lat": p[0], "lng": p[1]} for p in points]}
    ss.last_json["snap_payload"] = body

    stt, data = nb_post("/snapToRoad/v2", params=params, body=body)
    if stt == 404:
        stt, data = nb_post("/snaptoroad/v2", params=params, body=body)
    if stt == 404:
        stt, data = nb_post("/snapToRoad", params=params, body=body)

    ss.last_json["snap_response"] = data
    return stt, data

def isochrone(center_lat: float, center_lng: float, iso_type: str, iso_val: int, iso_mode: str) -> Tuple[int, Any]:
    # Per NextBillion Isochrone docs: HTTPS GET /isochrone/json with coordinates + contours_* :contentReference[oaicite:2]{index=2}
    params: Dict[str, Any] = {
        "key": NB_API_KEY,
        "coordinates": f"{float(center_lat)},{float(center_lng)}",
        "mode": iso_mode,
    }

    if iso_type == "time":
        params["contours_minutes"] = max(1, int(round(iso_val / 60)))
    else:
        params["contours_meters"] = int(iso_val)

    ss.last_json["iso_params"] = params
    stt, data = nb_get("/isochrone/json", params=params)
    if stt == 404:
        stt, data = nb_get("/isochrone/v2", params=params)  # fallback if enabled

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
                df["address"] = df["address"].astype("string")
                for addr in lines:
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

aaatabs = st.tabs(["Geocode & Map", "Places (Search + Generate Stops)", "Route + Optimize (Before vs After)", "Distance Matrix (NxN)", "Snap-to-Road + Isochrone", "Debug (Last API Call)"])

# -----------------------------
# TAB 1: GEOCODE & MAP
# -----------------------------
with tabs[0]:
    st.subheader("Geocode your stops and show them on the map")
    maybe_fix_session_stops(ref=(float(ss.center["lat"]), float(ss.center["lng"])))
    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        country = st.text_input("Country filter (3-letter)", value=ss.center["country"], key="geo_country")
    with gcol2:
        if st.button("🌍 Set country filter", key="geo_set_country"):
            ss.center["country"] = country.strip().upper()[:3] or ss.center["country"]

    df = ss.stops_df.copy()

    if st.button("🧭 Geocode all missing coordinates", key="geo_geocode_btn", width="stretch"):
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
    maybe_fix_session_stops(ref=(float(ss.center["lat"]), float(ss.center["lng"])))
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

    if st.button("🎲 Generate random stops around center", key="pl_gen_random", width="stretch"):
        pts = gen_random_points(float(ss.center["lat"]), float(ss.center["lng"]), int(n_rand), int(radius_m), int(seed))
        rows = [{"label": f"Rand {i}", "address": "", "lat": la, "lng": ln} for i, (la, ln) in enumerate(pts, start=1)]
        add_stops(rows, "Random around center")

        # Resolve addresses (reverse geocode) — FIXED indentation + dtype safety
                if resolve_addr == "Yes":
            df = ss.stops_df.copy()
            if "address" not in df.columns:
                df["address"] = ""
            df["address"] = df["address"].astype("string")

            # Resolve addresses for ALL stops that have valid lat/lng
            ref = (float(ss.center["lat"]), float(ss.center["lng"]))
            for ridx, row in df.iterrows():
                lat_c, lng_c, _sw = choose_latlng(row.get("lat"), row.get("lng"), ref=ref)
                if lat_c is None or lng_c is None:
                    continue

                # Only overwrite if empty / missing
                addr = str(row.get("address") or "").strip()
                if addr != "" and addr.lower() != "nan":
                    continue

                try:
                    status, rresp = geocode_reverse(float(lat_c), float(lng_c), language=language)
                    cand = extract_geocode_candidates(rresp)
                    if cand:
                        df.loc[ridx, "lat"] = lat_c
                        df.loc[ridx, "lng"] = lng_c
                        df.loc[ridx, "address"] = str(cand[0]["name"])
                        df.loc[ridx, "source"] = f"Reverse-geocode ({status})"
                except Exception:
                    continue

            replace_stops(df)

ss.mapsig["places"] += 1

    st.markdown("### Stops table")
    st.dataframe(ss.stops_df, width="stretch", height=260)

# -----------------------------
# TAB 3: ROUTE + OPTIMIZE
# -----------------------------
with tabs[2]:
    st.subheader("Compute route (Before) → run optimization → recompute route (After) + compare")
    maybe_fix_session_stops(ref=(float(ss.center["lat"]), float(ss.center["lng"])))

    if len(ss.stops_df) < 2:
        st.info("Add at least 2 stops first.")
    else:
        df_s, _ = sanitize_stops_df(ss.stops_df, ref=(float(ss.center['lat']), float(ss.center['lng'])))
        stops_ll = [latlng_str(float(r.lat), float(r.lng)) for r in df_s.itertuples()]

        left, right = st.columns([1.15, 1.0])

        with left:
            st.markdown("### Step 1 — Directions (Before)")
            dir_override_text = st.text_area("Optional: Directions query override (JSON). Leave {} for none.", value="{}", height=90, key="dir_before_override")

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", width="stretch"):
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

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", width="stretch"):
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

            if st.button("🔄 Fetch VRP result again", key="vrp_fetch_btn", width="stretch"):
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
                            st.warning("Result fetched, but order not found yet. Try again in a few seconds.")
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

            if st.button("🧭 Compute route (After)", key="rt_after_btn", width="stretch"):
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
    maybe_fix_session_stops(ref=(float(ss.center["lat"]), float(ss.center["lng"])))

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

        df_use_s, _ = sanitize_stops_df(df_use, ref=(float(ss.center['lat']), float(ss.center['lng'])))
        ll = [latlng_str(float(r.lat), float(r.lng)) for r in df_use_s.itertuples()]

        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", width="stretch"):
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
            st.dataframe(pd.DataFrame(dist_km).round(2), width="stretch", height=240)
            st.markdown("### Duration (min)")
            st.dataframe(pd.DataFrame(dur_min).round(1), width="stretch", height=240)

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

        if st.button("🧷 Snap to Road", key="snap_btn", width="stretch"):
            pts = coords[: int(n_path)] if coords else []
            if len(pts) < 2:
                st.warning("Need route geometry (2+ points). Compute directions first.")
            else:
                stt, data = snap_to_road(pts)
                ss.mapsig["snap"] += 1
                if stt != 200:
                    st.error(f"Snap-to-road failed: HTTP {stt}")
                st.json(data)

        m5 = make_map(center=(float(ss.center["lat"]), float(ss.center["lng"])), stops=ss.stops_df, route_pts=coords[: int(n_path)] if coords else None, clicked=ss.clicked_pin, zoom=12)
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (GET /isochrone/json)")
        iso_type = st.selectbox("Type", ["time", "distance"], index=0, key="iso_type")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")
        iso_val = st.number_input("Value (seconds if time, meters if distance)", min_value=60, max_value=2000000, value=1800, step=60, key="iso_val")

        if st.button("🟦 Compute Isochrone", key="iso_btn", width="stretch"):
            stt, data = isochrone(float(ss.center["lat"]), float(ss.center["lng"]), iso_type, int(iso_val), iso_mode)
            ss.mapsig["iso"] += 1
            if stt != 200:
                st.error(f"Isochrone failed: HTTP {stt}")
                ss.iso_geojson = None
            else:
                ss.iso_geojson = data if isinstance(data, dict) and data.get("type") == "FeatureCollection" else None
                st.success("Isochrone response received.")
            st.json(data)

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            clicked=ss.clicked_pin,
            zoom=12,
            geojson=ss.iso_geojson,
        )
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

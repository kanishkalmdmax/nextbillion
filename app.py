# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# FIXES:
# - VRP v2 payload: locations must include locations[i].location (fixes HTTP 400)
# - Directions: distance/time now parsed from directions response (no more 0.0), with matrix as best-effort
# - Directions: supports chunking for many stops (waypoint limit safe)
# - Snap-to-road: uses correct GET endpoint + mandatory path param
# - Isochrone: uses correct GET endpoint + contours_minutes/meters
# - Optional custom param box for each API call

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

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def merge_params(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if v is None:
            continue
        out[k] = v
    return out

def parse_custom_params(text: str) -> Dict[str, Any]:
    """
    User can paste JSON like:
    {"units":"metric","avoid":"toll"}
    """
    if not text or not text.strip():
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}

# -----------------------------
# HTTP (button-only + caching)
# -----------------------------
@dataclass(frozen=True)
class ReqSig:
    method: str
    url: str
    params_json: str
    body_json: str

def _sig(method: str, url: str, params: Dict[str, Any] | None, body: Dict[str, Any] | None) -> ReqSig:
    pj = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    bj = json.dumps(body or {}, sort_keys=True, separators=(",", ":"))
    return ReqSig(method=method.upper(), url=url, params_json=pj, body_json=bj)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_request(sig: ReqSig) -> Tuple[int, Any]:
    method = sig.method
    url = sig.url
    params = json.loads(sig.params_json)
    body = json.loads(sig.body_json)

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
    url = NB_BASE + path
    sig = _sig("GET", url, params=params, body=None)
    return _cached_request(sig)

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any]) -> Tuple[int, Any]:
    url = NB_BASE + path
    sig = _sig("POST", url, params=params, body=body)
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

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    if not isinstance(polyline_str, str):
        return []
    polyline_str = polyline_str.strip()
    if not polyline_str:
        return []

    index = 0
    lat = 0
    lng = 0
    coordinates: List[Tuple[float, float]] = []
    factor = 10 ** precision

    length = len(polyline_str)
    while index < length:
        result = 1
        shift = 0
        while True:
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
            b = ord(polyline_str[index]) - 63
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1F:
                break
        delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += delta_lng

        coordinates.append((lat / factor, lng / factor))

    return coordinates

def extract_route_geometry(resp: Any) -> List[Tuple[float, float]]:
    if not isinstance(resp, dict):
        return []

    # GeoJSON LineString
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

    routes = resp.get("routes") or resp.get("route") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    r0 = routes[0] if isinstance(routes[0], dict) else {}

    # common polyline fields
    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")
    if isinstance(poly, str) and poly.strip():
        # NextBillion sometimes uses polyline6; try both and prefer the one that looks sane.
        pts5 = decode_polyline(poly, precision=5)
        pts6 = decode_polyline(poly, precision=6)
        # pick non-empty; if both, prefer the longer one (usually more detailed)
        if pts6 and len(pts6) >= len(pts5):
            return pts6
        return pts5

    # legs / steps polyline
    legs = r0.get("legs") or []
    if isinstance(legs, dict):
        legs = [legs]
    pts2: List[Tuple[float, float]] = []
    for leg in legs:
        steps = (leg or {}).get("steps") or []
        for stp in steps:
            g = (stp or {}).get("geometry")
            if isinstance(g, str) and g.strip():
                pts2.extend(decode_polyline(g, precision=6) or decode_polyline(g, precision=5))
    return pts2

def extract_route_summary(resp: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (distance_m, duration_s) from directions response if available.
    """
    if not isinstance(resp, dict):
        return None, None
    routes = resp.get("routes") or resp.get("route") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list):
        return None, None
    r0 = routes[0] if isinstance(routes[0], dict) else {}

    # NB-style: distance.value, duration.value
    dist = (r0.get("distance") or {}).get("value")
    dur = (r0.get("duration") or {}).get("value")
    if dist is not None or dur is not None:
        return safe_float(dist, None), safe_float(dur, None)

    # Some variants: distance, duration directly
    if r0.get("distance") is not None and r0.get("duration") is not None:
        return safe_float(r0.get("distance"), None), safe_float(r0.get("duration"), None)

    return None, None

def extract_geojson_polygons(resp: Any) -> Optional[dict]:
    """
    Isochrone returns FeatureCollection (GeoJSON). We'll render it directly.
    """
    if isinstance(resp, dict) and resp.get("type") == "FeatureCollection" and resp.get("features"):
        return resp
    return None

# -----------------------------
# MAP RENDERING (stable)
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    clicked: Optional[Tuple[float, float]] = None,
    geojson_layer: Optional[dict] = None,
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")

    if clicked is not None:
        folium.Marker(
            location=[clicked[0], clicked[1]],
            popup="Pinned center",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    # stops
    for i, row in stops.reset_index(drop=True).iterrows():
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

    # optional isochrone polygons
    if geojson_layer:
        try:
            folium.GeoJson(geojson_layer, name="isochrone").add_to(m)
        except Exception:
            pass

    # route polyline + arrows
    if route_pts:
        folium.PolyLine(route_pts, weight=5, opacity=0.8).add_to(m)
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
# STOPS STATE
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
        ss.vrp = {"create": None, "result": None, "order": None}
    if "last_json" not in ss:
        ss.last_json = {}
    if "_optimized_stops" not in ss:
        ss._optimized_stops = None

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
            [
                df,
                pd.DataFrame([{
                    "label": str(label),
                    "address": str(addr),
                    "lat": lat_f,
                    "lng": lng_f,
                    "source": source,
                }]),
            ],
            ignore_index=True,
        )
    ss.stops_df = df.reset_index(drop=True)

def replace_stops(df_new: pd.DataFrame):
    st.session_state.stops_df = df_new.reset_index(drop=True)

def clear_stops():
    ss = st.session_state
    ss.stops_df = pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    ss.route_before = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.route_after = {"resp": None, "distance_m": None, "duration_s": None, "geometry": []}
    ss.vrp = {"create": None, "result": None, "order": None}
    ss._optimized_stops = None

# -----------------------------
# API CALL BUILDERS
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
    status, data = nb_get("/distancematrix/v2", params=params)
    return status, data

def directions_one(order_latlng: List[str], params_extra: Dict[str, Any]) -> Tuple[int, Any]:
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

    status, data = nb_get("/directions/v2", params=params)
    if status == 404:
        status, data = nb_get("/navigation/v2", params=params)
    if status == 404:
        status, data = nb_get("/directions", params=params)
    return status, data

def directions_multi_stop_chunked(order_latlng: List[str], params_extra: Dict[str, Any], max_points_per_call: int = 25) -> Tuple[int, dict]:
    """
    Many directions APIs enforce waypoint limits.
    We chunk the path into overlapping segments: [0..24], [24..48], etc.
    Merge geometry + sum distance/duration (from route summaries if provided).
    """
    if len(order_latlng) <= max_points_per_call:
        stt, resp = directions_one(order_latlng, params_extra=params_extra)
        return stt, {"_chunked": False, "responses": [resp], "status_codes": [stt], "merged": resp}

    responses = []
    codes = []
    merged_geom: List[Tuple[float, float]] = []
    total_dist = 0.0
    total_dur = 0.0

    start = 0
    while start < len(order_latlng) - 1:
        end = min(start + max_points_per_call, len(order_latlng))
        segment = order_latlng[start:end]

        stt, resp = directions_one(segment, params_extra=params_extra)
        codes.append(stt)
        responses.append(resp)

        if stt == 200 and isinstance(resp, dict):
            seg_geom = extract_route_geometry(resp)
            if seg_geom:
                if merged_geom and seg_geom:
                    # avoid duplicate seam point
                    if merged_geom[-1] == seg_geom[0]:
                        merged_geom.extend(seg_geom[1:])
                    else:
                        merged_geom.extend(seg_geom)
                else:
                    merged_geom.extend(seg_geom)

            d_m, t_s = extract_route_summary(resp)
            if d_m is not None:
                total_dist += float(d_m)
            if t_s is not None:
                total_dur += float(t_s)

        # overlap by 1 point so continuity holds
        start = end - 1

    merged = {
        "routes": [{
            "distance": {"value": total_dist},
            "duration": {"value": total_dur},
            # store geometry as polyline points for our renderer
            "geometry": [{"lat": p[0], "lng": p[1]} for p in merged_geom],
        }],
        "_note": "Chunked directions merge",
    }
    return 200 if all(c == 200 for c in codes) else (next((c for c in codes if c != 200), 500)), {
        "_chunked": True,
        "responses": responses,
        "status_codes": codes,
        "merged": merged,
    }

def vrp_create_v2(stops: pd.DataFrame, objective: str, custom_body_overrides: Dict[str, Any] | None = None) -> Tuple[int, Any]:
    """
    FIX: VRP v2 expects locations as list of objects containing "location".
    Your old payload was: locations: ["lat,lng", ...] which triggers:
      "Mandatory parameter 'locations.location' is missing"
    """
    if len(stops) < 2:
        return 0, {"error": "Need at least 2 stops"}

    df = stops.reset_index(drop=True)

    locations = []
    for i, r in df.iterrows():
        locations.append({
            "id": int(i),
            "location": latlng_str(float(r["lat"]), float(r["lng"])),
        })

    # Recommend: treat index 0 as depot, jobs are 1..n-1
    jobs = [{"id": int(i), "location_index": int(i)} for i in range(1, len(locations))]

    vehicles = [{"id": "vehicle_1", "start_index": 0, "end_index": 0}]

    body = {
        "locations": locations,
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {"objective": {"travel_cost": objective}},  # distance | duration
    }

    # allow user overrides
    if custom_body_overrides:
        # shallow merge (user can override top-level keys)
        for k, v in custom_body_overrides.items():
            body[k] = v

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
    routes = resp.get("routes") or resp.get("result", {}).get("routes") or resp.get("solution", {}).get("routes")
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

def snap_to_roads(path_points: List[Tuple[float, float]], custom_params: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Snap-To-Road is GET:
      /snapToRoads/json?key=...&path=lat,lng|lat,lng...
    """
    if len(path_points) < 2:
        return 0, {"error": "Need 2+ points to snap"}

    path = "|".join([latlng_str(p[0], p[1]) for p in path_points])
    params = merge_params({"key": NB_API_KEY, "path": path}, custom_params or {})
    status, data = nb_get("/snapToRoads/json", params=params)
    st.session_state.last_json["snap_request_params"] = params
    st.session_state.last_json["snap_response"] = data
    return status, data

def isochrone_get(center: Tuple[float, float], mode: str, contours_kind: str, contours_value: int, custom_params: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Isochrone is GET:
      /isochrone/json?key=...&coordinates=lat,lng&mode=car&contours_minutes=5
    """
    lat, lng = center
    params = {"key": NB_API_KEY, "coordinates": latlng_str(lat, lng), "mode": mode}

    if contours_kind == "minutes":
        params["contours_minutes"] = int(contours_value)
    else:
        params["contours_meters"] = int(contours_value)

    params = merge_params(params, custom_params or {})
    status, data = nb_get("/isochrone/json", params=params)
    st.session_state.last_json["iso_request_params"] = params
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
                start_n = len(df)
                for i, addr in enumerate(lines):
                    df = pd.concat([df, pd.DataFrame([{
                        "label": f"Stop {start_n + i + 1}",
                        "address": addr,
                        "lat": None,
                        "lng": None,
                        "source": "Pasted address",
                    }])], ignore_index=True)
                ss.stops_df = df.reset_index(drop=True)
            else:
                rows = []
                start_n = len(ss.stops_df)
                for i, ll in enumerate(lines):
                    parts = [p.strip() for p in ll.split(",")]
                    if len(parts) >= 2:
                        lat_f, lng_f = normalize_latlng(parts[0], parts[1])
                        if lat_f is not None:
                            rows.append({"label": f"Stop {start_n + i + 1}", "address": "", "lat": lat_f, "lng": lng_f})
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

    st.divider()
    st.markdown("### Optional: POI keyword search (best-effort) and add results as stops")

    kw = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="", key="pl_kw")
    kw_radius = st.slider("Search radius (m)", min_value=500, max_value=200000, value=5000, step=500, key="pl_kw_radius")
    kw_max = st.slider("Max results", min_value=1, max_value=100, value=10, step=1, key="pl_kw_max")

    if st.button("Search Places (POIs)", key="pl_search_pois", use_container_width=True):
        q = f"{kw}".strip()
        if not q:
            st.warning("Enter a keyword (or use Random Stops above).")
        else:
            params = {
                "key": NB_API_KEY, "q": q, "country": ss.center["country"], "language": language,
                "at": latlng_str(float(ss.center["lat"]), float(ss.center["lng"])),
                "radius": int(kw_radius), "limit": int(kw_max)
            }
            status, resp = nb_get("/geocode", params=params)
            if status == 404:
                status, resp = nb_get("/geocode/v1", params=params)
            ss.last_json["places_response"] = resp
            st.success(f"Places response: HTTP {status}")

            items = extract_geocode_candidates(resp)
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

            custom_dir_params = parse_custom_params(
                st.text_area(
                    "Optional custom Directions params (JSON)",
                    value="",
                    height=80,
                    key="custom_dir_params_before",
                    help='Example: {"units":"metric","alternatives":"true"}'
                )
            )

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", use_container_width=True):
                # Directions (chunked) – also yields distance/duration summary
                stt, pack = directions_multi_stop_chunked(stops_ll, params_extra=merge_params(GLOBAL_PARAMS, custom_dir_params))
                ss.last_json["directions_before_raw"] = pack

                merged = pack["merged"] if isinstance(pack, dict) else pack
                ss.route_before["resp"] = merged
                ss.route_before["geometry"] = extract_route_geometry(merged)

                # FIX: distance/time from directions summary
                d_m, t_s = extract_route_summary(merged)
                ss.route_before["distance_m"] = d_m
                ss.route_before["duration_s"] = t_s

                # Best-effort matrix along adjacent pairs (optional)
                try:
                    status_m, data_m = distance_matrix(stops_ll[:-1], stops_ll[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_before"] = data_m
                except Exception:
                    pass

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
            st.metric("Before distance (km)", f"{(bdist or 0)/1000:.2f}")
            st.metric("Before duration (min)", f"{(bdur or 0)/60:.1f}")

        with right:
            st.markdown("### Step 2 — Optimization (VRP v2)")

            objective = st.selectbox("Optimization objective", ["distance", "duration"], index=0, key="vrp_obj")

            custom_vrp_overrides = parse_custom_params(
                st.text_area(
                    "Optional VRP body overrides (JSON) — advanced",
                    value="",
                    height=80,
                    key="custom_vrp_overrides",
                    help='Example: {"vehicles":[{"id":"v1","start_index":0,"end_index":0,"capacity":[10]}]}'
                )
            )

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", use_container_width=True):
                status_c, data_c = vrp_create_v2(ss.stops_df, objective=objective, custom_body_overrides=custom_vrp_overrides)
                if status_c != 200:
                    st.error(f"VRP create failed: HTTP {status_c}")
                    st.json(data_c)
                else:
                    ss.vrp["create"] = data_c
                    job_id = data_c.get("id") or data_c.get("job_id") or data_c.get("jobId")
                    if not job_id:
                        st.warning("VRP create succeeded, but no job id found in response.")
                    else:
                        result = None
                        for _ in range(12):
                            stt, rr = vrp_result_v2(str(job_id))
                            if stt == 200:
                                result = rr
                                break
                            time.sleep(1.0)
                        ss.vrp["result"] = result
                        if result is None:
                            st.error("Could not fetch VRP result in time.")
                        else:
                            order = parse_vrp_order(result)
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
                                st.warning("Optimization result did not include an order we could parse.")

            st.markdown("### Step 3 — Recompute Directions (After)")

            df_after = ss._optimized_stops if ss._optimized_stops is not None else ss.stops_df.copy()

            custom_dir_params_after = parse_custom_params(
                st.text_area(
                    "Optional custom Directions params (JSON) — After",
                    value="",
                    height=80,
                    key="custom_dir_params_after",
                )
            )

            if st.button("🧭 Compute route (After)", key="rt_after_btn", use_container_width=True):
                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]

                stt2, pack2 = directions_multi_stop_chunked(ll_after, params_extra=merge_params(GLOBAL_PARAMS, custom_dir_params_after))
                ss.last_json["directions_after_raw"] = pack2
                merged2 = pack2["merged"] if isinstance(pack2, dict) else pack2

                ss.route_after["resp"] = merged2
                ss.route_after["geometry"] = extract_route_geometry(merged2)

                d2_m, t2_s = extract_route_summary(merged2)
                ss.route_after["distance_m"] = d2_m
                ss.route_after["duration_s"] = t2_s

                try:
                    status_m2, data_m2 = distance_matrix(ll_after[:-1], ll_after[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_after"] = data_m2
                except Exception:
                    pass

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
            st.metric("After distance (km)", f"{(adist or 0)/1000:.2f}")
            st.metric("After duration (min)", f"{(adur or 0)/60:.1f}")

            if (ss.route_before.get("distance_m") or 0) > 0 and (ss.route_after.get("distance_m") or 0) > 0:
                b = float(ss.route_before["distance_m"])
                a = float(ss.route_after["distance_m"])
                saved = b - a
                pct = (saved / b * 100.0) if b > 0 else 0.0
                st.success(f"Distance saved: {saved/1000:.2f} km ({pct:.1f}%)")

            if (ss.route_before.get("duration_s") or 0) > 0 and (ss.route_after.get("duration_s") or 0) > 0:
                b = float(ss.route_before["duration_s"])
                a = float(ss.route_after["duration_s"])
                saved = b - a
                pct = (saved / b * 100.0) if b > 0 else 0.0
                st.success(f"Time saved: {saved/60:.1f} min ({pct:.1f}%)")

        with st.expander("Download JSON (debug)", expanded=False):
            bundle = {
                "directions_before": ss.route_before.get("resp"),
                "directions_after": ss.route_after.get("resp"),
                "vrp_create_payload": ss.last_json.get("vrp_create_payload"),
                "vrp_create_response": ss.last_json.get("vrp_create_response"),
                "vrp_result_response": ss.last_json.get("vrp_result_response"),
                "directions_before_raw": ss.last_json.get("directions_before_raw"),
                "directions_after_raw": ss.last_json.get("directions_after_raw"),
                "matrix_pairs_before": ss.last_json.get("matrix_pairs_before"),
                "matrix_pairs_after": ss.last_json.get("matrix_pairs_after"),
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

        custom_matrix_params = parse_custom_params(
            st.text_area(
                "Optional custom Matrix params (JSON)",
                value="",
                height=80,
                key="custom_matrix_params",
            )
        )

        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", use_container_width=True):
            status, data = distance_matrix(ll, ll, params_extra=merge_params(GLOBAL_PARAMS, custom_matrix_params))
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
    st.subheader("Snap-to-Road + Isochrone (correct endpoints + stable UI; runs only on button click)")

    s1, s2 = st.columns([1, 1])

    with s1:
        st.markdown("### Snap-to-Road (GET + mandatory path)")
        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use Directions (After) geometry"], index=0, key="snap_src")
        geom = ss.route_before.get("geometry") if src.startswith("Use Directions (Before)") else ss.route_after.get("geometry")
        coords = geom or []

        if not coords:
            st.caption("Compute a route first (Before/After) to get geometry points.")

        max_pts = max(2, len(coords)) if coords else 2
        if max_pts <= 2:
            n_path = 2
        else:
            n_path = st.slider(
                "N geometry points to send (first N)",
                min_value=2,
                max_value=max_pts,
                value=min(300, max_pts),
                step=1,
                key="snap_n",
            )

        st.caption("SnapToRoads uses: /snapToRoads/json?key=...&path=lat,lng|lat,lng...")  # per docs
        custom_snap_params = parse_custom_params(
            st.text_area(
                "Optional custom SnapToRoads params (JSON)",
                value="",
                height=80,
                key="custom_snap_params",
                help='Example: {"interpolate":"true"}'
            )
        )

        if st.button("🧷 Snap to Road", key="snap_btn", use_container_width=True):
            pts = coords[: int(n_path)] if coords else []
            status, data = snap_to_roads(pts, custom_params=custom_snap_params)
            ss.mapsig["snap"] += 1

            if status != 200:
                st.error(f"Snap-to-road failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Snap-to-road computed (cached).")

        m5 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            route_pts=coords[: int(n_path)] if coords else None,
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (GET + contours_minutes/meters)")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")

        kind = st.selectbox("Contours kind", ["minutes", "meters"], index=0, key="iso_kind")

        # UI max was too low earlier — allow higher via number_input (API may enforce its own limits)
        default_val = 5 if kind == "minutes" else 5000
        iso_val = st.number_input(
            f"Contours value ({'minutes' if kind=='minutes' else 'meters'})",
            min_value=1,
            max_value=10_000_000,  # very high UI cap; API may still limit
            value=default_val,
            step=1 if kind == "minutes" else 100,
            key="iso_val",
        )

        st.caption("Isochrone uses: /isochrone/json?key=...&coordinates=lat,lng&mode=...&contours_minutes=... OR contours_meters=...")

        custom_iso_params = parse_custom_params(
            st.text_area(
                "Optional custom Isochrone params (JSON)",
                value="",
                height=80,
                key="custom_iso_params",
                help='Example: {"polygons":"true","denoise":"0.001"}'
            )
        )

        iso_geo = None
        if st.button("🟦 Compute Isochrone", key="iso_btn", use_container_width=True):
            status, data = isochrone_get(
                center=(float(ss.center["lat"]), float(ss.center["lng"])),
                mode=iso_mode,
                contours_kind=kind,
                contours_value=int(iso_val),
                custom_params=custom_iso_params,
            )
            ss.mapsig["iso"] += 1

            if status != 200:
                st.error(f"Isochrone failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Isochrone computed (cached).")

        # Render isochrone polygons if available
        iso_resp = ss.last_json.get("iso_response")
        iso_geo = extract_geojson_polygons(iso_resp)

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            clicked=ss.clicked_pin,
            geojson_layer=iso_geo,
            zoom=12,
        )
        render_map(m6, key=f"map_iso_{ss.mapsig['iso']}")

    with st.expander("Download JSON (debug)", expanded=False):
        bundle = {
            "snap_request_params": ss.last_json.get("snap_request_params"),
            "snap_response": ss.last_json.get("snap_response"),
            "iso_request_params": ss.last_json.get("iso_request_params"),
            "iso_response": ss.last_json.get("iso_response"),
        }
        st.download_button(
            "Download Snap+Iso JSON",
            data=json.dumps(bundle, indent=2).encode("utf-8"),
            file_name="snap_iso.json",
            mime="application/json",
            key="dl_snap_iso",
        )

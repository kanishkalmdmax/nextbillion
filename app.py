# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# FIXES:
# - VRP v2 payload: ensures `locations.location` exists (required by API) + normalization of overrides
# - Distance Matrix: endpoint fallbacks incl. /distancematrix/json + safe JSON parsing
# - Before/After totals: robust parsing so distance/time aren't stuck at 0
# - Polyline decoder: prevents IndexError on malformed/truncated polylines
# - Isochrone: draws polygon/outline on map when returned as GeoJSON-like
# - Snap-to-road: draws snapped geometry when returned
# - Global route options: flow into Directions/Matrix and VRP routing options

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
            r = requests.get(url, params=params, timeout=60)
        else:
            r = requests.post(url, params=params, json=body, timeout=60)

        ct = (r.headers.get("content-type", "") or "").lower()

        # Only parse JSON if content-type says so AND body isn't empty
        if "application/json" in ct:
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, r.text

        return r.status_code, r.text
    except Exception as e:
        return 0, {"error": str(e)}

def nb_get(path: str, params: Dict[str, Any]) -> Tuple[int, Any]:
    sig = _sig("GET", path, params=params, body=None)
    return _cached_request(sig)

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any]) -> Tuple[int, Any]:
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

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    # Safe Google polyline decoder (prevents IndexError on malformed/truncated strings)
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

    def _read_varint() -> Optional[int]:
        nonlocal index
        result = 0
        shift = 0
        while True:
            if index >= length:
                return None
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        return result

    while index < length:
        r1 = _read_varint()
        if r1 is None:
            break
        dlat = ~(r1 >> 1) if (r1 & 1) else (r1 >> 1)
        lat += dlat

        r2 = _read_varint()
        if r2 is None:
            break
        dlng = ~(r2 >> 1) if (r2 & 1) else (r2 >> 1)
        lng += dlng

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

    poly = r0.get("geometry") or r0.get("polyline") or (r0.get("overview_polyline") or {}).get("points")
    if isinstance(poly, str) and poly.strip():
        # Some providers use polyline6; try both, pick non-empty/longer
        pts5 = decode_polyline(poly, precision=5)
        pts6 = decode_polyline(poly, precision=6)
        if pts6 and len(pts6) >= len(pts5):
            return pts6
        return pts5

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
        if pts:
            return pts

    # legs / steps
    legs = r0.get("legs") or []
    if isinstance(legs, dict):
        legs = [legs]
    pts2: List[Tuple[float, float]] = []
    for leg in legs:
        steps = (leg or {}).get("steps") or []
        for stp in steps:
            g = (stp or {}).get("geometry")
            if isinstance(g, str) and g.strip():
                pts2.extend(decode_polyline(g, precision=5))
    return pts2

def extract_isoline_polygons(resp: Any) -> List[List[Tuple[float, float]]]:
    """
    Try to extract polygon rings from GeoJSON-like isochrone responses.
    Returns list of rings: [[(lat,lng),...], ...]
    """
    rings: List[List[Tuple[float, float]]] = []
    if not isinstance(resp, dict):
        return rings

    # GeoJSON FeatureCollection -> Polygon/MultiPolygon
    if resp.get("type") == "FeatureCollection":
        feats = resp.get("features") or []
        for f in feats:
            geom = (f or {}).get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            if gtype == "Polygon" and isinstance(coords, list) and coords:
                outer = coords[0]
                ring = []
                for c in outer:
                    if isinstance(c, list) and len(c) >= 2:
                        lng, lat = c[0], c[1]
                        lat_f, lng_f = normalize_latlng(lat, lng)
                        if lat_f is not None:
                            ring.append((lat_f, lng_f))
                if ring:
                    rings.append(ring)
            if gtype == "MultiPolygon" and isinstance(coords, list):
                for poly in coords:
                    if isinstance(poly, list) and poly:
                        outer = poly[0]
                        ring = []
                        for c in outer:
                            if isinstance(c, list) and len(c) >= 2:
                                lng, lat = c[0], c[1]
                                lat_f, lng_f = normalize_latlng(lat, lng)
                                if lat_f is not None:
                                    ring.append((lat_f, lng_f))
                        if ring:
                            rings.append(ring)

    # Some APIs return { "polygons": [ { "outer": [ [lng,lat], ...] } ] }
    if not rings and isinstance(resp.get("polygons"), list):
        for p in resp["polygons"]:
            outer = (p or {}).get("outer") or (p or {}).get("coordinates")
            if isinstance(outer, list):
                ring = []
                for c in outer:
                    if isinstance(c, list) and len(c) >= 2:
                        lng, lat = c[0], c[1]
                        lat_f, lng_f = normalize_latlng(lat, lng)
                        if lat_f is not None:
                            ring.append((lat_f, lng_f))
                if ring:
                    rings.append(ring)

    return rings

def parse_matrix_totals(data_m: Any) -> Tuple[float, float]:
    """
    Returns (distance_meters, duration_seconds) summed across pairwise rows.
    Supports common matrix schemas.
    """
    dist_total = 0.0
    dur_total = 0.0

    if not isinstance(data_m, dict):
        return dist_total, dur_total

    rows = data_m.get("rows") or []
    if not isinstance(rows, list):
        return dist_total, dur_total

    for row in rows:
        elems = (row or {}).get("elements") or []
        if not elems:
            continue
        # we are calling matrix with N origins and 1 destination each (pairs), so use first element
        e0 = elems[0] if isinstance(elems, list) else None
        if not isinstance(e0, dict):
            continue

        # distance: can be {"value": meters} or direct number
        d = e0.get("distance")
        if isinstance(d, dict):
            dist_total += float(d.get("value") or 0)
        elif isinstance(d, (int, float)):
            dist_total += float(d)

        t = e0.get("duration")
        if isinstance(t, dict):
            dur_total += float(t.get("value") or 0)
        elif isinstance(t, (int, float)):
            dur_total += float(t)

    return dist_total, dur_total

# -----------------------------
# MAP RENDERING (stable)
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    clicked: Optional[Tuple[float, float]] = None,
    iso_rings: Optional[List[List[Tuple[float, float]]]] = None,
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

    # Isochrone polygons (outline)
    if iso_rings:
        for ring in iso_rings:
            folium.Polygon(
                locations=ring,
                weight=3,
                fill=True,
                fill_opacity=0.15,
            ).add_to(m)

    if route_pts:
        folium.PolyLine(route_pts, weight=5, opacity=0.85).add_to(m)
        try:
            pl = folium.PolyLine(route_pts, weight=0, opacity=0)
            pl.add_to(m)
            PolyLineTextPath(pl, "  ➤  ", repeat=True, offset=7, attributes={"font-size": "14", "fill": "black"}).add_to(m)
        except Exception:
            pass

        lats = [p[0] for p in route_pts]
        lngs = [p[1] for p in route_pts]
        if lats and lngs:
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

    # Try common NB variants (some accounts don't have /v2)
    for ep in ["/distancematrix/json", "/distancematrix/v2", "/distancematrix", "/distancematrix/v1"]:
        status, data = nb_get(ep, params=params)
        if status != 404:
            return status, data
    return status, data

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

    for ep in ["/directions/v2", "/navigation/v2", "/directions"]:
        status, data = nb_get(ep, params=params)
        if status != 404:
            return status, data
    return status, data

# -------- VRP v2 normalization (CRITICAL FIX) --------
def normalize_vrp_body_to_v2_shape(body: Dict[str, Any], stops_df: Optional[pd.DataFrame] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Ensures VRP body has `locations.location` as a LIST of "lat,lng" strings.
    If user provided the older/wrong shape (locations as list of objects), convert it.
    """
    if not isinstance(body, dict):
        return False, "Body must be a JSON object.", {}

    # If user left {} and we have stops, we will build elsewhere.
    if body == {}:
        return True, "Using generated payload.", body

    locs = body.get("locations")

    # Case A: correct-ish: locations is an object with `location` list
    if isinstance(locs, dict) and isinstance(locs.get("location"), list):
        # ensure all are strings "lat,lng"
        fixed = []
        for x in locs["location"]:
            if isinstance(x, str) and "," in x:
                fixed.append(x.strip())
        body["locations"]["location"] = fixed
        if not fixed:
            return False, "locations.location exists but is empty/invalid.", body
        return True, "Override payload accepted (locations.location detected).", body

    # Case B: wrong common: locations is list of {id, location:"lat,lng"} -> convert
    if isinstance(locs, list):
        location_list: List[str] = []
        for it in locs:
            if isinstance(it, dict):
                s = it.get("location")
                if isinstance(s, str) and "," in s:
                    location_list.append(s.strip())
        if not location_list:
            return False, "Override has locations as list, but no valid {location:'lat,lng'} found.", body

        body["locations"] = {"location": location_list}
        return True, "Override normalized: locations[] -> locations.location[]", body

    # Case C: missing locations in override -> if stops_df present, build minimal
    if locs is None and stops_df is not None and len(stops_df) >= 2:
        location_list = [latlng_str(float(r["lat"]), float(r["lng"])) for _, r in stops_df.reset_index(drop=True).iterrows()]
        body["locations"] = {"location": location_list}
        return True, "Override missing locations; injected from current stops.", body

    return False, "Could not detect/normalize locations.location. Provide locations as {\"location\": [\"lat,lng\", ...] }.", body

def vrp_create_v2(stops: pd.DataFrame, objective: str, global_ui: Dict[str, Any], override_json_text: str) -> Tuple[int, Any, Dict[str, Any]]:
    """
    FIXED to match required `locations.location`.
    """
    if len(stops) < 2:
        return 0, {"error": "Need at least 2 stops"}, {}

    # Build generated v2 body (the required shape)
    loc_list = [latlng_str(float(r["lat"]), float(r["lng"])) for _, r in stops.reset_index(drop=True).iterrows()]

    jobs = [{"id": int(i), "location_index": int(i)} for i in range(len(loc_list))]
    vehicles = [{"id": "vehicle_1", "start_index": 0, "end_index": 0}]

    traffic_ts = int(global_ui.get("departure_time") or _now_unix())
    mode = global_ui.get("mode") or "car"

    generated = {
        "locations": {"location": loc_list},
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {
            "objective": {"travel_cost": objective},  # distance | duration
            "routing": {
                "mode": mode,
                "traffic_timestamp": traffic_ts,
            },
        },
    }

    # If override provided, merge/normalize
    body_to_send = generated
    override_json_text = (override_json_text or "").strip()
    if override_json_text and override_json_text != "{}":
        try:
            override = json.loads(override_json_text)
        except Exception as e:
            return 0, {"error": f"Invalid override JSON: {e}"}, generated

        ok, msg, normalized = normalize_vrp_body_to_v2_shape(override, stops_df=stops)
        if not ok:
            return 0, {"error": msg, "normalized_preview": normalized}, generated

        # Merge: override keys overwrite generated keys (after normalization)
        merged = dict(generated)
        for k, v in normalized.items():
            merged[k] = v
        body_to_send = merged

    params = {"key": NB_API_KEY}
    status, data = nb_post("/optimization/v2", params=params, body=body_to_send)

    st.session_state.last_json["vrp_create_payload"] = body_to_send
    st.session_state.last_json["vrp_create_response"] = data
    return status, data, body_to_send

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
                            rows.append({"label": f"Stop {len(ss.stops_df)+len(rows)+1}", "address": "", "lat": lat_f, "lng": lng_f})
                add_stops(rows, "Pasted lat/lng")

    with colB:
        if st.button("🗑️ Clear Stops", key="sb_clear"):
            clear_stops()

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

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", use_container_width=True):
                # pairwise totals via Distance Matrix along current order
                origins = stops_ll[:-1]
                dests = stops_ll[1:]
                status_m, data_m = distance_matrix(origins, dests, params_extra=GLOBAL_PARAMS)
                ss.last_json["matrix_pairs_before_raw"] = data_m

                dist_total, dur_total = parse_matrix_totals(data_m)
                ss.route_before["distance_m"] = dist_total
                ss.route_before["duration_s"] = dur_total

                status_d, resp_d = directions_multi_stop(stops_ll, params_extra=GLOBAL_PARAMS)
                ss.route_before["resp"] = resp_d
                ss.last_json["directions_before"] = resp_d
                ss.route_before["geometry"] = extract_route_geometry(resp_d)

                ss.mapsig["route_before"] += 1

            before_geom = ss.route_before.get("geometry") or []
            m_before = make_map(
                center=(float(ss.center["lat"]), float(ss.center["lng"])),
                stops=ss.stops_df,
                route_pts=before_geom if before_geom else None,
                clicked=ss.clicked_pin,
            )
            render_map(m_before, key=f"map_route_before_{ss.mapsig['route_before']}")

            bdist = ss.route_before.get("distance_m") or 0.0
            bdur = ss.route_before.get("duration_s") or 0.0
            st.metric("Before distance (km)", f"{bdist/1000:.2f}")
            st.metric("Before duration (min)", f"{bdur/60:.1f}")

        with right:
            st.markdown("### Step 2 — Optimization (VRP v2)")

            objective = st.selectbox("Optimization objective", ["distance", "duration"], index=0, key="vrp_obj")

            st.caption("Optional: Custom VRP request body override (JSON). Leave {} to use generated payload.")
            vrp_override = st.text_area("VRP override JSON", value="{}", height=140, key="vrp_override")

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", use_container_width=True):
                status_c, data_c, used_payload = vrp_create_v2(ss.stops_df, objective=objective, global_ui=global_ui, override_json_text=vrp_override)

                ss.last_json["vrp_create_payload_used"] = used_payload

                if status_c != 200:
                    st.error(f"VRP create failed: HTTP {status_c}")
                    st.json(data_c)
                else:
                    ss.vrp["create"] = data_c
                    job_id = (data_c.get("id") or data_c.get("job_id") or data_c.get("jobId"))
                    ss.vrp["job_id"] = job_id
                    if not job_id:
                        st.warning("VRP create succeeded, but no job id found in response.")
                    else:
                        # Poll
                        result = None
                        for _ in range(12):
                            stt, rr = vrp_result_v2(str(job_id))
                            if stt == 200 and isinstance(rr, dict):
                                result = rr
                                break
                            time.sleep(1.0)
                        ss.vrp["result"] = result

                        if result is None:
                            st.error("Could not fetch VRP result in time. Try 'Fetch VRP result again'.")
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

            if st.button("🔁 Fetch VRP result again", key="vrp_fetch_again", use_container_width=True):
                if ss.vrp.get("job_id"):
                    stt, rr = vrp_result_v2(str(ss.vrp["job_id"]))
                    ss.vrp["result"] = rr
                    if stt == 200:
                        order = parse_vrp_order(rr)
                        ss.vrp["order"] = order
                        if order:
                            df2 = ss.stops_df.copy().reset_index(drop=True)
                            valid = [i for i in order if 0 <= i < len(df2)]
                            missing = [i for i in range(len(df2)) if i not in valid]
                            final_order = valid + missing
                            ss._optimized_stops = df2.iloc[final_order].reset_index(drop=True)
                            st.success("Updated optimization order applied.")
                            ss.mapsig["route_after"] += 1
                    else:
                        st.error(f"Result fetch failed: HTTP {stt}")
                        st.json(rr)
                else:
                    st.info("No job id available yet. Run optimization first.")

            st.markdown("### Step 3 — Recompute Directions (After)")

            df_after = ss.get("_optimized_stops", None)
            if df_after is None:
                st.info("Run optimization to generate an optimized order.")
                df_after = ss.stops_df.copy()

            if st.button("🧭 Compute route (After)", key="rt_after_btn", use_container_width=True):
                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]

                status_m2, data_m2 = distance_matrix(ll_after[:-1], ll_after[1:], params_extra=GLOBAL_PARAMS)
                ss.last_json["matrix_pairs_after_raw"] = data_m2

                dist_total2, dur_total2 = parse_matrix_totals(data_m2)
                ss.route_after["distance_m"] = dist_total2
                ss.route_after["duration_s"] = dur_total2

                status_d2, resp_d2 = directions_multi_stop(ll_after, params_extra=GLOBAL_PARAMS)
                ss.route_after["resp"] = resp_d2
                ss.last_json["directions_after"] = resp_d2
                ss.route_after["geometry"] = extract_route_geometry(resp_d2)

                ss.mapsig["route_after"] += 1

            after_geom = ss.route_after.get("geometry") or []
            m_after = make_map(
                center=(float(ss.center["lat"]), float(ss.center["lng"])),
                stops=df_after,
                route_pts=after_geom if after_geom else None,
                clicked=ss.clicked_pin,
            )
            render_map(m_after, key=f"map_route_after_{ss.mapsig['route_after']}")

            adist = ss.route_after.get("distance_m") or 0.0
            adur = ss.route_after.get("duration_s") or 0.0
            st.metric("After distance (km)", f"{adist/1000:.2f}")
            st.metric("After duration (min)", f"{adur/60:.1f}")

            # Comparison
            b = float(ss.route_before.get("distance_m") or 0)
            a = float(ss.route_after.get("distance_m") or 0)
            if b > 0 and a > 0:
                saved = b - a
                pct = (saved / b * 100.0) if b else 0.0
                st.success(f"Distance saved: {saved/1000:.2f} km ({pct:.1f}%)")

            bt = float(ss.route_before.get("duration_s") or 0)
            at = float(ss.route_after.get("duration_s") or 0)
            if bt > 0 and at > 0:
                saved = bt - at
                pct = (saved / bt * 100.0) if bt else 0.0
                st.success(f"Time saved: {saved/60:.1f} min ({pct:.1f}%)")

        with st.expander("Download JSON (debug)", expanded=False):
            bundle = {
                "directions_before": ss.last_json.get("directions_before"),
                "directions_after": ss.last_json.get("directions_after"),
                "vrp_create_payload": ss.last_json.get("vrp_create_payload_used"),
                "vrp_create_response": ss.last_json.get("vrp_create_response"),
                "vrp_result_response": ss.last_json.get("vrp_result_response"),
                "matrix_pairs_before_raw": ss.last_json.get("matrix_pairs_before_raw"),
                "matrix_pairs_after_raw": ss.last_json.get("matrix_pairs_after_raw"),
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

        if st.button("📐 Compute Distance Matrix (NxN)", key="mx_btn", use_container_width=True):
            status, data = distance_matrix(ll, ll, params_extra=GLOBAL_PARAMS)
            ss.last_json["matrix_nxn"] = data
            ss.mapsig["matrix"] += 1
            if status != 200:
                st.error(f"Matrix failed: HTTP {status}")
                st.write(data)
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
                row_d = []
                row_t = []
                for e in elems:
                    d = e.get("distance")
                    t = e.get("duration")
                    dm = (d.get("value") if isinstance(d, dict) else d) or 0
                    ts = (t.get("value") if isinstance(t, dict) else t) or 0
                    row_d.append(float(dm) / 1000)
                    row_t.append(float(ts) / 60)
                dist_km.append(row_d)
                dur_min.append(row_t)

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
    st.subheader("Snap-to-Road + Isochrone (runs only on button click)")

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
            n_path = st.slider("N geometry points to send (first N)", min_value=2, max_value=max_pts, value=min(500, max_pts), step=1, key="snap_n")

        if st.button("🧷 Snap to Road", key="snap_btn", use_container_width=True):
            body = {"points": [{"lat": p[0], "lng": p[1]} for p in coords[: int(n_path)]]}
            params = {"key": NB_API_KEY}

            for ep in ["/snapToRoad/v2", "/snaptoroad/v2", "/snapToRoad"]:
                status, data = nb_post(ep, params=params, body=body)
                if status != 404:
                    break

            ss.last_json["snap"] = {"status": status, "response": data, "payload": body}
            ss.mapsig["snap"] += 1

            if status != 200:
                st.error(f"Snap-to-road failed: HTTP {status}")
                st.write(data)
            else:
                st.success("Snap-to-road computed (cached).")

        # Try to draw snapped geometry if returned
        snapped_pts: List[Tuple[float, float]] = []
        snap_resp = (ss.last_json.get("snap") or {}).get("response")
        if isinstance(snap_resp, dict):
            # common shapes: snappedPoints:[{location:{latitude,longitude}}]
            sp = snap_resp.get("snappedPoints") or snap_resp.get("snapped_points")
            if isinstance(sp, list):
                for it in sp:
                    loc = (it or {}).get("location") or {}
                    lat = loc.get("latitude") or loc.get("lat")
                    lng = loc.get("longitude") or loc.get("lng")
                    lat_f, lng_f = normalize_latlng(lat, lng)
                    if lat_f is not None:
                        snapped_pts.append((lat_f, lng_f))
            # or geojson-like
            if not snapped_pts:
                snapped_pts = extract_route_geometry(snap_resp)

        m5 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            route_pts=snapped_pts if snapped_pts else (coords[: int(n_path)] if coords else None),
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (best-effort)")
        iso_type = st.selectbox("Type", ["time", "distance"], index=0, key="iso_type")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")

        # Increase max limits as requested
        iso_val = st.number_input(
            "Value (seconds if time, meters if distance)",
            min_value=60,
            max_value=2_000_000,
            value=1800,
            step=60,
            key="iso_val",
        )

        if st.button("🟦 Compute Isochrone", key="iso_btn", use_container_width=True):
            params = {"key": NB_API_KEY}

            # Try your current working body first
            body1 = {
                "center": {"lat": float(ss.center["lat"]), "lng": float(ss.center["lng"])},
                "type": iso_type,
                "value": int(iso_val),
                "mode": iso_mode,
            }

            status, data = nb_post("/isochrone/v2", params=params, body=body1)
            if status == 404:
                status, data = nb_post("/isochrone", params=params, body=body1)

            ss.last_json["iso"] = {"status": status, "response": data, "payload": body1}
            ss.mapsig["iso"] += 1

            if status != 200:
                st.error(f"Isochrone failed: HTTP {status}")
                st.write(data)
            else:
                st.success("Isochrone computed (cached).")

        iso_resp = (ss.last_json.get("iso") or {}).get("response")
        iso_rings = extract_isoline_polygons(iso_resp)

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            clicked=ss.clicked_pin,
            iso_rings=iso_rings if iso_rings else None,
            zoom=12,
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

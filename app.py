# app.py
# NextBillion.ai — Visual API Tester (stable maps + 20+ stops + before/after optimize)
# Fixes:
# - Directions distance/time showing 0 (now parsed from directions route)
# - VRP v2 "locations.location missing" (now builds correct schema + auto-normalizes common wrong schemas)
# - Isochrone (correct GET /isochrone/json + draws polygon)
# - Snap-to-roads (correct /snapToRoads/json + draws snapped geometry)
# - Polyline decode IndexError (hardened decoder)
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

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        if not s or not str(s).strip():
            return None
        return json.loads(s)
    except Exception:
        return None

def deep_get(d: Any, path: List[Union[str, int]], default=None):
    cur = d
    for p in path:
        try:
            if isinstance(p, int) and isinstance(cur, list) and 0 <= p < len(cur):
                cur = cur[p]
            elif isinstance(p, str) and isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        except Exception:
            return default
    return cur

# -----------------------------
# HTTP (button-only + caching)
# -----------------------------
@dataclass(frozen=True)
class ReqSig:
    method: str
    url: str
    params_json: str
    body_json: str

def _sig(method: str, url: str, params: Dict[str, Any] | None, body: Any | None) -> ReqSig:
    pj = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    bj = json.dumps(body if body is not None else {}, sort_keys=True, separators=(",", ":"))
    return ReqSig(method=method.upper(), url=url, params_json=pj, body_json=bj)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_request(sig: ReqSig) -> Tuple[int, Any]:
    method = sig.method
    url = sig.url
    params = json.loads(sig.params_json)
    body = json.loads(sig.body_json)

    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=90)
        else:
            # for POST/PUT
            r = requests.post(url, params=params, json=body, timeout=90)

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

def nb_post(path: str, params: Dict[str, Any], body: Any) -> Tuple[int, Any]:
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

def extract_places_items(resp: Any) -> List[Dict[str, Any]]:
    return extract_geocode_candidates(resp)

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    # Hardened: never crash on malformed strings
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

    def read_value() -> Optional[int]:
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
        r1 = read_value()
        if r1 is None:
            break
        delta_lat = ~(r1 >> 1) if (r1 & 1) else (r1 >> 1)
        lat += delta_lat

        r2 = read_value()
        if r2 is None:
            break
        delta_lng = ~(r2 >> 1) if (r2 & 1) else (r2 >> 1)
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

    # Try common polyline keys
    poly = r0.get("geometry") or r0.get("polyline") or deep_get(r0, ["overview_polyline", "points"])
    if isinstance(poly, str) and poly.strip():
        # try polyline6 first; fallback to 5
        pts6 = decode_polyline(poly, precision=6)
        if pts6:
            return pts6
        pts5 = decode_polyline(poly, precision=5)
        if pts5:
            return pts5

    # legs / steps geometry (polyline per step)
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

def extract_route_totals(resp: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (distance_m, duration_s) from directions response if present.
    """
    if not isinstance(resp, dict):
        return None, None
    routes = resp.get("routes") or resp.get("route") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list) or not isinstance(routes[0], dict):
        return None, None
    r0 = routes[0]
    dist = r0.get("distance") or deep_get(r0, ["summary", "distance"])
    dur = r0.get("duration") or deep_get(r0, ["summary", "duration"])
    try:
        dist_m = float(dist) if dist is not None else None
    except Exception:
        dist_m = None
    try:
        dur_s = float(dur) if dur is not None else None
    except Exception:
        dur_s = None
    return dist_m, dur_s

def extract_matrix_pairs_totals(data_m: Any) -> Tuple[float, float]:
    """
    Matrix pairs call: rows[i].elements[0].distance.value (m), duration.value (s)
    """
    dist_total = 0.0
    dur_total = 0.0
    if not isinstance(data_m, dict):
        return dist_total, dur_total

    rows = data_m.get("rows") or []
    for row in rows:
        elems = (row or {}).get("elements") or []
        if not elems:
            continue
        e0 = elems[0]
        # Some schemas return numeric directly; prefer .value if dict
        d = e0.get("distance")
        t = e0.get("duration")
        if isinstance(d, dict):
            dist_total += float(d.get("value") or 0)
        else:
            dist_total += float(d or 0)

        if isinstance(t, dict):
            dur_total += float(t.get("value") or 0)
        else:
            dur_total += float(t or 0)

    return dist_total, dur_total

def extract_geojson_polygons(resp: Any) -> List[List[Tuple[float, float]]]:
    """
    For Isochrone GeoJSON FeatureCollection: polygons are [ [ [lng,lat], ... ] ]
    Return list of rings as lat,lng tuples (outer rings).
    """
    rings: List[List[Tuple[float, float]]] = []
    if not isinstance(resp, dict):
        return rings
    if resp.get("type") != "FeatureCollection":
        return rings
    feats = resp.get("features") or []
    for f in feats:
        geom = (f or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon" and isinstance(coords, list) and coords:
            # coords[0] is outer ring
            outer = coords[0]
            ring: List[Tuple[float, float]] = []
            for p in outer:
                if isinstance(p, list) and len(p) >= 2:
                    lng, lat = p[0], p[1]
                    lat_f, lng_f = normalize_latlng(lat, lng)
                    if lat_f is not None:
                        ring.append((lat_f, lng_f))
            if ring:
                rings.append(ring)
        elif gtype == "MultiPolygon" and isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list) and poly:
                    outer = poly[0]
                    ring: List[Tuple[float, float]] = []
                    for p in outer:
                        if isinstance(p, list) and len(p) >= 2:
                            lng, lat = p[0], p[1]
                            lat_f, lng_f = normalize_latlng(lat, lng)
                            if lat_f is not None:
                                ring.append((lat_f, lng_f))
                    if ring:
                        rings.append(ring)
    return rings

def extract_snap_geometry(resp: Any) -> List[Tuple[float, float]]:
    """
    Snap-to-roads returns GeoJSON sometimes in `geometry` or `snappedPoints`.
    We'll support:
    - FeatureCollection LineString
    - {"geometry": {"type":"LineString","coordinates":[[lng,lat],...]}}
    - {"snappedPoints":[{"location":{"latitude":..,"longitude":..}}, ...]}
    """
    if not isinstance(resp, dict):
        return []

    # FeatureCollection
    if resp.get("type") == "FeatureCollection":
        return extract_route_geometry(resp)

    geom = resp.get("geometry")
    if isinstance(geom, dict) and geom.get("type") == "LineString":
        pts = []
        for c in geom.get("coordinates") or []:
            if isinstance(c, list) and len(c) >= 2:
                lng, lat = c[0], c[1]
                lat_f, lng_f = normalize_latlng(lat, lng)
                if lat_f is not None:
                    pts.append((lat_f, lng_f))
        return pts

    snapped = resp.get("snappedPoints") or resp.get("snapped_points")
    if isinstance(snapped, list):
        pts = []
        for sp in snapped:
            loc = (sp or {}).get("location") or {}
            lat = loc.get("latitude") or loc.get("lat")
            lng = loc.get("longitude") or loc.get("lng")
            lat_f, lng_f = normalize_latlng(lat, lng)
            if lat_f is not None:
                pts.append((lat_f, lng_f))
        return pts

    return []

# -----------------------------
# MAP RENDERING (stable)
# -----------------------------
def make_map(
    center: Tuple[float, float],
    stops: pd.DataFrame,
    route_pts: Optional[List[Tuple[float, float]]] = None,
    clicked: Optional[Tuple[float, float]] = None,
    polygons: Optional[List[List[Tuple[float, float]]]] = None,
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True, tiles="OpenStreetMap")

    # Click marker (red)
    if clicked is not None:
        folium.Marker(
            location=[clicked[0], clicked[1]],
            popup="Pinned center",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    # Stops (numbered)
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

    # Polygons (Isochrone)
    if polygons:
        for ring in polygons:
            folium.Polygon(
                locations=ring,
                weight=3,
                opacity=0.8,
                fill=True,
                fill_opacity=0.2,
            ).add_to(m)

    # Route polyline + arrow
    if route_pts:
        folium.PolyLine(route_pts, weight=5, opacity=0.8).add_to(m)
        try:
            pl = folium.PolyLine(route_pts, weight=0, opacity=0)  # invisible base for arrows
            pl.add_to(m)
            PolyLineTextPath(pl, "  ➤  ", repeat=True, offset=7, attributes={"font-size": "14", "fill": "black"}).add_to(m)
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
        ss.clicked_pin = None  # (lat,lng)
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
                pd.DataFrame(
                    [{
                        "label": str(label),
                        "address": str(addr),
                        "lat": lat_f,
                        "lng": lng_f,
                        "source": source,
                    }]
                ),
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
    ss.vrp = {"create": None, "result": None, "order": None, "job_id": None}
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

def directions_multi_stop(order_latlng: List[str], params_extra: Dict[str, Any], custom_override: Optional[dict] = None) -> Tuple[int, Any]:
    """
    Directions endpoints are GET with params:
      origin, destination, waypoints=... (pipe-separated)
    """
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

    # Allow custom override for directions params (merge)
    if isinstance(custom_override, dict):
        params.update(custom_override)

    # try common endpoints in order
    status, data = nb_get("/directions/v2", params=params)
    if status == 404:
        status, data = nb_get("/navigation/v2", params=params)
    if status == 404:
        status, data = nb_get("/directions", params=params)
    return status, data

# -------- VRP v2 payload normalization (THIS is the big fix) --------
def normalize_vrp_v2_body(body: Any) -> Tuple[Optional[dict], Optional[str]]:
    """
    NextBillion Route Optimization v2 expects:
      locations: [ "lat,lng", ... ]   (index-based)
      jobs: [{id, location_index}, ...]
      vehicles: [{id, start_index, end_index}, ...]
      options: { objective: { travel_cost: "distance" | "duration" } }

    This function:
    - accepts correct dict
    - fixes common wrong formats like:
        {"locations":{"location":[...]} }
        {"locations":{"id":0,"location":[...]} }
        {"locations":[{"id":0,"location":"lat,lng"}, ...]}
    and converts them into the correct list-of-strings shape.
    """
    if not isinstance(body, dict):
        return None, "VRP body must be a JSON object"

    locs = body.get("locations")

    # Case A: correct already: list[str]
    if isinstance(locs, list) and all(isinstance(x, str) for x in locs):
        return body, None

    # Case B: locations is list[dict] with "location"
    if isinstance(locs, list) and all(isinstance(x, dict) for x in locs):
        loc_list = []
        for x in locs:
            loc = x.get("location")
            if isinstance(loc, str):
                loc_list.append(loc)
        if loc_list:
            body["locations"] = loc_list
            return body, None

    # Case C: locations is dict containing location list
    if isinstance(locs, dict):
        inner = locs.get("location") or locs.get("locations")
        if isinstance(inner, list) and all(isinstance(x, str) for x in inner):
            body["locations"] = inner
            return body, None
        # sometimes {"id":0,"location":[...]}
        if isinstance(inner, list):
            inner2 = [str(x) for x in inner if isinstance(x, (str, int, float))]
            if inner2:
                body["locations"] = inner2
                return body, None

    return None, "Could not normalize `locations`. Expected list of 'lat,lng' strings."

def build_vrp_v2_body_from_stops(stops: pd.DataFrame, objective: str) -> dict:
    loc_list = [latlng_str(float(r["lat"]), float(r["lng"])) for _, r in stops.reset_index(drop=True).iterrows()]
    jobs = [{"id": int(i), "location_index": int(i)} for i in range(len(loc_list))]
    vehicles = [{"id": "vehicle_1", "start_index": 0, "end_index": 0}]
    body = {
        "locations": loc_list,
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {"objective": {"travel_cost": objective}},
    }
    return body

def vrp_create_v2(stops: pd.DataFrame, objective: str, custom_override: Optional[Any] = None) -> Tuple[int, Any, Optional[dict]]:
    if len(stops) < 2:
        return 0, {"error": "Need at least 2 stops"}, None

    body = build_vrp_v2_body_from_stops(stops, objective=objective)

    # Apply override if provided (must be dict)
    if isinstance(custom_override, dict) and custom_override:
        # If user provides full body, use it; else shallow merge
        # Heuristic: if override has 'locations' or 'jobs' or 'vehicles', treat as full override
        if any(k in custom_override for k in ["locations", "jobs", "vehicles", "options"]):
            body = custom_override
        else:
            body.update(custom_override)

    fixed, err = normalize_vrp_v2_body(body)
    if err:
        return 0, {"error": err, "candidate_body": body}, None

    params = {"key": NB_API_KEY}
    status, data = nb_post("/optimization/v2", params=params, body=fixed)

    st.session_state.last_json["vrp_create_payload"] = fixed
    st.session_state.last_json["vrp_create_response"] = data

    return status, data, fixed

def vrp_result_v2(job_id: str) -> Tuple[int, Any]:
    params = {"key": NB_API_KEY, "id": job_id}
    status, data = nb_get("/optimization/v2/result", params=params)
    st.session_state.last_json["vrp_result_response"] = data
    return status, data

def parse_vrp_order(resp: Any) -> Optional[List[int]]:
    """
    Extract visit order as location indices.
    Supports:
      - resp.result.routes[0].steps[*].location_index
      - resp.routes[0].steps[*].location_index
    """
    if not isinstance(resp, dict):
        return None

    routes = resp.get("routes") or deep_get(resp, ["result", "routes"]) or deep_get(resp, ["solution", "routes"])
    if isinstance(routes, dict):
        routes = [routes]
    if not routes or not isinstance(routes, list):
        return None

    r0 = routes[0] if isinstance(routes[0], dict) else {}
    steps = r0.get("steps") or r0.get("activities") or r0.get("stops") or []
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
    return None

# -----------------------------
# ISOCHRONE (correct API)
# -----------------------------
def isochrone_get(center_lat: float, center_lng: float, mode: str, kind: str, value: int, extra_params: Optional[dict] = None) -> Tuple[int, Any]:
    """
    Isochrone API is GET:
      /isochrone/json?key=...&coordinates=lat,lng&mode=car&contours_minutes=5
    Only one of contours_minutes or contours_meters. 422 may occur due to underlying map issues. (NB docs)
    """
    params = {"key": NB_API_KEY, "coordinates": latlng_str(center_lat, center_lng), "mode": mode}

    if kind == "time":
        # value expected in minutes by API examples; accept seconds too (we convert if big)
        minutes = int(value)
        if minutes > 240:  # if user accidentally entered seconds
            minutes = max(1, int(round(minutes / 60.0)))
        params["contours_minutes"] = minutes
    else:
        meters = int(value)
        params["contours_meters"] = meters

    if isinstance(extra_params, dict):
        params.update(extra_params)

    return nb_get("/isochrone/json", params=params)

# -----------------------------
# SNAP TO ROADS (correct API)
# -----------------------------
def snap_to_roads(points: List[Tuple[float, float]], extra_params: Optional[dict] = None) -> Tuple[int, Any, dict]:
    """
    Snap To Roads API:
      POST https://api.nextbillion.io/snapToRoads/json?key=...
      Body example uses 'path' polyline-ish string or coordinates list depending on docs.
    We'll send `path` as "lat,lng|lat,lng|..."
    """
    params = {"key": NB_API_KEY}
    if isinstance(extra_params, dict):
        params.update(extra_params)

    path = "|".join([latlng_str(p[0], p[1]) for p in points])
    body = {"path": path, "interpolate": True}  # interpolate common option; safe if ignored

    status, data = nb_post("/snapToRoads/json", params=params, body=body)
    return status, data, body

# -----------------------------
# UI
# -----------------------------
ensure_state()
ss = st.session_state

st.title("NextBillion.ai — Visual API Tester")
st.caption(f"Stops loaded: {len(ss.stops_df)}")

# Sidebar: global route options (only applied when you click buttons)
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
            rows = []
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

# Tabs
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
        resp_sample = None
        for idx, row in missing.iterrows():
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            status, resp = geocode_forward(addr, country=ss.center["country"], language=language)
            resp_sample = resp
            candidates = extract_geocode_candidates(resp)
            if candidates:
                top = candidates[0]
                df.loc[idx, "lat"] = top["lat"]
                df.loc[idx, "lng"] = top["lng"]
                df.loc[idx, "source"] = f"Geocoded ({status})"
            else:
                df.loc[idx, "source"] = f"Geocode failed ({status})"

        replace_stops(df)
        ss.last_json["geocode_response_sample"] = resp_sample
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
                add_rows = []
                for i, it in enumerate(items[: int(kw_max)], start=1):
                    add_rows.append({"label": f"POI {i}", "address": it["name"], "lat": it["lat"], "lng": it["lng"]})
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

        # Custom override boxes
        with st.expander("Advanced: Custom request overrides (JSON) — optional", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                directions_override_txt = st.text_area("Directions override (query params JSON)", value="{}", height=120, key="adv_dir_override")
            with c2:
                vrp_override_txt = st.text_area("VRP override (full body JSON)", value="{}", height=120, key="adv_vrp_override")

        dir_override = safe_json_loads(directions_override_txt) or {}
        vrp_override = safe_json_loads(vrp_override_txt) or {}

        with left:
            st.markdown("### Step 1 — Directions (Before)")

            if st.button("🧭 Compute route (Before)", key="rt_before_btn", use_container_width=True):
                # Directions
                status_d, resp_d = directions_multi_stop(stops_ll, params_extra=GLOBAL_PARAMS, custom_override=dir_override)
                ss.route_before["resp"] = resp_d
                ss.last_json["directions_before"] = resp_d

                geom = extract_route_geometry(resp_d)
                ss.route_before["geometry"] = geom

                # Totals: prefer directions totals
                dist_m, dur_s = extract_route_totals(resp_d)

                # fallback to matrix pairs
                if dist_m is None or dur_s is None:
                    status_m, data_m = distance_matrix(stops_ll[:-1], stops_ll[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_before"] = data_m
                    d2, t2 = extract_matrix_pairs_totals(data_m)
                    dist_m = dist_m if dist_m is not None else d2
                    dur_s = dur_s if dur_s is not None else t2

                ss.route_before["distance_m"] = float(dist_m or 0.0)
                ss.route_before["duration_s"] = float(dur_s or 0.0)
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

            if st.button("⚙️ Run optimization (VRP v2)", key="vrp_run_btn", use_container_width=True):
                status_c, data_c, fixed_body = vrp_create_v2(ss.stops_df, objective=objective, custom_override=vrp_override)

                ss.last_json["vrp_create_payload"] = fixed_body
                ss.last_json["vrp_create_response"] = data_c

                if status_c != 200:
                    st.error(f"VRP create failed: HTTP {status_c}")
                    st.json(data_c)
                else:
                    ss.vrp["create"] = data_c
                    job_id = data_c.get("id") or data_c.get("job_id") or data_c.get("jobId")
                    ss.vrp["job_id"] = job_id
                    st.success(f"Optimization job created. Job id: {job_id}")

            if st.button("🔄 Fetch VRP result again", key="vrp_fetch_btn", use_container_width=True):
                job_id = ss.vrp.get("job_id")
                if not job_id:
                    st.warning("No job id yet. Run optimization first.")
                else:
                    # Poll with a few attempts (VRP can be async)
                    result = None
                    last = None
                    for _ in range(12):
                        stt, rr = vrp_result_v2(str(job_id))
                        last = rr
                        if stt == 200:
                            result = rr
                            break
                        time.sleep(1.0)

                    ss.vrp["result"] = result or last
                    if result is None:
                        st.error("Could not fetch VRP result (yet). Try again.")
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
                            st.warning("VRP result fetched, but order could not be parsed from result schema.")
                            st.json(result)

            st.markdown("### Step 3 — Recompute Directions (After)")

            df_after = ss._optimized_stops if ss._optimized_stops is not None else ss.stops_df.copy()
            if ss._optimized_stops is None:
                st.info("Run optimization + fetch result to apply an optimized order (otherwise After uses current order).")

            if st.button("🧭 Compute route (After)", key="rt_after_btn", use_container_width=True):
                ll_after = [latlng_str(float(r.lat), float(r.lng)) for r in df_after.itertuples()]

                status_d2, resp_d2 = directions_multi_stop(ll_after, params_extra=GLOBAL_PARAMS, custom_override=dir_override)
                ss.route_after["resp"] = resp_d2
                ss.last_json["directions_after"] = resp_d2

                geom2 = extract_route_geometry(resp_d2)
                ss.route_after["geometry"] = geom2

                dist_m2, dur_s2 = extract_route_totals(resp_d2)

                if dist_m2 is None or dur_s2 is None:
                    status_m2, data_m2 = distance_matrix(ll_after[:-1], ll_after[1:], params_extra=GLOBAL_PARAMS)
                    ss.last_json["matrix_pairs_after"] = data_m2
                    d2, t2 = extract_matrix_pairs_totals(data_m2)
                    dist_m2 = dist_m2 if dist_m2 is not None else d2
                    dur_s2 = dur_s2 if dur_s2 is not None else t2

                ss.route_after["distance_m"] = float(dist_m2 or 0.0)
                ss.route_after["duration_s"] = float(dur_s2 or 0.0)

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
            if ss.route_before.get("distance_m") is not None and ss.route_after.get("distance_m") is not None:
                b = float(ss.route_before["distance_m"] or 0)
                a = float(ss.route_after["distance_m"] or 0)
                if b > 0 and a > 0:
                    saved = b - a
                    pct = (saved / b * 100.0) if b > 0 else 0.0
                    st.success(f"Distance saved: {saved/1000:.2f} km ({pct:.1f}%)")

            if ss.route_before.get("duration_s") is not None and ss.route_after.get("duration_s") is not None:
                b = float(ss.route_before["duration_s"] or 0)
                a = float(ss.route_after["duration_s"] or 0)
                if b > 0 and a > 0:
                    saved = b - a
                    pct = (saved / b * 100.0) if b > 0 else 0.0
                    st.success(f"Time saved: {saved/60:.1f} min ({pct:.1f}%)")

        with st.expander("Download JSON (debug)", expanded=False):
            bundle = {
                "directions_before": ss.last_json.get("directions_before"),
                "directions_after": ss.last_json.get("directions_after"),
                "matrix_pairs_before": ss.last_json.get("matrix_pairs_before"),
                "matrix_pairs_after": ss.last_json.get("matrix_pairs_after"),
                "vrp_create_payload": ss.last_json.get("vrp_create_payload"),
                "vrp_create_response": ss.last_json.get("vrp_create_response"),
                "vrp_result_response": ss.last_json.get("vrp_result_response"),
                "parsed_vrp_order": ss.vrp.get("order"),
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
            n = st.slider("How many stops to include in NxN?", min_value=2, max_value=max_n, value=min(25, max_n), step=1, key="mx_n")

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
                dist_km.append([((e.get("distance") or {}).get("value") or e.get("distance") or 0) / 1000 for e in elems])
                dur_min.append([((e.get("duration") or {}).get("value") or e.get("duration") or 0) / 60 for e in elems])

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
    st.subheader("Snap-to-Road + Isochrone (correct endpoints; stable UI; runs only on button click)")

    with st.expander("Advanced: Custom request overrides (JSON) — optional", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            snap_override_txt = st.text_area("Snap-to-roads override (query/body merge JSON)", value="{}", height=110, key="adv_snap_override")
        with c2:
            iso_override_txt = st.text_area("Isochrone override (query params JSON)", value="{}", height=110, key="adv_iso_override")

    snap_override = safe_json_loads(snap_override_txt) or {}
    iso_override = safe_json_loads(iso_override_txt) or {}

    s1, s2 = st.columns([1, 1])

    with s1:
        st.markdown("### Snap-to-Roads")

        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use Directions (After) geometry"], index=0, key="snap_src")
        geom = ss.route_before.get("geometry") if src.startswith("Use Directions (Before)") else ss.route_after.get("geometry")
        coords = geom or []

        max_pts = len(coords) if coords else 0
        if max_pts < 2:
            st.caption("Not enough geometry points yet. Compute a route first.")
            n_path = 0
        else:
            n_path = st.slider(
                "N geometry points to send (first N)",
                min_value=2,
                max_value=max_pts,
                value=min(500, max_pts),
                step=1,
                key="snap_n",
            )

        snapped_pts: List[Tuple[float, float]] = []
        if st.button("🧷 Snap to Roads", key="snap_btn", use_container_width=True):
            pts = coords[: int(n_path)]
            status, data, payload = snap_to_roads(pts, extra_params=snap_override if isinstance(snap_override, dict) else None)
            ss.last_json["snap"] = {"status": status, "response": data, "payload": payload}
            ss.mapsig["snap"] += 1

            if status != 200:
                st.error(f"Snap-to-roads failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Snap-to-roads computed (cached).")

        snap_resp = deep_get(ss.last_json, ["snap", "response"])
        if isinstance(snap_resp, dict):
            snapped_pts = extract_snap_geometry(snap_resp)

        m5 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            route_pts=snapped_pts if snapped_pts else (coords[: int(n_path)] if coords and n_path else None),
            clicked=ss.clicked_pin,
            zoom=12,
        )
        render_map(m5, key=f"map_snap_{ss.mapsig['snap']}")

    with s2:
        st.markdown("### Isochrone (GET /isochrone/json)")

        iso_kind = st.selectbox("Type", ["time", "distance"], index=0, key="iso_kind")
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0, key="iso_mode")

        # Increase limits (you asked)
        if iso_kind == "time":
            iso_val = st.number_input("Value (minutes; if you paste seconds we auto-convert)", min_value=1, max_value=240, value=10, step=1, key="iso_val_min")
        else:
            iso_val = st.number_input("Value (meters)", min_value=100, max_value=200000, value=5000, step=100, key="iso_val_m")

        poly_rings: List[List[Tuple[float, float]]] = []
        if st.button("🟦 Compute Isochrone", key="iso_btn", use_container_width=True):
            status, data = isochrone_get(
                center_lat=float(ss.center["lat"]),
                center_lng=float(ss.center["lng"]),
                mode=iso_mode,
                kind=iso_kind,
                value=int(iso_val),
                extra_params=iso_override if isinstance(iso_override, dict) else None,
            )

            ss.last_json["iso"] = {"status": status, "response": data, "params": {"kind": iso_kind, "mode": iso_mode, "value": int(iso_val)}}
            ss.mapsig["iso"] += 1

            if status != 200:
                st.error(f"Isochrone failed: HTTP {status}")
                st.json(data)
            else:
                st.success("Isochrone computed (cached).")

        iso_resp = deep_get(ss.last_json, ["iso", "response"])
        if isinstance(iso_resp, dict):
            poly_rings = extract_geojson_polygons(iso_resp)

        m6 = make_map(
            center=(float(ss.center["lat"]), float(ss.center["lng"])),
            stops=ss.stops_df,
            clicked=ss.clicked_pin,
            polygons=poly_rings if poly_rings else None,
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

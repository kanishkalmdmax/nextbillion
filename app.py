# geocode_directions_nextbillion_visual_tester.py
# Streamlit Visual API Tester for NextBillion.ai
#
# Features (all stateful so maps/tables DON'T disappear):
# - Stops Manager: paste addresses or lat/lng; edit table; geocode (cached); numbered red markers
# - Places (Search + Generate Stops): search any region (country/state/city/locality) -> set center
#     -> generate random stops around center (NO keyword needed) OR search POIs (keyword) and add as stops
#     -> also supports generating around a pin-click on the map
# - Route + Optimize (Before vs After): multi-stop directions route map + totals (distance/duration)
#     -> run Optimization (VRP v2) -> get optimized stop order -> recompute directions -> second map
#     -> shows distance/time saved and % change
# - Distance Matrix (NxN) up to 20x20+ using POST
# - Snap-to-Road + Isochrone: draw a path (or use route geometry) -> snap -> show snapped polyline
#     -> isochrone polygon display
#
# Notes:
# - API endpoints are based on official docs:
#   - Directions: https://api.nextbillion.io/directions/json  (option=flexible supported)  (GET/POST)
#   - Distance Matrix: https://api.nextbillion.io/distancematrix/json (option=flexible supported) (GET/POST)
#   - Geocode: https://api.nextbillion.io/geocode
#   - Places Discover: https://api.nextbillion.io/discover  (Places docs “Discover”) – common NB endpoint
#   - Optimization: https://api.nextbillion.io/optimization/v2 (POST create, GET result)
#   - Snap To Roads: https://api.nextbillion.io/snapToRoads/json
#   - Isochrone: https://api.nextbillion.io/isochrone/json
#
# If any endpoint differs for your account/region, the app will surface the raw HTTP status + JSON.

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
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# =========================
# CONFIG
# =========================

API_KEY_DEFAULT = "a08a2b15af0f432c8e438403bc2b00e3"  # embedded as requested

NB_BASE = "https://api.nextbillion.io"

# Fallback lists for “avoid” / common route preferences to prevent spelling mistakes.
# (Docs mention tolls, highways, left/right turns, u-turns, service roads; plus bbox, etc.)
AVOID_PRESETS = [
    "toll",
    "highway",
    "ferry",
    "uturn",
    "left_turn",
    "right_turn",
    "service_road",
]

MODE_PRESETS = ["car", "truck", "motorbike", "bicycle", "walk"]

ROUTE_TYPE_PRESETS = ["fastest", "shortest"]

YESNO = ["false", "true"]


# =========================
# HELPERS
# =========================

def unix_to_human(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)

def human_to_unix(dt_str: str) -> Optional[int]:
    # Accepts: "YYYY-MM-DD HH:MM"
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(d2/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"

def safe_get(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        try:
            if isinstance(p, int) and isinstance(cur, list):
                cur = cur[p]
            elif isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return default
        except Exception:
            return default
        if cur is None:
            return default
    return cur

def nb_get(path: str, params: Dict[str, Any], timeout: int = 45) -> Tuple[int, Any]:
    url = NB_BASE + path
    try:
        r = requests.get(url, params=params, timeout=timeout)
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.status_code, r.json()
        return r.status_code, {"raw": r.text}
    except Exception as e:
        return 0, {"error": str(e)}

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any], timeout: int = 60) -> Tuple[int, Any]:
    url = NB_BASE + path
    try:
        r = requests.post(url, params=params, json=body, timeout=timeout)
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.status_code, r.json()
        return r.status_code, {"raw": r.text}
    except Exception as e:
        return 0, {"error": str(e)}

# Simple in-memory cache across reruns via session_state
def cache_get(cache_key: str):
    return st.session_state.get("_cache", {}).get(cache_key)

def cache_set(cache_key: str, value: Any):
    if "_cache" not in st.session_state:
        st.session_state["_cache"] = {}
    st.session_state["_cache"][cache_key] = value


# =========================
# POLYLINE DECODE (robust)
# =========================

def decode_polyline(polyline_str: Any, precision: int = 5) -> List[Tuple[float, float]]:
    """
    Robust decoder for Google-encoded polyline strings.
    Handles:
    - None -> []
    - dict -> tries common keys
    - list of coords -> returns normalized
    """
    if polyline_str is None:
        return []

    # If geometry already looks like coordinates
    if isinstance(polyline_str, list):
        if len(polyline_str) == 0:
            return []
        if isinstance(polyline_str[0], (list, tuple)) and len(polyline_str[0]) >= 2:
            # might be [lng,lat] or [lat,lng]; we assume [lat,lng] if abs(first)<=90 and abs(second)<=180
            pts = []
            for a, b in polyline_str:
                if abs(a) <= 90 and abs(b) <= 180:
                    pts.append((float(a), float(b)))
                else:
                    pts.append((float(b), float(a)))
            return pts
        return []

    if isinstance(polyline_str, dict):
        for k in ["polyline", "geometry", "encoded", "points"]:
            if k in polyline_str:
                polyline_str = polyline_str[k]
                break

    if not isinstance(polyline_str, str):
        # last resort: stringify
        polyline_str = str(polyline_str)

    # Remove any weird invisible chars that can break ord()
    polyline_str = "".join(ch for ch in polyline_str if ord(ch) >= 32)

    index = 0
    lat = 0
    lng = 0
    coordinates: List[Tuple[float, float]] = []
    factor = 10 ** precision

    try:
        while index < len(polyline_str):
            shift = 0
            result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            shift = 0
            result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            coordinates.append((lat / factor, lng / factor))
    except Exception:
        return []

    return coordinates


# =========================
# STATE INIT
# =========================

def ensure_state():
    st.session_state.setdefault("api_key", API_KEY_DEFAULT)

    st.session_state.setdefault("stops", [])  # list of dicts: {label,address,lat,lng,source}

    st.session_state.setdefault("center", {"lat": 28.6139, "lng": 77.2090, "label": "Delhi, India"})
    st.session_state.setdefault("country_filter", "IND")  # countryCode

    # results
    st.session_state.setdefault("geocode_result", None)
    st.session_state.setdefault("places_result", None)
    st.session_state.setdefault("places_map_data", None)

    st.session_state.setdefault("directions_before", None)
    st.session_state.setdefault("directions_after", None)

    st.session_state.setdefault("opt_create", None)
    st.session_state.setdefault("opt_result", None)
    st.session_state.setdefault("opt_order", None)

    st.session_state.setdefault("dm_result", None)

    st.session_state.setdefault("snap_result", None)
    st.session_state.setdefault("iso_result", None)

    st.session_state.setdefault("pin_center", None)  # from map click lat/lng


# =========================
# STOPS UTILITIES
# =========================

def stops_df() -> pd.DataFrame:
    rows = []
    for i, s in enumerate(st.session_state["stops"], start=1):
        rows.append({
            "#": i,
            "label": s.get("label", f"Stop {i}"),
            "address": s.get("address", ""),
            "lat": s.get("lat", None),
            "lng": s.get("lng", None),
            "source": s.get("source", ""),
        })
    return pd.DataFrame(rows)

def normalize_stops_from_text(lines: str, mode: str) -> List[Dict[str, Any]]:
    out = []
    raw = [l.strip() for l in lines.splitlines() if l.strip()]
    for i, line in enumerate(raw, start=1):
        if mode == "addresses":
            out.append({"label": f"Stop {i}", "address": line, "lat": None, "lng": None, "source": "addresses"})
        else:
            # lat,lng
            parts = [p.strip() for p in line.replace(" ", "").split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0]); lng = float(parts[1])
                    out.append({"label": f"Stop {i}", "address": "", "lat": lat, "lng": lng, "source": "latlng"})
                except Exception:
                    out.append({"label": f"Stop {i}", "address": line, "lat": None, "lng": None, "source": "invalid"})
            else:
                out.append({"label": f"Stop {i}", "address": line, "lat": None, "lng": None, "source": "invalid"})
    return out

def set_stops(new_stops: List[Dict[str, Any]], append: bool = False):
    if append:
        st.session_state["stops"].extend(new_stops)
    else:
        st.session_state["stops"] = new_stops
    # clear downstream results (because stops changed)
    st.session_state["directions_before"] = None
    st.session_state["directions_after"] = None
    st.session_state["opt_create"] = None
    st.session_state["opt_result"] = None
    st.session_state["opt_order"] = None
    st.session_state["dm_result"] = None
    st.session_state["snap_result"] = None
    st.session_state["iso_result"] = None

def stops_have_coords(stops: List[Dict[str, Any]]) -> bool:
    ok = 0
    for s in stops:
        if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float)):
            ok += 1
    return ok >= 2


# =========================
# API WRAPPERS (cached)
# =========================

def geocode_one(api_key: str, q: str, country: Optional[str], at_latlng: Optional[str]) -> Tuple[int, Any]:
    params = {"key": api_key, "q": q}
    if country:
        params["in"] = f"countryCode:{country}"
    if at_latlng:
        params["at"] = at_latlng
    return nb_get("/geocode", params=params, timeout=45)

def geocode_one_cached(api_key: str, q: str, country: Optional[str], at_latlng: Optional[str]) -> Tuple[int, Any]:
    ck = f"geocode|{country}|{at_latlng}|{q}"
    hit = cache_get(ck)
    if hit is not None:
        return hit
    res = geocode_one(api_key, q, country, at_latlng)
    cache_set(ck, res)
    return res

def discover_places(api_key: str, q: str, at_latlng: str, country: Optional[str], radius_m: int, limit: int = 20) -> Tuple[int, Any]:
    # Places “Discover” commonly uses in=circle:lat,lng;r=... and optional countryCode
    params = {"key": api_key, "q": q, "at": at_latlng, "limit": limit}
    params["in"] = f"circle:{at_latlng};r={int(radius_m)}"
    if country:
        params["in"] += f"&in=countryCode:{country}"
    return nb_get("/discover", params=params, timeout=45)

def directions_route(
    api_key: str,
    coords: List[Tuple[float, float]],
    option: str,
    mode: str,
    avoid: List[str],
    route_type: str,
    departure_time: Optional[int],
    alternatives: bool,
) -> Tuple[int, Any]:
    # Use POST to support 20+ waypoints reliably (docs: up to 200 waypoints POST).
    params = {"key": api_key}
    if option == "flexible":
        params["option"] = "flexible"

    origin = latlng_str(coords[0][0], coords[0][1])
    destination = latlng_str(coords[-1][0], coords[-1][1])
    waypoints = [latlng_str(a, b) for (a, b) in coords[1:-1]]

    body: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "waypoints": waypoints,
        "alternatives": alternatives,
    }
    if route_type:
        body["route_type"] = route_type
    if avoid:
        # For NB directions, avoid is typically a string list or pipe-separated; we send comma-separated.
        body["avoid"] = ",".join(avoid)
    if departure_time:
        body["departure_time"] = int(departure_time)

    return nb_post("/directions/json", params=params, body=body, timeout=60)

def distance_matrix(
    api_key: str,
    coords: List[Tuple[float, float]],
    option: str,
    mode: str,
    avoid: List[str],
    route_type: str,
    departure_time: Optional[int],
) -> Tuple[int, Any]:
    params = {"key": api_key}
    if option == "flexible":
        params["option"] = "flexible"

    locs = [latlng_str(a, b) for (a, b) in coords]
    body: Dict[str, Any] = {
        "origins": locs,
        "destinations": locs,
        "mode": mode,
    }
    if route_type:
        body["route_type"] = route_type
    if avoid:
        body["avoid"] = ",".join(avoid)
    if departure_time:
        body["departure_time"] = int(departure_time)

    return nb_post("/distancematrix/json", params=params, body=body, timeout=90)

def vrp_create_v2(api_key: str, locations: List[Dict[str, Any]], jobs: List[Dict[str, Any]], vehicles: List[Dict[str, Any]], objective: str) -> Tuple[int, Any]:
    # Based on tutorial + error you saw:
    # - expects locations as objects, each with a 'location' field
    # - options.objective expects an object (dto.ObjectiveOption), not a string.
    body = {
        "locations": locations,
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {
            "objective": {
                "travel_cost": objective  # "duration" or "distance"
            }
        }
    }
    return nb_post("/optimization/v2", params={"key": api_key}, body=body, timeout=90)

def vrp_result_v2(api_key: str, job_id: str) -> Tuple[int, Any]:
    return nb_get("/optimization/v2/result", params={"key": api_key, "id": job_id}, timeout=90)

def snap_to_roads(api_key: str, path_coords: List[Tuple[float, float]], interpolate: bool, include_geometry: bool) -> Tuple[int, Any]:
    params = {"key": api_key}
    # API expects "path" as pipe separated lat,lng pairs in query params for GET
    path_str = "|".join([latlng_str(a, b) for a, b in path_coords])
    params["path"] = path_str
    params["interpolate"] = str(interpolate).lower()
    params["include_geometry"] = str(include_geometry).lower()
    return nb_get("/snapToRoads/json", params=params, timeout=60)

def isochrone(api_key: str, center: Tuple[float, float], mode: str, contours_minutes: int, departure_time: Optional[int]) -> Tuple[int, Any]:
    params = {"key": api_key}
    params["location"] = latlng_str(center[0], center[1])
    params["mode"] = mode
    params["contours_minutes"] = int(contours_minutes)
    if departure_time:
        params["departure_time"] = int(departure_time)
    return nb_get("/isochrone/json", params=params, timeout=60)


# =========================
# MAP RENDERING
# =========================

def build_base_map(center_lat: float, center_lng: float, zoom: int = 11) -> folium.Map:
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, control_scale=True)
    return m

def add_numbered_markers(m: folium.Map, coords: List[Tuple[float, float]], labels: List[str]):
    for i, ((lat, lng), lab) in enumerate(zip(coords, labels), start=1):
        html = f"""
        <div style="
            background:#E53935;
            color:white;
            border-radius:18px;
            width:28px;height:28px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight:700;
            border:2px solid white;
            box-shadow:0 1px 4px rgba(0,0,0,0.3);
        ">{i}</div>
        """
        icon = folium.DivIcon(html=html)
        folium.Marker([lat, lng], tooltip=f"{i}. {lab}", icon=icon).add_to(m)

def add_polyline(m: folium.Map, coords: List[Tuple[float, float]]):
    if coords and len(coords) >= 2:
        folium.PolyLine([(a, b) for a, b in coords], weight=6, opacity=0.9).add_to(m)

def fit_map(m: folium.Map, coords: List[Tuple[float, float]]):
    if not coords:
        return
    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    sw = [min(lats), min(lngs)]
    ne = [max(lats), max(lngs)]
    m.fit_bounds([sw, ne])

def render_map_with_click(m: folium.Map, key: str, height: int = 520) -> Dict[str, Any]:
    # Unique key per map -> avoids StreamlitDuplicateElementKey
    return st_folium(m, height=height, use_container_width=True, key=key)


# =========================
# UI
# =========================

st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")
ensure_state()

st.title("NextBillion.ai — Visual API Tester")
st.caption("Stateful maps/tables (no disappearing). Multi-stop workflow: Stops → Directions → Optimize → Compare.")

with st.sidebar:
    st.header("Config")
    st.session_state["api_key"] = st.text_input("NextBillion API Key", value=st.session_state["api_key"], type="password")
    st.write(f"Stops loaded: **{len(st.session_state['stops'])}**")

    st.subheader("Global route options (Directions / Matrix / Optimize)")
    c1, c2 = st.columns(2)
    with c1:
        opt_option = st.selectbox("API option", ["fast", "flexible"], index=1)
        opt_mode = st.selectbox("Mode", MODE_PRESETS, index=0)
        opt_route_type = st.selectbox("Route type", ROUTE_TYPE_PRESETS, index=0)
    with c2:
        avoid_sel = st.multiselect("Avoid (safe presets)", AVOID_PRESETS, default=[])
        alt_routes = st.selectbox("Alternatives", YESNO, index=0) == "true"

    st.subheader("Departure time")
    dt_str = st.text_input("Departure (UTC) — YYYY-MM-DD HH:MM (optional)", value="")
    dep_unix = human_to_unix(dt_str) if dt_str.strip() else None
    if dep_unix:
        st.caption(f"Unix: {dep_unix}  •  Local: {unix_to_human(dep_unix)}")

tabs = st.tabs([
    "Stops Manager",
    "Geocode & Map",
    "Places (Search + Generate Stops)",
    "Route + Optimize (Before vs After)",
    "Distance Matrix (NxN)",
    "Snap-to-Road + Isochrone",
])


# =========================
# TAB 0 — Stops Manager
# =========================
with tabs[0]:
    st.subheader("Stops Manager (20+ stops supported)")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Easy input (no code)")
        mode = st.radio("How to input stops?", ["Addresses (one per line)", "Lat/Lng (one per line)"], index=0)
        sample = ""
        if mode.startswith("Addresses"):
            sample = "Connaught Place, New Delhi\nIndia Gate, New Delhi\nDLF Cyber Hub, Gurugram"
        else:
            sample = "28.6315,77.2167\n28.6129,77.2295\n28.4947,77.0884"
        txt = st.text_area("Paste at least 2 lines", value=sample, height=160)

        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Replace Stops", use_container_width=True):
                new_stops = normalize_stops_from_text(txt, "addresses" if mode.startswith("Addresses") else "latlng")
                set_stops(new_stops, append=False)
        with colB:
            if st.button("Add Stops", use_container_width=True):
                new_stops = normalize_stops_from_text(txt, "addresses" if mode.startswith("Addresses") else "latlng")
                set_stops(new_stops, append=True)
        with colC:
            if st.button("Clear Stops", use_container_width=True):
                set_stops([], append=False)

        st.markdown("---")
        st.markdown("#### Country filter (helps disambiguate geocoding)")
        st.session_state["country_filter"] = st.text_input("countryCode (e.g., IND, USA, DEU)", value=st.session_state["country_filter"])

    with right:
        st.markdown("#### Editable table")
        df = stops_df()
        if df.empty:
            st.info("Add stops from the left panel.")
        else:
            edited = st.data_editor(
                df.drop(columns=["#"]),
                use_container_width=True,
                num_rows="dynamic",
                key="stops_editor",
            )
            if st.button("Save table edits", use_container_width=True):
                new_list = []
                for i, row in edited.reset_index(drop=True).iterrows():
                    new_list.append({
                        "label": row.get("label") or f"Stop {i+1}",
                        "address": row.get("address") or "",
                        "lat": float(row["lat"]) if str(row.get("lat")).strip() not in ["", "None", "nan"] else None,
                        "lng": float(row["lng"]) if str(row.get("lng")).strip() not in ["", "None", "nan"] else None,
                        "source": row.get("source") or "table",
                    })
                set_stops(new_list, append=False)
                st.success("Saved.")

    st.markdown("---")
    st.markdown("#### Quick JSON export (your current stops)")
    st.download_button(
        "Download Stops JSON",
        data=json.dumps(st.session_state["stops"], indent=2),
        file_name="stops.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================
# TAB 1 — Geocode & Map
# =========================
with tabs[1]:
    st.subheader("Geocode your stops and show them on the map (stateful)")

    api_key = st.session_state["api_key"]
    stops = st.session_state["stops"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Geocode (cached to reduce API calls)")
        if st.button("Geocode all stops (cached)", use_container_width=True):
            center_at = latlng_str(st.session_state["center"]["lat"], st.session_state["center"]["lng"])
            updated = []
            for s in stops:
                if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float)):
                    updated.append(s)
                    continue
                q = (s.get("address") or "").strip()
                if not q:
                    updated.append(s)
                    continue
                status, data = geocode_one_cached(api_key, q, st.session_state["country_filter"], center_at)
                if status == 200 and isinstance(data, dict) and data.get("items"):
                    item0 = data["items"][0]
                    pos = item0.get("position") or {}
                    s2 = dict(s)
                    s2["lat"] = pos.get("lat")
                    s2["lng"] = pos.get("lng")
                    s2["address"] = safe_get(item0, ["address", "label"], s2.get("address", ""))
                    s2["source"] = "geocode"
                    updated.append(s2)
                else:
                    updated.append(s)
            set_stops(updated, append=False)
            st.session_state["geocode_result"] = {"status": 200, "note": "Completed geocoding (see table/map)."}
            st.success("Geocode completed (cached).")

        st.download_button(
            "Download Geocode+Stops JSON",
            data=json.dumps(st.session_state["stops"], indent=2),
            file_name="geocode_stops.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("#### Stops table")
        st.dataframe(stops_df(), use_container_width=True, height=320)

    with col2:
        st.markdown("#### Map (numbered red pins)")
        coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
        labels = [s.get("label") or s.get("address") or f"Stop {i+1}" for i, s in enumerate(stops) if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

        if coords:
            m = build_base_map(coords[0][0], coords[0][1], zoom=11)
            add_numbered_markers(m, coords, labels)
            fit_map(m, coords)
            render_map_with_click(m, key="map_geocode")
        else:
            st.info("No coordinates yet. Run geocode or enter lat/lng stops.")


# =========================
# TAB 2 — Places (Search + Generate Stops)
# =========================
with tabs[2]:
    st.subheader("Search any region → set center → generate random stops (no keyword required) OR add POIs")

    api_key = st.session_state["api_key"]

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Step 1 — Search Region (country/state/city/locality)")
        with st.form("region_form", clear_on_submit=False):
            region_q = st.text_input("Region query", value=st.session_state["center"]["label"])
            region_country = st.text_input("countryCode filter (optional)", value=st.session_state["country_filter"])
            submitted = st.form_submit_button("Search Region", use_container_width=True)

        if submitted:
            at_hint = None  # region search should be global; avoid narrowing too early
            status, data = geocode_one_cached(api_key, region_q, region_country.strip() or None, at_hint)
            st.session_state["places_result"] = {"region_search": {"status": status, "data": data}}
            if status == 200 and data.get("items"):
                items = data["items"][:10]
                options = []
                for it in items:
                    label = it.get("title") or safe_get(it, ["address", "label"], "Unknown")
                    pos = it.get("position") or {}
                    options.append((label, pos.get("lat"), pos.get("lng")))
                st.session_state["region_candidates"] = options
            else:
                st.session_state["region_candidates"] = []

        candidates = st.session_state.get("region_candidates", [])
        if candidates:
            pick = st.selectbox("Pick a region result", list(range(len(candidates))), format_func=lambda i: candidates[i][0])
            if st.button("Use picked region as center", use_container_width=True):
                label, lat, lng = candidates[pick]
                if lat is not None and lng is not None:
                    st.session_state["center"] = {"lat": float(lat), "lng": float(lng), "label": label}
                    st.success(f"Center set: {label} ({lat:.5f},{lng:.5f})")

        st.markdown("---")
        st.markdown("#### Step 2 — Generate random stops around center (no keyword needed)")
        gen_n = st.slider("How many stops to generate?", min_value=5, max_value=50, value=20, step=1)
        gen_radius = st.slider("Radius (meters)", min_value=500, max_value=30000, value=8000, step=500)

        if st.button("Generate random stops around center", use_container_width=True):
            c = st.session_state["center"]
            lat0, lng0 = c["lat"], c["lng"]
            new = []
            for i in range(gen_n):
                # random point in circle (roughly)
                r = gen_radius * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                dlat = (r * math.cos(theta)) / 111320.0
                dlng = (r * math.sin(theta)) / (111320.0 * math.cos(math.radians(lat0)) + 1e-9)
                lat = lat0 + dlat
                lng = lng0 + dlng
                new.append({
                    "label": f"Stop {len(st.session_state['stops']) + i + 1}",
                    "address": f"Random near {c['label']}",
                    "lat": lat,
                    "lng": lng,
                    "source": "random",
                })
            set_stops(new, append=True)
            st.success(f"Added {gen_n} random stops.")

    with right:
        c = st.session_state["center"]
        st.caption(f"Current center: **{c['label']}**  |  {c['lat']:.5f},{c['lng']:.5f}  |  countryCode: {st.session_state['country_filter']}")

        # Map: show center + current stops, allow click-to-set pin_center
        coords = [(s["lat"], s["lng"]) for s in st.session_state["stops"] if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
        labels = [s.get("label") or s.get("address") or "Stop" for s in st.session_state["stops"] if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

        m = build_base_map(c["lat"], c["lng"], zoom=12)
        folium.Marker([c["lat"], c["lng"]], tooltip="Center", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

        if coords:
            add_numbered_markers(m, coords, labels)
            fit_map(m, coords + [(c["lat"], c["lng"])])
        else:
            fit_map(m, [(c["lat"], c["lng"])])

        st.markdown("#### Pin-based generation (click map → generate around clicked pin)")
        out = render_map_with_click(m, key="map_places", height=520)
        click = out.get("last_clicked")
        if click:
            st.session_state["pin_center"] = {"lat": click["lat"], "lng": click["lng"]}
        pc = st.session_state.get("pin_center")
        if pc:
            st.caption(f"Pin center: {pc['lat']:.5f},{pc['lng']:.5f}")

        colX, colY = st.columns(2)
        with colX:
            pin_n = st.number_input("Stops around pin", min_value=1, max_value=100, value=20, step=1)
        with colY:
            pin_r = st.number_input("Pin radius (m)", min_value=100, max_value=50000, value=5000, step=100)

        if st.button("Generate stops around PIN", use_container_width=True, disabled=(pc is None)):
            lat0, lng0 = pc["lat"], pc["lng"]
            new = []
            for i in range(int(pin_n)):
                r = float(pin_r) * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                dlat = (r * math.cos(theta)) / 111320.0
                dlng = (r * math.sin(theta)) / (111320.0 * math.cos(math.radians(lat0)) + 1e-9)
                new.append({
                    "label": f"Stop {len(st.session_state['stops']) + i + 1}",
                    "address": "Random near pin",
                    "lat": lat0 + dlat,
                    "lng": lng0 + dlng,
                    "source": "pin_random",
                })
            set_stops(new, append=True)
            st.success(f"Added {pin_n} stops around pin.")

    st.markdown("---")
    st.markdown("### Optional: POI keyword search (Discover) and add results as stops")
    with st.form("poi_form", clear_on_submit=False):
        poi_q = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="")
        poi_radius = st.slider("Search radius (m)", min_value=500, max_value=30000, value=5000, step=500)
        poi_limit = st.slider("Max results", min_value=5, max_value=50, value=20, step=1)
        run_poi = st.form_submit_button("Search Places (POIs)", use_container_width=True)

    if run_poi:
        if not poi_q.strip():
            st.warning("Enter a keyword to use POI search. (Random stop generation works without keyword.)")
        else:
            at = latlng_str(st.session_state["center"]["lat"], st.session_state["center"]["lng"])
            status, data = discover_places(api_key, poi_q.strip(), at, st.session_state["country_filter"], poi_radius, poi_limit)
            st.session_state["places_result"] = {"poi_search": {"status": status, "data": data}}
            st.success(f"Places response: HTTP {status}")

    places_blob = st.session_state.get("places_result")
    if places_blob:
        st.download_button(
            "Download Places JSON",
            data=json.dumps(places_blob, indent=2),
            file_name="places.json",
            mime="application/json",
            use_container_width=True,
        )

    poi_data = safe_get(places_blob, ["poi_search", "data"], default=None) if places_blob else None
    items = (poi_data.get("items") if isinstance(poi_data, dict) else None) or []
    if items:
        rows = []
        for it in items:
            title = it.get("title") or safe_get(it, ["address", "label"], "")
            pos = it.get("position") or {}
            rows.append({"title": title, "lat": pos.get("lat"), "lng": pos.get("lng"), "id": it.get("id")})
        dfp = pd.DataFrame(rows)
        st.dataframe(dfp, use_container_width=True, height=260)

        if st.button("Add ALL POI results as stops", use_container_width=True):
            new = []
            for r in rows:
                if r["lat"] is None or r["lng"] is None:
                    continue
                new.append({
                    "label": f"Stop {len(st.session_state['stops']) + len(new) + 1}",
                    "address": r["title"],
                    "lat": float(r["lat"]),
                    "lng": float(r["lng"]),
                    "source": "places_discover",
                })
            set_stops(new, append=True)
            st.success(f"Added {len(new)} stops from POI results.")


# =========================
# TAB 3 — Route + Optimize (Before vs After)
# =========================
with tabs[3]:
    st.subheader("Route + Optimize (Before vs After) — Compare distance/time saved")

    api_key = st.session_state["api_key"]
    stops = st.session_state["stops"]

    coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
    labels = [s.get("label") or s.get("address") or "Stop" for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

    if len(coords) < 2:
        st.warning("Need at least 2 stops with coordinates. Use Geocode & Map or generate random stops.")
    else:
        st.markdown("#### Step 1 — Compute Directions route for current stop order")
        if st.button("Compute Route (Directions)", use_container_width=True):
            status, data = directions_route(
                api_key=api_key,
                coords=coords,
                option=("flexible" if opt_option == "flexible" else "fast"),
                mode=opt_mode,
                avoid=avoid_sel,
                route_type=opt_route_type,
                departure_time=dep_unix,
                alternatives=alt_routes,
            )
            st.session_state["directions_before"] = {"status": status, "data": data, "order": list(range(1, len(coords)+1))}
            st.success(f"Directions response: HTTP {status}")

        before = st.session_state.get("directions_before")
        if before:
            st.download_button(
                "Download Directions (Before) JSON",
                data=json.dumps(before, indent=2),
                file_name="directions_before.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Step 2 — Optimize (VRP v2) then recompute Directions for optimized order")

        objective = st.selectbox("Optimization objective (travel_cost)", ["duration", "distance"], index=0)

        # Build a basic VRP request from stops:
        # - location 0 is depot (first stop)
        # - remaining are jobs, each referencing a location by id
        if st.button("Run optimization (VRP v2)", use_container_width=True):
            locs = []
            for i, (lat, lng) in enumerate(coords):
                locs.append({"id": i, "location": latlng_str(lat, lng)})

            jobs = []
            # jobs for stops 1..n-1
            for i in range(1, len(coords)):
                jobs.append({
                    "id": f"job_{i}",
                    "location_index": i,
                    # You can add service, time windows, priority etc. later (edge cases)
                })

            vehicles = [{
                "id": "vehicle_1",
                "start_location_index": 0,
                "end_location_index": 0,
            }]

            status, data = vrp_create_v2(api_key, locs, jobs, vehicles, objective=objective)
            st.session_state["opt_create"] = {"status": status, "data": data}
            st.success(f"Optimization create: HTTP {status}")

            # Try extracting job_id from response
            job_id = data.get("id") or data.get("job_id") or data.get("result", {}).get("id")
            if job_id:
                st.session_state["opt_job_id"] = job_id
            else:
                st.session_state["opt_job_id"] = None

        opt_create = st.session_state.get("opt_create")
        if opt_create:
            st.download_button(
                "Download Optimization Create JSON",
                data=json.dumps(opt_create, indent=2),
                file_name="optimization_create.json",
                mime="application/json",
                use_container_width=True,
            )

        job_id = st.session_state.get("opt_job_id")
        if job_id:
            st.caption(f"Optimization job id: **{job_id}**")
            if st.button("Fetch Optimization Result", use_container_width=True):
                status, data = vrp_result_v2(api_key, job_id)
                st.session_state["opt_result"] = {"status": status, "data": data}
                st.success(f"Optimization result: HTTP {status}")

                # Extract an order (best-effort; schema can vary by account)
                # We look for a sequence of location indices or job ids.
                order = None

                # common patterns
                # 1) data.routes[0].steps[*].location_index
                steps = safe_get(data, ["routes", 0, "steps"], default=None)
                if isinstance(steps, list) and steps:
                    tmp = []
                    for stp in steps:
                        li = stp.get("location_index")
                        if li is not None:
                            tmp.append(int(li))
                    # convert to 1-based order for display
                    if tmp:
                        # keep unique while preserving sequence
                        seen = set()
                        seq = []
                        for x in tmp:
                            if x not in seen:
                                seq.append(x)
                                seen.add(x)
                        order = seq

                # 2) data.solution.routes[0].activities[*].location_index
                if order is None:
                    acts = safe_get(data, ["solution", "routes", 0, "activities"], default=None)
                    if isinstance(acts, list) and acts:
                        tmp = []
                        for a in acts:
                            li = a.get("location_index")
                            if li is not None:
                                tmp.append(int(li))
                        if tmp:
                            seen = set()
                            seq = []
                            for x in tmp:
                                if x not in seen:
                                    seq.append(x)
                                    seen.add(x)
                            order = seq

                # 3) data.routes[0].route (list of indices)
                if order is None:
                    rt = safe_get(data, ["routes", 0, "route"], default=None)
                    if isinstance(rt, list) and rt:
                        order = [int(x) for x in rt if isinstance(x, (int, float, str))]

                # Final: ensure depot start/end included
                if order:
                    # ensure start=0 present
                    if order[0] != 0:
                        order = [0] + [x for x in order if x != 0]
                    # ensure end=0
                    if order[-1] != 0:
                        order = order + [0]
                    st.session_state["opt_order"] = order
                else:
                    st.session_state["opt_order"] = None

        opt_result = st.session_state.get("opt_result")
        if opt_result:
            st.download_button(
                "Download Optimization Result JSON",
                data=json.dumps(opt_result, indent=2),
                file_name="optimization_result.json",
                mime="application/json",
                use_container_width=True,
            )

        order = st.session_state.get("opt_order")
        if order:
            st.markdown("**Optimized location_index order:** " + " → ".join(str(x) for x in order))

            # Convert optimized order indices to coords
            opt_coords = [coords[i] for i in order if 0 <= i < len(coords)]
            opt_labels = [labels[i] for i in order if 0 <= i < len(labels)]

            if st.button("Recompute Directions for optimized order", use_container_width=True):
                status, data = directions_route(
                    api_key=api_key,
                    coords=opt_coords,
                    option=("flexible" if opt_option == "flexible" else "fast"),
                    mode=opt_mode,
                    avoid=avoid_sel,
                    route_type=opt_route_type,
                    departure_time=dep_unix,
                    alternatives=False,
                )
                st.session_state["directions_after"] = {"status": status, "data": data, "order": order}
                st.success(f"Directions (After) response: HTTP {status}")

        after = st.session_state.get("directions_after")
        if after:
            st.download_button(
                "Download Directions (After) JSON",
                data=json.dumps(after, indent=2),
                file_name="directions_after.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Before vs After maps")

        m1c, m2c = st.columns(2)

        def totals_from_directions(resp: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], List[Tuple[float, float]]]:
            data = resp.get("data") if resp else None
            route = safe_get(data, ["routes", 0], default={}) if isinstance(data, dict) else {}
            dist_m = safe_get(route, ["distance"], default=None)
            dur_s = safe_get(route, ["duration"], default=None)
            geom = safe_get(route, ["geometry"], default=None) or safe_get(route, ["overview_polyline", "points"], default=None)
            pts = decode_polyline(geom) if geom else []
            return dist_m, dur_s, pts

        before_geom = []
        after_geom = []

        with m1c:
            st.markdown("#### Route (Before)")
            if before:
                dist_m, dur_s, pts = totals_from_directions(before)
                before_geom = pts
                if dist_m is not None and dur_s is not None:
                    st.metric("Distance (km)", f"{dist_m/1000:.2f}")
                    st.metric("Duration (min)", f"{dur_s/60:.1f}")

                m = build_base_map(coords[0][0], coords[0][1], zoom=11)
                add_numbered_markers(m, coords, labels)
                if pts:
                    add_polyline(m, pts)
                fit_map(m, coords)
                render_map_with_click(m, key="map_route_before")
            else:
                st.info("Compute Directions first.")

        with m2c:
            st.markdown("#### Route (After optimization)")
            if after and order:
                opt_coords = [coords[i] for i in order if 0 <= i < len(coords)]
                opt_labels = [labels[i] for i in order if 0 <= i < len(labels)]
                dist_m, dur_s, pts = totals_from_directions(after)
                after_geom = pts
                if dist_m is not None and dur_s is not None:
                    st.metric("Distance (km)", f"{dist_m/1000:.2f}")
                    st.metric("Duration (min)", f"{dur_s/60:.1f}")

                m = build_base_map(opt_coords[0][0], opt_coords[0][1], zoom=11)
                add_numbered_markers(m, opt_coords, opt_labels)
                if pts:
                    add_polyline(m, pts)
                fit_map(m, opt_coords)
                render_map_with_click(m, key="map_route_after")
            else:
                st.info("Run optimization + recompute directions for optimized order.")

        # Savings summary
        if before and after:
            bdist, bdur, _ = totals_from_directions(before)
            adist, adur, _ = totals_from_directions(after)
            if bdist and adist and bdur and adur:
                dist_saved = bdist - adist
                dur_saved = bdur - adur
                st.markdown("---")
                st.markdown("### Savings")
                cA, cB, cC, cD = st.columns(4)
                cA.metric("Distance saved (km)", f"{dist_saved/1000:.2f}")
                cB.metric("Time saved (min)", f"{dur_saved/60:.1f}")
                cC.metric("Distance % change", f"{(dist_saved/bdist*100):.1f}%")
                cD.metric("Time % change", f"{(dur_saved/bdur*100):.1f}%")


# =========================
# TAB 4 — Distance Matrix (NxN)
# =========================
with tabs[4]:
    st.subheader("Distance Matrix (NxN) — do 20x20+ (POST)")

    api_key = st.session_state["api_key"]
    stops = st.session_state["stops"]
    coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

    if len(coords) < 2:
        st.warning("Need at least 2 stops with coordinates.")
    else:
        st.caption("Tip: NxN cost grows quickly. Start with 20 stops for testing, then expand.")
        max_n = min(len(coords), 60)
        n = st.slider("Use first N stops (from your current stops list)", min_value=2, max_value=max_n, value=min(20, max_n), step=1)

        if st.button("Compute Distance Matrix (NxN)", use_container_width=True):
            use_coords = coords[:n]
            status, data = distance_matrix(
                api_key=api_key,
                coords=use_coords,
                option=("flexible" if opt_option == "flexible" else "fast"),
                mode=opt_mode,
                avoid=avoid_sel,
                route_type=opt_route_type,
                departure_time=dep_unix,
            )
            st.session_state["dm_result"] = {"status": status, "data": data, "n": n}
            st.success(f"Distance Matrix response: HTTP {status}")

        dm = st.session_state.get("dm_result")
        if dm:
            st.download_button(
                "Download Distance Matrix JSON",
                data=json.dumps(dm, indent=2),
                file_name="distance_matrix.json",
                mime="application/json",
                use_container_width=True,
            )

            data = dm.get("data", {})
            rows = data.get("rows", [])
            # Build a numeric table of distances (km) and durations (min)
            dist_tbl = []
            dur_tbl = []
            for r in rows:
                elems = r.get("elements", [])
                dist_row = []
                dur_row = []
                for e in elems:
                    dist_row.append((e.get("distance", {}) or {}).get("value", None))
                    dur_row.append((e.get("duration", {}) or {}).get("value", None))
                dist_tbl.append(dist_row)
                dur_tbl.append(dur_row)

            if dist_tbl and dur_tbl:
                dist_km = pd.DataFrame(dist_tbl).applymap(lambda x: None if x is None else round(x/1000, 2))
                dur_min = pd.DataFrame(dur_tbl).applymap(lambda x: None if x is None else round(x/60, 1))
                st.markdown("#### Distances (km)")
                st.dataframe(dist_km, use_container_width=True, height=340)
                st.markdown("#### Durations (minutes)")
                st.dataframe(dur_min, use_container_width=True, height=340)
            else:
                st.info("Response schema differed. Check JSON download.")


# =========================
# TAB 5 — Snap-to-Road + Isochrone
# =========================
with tabs[5]:
    st.subheader("Snap-to-Road + Isochrone (stateful maps)")

    api_key = st.session_state["api_key"]
    stops = st.session_state["stops"]
    coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Snap-to-Road")
        st.caption("Pick a path from your current route geometry, or from first N stops, then snap to roads.")

        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use first N stops"], index=0)
        n = st.slider("N stops for path (if using first N)", min_value=2, max_value=max(2, min(50, len(coords))), value=min(10, len(coords)) if len(coords) >= 2 else 2)

        interpolate = st.selectbox("Interpolate", YESNO, index=1) == "true"
        include_geometry = st.selectbox("Include geometry", YESNO, index=1) == "true"

        if st.button("Run Snap-to-Road", use_container_width=True):
            path_pts = []
            if src.startswith("Use Directions") and st.session_state.get("directions_before"):
                # decode geometry from directions_before
                before = st.session_state["directions_before"]
                route = safe_get(before, ["data", "routes", 0], default={})
                geom = route.get("geometry") or safe_get(route, ["overview_polyline", "points"], default=None)
                path_pts = decode_polyline(geom) if geom else []
                # If decode fails, fallback to stop coords
                if len(path_pts) < 2:
                    path_pts = coords[:n]
            else:
                path_pts = coords[:n]

            if len(path_pts) < 2:
                st.warning("Not enough points to snap.")
            else:
                status, data = snap_to_roads(api_key, path_pts, interpolate, include_geometry)
                st.session_state["snap_result"] = {"status": status, "data": data}
                st.success(f"Snap-to-road response: HTTP {status}")

        snap = st.session_state.get("snap_result")
        if snap:
            st.download_button(
                "Download Snap JSON",
                data=json.dumps(snap, indent=2),
                file_name="snap_to_road.json",
                mime="application/json",
                use_container_width=True,
            )

            data = snap.get("data", {})
            # Try reading snapped points + geometry
            snapped = []
            pts = data.get("snappedPoints") or data.get("snapped_points") or []
            for p in pts:
                loc = p.get("location") or {}
                lat = loc.get("latitude") or loc.get("lat")
                lng = loc.get("longitude") or loc.get("lng")
                if lat is not None and lng is not None:
                    snapped.append((float(lat), float(lng)))

            geom = data.get("geometry")
            geom_pts = decode_polyline(geom) if geom else []

            draw_pts = geom_pts if len(geom_pts) >= 2 else snapped

            if draw_pts:
                m = build_base_map(draw_pts[0][0], draw_pts[0][1], zoom=12)
                add_polyline(m, draw_pts)
                fit_map(m, draw_pts)
                render_map_with_click(m, key="map_snap")
            else:
                st.info("No geometry/snapped points found in response. Check JSON.")

    with c2:
        st.markdown("### Isochrone")
        st.caption("Compute reachable area polygon from center or pin; overlays polygon on map.")

        center_src = st.selectbox("Isochrone center", ["Use current center", "Use last clicked pin (Places map)"], index=0)
        if center_src.startswith("Use last") and st.session_state.get("pin_center"):
            cen = (st.session_state["pin_center"]["lat"], st.session_state["pin_center"]["lng"])
        else:
            c = st.session_state["center"]
            cen = (c["lat"], c["lng"])

        contours = st.slider("Contours minutes", min_value=5, max_value=120, value=30, step=5)
        iso_mode = st.selectbox("Mode (Isochrone)", MODE_PRESETS, index=0)

        if st.button("Run Isochrone", use_container_width=True):
            status, data = isochrone(api_key, cen, iso_mode, contours, dep_unix)
            st.session_state["iso_result"] = {"status": status, "data": data, "center": cen}
            st.success(f"Isochrone response: HTTP {status}")

        iso = st.session_state.get("iso_result")
        if iso:
            st.download_button(
                "Download Isochrone JSON",
                data=json.dumps(iso, indent=2),
                file_name="isochrone.json",
                mime="application/json",
                use_container_width=True,
            )

            data = iso.get("data", {})
            # Common: "polygons":[{"coordinates":[[[lng,lat],...]]}]
            polys = data.get("polygons") or data.get("features") or []
            poly_coords = None

            if isinstance(polys, list) and polys:
                # try polygons schema
                p0 = polys[0]
                coords0 = p0.get("coordinates")
                if coords0:
                    poly_coords = coords0
                # GeoJSON feature
                if poly_coords is None and p0.get("geometry"):
                    poly_coords = p0["geometry"].get("coordinates")

            m = build_base_map(cen[0], cen[1], zoom=12)
            folium.Marker([cen[0], cen[1]], tooltip="Isochrone center", icon=folium.Icon(color="blue")).add_to(m)

            if poly_coords:
                # normalize: take first ring
                ring = None
                # could be [[[lng,lat]...]] or [[lng,lat]...]
                if isinstance(poly_coords, list) and poly_coords:
                    if isinstance(poly_coords[0], list) and poly_coords and isinstance(poly_coords[0][0], list):
                        ring = poly_coords[0]
                    else:
                        ring = poly_coords

                if ring and len(ring) >= 3:
                    latlngs = [(pt[1], pt[0]) for pt in ring if isinstance(pt, list) and len(pt) >= 2]
                    folium.Polygon(latlngs, opacity=0.6, weight=3).add_to(m)
                    fit_map(m, latlngs + [cen])
                render_map_with_click(m, key="map_iso")
            else:
                render_map_with_click(m, key="map_iso_center")
                st.info("No polygon found in response (check JSON).")


# Footer debug panel
with st.expander("Debug: current session_state keys", expanded=False):
    st.json({
        "center": st.session_state["center"],
        "country_filter": st.session_state["country_filter"],
        "stops_count": len(st.session_state["stops"]),
        "has_directions_before": st.session_state["directions_before"] is not None,
        "has_opt_result": st.session_state["opt_result"] is not None,
        "has_dm_result": st.session_state["dm_result"] is not None,
    })

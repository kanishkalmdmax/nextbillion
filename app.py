# geocode_directions_nextbillion_visual_tester.py
# ✅ Updated to fix:
# 1) StreamlitAPIException: slider min_value must be < max_value (when values are 2 and 2)
# 2) Streamlit warning: use_container_width -> width="stretch" (for buttons + downloads where supported)
#
# NOTE: Some Streamlit elements still only accept use_container_width in certain versions.
# This file uses width="stretch" for st.button / st.download_button where supported.

import json
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

import folium
from streamlit_folium import st_folium


# =========================
# CONFIG
# =========================
API_KEY_DEFAULT = "a08a2b15af0f432c8e438403bc2b00e3"  # embedded as requested
NB_BASE = "https://api.nextbillion.io"

AVOID_PRESETS = ["toll", "highway", "ferry", "uturn", "left_turn", "right_turn", "service_road"]
MODE_PRESETS = ["car", "truck", "motorbike", "bicycle", "walk"]
ROUTE_TYPE_PRESETS = ["fastest", "shortest"]
YESNO = ["false", "true"]


# =========================
# HELPERS
# =========================
def wstretch() -> Dict[str, Any]:
    return {"width": "stretch"}

def unix_to_human(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)

def human_to_unix(dt_str: str) -> Optional[int]:
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None

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
    if polyline_str is None:
        return []

    if isinstance(polyline_str, list):
        if len(polyline_str) == 0:
            return []
        if isinstance(polyline_str[0], (list, tuple)) and len(polyline_str[0]) >= 2:
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
        polyline_str = str(polyline_str)

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
    ss = st.session_state
    ss.setdefault("api_key", API_KEY_DEFAULT)
    ss.setdefault("stops", [])
    ss.setdefault("center", {"lat": 28.6139, "lng": 77.2090, "label": "Delhi, India"})
    ss.setdefault("country_filter", "IND")
    ss.setdefault("geocode_result", None)
    ss.setdefault("places_result", None)
    ss.setdefault("directions_before", None)
    ss.setdefault("directions_after", None)
    ss.setdefault("opt_create", None)
    ss.setdefault("opt_result", None)
    ss.setdefault("opt_order", None)
    ss.setdefault("dm_result", None)
    ss.setdefault("snap_result", None)
    ss.setdefault("iso_result", None)
    ss.setdefault("pin_center", None)
    ss.setdefault("region_candidates", [])
    ss.setdefault("opt_job_id", None)


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

    # reset downstream outputs
    st.session_state["directions_before"] = None
    st.session_state["directions_after"] = None
    st.session_state["opt_create"] = None
    st.session_state["opt_result"] = None
    st.session_state["opt_order"] = None
    st.session_state["opt_job_id"] = None
    st.session_state["dm_result"] = None
    st.session_state["snap_result"] = None
    st.session_state["iso_result"] = None


# =========================
# API WRAPPERS (cached)
# =========================
def geocode_one_cached(api_key: str, q: str, country: Optional[str], at_latlng: Optional[str]) -> Tuple[int, Any]:
    ck = f"geocode|{country}|{at_latlng}|{q}"
    hit = cache_get(ck)
    if hit is not None:
        return hit
    params = {"key": api_key, "q": q}
    if country:
        params["in"] = f"countryCode:{country}"
    if at_latlng:
        params["at"] = at_latlng
    res = nb_get("/geocode", params=params, timeout=45)
    cache_set(ck, res)
    return res

def discover_places(api_key: str, q: str, at_latlng: str, country: Optional[str], radius_m: int, limit: int = 20) -> Tuple[int, Any]:
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
    body: Dict[str, Any] = {"origins": locs, "destinations": locs, "mode": mode}
    if route_type:
        body["route_type"] = route_type
    if avoid:
        body["avoid"] = ",".join(avoid)
    if departure_time:
        body["departure_time"] = int(departure_time)

    return nb_post("/distancematrix/json", params=params, body=body, timeout=90)

def vrp_create_v2(api_key: str, locations: List[Dict[str, Any]], jobs: List[Dict[str, Any]], vehicles: List[Dict[str, Any]], objective: str) -> Tuple[int, Any]:
    body = {
        "locations": locations,
        "jobs": jobs,
        "vehicles": vehicles,
        "options": {"objective": {"travel_cost": objective}},  # ✅ correct type (object)
    }
    return nb_post("/optimization/v2", params={"key": api_key}, body=body, timeout=90)

def vrp_result_v2(api_key: str, job_id: str) -> Tuple[int, Any]:
    return nb_get("/optimization/v2/result", params={"key": api_key, "id": job_id}, timeout=90)

def snap_to_roads(api_key: str, path_coords: List[Tuple[float, float]], interpolate: bool, include_geometry: bool) -> Tuple[int, Any]:
    params = {"key": api_key}
    params["path"] = "|".join([latlng_str(a, b) for a, b in path_coords])
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
    return folium.Map(location=[center_lat, center_lng], zoom_start=zoom, control_scale=True)

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
    m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

def render_map_with_click(m: folium.Map, key: str, height: int = 520) -> Dict[str, Any]:
    return st_folium(m, height=height, use_container_width=True, key=key)  # st_folium still uses this


# =========================
# APP
# =========================
st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")
ensure_state()

st.title("NextBillion.ai — Visual API Tester")
st.caption("Stateful maps/tables (no disappearing). Stops → Directions → Optimize → Compare.")

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
        mode = st.radio("How to input stops?", ["Addresses (one per line)", "Lat/Lng (one per line)"], index=0)
        sample = "Connaught Place, New Delhi\nIndia Gate, New Delhi\nDLF Cyber Hub, Gurugram" if mode.startswith("Addresses") else "28.6315,77.2167\n28.6129,77.2295\n28.4947,77.0884"
        txt = st.text_area("Paste at least 2 lines", value=sample, height=160)

        a, b, c = st.columns(3)
        with a:
            if st.button("Replace Stops", **wstretch()):
                new_stops = normalize_stops_from_text(txt, "addresses" if mode.startswith("Addresses") else "latlng")
                set_stops(new_stops, append=False)
        with b:
            if st.button("Add Stops", **wstretch()):
                new_stops = normalize_stops_from_text(txt, "addresses" if mode.startswith("Addresses") else "latlng")
                set_stops(new_stops, append=True)
        with c:
            if st.button("Clear Stops", **wstretch()):
                set_stops([], append=False)

        st.session_state["country_filter"] = st.text_input("countryCode (e.g., IND, USA, DEU)", value=st.session_state["country_filter"])

    with right:
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
            if st.button("Save table edits", **wstretch()):
                new_list = []
                for i, row in edited.reset_index(drop=True).iterrows():
                    latv = row.get("lat")
                    lngv = row.get("lng")
                    new_list.append({
                        "label": row.get("label") or f"Stop {i+1}",
                        "address": row.get("address") or "",
                        "lat": float(latv) if str(latv).strip() not in ["", "None", "nan"] else None,
                        "lng": float(lngv) if str(lngv).strip() not in ["", "None", "nan"] else None,
                        "source": row.get("source") or "table",
                    })
                set_stops(new_list, append=False)
                st.success("Saved.")

    st.download_button(
        "Download Stops JSON",
        data=json.dumps(st.session_state["stops"], indent=2),
        file_name="stops.json",
        mime="application/json",
        **wstretch(),
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
        if st.button("Geocode all stops (cached)", **wstretch()):
            center_at = latlng_str(st.session_state["center"]["lat"], st.session_state["center"]["lng"])
            updated = []
            for s in stops:
                if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float)):
                    updated.append(s); continue
                q = (s.get("address") or "").strip()
                if not q:
                    updated.append(s); continue
                status, data = geocode_one_cached(api_key, q, st.session_state["country_filter"], center_at)
                if status == 200 and isinstance(data, dict) and data.get("items"):
                    it0 = data["items"][0]
                    pos = it0.get("position") or {}
                    s2 = dict(s)
                    s2["lat"] = pos.get("lat")
                    s2["lng"] = pos.get("lng")
                    s2["address"] = safe_get(it0, ["address", "label"], s2.get("address", ""))
                    s2["source"] = "geocode"
                    updated.append(s2)
                else:
                    updated.append(s)
            set_stops(updated, append=False)
            st.success("Geocode completed (cached).")

        st.download_button(
            "Download Geocode+Stops JSON",
            data=json.dumps(st.session_state["stops"], indent=2),
            file_name="geocode_stops.json",
            mime="application/json",
            **wstretch(),
        )
        st.dataframe(stops_df(), use_container_width=True, height=320)

    with col2:
        coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
        labels = [s.get("label") or s.get("address") or "Stop" for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
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
        with st.form("region_form", clear_on_submit=False):
            region_q = st.text_input("Region query", value=st.session_state["center"]["label"])
            region_country = st.text_input("countryCode filter (optional)", value=st.session_state["country_filter"])
            submitted = st.form_submit_button("Search Region", **wstretch())

        if submitted:
            status, data = geocode_one_cached(api_key, region_q, region_country.strip() or None, None)
            if status == 200 and data.get("items"):
                items = data["items"][:10]
                st.session_state["region_candidates"] = [
                    (
                        (it.get("title") or safe_get(it, ["address", "label"], "Unknown")),
                        (it.get("position") or {}).get("lat"),
                        (it.get("position") or {}).get("lng"),
                    )
                    for it in items
                ]
            else:
                st.session_state["region_candidates"] = []
                st.warning(f"Region search HTTP {status}")

        candidates = st.session_state.get("region_candidates", [])
        if candidates:
            pick = st.selectbox("Pick a region result", list(range(len(candidates))), format_func=lambda i: candidates[i][0])
            if st.button("Use picked region as center", **wstretch()):
                label, lat, lng = candidates[pick]
                if lat is not None and lng is not None:
                    st.session_state["center"] = {"lat": float(lat), "lng": float(lng), "label": label}
                    st.success(f"Center set: {label} ({lat:.5f},{lng:.5f})")

        st.markdown("---")
        gen_n = st.slider("How many stops to generate?", min_value=5, max_value=50, value=20, step=1)
        gen_radius = st.slider("Radius (meters)", min_value=500, max_value=30000, value=8000, step=500)

        if st.button("Generate random stops around center", **wstretch()):
            c = st.session_state["center"]
            lat0, lng0 = c["lat"], c["lng"]
            new = []
            for i in range(gen_n):
                r = gen_radius * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                dlat = (r * math.cos(theta)) / 111320.0
                dlng = (r * math.sin(theta)) / (111320.0 * math.cos(math.radians(lat0)) + 1e-9)
                new.append({
                    "label": f"Stop {len(st.session_state['stops']) + i + 1}",
                    "address": f"Random near {c['label']}",
                    "lat": lat0 + dlat,
                    "lng": lng0 + dlng,
                    "source": "random",
                })
            set_stops(new, append=True)
            st.success(f"Added {gen_n} random stops.")

    with right:
        c = st.session_state["center"]
        st.caption(f"Current center: **{c['label']}** | {c['lat']:.5f},{c['lng']:.5f} | countryCode: {st.session_state['country_filter']}")

        coords = [(s["lat"], s["lng"]) for s in st.session_state["stops"] if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]
        labels = [s.get("label") or s.get("address") or "Stop" for s in st.session_state["stops"] if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

        m = build_base_map(c["lat"], c["lng"], zoom=12)
        folium.Marker([c["lat"], c["lng"]], tooltip="Center", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        if coords:
            add_numbered_markers(m, coords, labels)
            fit_map(m, coords + [(c["lat"], c["lng"])])
        else:
            fit_map(m, [(c["lat"], c["lng"])])

        out = render_map_with_click(m, key="map_places", height=520)
        click = out.get("last_clicked")
        if click:
            st.session_state["pin_center"] = {"lat": click["lat"], "lng": click["lng"]}

        pc = st.session_state.get("pin_center")
        if pc:
            st.caption(f"Pin center: {pc['lat']:.5f},{pc['lng']:.5f}")

        pin_n = st.number_input("Stops around pin", min_value=1, max_value=100, value=20, step=1)
        pin_r = st.number_input("Pin radius (m)", min_value=100, max_value=50000, value=5000, step=100)

        if st.button("Generate stops around PIN", disabled=(pc is None), **wstretch()):
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
        run_poi = st.form_submit_button("Search Places (POIs)", **wstretch())

    if run_poi:
        if not poi_q.strip():
            st.warning("Enter a keyword for POI search. (Random stop generation works without keyword.)")
        else:
            at = latlng_str(st.session_state["center"]["lat"], st.session_state["center"]["lng"])
            status, data = discover_places(api_key, poi_q.strip(), at, st.session_state["country_filter"], poi_radius, poi_limit)
            st.session_state["places_result"] = {"poi_search": {"status": status, "data": data}}
            st.success(f"Places response: HTTP {status}")

    if st.session_state.get("places_result"):
        st.download_button(
            "Download Places JSON",
            data=json.dumps(st.session_state["places_result"], indent=2),
            file_name="places.json",
            mime="application/json",
            **wstretch(),
        )


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
        if st.button("Compute Route (Directions)", **wstretch()):
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
            st.session_state["directions_before"] = {"status": status, "data": data, "order": list(range(1, len(coords) + 1))}
            st.success(f"Directions response: HTTP {status}")

        objective = st.selectbox("Optimization objective (travel_cost)", ["duration", "distance"], index=0)

        if st.button("Run optimization (VRP v2)", **wstretch()):
            locs = [{"id": i, "location": latlng_str(lat, lng)} for i, (lat, lng) in enumerate(coords)]
            jobs = [{"id": f"job_{i}", "location_index": i} for i in range(1, len(coords))]
            vehicles = [{"id": "vehicle_1", "start_location_index": 0, "end_location_index": 0}]

            status, data = vrp_create_v2(api_key, locs, jobs, vehicles, objective=objective)
            st.session_state["opt_create"] = {"status": status, "data": data}
            st.success(f"Optimization create: HTTP {status}")

            job_id = data.get("id") or data.get("job_id") or safe_get(data, ["result", "id"], None)
            st.session_state["opt_job_id"] = job_id

        job_id = st.session_state.get("opt_job_id")
        if job_id:
            if st.button("Fetch Optimization Result", **wstretch()):
                status, data = vrp_result_v2(api_key, job_id)
                st.session_state["opt_result"] = {"status": status, "data": data}
                st.success(f"Optimization result: HTTP {status}")

        # Two maps:
        cA, cB = st.columns(2)

        def totals_and_geom(resp: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], List[Tuple[float, float]]]:
            data = resp.get("data") if resp else None
            route = safe_get(data, ["routes", 0], default={}) if isinstance(data, dict) else {}
            dist_m = safe_get(route, ["distance"], default=None)
            dur_s = safe_get(route, ["duration"], default=None)
            geom = route.get("geometry") or safe_get(route, ["overview_polyline", "points"], default=None)
            pts = decode_polyline(geom) if geom else []
            return dist_m, dur_s, pts

        before = st.session_state.get("directions_before")

        with cA:
            st.markdown("#### Route (Before)")
            if before:
                dist_m, dur_s, pts = totals_and_geom(before)
                if dist_m is not None:
                    st.metric("Distance (km)", f"{dist_m/1000:.2f}")
                if dur_s is not None:
                    st.metric("Duration (min)", f"{dur_s/60:.1f}")

                m = build_base_map(coords[0][0], coords[0][1], zoom=11)
                add_numbered_markers(m, coords, labels)
                if pts:
                    add_polyline(m, pts)
                fit_map(m, coords)
                render_map_with_click(m, key="map_route_before")
            else:
                st.info("Compute Directions first.")

        with cB:
            st.markdown("#### Route (After) — (requires your optimized order extraction)")
            st.caption("Once your optimization schema is confirmed, we can draw the optimized order + route here.")


# =========================
# TAB 4 — Distance Matrix (NxN)
# =========================
with tabs[4]:
    st.subheader("Distance Matrix (NxN) — 20x20+ (POST)")
    api_key = st.session_state["api_key"]
    stops = st.session_state["stops"]
    coords = [(s["lat"], s["lng"]) for s in stops if isinstance(s.get("lat"), (int, float)) and isinstance(s.get("lng"), (int, float))]

    if len(coords) < 2:
        st.warning("Need at least 2 stops with coordinates.")
    else:
        max_n = min(len(coords), 60)
        n = st.slider("Use first N stops", min_value=2, max_value=max_n, value=min(20, max_n), step=1)

        if st.button("Compute Distance Matrix (NxN)", **wstretch()):
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
                **wstretch(),
            )


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
        src = st.selectbox("Path source", ["Use Directions (Before) geometry", "Use first N stops"], index=0)

        # ✅ FIX: slider must have min < max. If only 2 points exist, don't render slider at all.
        max_n = max(2, min(50, len(coords)))
        if max_n <= 2:
            n_path = 2
            st.info("Only 2 coordinate points available — using N=2 for snap path.")
        else:
            n_path = st.slider("N stops for path (if using first N)", min_value=2, max_value=max_n, value=min(10, max_n), step=1)

        interpolate = st.selectbox("Interpolate", YESNO, index=1) == "true"
        include_geometry = st.selectbox("Include geometry", YESNO, index=1) == "true"

        if st.button("Run Snap-to-Road", **wstretch()):
            path_pts = []
            if src.startswith("Use Directions") and st.session_state.get("directions_before"):
                before = st.session_state["directions_before"]
                route = safe_get(before, ["data", "routes", 0], default={})
                geom = route.get("geometry") or safe_get(route, ["overview_polyline", "points"], default=None)
                path_pts = decode_polyline(geom) if geom else []
                if len(path_pts) < 2:
                    path_pts = coords[:n_path]
            else:
                path_pts = coords[:n_path]

            if len(path_pts) < 2:
                st.warning("Not enough points to snap.")
            else:
                status, data = snap_to_roads(api_key, path_pts, interpolate, include_geometry)
                st.session_state["snap_result"] = {"status": status, "data": data}
                st.success(f"Snap-to-road response: HTTP {status}")

    with c2:
        st.markdown("### Isochrone")
        center_src = st.selectbox("Isochrone center", ["Use current center", "Use last clicked pin (Places map)"], index=0)
        if center_src.startswith("Use last") and st.session_state.get("pin_center"):
            cen = (st.session_state["pin_center"]["lat"], st.session_state["pin_center"]["lng"])
        else:
            c = st.session_state["center"]
            cen = (c["lat"], c["lng"])

        contours = st.slider("Contours minutes", min_value=5, max_value=120, value=30, step=5)
        iso_mode = st.selectbox("Mode (Isochrone)", MODE_PRESETS, index=0)

        if st.button("Run Isochrone", **wstretch()):
            status, data = isochrone(api_key, cen, iso_mode, contours, dep_unix)
            st.session_state["iso_result"] = {"status": status, "data": data, "center": cen}
            st.success(f"Isochrone response: HTTP {status}")

# Debug
with st.expander("Debug", expanded=False):
    st.json({
        "stops_count": len(st.session_state["stops"]),
        "center": st.session_state["center"],
        "country_filter": st.session_state["country_filter"],
        "has_directions_before": st.session_state["directions_before"] is not None,
        "has_opt_result": st.session_state["opt_result"] is not None,
    })
    

# NextBillion.ai — Visual API Tester (Streamlit)
# pip install streamlit requests pandas folium streamlit-folium

import math
import random
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium


# =========================
# CONFIG
# =========================
DEFAULT_API_KEY = ""  # Leave blank in code; set via Streamlit Secrets (NEXTBILLION_API_KEY) or env var.
DEFAULT_API_KEY = st.secrets.get("NEXTBILLION_API_KEY", os.getenv("NEXTBILLION_API_KEY", DEFAULT_API_KEY))
NB_BASE = "https://api.nextbillion.io"
UA = {"User-Agent": "NextBillion-Visual-Tester/1.0"}

st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")


# =========================
# HELPERS
# =========================
def now_unix() -> int:
    return int(time.time())


def dt_to_unix(dt: datetime) -> int:
    # treat naive as UTC
    if dt.tzinfo is None:
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    return int(dt.timestamp())


def fmt_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return "-"
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def fmt_meters(m: Optional[float]) -> str:
    if m is None:
        return "-"
    m = float(m)
    if m >= 1000:
        return f"{m/1000:.2f} km"
    return f"{m:.0f} m"


def safe_get(d: Dict, path: List, default=None):
    cur = d
    for p in path:
        try:
            cur = cur[p]
        except Exception:
            return default
    return cur


def nb_get(path: str, params: Dict, timeout: int = 60) -> Tuple[int, Dict]:
    url = f"{NB_BASE}{path}"
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def nb_post(path: str, params: Dict, body: Dict, timeout: int = 60) -> Tuple[int, Dict]:
    url = f"{NB_BASE}{path}"
    r = requests.post(url, params=params, json=body, headers=UA, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"


def parse_latlng(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    s = s.strip()
    if "," not in s:
        return None
    a, b = s.split(",", 1)
    try:
        return float(a.strip()), float(b.strip())
    except Exception:
        return None


# Google polyline decode (precision 5)
def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    if not isinstance(polyline_str, str) or not polyline_str:
        return []

    index, lat, lng = 0, 0, 0
    coordinates = []
    factor = 10 ** precision

    while index < len(polyline_str):
        shift, result = 0, 0
        while True:
            if index >= len(polyline_str):
                return coordinates
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            if index >= len(polyline_str):
                return coordinates
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / factor, lng / factor))
    return coordinates


def extract_route_coords(directions_json: Dict) -> List[Tuple[float, float]]:
    """
    NextBillion Directions can resemble Google-like responses, but fields vary.
    We try multiple known shapes to reliably extract a polyline:
    - routes[0].geometry (encoded polyline)
    - routes[0].overview_polyline.points
    - routes[0].polyline / routes[0].points
    - legs/steps geometries (fallback)
    """
    routes = directions_json.get("routes") or []
    if not routes:
        return []

    r0 = routes[0]

    # common direct fields
    geom = r0.get("geometry")
    if isinstance(geom, str) and geom:
        return decode_polyline(geom, precision=5)

    ov = r0.get("overview_polyline")
    if isinstance(ov, dict):
        pts = ov.get("points")
        if isinstance(pts, str) and pts:
            return decode_polyline(pts, precision=5)

    poly = r0.get("polyline") or r0.get("points")
    if isinstance(poly, str) and poly:
        return decode_polyline(poly, precision=5)

    # fallback: stitch step geometries if present
    coords: List[Tuple[float, float]] = []
    legs = r0.get("legs") or []
    for leg in legs:
        steps = leg.get("steps") or []
        for stp in steps:
            g = stp.get("geometry") or stp.get("polyline")
            if isinstance(g, str) and g:
                part = decode_polyline(g, precision=5)
                if part:
                    if coords and part and coords[-1] == part[0]:
                        coords.extend(part[1:])
                    else:
                        coords.extend(part)
    return coords


def infer_center(stops: List[Dict], fallback: Tuple[float, float]) -> Tuple[float, float]:
    pts = [(s["lat"], s["lng"]) for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if not pts:
        return fallback
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def numbered_icon(num: int) -> folium.DivIcon:
    html = f"""
    <div style="
        width: 28px; height: 28px;
        background: #e53935;
        border-radius: 50%;
        border: 2px solid white;
        color: white;
        text-align: center;
        line-height: 24px;
        font-weight: 700;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    ">{num}</div>
    """
    return folium.DivIcon(html=html)


def build_map(
    center: Tuple[float, float],
    stops: List[Dict],
    route_coords: Optional[List[Tuple[float, float]]] = None,
    zoom: int = 11,
    show_arrows: bool = True,
) -> folium.Map:
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, control_scale=True)

    # numbered red pins (order matters)
    for i, s in enumerate(stops):
        if s.get("lat") is None or s.get("lng") is None:
            continue
        label = s.get("label") or f"Stop {i+1}"
        addr = s.get("address") or ""
        popup = f"<b>{label}</b><br/>{addr}<br/>{s['lat']:.6f}, {s['lng']:.6f}"
        folium.Marker(
            [s["lat"], s["lng"]],
            popup=popup,
            tooltip=f"{i+1}. {label}",
            icon=numbered_icon(i + 1),
        ).add_to(m)

    # route polyline + arrows
    if route_coords and len(route_coords) >= 2:
        pl = folium.PolyLine(route_coords, weight=6, opacity=0.9)
        pl.add_to(m)

        if show_arrows:
            # repeat arrowheads along the line
            PolyLineTextPath(
                pl,
                "▶",
                repeat=True,
                offset=7,
                attributes={"font-size": "16", "fill": "#111", "font-weight": "700"},
            ).add_to(m)

    return m


def ensure_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = DEFAULT_API_KEY

    if "center" not in st.session_state:
        st.session_state.center = (28.6139, 77.2090)  # New Delhi

    if "country_filter" not in st.session_state:
        st.session_state.country_filter = "IND"

    if "stops" not in st.session_state:
        st.session_state.stops = []

    # persistence buckets
    for k in [
        "region_results",
        "places_results",
        "last_geocode_json",
        "last_places_json",
        "last_directions_json_before",
        "last_directions_json_after",
        "last_matrix_json",
        "last_snap_json",
        "last_iso_json",
        "last_opt_create_json",
        "last_opt_result_json",
        "before_metrics",
        "after_metrics",
        "before_route_coords",
        "after_route_coords",
        "vrp_job_id",
        "map_click",
        "map_version",
    ]:
        if k not in st.session_state:
            st.session_state[k] = None

    if st.session_state.map_version is None:
        st.session_state.map_version = 1


ensure_state()


# =========================
# CACHING
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_forward_geocode(api_key: str, query: str, country_code: Optional[str], lang: str):
    params = {"key": api_key, "q": query, "language": lang}
    if country_code:
        params["countryCode"] = country_code
    return nb_get("/geocode", params=params, timeout=60)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_reverse_geocode(api_key: str, lat: float, lng: float, lang: str):
    params = {"key": api_key, "at": latlng_str(lat, lng), "language": lang}
    return nb_get("/reversegeocode", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_places_discover(api_key: str, at: str, q: str, radius: int, limit: int, country_filter: Optional[str], lang: str):
    # Docs confirm /discover endpoint.  :contentReference[oaicite:2]{index=2}
    params = {"key": api_key, "at": at, "q": q, "limit": limit, "language": lang}
    if radius:
        params["radius"] = radius
    if country_filter:
        params["in"] = f"countryCode:{country_filter}"
    return nb_get("/discover", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_directions(api_key: str, origin: str, destination: str, waypoints: str, mode: str, option: str,
                      avoid: str, alternatives: bool, departure_time: Optional[int], route_type: str, lang: str):
    params = {"key": api_key, "origin": origin, "destination": destination, "mode": mode, "language": lang}

    if option == "flexible":
        params["option"] = "flexible"
        if route_type:
            params["route_type"] = route_type

    if waypoints:
        params["waypoints"] = waypoints
    if avoid:
        params["avoid"] = avoid
    if alternatives:
        params["alternatives"] = "true"
    if departure_time:
        params["departure_time"] = int(departure_time)

    return nb_get("/directions/json", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_distance_matrix(api_key: str, origins: str, destinations: str, mode: str, option: str,
                           avoid: str, departure_time: Optional[int], route_type: str, lang: str):
    params = {"key": api_key, "origins": origins, "destinations": destinations, "mode": mode, "language": lang}
    if option == "flexible":
        params["option"] = "flexible"
        if route_type:
            params["route_type"] = route_type
    if avoid:
        params["avoid"] = avoid
    if departure_time:
        params["departure_time"] = int(departure_time)
    return nb_get("/distancematrix/json", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_snap_to_roads(api_key: str, path_str: str, radiuses: Optional[str], timestamps: Optional[str], mode: str, geometry: str):
    params = {"key": api_key, "path": path_str, "mode": mode}
    if radiuses:
        params["radiuses"] = radiuses
    if timestamps:
        params["timestamps"] = timestamps
    if geometry:
        params["geometry"] = geometry
    return nb_get("/snapToRoads/json", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_isochrone(api_key: str, coordinates: str, mode: str, contours_minutes: str, departure_time: Optional[int]):
    params = {"key": api_key, "coordinates": coordinates, "mode": mode, "contours_minutes": contours_minutes}
    if departure_time:
        params["departure_time"] = int(departure_time)
    return nb_get("/isochrone/json", params=params, timeout=60)


@st.cache_data(ttl=300, show_spinner=False)
def cached_vrp_create(api_key: str, body: Dict):
    return nb_post("/optimization/v2", params={"key": api_key}, body=body, timeout=90)


def vrp_result_no_cache(api_key: str, job_id: str) -> Tuple[int, Dict]:
    # No caching: needed for polling until ready
    return nb_get("/optimization/v2/result", params={"key": api_key, "id": job_id}, timeout=90)


# =========================
# UI — SIDEBAR GLOBAL OPTIONS
# =========================
st.sidebar.title("Config")

st.session_state.api_key = st.sidebar.text_input(
    "NextBillion API Key",
    value=st.session_state.api_key,
    type="password",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Stops (20+ supported)")
st.sidebar.caption("Stops are shared across ALL tabs.")

lang = st.sidebar.selectbox("Language", ["en-US", "en", "de", "fr", "hi"], index=0)
mode = st.sidebar.selectbox("Mode", ["car", "truck", "motorbike", "bicycle", "pedestrian"], index=0)
option = st.sidebar.selectbox("Directions/Matrix option", ["fast", "flexible"], index=0)

route_type = ""
if option == "flexible":
    route_type = st.sidebar.selectbox("Route type (flexible)", ["", "fastest", "shortest"], index=1)

# Avoid dropdown
avoid_choices = {
    "None": "",
    "Avoid tolls": "toll",
    "Avoid highways": "highway",
    "Avoid ferries": "ferry",
    "Avoid highways + tolls": "highway|toll",
    "Avoid ferries + tolls": "ferry|toll",
    "Avoid ferries + highways": "ferry|highway",
    "Avoid ferries + highways + tolls": "ferry|highway|toll",
}
avoid_label = st.sidebar.selectbox("Avoid", list(avoid_choices.keys()), index=0)
avoid = avoid_choices[avoid_label]

alternatives = st.sidebar.checkbox("Alternative routes", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Departure time (optional)")
dep_dt = st.sidebar.date_input("Date", value=datetime.now().date())
dep_tm = st.sidebar.time_input("Time", value=datetime.now().time().replace(microsecond=0))
use_departure = st.sidebar.checkbox("Use departure_time", value=False)

departure_unix = None
if use_departure:
    dep = datetime(dep_dt.year, dep_dt.month, dep_dt.day, dep_tm.hour, dep_tm.minute, dep_tm.second)
    departure_unix = dt_to_unix(dep)
    st.sidebar.caption(f"departure_time: {departure_unix}  ({dep} UTC)")

st.sidebar.markdown("---")
st.sidebar.subheader("Stops loaded")
st.sidebar.write(len(st.session_state.stops))


# =========================
# HEADER
# =========================
st.title("NextBillion.ai — Visual API Tester")
st.caption("Workflow: search region → set center → generate stops → compute route (before) → optimize (after) → compare savings")

tabs = st.tabs([
    "Stop Manager (Search + Generate 20+)",
    "Geocode & Map",
    "Places (POI Search → Add Stops)",
    "Route + Optimize (Before vs After)",
    "Distance Matrix (NxN)",
    "Snap-to-Roads + Isochrone",
])


# =========================
# TAB 0 — STOP MANAGER
# =========================
with tabs[0]:
    st.subheader("1) Pick a region/city → set center → generate stops (no keyword required)")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### A) Search region/city/state/country (universal)")
        st.caption("Uses Forward Geocode for region search. Only runs when you click Search.")

        with st.form("region_search_form", clear_on_submit=False):
            region_q = st.text_input("Region/City/State/Country", value="Delhi")
            country_hint = st.text_input("Country hint (optional, 3-letter code like IND/DEU/USA)", value="")
            submitted_region = st.form_submit_button("Search Region")

        if submitted_region and region_q.strip():
            status, data = cached_forward_geocode(
                st.session_state.api_key,
                region_q.strip(),
                country_hint.strip() if country_hint.strip() else None,
                lang,
            )
            st.session_state.last_geocode_json = data
            st.write(f"Geocode response: HTTP {status}")

            items = data.get("items") or data.get("results") or []
            region_results = []
            for it in items[:15]:
                pos = it.get("position") or {}
                title = it.get("title") or it.get("address", {}).get("label") or str(it.get("id", "result"))
                cc = (it.get("address", {}) or {}).get("countryCode") or it.get("countryCode")
                region_results.append({
                    "title": title,
                    "lat": pos.get("lat"),
                    "lng": pos.get("lng"),
                    "countryCode": cc,
                    "raw": it,
                })
            st.session_state.region_results = region_results

        # Persist selection even after reruns
        if st.session_state.region_results:
            pick = st.selectbox(
                "Pick a region result",
                options=list(range(len(st.session_state.region_results))),
                format_func=lambda i: f"{st.session_state.region_results[i]['title']} ({st.session_state.region_results[i]['countryCode'] or '-'})",
                key="region_pick_selectbox",
            )
            if st.button("Use picked region as center"):
                chosen = st.session_state.region_results[pick]
                if chosen.get("lat") is not None and chosen.get("lng") is not None:
                    st.session_state.center = (chosen["lat"], chosen["lng"])
                    if chosen.get("countryCode"):
                        st.session_state.country_filter = chosen["countryCode"]
                    st.session_state.map_version += 1
                    st.success(f"Center set: {st.session_state.center} | Country filter: {st.session_state.country_filter}")
                else:
                    st.error("Selected result has no lat/lng. Try another result.")

        st.markdown("### B) Paste stops (easy input)")
        input_mode = st.radio("Input type", ["Addresses (one per line)", "Lat,Lng (one per line)"], horizontal=True)

        pasted = st.text_area("Paste at least 20 lines", height=180, placeholder="Example:\nConnaught Place, New Delhi\nIGI Airport, Delhi\n...")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Add / Replace Stops"):
                lines = [x.strip() for x in pasted.splitlines() if x.strip()]
                new_stops = []
                for i, line in enumerate(lines):
                    if input_mode.startswith("Lat"):
                        ll = parse_latlng(line)
                        if not ll:
                            continue
                        new_stops.append({"label": f"Stop {i+1}", "address": "", "lat": ll[0], "lng": ll[1], "source": "pasted_latlng"})
                    else:
                        new_stops.append({"label": f"Stop {i+1}", "address": line, "lat": None, "lng": None, "source": "pasted_address"})
                st.session_state.stops = new_stops
                st.success(f"Loaded {len(st.session_state.stops)} stops")
        with colB:
            if st.button("Clear Stops"):
                st.session_state.stops = []
                st.success("Stops cleared")

    with c2:
        st.markdown("### C) Set center by clicking on the map (pin shows immediately)")
        st.caption("Click the map → a red pin appears. Then press 'Use clicked point as center'.")

        center = st.session_state.center
        pick_map = folium.Map(location=[center[0], center[1]], zoom_start=11, control_scale=True)
        folium.Marker([center[0], center[1]], tooltip="Current center", icon=folium.Icon(color="blue")).add_to(pick_map)

        # If user already clicked, show that pin too
        if st.session_state.map_click:
            folium.Marker(
                [st.session_state.map_click[0], st.session_state.map_click[1]],
                tooltip="Clicked point",
                icon=folium.Icon(color="red", icon="map-marker"),
            ).add_to(pick_map)

        map_click = st_folium(
            pick_map,
            height=420,
            use_container_width=True,
            key=f"pick_center_map_v{st.session_state.map_version}",
        )
        clicked = map_click.get("last_clicked")
        if clicked:
            st.session_state.map_click = (clicked["lat"], clicked["lng"])
            st.info(f"Clicked: {clicked['lat']:.6f}, {clicked['lng']:.6f}")

        if st.button("Use clicked point as center"):
            if st.session_state.map_click:
                st.session_state.center = st.session_state.map_click
                st.session_state.map_version += 1
                st.success(f"Center updated: {st.session_state.center}")

        st.markdown("### D) Generate random stops around center (NO keyword required)")
        st.caption("Generates random lat/lng. Optional reverse-geocode first N into human-readable addresses.")

        with st.form("random_gen_form", clear_on_submit=False):
            n = st.number_input("How many stops?", min_value=5, max_value=200, value=25, step=1)
            radius_m = st.number_input("Radius (meters)", min_value=200, max_value=50000, value=8000, step=100)
            resolve_n = st.number_input("Reverse-geocode first N (optional)", min_value=0, max_value=200, value=0, step=1)
            gen = st.form_submit_button("Generate Stops Around Center")

        if gen:
            cx, cy = st.session_state.center
            new_stops = []
            for i in range(int(n)):
                r = radius_m * math.sqrt(random.random())
                theta = 2 * math.pi * random.random()
                dx = r * math.cos(theta)
                dy = r * math.sin(theta)

                dlat = dy / 111320.0
                dlng = dx / (111320.0 * math.cos(math.radians(cx)) + 1e-9)

                lat = cx + dlat
                lng = cy + dlng
                new_stops.append({"label": f"Stop {i+1}", "address": "", "lat": lat, "lng": lng, "source": "random"})

            if resolve_n and resolve_n > 0:
                for i in range(min(int(resolve_n), len(new_stops))):
                    status, data = cached_reverse_geocode(st.session_state.api_key, new_stops[i]["lat"], new_stops[i]["lng"], lang)
                    items = data.get("items") or data.get("results") or []
                    if items:
                        it0 = items[0]
                        label = it0.get("title") or (it0.get("address", {}) or {}).get("label") or ""
                        new_stops[i]["address"] = label

            st.session_state.stops = new_stops
            st.session_state.map_version += 1
            st.success(f"Generated {len(st.session_state.stops)} stops around {st.session_state.center}")

        st.markdown("### Stops table (editable)")
        df = pd.DataFrame(st.session_state.stops)
        edited = st.data_editor(df, num_rows="dynamic", width="stretch", key="stops_editor"
        st.session_state.stops = edited.to_dict(orient="records")

        st.markdown("### Stops map (red numbered pins)")
        center2 = infer_center(st.session_state.stops, st.session_state.center)
        m = build_map(center2, st.session_state.stops, zoom=11, show_arrows=False)
        st_folium(m, height=520, use_container_width=True, key=f"stops_overview_map_v{st.session_state.map_version}")


# =========================
# TAB 1 — GEOCODE & MAP
# =========================
with tabs[1]:
    st.subheader("2) Geocode addresses → show pins on map (cached)")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if st.button("Geocode all stops missing lat/lng (cached)"):
            updated = []
            for s in st.session_state.stops:
                if s.get("lat") is not None and s.get("lng") is not None:
                    updated.append(s)
                    continue
                addr = (s.get("address") or "").strip()
                if not addr:
                    updated.append(s)
                    continue
                status, data = cached_forward_geocode(st.session_state.api_key, addr, st.session_state.country_filter, lang)
                st.session_state.last_geocode_json = data
                items = data.get("items") or data.get("results") or []
                if items:
                    pos = items[0].get("position") or {}
                    s["lat"] = pos.get("lat")
                    s["lng"] = pos.get("lng")
                    if not s.get("address"):
                        s["address"] = items[0].get("title") or (items[0].get("address", {}) or {}).get("label") or ""
                    s["source"] = (s.get("source") or "") + "+geocoded"
                updated.append(s)

            st.session_state.stops = updated
            st.session_state.map_version += 1
            st.success("Geocoding done.")

        if st.session_state.last_geocode_json:
            st.download_button(
                "Download last Geocode JSON",
                data=json.dumps(st.session_state.last_geocode_json, indent=2, default=str),
                file_name="geocode_last.json",
                mime="application/json",
            )

    with col2:
        st.write(f"Country filter: **{st.session_state.country_filter}**")
        st.write(f"Center: **{st.session_state.center[0]:.6f}, {st.session_state.center[1]:.6f}**")

    st.markdown("### Map (pins persist)")
    center = infer_center(st.session_state.stops, st.session_state.center)
    m = build_map(center, st.session_state.stops, zoom=11, show_arrows=False)
    st_folium(m, height=560, use_container_width=True, key=f"geocode_map_v{st.session_state.map_version}")


# =========================
# TAB 2 — PLACES (POI SEARCH)
# =========================
with tabs[2]:
    st.subheader("3) Places — Search POIs and add as stops")
    st.caption("Uses /discover. Runs only when you click Search Places. Results persist until next search.")

    st.write(f"Search center: **{st.session_state.center[0]:.6f}, {st.session_state.center[1]:.6f}** | Country filter: **{st.session_state.country_filter}**")

    with st.form("places_form", clear_on_submit=False):
        q = st.text_input("POI keyword (required for Places)", value="petrol pump")
        radius = st.slider("Radius (meters)", min_value=500, max_value=50000, value=15000, step=500)
        limit = st.slider("Limit", min_value=5, max_value=100, value=30, step=5)
        use_country = st.checkbox("Apply country filter", value=True)
        do_search = st.form_submit_button("Search Places")

    if do_search:
        if not q.strip():
            st.warning("Places needs a keyword. For random stops without keyword, use Stop Manager → Generate random stops.")
        else:
            status, data = cached_places_discover(
                st.session_state.api_key,
                at=latlng_str(*st.session_state.center),
                q=q.strip(),
                radius=int(radius),
                limit=int(limit),
                country_filter=st.session_state.country_filter if use_country else None,
                lang=lang,
            )
            st.session_state.last_places_json = data
            st.info(f"Places response: HTTP {status}")

            items = data.get("items") or data.get("results") or data.get("data") or []
            results = []
            for it in items:
                pos = it.get("position") or it.get("location") or {}
                title = it.get("title") or (it.get("address", {}) or {}).get("label") or it.get("name") or "POI"
                addr_label = (it.get("address", {}) or {}).get("label") or title
                results.append({
                    "title": title,
                    "lat": pos.get("lat"),
                    "lng": pos.get("lng"),
                    "address": addr_label,
                    "raw": it,
                })

            st.session_state.places_results = results

    results = st.session_state.places_results or []

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        if st.session_state.last_places_json:
            st.download_button(
                "Download last Places JSON",
                data=json.dumps(st.session_state.last_places_json, indent=2, default=str),
                file_name="places_last.json",
                mime="application/json",
            )

        if results:
            st.markdown("### Select POIs to add as stops")
            add_idx = st.multiselect(
                "Pick results",
                options=list(range(len(results))),
                format_func=lambda i: results[i]["title"],
                key="places_pick_multi",
            )
            if st.button("Add selected POIs to stops"):
                for i in add_idx:
                    r = results[i]
                    if r.get("lat") is None or r.get("lng") is None:
                        continue
                    st.session_state.stops.append({
                        "label": r["title"],
                        "address": r["address"],
                        "lat": r["lat"],
                        "lng": r["lng"],
                        "source": "places_discover",
                    })
                st.session_state.map_version += 1
                st.success(f"Added {len(add_idx)} stops. Total stops: {len(st.session_state.stops)}")

            dfp = pd.DataFrame([{"title": r["title"], "lat": r["lat"], "lng": r["lng"]} for r in results])
            st.dataframe(dfp, width="stretch")
        else:
            st.warning("No items found. If your raw JSON has items but UI shows none, it’s usually schema variance—download JSON and share a sample item.")

    with colR:
        st.markdown("### Places map (pins persist)")
        preview_stops = [
            {"label": r["title"], "address": r["address"], "lat": r["lat"], "lng": r["lng"]}
            for r in results if r.get("lat") is not None and r.get("lng") is not None
        ]
        m = build_map(st.session_state.center, preview_stops, zoom=12, show_arrows=False)
        st_folium(m, height=560, use_container_width=True, key=f"places_map_v{st.session_state.map_version}")


# =========================
# TAB 3 — ROUTE + OPTIMIZE (BEFORE VS AFTER)
# =========================
with tabs[3]:
    st.subheader("4) Route + Optimize (Before vs After)")
    st.caption("Step 1 draws the route line. Step 2 (ONE button) optimizes + polls result + recomputes route after.")

    stops = [s for s in st.session_state.stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(stops) < 2:
        st.warning("Need at least 2 stops with lat/lng. Use Stop Manager (generate) or Geocode tab.")
    else:
        n_use = st.slider("Stops used in route/opt", min_value=2, max_value=min(60, len(stops)), value=min(10, len(stops)))
        stops_use = stops[:n_use]

        # ---- STEP 1 (BEFORE)
        st.markdown("### Step 1 — Compute route (BEFORE)")
        if st.button("Compute Route (Before)"):
            origin = latlng_str(stops_use[0]["lat"], stops_use[0]["lng"])
            destination = latlng_str(stops_use[-1]["lat"], stops_use[-1]["lng"])
            wps = "|".join(latlng_str(s["lat"], s["lng"]) for s in stops_use[1:-1]) if len(stops_use) > 2 else ""

            status, data = cached_directions(
                st.session_state.api_key,
                origin=origin,
                destination=destination,
                waypoints=wps,
                mode=mode,
                option=option,
                avoid=avoid,
                alternatives=alternatives,
                departure_time=departure_unix,
                route_type=route_type,
                lang=lang,
            )
            st.session_state.last_directions_json_before = data
            st.info(f"Directions (Before): HTTP {status}")

            routes = data.get("routes") or []
            dist_m = safe_get(routes, [0, "distance"], None)
            dur_s = safe_get(routes, [0, "duration"], None)

            coords = extract_route_coords(data)
            st.session_state.before_route_coords = coords

            st.session_state.before_metrics = {"distance_m": dist_m, "duration_s": dur_s}
            st.session_state.map_version += 1

        if st.session_state.before_metrics:
            bm = st.session_state.before_metrics
            st.write(f"**Before:** {fmt_meters(bm['distance_m'])} | ETA: {fmt_seconds(bm['duration_s'])}")
            if not st.session_state.before_route_coords:
                st.warning("No route geometry found in response → route line cannot draw. Download JSON and share if this repeats.")

        # ---- STEP 2 (ONE BUTTON): optimize + poll + recompute after
        st.markdown("---")
        st.markdown("### Step 2 — Optimize + recompute route (AFTER) (ONE button)")
        obj = st.selectbox("Optimization objective (travel_cost)", ["duration", "distance"], index=0)

        if st.button("Optimize + Recompute After Route"):
            # Build VRP v2 request body per docs:
            # - locations must be an object with `location` list => fixes "locations.location missing"
            # - objective is object => options.objective.travel_cost
            #   :contentReference[oaicite:3]{index=3}
            locations_list = [latlng_str(s["lat"], s["lng"]) for s in stops_use]

            jobs = [{"id": f"job_{i}", "location_index": i} for i in range(1, len(locations_list))]

            body = {
                "locations": {"id": 1, "location": locations_list},
                "jobs": jobs,
                "vehicles": [{"id": "vehicle_1", "start_index": 0, "end_index": 0}],
                "options": {"objective": {"travel_cost": obj}},
            }

            status, data = cached_vrp_create(st.session_state.api_key, body)
            st.session_state.last_opt_create_json = data
            st.info(f"VRP create: HTTP {status}")

            job_id = safe_get(data, ["data", "id"], None) or data.get("id") or safe_get(data, ["result", "id"], None)
            st.session_state.vrp_job_id = job_id

            if not job_id:
                st.error("Optimization job id not found. Check JSON below.")
                st.json(data)
            else:
                st.success(f"Optimization job created: {job_id}. Polling for result...")

                # Poll until result is ready
                result_data = None
                max_wait_s = 60
                step_s = 2
                start = time.time()

                with st.status("Waiting for optimization result...", expanded=False) as status_box:
                    while time.time() - start < max_wait_s:
                        rs, rd = vrp_result_no_cache(st.session_state.api_key, job_id)
                        st.session_state.last_opt_result_json = rd

                        # "ready" heuristics: routes exist
                        routes = safe_get(rd, ["result", "routes"], None) or rd.get("routes")
                        if routes:
                            result_data = rd
                            status_box.update(label="Optimization result received.", state="complete")
                            break

                        time.sleep(step_s)

                    if result_data is None:
                        status_box.update(label="Optimization result not ready yet.", state="error")

                if result_data is None:
                    st.error("Optimization result not ready or empty. Click the button again after a few seconds.")
                else:
                    # Extract order from VRP result
                    order = [0]
                    routes = safe_get(result_data, ["result", "routes"], []) or result_data.get("routes") or []
                    if routes:
                        steps = routes[0].get("steps") or []
                        for step in steps:
                            li = step.get("location_index") or step.get("locationIndex")
                            if li is not None and int(li) != 0:
                                order.append(int(li))
                    if order[-1] != 0:
                        order.append(0)

                    # fallback
                    if len(order) <= 2:
                        order = list(range(len(locations_list)))

                    # build after sequence (skip duplicate depot)
                    seq = []
                    for idx in order:
                        if seq and idx == seq[-1]:
                            continue
                        seq.append(idx)

                    seq_coords = [locations_list[i] for i in seq if i < len(locations_list)]
                    if len(seq_coords) >= 2:
                        origin2 = seq_coords[0]
                        dest2 = seq_coords[-1]
                        wps2 = "|".join(seq_coords[1:-1]) if len(seq_coords) > 2 else ""

                        s2, d2 = cached_directions(
                            st.session_state.api_key,
                            origin=origin2,
                            destination=dest2,
                            waypoints=wps2,
                            mode=mode,
                            option=option,
                            avoid=avoid,
                            alternatives=alternatives,
                            departure_time=departure_unix,
                            route_type=route_type,
                            lang=lang,
                        )
                        st.session_state.last_directions_json_after = d2
                        st.info(f"Directions (After): HTTP {s2}")

                        routes2 = d2.get("routes") or []
                        dist2 = safe_get(routes2, [0, "distance"], None)
                        dur2 = safe_get(routes2, [0, "duration"], None)
                        coords2 = extract_route_coords(d2)

                        st.session_state.after_metrics = {"distance_m": dist2, "duration_s": dur2}
                        st.session_state.after_route_coords = coords2

                        # Save the optimized stop order for mapping with correct numbering
                        # Create a re-ordered stop list (without final depot repeat)
                        order_wo_last_depot = [i for i in seq if i != 0]
                        # If depot included only once, this works. Otherwise fallback.
                        if order_wo_last_depot:
                            reordered = [stops_use[0]] + [stops_use[i] for i in order_wo_last_depot if i < len(stops_use)]
                            # De-dup in case of oddities
                            seen = set()
                            cleaned = []
                            for s in reordered:
                                key = (s.get("lat"), s.get("lng"))
                                if key not in seen:
                                    seen.add(key)
                                    cleaned.append(s)
                            st.session_state.optimized_stops_use = cleaned
                        else:
                            st.session_state.optimized_stops_use = stops_use

                        st.session_state.map_version += 1

        # ---- COMPARISON
        if st.session_state.before_metrics and st.session_state.after_metrics:
            bm = st.session_state.before_metrics
            am = st.session_state.after_metrics

            dist_saved = None
            dur_saved = None
            pct_dist = None
            pct_dur = None

            if bm["distance_m"] is not None and am["distance_m"] is not None and bm["distance_m"]:
                dist_saved = bm["distance_m"] - am["distance_m"]
                pct_dist = dist_saved / bm["distance_m"] * 100

            if bm["duration_s"] is not None and am["duration_s"] is not None and bm["duration_s"]:
                dur_saved = bm["duration_s"] - am["duration_s"]
                pct_dur = dur_saved / bm["duration_s"] * 100

            st.markdown("## Results")
            st.write(f"**Results (Before):** Distance: {fmt_meters(bm['distance_m'])} | ETA: {fmt_seconds(bm['duration_s'])}")
            st.write(f"**Results (After Optimization):** Distance: {fmt_meters(am['distance_m'])} | ETA: {fmt_seconds(am['duration_s'])}")

            if dist_saved is not None and pct_dist is not None:
                st.success(f"Distance saved: {fmt_meters(dist_saved)} ({pct_dist:.1f}%)")
            if dur_saved is not None and pct_dur is not None:
                st.success(f"Time saved: {fmt_seconds(dur_saved)} ({pct_dur:.1f}%)")

        # ---- TWO MAPS SIDE BY SIDE (BEFORE + AFTER) + ARROWS
        st.markdown("### Maps (Before vs After) — numbered stops + direction arrows")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Map — Before")
            center_b = infer_center(stops_use, st.session_state.center)
            m1 = build_map(
                center=center_b,
                stops=stops_use,
                route_coords=st.session_state.before_route_coords,
                zoom=11,
                show_arrows=True,
            )
            st_folium(m1, height=520, use_container_width=True, key=f"map_before_v{st.session_state.map_version}")

        with right:
            st.markdown("#### Map — After Optimization")
            after_stops = st.session_state.get("optimized_stops_use") or stops_use
            center_a = infer_center(after_stops, st.session_state.center)
            m2 = build_map(
                center=center_a,
                stops=after_stops,
                route_coords=st.session_state.after_route_coords,
                zoom=11,
                show_arrows=True,
            )
            st_folium(m2, height=520, use_container_width=True, key=f"map_after_v{st.session_state.map_version}")

        # downloads
        colD1, colD2, colD3, colD4 = st.columns(4)
        with colD1:
            if st.session_state.last_directions_json_before:
                st.download_button(
                    "Download Directions (Before)",
                    data=json.dumps(st.session_state.last_directions_json_before, indent=2, default=str),
                    file_name="directions_before.json",
                    mime="application/json",
                )
        with colD2:
            if st.session_state.last_directions_json_after:
                st.download_button(
                    "Download Directions (After)",
                    data=json.dumps(st.session_state.last_directions_json_after, indent=2, default=str),
                    file_name="directions_after.json",
                    mime="application/json",
                )
        with colD3:
            if st.session_state.last_opt_create_json:
                st.download_button(
                    "Download VRP Create JSON",
                    data=json.dumps(st.session_state.last_opt_create_json, indent=2, default=str),
                    file_name="opt_create_last.json",
                    mime="application/json",
                )
        with colD4:
            if st.session_state.last_opt_result_json:
                st.download_button(
                    "Download VRP Result JSON",
                    data=json.dumps(st.session_state.last_opt_result_json, indent=2, default=str),
                    file_name="opt_result_last.json",
                    mime="application/json",
                )


# =========================
# TAB 4 — DISTANCE MATRIX (NxN)
# =========================
with tabs[4]:
    st.subheader("5) Distance Matrix (NxN for 20+ points)")
    st.caption("Many-to-many: origins and destinations are 'lat,lng|lat,lng|...'")

    stops = [s for s in st.session_state.stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(stops) < 2:
        st.warning("Need at least 2 geocoded stops.")
    else:
        n_use = st.slider("Stops used in NxN", min_value=2, max_value=min(50, len(stops)), value=min(20, len(stops)))
        use_stops = stops[:n_use]

        origins = "|".join(latlng_str(s["lat"], s["lng"]) for s in use_stops)
        destinations = origins

        if st.button("Compute Distance Matrix (NxN)"):
            status, data = cached_distance_matrix(
                st.session_state.api_key,
                origins=origins,
                destinations=destinations,
                mode=mode,
                option=option,
                avoid=avoid,
                departure_time=departure_unix,
                route_type=route_type,
                lang=lang,
            )
            st.session_state.last_matrix_json = data
            st.info(f"Distance Matrix: HTTP {status}")

        if st.session_state.last_matrix_json:
            dm = st.session_state.last_matrix_json
            st.download_button(
                "Download Distance Matrix JSON",
                data=json.dumps(dm, indent=2, default=str),
                file_name="distance_matrix_last.json",
                mime="application/json",
            )

            rows = dm.get("rows") or []
            matrix_dist = []
            matrix_dur = []
            for r in rows:
                elems = r.get("elements") if isinstance(r, dict) else None
                dist_row = []
                dur_row = []
                for e in (elems or []):
                    dist_row.append(safe_get(e, ["distance", "value"], None))
                    dur_row.append(safe_get(e, ["duration", "value"], None))
                if dist_row:
                    matrix_dist.append(dist_row)
                    matrix_dur.append(dur_row)

            if matrix_dist:
                st.markdown("### Distance (km)")
                df_dist = pd.DataFrame(matrix_dist) / 1000.0
                st.dataframe(df_dist.style.format("{:.2f}"), width="stretch")

            if matrix_dur:
                st.markdown("### Duration (minutes)")
                df_dur = pd.DataFrame(matrix_dur) / 60.0
                st.dataframe(df_dur.style.format("{:.1f}"), width="stretch")


# =========================
# TAB 5 — SNAP TO ROADS + ISOCHRONE
# =========================
with tabs[5]:
    st.subheader("6) Snap-to-Roads + Isochrone (edge-case testing)")
    st.caption("Snap-to-Roads: snaps your path to road network. Isochrone: reachable area within X minutes.")

    st.markdown("### A) Snap-to-Roads")
    stops = [s for s in st.session_state.stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(stops) < 2:
        st.warning("Need at least 2 geocoded stops.")
    else:
        n_use = st.slider("Points for snap path", min_value=2, max_value=min(100, len(stops)), value=min(20, len(stops)))
        pts = stops[:n_use]
        path_str = "|".join(latlng_str(p["lat"], p["lng"]) for p in pts)

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            geometry = st.selectbox("Snap geometry", ["", "geojson"], index=1)
            use_radiuses = st.checkbox("Add radiuses", value=False)
            use_timestamps = st.checkbox("Add timestamps", value=False)

        radiuses = "|".join(["30"] * len(pts)) if use_radiuses else None
        timestamps = None
        if use_timestamps:
            t0 = now_unix() - 600
            timestamps = "|".join(str(t0 + i * 60) for i in range(len(pts)))

        if st.button("Snap To Roads"):
            status, data = cached_snap_to_roads(
                st.session_state.api_key,
                path_str=path_str,
                radiuses=radiuses,
                timestamps=timestamps,
                mode=mode if mode in ["car", "truck"] else "car",
                geometry=geometry,
            )
            st.info(f"SnapToRoads: HTTP {status}")
            st.session_state.last_snap_json = data
            st.session_state.map_version += 1

        if st.session_state.last_snap_json:
            data = st.session_state.last_snap_json
            st.download_button(
                "Download SnapToRoads JSON",
                data=json.dumps(data, indent=2, default=str),
                file_name="snap_to_roads_last.json",
                mime="application/json",
            )

            # draw snapped geometry if present
            geom = data.get("geometry")
            coords = []
            if geometry == "geojson" and isinstance(geom, dict):
                try:
                    if geom.get("type") == "LineString":
                        coords = [(c[1], c[0]) for c in geom.get("coordinates", [])]
                except Exception:
                    coords = []
            elif isinstance(geom, str):
                coords = decode_polyline(geom, precision=5)

            center = infer_center(pts, st.session_state.center)
            m = build_map(center, pts, route_coords=coords, zoom=12, show_arrows=True)
            st_folium(m, height=520, use_container_width=True, key=f"snap_map_v{st.session_state.map_version}")

    st.markdown("---")
    st.markdown("### B) Isochrone")
    cA, cB = st.columns([1, 1], gap="large")
    with cA:
        coords_in = st.text_input("Coordinates (lat,lng)", value=latlng_str(*st.session_state.center))
        contours = st.text_input("contours_minutes (comma-separated)", value="5,10,15")
    with cB:
        iso_mode = st.selectbox("Isochrone mode", ["car", "truck"], index=0)

    if st.button("Compute Isochrone"):
        status, data = cached_isochrone(
            st.session_state.api_key,
            coordinates=coords_in,
            mode=iso_mode,
            contours_minutes=contours,
            departure_time=departure_unix,
        )
        st.info(f"Isochrone: HTTP {status}")
        st.session_state.last_iso_json = data
        st.session_state.map_version += 1

    if st.session_state.last_iso_json:
        data = st.session_state.last_iso_json
        st.download_button(
            "Download Isochrone JSON",
            data=json.dumps(data, indent=2, default=str),
            file_name="isochrone_last.json",
            mime="application/json",
        )

        features = data.get("features") or []
        center_ll = parse_latlng(coords_in) or st.session_state.center
        m = folium.Map(location=[center_ll[0], center_ll[1]], zoom_start=12, control_scale=True)
        folium.Marker([center_ll[0], center_ll[1]], tooltip="Center", icon=folium.Icon(color="blue")).add_to(m)

        try:
            for f in features:
                gj = {"type": "Feature", "geometry": f.get("geometry"), "properties": f.get("properties", {})}
                folium.GeoJson(gj).add_to(m)
        except Exception:
            pass

        st_folium(m, height=520, use_container_width=True, key=f"iso_map_v{st.session_state.map_version}")

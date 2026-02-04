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
from folium import plugins
from streamlit_folium import st_folium

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")

# User explicitly asked to embed key:
API_KEY = "a08a2b15af0f432c8e438403bc2b00e3"
BASE_URL = "https://api.nextbillion.io"

YESNO = ["No", "Yes"]

# Navigation API avoid values (from docs/examples)
NAV_AVOID_OPTIONS = [
    "toll",
    "highway",
    "ferry",
    "sharp_turn",
]

# Travel modes commonly used in routing APIs
TRAVEL_MODES = ["car", "truck", "bike", "pedestrian"]

# Optimization objective travel_cost used in v2 example payloads
OPT_OBJECTIVES = ["distance", "duration"]

# =========================
# HELPERS
# =========================

def human_unix(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)

def km(meters: float) -> float:
    return meters / 1000.0

def mins(seconds: float) -> float:
    return seconds / 60.0

def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"

def parse_latlng_line(line: str) -> Optional[Tuple[float, float]]:
    line = line.strip()
    if not line:
        return None
    # accept "lat,lng" or "lat lng"
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = [p.strip() for p in line.split()]
    if len(parts) != 2:
        return None
    lat = safe_float(parts[0])
    lng = safe_float(parts[1])
    if lat is None or lng is None:
        return None
    return lat, lng

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))

def random_points_around(center: Tuple[float, float], radius_m: int, n: int) -> List[Tuple[float, float]]:
    """Generate random lat/lng points within radius meters around center."""
    lat0, lng0 = center
    out = []
    for _ in range(n):
        # Random distance and bearing
        r = radius_m * math.sqrt(random.random())
        theta = 2 * math.pi * random.random()
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)

        # Convert meters to degrees
        dlat = dy / 111_320.0
        dlng = dx / (111_320.0 * math.cos(math.radians(lat0)) + 1e-9)
        out.append((lat0 + dlat, lng0 + dlng))
    return out

def nb_get(path: str, params: Dict[str, Any], timeout: int = 30) -> Tuple[int, Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    params = dict(params or {})
    params["key"] = API_KEY
    r = requests.get(url, params=params, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def nb_post(path: str, params: Dict[str, Any], body: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    params = dict(params or {})
    params["key"] = API_KEY
    r = requests.post(url, params=params, json=body, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

@st.cache_data(show_spinner=False, ttl=3600)
def cached_get(path: str, params_json: str) -> Tuple[int, Dict[str, Any]]:
    params = json.loads(params_json)
    return nb_get(path, params=params)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_post(path: str, params_json: str, body_json: str) -> Tuple[int, Dict[str, Any]]:
    params = json.loads(params_json)
    body = json.loads(body_json)
    return nb_post(path, params=params, body=body)

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    """Standard Google polyline decoding. Works for string."""
    if not isinstance(polyline_str, str) or len(polyline_str) == 0:
        return []
    index = 0
    lat = 0
    lng = 0
    coordinates = []
    factor = 10 ** precision

    length = len(polyline_str)
    while index < length:
        result = 1
        shift = 0
        b = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result += (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result = 1
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result += (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / factor, lng / factor))
    return coordinates

def extract_route_coords(resp: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Try multiple possible response shapes:
    - routes[0].overview_polyline.points
    - routes[0].geometry (polyline string)
    - routes[0].geometry.coordinates (GeoJSON)
    - routes[0].legs[].steps[].polyline / geometry
    """
    # 1) Google-ish
    try:
        pts = resp["routes"][0]["overview_polyline"]["points"]
        coords = decode_polyline(pts, precision=5)
        if coords:
            return coords
    except Exception:
        pass

    # 2) geometry polyline string
    try:
        geom = resp["routes"][0].get("geometry")
        if isinstance(geom, str):
            coords = decode_polyline(geom, precision=5)
            if coords:
                return coords
    except Exception:
        pass

    # 3) GeoJSON coordinates
    try:
        coords = resp["routes"][0]["geometry"]["coordinates"]
        # assume [lng,lat]
        out = [(c[1], c[0]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        if out:
            return out
    except Exception:
        pass

    # 4) accumulate from legs/steps
    out = []
    try:
        legs = resp["routes"][0].get("legs", [])
        for leg in legs:
            for step in leg.get("steps", []):
                if "polyline" in step and isinstance(step["polyline"], str):
                    out.extend(decode_polyline(step["polyline"], precision=5))
                elif "geometry" in step and isinstance(step["geometry"], str):
                    out.extend(decode_polyline(step["geometry"], precision=5))
        if out:
            return out
    except Exception:
        pass

    return []

def compute_directions_totals(resp: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (distance_m, duration_s)"""
    try:
        r0 = resp["routes"][0]
        dist = r0.get("distance")
        dur = r0.get("duration")
        if dist is not None and dur is not None:
            return float(dist), float(dur)
    except Exception:
        pass

    # fallback sum legs
    try:
        dist = 0.0
        dur = 0.0
        legs = resp["routes"][0].get("legs", [])
        for leg in legs:
            dist += float(leg.get("distance", 0))
            dur += float(leg.get("duration", 0))
        if dist > 0 or dur > 0:
            return dist, dur
    except Exception:
        pass
    return None, None

def make_numbered_icon_html(num: int) -> str:
    # Red pin-like badge with number
    return f"""
    <div style="
        background:#E53935;
        color:white;
        border-radius:18px;
        width:28px;height:28px;
        display:flex;align-items:center;justify-content:center;
        font-weight:700;font-size:14px;
        box-shadow:0 2px 6px rgba(0,0,0,0.35);
        border:2px solid white;
    ">{num}</div>
    """

def build_map(
    center: Tuple[float, float],
    stops: List[Dict[str, Any]],
    route_coords: Optional[List[Tuple[float, float]]] = None,
    route_color: str = "#1976D2",
    click_marker: Optional[Tuple[float, float]] = None,
    zoom: int = 12,
) -> folium.Map:
    m = folium.Map(location=list(center), zoom_start=zoom, control_scale=True)

    # Stops
    for idx, s in enumerate(stops, start=1):
        lat = s.get("lat")
        lng = s.get("lng")
        if lat is None or lng is None:
            continue
        html = make_numbered_icon_html(idx)
        icon = folium.DivIcon(html=html)
        tooltip = s.get("label") or f"Stop {idx}"
        popup = folium.Popup(f"<b>{tooltip}</b><br/>{s.get('address','')}", max_width=350)
        folium.Marker([lat, lng], tooltip=tooltip, popup=popup, icon=icon).add_to(m)

    # Optional click marker (plain pin)
    if click_marker:
        folium.Marker(
            list(click_marker),
            tooltip="Clicked center",
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(m)

    # Route polyline
    if route_coords and len(route_coords) >= 2:
        folium.PolyLine(route_coords, weight=5, opacity=0.9, color=route_color).add_to(m)

    plugins.Fullscreen().add_to(m)
    return m

def ensure_state():
    ss = st.session_state
    if "stops" not in ss:
        ss.stops = []  # list of dicts: label,address,lat,lng,source
    if "center" not in ss:
        ss.center = (28.6139, 77.2090)  # default Delhi
    if "country_filter" not in ss:
        ss.country_filter = "IND"
    if "places_last" not in ss:
        ss.places_last = None
    if "geocode_last" not in ss:
        ss.geocode_last = None
    if "directions_before" not in ss:
        ss.directions_before = None
    if "directions_after" not in ss:
        ss.directions_after = None
    if "vrp_job" not in ss:
        ss.vrp_job = None  # {"id":..., "create_resp":...}
    if "vrp_result" not in ss:
        ss.vrp_result = None
    if "map_versions" not in ss:
        ss.map_versions = {"geocode": 0, "places": 0, "before": 0, "after": 0, "snap": 0, "iso": 0}
    if "places_click" not in ss:
        ss.places_click = None

ensure_state()
ss = st.session_state

# =========================
# HEADER
# =========================
st.title("NextBillion.ai — Visual API Tester")
st.caption("Stops are shared across all tabs. Nothing calls the API until you click a button (forms prevent auto-reruns).")

# =========================
# SIDEBAR: STOP MANAGER
# =========================
with st.sidebar:
    st.header("Stops (shared)")

    st.write(f"Stops loaded: **{len(ss.stops)}**")
    st.write(f"Current center: **{ss.center[0]:.5f},{ss.center[1]:.5f}** | Country filter: **{ss.country_filter}**")

    st.divider()
    st.subheader("Add/Replace Stops (easy input)")

    input_mode = st.radio("Input type", ["Addresses (one per line)", "Lat/Lng (one per line)"], index=0)

    default_text = ""
    pasted = st.text_area("Paste at least 2 lines (supports 20+)", value=default_text, height=180)

    colA, colB = st.columns(2)
    with colA:
        add_btn = st.button("➕ Add", use_container_width=True)
    with colB:
        replace_btn = st.button("♻️ Replace", use_container_width=True)

    clear_btn = st.button("🗑️ Clear stops", use_container_width=True)

    if clear_btn:
        ss.stops = []
        ss.directions_before = None
        ss.directions_after = None
        ss.vrp_job = None
        ss.vrp_result = None

    def add_parsed(lines: List[str], replace: bool):
        items = []
        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            if input_mode.startswith("Lat/Lng"):
                ll = parse_latlng_line(line)
                if ll:
                    items.append({"label": f"Stop {len(items)+1}", "address": "", "lat": ll[0], "lng": ll[1], "source": "manual latlng"})
            else:
                items.append({"label": f"Stop {len(items)+1}", "address": line, "lat": None, "lng": None, "source": "manual address"})
        if replace:
            ss.stops = items
        else:
            # append, re-label later
            ss.stops.extend(items)

    if add_btn or replace_btn:
        lines = pasted.splitlines()
        add_parsed(lines, replace=bool(replace_btn))
        # re-label sequentially
        for idx, s in enumerate(ss.stops, start=1):
            s["label"] = f"Stop {idx}"

    st.divider()
    st.subheader("Generate random stops")
    gen_n = st.number_input("How many", min_value=2, max_value=60, value=20, step=1)
    gen_r = st.slider("Radius (meters)", min_value=200, max_value=30000, value=5000, step=100)
    gen_btn = st.button("🎲 Generate around current center", use_container_width=True)

    if gen_btn:
        pts = random_points_around(ss.center, int(gen_r), int(gen_n))
        ss.stops = [{"label": f"Stop {i+1}", "address": "", "lat": p[0], "lng": p[1], "source": "random"} for i, p in enumerate(pts)]
        ss.directions_before = None
        ss.directions_after = None
        ss.vrp_job = None
        ss.vrp_result = None

# =========================
# GLOBAL ROUTE OPTIONS
# =========================
with st.expander("Global route options (Directions / Matrix / Optimize)", expanded=False):
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])

    with c1:
        mode = st.selectbox("Mode", TRAVEL_MODES, index=0, key="mode")
    with c2:
        traffic = st.selectbox("Traffic", ["No", "Yes"], index=0, key="traffic")
    with c3:
        avoid = st.multiselect("Avoid", NAV_AVOID_OPTIONS, default=[], key="avoid")
    with c4:
        dep = st.number_input("Departure time (unix seconds)", min_value=0, value=0, step=60, key="dep")
        st.caption(f"Human readable: **{human_unix(dep)}**" if dep else "0 = not set")

# =========================
# TABS
# =========================
tab_geocode, tab_places, tab_route_opt, tab_matrix, tab_snap_iso = st.tabs(
    ["Geocode & Map", "Places (Search + Generate Stops)", "Route + Optimize (Before vs After)", "Distance Matrix (NxN)", "Snap-to-Road + Isochrone"]
)

# ==========================================================
# TAB 1: GEOCODE & MAP
# ==========================================================
with tab_geocode:
    st.subheader("Geocode your stops and show them on the map")
    st.caption("Tip: Geocode once → reuse lat/lng across tabs (saves API calls).")

    if len(ss.stops) == 0:
        st.info("Add stops from the sidebar (paste addresses or lat/lng, or generate random stops).")

    with st.form("geocode_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            in_country = st.text_input("Country filter (ISO3 like IND/USA/DEU)", value=ss.country_filter)
        with col2:
            lang = st.text_input("Language", value="en-US")
        do_geocode = st.form_submit_button("🌍 Geocode all address stops (cached)", use_container_width=True)

    if do_geocode and len(ss.stops) > 0:
        ss.country_filter = in_country.strip().upper() or ss.country_filter

        geocoded = 0
        last_status = None
        last_resp = None

        for s in ss.stops:
            if s.get("lat") is not None and s.get("lng") is not None:
                continue
            addr = (s.get("address") or "").strip()
            if not addr:
                continue

            # Geocoding API: /geocode?at=...&q=...&in=countryCode:... :contentReference[oaicite:9]{index=9}
            params = {
                "q": addr,
                "in": f"countryCode:{ss.country_filter}",
                "lang": lang,
            }
            # use center bias if available
            params["at"] = latlng_str(ss.center[0], ss.center[1])

            status, resp = cached_get("/geocode", json.dumps(params, sort_keys=True))
            last_status, last_resp = status, resp

            items = resp.get("items") if isinstance(resp, dict) else None
            if status == 200 and items and isinstance(items, list) and len(items) > 0:
                pos = items[0].get("position", {})
                lat = pos.get("lat")
                lng = pos.get("lng")
                if lat is not None and lng is not None:
                    s["lat"] = float(lat)
                    s["lng"] = float(lng)
                    s["source"] = f"geocode:{ss.country_filter}"
                    geocoded += 1

        ss.geocode_last = {"status": last_status, "resp": last_resp}
        ss.map_versions["geocode"] += 1

        st.success(f"Geocoded {geocoded} stop(s).")

    # Show stops table
    if len(ss.stops) > 0:
        df = pd.DataFrame(ss.stops)
        st.dataframe(df, width="stretch", hide_index=True)

    # Map always renders (persisted)
    center = ss.center
    # auto-center to average of known stops
    coords = [(s["lat"], s["lng"]) for s in ss.stops if s.get("lat") is not None and s.get("lng") is not None]
    if coords:
        lat_avg = sum([c[0] for c in coords]) / len(coords)
        lng_avg = sum([c[1] for c in coords]) / len(coords)
        center = (lat_avg, lng_avg)

    m = build_map(center=center, stops=ss.stops, zoom=11)
    st_folium(m, height=520, width="stretch", key=f"map_geocode_{ss.map_versions['geocode']}")

    # Export JSON
    if len(ss.stops) > 0:
        st.download_button(
            "⬇️ Download Stops JSON",
            data=json.dumps(ss.stops, indent=2),
            file_name="stops.json",
            mime="application/json",
            use_container_width=True,
        )

    if ss.geocode_last:
        with st.expander("Debug: last geocode response"):
            st.write(f"HTTP {ss.geocode_last.get('status')}")
            st.json(ss.geocode_last.get("resp", {}))

# ==========================================================
# TAB 2: PLACES (SEARCH + GENERATE STOPS)
# ==========================================================
with tab_places:
    st.subheader("Search region/city → set center → (optional) search POIs → generate 20+ stops")

    # ---- Region search using geocode (works globally)
    with st.form("region_form", clear_on_submit=False):
        region_q = st.text_input("Region/City/State/Country", value="Delhi")
        region_country = st.text_input("Country filter (ISO3, optional)", value=ss.country_filter)
        region_btn = st.form_submit_button("🔎 Search Region", use_container_width=True)

    region_results = []
    if region_btn:
        params = {
            "q": region_q,
            "lang": "en-US",
        }
        if region_country.strip():
            params["in"] = f"countryCode:{region_country.strip().upper()}"
        status, resp = cached_get("/geocode", json.dumps(params, sort_keys=True))
        items = resp.get("items", []) if isinstance(resp, dict) else []
        region_results = items if isinstance(items, list) else []
        ss.places_last = {"region": {"status": status, "resp": resp}}
        st.success(f"Region search: HTTP {status}")

    if region_results:
        options = []
        for it in region_results[:10]:
            title = it.get("title") or it.get("address", {}).get("label") or "Result"
            pos = it.get("position", {})
            options.append((title, pos.get("lat"), pos.get("lng"), it))
        pick = st.selectbox("Pick a region result", options, format_func=lambda x: x[0])
        if st.button("✅ Use picked region as center", use_container_width=True):
            if pick[1] is not None and pick[2] is not None:
                ss.center = (float(pick[1]), float(pick[2]))
                if region_country.strip():
                    ss.country_filter = region_country.strip().upper()
                ss.map_versions["places"] += 1
                st.success(f"Center set to {ss.center[0]:.6f},{ss.center[1]:.6f}")

    st.write(f"**Current center:** {ss.center[0]:.6f},{ss.center[1]:.6f} | **Country filter:** {ss.country_filter}")

    # ---- Click map to set center
    st.caption("Option: click on map to drop a pin, then use it as center.")
    click_map = build_map(center=ss.center, stops=[], click_marker=ss.places_click, zoom=11)
    click_data = st_folium(click_map, height=420, width="stretch", key=f"map_places_click_{ss.map_versions['places']}")
    if click_data and click_data.get("last_clicked"):
        lat = click_data["last_clicked"]["lat"]
        lng = click_data["last_clicked"]["lng"]
        ss.places_click = (lat, lng)

    colx, coly = st.columns(2)
    with colx:
        if st.button("📍 Use clicked pin as center", use_container_width=True):
            if ss.places_click:
                ss.center = ss.places_click
                ss.map_versions["places"] += 1
    with coly:
        if st.button("❌ Clear clicked pin", use_container_width=True):
            ss.places_click = None
            ss.map_versions["places"] += 1

    st.divider()

    # ---- Discover (POI keyword search) :contentReference[oaicite:10]{index=10}
    st.markdown("### Optional: POI keyword search (Discover) and add results as stops")
    with st.form("poi_form", clear_on_submit=False):
        poi_q = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="petrol pump")
        poi_radius = st.slider("Search radius (m)", min_value=500, max_value=50000, value=5000, step=100)
        poi_limit = st.slider("Max results", min_value=1, max_value=50, value=20, step=1)
        poi_btn = st.form_submit_button("🔎 Search Places (POIs)", use_container_width=True)

    poi_items = []
    if poi_btn:
        params = {
            "at": latlng_str(ss.center[0], ss.center[1]),
            "q": poi_q,
            "limit": int(poi_limit),
            "in": f"countryCode:{ss.country_filter}",
        }
        status, resp = cached_get("/discover", json.dumps(params, sort_keys=True))
        ss.places_last = {"poi": {"status": status, "resp": resp}}
        st.success(f"Places response: HTTP {status}")
        poi_items = resp.get("items", []) if isinstance(resp, dict) else []

    if poi_items:
        rows = []
        for it in poi_items:
            title = it.get("title") or it.get("address", {}).get("label") or it.get("id", "Place")
            pos = it.get("position", {})
            lat = pos.get("lat")
            lng = pos.get("lng")
            rows.append({"title": title, "lat": lat, "lng": lng, "raw": it})

        dfp = pd.DataFrame([{"title": r["title"], "lat": r["lat"], "lng": r["lng"]} for r in rows])
        st.dataframe(dfp, width="stretch", hide_index=True)

        if st.button("➕ Add all POI results as stops", use_container_width=True):
            new = []
            for r in rows:
                if r["lat"] is None or r["lng"] is None:
                    continue
                new.append({
                    "label": f"Stop {len(ss.stops) + len(new) + 1}",
                    "address": r["title"],
                    "lat": float(r["lat"]),
                    "lng": float(r["lng"]),
                    "source": "discover",
                })
            ss.stops.extend(new)
            # re-label
            for i, s in enumerate(ss.stops, start=1):
                s["label"] = f"Stop {i}"
            ss.map_versions["places"] += 1
            st.success(f"Added {len(new)} stop(s).")

        st.download_button(
            "⬇️ Download Places JSON",
            data=json.dumps(ss.places_last, indent=2),
            file_name="places.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("No POI results yet. Try a broader keyword (e.g., 'gas', 'market') or increase radius.")

    st.divider()

    # ---- Generate random stops around center (without keyword)
    st.markdown("### Generate random test stops around the current center (no keyword needed)")
    with st.form("gen_form", clear_on_submit=False):
        gn = st.number_input("How many stops", min_value=2, max_value=60, value=20, step=1)
        gr = st.slider("Generation radius (m)", min_value=200, max_value=50000, value=8000, step=100)
        gbtn = st.form_submit_button("🎲 Generate stops", use_container_width=True)

    if gbtn:
        pts = random_points_around(ss.center, int(gr), int(gn))
        ss.stops = [{"label": f"Stop {i+1}", "address": "", "lat": p[0], "lng": p[1], "source": "random(center)"} for i, p in enumerate(pts)]
        ss.directions_before = None
        ss.directions_after = None
        ss.vrp_job = None
        ss.vrp_result = None
        ss.map_versions["places"] += 1
        st.success(f"Generated {len(ss.stops)} stops.")

    # Show stops on map (persisted)
    if len(ss.stops) > 0:
        center2 = ss.center
        coords2 = [(s["lat"], s["lng"]) for s in ss.stops if s.get("lat") is not None and s.get("lng") is not None]
        if coords2:
            lat_avg = sum([c[0] for c in coords2]) / len(coords2)
            lng_avg = sum([c[1] for c in coords2]) / len(coords2)
            center2 = (lat_avg, lng_avg)
        m2 = build_map(center=center2, stops=ss.stops, zoom=11)
        st_folium(m2, height=520, width="stretch", key=f"map_places_stops_{ss.map_versions['places']}")

# ==========================================================
# TAB 3: ROUTE + OPTIMIZE (BEFORE VS AFTER)
# ==========================================================
with tab_route_opt:
    st.subheader("Compute route (Before) → run optimization → recompute route (After) + compare")

    if len(ss.stops) < 2:
        st.warning("Need at least 2 stops. Add or generate stops first.")
    else:
        coords = [(s["lat"], s["lng"]) for s in ss.stops if s.get("lat") is not None and s.get("lng") is not None]
        if len(coords) < 2:
            st.warning("Stops must have lat/lng. Use Geocode tab or generate random lat/lng stops.")
        else:
            # Directions form
            with st.form("dir_form", clear_on_submit=False):
                st.markdown("### Step 1 — Directions (Before)")
                o = coords[0]
                d = coords[-1]
                way = coords[1:-1]  # up to 18+ fine

                compute_btn = st.form_submit_button("🧭 Compute route (Before)", use_container_width=True)

            if compute_btn:
                params = {
                    "origin": latlng_str(o[0], o[1]),
                    "destination": latlng_str(d[0], d[1]),
                    "mode": st.session_state.get("mode", "car"),
                }
                if way:
                    params["waypoints"] = "|".join([latlng_str(x[0], x[1]) for x in way])
                if st.session_state.get("avoid"):
                    params["avoid"] = "|".join(st.session_state["avoid"])
                if st.session_state.get("traffic") == "Yes":
                    params["traffic"] = "true"
                if st.session_state.get("dep", 0):
                    params["departure_time"] = int(st.session_state["dep"])

                # Navigation API: /navigation/json :contentReference[oaicite:11]{index=11}
                status, resp = cached_get("/navigation/json", json.dumps(params, sort_keys=True))
                ss.directions_before = {"status": status, "resp": resp, "params": params}
                ss.map_versions["before"] += 1

                if status == 200:
                    st.success("Directions (Before) computed.")
                else:
                    st.error(f"Directions (Before) failed: HTTP {status}")
                    st.json(resp)

            # Show BEFORE map + metrics
            before_cols = st.columns(2)
            with before_cols[0]:
                st.markdown("#### Route map (Before)")
                route_coords = []
                dist_m = dur_s = None
                if ss.directions_before and ss.directions_before.get("status") == 200:
                    route_coords = extract_route_coords(ss.directions_before["resp"])
                    dist_m, dur_s = compute_directions_totals(ss.directions_before["resp"])

                map_center = ss.center
                if coords:
                    map_center = coords[0]
                m_before = build_map(center=map_center, stops=ss.stops, route_coords=route_coords, route_color="#1976D2", zoom=11)
                st_folium(m_before, height=520, width="stretch", key=f"map_before_{ss.map_versions['before']}")

                if dist_m is not None and dur_s is not None:
                    st.metric("Before distance (km)", f"{km(dist_m):.2f}")
                    st.metric("Before duration (min)", f"{mins(dur_s):.1f}")

                if ss.directions_before:
                    st.download_button(
                        "⬇️ Download Directions (Before) JSON",
                        data=json.dumps(ss.directions_before, indent=2),
                        file_name="directions_before.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            # Optimization controls
            with before_cols[1]:
                st.markdown("#### Step 2 — Optimization (VRP v2)")
                st.caption("Uses the v2 payload format with `locations` as a list of `lat,lng` strings + jobs referencing location_index. :contentReference[oaicite:12]{index=12}")

                with st.form("opt_form", clear_on_submit=False):
                    objective = st.selectbox("Optimization objective", OPT_OBJECTIVES, index=OPT_OBJECTIVES.index("duration"))
                    run_opt = st.form_submit_button("⚙️ Run optimization (VRP v2)", use_container_width=True)

                if run_opt:
                    # Build request similar to user's working sample structure
                    locations = [latlng_str(lat, lng) for (lat, lng) in coords]  # includes depot at index 0
                    jobs = []
                    for i in range(1, len(locations)):
                        jobs.append({"id": i, "location_index": i})

                    body = {
                        "locations": locations,
                        "vehicles": [{"id": 1, "start_location_index": 0, "end_location_index": 0}],
                        "jobs": jobs,
                        "options": {
                            "objective": {"travel_cost": objective}
                        },
                    }

                    status, resp = cached_post("/optimization/v2", "{}", json.dumps(body, sort_keys=True))
                    ss.vrp_job = {"status": status, "resp": resp, "request": body}
                    ss.vrp_result = None

                    if status != 200:
                        st.error(f"VRP create failed: HTTP {status}")
                        st.json(resp)
                    else:
                        st.success("VRP job created.")
                        # Expect an id in response
                        job_id = resp.get("id") or resp.get("job_id") or resp.get("result", {}).get("id")
                        ss.vrp_job["job_id"] = job_id

                # Poll result
                if ss.vrp_job and ss.vrp_job.get("job_id"):
                    job_id = ss.vrp_job["job_id"]
                    poll_btn = st.button("📥 Fetch optimization result", use_container_width=True)

                    if poll_btn:
                        with st.spinner("Fetching result..."):
                            # Poll up to ~10 times but keep light
                            result = None
                            last = None
                            for _ in range(10):
                                status, rr = cached_get("/optimization/v2/result", json.dumps({"id": job_id}, sort_keys=True))
                                last = {"status": status, "resp": rr}
                                # heuristic: if routes present, done
                                if isinstance(rr, dict) and (rr.get("routes") or rr.get("result") or rr.get("solution")):
                                    result = last
                                    break
                                time.sleep(1)
                            ss.vrp_result = result or last

                # Show AFTER if possible
                st.markdown("#### Step 3 — Recompute Directions (After)")
                after_route_coords = []
                after_dist_m = after_dur_s = None
                ordered_coords = None

                if ss.vrp_result and ss.vrp_result.get("status") == 200:
                    rr = ss.vrp_result["resp"]

                    # Try to extract an optimized order: look for routes[0].steps with location_index
                    seq = []
                    try:
                        routes = rr.get("routes") or rr.get("result", {}).get("routes") or []
                        r0 = routes[0] if routes else None
                        steps = r0.get("steps", []) if r0 else []
                        for stp in steps:
                            li = stp.get("location_index")
                            if li is not None:
                                seq.append(int(li))
                    except Exception:
                        seq = []

                    # Fallback: if no steps, just keep original order
                    if not seq:
                        seq = list(range(len(coords)))

                    # Build optimized stop order (unique, preserve)
                    seen = set()
                    seq2 = []
                    for i in seq:
                        if 0 <= i < len(coords) and i not in seen:
                            seq2.append(i)
                            seen.add(i)
                    if 0 not in seen:
                        seq2 = [0] + seq2

                    ordered_coords = [coords[i] for i in seq2]

                    # Compute directions using optimized order
                    if len(ordered_coords) >= 2:
                        o2 = ordered_coords[0]
                        d2 = ordered_coords[-1]
                        w2 = ordered_coords[1:-1]

                        params2 = {
                            "origin": latlng_str(o2[0], o2[1]),
                            "destination": latlng_str(d2[0], d2[1]),
                            "mode": st.session_state.get("mode", "car"),
                        }
                        if w2:
                            params2["waypoints"] = "|".join([latlng_str(x[0], x[1]) for x in w2])
                        if st.session_state.get("avoid"):
                            params2["avoid"] = "|".join(st.session_state["avoid"])
                        if st.session_state.get("traffic") == "Yes":
                            params2["traffic"] = "true"
                        if st.session_state.get("dep", 0):
                            params2["departure_time"] = int(st.session_state["dep"])

                        status2, resp2 = cached_get("/navigation/json", json.dumps(params2, sort_keys=True))
                        ss.directions_after = {"status": status2, "resp": resp2, "params": params2, "order": seq2}
                        ss.map_versions["after"] += 1

                # Show AFTER map + comparison
                if ss.directions_after and ss.directions_after.get("status") == 200:
                    after_route_coords = extract_route_coords(ss.directions_after["resp"])
                    after_dist_m, after_dur_s = compute_directions_totals(ss.directions_after["resp"])

                st.markdown("#### Optimized route map (After)")
                # Show stops in optimized order on map by reordering markers
                map_stops_after = ss.stops
                if ss.directions_after and ss.directions_after.get("order"):
                    order = ss.directions_after["order"]
                    # reorder stops for numbering on the "After" map
                    reordered = []
                    for idx in order:
                        if 0 <= idx < len(ss.stops):
                            reordered.append(ss.stops[idx])
                    # append any missing
                    seen_ids = {id(x) for x in reordered}
                    for s in ss.stops:
                        if id(s) not in seen_ids:
                            reordered.append(s)
                    map_stops_after = reordered

                m_after = build_map(center=ss.center, stops=map_stops_after, route_coords=after_route_coords, route_color="#2E7D32", zoom=11)
                st_folium(m_after, height=520, width="stretch", key=f"map_after_{ss.map_versions['after']}")

                # Comparison metrics
                before_dist_m = before_dur_s = None
                if ss.directions_before and ss.directions_before.get("status") == 200:
                    before_dist_m, before_dur_s = compute_directions_totals(ss.directions_before["resp"])

                if before_dist_m and before_dur_s and after_dist_m and after_dur_s:
                    dist_saved = before_dist_m - after_dist_m
                    dur_saved = before_dur_s - after_dur_s
                    dist_pct = (dist_saved / before_dist_m) * 100.0 if before_dist_m else 0
                    dur_pct = (dur_saved / before_dur_s) * 100.0 if before_dur_s else 0

                    st.markdown("### Before vs After (Savings)")
                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("Distance saved (km)", f"{km(dist_saved):.2f}", f"{dist_pct:.1f}%")
                    cB.metric("Time saved (min)", f"{mins(dur_saved):.1f}", f"{dur_pct:.1f}%")
                    cC.metric("After distance (km)", f"{km(after_dist_m):.2f}")
                    cD.metric("After duration (min)", f"{mins(after_dur_s):.1f}")

                # Downloads
                if ss.vrp_job:
                    st.download_button(
                        "⬇️ Download VRP Create JSON",
                        data=json.dumps(ss.vrp_job, indent=2),
                        file_name="vrp_create.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                if ss.vrp_result:
                    st.download_button(
                        "⬇️ Download VRP Result JSON",
                        data=json.dumps(ss.vrp_result, indent=2),
                        file_name="vrp_result.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                if ss.directions_after:
                    st.download_button(
                        "⬇️ Download Directions (After) JSON",
                        data=json.dumps(ss.directions_after, indent=2),
                        file_name="directions_after.json",
                        mime="application/json",
                        use_container_width=True,
                    )

# ==========================================================
# TAB 4: DISTANCE MATRIX (NxN)
# ==========================================================
with tab_matrix:
    st.subheader("Distance Matrix (NxN) — supports 20+ points")
    if len(ss.stops) < 2:
        st.warning("Need at least 2 stops.")
    else:
        coords = [(s["lat"], s["lng"]) for s in ss.stops if s.get("lat") is not None and s.get("lng") is not None]
        if len(coords) < 2:
            st.warning("Stops must have lat/lng. Use Geocode tab or generate random lat/lng stops.")
        else:
            max_n = min(60, len(coords))
            n = st.number_input("Use first N stops", min_value=2, max_value=max_n, value=min(20, max_n), step=1)

            with st.form("dm_form", clear_on_submit=False):
                run_dm = st.form_submit_button("📏 Compute NxN Distance Matrix", use_container_width=True)

            if run_dm:
                sub = coords[: int(n)]
                origins = "|".join([latlng_str(a[0], a[1]) for a in sub])
                destinations = origins

                params = {
                    "origins": origins,
                    "destinations": destinations,
                    "mode": st.session_state.get("mode", "car"),
                }
                if st.session_state.get("avoid"):
                    params["avoid"] = "|".join(st.session_state["avoid"])
                if st.session_state.get("traffic") == "Yes":
                    params["traffic"] = "true"
                if st.session_state.get("dep", 0):
                    params["departure_time"] = int(st.session_state["dep"])

                # Distance Matrix API endpoint (product docs list it; common path is /distancematrix/json)
                status, resp = cached_get("/distancematrix/json", json.dumps(params, sort_keys=True))
                st.success(f"Distance Matrix: HTTP {status}")

                if status != 200:
                    st.error("Distance Matrix failed.")
                    st.json(resp)
                else:
                    # Parse into dataframe
                    rows = resp.get("rows", [])
                    mat_km = []
                    mat_min = []
                    for r in rows:
                        elems = r.get("elements", [])
                        km_row = []
                        min_row = []
                        for e in elems:
                            d = e.get("distance", {}).get("value")
                            t = e.get("duration", {}).get("value")
                            km_row.append(km(float(d)) if d is not None else None)
                            min_row.append(mins(float(t)) if t is not None else None)
                        mat_km.append(km_row)
                        mat_min.append(min_row)

                    df_km = pd.DataFrame(mat_km)
                    df_min = pd.DataFrame(mat_min)
                    st.markdown("#### Distance (km)")
                    st.dataframe(df_km, width="stretch")
                    st.markdown("#### Duration (min)")
                    st.dataframe(df_min, width="stretch")

                    st.download_button(
                        "⬇️ Download Distance Matrix JSON",
                        data=json.dumps(resp, indent=2),
                        file_name="distance_matrix.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇️ Download Distance (km) CSV",
                        data=df_km.to_csv(index=False),
                        file_name="distance_km.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇️ Download Duration (min) CSV",
                        data=df_min.to_csv(index=False),
                        file_name="duration_min.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# ==========================================================
# TAB 5: SNAP-TO-ROAD + ISOCHRONE
# ==========================================================
with tab_snap_iso:
    st.subheader("Snap-to-road + Isochrone (stable maps)")

    coords = [(s["lat"], s["lng"]) for s in ss.stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(coords) < 2:
        st.info("Add at least 2 stops with lat/lng to test Snap-to-road. Isochrone works with center only.")
    else:
        # ----- Snap-to-road 
        st.markdown("### Snap-to-road")
        max_n = min(60, len(coords))
        nmax = max(2, max_n)
        # Fix slider min/max issue:
        if nmax <= 2:
            n_path = 2
            st.info("Only 2 coordinate points available for Snap-to-road path.")
        else:
            n_path = st.slider("N points for path (use first N)", min_value=2, max_value=nmax, value=min(10, nmax), step=1)

        with st.form("snap_form", clear_on_submit=False):
            run_snap = st.form_submit_button("🧲 Snap path to road", use_container_width=True)

        snap_coords = None
        if run_snap:
            path_pts = coords[: int(n_path)]
            path = "|".join([latlng_str(a[0], a[1]) for a in path_pts])
            params = {"path": path}
            status, resp = cached_get("/snapToRoads/json", json.dumps(params, sort_keys=True))
            st.success(f"Snap-to-road: HTTP {status}")

            if status != 200:
                st.error("Snap-to-road failed.")
                st.json(resp)
            else:
                # Response commonly includes snappedPoints or geometry; handle both
                snapped = []
                if "snappedPoints" in resp and isinstance(resp["snappedPoints"], list):
                    for p in resp["snappedPoints"]:
                        loc = p.get("location", {})
                        lat = loc.get("latitude")
                        lng = loc.get("longitude")
                        if lat is not None and lng is not None:
                            snapped.append((float(lat), float(lng)))
                elif "coordinates" in resp:
                    # if GeoJSON-like [lng,lat]
                    try:
                        snapped = [(c[1], c[0]) for c in resp["coordinates"]]
                    except Exception:
                        snapped = []

                snap_coords = snapped if len(snapped) >= 2 else None
                ss.map_versions["snap"] += 1

                st.download_button(
                    "⬇️ Download Snap-to-road JSON",
                    data=json.dumps(resp, indent=2),
                    file_name="snap_to_road.json",
                    mime="application/json",
                    use_container_width=True,
                )

        m_snap = build_map(center=ss.center, stops=ss.stops[: int(min(20, len(ss.stops)))], route_coords=snap_coords, route_color="#6A1B9A", zoom=11)
        st_folium(m_snap, height=520, width="stretch", key=f"map_snap_{ss.map_versions['snap']}")

    st.divider()

    # ----- Isochrone :contentReference[oaicite:14]{index=14}
    st.markdown("### Isochrone")
    with st.form("iso_form", clear_on_submit=False):
        iso_mode = st.selectbox("Mode", TRAVEL_MODES, index=0, key="iso_mode")
        iso_type = st.selectbox("Range type", ["time", "distance"], index=0)
        iso_val = st.number_input("Range value (seconds if time, meters if distance)", min_value=60, value=900, step=60)
        iso_btn = st.form_submit_button("🟦 Compute isochrone", use_container_width=True)

    iso_geojson = None
    if iso_btn:
        # Isochrone endpoint: /isochrone/json :contentReference[oaicite:15]{index=15}
        params = {
            "mode": iso_mode,
            "lat": ss.center[0],
            "lng": ss.center[1],
        }
        if iso_type == "time":
            params["range_type"] = "time"
            params["range"] = int(iso_val)
        else:
            params["range_type"] = "distance"
            params["range"] = int(iso_val)

        status, resp = cached_get("/isochrone/json", json.dumps(params, sort_keys=True))
        st.success(f"Isochrone: HTTP {status}")
        if status != 200:
            st.error("Isochrone failed.")
            st.json(resp)
        else:
            # Expect GeoJSON-ish polygons
            iso_geojson = resp.get("features") or resp.get("polygons") or resp
            ss.map_versions["iso"] += 1
            st.download_button(
                "⬇️ Download Isochrone JSON",
                data=json.dumps(resp, indent=2),
                file_name="isochrone.json",
                mime="application/json",
                use_container_width=True,
            )

    # Render isochrone map persistently
    m_iso = folium.Map(location=list(ss.center), zoom_start=11, control_scale=True)
    folium.Marker(list(ss.center), tooltip="Center", icon=folium.Icon(color="red")).add_to(m_iso)
    if iso_geojson:
        try:
            # If it's full feature collection
            if isinstance(iso_geojson, dict) and iso_geojson.get("type") == "FeatureCollection":
                folium.GeoJson(iso_geojson).add_to(m_iso)
            # If list of features
            elif isinstance(iso_geojson, list):
                folium.GeoJson({"type": "FeatureCollection", "features": iso_geojson}).add_to(m_iso)
            # Else try as-is
            else:
                folium.GeoJson(iso_geojson).add_to(m_iso)
        except Exception:
            pass
    plugins.Fullscreen().add_to(m_iso)
    st_folium(m_iso, height=520, width="stretch", key=f"map_iso_{ss.map_versions['iso']}")

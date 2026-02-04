# app.py — NextBillion.ai Visual API Tester (stable maps + VRP v2 payload fix)
# Run: streamlit run app.py

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
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# =============================
# CONFIG
# =============================
NB_API_KEY = "a08a2b15af0f432c8e438403bc2b00e3"  # embedded as requested
BASE_URL = "https://api.nextbillion.io"          # NextBillion public API base
UA = "NB-Visual-Tester/1.0"

DEFAULT_CENTER = (28.6139, 77.2090)  # New Delhi
DEFAULT_COUNTRY = "IND"              # ISO3-like used by many NB endpoints

st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")

# =============================
# SESSION STATE INIT
# =============================
def ss_init():
    ss = st.session_state
    ss.setdefault("api_key", NB_API_KEY)

    # center: where we generate/search around
    ss.setdefault("center", {"lat": DEFAULT_CENTER[0], "lng": DEFAULT_CENTER[1], "country": DEFAULT_COUNTRY})
    ss.setdefault("last_clicked", None)  # {"lat":..., "lng":...}

    # stops table
    ss.setdefault("stops", [])  # list of dicts {label,address,lat,lng,source}

    # persisted results (so maps never disappear)
    ss.setdefault("geocode_last", None)
    ss.setdefault("places_last", None)
    ss.setdefault("directions_before", None)
    ss.setdefault("directions_after", None)
    ss.setdefault("vrp_last", None)
    ss.setdefault("dm_last", None)
    ss.setdefault("snap_last", None)
    ss.setdefault("iso_last", None)

    # settings / options
    ss.setdefault("global_opts", {
        "mode": "car",
        "avoid": [],
        "units": "metric",
        "traffic": False,
        "departure_time": None,  # unix seconds
        "alternatives": False,
    })

ss_init()

# =============================
# HELPERS
# =============================
def http_get(path: str, params: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    params = dict(params or {})
    params["key"] = st.session_state.api_key
    headers = {"User-Agent": UA}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def http_post(path: str, payload: Dict[str, Any], params: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    params = dict(params or {})
    params["key"] = st.session_state.api_key
    headers = {"User-Agent": UA, "Content-Type": "application/json"}
    r = requests.post(url, params=params, headers=headers, data=json.dumps(payload), timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def normalize_latlng_str(lat: float, lng: float) -> str:
    return f"{lat:.6f},{lng:.6f}"

def epoch_to_human(ts: Optional[int]) -> str:
    if not ts:
        return "None"
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def human_to_epoch(dt_str: str) -> Optional[int]:
    # expects "YYYY-mm-dd HH:MM" local-ish, convert to UTC naive
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
        # treat as local time then convert to epoch as-is (good enough for testing)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None

def parse_stops_df() -> pd.DataFrame:
    rows = st.session_state.stops
    if not rows:
        return pd.DataFrame(columns=["label", "address", "lat", "lng", "source"])
    return pd.DataFrame(rows)

def set_stops_from_df(df: pd.DataFrame):
    df = df.copy()
    for c in ["label", "address", "lat", "lng", "source"]:
        if c not in df.columns:
            df[c] = None
    stops = []
    for _, r in df.iterrows():
        stops.append({
            "label": str(r.get("label") or "").strip() or None,
            "address": str(r.get("address") or "").strip() or None,
            "lat": None if pd.isna(r.get("lat")) else float(r.get("lat")),
            "lng": None if pd.isna(r.get("lng")) else float(r.get("lng")),
            "source": str(r.get("source") or "").strip() or None,
        })
    st.session_state.stops = stops

def add_stops(new_stops: List[Dict[str, Any]]):
    # avoid duplicates by (lat,lng,address)
    existing = {(s.get("lat"), s.get("lng"), (s.get("address") or "")) for s in st.session_state.stops}
    for s in new_stops:
        key = (s.get("lat"), s.get("lng"), (s.get("address") or ""))
        if key not in existing:
            st.session_state.stops.append(s)
            existing.add(key)

def clear_results():
    ss = st.session_state
    ss.geocode_last = None
    ss.places_last = None
    ss.directions_before = None
    ss.directions_after = None
    ss.vrp_last = None
    ss.dm_last = None
    ss.snap_last = None
    ss.iso_last = None

# ---------- Polyline decode (robust) ----------
def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    # Standard polyline decoding (Google/HERE style)
    if not isinstance(polyline_str, str) or not polyline_str:
        return []
    index, lat, lng = 0, 0, 0
    coordinates = []
    factor = 10 ** precision

    while index < len(polyline_str):
        result, shift = 0, 0
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

        result, shift = 0, 0
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

def extract_route_coords(directions_json: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Tries multiple schemas:
    - routes[0].geometry (polyline string)
    - routes[0].polyline (polyline string)
    - routes[0].shape (list of "lat,lng")
    - routes[0].legs[*].steps[*].geometry (polyline string)
    """
    if not isinstance(directions_json, dict):
        return []

    routes = directions_json.get("routes") or directions_json.get("route") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    r0 = routes[0]
    for k in ["geometry", "polyline"]:
        if isinstance(r0.get(k), str):
            coords = decode_polyline(r0[k], precision=5)
            if coords:
                return coords

    if isinstance(r0.get("shape"), list) and r0["shape"]:
        coords = []
        for s in r0["shape"]:
            if isinstance(s, str) and "," in s:
                a, b = s.split(",", 1)
                coords.append((float(a), float(b)))
        if coords:
            return coords

    # deep steps
    legs = r0.get("legs") or []
    coords = []
    for leg in legs:
        steps = leg.get("steps") or []
        for stp in steps:
            g = stp.get("geometry")
            if isinstance(g, str):
                coords.extend(decode_polyline(g, precision=5))
    return coords

def extract_total_distance_duration(directions_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (distance_km, duration_min) if possible.
    Tries common fields:
    - routes[0].distance (meters) & routes[0].duration (seconds)
    - routes[0].summary.distance / summary.duration
    """
    if not isinstance(directions_json, dict):
        return None, None
    routes = directions_json.get("routes") or []
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return None, None
    r0 = routes[0]
    dist_m = None
    dur_s = None
    if isinstance(r0.get("distance"), (int, float)):
        dist_m = float(r0["distance"])
    if isinstance(r0.get("duration"), (int, float)):
        dur_s = float(r0["duration"])
    summ = r0.get("summary") or {}
    if dist_m is None and isinstance(summ.get("distance"), (int, float)):
        dist_m = float(summ["distance"])
    if dur_s is None and isinstance(summ.get("duration"), (int, float)):
        dur_s = float(summ["duration"])

    dist_km = None if dist_m is None else dist_m / 1000.0
    dur_min = None if dur_s is None else dur_s / 60.0
    return dist_km, dur_min

# ---------- Map rendering (stable) ----------
def base_map(center_lat: float, center_lng: float, zoom: int = 12) -> folium.Map:
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, control_scale=True)
    return m

def add_numbered_markers(m: folium.Map, stops: List[Dict[str, Any]], color: str = "red"):
    for i, s in enumerate(stops, start=1):
        lat, lng = s.get("lat"), s.get("lng")
        if lat is None or lng is None:
            continue
        popup = folium.Popup(
            f"<b>{i}. {s.get('label') or ''}</b><br/>{s.get('address') or ''}<br/>{lat:.6f},{lng:.6f}",
            max_width=300
        )
        folium.Marker(
            location=[lat, lng],
            popup=popup,
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:{color};
                    color:white;
                    border-radius:16px;
                    width:28px;
                    height:28px;
                    text-align:center;
                    line-height:28px;
                    font-weight:700;
                    border:2px solid white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                ">{i}</div>
                """
            ),
        ).add_to(m)

def add_route_polyline_with_arrows(m: folium.Map, coords: List[Tuple[float, float]]):
    if not coords or len(coords) < 2:
        return
    pl = folium.PolyLine(coords, weight=6, opacity=0.8)
    pl.add_to(m)
    # arrows
    PolyLineTextPath(
        pl,
        "▶",
        repeat=True,
        offset=10,
        attributes={"fill": "black", "font-weight": "bold", "font-size": "16"},
    ).add_to(m)

def render_map(m: folium.Map, key: str, height: int = 520) -> Dict[str, Any]:
    # stable render; key MUST remain constant per map slot
    return st_folium(m, height=height, width="stretch", key=key)

# =============================
# UI — SIDEBAR (Stops input)
# =============================
st.title("NextBillion.ai — Visual API Tester")

ss = st.session_state
st.caption(f"Stops loaded: **{len(ss.stops)}**")

with st.sidebar:
    st.header("Config")
    ss.api_key = st.text_input("NextBillion API Key", value=ss.api_key, type="password")

    st.divider()
    st.subheader("Stops (20+ addresses)")

    mode = st.radio("How to input stops?", ["Addresses (one per line)", "Lat/Lng (one per line)"], index=0)

    stop_text = st.text_area(
        "Paste at least 2 lines",
        height=180,
        placeholder=(
            "Example (addresses):\n"
            "78-14 138th St Flushing, NY 11367\n"
            "2451 Webb Ave Bronx, NY 10468\n\n"
            "Example (lat,lng):\n"
            "28.6139,77.2090\n"
            "28.7041,77.1025"
        )
    )

    c1, c2 = st.columns(2)
    with c1:
        add_btn = st.button("➕ Add", use_container_width=True)
    with c2:
        replace_btn = st.button("♻️ Replace", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        clear_btn = st.button("🧹 Clear Stops", use_container_width=True)
    with c4:
        clear_res_btn = st.button("🧽 Clear Results", use_container_width=True)

    if clear_btn:
        ss.stops = []
    if clear_res_btn:
        clear_results()

    if add_btn or replace_btn:
        lines = [x.strip() for x in stop_text.splitlines() if x.strip()]
        new_stops = []
        if mode.startswith("Addresses"):
            for i, addr in enumerate(lines, start=1):
                new_stops.append({"label": f"Stop {i}", "address": addr, "lat": None, "lng": None, "source": "Pasted addresses"})
        else:
            for i, ll in enumerate(lines, start=1):
                if "," not in ll:
                    continue
                a, b = ll.split(",", 1)
                try:
                    lat = float(a.strip())
                    lng = float(b.strip())
                    new_stops.append({"label": f"Stop {i}", "address": None, "lat": lat, "lng": lng, "source": "Pasted lat,lng"})
                except Exception:
                    continue

        if replace_btn:
            ss.stops = new_stops
        else:
            add_stops(new_stops)

    st.divider()
    st.subheader("Center / Country")
    ss.center["country"] = st.text_input("Country filter (3-letter)", value=ss.center["country"])
    st.caption(f"Current center: {ss.center['lat']:.6f},{ss.center['lng']:.6f} | Country: {ss.center['country']}")

# =============================
# GLOBAL ROUTE OPTIONS
# =============================
with st.expander("Global route options (Directions / Matrix / Optimize)", expanded=False):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        ss.global_opts["mode"] = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0)
    with colB:
        ss.global_opts["units"] = st.selectbox("Units", ["metric", "imperial"], index=0)
    with colC:
        ss.global_opts["traffic"] = st.selectbox("Use traffic", ["No", "Yes"], index=0) == "Yes"
    with colD:
        ss.global_opts["alternatives"] = st.selectbox("Alternatives", ["No", "Yes"], index=0) == "Yes"

    avoid_choices = [
        "toll_roads", "highways", "ferries", "tunnels", "unpaved_roads",
        "sharp_turns", "u_turns", "borders"
    ]
    ss.global_opts["avoid"] = st.multiselect("Avoid (dropdown to prevent typos)", avoid_choices, default=ss.global_opts["avoid"])

    dep_col1, dep_col2 = st.columns([2, 1])
    with dep_col1:
        dt_str = st.text_input("Departure time (human) — format: YYYY-mm-dd HH:MM", value="")
    with dep_col2:
        if st.button("Set departure", use_container_width=True):
            ss.global_opts["departure_time"] = human_to_epoch(dt_str)

    st.caption(f"Departure (unix): {ss.global_opts['departure_time']}  |  {epoch_to_human(ss.global_opts['departure_time'])}")

# =============================
# TABS
# =============================
tabs = st.tabs([
    "Geocode & Map",
    "Places (Search + Generate Stops)",
    "Route + Optimize (Before vs After)",
    "Distance Matrix (NxN)",
    "Snap-to-Road + Isochrone",
])

# =============================
# TAB 1 — GEOCODE & MAP
# =============================
with tabs[0]:
    st.subheader("Geocode your stops and show them on the map")

    df = parse_stops_df()
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", key="stops_editor_main")
    set_stops_from_df(edited)

    c1, c2 = st.columns([1, 1])
    with c1:
        geocode_btn = st.button("🧭 Geocode all stops (cached)", use_container_width=True)
    with c2:
        create_tbl_btn = st.button("🧾 Create a clean table", use_container_width=True)

    if create_tbl_btn:
        # normalize labels if missing
        df = parse_stops_df()
        if not df.empty:
            for i in range(len(df)):
                if not df.loc[i, "label"] or str(df.loc[i, "label"]).strip() == "None":
                    df.loc[i, "label"] = f"Stop {i+1}"
            set_stops_from_df(df)

    if geocode_btn:
        # Prefer Batch Geocode if available; else fall back to Forward Geocode per address.
        # We keep it simple and reliable.
        df = parse_stops_df()
        addrs = [a for a in df["address"].tolist() if isinstance(a, str) and a.strip()]
        if not addrs:
            st.warning("No addresses found to geocode. (Lat/Lng-only stops are already usable.)")
        else:
            payload = {"addresses": addrs}
            # Try Batch Geocode (docs list it); if it fails, fall back.
            status, data = http_post("/batch_geocode", payload, timeout=60)
            if status != 200:
                # fallback forward geocode
                resolved = []
                prog = st.progress(0)
                for i, addr in enumerate(addrs):
                    params = {"q": addr}
                    # Many place APIs use /geocode?; NextBillion docs list Forward Geocode but
                    # path differs by deployment; we try /geocode first, then /forward.
                    s1, d1 = http_get("/geocode", params=params, timeout=30)
                    if s1 != 200:
                        s1, d1 = http_get("/places/v1/geocode", params=params, timeout=30)
                    if s1 == 200:
                        resolved.append({"address": addr, "data": d1})
                    prog.progress((i + 1) / max(1, len(addrs)))
                data = {"fallback": resolved}

            ss.geocode_last = {"status": status, "data": data, "ts": time.time()}

            # Apply results back to stops if we can detect lat/lng
            df = parse_stops_df()
            updated = df.copy()
            addr_to_latlng = {}

            # attempt parse batch style
            if isinstance(data, dict):
                # common patterns
                results = data.get("results") or data.get("items") or data.get("data") or None
                if isinstance(results, list):
                    for r in results:
                        addr = r.get("address") or r.get("input") or r.get("query")
                        pos = r.get("position") or r.get("location") or r.get("pos")
                        if isinstance(pos, dict) and "lat" in pos and "lng" in pos and addr:
                            addr_to_latlng[addr] = (float(pos["lat"]), float(pos["lng"]))
                # fallback pattern
                if "fallback" in data and isinstance(data["fallback"], list):
                    for item in data["fallback"]:
                        addr = item.get("address")
                        d1 = item.get("data") or {}
                        items = d1.get("items") or d1.get("results") or []
                        if items and isinstance(items, list):
                            pos = items[0].get("position") or items[0].get("location") or {}
                            if isinstance(pos, dict) and "lat" in pos and "lng" in pos and addr:
                                addr_to_latlng[addr] = (float(pos["lat"]), float(pos["lng"]))

            for i in range(len(updated)):
                addr = updated.loc[i, "address"]
                if isinstance(addr, str) and addr in addr_to_latlng:
                    updated.loc[i, "lat"] = addr_to_latlng[addr][0]
                    updated.loc[i, "lng"] = addr_to_latlng[addr][1]
                    if not updated.loc[i, "source"]:
                        updated.loc[i, "source"] = "Geocoded"

            set_stops_from_df(updated)

    # Map (always re-renders from session_state; never disappears)
    df = parse_stops_df()
    valid = df.dropna(subset=["lat", "lng"])
    if not valid.empty:
        lat0 = float(valid.iloc[0]["lat"])
        lng0 = float(valid.iloc[0]["lng"])
        m = base_map(lat0, lng0, zoom=12)
        add_numbered_markers(m, st.session_state.stops, color="red")
        render_map(m, key="map_geocode", height=520)
    else:
        st.info("Add stops with lat/lng (or geocode addresses) to see pins on the map.")

    if ss.geocode_last:
        with st.expander("Download / View Geocode JSON", expanded=False):
            st.download_button(
                "⬇️ Download Geocode JSON",
                data=json.dumps(ss.geocode_last, indent=2),
                file_name="geocode_last.json",
                mime="application/json",
                use_container_width=True,
            )
            st.json(ss.geocode_last)

# =============================
# TAB 2 — PLACES (REGION SEARCH + GENERATE STOPS)
# =============================
with tabs[1]:
    st.subheader("Search region/city → set center → generate random stops OR search POIs → add as stops")

    # ----- Step A: Region search (universal) -----
    with st.form("region_search_form", clear_on_submit=False):
        region_q = st.text_input("Region/City/State/Country", value="Delhi")
        submit_region = st.form_submit_button("Search Region")

    region_results = []
    if submit_region:
        # Use forward geocode to get a region center
        status, data = http_get("/geocode", params={"q": region_q}, timeout=30)
        if status != 200:
            status, data = http_get("/places/v1/geocode", params={"q": region_q}, timeout=30)
        ss.places_last = {"region_search": {"status": status, "data": data, "q": region_q}, "ts": time.time()}

        items = data.get("items") or data.get("results") or []
        if isinstance(items, list):
            for it in items[:10]:
                title = it.get("title") or it.get("name") or it.get("label") or "Result"
                pos = it.get("position") or it.get("location") or {}
                if isinstance(pos, dict) and "lat" in pos and "lng" in pos:
                    region_results.append((title, float(pos["lat"]), float(pos["lng"])))

    if region_results:
        pick = st.selectbox("Pick a region result", [r[0] for r in region_results], key="region_pick")
        chosen = next(r for r in region_results if r[0] == pick)
        if st.button("Use picked region as center", use_container_width=True):
            ss.center["lat"], ss.center["lng"] = chosen[1], chosen[2]

    st.caption(f"Current center: {ss.center['lat']:.6f},{ss.center['lng']:.6f} | Country filter: {ss.center['country']}")

    # ----- Click-to-set-center map (shows clicked pin) -----
    st.markdown("### Optional: Click map to set center")
    m = base_map(ss.center["lat"], ss.center["lng"], zoom=12)
    # show center marker
    folium.Marker([ss.center["lat"], ss.center["lng"]], tooltip="Current center").add_to(m)

    if ss.last_clicked:
        folium.Marker([ss.last_clicked["lat"], ss.last_clicked["lng"]], tooltip="Clicked point", icon=folium.Icon(color="red")).add_to(m)

    out = render_map(m, key="map_places_center", height=420)
    if out and out.get("last_clicked"):
        ss.last_clicked = {"lat": out["last_clicked"]["lat"], "lng": out["last_clicked"]["lng"]}
        st.success(f"Clicked: {ss.last_clicked['lat']:.6f},{ss.last_clicked['lng']:.6f}")

        cA, cB = st.columns(2)
        with cA:
            if st.button("Use clicked point as center", use_container_width=True):
                ss.center["lat"], ss.center["lng"] = ss.last_clicked["lat"], ss.last_clicked["lng"]
        with cB:
            if st.button("Clear clicked point", use_container_width=True):
                ss.last_clicked = None

    # ----- Generate random stops around center (NO keyword required) -----
    st.markdown("### Generate random stops around current center (no keyword needed)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        n_random = st.number_input("How many stops?", min_value=2, max_value=60, value=20, step=1)
    with col2:
        radius_m = st.number_input("Radius (meters)", min_value=200, max_value=50000, value=4000, step=200)
    with col3:
        jitter = st.selectbox("Distribution", ["Uniform", "Clustered"], index=0)

    if st.button("🎲 Generate random stops", use_container_width=True):
        lat0, lng0 = ss.center["lat"], ss.center["lng"]
        new_stops = []
        for i in range(int(n_random)):
            # meters -> degrees approx
            r = random.random()
            if jitter == "Clustered":
                r = r * r
            ang = random.random() * 2 * math.pi
            dist = r * radius_m
            dlat = (dist * math.cos(ang)) / 111_320.0
            dlng = (dist * math.sin(ang)) / (111_320.0 * math.cos(math.radians(lat0)) + 1e-9)
            lat = lat0 + dlat
            lng = lng0 + dlng
            new_stops.append({
                "label": f"Rand {i+1}",
                "address": None,
                "lat": float(lat),
                "lng": float(lng),
                "source": f"Random within {radius_m}m",
            })
        add_stops(new_stops)

    # ----- Optional POI keyword search (anchored around center) -----
    st.markdown("### Optional: POI keyword search (Discover) and add results as stops")
    with st.form("poi_search_form", clear_on_submit=False):
        poi_q = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="petrol pump")
        poi_radius = st.slider("Search radius (m)", min_value=500, max_value=50000, value=5000, step=500)
        poi_limit = st.slider("Max results", min_value=5, max_value=50, value=10, step=1)
        submit_poi = st.form_submit_button("Search Places (POIs)")

    poi_items = []
    if submit_poi:
        # Many NB place search APIs return HERE-like schema (items with position)
        # We'll call Search Places APIs if available; if not, keep your previous /discover style.
        center = normalize_latlng_str(ss.center["lat"], ss.center["lng"])
        params = {
            "at": center,
            "q": poi_q,
            "limit": int(poi_limit),
            "radius": int(poi_radius),
            "in": f"countryCode:{ss.center['country']}",
        }

        # Try a couple common paths (NB deployments differ)
        status, data = http_get("/discover", params=params, timeout=30)
        if status == 404:
            status, data = http_get("/places/v1/discover", params=params, timeout=30)
        if status == 404:
            status, data = http_get("/places/v1/search", params=params, timeout=30)

        ss.places_last = {"poi_search": {"status": status, "data": data, "params": params}, "ts": time.time()}

        items = data.get("items") or data.get("results") or []
        if isinstance(items, list):
            for it in items:
                title = it.get("title") or it.get("name") or it.get("label") or ""
                addr = None
                if isinstance(it.get("address"), dict):
                    addr = it["address"].get("label") or it["address"].get("freeformAddress")
                addr = addr or title or None
                pos = it.get("position") or it.get("location") or {}
                if isinstance(pos, dict) and "lat" in pos and "lng" in pos:
                    poi_items.append({"title": title, "address": addr, "lat": float(pos["lat"]), "lng": float(pos["lng"])})

        st.success(f"Places response: HTTP {status}")

        if poi_items:
            # show + add
            dfpoi = pd.DataFrame(poi_items)
            st.dataframe(dfpoi, width="stretch", height=220)

            if st.button("➕ Add POI results as stops", use_container_width=True):
                new_stops = []
                base_n = len(ss.stops)
                for i, r in enumerate(poi_items, start=1):
                    new_stops.append({
                        "label": f"POI {base_n+i}",
                        "address": r.get("address"),
                        "lat": r.get("lat"),
                        "lng": r.get("lng"),
                        "source": f"POI: {poi_q}",
                    })
                add_stops(new_stops)
        else:
            st.warning("No POI items found (or response schema differed). Try a larger radius or different keyword.")

    # Map showing current stops (always visible)
    st.markdown("### Stops map (current)")
    df = parse_stops_df()
    valid = df.dropna(subset=["lat", "lng"])
    if not valid.empty:
        m2 = base_map(ss.center["lat"], ss.center["lng"], zoom=12)
        add_numbered_markers(m2, ss.stops, color="red")
        render_map(m2, key="map_places_stops", height=520)
    else:
        st.info("No valid stops with lat/lng yet.")

    if ss.places_last:
        with st.expander("Download / View Places JSON", expanded=False):
            st.download_button(
                "⬇️ Download Places JSON",
                data=json.dumps(ss.places_last, indent=2),
                file_name="places_last.json",
                mime="application/json",
                use_container_width=True,
            )
            st.json(ss.places_last)

# =============================
# TAB 3 — ROUTE + OPTIMIZE (BEFORE VS AFTER)
# =============================
with tabs[2]:
    st.subheader("Compute route (Before) → run optimization → recompute route (After) + compare")

    df = parse_stops_df()
    valid = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    if len(valid) < 2:
        st.warning("Need at least 2 stops with lat/lng to compute directions.")
    else:
        # Build waypoints list
        coords = [(float(r["lat"]), float(r["lng"])) for _, r in valid.iterrows()]

        # Avoid slider crash: only show if meaningful
        maxN = max(2, len(coords))
        minN = 2
        if maxN <= minN:
            n_for_path = 2
            st.caption("N stops for path: 2 (only two coordinates available)")
        else:
            n_for_path = st.slider("N stops for path (if using first N)", min_value=2, max_value=maxN, value=min(10, maxN))

        coords_use = coords[:n_for_path]

        def directions_call(coords_list: List[Tuple[float, float]]) -> Tuple[int, Dict[str, Any]]:
            # Many Directions APIs accept origin/destination + waypoints or "locations" list.
            # We'll try a flexible approach:
            origin = normalize_latlng_str(coords_list[0][0], coords_list[0][1])
            destination = normalize_latlng_str(coords_list[-1][0], coords_list[-1][1])
            waypoints = [normalize_latlng_str(a, b) for (a, b) in coords_list[1:-1]]

            params = {
                "origin": origin,
                "destination": destination,
                "mode": ss.global_opts["mode"],
                "units": ss.global_opts["units"],
                "alternatives": "true" if ss.global_opts["alternatives"] else "false",
                "geometry": "polyline",  # request route geometry
            }

            if ss.global_opts["avoid"]:
                params["avoid"] = ",".join(ss.global_opts["avoid"])
            if ss.global_opts["traffic"]:
                params["traffic"] = "true"
            if ss.global_opts["departure_time"]:
                params["departure_time"] = int(ss.global_opts["departure_time"])

            if waypoints:
                params["waypoints"] = "|".join(waypoints)

            status, data = http_get("/directions", params=params, timeout=60)
            if status == 404:
                status, data = http_get("/directions/v2", params=params, timeout=60)
            return status, data

        # Step 1 — Directions Before
        colL, colR = st.columns([1, 1])
        with colL:
            if st.button("🧭 Compute route (Before)", use_container_width=True):
                status, data = directions_call(coords_use)
                ss.directions_before = {"status": status, "data": data, "ts": time.time(), "coords_used": coords_use}
        with colR:
            if ss.directions_before:
                st.caption(f"Directions (Before): HTTP {ss.directions_before['status']}")

        # Render Before map + metrics
        if ss.directions_before and ss.directions_before["status"] == 200:
            before_coords = extract_route_coords(ss.directions_before["data"])
            before_km, before_min = extract_total_distance_duration(ss.directions_before["data"])

            left, right = st.columns([1, 1])
            with left:
                st.markdown("### Route map (Before)")
                mB = base_map(coords_use[0][0], coords_use[0][1], zoom=12)
                add_numbered_markers(mB, [{"lat": a, "lng": b, "label": f"{i+1}"} for i, (a, b) in enumerate(coords_use)], color="red")
                add_route_polyline_with_arrows(mB, before_coords)
                render_map(mB, key="map_route_before", height=520)

                if before_km is not None:
                    st.metric("Before distance (km)", f"{before_km:.2f}")
                if before_min is not None:
                    st.metric("Before duration (min)", f"{before_min:.1f}")

            with right:
                st.markdown("### Step 2 — Optimization (VRP v2)")
                st.caption("Uses VRP v2 payload with **locations.location** present + objective object.")

                objective = st.selectbox("Optimization objective (travel_cost)", ["distance", "duration"], index=0)

                if st.button("⚙️ Run optimization (VRP v2)", use_container_width=True):
                    # Build VRP v2 payload:
                    # - locations: { id, location: ["lat,lng", ...] }  ✅ fixes locations.location missing
                    # - jobs reference location_index
                    # - options.objective must be an object (not a string) ✅ fixes dto.ObjectiveOption error
                    loc_list = [normalize_latlng_str(a, b) for (a, b) in coords_use]
                    body = {
                        "locations": {"id": 1, "location": loc_list},
                        "vehicles": [{
                            "id": 1,
                            "start_location_index": 0,
                            "end_location_index": 0
                        }],
                        "jobs": [{"id": i + 1, "location_index": i} for i in range(1, len(loc_list))],
                        "options": {
                            "objective": {"travel_cost": objective},
                        }
                    }
                    status, data = http_post("/optimization/v2", body, timeout=60)
                    ss.vrp_last = {"status": status, "data": data, "ts": time.time(), "request": body}

                    if status != 200:
                        st.error(f"VRP create failed: HTTP {status}")
                        st.json(data)
                    else:
                        st.success("VRP job created.")

                # Show VRP JSON
                if ss.vrp_last:
                    st.download_button(
                        "⬇️ Download VRP create JSON",
                        data=json.dumps(ss.vrp_last, indent=2),
                        file_name="vrp_create_last.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                st.markdown("### Step 3 — Recompute Directions (After)")
                if ss.vrp_last and ss.vrp_last.get("status") == 200:
                    # Try to extract optimized order.
                    # Different schemas exist; we handle multiple.
                    data = ss.vrp_last.get("data") or {}
                    order_indices = None

                    # candidate paths
                    # e.g., routes[0].steps[*].location_index, or solution.routes[0].jobs, etc.
                    if isinstance(data, dict):
                        # 1) solution.routes[0].steps
                        sol = data.get("solution") or data.get("result") or data
                        routes = sol.get("routes") if isinstance(sol, dict) else None
                        if isinstance(routes, list) and routes:
                            r0 = routes[0]
                            steps = r0.get("steps") or r0.get("activities") or []
                            idxs = []
                            for s in steps:
                                if "location_index" in s:
                                    idxs.append(int(s["location_index"]))
                            if idxs:
                                order_indices = idxs

                        # 2) route with "jobs" list of indices
                        if order_indices is None:
                            routes = data.get("routes")
                            if isinstance(routes, list) and routes:
                                r0 = routes[0]
                                jobs = r0.get("jobs") or []
                                idxs = []
                                for j in jobs:
                                    if "location_index" in j:
                                        idxs.append(int(j["location_index"]))
                                if idxs:
                                    order_indices = [0] + idxs + [0]

                    if order_indices:
                        # Build optimized coords (ensure start at 0)
                        opt_coords = []
                        for idx in order_indices:
                            if 0 <= idx < len(coords_use):
                                opt_coords.append(coords_use[idx])
                        # if still too short, fallback to original
                        if len(opt_coords) < 2:
                            opt_coords = coords_use

                        if st.button("🧭 Recompute route (After)", use_container_width=True):
                            status2, data2 = directions_call(opt_coords)
                            ss.directions_after = {"status": status2, "data": data2, "ts": time.time(), "coords_used": opt_coords}

                    else:
                        st.info("Optimization succeeded, but route order not found in response schema. (We can still show raw JSON below.)")

                # Render After map + compare
                if ss.directions_after and ss.directions_after["status"] == 200 and ss.directions_before and ss.directions_before["status"] == 200:
                    after_coords = extract_route_coords(ss.directions_after["data"])
                    after_km, after_min = extract_total_distance_duration(ss.directions_after["data"])
                    before_km, before_min = extract_total_distance_duration(ss.directions_before["data"])

                    st.markdown("### Optimized route map (After)")
                    cu = ss.directions_after.get("coords_used") or coords_use

                    mA = base_map(cu[0][0], cu[0][1], zoom=12)
                    add_numbered_markers(mA, [{"lat": a, "lng": b, "label": f"{i+1}"} for i, (a, b) in enumerate(cu)], color="red")
                    add_route_polyline_with_arrows(mA, after_coords)
                    render_map(mA, key="map_route_after", height=520)

                    # Compare metrics
                    st.markdown("### Comparison")
                    c1, c2, c3 = st.columns(3)

                    def fmt_delta(before, after):
                        if before is None or after is None:
                            return None, None
                        delta = after - before
                        pct = (delta / before * 100.0) if before else None
                        return delta, pct

                    if before_km is not None and after_km is not None:
                        d, p = fmt_delta(before_km, after_km)
                        c1.metric("Distance (km) After", f"{after_km:.2f}", delta=f"{d:+.2f} km" if d is not None else None)
                        if p is not None:
                            c1.caption(f"% change: {p:+.1f}%")
                    if before_min is not None and after_min is not None:
                        d, p = fmt_delta(before_min, after_min)
                        c2.metric("Duration (min) After", f"{after_min:.1f}", delta=f"{d:+.1f} min" if d is not None else None)
                        if p is not None:
                            c2.caption(f"% change: {p:+.1f}%")

                    if (before_km is not None and after_km is not None) and (before_min is not None and after_min is not None):
                        saved_km = before_km - after_km
                        saved_min = before_min - after_min
                        c3.metric("Saved (km)", f"{saved_km:.2f}")
                        c3.metric("Saved (min)", f"{saved_min:.1f}")

        # Downloads
        st.divider()
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            if ss.directions_before:
                st.download_button(
                    "⬇️ Download Directions (Before)",
                    data=json.dumps(ss.directions_before, indent=2),
                    file_name="directions_before.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with dl2:
            if ss.vrp_last:
                st.download_button(
                    "⬇️ Download VRP (Create/Result)",
                    data=json.dumps(ss.vrp_last, indent=2),
                    file_name="vrp_last.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with dl3:
            if ss.directions_after:
                st.download_button(
                    "⬇️ Download Directions (After)",
                    data=json.dumps(ss.directions_after, indent=2),
                    file_name="directions_after.json",
                    mime="application/json",
                    use_container_width=True,
                )

# =============================
# TAB 4 — DISTANCE MATRIX (NxN)
# =============================
with tabs[3]:
    st.subheader("Distance Matrix (NxN) — supports 20+ points")

    df = parse_stops_df()
    valid = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    if len(valid) < 2:
        st.warning("Need at least 2 stops with lat/lng.")
    else:
        max_points = min(60, len(valid))
        n = st.number_input("How many stops to include", min_value=2, max_value=max_points, value=min(20, max_points))

        if st.button("📏 Compute Distance Matrix", use_container_width=True):
            subset = valid.iloc[:n]
            locs = [normalize_latlng_str(float(r["lat"]), float(r["lng"])) for _, r in subset.iterrows()]
            # common matrix API uses origins & destinations
            params = {
                "origins": "|".join(locs),
                "destinations": "|".join(locs),
                "mode": ss.global_opts["mode"],
                "units": ss.global_opts["units"],
            }
            if ss.global_opts["traffic"]:
                params["traffic"] = "true"
            if ss.global_opts["departure_time"]:
                params["departure_time"] = int(ss.global_opts["departure_time"])
            if ss.global_opts["avoid"]:
                params["avoid"] = ",".join(ss.global_opts["avoid"])

            status, data = http_get("/distancematrix", params=params, timeout=60)
            if status == 404:
                status, data = http_get("/distance_matrix", params=params, timeout=60)

            ss.dm_last = {"status": status, "data": data, "ts": time.time(), "params": params}

        if ss.dm_last:
            st.caption(f"Distance Matrix: HTTP {ss.dm_last['status']}")
            st.download_button(
                "⬇️ Download Distance Matrix JSON",
                data=json.dumps(ss.dm_last, indent=2),
                file_name="distance_matrix_last.json",
                mime="application/json",
                use_container_width=True,
            )
            # Try render a table if the schema matches common patterns
            data = ss.dm_last.get("data") or {}
            rows = data.get("rows")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "elements" in rows[0]:
                mat = []
                for r in rows:
                    elems = r.get("elements") or []
                    mat.append([e.get("distance", {}).get("value") for e in elems])
                dfm = pd.DataFrame(mat)
                st.dataframe(dfm, width="stretch", height=320)
            else:
                st.json(data)

# =============================
# TAB 5 — SNAP TO ROAD + ISOCHRONE
# =============================
with tabs[4]:
    st.subheader("Snap-to-Road + Isochrone (stable outputs)")

    df = parse_stops_df()
    valid = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    st.markdown("### Snap-to-Road")
    st.caption("Pick a path from your current stops; output stays visible after button click.")

    if len(valid) < 2:
        st.warning("Need at least 2 stops with lat/lng.")
    else:
        max_points = min(50, len(valid))
        n_path = st.number_input("N stops for path", min_value=2, max_value=max_points, value=min(10, max_points))
        subset = valid.iloc[:n_path]
        path = [normalize_latlng_str(float(r["lat"]), float(r["lng"])) for _, r in subset.iterrows()]

        if st.button("🧷 Snap path to road", use_container_width=True):
            # Many snap-to-road APIs accept path as "path=lat,lng|lat,lng"
            params = {"path": "|".join(path)}
            status, data = http_get("/snapToRoad", params=params, timeout=60)
            if status == 404:
                status, data = http_get("/snap-to-road", params=params, timeout=60)
            ss.snap_last = {"status": status, "data": data, "ts": time.time(), "params": params}

    if ss.snap_last:
        st.caption(f"Snap-to-Road: HTTP {ss.snap_last['status']}")
        st.download_button(
            "⬇️ Download Snap-to-Road JSON",
            data=json.dumps(ss.snap_last, indent=2),
            file_name="snap_to_road_last.json",
            mime="application/json",
            use_container_width=True,
        )

        # Render snapped polyline if available
        d = ss.snap_last.get("data") or {}
        snapped_coords = []
        if isinstance(d.get("geometry"), str):
            snapped_coords = decode_polyline(d["geometry"], precision=5)
        elif isinstance(d.get("snappedPoints"), list):
            for p in d["snappedPoints"]:
                loc = p.get("location") or p.get("position") or {}
                if "latitude" in loc and "longitude" in loc:
                    snapped_coords.append((float(loc["latitude"]), float(loc["longitude"])))
                elif "lat" in loc and "lng" in loc:
                    snapped_coords.append((float(loc["lat"]), float(loc["lng"])))

        if snapped_coords:
            mS = base_map(snapped_coords[0][0], snapped_coords[0][1], zoom=13)
            add_route_polyline_with_arrows(mS, snapped_coords)
            render_map(mS, key="map_snap", height=420)
        else:
            st.json(d)

    st.divider()
    st.markdown("### Isochrone")
    st.caption("Compute reachable area from the current center; output stays visible after button click.")

    iso_col1, iso_col2, iso_col3 = st.columns([1, 1, 1])
    with iso_col1:
        iso_minutes = st.selectbox("Range (minutes)", [5, 10, 15, 20, 30, 45, 60], index=2)
    with iso_col2:
        iso_type = st.selectbox("Type", ["time", "distance"], index=0)
    with iso_col3:
        iso_mode = st.selectbox("Mode", ["car", "truck", "bike", "walk"], index=0)

    if st.button("🟦 Compute Isochrone", use_container_width=True):
        center = normalize_latlng_str(ss.center["lat"], ss.center["lng"])
        params = {
            "location": center,
            "mode": iso_mode,
        }
        if iso_type == "time":
            params["range"] = iso_minutes * 60
        else:
            params["range"] = iso_minutes * 1000

        status, data = http_get("/isochrone", params=params, timeout=60)
        if status == 404:
            status, data = http_get("/isoline", params=params, timeout=60)
        ss.iso_last = {"status": status, "data": data, "ts": time.time(), "params": params}

    if ss.iso_last:
        st.caption(f"Isochrone: HTTP {ss.iso_last['status']}")
        st.download_button(
            "⬇️ Download Isochrone JSON",
            data=json.dumps(ss.iso_last, indent=2),
            file_name="isochrone_last.json",
            mime="application/json",
            use_container_width=True,
        )
        d = ss.iso_last.get("data") or {}
        # If GeoJSON present, draw it
        gj = d.get("geojson") or d.get("data") or None
        if isinstance(gj, dict) and gj.get("type") in ("Feature", "FeatureCollection"):
            mI = base_map(ss.center["lat"], ss.center["lng"], zoom=12)
            folium.GeoJson(gj).add_to(mI)
            render_map(mI, key="map_iso", height=420)
        else:
            st.json(d)

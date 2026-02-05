# app.py
# NextBillion.ai — Visual API Tester (Stops + Pin + Stable Maps)
# Focus: restore location-based stop generator + pin-click generation + stable map rendering

import os
import json
import math
import random
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import streamlit as st
import folium
from folium.features import DivIcon

try:
    from streamlit_folium import st_folium
except Exception:
    st.error("Missing dependency: streamlit-folium. Install with: pip install streamlit-folium")
    raise

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="NextBillion.ai — Visual API Tester", layout="wide")

APP_VERSION = "v3.2 (Stops+Pin restored, stable maps, safe keys)"
DEFAULT_CENTER = {"lat": 28.6139, "lng": 77.2090, "label": "New Delhi", "country": "IND"}  # 3-letter

# -----------------------------
# Helpers: safe unique keys
# -----------------------------
def k(prefix: str) -> str:
    # stable unique keys across reruns
    return f"{prefix}__{APP_VERSION}"

# -----------------------------
# API key handling (NO hardcoding)
# -----------------------------
def get_api_key() -> str:
    # priority: st.secrets, env var, or sidebar input
    if "NEXTBILLION_API_KEY" in st.secrets:
        return st.secrets["NEXTBILLION_API_KEY"]
    env_key = os.getenv("NEXTBILLION_API_KEY", "").strip()
    if env_key:
        return env_key
    return st.session_state.get("api_key_input", "").strip()

# -----------------------------
# HTTP helpers
# -----------------------------
NB_BASE = "https://api.nextbillion.io"

def nb_get(path: str, params: dict, timeout=60):
    url = f"{NB_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    return r.status_code, safe_json(r)

def nb_post(path: str, params: dict, body: dict, timeout=120):
    url = f"{NB_BASE}{path}"
    r = requests.post(url, params=params, json=body, timeout=timeout)
    return r.status_code, safe_json(r)

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}

def download_button(label: str, obj: dict, filename: str, key: str):
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(label, data=data, file_name=filename, mime="application/json", key=key)

# -----------------------------
# Session state init
# -----------------------------
ss = st.session_state
if "center" not in ss:
    ss.center = dict(DEFAULT_CENTER)

if "stops" not in ss:
    # stops = list of {"label","address","lat","lng","source"}
    ss.stops = []

if "pin" not in ss:
    ss.pin = None  # {"lat","lng"}

if "places_last" not in ss:
    ss.places_last = None

if "debug" not in ss:
    ss.debug = False

# Cache buckets for reducing API calls
if "cache_geocode" not in ss:
    ss.cache_geocode = {}
if "cache_region" not in ss:
    ss.cache_region = {}
if "cache_places" not in ss:
    ss.cache_places = {}
if "cache_directions" not in ss:
    ss.cache_directions = {}

# -----------------------------
# Map builders (stable rendering)
# -----------------------------
def numbered_marker(lat, lng, n: int, tooltip: str = ""):
    # Red numbered marker using DivIcon (easy + no extra plugins)
    html = f"""
    <div style="
      background:#e11d48;
      color:white;
      border-radius:9999px;
      width:30px;height:30px;
      display:flex;align-items:center;justify-content:center;
      font-weight:700;
      border:2px solid white;
      box-shadow:0 2px 6px rgba(0,0,0,0.25);
    ">{n}</div>
    """
    return folium.Marker(
        location=[lat, lng],
        tooltip=tooltip,
        icon=DivIcon(icon_size=(30, 30), icon_anchor=(15, 15), html=html),
    )

def build_map(center_lat, center_lng, stops=None, pin=None, zoom=12):
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, control_scale=True)

    # show pin if set
    if pin:
        folium.Marker(
            location=[pin["lat"], pin["lng"]],
            tooltip="📍 Clicked pin center",
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(m)
        folium.CircleMarker(
            location=[pin["lat"], pin["lng"]],
            radius=6,
            weight=2,
            color="#ef4444",
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    # stops with numbers
    if stops:
        for i, s in enumerate(stops, start=1):
            tip = f"{i}. {s.get('label','Stop')} | {s.get('address','')}"
            numbered_marker(s["lat"], s["lng"], i, tooltip=tip).add_to(m)

    return m

# -----------------------------
# Stop utilities
# -----------------------------
def normalize_latlng(lat, lng):
    return float(lat), float(lng)

def random_points_around(lat, lng, radius_m, n):
    # uniform-ish scatter within circle (meters)
    pts = []
    for _ in range(n):
        r = radius_m * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)
        # meters to degrees
        dlat = dy / 111_320.0
        dlng = dx / (111_320.0 * math.cos(math.radians(lat)) + 1e-9)
        pts.append((lat + dlat, lng + dlng))
    return pts

def set_center(lat, lng, label=None, country=None):
    ss.center["lat"] = float(lat)
    ss.center["lng"] = float(lng)
    if label:
        ss.center["label"] = label
    if country:
        ss.center["country"] = country

def clear_stops():
    ss.stops = []

def append_stop(lat, lng, label, address="", source="generated"):
    ss.stops.append(
        {"label": label, "address": address, "lat": float(lat), "lng": float(lng), "source": source}
    )

def stops_df():
    if not ss.stops:
        return pd.DataFrame(columns=["#", "label", "address", "lat", "lng", "source"])
    rows = []
    for i, s in enumerate(ss.stops, start=1):
        rows.append(
            {"#": i, "label": s.get("label",""), "address": s.get("address",""), "lat": s["lat"], "lng": s["lng"], "source": s.get("source","")}
        )
    return pd.DataFrame(rows)

# -----------------------------
# REGION SEARCH (use Geocode endpoint as "universal" region finder)
# Note: If the API you have doesn’t support an explicit "region search" endpoint,
# the most reliable universal option is geocoding a query string and using its lat/lng.
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def region_search_cached(api_key: str, query: str, country: str = ""):
    # NextBillion Geocode API: /geocode?key=...&q=...
    # Many implementations support "country" or "countryCode". If yours differs, keep only q + key.
    params = {"key": api_key, "q": query}
    if country:
        params["country"] = country  # if unsupported by your key/account, harmless to remove
    return nb_get("/geocode", params=params, timeout=60)

def parse_region_candidates(data: dict):
    # best-effort parsing
    candidates = []
    for key in ["items", "results", "features"]:
        if key in data and isinstance(data[key], list):
            for it in data[key]:
                lat = None
                lng = None
                label = it.get("title") or it.get("name") or it.get("formatted_address") or it.get("address") or ""
                # common coordinate patterns
                if "position" in it and isinstance(it["position"], dict):
                    lat = it["position"].get("lat")
                    lng = it["position"].get("lng")
                if (lat is None or lng is None) and "location" in it and isinstance(it["location"], dict):
                    lat = it["location"].get("lat")
                    lng = it["location"].get("lng")
                if (lat is None or lng is None) and "geometry" in it and isinstance(it["geometry"], dict):
                    coords = it["geometry"].get("coordinates")
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        lng, lat = coords[0], coords[1]
                if lat is not None and lng is not None:
                    candidates.append({"label": label, "lat": float(lat), "lng": float(lng)})
            break
    return candidates

# -----------------------------
# PLACES/POI SEARCH (optional keyword)
# If your account/plan has a specific Places endpoint (e.g., /discover),
# keep the path below aligned to your working JSON.
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def places_search_cached(api_key: str, lat: float, lng: float, keyword: str, radius_m: int, max_results: int, country: str = ""):
    # Many NextBillion implementations use /discover for POI search.
    # If yours uses a different path, change ONLY the path here.
    params = {
        "key": api_key,
        "at": f"{lat},{lng}",
        "q": keyword if keyword else "",
        "limit": max_results,
        "radius": radius_m,
    }
    if country:
        params["country"] = country
    return nb_get("/discover", params=params, timeout=60)

def parse_places_items(data: dict):
    items = []
    for key in ["items", "results"]:
        if key in data and isinstance(data[key], list):
            for it in data[key]:
                name = it.get("title") or it.get("name") or ""
                addr = it.get("address", {}).get("label") if isinstance(it.get("address"), dict) else it.get("address", "")
                lat = None
                lng = None
                if "position" in it and isinstance(it["position"], dict):
                    lat = it["position"].get("lat")
                    lng = it["position"].get("lng")
                if (lat is None or lng is None) and "location" in it and isinstance(it["location"], dict):
                    lat = it["location"].get("lat")
                    lng = it["location"].get("lng")
                if lat is not None and lng is not None:
                    items.append({"label": name or "POI", "address": addr or "", "lat": float(lat), "lng": float(lng)})
            break
    return items

# -----------------------------
# SIDEBAR: Controls
# -----------------------------
st.sidebar.markdown("## Config")
api_key = get_api_key()

if not api_key:
    st.sidebar.text_input("NextBillion API Key", type="password", key="api_key_input")
    api_key = get_api_key()

st.sidebar.caption(f"App build: {APP_VERSION}")

st.sidebar.markdown("---")
st.sidebar.markdown("## Stops (20+)")
st.sidebar.write(f"Stops loaded: **{len(ss.stops)}**")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("➕ Add / Replace", key=k("add_replace_btn")):
        # no-op; UI below does the real add/replace
        pass
with c2:
    if st.button("🧹 Clear stops", key=k("clear_stops_btn")):
        clear_stops()

ss.debug = st.sidebar.checkbox("Debug", value=ss.debug, key=k("debug_chk"))

st.sidebar.markdown("---")
st.sidebar.markdown("## Current center")
st.sidebar.write(f"**{ss.center.get('label','Center')}**")
st.sidebar.write(f"{ss.center['lat']:.5f}, {ss.center['lng']:.5f}  |  Country: {ss.center.get('country','')}")
st.sidebar.caption("Tip: click a pin on map to set center, then generate stops around it.")

# -----------------------------
# MAIN
# -----------------------------
st.title("NextBillion.ai — Visual API Tester")
st.caption("Stops → Places/Generate → (later) Directions/Matrix/Optimize. This build restores location-based stop generation + pin-click center.")
st.markdown(f"**Stops loaded:** {len(ss.stops)}")

tabs = st.tabs([
    "Geocode & Map",
    "Places (Search + Generate Stops)",
    "Directions (Multi-stop) — placeholder",
    "Distance Matrix (NxN) — placeholder",
    "Optimization (VRP v2) — placeholder",
    "Snap-to-Road + Isochrone — placeholder",
])

# =========================================================
# TAB 1: Geocode & Map (simple address/latlng input)
# =========================================================
with tabs[0]:
    st.subheader("Geocode your stops and show them on the map")
    st.caption("This tab focuses on stable map rendering + stop table. Places tab is the main stop generator.")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Stops table")
        df = stops_df()
        st.dataframe(df, width="stretch", height=320, key=k("stops_table_geocode"))

        st.markdown("#### Paste stops (one per line)")
        input_mode = st.radio(
            "Input type",
            ["Addresses (one per line)", "Lat/Lng (one per line)"],
            index=0,
            key=k("geocode_input_mode"),
        )
        raw = st.text_area(
            "Paste here",
            height=160,
            placeholder="e.g.\nConnaught Place, New Delhi\n28.6139,77.2090",
            key=k("geocode_paste"),
        )

        add_replace = st.selectbox("When I apply these:", ["Add to existing", "Replace existing"], index=0, key=k("geocode_add_replace"))
        apply_btn = st.button("Apply pasted stops", key=k("apply_paste_stops"), width="stretch")

        if apply_btn:
            lines = [x.strip() for x in raw.splitlines() if x.strip()]
            if not lines:
                st.warning("Paste at least 1 line.")
            else:
                if add_replace == "Replace existing":
                    clear_stops()
                if input_mode.startswith("Lat/Lng"):
                    ok = 0
                    for i, line in enumerate(lines, start=1):
                        parts = [p.strip() for p in line.replace(";", ",").split(",")]
                        if len(parts) >= 2:
                            try:
                                lat = float(parts[0]); lng = float(parts[1])
                                append_stop(lat, lng, label=f"Stop {len(ss.stops)+1}", address="", source="latlng paste")
                                ok += 1
                            except:
                                pass
                    st.success(f"Added {ok} stops from lat/lng.")
                else:
                    # In a production build you'd call geocode per line (costly).
                    # Here we store as address-only until you geocode with your own flow.
                    for line in lines:
                        append_stop(ss.center["lat"], ss.center["lng"], label=f"Stop {len(ss.stops)+1}", address=line, source="address paste (not geocoded)")
                    st.info("Added address-only stops at current center (no geocode call in this lightweight build).")

    with colB:
        st.markdown("### Map")
        m = build_map(ss.center["lat"], ss.center["lng"], stops=ss.stops, pin=ss.pin, zoom=12)

        # IMPORTANT: stable key so map does not disappear
        map_ret = st_folium(
            m,
            height=520,
            width="stretch",
            key=k("map_geocode"),
            returned_objects=["last_clicked"],
        )

        # pin click handler (works across app)
        if map_ret and map_ret.get("last_clicked"):
            lc = map_ret["last_clicked"]
            ss.pin = {"lat": lc["lat"], "lng": lc["lng"]}
            st.success(f"Pin selected: {ss.pin['lat']:.5f}, {ss.pin['lng']:.5f}")

        if ss.pin:
            if st.button("Use pin as center", key=k("use_pin_center_btn"), width="stretch"):
                set_center(ss.pin["lat"], ss.pin["lng"], label="Pinned center")

# =========================================================
# TAB 2: PLACES + GENERATE STOPS (RESTORED)
# =========================================================
with tabs[1]:
    st.subheader("Search region/city → set center → generate 20+ stops (no keyword required)")
    st.caption("This restores your missing workflow: location search + pin center + random stops generator + optional POI search.")

    # -----------------------------
    # Step 1: Region search (button-run only)
    # -----------------------------
    st.markdown("### Step 1 — Search for a region/city/state/country (universal)")
    with st.form(key=k("region_search_form"), clear_on_submit=False):
        q = st.text_input("Region/City/State/Country", value=ss.center.get("label", ""), key=k("region_q"))
        country = st.text_input("Country filter (3-letter, optional)", value=ss.center.get("country",""), key=k("region_country"))
        submitted = st.form_submit_button("Search region", use_container_width=True)

    candidates = []
    region_raw = None
    if submitted:
        if not api_key:
            st.error("Add API key first.")
        else:
            status, data = region_search_cached(api_key, q, country=country)
            region_raw = {"http_status": status, "data": data}
            candidates = parse_region_candidates(data)
            ss.cache_region[(q, country)] = region_raw

    # show last cached if available
    if not candidates and (q, country) in ss.cache_region:
        region_raw = ss.cache_region[(q, country)]
        candidates = parse_region_candidates(region_raw["data"])

    if candidates:
        labels = [c["label"] or f"{c['lat']},{c['lng']}" for c in candidates[:10]]
        pick = st.selectbox("Pick a region result", labels, key=k("pick_region_sel"))
        picked = next(c for c in candidates[:10] if (c["label"] or f"{c['lat']},{c['lng']}") == pick)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Use picked region as center", key=k("use_picked_center"), width="stretch"):
                set_center(picked["lat"], picked["lng"], label=pick, country=country or ss.center.get("country",""))
                st.success(f"Center set to: {pick}")
        with col2:
            st.write(f"**Center preview:** {picked['lat']:.5f}, {picked['lng']:.5f}")

    if region_raw:
        with st.expander("Region search debug"):
            st.write(f"HTTP {region_raw['http_status']}")
            download_button("Download region JSON", region_raw, "region_search.json", key=k("dl_region_json"))

    st.markdown("---")

    # -----------------------------
    # Step 2: Click-pin center + generator
    # -----------------------------
    st.markdown("### Step 2 — Click on map to set a pin center (pin is visible)")

    left, right = st.columns([1.1, 0.9])
    with left:
        m2 = build_map(ss.center["lat"], ss.center["lng"], stops=ss.stops, pin=ss.pin, zoom=12)
        ret2 = st_folium(
            m2,
            height=520,
            width="stretch",
            key=k("map_places"),
            returned_objects=["last_clicked"],
        )
        if ret2 and ret2.get("last_clicked"):
            lc = ret2["last_clicked"]
            ss.pin = {"lat": lc["lat"], "lng": lc["lng"]}
            st.success(f"Pin selected: {ss.pin['lat']:.5f}, {ss.pin['lng']:.5f}")

    with right:
        st.markdown("#### Center controls")
        st.write(f"Current center: **{ss.center.get('label','')}**")
        st.write(f"{ss.center['lat']:.5f}, {ss.center['lng']:.5f} | Country: {ss.center.get('country','')}")

        if ss.pin:
            st.write(f"Pin: {ss.pin['lat']:.5f}, {ss.pin['lng']:.5f}")
            if st.button("Use pin as center", key=k("places_use_pin_center"), width="stretch"):
                set_center(ss.pin["lat"], ss.pin["lng"], label="Pinned center")
                st.success("Center updated from pin.")

        st.markdown("#### Generate random stops (NO keyword needed)")
        gen_n = st.number_input("How many stops?", min_value=2, max_value=200, value=20, step=1, key=k("gen_n"))
        gen_radius = st.slider("Radius (meters)", min_value=200, max_value=30000, value=5000, step=100, key=k("gen_radius"))
        gen_mode = st.selectbox("Add/Replace", ["Add to existing", "Replace existing"], index=0, key=k("gen_add_replace"))

        if st.button("🎲 Generate random stops around center", key=k("gen_random_btn"), width="stretch"):
            if gen_mode == "Replace existing":
                clear_stops()

            base_lat, base_lng = ss.center["lat"], ss.center["lng"]
            pts = random_points_around(base_lat, base_lng, int(gen_radius), int(gen_n))
            for (lat, lng) in pts:
                append_stop(lat, lng, label=f"Stop {len(ss.stops)+1}", address="", source="random generator")
            st.success(f"Generated {gen_n} stops around center.")

    st.markdown("---")

    # -----------------------------
    # Step 3: Optional POI search (keyword)
    # -----------------------------
    st.markdown("### Optional — POI keyword search (Discover) and add results as stops")

    with st.form(key=k("poi_form"), clear_on_submit=False):
        keyword = st.text_input("POI keyword (e.g., petrol, hospital, warehouse)", value="petrol pump", key=k("poi_kw"))
        radius_poi = st.slider("Search radius (m)", 200, 30000, 5000, 100, key=k("poi_radius"))
        max_results = st.slider("Max results", 1, 50, 10, 1, key=k("poi_max"))
        add_mode = st.selectbox("Add results as stops", ["Add to existing", "Replace existing"], index=0, key=k("poi_add_mode"))
        go_poi = st.form_submit_button("Search Places (POIs)", use_container_width=True)

    poi_raw = None
    poi_items = []
    if go_poi:
        if not api_key:
            st.error("Add API key first.")
        else:
            lat0, lng0 = ss.center["lat"], ss.center["lng"]
            status, data = places_search_cached(api_key, lat0, lng0, keyword, int(radius_poi), int(max_results), country=ss.center.get("country",""))
            poi_raw = {"http_status": status, "data": data}
            poi_items = parse_places_items(data)
            ss.places_last = poi_raw

            st.success(f"Places response: HTTP {status}")
            if poi_items:
                if add_mode == "Replace existing":
                    clear_stops()
                for it in poi_items:
                    append_stop(it["lat"], it["lng"], label=it["label"], address=it["address"], source="places/poi")
                st.success(f"Added {len(poi_items)} POI stops.")
            else:
                st.warning("No POIs parsed from response (schema may differ for your endpoint).")

    if ss.places_last:
        download_button("Download Places JSON", ss.places_last, "places.json", key=k("dl_places"))

    st.markdown("### Stops table (live)")
    st.dataframe(stops_df(), width="stretch", height=260, key=k("stops_table_places"))

# =========================================================
# Remaining tabs placeholders
# (kept minimal; your current request was restoring stop generator + pin action)
# =========================================================
with tabs[2]:
    st.subheader("Directions (Multi-stop) — placeholder")
    st.info("You can now generate 20+ stops reliably. Next step is wiring multi-stop directions + route polyline on map.")

with tabs[3]:
    st.subheader("Distance Matrix (NxN) — placeholder")
    st.info("Next step: NxN matrix using all stops. The stop engine is now stable.")

with tabs[4]:
    st.subheader("Optimization (VRP v2) — placeholder")
    st.info("Next step: build VRP payload using locations[] entries that include 'location' field to avoid: locations.location missing.")

    st.markdown(
        "Your error `Mandatory parameter 'locations.location' is missing` means each item in `locations` must include a `location` field. "
        "The Route Optimization API docs show `locations` is required and includes coordinates + id. "
        "See docs: the tutorials explicitly define `locations` object before jobs/vehicles. "
    )
    st.caption("Reference: NextBillion Route Optimization API tutorials (locations object).")

with tabs[5]:
    st.subheader("Snap-to-Road + Isochrone — placeholder")
    st.info("Stop + map stability is now fixed. Next: wire snap-to-road + isochrone calls with persistent map output.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("If your POI endpoint path is not `/discover` in your account, change only `places_search_cached()` path.")

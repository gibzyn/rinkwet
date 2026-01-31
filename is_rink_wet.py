# is_rink_wet.py
import math
import json
import time
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# Google Sheets feedback + notes
import gspread
from google.oauth2.service_account import Credentials


# ----------------------------
# Timezone (force rink local time)
# ----------------------------
RINK_TZ = ZoneInfo("America/Los_Angeles")  # PST/PDT automatically


# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="RinkWet", page_icon="🏒", layout="centered")


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_LABEL = "Freedom Park Inline Hockey Arena (Camarillo)"
DEFAULT_LAT = 34.2138
DEFAULT_LON = -119.0856

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"

MAX_FORECAST_DAYS = 16
APP_URL = "https://rinkwet.streamlit.app/"


# ----------------------------
# Requests session + basic retry
# ----------------------------
_SESSION = requests.Session()


def request_json(url: str, params: dict, timeout: int = 15, retries: int = 2, backoff: float = 0.6):
    """Lightweight retry wrapper for transient network/API hiccups."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                raise last_exc


# ----------------------------
# Google Sheets (feedback + notes)
# ----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    # Keep Drive scope if your environment needs it for open_by_key / metadata.
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource
def get_gsheet_client():
    sa_info = st.secrets["gcp_service_account"]
    if isinstance(sa_info, str):
        sa_info = json.loads(sa_info)
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource
def get_spreadsheet():
    gc = get_gsheet_client()
    return gc.open_by_key(st.secrets["sheet_id"])


@st.cache_resource
def get_worksheet_feedback():
    """Prefer tab name 'feedback', then 'Sheet1', else first tab."""
    sh = get_spreadsheet()
    for name in ("feedback", "Sheet1"):
        try:
            return sh.worksheet(name)
        except Exception:
            pass
    return sh.get_worksheet(0)


@st.cache_resource
def get_worksheet_notes():
    """
    Notes tab (create if missing). Safe to call; only creates if not present.
    """
    sh = get_spreadsheet()
    try:
        return sh.worksheet("notes")
    except Exception:
        # Create a notes sheet if it doesn't exist
        ws = sh.add_worksheet(title="notes", rows=1000, cols=10)
        ws.append_row(["ts_utc", "label", "note_type", "note_text"], value_input_option="RAW")
        return ws


def ensure_feedback_header():
    """
    Backward compatible:
      Old header: ts_utc,label,target_time_local,verdict,score,thumbs
      New header: ts_utc,label,target_time_local,verdict,score,thumbs,observed
    If header is missing, create the NEW header.
    If header exists but differs, we do not overwrite.
    """
    ws = get_worksheet_feedback()
    values = ws.get_all_values()
    expected_new = ["ts_utc", "label", "target_time_local", "verdict", "score", "thumbs", "observed"]

    if not values:
        ws.append_row(expected_new, value_input_option="RAW")
        return expected_new

    header = [h.strip() for h in values[0]]
    return header


def append_feedback_row(row_dict: dict):
    """
    Append a feedback row using existing sheet header as the source of truth.
    If header includes fields we don't have, fill empty.
    If row_dict includes extra fields, append them as extra columns (rare).
    """
    ws = get_worksheet_feedback()
    header = ensure_feedback_header()
    ts_utc = datetime.utcnow().isoformat(timespec="seconds")
    row_dict = {"ts_utc": ts_utc, **row_dict}

    # If header is empty for some reason, fall back to writing keys
    if not header:
        header = list(row_dict.keys())
        ws.append_row(header, value_input_option="RAW")

    row = [row_dict.get(col, "") for col in header]

    # If there are extra keys not in header, append them as new columns on the right
    extra_keys = [k for k in row_dict.keys() if k not in header]
    if extra_keys:
        row.extend([row_dict[k] for k in extra_keys])

    ws.append_row(row, value_input_option="RAW")


@st.cache_data(ttl=30)
def read_feedback_stats():
    ws = get_worksheet_feedback()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return {
            "total": 0,
            "up": 0,
            "down": 0,
            "thumbs_up_rate": None,
            "observed_total": 0,
            "observed_accuracy": None,
            "cm": None,
        }

    header = [h.strip() for h in values[0]]

    def idx(name: str):
        try:
            return header.index(name)
        except ValueError:
            return None

    thumbs_idx = idx("thumbs")
    observed_idx = idx("observed")
    verdict_idx = idx("verdict")

    up = down = total = 0
    observed_total = 0

    # Confusion matrix: predicted (rows) vs observed (cols)
    # Classes: dry, damp, wet
    classes = ["dry", "damp", "wet"]
    cm = {p: {o: 0 for o in classes} for p in classes}

    def norm_obs(x: str):
        x = (x or "").strip().lower()
        if x in ("dry", "drier"):
            return "dry"
        if x in ("damp", "borderline", "maybe"):
            return "damp"
        if x in ("wet", "soaked"):
            return "wet"
        return None

    def norm_pred(verdict: str):
        v = (verdict or "").lower()
        if "no" in v:
            return "dry"
        if "maybe" in v:
            return "damp"
        if "yes" in v:
            return "wet"
        return None

    correct = 0

    for row in values[1:]:
        if not row:
            continue
        total += 1

        if thumbs_idx is not None and thumbs_idx < len(row):
            t = (row[thumbs_idx] or "").strip().lower()
            if t == "up":
                up += 1
            elif t == "down":
                down += 1

        if observed_idx is not None and verdict_idx is not None:
            if observed_idx < len(row) and verdict_idx < len(row):
                o = norm_obs(row[observed_idx])
                p = norm_pred(row[verdict_idx])
                if o and p:
                    observed_total += 1
                    cm[p][o] += 1
                    if o == p:
                        correct += 1

    thumbs_up_rate = (up / total) * 100 if total else None
    observed_accuracy = (correct / observed_total) * 100 if observed_total else None

    return {
        "total": total,
        "up": up,
        "down": down,
        "thumbs_up_rate": thumbs_up_rate,
        "observed_total": observed_total,
        "observed_accuracy": observed_accuracy,
        "cm": cm if observed_total else None,
    }


def refresh_feedback_stats():
    read_feedback_stats.clear()


@st.cache_data(ttl=30)
def read_recent_notes(label: str, limit: int = 5):
    """
    Read last N notes for the given rink label (most recent first).
    Uses in-memory filtering since sheet is likely small. TTL caches.
    """
    ws = get_worksheet_notes()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return []

    header = [h.strip() for h in values[0]]

    def idx(name: str):
        try:
            return header.index(name)
        except ValueError:
            return None

    ts_idx = idx("ts_utc")
    label_idx = idx("label")
    type_idx = idx("note_type")
    text_idx = idx("note_text")

    rows = []
    for r in values[1:]:
        if not r:
            continue
        if label_idx is None or label_idx >= len(r):
            continue
        if (r[label_idx] or "").strip() != label:
            continue
        rows.append(
            {
                "ts_utc": r[ts_idx] if ts_idx is not None and ts_idx < len(r) else "",
                "note_type": r[type_idx] if type_idx is not None and type_idx < len(r) else "",
                "note_text": r[text_idx] if text_idx is not None and text_idx < len(r) else "",
            }
        )

    # newest first
    rows.reverse()
    return rows[:limit]


def refresh_notes():
    read_recent_notes.clear()


def append_note(label: str, note_type: str, note_text: str):
    ws = get_worksheet_notes()
    ts_utc = datetime.utcnow().isoformat(timespec="seconds")
    ws.append_row([ts_utc, label, note_type, note_text], value_input_option="RAW")
    refresh_notes()


# ----------------------------
# Weather + scoring helpers
# ----------------------------
def geocode(query: str):
    return request_json(
        OPEN_METEO_GEOCODE,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
        retries=2,
    ).get("results") or []


def fetch_weather(lat: float, lon: float, forecast_days: int):
    """
    Force Open-Meteo to return times in America/Los_Angeles.

    Notes on precipitation:
      - hourly.precipitation is the *preceding hour sum* (mm), not a rate.
      - minutely_15.precipitation is the *preceding 15 minutes sum* (mm).
    """
    forecast_days = max(1, min(MAX_FORECAST_DAYS, int(forecast_days)))

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "America/Los_Angeles",
        "forecast_days": forecast_days,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,is_day",
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,is_day",
        "minutely_15": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,is_day",
    }
    return request_json(OPEN_METEO_FORECAST, params=params, timeout=15, retries=2)


def c_to_f(c):
    return None if c is None else (c * 9 / 5) + 32


def compute_dewpoint_c(temp_c: float, rh: float):
    # Magnus approximation
    a, b = 17.62, 243.12
    gamma = (a * temp_c / (b + temp_c)) + math.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def at(block: dict, key: str, idx: int):
    arr = (block or {}).get(key) or []
    return arr[idx] if 0 <= idx < len(arr) else None


def parse_times_local(times: list[str]):
    """Interpret Open-Meteo timezone-local strings as rink-local aware datetimes."""
    out = []
    for t in times or []:
        dt_naive = datetime.fromisoformat(t)
        out.append(dt_naive.replace(tzinfo=RINK_TZ))
    return out


def pick_index(times_local: list[datetime], target_dt_local: datetime) -> int:
    if not times_local:
        return 0
    return min(
        range(len(times_local)),
        key=lambda i: abs((times_local[i] - target_dt_local).total_seconds()),
    )


def needed_forecast_days_for(target_dt_local: datetime) -> int:
    today_local = datetime.now(RINK_TZ).date()
    delta_days = (target_dt_local.date() - today_local).days
    return max(1, min(MAX_FORECAST_DAYS, delta_days + 2))


def sum_recent(arr: list, end_idx: int, lookback: int):
    """Sum numeric arr over a window ending at end_idx."""
    if not arr:
        return 0.0
    s = 0.0
    start = max(0, end_idx - lookback + 1)
    for j in range(start, end_idx + 1):
        if j >= len(arr):
            continue
        v = arr[j]
        if v is None:
            continue
        try:
            s += float(v)
        except Exception:
            continue
    return s


def surface_adjustments(surface_type: str):
    """
    Simple tuning knobs (feel-based, not "scientific"):
      - Tile often stays slick when damp and can "feel wet" longer
      - Concrete dries a bit more predictably
      - Sport court sits between
    Returns: dict with additive adjustments to components.
    """
    s = (surface_type or "").strip().lower()
    if s.startswith("tile"):
        return {
            "dew_bonus": 6,       # dew/condensation feels worse
            "wind_dry_bonus": -2, # wind drying slightly less effective
            "rain_bonus": 5,      # rain wetness lingers
        }
    if s.startswith("sport"):
        return {"dew_bonus": 3, "wind_dry_bonus": -1, "rain_bonus": 2}
    # concrete default
    return {"dew_bonus": 0, "wind_dry_bonus": 0, "rain_bonus": 0}


def compute_confidence(match_delta_min: int | None, dewpoint_source: str, precip_present: bool):
    """
    Confidence is about data quality / alignment, not wetness probability.
    """
    score = 0

    # Time alignment
    if match_delta_min is None:
        score -= 1
    elif match_delta_min <= 15:
        score += 2
    elif match_delta_min <= 30:
        score += 1
    else:
        score -= 1

    # Dew point source
    # "api" is better than computed from temp/RH
    if dewpoint_source == "api":
        score += 2
    elif dewpoint_source == "computed":
        score += 0

    # Precip data presence
    if precip_present:
        score += 1
    else:
        score -= 1

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def wet_assess(
    temp_f,
    dew_f,
    rh,
    wind_mph,
    precip_recent_mm: float | None,
    precip_unit_label: str,
    is_day: int | None,
    matched_time_delta_minutes: int | None,
    surface_type: str,
):
    """
    Returns: verdict, score(0-100), grouped_reasons_dict
    """
    adj = surface_adjustments(surface_type)
    score = 0

    reasons = {
        "Rain / recent moisture": [],
        "Condensation / dew": [],
        "Drying factors": [],
        "Data quality": [],
    }

    # Recent precipitation
    if precip_recent_mm is not None:
        if precip_recent_mm >= 1.0:
            score += 60 + adj["rain_bonus"]
            reasons["Rain / recent moisture"].append(
                f"Recent precipitation ~{precip_recent_mm:.2f} mm over the last {precip_unit_label}."
            )
        elif precip_recent_mm >= 0.2:
            score += 40 + adj["rain_bonus"]
            reasons["Rain / recent moisture"].append(
                f"Some recent precipitation ~{precip_recent_mm:.2f} mm over the last {precip_unit_label}."
            )
        else:
            reasons["Rain / recent moisture"].append(
                f"Little/no recent precipitation (~{precip_recent_mm:.2f} mm over the last {precip_unit_label})."
            )
    else:
        reasons["Rain / recent moisture"].append("Missing precipitation data.")

    # Dew/condensation risk
    if temp_f is not None and dew_f is not None:
        spread = temp_f - dew_f
        if spread <= 2:
            score += 55 + adj["dew_bonus"]
            reasons["Condensation / dew"].append(f"Very tight temp–dewpoint spread ({spread:.1f}°F).")
        elif spread <= 5:
            score += 35 + adj["dew_bonus"]
            reasons["Condensation / dew"].append(f"Small temp–dewpoint spread ({spread:.1f}°F).")
        elif spread <= 8:
            score += 18 + max(0, adj["dew_bonus"] - 2)
            reasons["Condensation / dew"].append(f"Moderate temp–dewpoint spread ({spread:.1f}°F).")
        else:
            reasons["Condensation / dew"].append(f"Wide temp–dewpoint spread ({spread:.1f}°F).")
    else:
        reasons["Condensation / dew"].append("Missing temperature or dew point.")

    # Humidity supports dew risk
    if rh is not None:
        if rh >= 95:
            score += 18
            reasons["Condensation / dew"].append(f"Humidity extremely high ({rh:.0f}%).")
        elif rh >= 85:
            score += 10
            reasons["Condensation / dew"].append(f"Humidity high ({rh:.0f}%).")
        elif rh <= 60:
            score -= 6
            reasons["Condensation / dew"].append(f"Humidity moderate/low ({rh:.0f}%), helps dry.")
    else:
        reasons["Condensation / dew"].append("Missing humidity data.")

    # Night factor (surfaces radiatively cool)
    if is_day is not None and int(is_day) == 0 and rh is not None and rh >= 80:
        score += 8
        reasons["Condensation / dew"].append("Nighttime + high humidity increases dew risk.")

    # Wind dries
    if wind_mph is not None:
        if wind_mph >= 10:
            score -= 14 + adj["wind_dry_bonus"]
            reasons["Drying factors"].append(f"Decent wind ({wind_mph:.1f} mph) helps drying.")
        elif wind_mph <= 3:
            score += 8
            reasons["Drying factors"].append(f"Very light wind ({wind_mph:.1f} mph) allows moisture to linger.")
        else:
            reasons["Drying factors"].append(f"Light/moderate wind ({wind_mph:.1f} mph).")
    else:
        reasons["Drying factors"].append("Missing wind data.")

    # Time match uncertainty
    if matched_time_delta_minutes is not None:
        if matched_time_delta_minutes <= 15:
            reasons["Data quality"].append(f"Forecast time match: within {matched_time_delta_minutes} min.")
        elif matched_time_delta_minutes <= 30:
            reasons["Data quality"].append(f"Forecast time match: within {matched_time_delta_minutes} min (some uncertainty).")
            score += 2
        else:
            reasons["Data quality"].append(f"Forecast time match: within {matched_time_delta_minutes} min (higher uncertainty).")
            score += 4
    else:
        reasons["Data quality"].append("Forecast time match unknown.")

    score = max(0, min(100, int(round(score))))

    if score >= 65:
        verdict = "YES — likely wet"
    elif score >= 45:
        verdict = "MAYBE — likely damp/borderline"
    else:
        verdict = "NO — likely dry"

    return verdict, score, reasons


def action_recommendation(verdict: str, surface_type: str):
    s = (surface_type or "").strip().lower()
    surface_hint = "tile" if s.startswith("tile") else ("sport court" if s.startswith("sport") else "concrete")

    v = (verdict or "").lower()
    if "yes" in v:
        return f"Action: Expect slick spots. Consider softer wheels and cautious corners (surface: {surface_hint}). If you can, check again in 30–60 min."
    if "maybe" in v:
        return f"Action: Borderline. Plan for damp patches—do a quick surface check on arrival (surface: {surface_hint})."
    return f"Action: Likely normal conditions. Bring your usual setup (surface: {surface_hint})."


# ----------------------------
# Favorites + shareable query params
# ----------------------------
def qp_get():
    try:
        return st.query_params
    except Exception:
        # older streamlit compatibility
        return {}


def qp_set(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        try:
            st.experimental_set_query_params(**kwargs)
        except Exception:
            pass


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ----------------------------
# UI
# ----------------------------
st.title("🏒 RinkWet")
APP_VERSION = "v0.4.0"

left, right = st.columns([4, 1])
with left:
    st.info(
        "🚧 **UNDER DEVELOPMENT** — RinkWet is in active development. "
        "Some features may be incomplete or temporarily unavailable.",
        icon="🚧",
    )
with right:
    st.caption(f"**Version:** {APP_VERSION}")

st.caption("Weather-based estimate for wet rink conditions (dew/condensation + recent rain + wind).")

# Session state init
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "debug_safe" not in st.session_state:
    st.session_state.debug_safe = None
if "favorites" not in st.session_state:
    st.session_state.favorites = [
        {"label": DEFAULT_LABEL, "lat": DEFAULT_LAT, "lon": DEFAULT_LON}
    ]
if "last_check_payload" not in st.session_state:
    st.session_state.last_check_payload = None  # stores computed arrays for timeline/next-dry
if "pending_observed" not in st.session_state:
    st.session_state.pending_observed = None  # store selection before submit
if "geocode_results" not in st.session_state:
    st.session_state.geocode_results = []

# Surface type toggle
surface_type = st.selectbox(
    "Surface type",
    ["Concrete (default)", "Sport court", "Tile (plastic)"],
    index=0,
    help="This slightly tunes how dew and drying are weighted.",
)

mode = st.radio("Check for", ["Now (arrival in X minutes)", "Pick a date & time"], horizontal=True)

# Load shared query params if present (for shareable rink link)
qp = qp_get()
shared_lat = parse_float(qp.get("lat")) if hasattr(qp, "get") else None
shared_lon = parse_float(qp.get("lon")) if hasattr(qp, "get") else None
shared_label = (qp.get("label") if hasattr(qp, "get") else None) or None

# Favorites selector
fav_labels = [f["label"] for f in st.session_state.favorites]
fav_default_idx = 0

# If shared rink exists, ensure it appears in favorites (session-only)
if shared_lat is not None and shared_lon is not None and shared_label:
    if shared_label not in fav_labels:
        st.session_state.favorites.insert(0, {"label": shared_label, "lat": shared_lat, "lon": shared_lon})
        fav_labels = [f["label"] for f in st.session_state.favorites]
        fav_default_idx = 0

selected_fav_label = st.selectbox("My rinks", fav_labels, index=fav_default_idx)
selected_fav = next((f for f in st.session_state.favorites if f["label"] == selected_fav_label), None)

use_custom = st.toggle("Search a different location", value=False)
place = ""
chosen_geo = None

if use_custom:
    place = st.text_input("Search location", placeholder="e.g., 'Ventura, CA' or 'Freedom Park Camarillo'")
    colg1, colg2 = st.columns([1, 1])
    if colg1.button("Search"):
        if not place.strip():
            st.warning("Type a location first.")
        else:
            st.session_state.geocode_results = geocode(place.strip())

    results = st.session_state.geocode_results or []
    if results:
        options = []
        for r in results:
            name = r.get("name", "")
            admin1 = r.get("admin1", "")
            country = r.get("country", "")
            lat = r.get("latitude", "")
            lon = r.get("longitude", "")
            options.append(f"{name}, {admin1}, {country}  ({lat:.4f}, {lon:.4f})")

        idx = st.selectbox("Choose result", list(range(len(options))), format_func=lambda i: options[i])
        chosen_geo = results[idx]

        col_save, col_share = st.columns(2)
        if col_save.button("⭐ Save to My rinks"):
            try:
                label = f'{chosen_geo.get("name","")} {chosen_geo.get("admin1","")} {chosen_geo.get("country","")}'.strip()
                lat = float(chosen_geo["latitude"])
                lon = float(chosen_geo["longitude"])
                if label and all(f["label"] != label for f in st.session_state.favorites):
                    st.session_state.favorites.append({"label": label, "lat": lat, "lon": lon})
                    st.success("Saved to My rinks (session only).")
                else:
                    st.info("Already in My rinks.")
            except Exception:
                st.error("Couldn’t save this location.")

        if col_share.button("🔗 Make shareable link"):
            try:
                label = f'{chosen_geo.get("name","")} {chosen_geo.get("admin1","")} {chosen_geo.get("country","")}'.strip()
                lat = float(chosen_geo["latitude"])
                lon = float(chosen_geo["longitude"])
                qp_set(lat=f"{lat:.5f}", lon=f"{lon:.5f}", label=label)
                st.success("Link updated. Copy the URL from your browser bar to share.")
            except Exception:
                st.error("Couldn’t set link parameters.")


# Time inputs
now_local = datetime.now(RINK_TZ)
today_local = now_local.date()
max_day = today_local + timedelta(days=MAX_FORECAST_DAYS - 1)

if mode == "Now (arrival in X minutes)":
    arrival_min = st.slider("Arriving in (minutes)", 0, 180, 20, 5)
    target_dt = now_local + timedelta(minutes=arrival_min)
else:
    d = st.date_input("Date", value=today_local, min_value=today_local, max_value=max_day)
    next_hour = (now_local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    t = st.time_input("Time", value=next_hour.time())
    target_dt = datetime.combine(d, t).replace(tzinfo=RINK_TZ)

st.caption(f"Target time (America/Los_Angeles): {target_dt.strftime('%Y-%m-%d %H:%M')}")

# Determine which location we’re using
def resolve_location():
    if use_custom and chosen_geo:
        lat = float(chosen_geo["latitude"])
        lon = float(chosen_geo["longitude"])
        label = f'{chosen_geo.get("name","")} {chosen_geo.get("admin1","")} {chosen_geo.get("country","")}'.strip()
        return label, lat, lon
    if selected_fav:
        return selected_fav["label"], float(selected_fav["lat"]), float(selected_fav["lon"])
    return DEFAULT_LABEL, DEFAULT_LAT, DEFAULT_LON


# ----------------------------
# Check computation
# ----------------------------
def compute_hourly_scores_for_window(hourly: dict, times_local: list[datetime], start_idx: int, hours: int, surface_type: str):
    """
    Compute wet scores for a window of hourly indices using:
      - hourly temp/rh/dew/wind/is_day
      - precip recent window = sum of last 3 hourly precip buckets
    Returns list of dicts: {"dt": datetime, "score": int, "verdict": str}
    """
    out = []
    precip_arr = (hourly or {}).get("precipitation") or []

    for k in range(hours):
        i = start_idx + k
        if i < 0 or i >= len(times_local):
            continue

        temp_c = at(hourly, "temperature_2m", i)
        rh = at(hourly, "relative_humidity_2m", i)
        dew_c = at(hourly, "dew_point_2m", i)
        wind_kmh = at(hourly, "wind_speed_10m", i)
        is_day = at(hourly, "is_day", i)

        dew_source = "api"
        if dew_c is None and temp_c is not None and rh is not None:
            dew_c = compute_dewpoint_c(temp_c, rh)
            dew_source = "computed"

        temp_f = c_to_f(temp_c)
        dew_f = c_to_f(dew_c)
        wind_mph = None if wind_kmh is None else float(wind_kmh) * 0.621371

        precip_last3h = sum_recent(precip_arr, i, lookback=3) if precip_arr else None

        verdict, score, _reasons = wet_assess(
            temp_f=temp_f,
            dew_f=dew_f,
            rh=rh,
            wind_mph=wind_mph,
            precip_recent_mm=precip_last3h,
            precip_unit_label="3 hours",
            is_day=is_day,
            matched_time_delta_minutes=None,
            surface_type=surface_type,
        )
        out.append({"dt": times_local[i], "score": score, "verdict": verdict, "dew_source": dew_source})
    return out


if st.button("Check"):
    try:
        label, lat, lon = resolve_location()
        forecast_days = needed_forecast_days_for(target_dt)
        wx = fetch_weather(lat, lon, forecast_days=forecast_days)

        # 15-min block for "Now"
        m15 = wx.get("minutely_15") or {}
        m15_times_raw = m15.get("time") or []
        m15_times_local = parse_times_local(m15_times_raw)
        i_now_m15 = pick_index(m15_times_local, datetime.now(RINK_TZ)) if m15_times_local else 0
        used_m15_time = m15_times_raw[i_now_m15] if m15_times_raw else ""

        delta_now_min = None
        if m15_times_local:
            delta_now_min = int(round(abs((m15_times_local[i_now_m15] - datetime.now(RINK_TZ)).total_seconds()) / 60))

        # Hourly for target + planning
        hourly = wx.get("hourly") or {}
        h_times_raw = hourly.get("time") or []
        h_times_local = parse_times_local(h_times_raw)
        i_t_h = pick_index(h_times_local, target_dt) if h_times_local else 0
        used_h_time = h_times_raw[i_t_h] if h_times_raw else ""

        delta_t_min = None
        if h_times_local:
            delta_t_min = int(round(abs((h_times_local[i_t_h] - target_dt).total_seconds()) / 60))

        # ----------------------------
        # NOW (consistent 15-min)
        # ----------------------------
        temp_c_now = at(m15, "temperature_2m", i_now_m15)
        rh_now = at(m15, "relative_humidity_2m", i_now_m15)
        wind_kmh_now = at(m15, "wind_speed_10m", i_now_m15)
        precip_now_15 = at(m15, "precipitation", i_now_m15)  # preceding 15-min sum (mm)
        dew_c_now = at(m15, "dew_point_2m", i_now_m15)
        is_day_now = at(m15, "is_day", i_now_m15)

        dew_source_now = "api"
        if dew_c_now is None and temp_c_now is not None and rh_now is not None:
            dew_c_now = compute_dewpoint_c(temp_c_now, rh_now)
            dew_source_now = "computed"

        temp_f_now = c_to_f(temp_c_now)
        dew_f_now = c_to_f(dew_c_now)
        wind_mph_now = None if wind_kmh_now is None else float(wind_kmh_now) * 0.621371

        verdict_now, score_now, reasons_now = wet_assess(
            temp_f=temp_f_now,
            dew_f=dew_f_now,
            rh=rh_now,
            wind_mph=wind_mph_now,
            precip_recent_mm=(float(precip_now_15) if precip_now_15 is not None else None),
            precip_unit_label="15 min",
            is_day=is_day_now,
            matched_time_delta_minutes=delta_now_min,
            surface_type=surface_type,
        )

        now_conf = compute_confidence(delta_now_min, dew_source_now, precip_now_15 is not None)

        # ----------------------------
        # TARGET (hourly + recent rain window)
        # ----------------------------
        temp_c_t = at(hourly, "temperature_2m", i_t_h)
        rh_t = at(hourly, "relative_humidity_2m", i_t_h)
        wind_kmh_t = at(hourly, "wind_speed_10m", i_t_h)
        precip_t_1h = at(hourly, "precipitation", i_t_h)  # preceding hour sum (mm)
        dew_c_t = at(hourly, "dew_point_2m", i_t_h)
        is_day_t = at(hourly, "is_day", i_t_h)

        dew_source_t = "api"
        if dew_c_t is None and temp_c_t is not None and rh_t is not None:
            dew_c_t = compute_dewpoint_c(temp_c_t, rh_t)
            dew_source_t = "computed"

        temp_f_t = c_to_f(temp_c_t)
        dew_f_t = c_to_f(dew_c_t)
        wind_mph_t = None if wind_kmh_t is None else float(wind_kmh_t) * 0.621371

        precip_arr = (hourly or {}).get("precipitation") or []
        precip_last3h = sum_recent(precip_arr, i_t_h, lookback=3) if precip_arr else None
        precip_last1h = None if precip_t_1h is None else float(precip_t_1h)

        verdict_t, score_t, reasons_t = wet_assess(
            temp_f=temp_f_t,
            dew_f=dew_f_t,
            rh=rh_t,
            wind_mph=wind_mph_t,
            precip_recent_mm=(precip_last3h if precip_last3h is not None else precip_last1h),
            precip_unit_label=("3 hours" if precip_last3h is not None else "1 hour"),
            is_day=is_day_t,
            matched_time_delta_minutes=delta_t_min,
            surface_type=surface_type,
        )

        # Add explicit target-hour precip note
        if precip_last1h is not None:
            reasons_t["Rain / recent moisture"].insert(
                0, f"Target-hour precipitation bucket: {precip_last1h:.2f} mm (preceding 1 hour sum)."
            )

        target_conf = compute_confidence(delta_t_min, dew_source_t, precip_last1h is not None or precip_last3h is not None)

        # Save for feedback + planning tools
        st.session_state.last_result = {
            "label": label,
            "lat": lat,
            "lon": lon,
            "target_local": target_dt.strftime("%Y-%m-%d %H:%M"),
            "verdict": verdict_t,
            "score": score_t,
            "surface_type": surface_type,
        }

        # Planning window (timeline + next-dry)
        # start from current nearest hourly index for timeline
        i_now_h = pick_index(h_times_local, datetime.now(RINK_TZ)) if h_times_local else 0
        window = compute_hourly_scores_for_window(hourly, h_times_local, start_idx=i_now_h, hours=12, surface_type=surface_type)
        st.session_state.last_check_payload = {
            "hourly_window": window,
            "hourly_used_start_idx": i_now_h,
            "label": label,
            "lat": lat,
            "lon": lon,
        }

        # Debug info
        st.session_state.debug_safe = {
            "now_local": datetime.now(RINK_TZ).strftime("%Y-%m-%d %H:%M %Z"),
            "target_dt": target_dt.strftime("%Y-%m-%d %H:%M %Z"),
            "m15_index_now": i_now_m15,
            "hourly_index_target": i_t_h,
            "used_m15_time": used_m15_time,
            "used_hourly_time": used_h_time,
            "match_delta_now_min": delta_now_min,
            "match_delta_target_min": delta_t_min,
            "dew_source_now": dew_source_now,
            "dew_source_target": dew_source_t,
            "label": label,
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "surface_type": surface_type,
        }

        # ----------------------------
        # Display
        # ----------------------------
        st.subheader(f"📍 {label}")

        # Community notes (last 5)
        with st.expander("🗒️ Rink notes (latest)", expanded=False):
            try:
                notes = read_recent_notes(label, limit=5)
                if not notes:
                    st.write("No notes yet.")
                else:
                    for n in notes:
                        st.write(f"- **{n['note_type']}** — {n['note_text']}  _(UTC: {n['ts_utc']})_")
            except Exception:
                st.write("Notes not available.")

            st.divider()
            st.write("Add a note (helps everyone):")
            nt = st.selectbox("Note type", ["Irrigation", "Shade", "Drainage", "Sticky spot", "Other"], index=0)
            ntext = st.text_input("Note", placeholder="e.g., 'Back corner stays wet after rain' (keep it short)")
            if st.button("Submit note"):
                if not ntext.strip():
                    st.warning("Type a note first.")
                else:
                    try:
                        append_note(label, nt, ntext.strip())
                        st.success("Note added. Thanks.")
                    except Exception:
                        st.error("Couldn’t write note (check your Sheets permissions).")

        st.write("**Now (15-minute snapshot)**")
        cols = st.columns(5)
        cols[0].metric("Temp", "—" if temp_f_now is None else f"{temp_f_now:.1f}°F")
        cols[1].metric("Dew point", "—" if dew_f_now is None else f"{dew_f_now:.1f}°F")
        cols[2].metric("Humidity", "—" if rh_now is None else f"{rh_now:.0f}%")
        cols[3].metric("Wind", "—" if wind_mph_now is None else f"{wind_mph_now:.1f} mph")
        cols[4].metric("Confidence", now_conf)

        st.write("**Target (nearest hourly forecast)**")
        cols2 = st.columns(5)
        cols2[0].metric("Temp", "—" if temp_f_t is None else f"{temp_f_t:.1f}°F")
        cols2[1].metric("Dew point", "—" if dew_f_t is None else f"{dew_f_t:.1f}°F")
        cols2[2].metric("Humidity", "—" if rh_t is None else f"{rh_t:.0f}%")
        cols2[3].metric("Wind", "—" if wind_mph_t is None else f"{wind_mph_t:.1f} mph")
        cols2[4].metric("Confidence", target_conf)

        st.divider()
        colA, colB = st.columns(2)

        with colA:
            st.write("### Now verdict")
            if score_now >= 65:
                st.error(f"{verdict_now} (Risk: {score_now}/100)")
            elif score_now >= 45:
                st.warning(f"{verdict_now} (Risk: {score_now}/100)")
            else:
                st.success(f"{verdict_now} (Risk: {score_now}/100)")
            st.caption(action_recommendation(verdict_now, surface_type))

        with colB:
            st.write("### Target verdict")
            if score_t >= 65:
                st.error(f"{verdict_t} (Risk: {score_t}/100)")
            elif score_t >= 45:
                st.warning(f"{verdict_t} (Risk: {score_t}/100)")
            else:
                st.success(f"{verdict_t} (Risk: {score_t}/100)")
            st.caption(action_recommendation(verdict_t, surface_type))

        # Grouped reasons (more scannable)
        st.write("**Why (target):**")
        for group, items in reasons_t.items():
            if not items:
                continue
            st.write(f"**{group}:**")
            for it in items:
                st.write(f"- {it}")

        if used_h_time:
            st.caption(f"Target forecast hour used (America/Los_Angeles): {used_h_time}  (match: ±{delta_t_min} min)")

        # ----------------------------
        # Timeline chart (next 12 hours)
        # ----------------------------
        payload = st.session_state.get("last_check_payload") or {}
        window = payload.get("hourly_window") or []
        if window:
            st.subheader("📈 Next 12 hours risk trend")
            times = [w["dt"].strftime("%H:%M") for w in window]
            scores = [w["score"] for w in window]
            st.line_chart({"Wet risk (0–100)": scores})

            # show labels underneath for context
            with st.expander("Show times used"):
                st.write(", ".join(times))

        # ----------------------------
        # Next dry window (planner)
        # ----------------------------
        st.subheader("🕒 When will it be driest?")
        if st.button("Find next driest window"):
            if not window:
                st.info("Run a check first.")
            else:
                # choose best 2 hours (lowest risk)
                ranked = sorted(window, key=lambda x: x["score"])
                top = ranked[:2]
                st.write("Best upcoming windows (lowest risk):")
                for tbest in top:
                    st.write(f"- **{tbest['dt'].strftime('%a %H:%M')}** — {tbest['verdict']} ({tbest['score']}/100)")

        # Save shareable link for current rink quickly
        with st.expander("🔗 Share this rink link"):
            st.write("This adds rink coordinates to the URL so someone else opens the same rink by default.")
            if st.button("Update URL with this rink"):
                qp_set(lat=f"{lat:.5f}", lon=f"{lon:.5f}", label=label)
                st.success("URL updated. Copy from browser address bar.")

    except requests.RequestException as e:
        st.error(f"Network/API error: {e}")
    except Exception as e:
        st.error(f"App error: {e}")


# ----------------------------
# Feedback UI (Google Sheets)
# ----------------------------
st.divider()
st.subheader("✅ Was the prediction accurate?")

try:
    stats = read_feedback_stats()
    a, b, c, d = st.columns(4)
    a.metric("Total votes", stats["total"])
    b.metric("👍", stats["up"])
    c.metric("Thumbs-up rate", "—" if stats["thumbs_up_rate"] is None else f"{stats['thumbs_up_rate']:.0f}%")
    d.metric("Observed accuracy", "—" if stats["observed_accuracy"] is None else f"{stats['observed_accuracy']:.0f}%")

    last = st.session_state.get("last_result")
    if not last:
        st.info("Run a check first, then vote 👍 or 👎 based on what you actually saw at the rink.")
    else:
        st.caption(
            f"Last check: {last['label']} @ {last['target_local']} → {last['verdict']} ({last['score']}/100)"
        )

        # Ask for observed condition (enables real accuracy)
        observed = st.selectbox(
            "What was it actually?",
            ["(choose)", "Dry", "Damp/Borderline", "Wet"],
            index=0,
            help="This helps the model improve and enables real accuracy tracking.",
        )

        col1, col2 = st.columns(2)
        if col1.button("👍 Accurate"):
            append_feedback_row(
                {
                    "label": last["label"],
                    "target_time_local": last["target_local"],
                    "verdict": last["verdict"],
                    "score": int(last["score"]),
                    "thumbs": "up",
                    "observed": "" if observed == "(choose)" else observed,
                }
            )
            refresh_feedback_stats()
            st.success("Logged 👍. Thank you for your feedback.")

        if col2.button("👎 Not accurate"):
            append_feedback_row(
                {
                    "label": last["label"],
                    "target_time_local": last["target_local"],
                    "verdict": last["verdict"],
                    "score": int(last["score"]),
                    "thumbs": "down",
                    "observed": "" if observed == "(choose)" else observed,
                }
            )
            refresh_feedback_stats()
            st.success("Logged 👎. Thank you for your feedback.")

        # Confusion matrix (if we have observed labels)
        if stats.get("cm"):
            with st.expander("📊 Model report (observed vs predicted)"):
                st.write("Rows = predicted, Columns = observed")
                cm = stats["cm"]
                st.write(
                    {
                        "pred_dry": cm["dry"],
                        "pred_damp": cm["damp"],
                        "pred_wet": cm["wet"],
                    }
                )
        else:
            st.caption("Tip: pick an observed condition above to enable true accuracy tracking over time.")

except Exception:
    st.error(
        "Feedback system isn't connected yet. Make sure Streamlit Secrets contain gcp_service_account + sheet_id, "
        "and your service account is shared as Editor on the sheet."
    )


# ----------------------------
# Share + Debug (bottom)
# ----------------------------
st.divider()

with st.expander("📣 Share this app"):
    st.write("Share this link:")
    st.code(APP_URL, language="text")

    qr_url = "https://api.qrserver.com/v1/create-qr-code/?" + urllib.parse.urlencode(
        {"size": "220x220", "data": APP_URL}
    )
    st.image(qr_url, caption="Scan to open RinkWet")

with st.expander("Debug (safe)"):
    dbg = st.session_state.get("debug_safe")
    if not dbg:
        st.write("Run a check first.")
    else:
        st.json(dbg)


# ----------------------------
# Disclaimer
# ----------------------------
st.caption(
    "Disclaimer: This app provides a weather-based estimate only. Surface conditions may differ due to irrigation, "
    "shade, drainage, surface material, or microclimate. Use at your own risk."
)


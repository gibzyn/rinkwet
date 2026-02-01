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
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="RinkWet", page_icon="🏒", layout="centered")


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_LABEL = "Freedom Park Inline Hockey Arena (Camarillo)"
DEFAULT_LAT = 34.2138
DEFAULT_LON = -119.0856
DEFAULT_TZ = "America/Los_Angeles"  # fallback only

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"

# Free POI/name search fallback (OpenStreetMap)
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

MAX_FORECAST_DAYS = 16
APP_URL = "https://rinkwet.streamlit.app/"


# ----------------------------
# Requests session + basic retry
# ----------------------------
_SESSION = requests.Session()

# Nominatim usage expectations:
# - include a descriptive User-Agent
# - rate limit (don’t hammer)
NOMINATIM_HEADERS = {
    "User-Agent": f"RinkWet/0.5.2 (+{APP_URL})"  # bumped for this fix
}
_NOMINATIM_MIN_INTERVAL_SEC = 1.0  # conservative: ~1 request/sec


def _throttle_nominatim():
    """
    Basic local throttle. Streamlit reruns can still cause multiple calls,
    so we keep the throttle timestamp in session_state.
    """
    last = st.session_state.get("_nominatim_last_ts", 0.0)
    now = time.time()
    wait = _NOMINATIM_MIN_INTERVAL_SEC - (now - last)
    if wait > 0:
        time.sleep(wait)
    st.session_state["_nominatim_last_ts"] = time.time()


def request_json(
    url: str,
    params: dict,
    timeout: int = 15,
    retries: int = 2,
    backoff: float = 0.6,
    headers: dict | None = None,
    throttle: bool = False,
):
    """Lightweight retry wrapper for transient network/API hiccups."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            if throttle:
                _throttle_nominatim()
            r = _SESSION.get(url, params=params, timeout=timeout, headers=headers)
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
    ws = get_worksheet_feedback()
    header = ensure_feedback_header()
    ts_utc = datetime.utcnow().isoformat(timespec="seconds")
    row_dict = {"ts_utc": ts_utc, **row_dict}

    if not header:
        header = list(row_dict.keys())
        ws.append_row(header, value_input_option="RAW")

    row = [row_dict.get(col, "") for col in header]

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
# Location helpers (lat/lon + geocode fallback + timezone derivation)
# ----------------------------
def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_latlon(text: str):
    """
    Accepts:
      '34.2138, -119.0856' OR '34.2138 -119.0856'
    Returns (lat, lon) or None
    """
    if not text:
        return None
    t = text.strip().replace("(", "").replace(")", "")
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in t.split() if p.strip()]

    if len(parts) != 2:
        return None

    lat = parse_float(parts[0])
    lon = parse_float(parts[1])
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def simplify_query(q: str):
    """
    Light cleanup so Open-Meteo has a better chance, but don't over-strip.
    """
    if not q:
        return q
    s = q.strip()
    for ch in ["|", "\\", "#", "@"]:
        s = s.replace(ch, " ")
    # keep commas because "City, State" helps
    s = " ".join(s.split())
    return s


def geocode_open_meteo(query: str, count: int = 10):
    return request_json(
        OPEN_METEO_GEOCODE,
        params={"name": query, "count": count, "language": "en", "format": "json"},
        timeout=15,
        retries=2,
    ).get("results") or []


@st.cache_data(ttl=3600)
def nominatim_search(query: str, limit: int = 6):
    """
    Free POI/name search fallback for rink/business-like queries.
    TTL cached to reduce calls and avoid rate limiting.
    """
    q = (query or "").strip()
    if not q:
        return []

    data = request_json(
        NOMINATIM_SEARCH,
        params={
            "q": q,
            "format": "json",
            "addressdetails": 1,
            "limit": limit,
            # ✅ FIX: keep results in the US when users give a US query
            "countrycodes": "us",
        },
        timeout=15,
        retries=1,
        headers=NOMINATIM_HEADERS,
        throttle=True,
    )
    return data if isinstance(data, list) else []


def _norm_admin_country_from_nominatim(addr: dict):
    """
    Extract something useful for display + saved label.
    """
    if not isinstance(addr, dict):
        return "", ""
    admin = (
        addr.get("state")
        or addr.get("province")
        or addr.get("region")
        or addr.get("county")
        or ""
    )
    country = addr.get("country") or ""
    return admin, country


def _build_nominatim_queries(original: str) -> list[str]:
    """
    Nominatim is picky. Generate a few variants that improve hit-rate for:
      - parks with "rink" in the user query
      - "Outdoor Rink" style wording
      - CA queries missing comma formatting
    """
    q = (original or "").strip()
    if not q:
        return []

    out: list[str] = []

    # 1) Original
    out.append(q)

    # 2) Simplified punctuation
    q2 = simplify_query(q)
    if q2 and q2 != q:
        out.append(q2)

    q2_lower = q2.lower()

    # 3) Expand common abbreviation: SB -> Santa Barbara
    if q2_lower.startswith("sb ") and "santa barbara" not in q2_lower:
        out.append("Santa Barbara " + q2[3:])
        out.append("Santa Barbara, CA " + q2[3:])

    # 4) Strip common "junk" tokens that often reduce POI matching
    #    (keep location tokens like Calabasas/CA/Street names etc.)
    junk = {
        "outdoor", "rink", "roller", "inline", "hockey", "ice", "arena",
        "court", "skate", "skating",
    }

    tokens = [t for t in q2.split() if t and t.lower() not in junk]
    cleaned = " ".join(tokens).strip()
    if cleaned and cleaned not in out:
        out.append(cleaned)

    # 5) If we end with " CA" but no comma, try comma form
    #    "De Anza Park Calabasas CA" -> "De Anza Park Calabasas, CA"
    if q2.endswith(" CA") and "," not in q2:
        out.append(q2[:-3].strip() + ", CA")
    if cleaned.endswith(" CA") and "," not in cleaned:
        out.append(cleaned[:-3].strip() + ", CA")

    # 6) If the query mentions "park", try park-only + city/state (if present)
    #    (This helps cases where the rink itself isn't mapped, but the park is.)
    if "park" in q2_lower:
        park_tokens = [t for t in q2.split() if t.lower() not in junk]
        park_clean = " ".join(park_tokens).strip()
        if park_clean and park_clean not in out:
            out.append(park_clean)

    # 7) Small helpful hints if query is short / ambiguous
    if len(q2.split()) <= 6:
        if "park" not in q2_lower:
            out.append(q2 + " park")

    # De-dupe while preserving order
    deduped: list[str] = []
    for s in out:
        s2 = " ".join((s or "").split())
        if s2 and s2 not in deduped:
            deduped.append(s2)

    return deduped[:6]


@st.cache_data(ttl=60 * 60 * 24 * 30)  # 30 days
def tz_from_coords(lat: float, lon: float) -> str:
    """
    Derive an IANA timezone for any lat/lon using Open-Meteo.
    Uses timezone=auto, then reads 'timezone' field from response.
    Cached to avoid repeated calls.
    """
    data = request_json(
        OPEN_METEO_FORECAST,
        params={
            "latitude": float(lat),
            "longitude": float(lon),
            "timezone": "auto",
            "forecast_days": 1,
            "current": "temperature_2m",
        },
        timeout=15,
        retries=2,
    )
    tz = (data.get("timezone") or "").strip()
    return tz or DEFAULT_TZ


def geocode_with_fallback(query: str, count: int = 10):
    """
    Strategy:
      1) Open-Meteo (great for cities/regions)
      2) Open-Meteo with simplified query
      3) Nominatim (great for POI/business names) with multiple variants
    Returns a unified list of result dicts.
    """
    q = (query or "").strip()
    if not q:
        return []

    # 1) Open-Meteo
    r1 = geocode_open_meteo(q, count=count)
    if r1:
        for r in r1:
            r["source"] = "open-meteo"
        return r1

    # 2) Open-Meteo simplified
    q2 = simplify_query(q)
    if q2 != q:
        r2 = geocode_open_meteo(q2, count=count)
        if r2:
            for r in r2:
                r["source"] = "open-meteo"
            return r2

    # 3) Nominatim fallback (try several query variants)
    variants = _build_nominatim_queries(q)
    out = []
    seen = set()

    for vq in variants:
        nom = nominatim_search(vq, limit=min(6, count))
        for r in nom:
            try:
                lat = float(r.get("lat"))
                lon = float(r.get("lon"))
                display = (r.get("display_name") or "").strip()
                addr = r.get("address") or {}
                admin1, country = _norm_admin_country_from_nominatim(addr)

                short = display.split(",")[0].strip() if display else "Unknown place"

                key = (round(lat, 5), round(lon, 5), short.lower())
                if key in seen:
                    continue
                seen.add(key)

                out.append(
                    {
                        "name": short,
                        "admin1": admin1,
                        "country": country,
                        "latitude": lat,
                        "longitude": lon,
                        "display_name": display,
                        "source": "nominatim",
                        "timezone": None,  # filled lazily when selected / used
                    }
                )
            except Exception:
                continue

        # If we got any results on a variant, stop early
        if out:
            break

    return out


# ----------------------------
# Weather + scoring helpers
# ----------------------------
def fetch_weather(lat: float, lon: float, forecast_days: int):
    """
    IMPORTANT:
      - timezone="auto" makes Open-Meteo return times in the *location's* timezone
      - hourly.precipitation is the preceding hour sum (mm)
      - minutely_15.precipitation is the preceding 15 min sum (mm)
    """
    forecast_days = max(1, min(MAX_FORECAST_DAYS, int(forecast_days)))

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "forecast_days": forecast_days,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,is_day",
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,is_day",
        "minutely_15": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,is_day",
    }
    return request_json(OPEN_METEO_FORECAST, params=params, timeout=15, retries=2)


def c_to_f(c):
    return None if c is None else (c * 9 / 5) + 32


def compute_dewpoint_c(temp_c: float, rh: float):
    a, b = 17.62, 243.12
    gamma = (a * temp_c / (b + temp_c)) + math.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def at(block: dict, key: str, idx: int):
    arr = (block or {}).get(key) or []
    return arr[idx] if 0 <= idx < len(arr) else None


def parse_times_local(times: list[str], tz: ZoneInfo):
    """Interpret Open-Meteo timezone-local strings as timezone-aware datetimes."""
    out = []
    for t in times or []:
        dt_naive = datetime.fromisoformat(t)
        out.append(dt_naive.replace(tzinfo=tz))
    return out


def pick_index(times_local: list[datetime], target_dt_local: datetime) -> int:
    if not times_local:
        return 0
    return min(
        range(len(times_local)),
        key=lambda i: abs((times_local[i] - target_dt_local).total_seconds()),
    )


def needed_forecast_days_for(target_dt_local: datetime, now_local: datetime) -> int:
    delta_days = (target_dt_local.date() - now_local.date()).days
    return max(1, min(MAX_FORECAST_DAYS, delta_days + 2))


def sum_recent(arr: list, end_idx: int, lookback: int):
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
    s = (surface_type or "").strip().lower()
    if s.startswith("tile"):
        return {"dew_bonus": 6, "wind_dry_bonus": -2, "rain_bonus": 5}
    if s.startswith("sport"):
        return {"dew_bonus": 3, "wind_dry_bonus": -1, "rain_bonus": 2}
    return {"dew_bonus": 0, "wind_dry_bonus": 0, "rain_bonus": 0}


def compute_confidence(match_delta_min: int | None, dewpoint_source: str, precip_present: bool):
    score = 0
    if match_delta_min is None:
        score -= 1
    elif match_delta_min <= 15:
        score += 2
    elif match_delta_min <= 30:
        score += 1
    else:
        score -= 1

    if dewpoint_source == "api":
        score += 2

    score += 1 if precip_present else -1

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
    adj = surface_adjustments(surface_type)
    score = 0

    reasons = {
        "Rain / recent moisture": [],
        "Condensation / dew": [],
        "Drying factors": [],
        "Data quality": [],
    }

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

    if is_day is not None and int(is_day) == 0 and rh is not None and rh >= 80:
        score += 8
        reasons["Condensation / dew"].append("Nighttime + high humidity increases dew risk.")

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

    if matched_time_delta_minutes is not None:
        if matched_time_delta_minutes <= 15:
            reasons["Data quality"].append(f"Forecast time match: within {matched_time_delta_minutes} min.")
        elif matched_time_delta_minutes <= 30:
            reasons["Data quality"].append(
                f"Forecast time match: within {matched_time_delta_minutes} min (some uncertainty)."
            )
            score += 2
        else:
            reasons["Data quality"].append(
                f"Forecast time match: within {matched_time_delta_minutes} min (higher uncertainty)."
            )
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
        return (
            f"Action: Expect slick spots. Consider softer wheels and cautious corners (surface: {surface_hint}). "
            "If you can, check again in 30–60 min."
        )
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
        return {}


def qp_set(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        try:
            st.experimental_set_query_params(**kwargs)
        except Exception:
            pass


# ----------------------------
# UI
# ----------------------------
st.title("🏒 RinkWet")
APP_VERSION = "v0.5.2"

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
    st.session_state.favorites = [{"label": DEFAULT_LABEL, "lat": DEFAULT_LAT, "lon": DEFAULT_LON}]
if "last_check_payload" not in st.session_state:
    st.session_state.last_check_payload = None
if "geocode_results" not in st.session_state:
    st.session_state.geocode_results = []
if "geo_selected_idx" not in st.session_state:
    st.session_state.geo_selected_idx = 0
if "geo_selected" not in st.session_state:
    st.session_state.geo_selected = None

# Surface type toggle
surface_type = st.selectbox(
    "Surface type",
    ["Tile (default)", "Sport court", "Concrete"],
    index=0,
    help="This slightly tunes how dew and drying are weighted.",
)

mode = st.radio("Check for", ["Now (arrival in X minutes)", "Pick a date & time"], horizontal=True)

# Load shared query params if present (for shareable rink link)
qp = qp_get()
shared_lat = parse_float(qp.get("lat")) if hasattr(qp, "get") else None
shared_lon = parse_float(qp.get("lon")) if hasattr(qp, "get") else None
shared_label = (qp.get("label") if hasattr(qp, "get") else None) or None

fav_labels = [f["label"] for f in st.session_state.favorites]
fav_default_idx = 0

if shared_lat is not None and shared_lon is not None and shared_label:
    if shared_label not in fav_labels:
        st.session_state.favorites.insert(0, {"label": shared_label, "lat": shared_lat, "lon": shared_lon})
        fav_labels = [f["label"] for f in st.session_state.favorites]
        fav_default_idx = 0

selected_fav_label = st.selectbox("My rinks", fav_labels, index=fav_default_idx)
selected_fav = next((f for f in st.session_state.favorites if f["label"] == selected_fav_label), None)

# Custom search
use_custom = st.toggle("Search a different location", value=False)
place = ""

if use_custom:
    st.caption("Tip: best results are **Place name + City, State** (example: `SB Roller Hockey Rink, Santa Barbara, CA`).")
    place = st.text_input(
        "Search location (City/State, rink name, or paste coords like `34.21, -119.08`)",
        placeholder="e.g., 'SB Roller Hockey Rink, Santa Barbara, CA' or '34.2138, -119.0856'",
        key="geo_query",
    )

    colg1, _colg2 = st.columns([1, 1])
    if colg1.button("Search", key="geo_search_btn"):
        if not place.strip():
            st.warning("Type a location first.")
        else:
            q = place.strip()

            ll = parse_latlon(q)
            if ll:
                lat, lon = ll
                # Derive timezone immediately for coords
                tz_guess = tz_from_coords(lat, lon)
                st.session_state.geocode_results = [{
                    "name": "Custom coordinates",
                    "admin1": "",
                    "country": "",
                    "latitude": lat,
                    "longitude": lon,
                    "display_name": "Custom coordinates",
                    "source": "coords",
                    "timezone": tz_guess,
                }]
                st.session_state.geo_selected_idx = 0
                st.session_state.geo_selected = st.session_state.geocode_results[0]
            else:
                try:
                    with st.spinner("Searching locations..."):
                        st.session_state.geocode_results = geocode_with_fallback(q, count=10)

                    results = st.session_state.geocode_results or []
                    if not results:
                        st.warning(
                            "No results.\n\n"
                            "Try **Place + City, State** (example: `SB Roller Hockey Rink, Santa Barbara, CA`) "
                            "or paste **coordinates** like `34.2138, -119.0856`."
                        )
                        st.session_state.geo_selected = None
                    else:
                        st.session_state.geo_selected_idx = 0
                        st.session_state.geo_selected = results[0]
                except Exception as e:
                    st.error("Geocoding failed. Exact error:")
                    st.code(str(e))
                    st.session_state.geocode_results = []
                    st.session_state.geo_selected = None

    results = st.session_state.geocode_results or []
    if results:
        options = []
        for r in results:
            lat = r.get("latitude", 0.0)
            lon = r.get("longitude", 0.0)

            display_name = (r.get("display_name") or "").strip()
            if display_name:
                label_line = display_name
            else:
                name = (r.get("name", "") or "").strip()
                admin1 = (r.get("admin1", "") or "").strip()
                country = (r.get("country", "") or "").strip()
                label_line = ", ".join([x for x in [name, admin1, country] if x])

            source = r.get("source", "")
            suffix = f"[{source}]" if source else ""
            options.append(f"{label_line}  ({lat:.4f}, {lon:.4f}) {suffix}")

        st.session_state.geo_selected_idx = st.selectbox(
            "Choose result",
            list(range(len(options))),
            index=min(st.session_state.geo_selected_idx, len(options) - 1),
            format_func=lambda i: options[i],
            key="geo_select_idx",
        )

        st.session_state.geo_selected = results[st.session_state.geo_selected_idx]
        chosen_geo = st.session_state.geo_selected

        # Lazy-fill timezone for any selected result (Nominatim + coords especially)
        try:
            if not chosen_geo.get("timezone"):
                chosen_geo["timezone"] = tz_from_coords(float(chosen_geo["latitude"]), float(chosen_geo["longitude"]))
        except Exception:
            chosen_geo["timezone"] = None

        if chosen_geo.get("timezone"):
            st.caption(f"Derived timezone for selected location: **{chosen_geo['timezone']}**")

        col_save, col_share = st.columns(2)
        if col_save.button("⭐ Save to My rinks", key="geo_save_btn"):
            try:
                display_name = (chosen_geo.get("display_name") or "").strip()
                if display_name:
                    label = display_name
                else:
                    label = f'{chosen_geo.get("name","")} {chosen_geo.get("admin1","")} {chosen_geo.get("country","")}'.strip()

                label = " ".join(label.split())
                if len(label) > 80:
                    label = label[:77] + "..."

                lat = float(chosen_geo["latitude"])
                lon = float(chosen_geo["longitude"])

                if label and all(f["label"] != label for f in st.session_state.favorites):
                    st.session_state.favorites.append({"label": label, "lat": lat, "lon": lon})
                    st.success("Saved to My rinks (session only).")
                else:
                    st.info("Already in My rinks.")
            except Exception as e:
                st.error("Couldn’t save this location. Exact error:")
                st.code(str(e))

        if col_share.button("🔗 Make shareable link", key="geo_share_btn"):
            try:
                display_name = (chosen_geo.get("display_name") or "").strip()
                if display_name:
                    label = display_name
                else:
                    label = f'{chosen_geo.get("name","")} {chosen_geo.get("admin1","")} {chosen_geo.get("country","")}'.strip()
                label = " ".join(label.split())
                if len(label) > 80:
                    label = label[:77] + "..."

                lat = float(chosen_geo["latitude"])
                lon = float(chosen_geo["longitude"])
                qp_set(lat=f"{lat:.5f}", lon=f"{lon:.5f}", label=label)
                st.success("Link updated. Copy the URL from your browser bar to share.")
            except Exception as e:
                st.error("Couldn’t set link parameters. Exact error:")
                st.code(str(e))
else:
    st.session_state.geo_selected = None


def resolve_location():
    """
    Returns (label, lat, lon, tz_name_or_None)
    """
    if use_custom and st.session_state.get("geo_selected"):
        g = st.session_state.geo_selected
        lat = float(g["latitude"])
        lon = float(g["longitude"])

        display_name = (g.get("display_name") or "").strip()
        if display_name:
            label = display_name
        else:
            label = f'{g.get("name","")} {g.get("admin1","")} {g.get("country","")}'.strip()

        label = " ".join(label.split())
        if len(label) > 80:
            label = label[:77] + "..."

        tz_name = (g.get("timezone") or "").strip() or None
        return label, lat, lon, tz_name

    if selected_fav:
        return selected_fav["label"], float(selected_fav["lat"]), float(selected_fav["lon"]), None

    return DEFAULT_LABEL, DEFAULT_LAT, DEFAULT_LON, DEFAULT_TZ


# Time inputs (NOTE: we’ll display + compute in the selected location’s timezone after Check)
now_fallback = datetime.now(ZoneInfo(DEFAULT_TZ))
today_fallback = now_fallback.date()
max_day = today_fallback + timedelta(days=MAX_FORECAST_DAYS - 1)

if mode == "Now (arrival in X minutes)":
    arrival_min = st.slider("Arriving in (minutes)", 0, 180, 20, 5)
    target_naive = now_fallback + timedelta(minutes=arrival_min)
else:
    d = st.date_input("Date", value=today_fallback, min_value=today_fallback, max_value=max_day)
    next_hour = (now_fallback + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    t = st.time_input("Time", value=next_hour.time())
    target_naive = datetime.combine(d, t)

st.caption("Target time will be interpreted in the selected location’s local timezone after you press **Check**.")


# ----------------------------
# Check computation
# ----------------------------
def compute_hourly_scores_for_window(hourly: dict, times_local: list[datetime], start_idx: int, hours: int, surface_type: str):
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

        verdict, score, _ = wet_assess(
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
        label, lat, lon, tz_hint = resolve_location()

        # Derive timezone from lat/lon (works for Open-Meteo, Nominatim, and pasted coords)
        tz_name = (tz_hint or "").strip() or tz_from_coords(lat, lon)
        tz_loc = ZoneInfo(tz_name)

        now_local = datetime.now(tz_loc)

        # Reinterpret the UI target as location-local time
        if mode == "Now (arrival in X minutes)":
            target_dt = now_local + timedelta(minutes=arrival_min)
        else:
            target_dt = target_naive.replace(tzinfo=tz_loc)

        forecast_days = needed_forecast_days_for(target_dt, now_local)
        wx = fetch_weather(lat, lon, forecast_days=forecast_days)

        # Use timezone from full response (should match tz_from_coords, but trust API if present)
        tz_name = (wx.get("timezone") or tz_name).strip()
        tz_loc = ZoneInfo(tz_name)

        # 15-min block for "Now"
        m15 = wx.get("minutely_15") or {}
        m15_times_raw = m15.get("time") or []
        m15_times_local = parse_times_local(m15_times_raw, tz_loc)
        i_now_m15 = pick_index(m15_times_local, datetime.now(tz_loc)) if m15_times_local else 0
        used_m15_time = m15_times_raw[i_now_m15] if m15_times_raw else ""

        delta_now_min = None
        if m15_times_local:
            delta_now_min = int(round(abs((m15_times_local[i_now_m15] - datetime.now(tz_loc)).total_seconds()) / 60))

        # Hourly for target + planning
        hourly = wx.get("hourly") or {}
        h_times_raw = hourly.get("time") or []
        h_times_local = parse_times_local(h_times_raw, tz_loc)
        i_t_h = pick_index(h_times_local, target_dt) if h_times_local else 0
        used_h_time = h_times_raw[i_t_h] if h_times_raw else ""

        delta_t_min = None
        if h_times_local:
            delta_t_min = int(round(abs((h_times_local[i_t_h] - target_dt).total_seconds()) / 60))

        # NOW (15-min)
        temp_c_now = at(m15, "temperature_2m", i_now_m15)
        rh_now = at(m15, "relative_humidity_2m", i_now_m15)
        wind_kmh_now = at(m15, "wind_speed_10m", i_now_m15)
        precip_now_15 = at(m15, "precipitation", i_now_m15)
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

        # TARGET (hourly)
        temp_c_t = at(hourly, "temperature_2m", i_t_h)
        rh_t = at(hourly, "relative_humidity_2m", i_t_h)
        wind_kmh_t = at(hourly, "wind_speed_10m", i_t_h)
        precip_t_1h = at(hourly, "precipitation", i_t_h)
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

        if precip_last1h is not None:
            reasons_t["Rain / recent moisture"].insert(
                0, f"Target-hour precipitation bucket: {precip_last1h:.2f} mm (preceding 1 hour sum)."
            )

        target_conf = compute_confidence(delta_t_min, dew_source_t, precip_last1h is not None or precip_last3h is not None)

        st.session_state.last_result = {
            "label": label,
            "lat": lat,
            "lon": lon,
            "target_local": target_dt.strftime("%Y-%m-%d %H:%M"),
            "verdict": verdict_t,
            "score": score_t,
            "surface_type": surface_type,
        }

        i_now_h = pick_index(h_times_local, datetime.now(tz_loc)) if h_times_local else 0
        window = compute_hourly_scores_for_window(hourly, h_times_local, start_idx=i_now_h, hours=12, surface_type=surface_type)
        st.session_state.last_check_payload = {
            "hourly_window": window,
            "label": label,
            "lat": lat,
            "lon": lon,
            "timezone": tz_name,
        }

        st.session_state.debug_safe = {
            "timezone_used": tz_name,
            "now_local": datetime.now(tz_loc).strftime("%Y-%m-%d %H:%M %Z"),
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

        # Display
        st.subheader(f"📍 {label}")
        st.caption(f"Local time zone for this location: **{tz_name}**")

        with st.expander("🗒️ Rink notes (latest)", expanded=False):
            try:
                notes = read_recent_notes(label, limit=5)
                if not notes:
                    st.write("No notes yet.")
                else:
                    for n in notes:
                        st.write(f"- **{n['note_type']}** — {n['note_text']}  _(UTC: {n['ts_utc']})_")
            except Exception as e:
                    st.error("Notes not available. Exact error:")
                    st.code(str(e))


            st.divider()
            st.write("Add a note (helps everyone):")

            nt = st.selectbox(
                "Note type",
                ["Irrigation", "Shade", "Drainage", "Sticky spot", "Other"],
                index=0,
                key=f"note_type::{label}",
            )
            ntext = st.text_input(
                "Note",
                placeholder="e.g., 'Back corner stays wet after rain' (keep it short)",
                key=f"note_text::{label}",
            )
            if st.button("Submit note", key=f"submit_note::{label}"):
                if not ntext.strip():
                    st.warning("Type a note first.")
                else:
                    try:
                        append_note(label, nt, ntext.strip())
                        st.success("Note added. Thanks.")
                        st.session_state[f"note_text::{label}"] = ""
                    except Exception as e:
                        st.error("Couldn’t write note. Exact error:")
                        st.code(str(e))

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

        st.write("**Why (target):**")
        for group, items in reasons_t.items():
            if not items:
                continue
            st.write(f"**{group}:**")
            for it in items:
                st.write(f"- {it}")

        if used_h_time:
            st.caption(f"Target forecast hour used (local): {used_h_time}  (match: ±{delta_t_min} min)")

        payload = st.session_state.get("last_check_payload") or {}
        window = payload.get("hourly_window") or []
        if window:
            st.subheader("📈 Next 12 hours risk trend")
            scores = [w["score"] for w in window]
            st.line_chart({"Wet risk (0–100)": scores})

        st.subheader("🕒 When will it be driest?")
        if st.button("Find next driest window"):
            if not window:
                st.info("Run a check first.")
            else:
                ranked = sorted(window, key=lambda x: x["score"])
                top = ranked[:2]
                st.write("Best upcoming windows (lowest risk):")
                for tbest in top:
                    st.write(f"- **{tbest['dt'].strftime('%a %H:%M')}** — {tbest['verdict']} ({tbest['score']}/100)")

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
        st.caption(f"Last check: {last['label']} @ {last['target_local']} → {last['verdict']} ({last['score']}/100)")

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

        if stats.get("cm"):
            with st.expander("📊 Model report (observed vs predicted)"):
                st.write("Rows = predicted, Columns = observed")
                cm = stats["cm"]
                st.write({"pred_dry": cm["dry"], "pred_damp": cm["damp"], "pred_wet": cm["wet"]})
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



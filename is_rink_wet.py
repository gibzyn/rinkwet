# is_rink_wet.py
import math
import json
import time
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# Google Sheets feedback
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
    """
    Lightweight retry wrapper for transient network/API hiccups.
    """
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
# Google Sheets feedback
# ----------------------------
SCOPES = [
    # Least-privilege note:
    # - If your service account has direct access to the Sheet, this is often enough.
    "https://www.googleapis.com/auth/spreadsheets",
    # Keep Drive scope if your environment needs it for open_by_key / file metadata.
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource
def get_worksheet():
    """
    Uses Streamlit Secrets:
      - st.secrets["gcp_service_account"]  (JSON string OR TOML dict)
      - st.secrets["sheet_id"]             (Google Sheet ID)
    Returns a worksheet, preferring tab name 'feedback', then 'Sheet1', else first tab.
    """
    sa_info = st.secrets["gcp_service_account"]
    if isinstance(sa_info, str):
        sa_info = json.loads(sa_info)

    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["sheet_id"])

    for name in ("feedback", "Sheet1"):
        try:
            return sh.worksheet(name)
        except Exception:
            pass
    return sh.get_worksheet(0)


def ensure_header():
    """
    Ensures the first row has the expected headers.
    Expected: ts_utc,label,target_time_local,verdict,score,thumbs
    """
    ws = get_worksheet()
    values = ws.get_all_values()
    expected = ["ts_utc", "label", "target_time_local", "verdict", "score", "thumbs"]

    if not values:
        ws.append_row(expected, value_input_option="RAW")
        return

    header = [h.strip() for h in values[0]]
    if header != expected:
        # Don't overwrite user sheet automatically; just leave it.
        pass


def append_vote(label: str, target_local: str, verdict: str, score: int, thumbs: str):
    ensure_header()
    ws = get_worksheet()
    ts_utc = datetime.utcnow().isoformat(timespec="seconds")
    ws.append_row([ts_utc, label, target_local, verdict, int(score), thumbs], value_input_option="RAW")


@st.cache_data(ttl=30)
def read_stats():
    ensure_header()
    ws = get_worksheet()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return {"total": 0, "up": 0, "down": 0, "thumbs_up_rate": None}

    header = values[0]
    try:
        thumbs_idx = header.index("thumbs")
    except ValueError:
        thumbs_idx = len(header) - 1

    up = down = total = 0
    for row in values[1:]:
        if not row:
            continue
        total += 1
        t = row[thumbs_idx].strip().lower() if thumbs_idx < len(row) else ""
        if t == "up":
            up += 1
        elif t == "down":
            down += 1

    thumbs_up_rate = (up / total) * 100 if total else None
    return {"total": total, "up": up, "down": down, "thumbs_up_rate": thumbs_up_rate}


def refresh_stats():
    read_stats.clear()


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
    Force Open-Meteo to return times in America/Los_Angeles so they match rink local time.

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
        # Use 15-min data for a more consistent "Now" readout
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


def hourly_at(block: dict, key: str, idx: int):
    arr = (block or {}).get(key) or []
    return arr[idx] if 0 <= idx < len(arr) else None


def parse_times_local(times: list[str]):
    """
    Open-Meteo with timezone=America/Los_Angeles returns times like '2026-01-30T08:00'
    (no offset). Interpret as rink-local timezone-aware datetimes.
    """
    out = []
    for t in times or []:
        dt_naive = datetime.fromisoformat(t)  # naive
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


def sum_recent_precip(arr: list, end_idx: int, lookback: int):
    """
    Sum precipitation over [end_idx-lookback+1 .. end_idx], skipping None.
    """
    if not arr:
        return 0.0
    s = 0.0
    start = max(0, end_idx - lookback + 1)
    for j in range(start, end_idx + 1):
        v = arr[j] if j < len(arr) else None
        if v is None:
            continue
        try:
            s += float(v)
        except Exception:
            continue
    return s


def wet_assess(
    temp_f,
    dew_f,
    rh,
    wind_mph,
    precip_recent_mm: float,
    precip_unit_label: str,
    is_day: int | None,
    matched_time_delta_minutes: int | None,
):
    """
    precip_recent_mm: sum over a recent window (e.g., last 15 min, last 3 hours)
    precip_unit_label: "15 min" or "3 hours" or similar, used for explanation
    """
    reasons = []
    score = 0

    # Recent precipitation strongly suggests wet surface
    if precip_recent_mm is not None:
        if precip_recent_mm >= 1.0:
            score += 60
            reasons.append(
                f"Recent precipitation (~{precip_recent_mm:.2f} mm over the last {precip_unit_label}) strongly suggests a wet surface."
            )
        elif precip_recent_mm >= 0.2:
            score += 40
            reasons.append(
                f"Some recent precipitation (~{precip_recent_mm:.2f} mm over the last {precip_unit_label}) suggests the surface may be wet."
            )
        else:
            reasons.append(
                f"Little/no recent precipitation (~{precip_recent_mm:.2f} mm over the last {precip_unit_label})."
            )
    else:
        reasons.append("Missing precipitation data, so recent-rain wet risk is less certain.")

    # Dew/condensation risk from temp–dewpoint spread
    if temp_f is not None and dew_f is not None:
        spread = temp_f - dew_f
        if spread <= 2:
            score += 55
            reasons.append(f"Very tight temp–dewpoint spread ({spread:.1f}°F) = high dew/condensation risk.")
        elif spread <= 5:
            score += 35
            reasons.append(f"Small temp–dewpoint spread ({spread:.1f}°F) = moderate dew risk.")
        elif spread <= 8:
            score += 18
            reasons.append(f"Moderate temp–dewpoint spread ({spread:.1f}°F) = mild dew risk.")
        else:
            reasons.append(f"Wide temp–dewpoint spread ({spread:.1f}°F) = low dew risk.")
    else:
        reasons.append("Missing temperature or dew point, so dew/condensation risk is less certain.")

    # Humidity supports risk
    if rh is not None:
        if rh >= 95:
            score += 18
            reasons.append(f"Humidity is extremely high ({rh:.0f}%).")
        elif rh >= 85:
            score += 10
            reasons.append(f"Humidity is high ({rh:.0f}%).")
        elif rh <= 60:
            score -= 6
            reasons.append(f"Humidity is moderate/low ({rh:.0f}%), helping the surface stay drier.")

    # Night factor (surfaces can radiatively cool, increasing dew formation risk)
    # is_day is 1 (day) or 0 (night) per Open-Meteo.
    if is_day is not None:
        if int(is_day) == 0 and rh is not None and rh >= 80:
            score += 8
            reasons.append("Nighttime + high humidity increases dew/condensation risk on the surface.")

    # Wind dries
    if wind_mph is not None:
        if wind_mph >= 10:
            score -= 14
            reasons.append(f"Wind is decent ({wind_mph:.1f} mph), which helps the rink dry.")
        elif wind_mph <= 3:
            score += 8
            reasons.append(f"Wind is very light ({wind_mph:.1f} mph), so moisture lingers.")

    # Time match confidence
    if matched_time_delta_minutes is not None:
        if matched_time_delta_minutes <= 15:
            reasons.append(f"Forecast time match is tight (within {matched_time_delta_minutes} minutes).")
        else:
            score += 4  # slight uncertainty bump
            reasons.append(f"Forecast time match is looser (within {matched_time_delta_minutes} minutes), adding uncertainty.")

    score = max(0, min(100, int(round(score))))

    if score >= 65:
        verdict = "YES — likely wet"
    elif score >= 45:
        verdict = "MAYBE — likely damp/borderline"
    else:
        verdict = "NO — likely dry"

    return verdict, score, reasons


# ----------------------------
# UI
# ----------------------------
st.title("🏒 RinkWet")
st.caption("Forecast-based estimate for wet rink conditions (dew/condensation + recent rain + wind).")

mode = st.radio("Check for", ["Now (arrival in X minutes)", "Pick a date & time"], horizontal=True)

use_default = st.toggle("Use Freedom Park default", value=True)
place = "" if use_default else st.text_input("Other location", placeholder="e.g., 'Ventura, CA'")

now_local = datetime.now(RINK_TZ)
today_local = now_local.date()
max_day = today_local + timedelta(days=MAX_FORECAST_DAYS - 1)

if mode == "Now (arrival in X minutes)":
    arrival_min = st.slider("Arriving in (minutes)", 0, 180, 20, 5)
    target_dt = now_local + timedelta(minutes=arrival_min)
else:
    d = st.date_input("Date", value=today_local, min_value=today_local, max_value=max_day)
    # default time: next full hour in rink local time
    next_hour = (now_local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    t = st.time_input("Time", value=next_hour.time())
    target_dt = datetime.combine(d, t).replace(tzinfo=RINK_TZ)

st.caption(f"Target time (America/Los_Angeles): {target_dt.strftime('%Y-%m-%d %H:%M')}")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "debug_safe" not in st.session_state:
    st.session_state.debug_safe = None


if st.button("Check"):
    try:
        # Location
        if use_default:
            lat, lon = DEFAULT_LAT, DEFAULT_LON
            label = DEFAULT_LABEL
        else:
            if not place.strip():
                st.error("Enter a location or turn on Freedom Park default.")
                st.stop()

            results = geocode(place.strip())
            if not results:
                st.error("Couldn't geocode that location. Try 'City, State' or a fuller place name.")
                st.stop()

            best = results[0]
            lat, lon = best["latitude"], best["longitude"]
            label = f'{best.get("name","")} {best.get("admin1","")} {best.get("country","")}'.strip()

        forecast_days = needed_forecast_days_for(target_dt)
        wx = fetch_weather(lat, lon, forecast_days=forecast_days)

        current = wx.get("current") or {}

        # 15-minute block for "Now"
        m15 = wx.get("minutely_15") or {}
        m15_times_raw = m15.get("time") or []
        m15_times_local = parse_times_local(m15_times_raw)

        # hourly block for "Target"
        hourly = wx.get("hourly") or {}
        h_times_raw = hourly.get("time") or []
        h_times_local = parse_times_local(h_times_raw)

        # Indexes
        i_now_m15 = pick_index(m15_times_local, datetime.now(RINK_TZ)) if m15_times_local else 0
        i_t_h = pick_index(h_times_local, target_dt) if h_times_local else 0

        used_m15_time = m15_times_raw[i_now_m15] if m15_times_raw else ""
        used_h_time = h_times_raw[i_t_h] if h_times_raw else ""

        # Time deltas for match confidence
        delta_now_min = None
        if m15_times_local:
            delta_now_min = int(round(abs((m15_times_local[i_now_m15] - datetime.now(RINK_TZ)).total_seconds()) / 60))

        delta_t_min = None
        if h_times_local:
            delta_t_min = int(round(abs((h_times_local[i_t_h] - target_dt).total_seconds()) / 60))

        # ----------------------------
        # NOW (use minutely_15 consistently)
        # ----------------------------
        temp_c_now = hourly_at(m15, "temperature_2m", i_now_m15)
        rh_now = hourly_at(m15, "relative_humidity_2m", i_now_m15)
        wind_kmh_now = hourly_at(m15, "wind_speed_10m", i_now_m15)
        precip_now_15 = hourly_at(m15, "precipitation", i_now_m15)  # preceding 15 minutes sum (mm)
        dew_c_now = hourly_at(m15, "dew_point_2m", i_now_m15)
        is_day_now = hourly_at(m15, "is_day", i_now_m15)

        if dew_c_now is None and temp_c_now is not None and rh_now is not None:
            dew_c_now = compute_dewpoint_c(temp_c_now, rh_now)

        temp_f_now = c_to_f(temp_c_now)
        dew_f_now = c_to_f(dew_c_now)
        wind_mph_now = None if wind_kmh_now is None else wind_kmh_now * 0.621371

        verdict_now, score_now, reasons_now = wet_assess(
            temp_f_now,
            dew_f_now,
            rh_now,
            wind_mph_now,
            precip_recent_mm=(float(precip_now_15) if precip_now_15 is not None else None),
            precip_unit_label="15 min",
            is_day=is_day_now,
            matched_time_delta_minutes=delta_now_min,
        )

        # ----------------------------
        # TARGET (nearest hourly forecast; incorporate recent rain window)
        # ----------------------------
        temp_c_t = hourly_at(hourly, "temperature_2m", i_t_h)
        rh_t = hourly_at(hourly, "relative_humidity_2m", i_t_h)
        wind_kmh_t = hourly_at(hourly, "wind_speed_10m", i_t_h)
        precip_t_1h = hourly_at(hourly, "precipitation", i_t_h)  # preceding hour sum (mm)
        dew_c_t = hourly_at(hourly, "dew_point_2m", i_t_h)
        is_day_t = hourly_at(hourly, "is_day", i_t_h)

        if dew_c_t is None and temp_c_t is not None and rh_t is not None:
            dew_c_t = compute_dewpoint_c(temp_c_t, rh_t)

        temp_f_t = c_to_f(temp_c_t)
        dew_f_t = c_to_f(dew_c_t)
        wind_mph_t = None if wind_kmh_t is None else wind_kmh_t * 0.621371

        # Recent rain: sum last 3 hours INCLUDING the target hour bucket
        precip_arr = (hourly or {}).get("precipitation") or []
        precip_last3h = sum_recent_precip(precip_arr, i_t_h, lookback=3)
        # Also keep the target hour bucket for display
        precip_last1h = None if precip_t_1h is None else float(precip_t_1h)

        verdict_t, score_t, reasons_t = wet_assess(
            temp_f_t,
            dew_f_t,
            rh_t,
            wind_mph_t,
            precip_recent_mm=(precip_last3h if precip_arr else (precip_last1h if precip_last1h is not None else None)),
            precip_unit_label=("3 hours" if precip_arr else "1 hour"),
            is_day=is_day_t,
            matched_time_delta_minutes=delta_t_min,
        )

        # Add explicit note about the target hour bucket (helpful for debugging/trust)
        if precip_last1h is not None:
            reasons_t.insert(0, f"Target-hour precipitation bucket: ~{precip_last1h:.2f} mm (preceding 1 hour sum).")

        # Save for feedback buttons
        st.session_state.last_result = {
            "label": label,
            "target_local": target_dt.strftime("%Y-%m-%d %H:%M"),
            "verdict": verdict_t,
            "score": score_t,
        }

        # Save safe debug info for bottom Debug expander
        st.session_state.debug_safe = {
            "now_local": datetime.now(RINK_TZ).strftime("%Y-%m-%d %H:%M %Z"),
            "target_dt": target_dt.strftime("%Y-%m-%d %H:%M %Z"),
            "m15_index_now": i_now_m15,
            "hourly_index_target": i_t_h,
            "used_m15_time": used_m15_time,
            "used_hourly_time": used_h_time,
            "match_delta_now_min": delta_now_min,
            "match_delta_target_min": delta_t_min,
            "label": label,
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
        }

        # ----------------------------
        # Display
        # ----------------------------
        st.subheader(f"📍 {label}")

        st.write("**Now (15-minute snapshot)**")
        cols = st.columns(4)
        cols[0].metric("Temp", "—" if temp_f_now is None else f"{temp_f_now:.1f}°F")
        cols[1].metric("Dew point", "—" if dew_f_now is None else f"{dew_f_now:.1f}°F")
        cols[2].metric("Humidity", "—" if rh_now is None else f"{rh_now:.0f}%")
        cols[3].metric("Wind", "—" if wind_mph_now is None else f"{wind_mph_now:.1f} mph")

        # Extra now precipitation + match delta
        subcols_now = st.columns(2)
        subcols_now[0].metric(
            "Precip (last 15 min)",
            "—" if precip_now_15 is None else f"{float(precip_now_15):.2f} mm",
        )
        subcols_now[1].metric(
            "Time match",
            "—" if delta_now_min is None else f"±{delta_now_min} min",
        )

        st.write("**Target (nearest hourly forecast)**")
        cols2 = st.columns(4)
        cols2[0].metric("Temp", "—" if temp_f_t is None else f"{temp_f_t:.1f}°F")
        cols2[1].metric("Dew point", "—" if dew_f_t is None else f"{dew_f_t:.1f}°F")
        cols2[2].metric("Humidity", "—" if rh_t is None else f"{rh_t:.0f}%")
        cols2[3].metric("Wind", "—" if wind_mph_t is None else f"{wind_mph_t:.1f} mph")

        # Extra target precipitation + match delta
        subcols_t = st.columns(3)
        subcols_t[0].metric(
            "Precip (target hr)",
            "—" if precip_last1h is None else f"{precip_last1h:.2f} mm",
        )
        subcols_t[1].metric(
            "Precip (last 3 hrs)",
            "—" if precip_arr is None else f"{precip_last3h:.2f} mm",
        )
        subcols_t[2].metric(
            "Time match",
            "—" if delta_t_min is None else f"±{delta_t_min} min",
        )

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

        with colB:
            st.write("### Target verdict")
            if score_t >= 65:
                st.error(f"{verdict_t} (Risk: {score_t}/100)")
            elif score_t >= 45:
                st.warning(f"{verdict_t} (Risk: {score_t}/100)")
            else:
                st.success(f"{verdict_t} (Risk: {score_t}/100)")

        st.write("**Why (target):**")
        for r in reasons_t:
            st.write(f"- {r}")

        if used_m15_time:
            st.caption(f"Now data timestamp used (America/Los_Angeles): {used_m15_time}")
        if used_h_time:
            st.caption(f"Target forecast hour used (America/Los_Angeles): {used_h_time}")

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
    stats = read_stats()
    a, b, c = st.columns(3)
    a.metric("Total votes", stats["total"])
    b.metric("👍", stats["up"])
    c.metric("Thumbs-up rate", "—" if stats["thumbs_up_rate"] is None else f"{stats['thumbs_up_rate']:.0f}%")

    last = st.session_state.get("last_result")
    if not last:
        st.info("Run a check first, then vote 👍 or 👎 based on what you actually saw at the rink.")
    else:
        col1, col2 = st.columns(2)
        if col1.button("👍 Accurate"):
            append_vote(last["label"], last["target_local"], last["verdict"], last["score"], "up")
            refresh_stats()
            st.success("Logged 👍. Thank you for your feedback.")

        if col2.button("👎 Not accurate"):
            append_vote(last["label"], last["target_local"], last["verdict"], last["score"], "down")
            refresh_stats()
            st.success("Logged 👎. Thank you for your feedback.")

        st.caption(
            f"Last check: {last['label']} @ {last['target_local']} → {last['verdict']} ({last['score']}/100)"
        )

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
    "shade, drainage, or microclimate. Use at your own risk."
)

# is_rink_wet.py
import math
import json
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
# Google Sheets feedback
# ----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
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
        # If you want strict enforcement, delete row 1 and re-add.
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
        return {"total": 0, "up": 0, "down": 0, "accuracy": None}

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

    accuracy = (up / total) * 100 if total else None
    return {"total": total, "up": up, "down": down, "accuracy": accuracy}


def refresh_stats():
    read_stats.clear()


# ----------------------------
# Weather + scoring helpers
# ----------------------------
def geocode(query: str):
    r = requests.get(
        OPEN_METEO_GEOCODE,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    return (r.json().get("results") or [])


def fetch_weather(lat: float, lon: float, forecast_days: int):
    """
    Force Open-Meteo to return hourly times in America/Los_Angeles so they match your rink local time.
    IMPORTANT: Do NOT include 'time' inside hourly param list (Open-Meteo returns it automatically).
    """
    forecast_days = max(1, min(MAX_FORECAST_DAYS, int(forecast_days)))

    r = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m",
            "forecast_days": forecast_days,
            # THIS is the key: make the API give times in rink-local TZ
            "timezone": "America/Los_Angeles",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def c_to_f(c):
    return None if c is None else (c * 9 / 5) + 32


def compute_dewpoint_c(temp_c: float, rh: float):
    # Magnus approximation
    a, b = 17.62, 243.12
    gamma = (a * temp_c / (b + temp_c)) + math.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def hourly_at(hourly: dict, key: str, idx: int):
    arr = (hourly or {}).get(key) or []
    return arr[idx] if 0 <= idx < len(arr) else None


def parse_hourly_times(times: list[str]):
    """
    Open-Meteo with timezone=America/Los_Angeles returns times like '2026-01-30T08:00'
    (no offset). We interpret those as rink-local timezone-aware datetimes.
    """
    out = []
    for t in times or []:
        dt_naive = datetime.fromisoformat(t)  # naive
        out.append(dt_naive.replace(tzinfo=RINK_TZ))
    return out


def pick_hour_index(hourly_times_local: list[datetime], target_dt_local: datetime) -> int:
    if not hourly_times_local:
        return 0
    return min(
        range(len(hourly_times_local)),
        key=lambda i: abs((hourly_times_local[i] - target_dt_local).total_seconds()),
    )


def needed_forecast_days_for(target_dt_local: datetime) -> int:
    today_local = datetime.now(RINK_TZ).date()
    delta_days = (target_dt_local.date() - today_local).days
    return max(1, min(MAX_FORECAST_DAYS, delta_days + 2))


def wet_assess(temp_f, dew_f, rh, wind_mph, precip_mm_hr):
    reasons = []
    score = 0

    # Precip overrides
    if precip_mm_hr is not None and precip_mm_hr > 0:
        score += 55
        reasons.append(f"Active precipitation (~{precip_mm_hr:.2f} mm/hr) strongly suggests a wet surface.")

    # Dew/condensation risk from spread
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

    # Wind dries
    if wind_mph is not None:
        if wind_mph >= 10:
            score -= 14
            reasons.append(f"Wind is decent ({wind_mph:.1f} mph), which helps the rink dry.")
        elif wind_mph <= 3:
            score += 8
            reasons.append(f"Wind is very light ({wind_mph:.1f} mph), so moisture lingers.")

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
st.caption("Forecast-based estimate for wet rink conditions (dew/condensation + rain + wind).")

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
        hourly = wx.get("hourly") or {}
        times_raw = hourly.get("time") or []
        times_local = parse_hourly_times(times_raw)

        # Indexes
        i_now = pick_hour_index(times_local, datetime.now(RINK_TZ)) if times_local else 0
        i_t = pick_hour_index(times_local, target_dt) if times_local else 0
        used_hour = times_raw[i_t] if times_raw else ""

        # NOW (current metrics + nearest hourly dew)
        temp_c_now = current.get("temperature_2m")
        rh_now = current.get("relative_humidity_2m")
        wind_kmh_now = current.get("wind_speed_10m")
        precip_now = current.get("precipitation")

        dew_c_now = hourly_at(hourly, "dew_point_2m", i_now)
        if dew_c_now is None and temp_c_now is not None and rh_now is not None:
            dew_c_now = compute_dewpoint_c(temp_c_now, rh_now)

        temp_f_now = c_to_f(temp_c_now)
        dew_f_now = c_to_f(dew_c_now)
        wind_mph_now = None if wind_kmh_now is None else wind_kmh_now * 0.621371

        verdict_now, score_now, _ = wet_assess(
            temp_f_now, dew_f_now, rh_now, wind_mph_now, precip_now
        )

        # TARGET (nearest hourly forecast)
        temp_c_t = hourly_at(hourly, "temperature_2m", i_t)
        rh_t = hourly_at(hourly, "relative_humidity_2m", i_t)
        wind_kmh_t = hourly_at(hourly, "wind_speed_10m", i_t)
        precip_t = hourly_at(hourly, "precipitation", i_t)
        dew_c_t = hourly_at(hourly, "dew_point_2m", i_t)

        if dew_c_t is None and temp_c_t is not None and rh_t is not None:
            dew_c_t = compute_dewpoint_c(temp_c_t, rh_t)

        temp_f_t = c_to_f(temp_c_t)
        dew_f_t = c_to_f(dew_c_t)
        wind_mph_t = None if wind_kmh_t is None else wind_kmh_t * 0.621371

        verdict_t, score_t, reasons_t = wet_assess(
            temp_f_t, dew_f_t, rh_t, wind_mph_t, precip_t
        )

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
            "i_now": i_now,
            "i_t": i_t,
            "used_hour": used_hour,
            "label": label,
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
        }

        # Display
        st.subheader(f"📍 {label}")

        st.write("**Now (current)**")
        cols = st.columns(4)
        cols[0].metric("Temp", "—" if temp_f_now is None else f"{temp_f_now:.1f}°F")
        cols[1].metric("Dew point", "—" if dew_f_now is None else f"{dew_f_now:.1f}°F")
        cols[2].metric("Humidity", "—" if rh_now is None else f"{rh_now:.0f}%")
        cols[3].metric("Wind", "—" if wind_mph_now is None else f"{wind_mph_now:.1f} mph")

        st.write("**Target (nearest hourly forecast)**")
        cols2 = st.columns(4)
        cols2[0].metric("Temp", "—" if temp_f_t is None else f"{temp_f_t:.1f}°F")
        cols2[1].metric("Dew point", "—" if dew_f_t is None else f"{dew_f_t:.1f}°F")
        cols2[2].metric("Humidity", "—" if rh_t is None else f"{rh_t:.0f}%")
        cols2[3].metric("Wind", "—" if wind_mph_t is None else f"{wind_mph_t:.1f} mph")

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

        if used_hour:
            st.caption(f"Target forecast hour used (America/Los_Angeles): {used_hour}")

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
    c.metric("Accuracy", "—" if stats["accuracy"] is None else f"{stats['accuracy']:.0f}%")

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



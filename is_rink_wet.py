import math
import os
import csv
from datetime import datetime, timedelta

import requests
import streamlit as st

st.set_page_config(page_title="RinkWet", page_icon="🛼", layout="centered")

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_LABEL = "Freedom Park Inline Hockey Arena (Camarillo)"
DEFAULT_LAT = 34.2138
DEFAULT_LON = -119.0856

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"

MAX_FORECAST_DAYS = 16  # safe cap

# Local feedback file (desktop/local)
FEEDBACK_FILE = "feedback.csv"


# ----------------------------
# Feedback helpers (local CSV)
# ----------------------------
def save_feedback(verdict: str, score: int, label: str, target_iso: str, thumbs: str):
    """
    thumbs: 'up' or 'down'
    Writes one row to feedback.csv (local file).
    """
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["ts_utc", "label", "target_time", "verdict", "score", "thumbs"])
        w.writerow(
            [
                datetime.utcnow().isoformat(timespec="seconds"),
                label,
                target_iso,
                verdict,
                score,
                thumbs,
            ]
        )


def load_feedback_stats():
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "up": 0, "down": 0, "accuracy": None}

    up = down = total = 0
    with open(FEEDBACK_FILE, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            if row.get("thumbs") == "up":
                up += 1
            elif row.get("thumbs") == "down":
                down += 1

    accuracy = (up / total) * 100 if total else None
    return {"total": total, "up": up, "down": down, "accuracy": accuracy}


# ----------------------------
# Weather + computation
# ----------------------------
def geocode(query: str):
    r = requests.get(
        OPEN_METEO_GEOCODE,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    return (r.json().get("results") or [])


def fetch_weather(lat: float, lon: float, forecast_days: int = 2):
    """
    IMPORTANT: Do NOT include 'time' inside hourly variable list.
    Open-Meteo returns hourly['time'] automatically.
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
            "timezone": "auto",
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


def pick_hour_index(hourly_times: list[str], target_dt: datetime) -> int:
    if not hourly_times:
        return 0
    dts = [datetime.fromisoformat(t) for t in hourly_times]
    return min(range(len(dts)), key=lambda i: abs((dts[i] - target_dt).total_seconds()))


def needed_forecast_days_for(target_dt: datetime) -> int:
    today = datetime.now().date()
    delta_days = (target_dt.date() - today).days
    return max(1, min(MAX_FORECAST_DAYS, delta_days + 2))


def wet_assess(temp_f, dew_f, rh, wind_mph, precip_mm_hr):
    """
    Returns: (verdict_str, score_int, reasons_list)
    """
    reasons = []
    score = 0

    # Precip = wet
    if precip_mm_hr is not None and precip_mm_hr > 0:
        score += 55
        reasons.append(f"Active precipitation (~{precip_mm_hr:.2f} mm/hr) strongly suggests a wet surface.")

    # Dew/condensation risk
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
st.title("🛼 RinkWet")
st.caption("Forecast-based estimate for wet rink conditions (dew/condensation + rain + wind).")

mode = st.radio("Check for", ["Now (arrival in X minutes)", "Pick a date & time"], horizontal=True)

use_default = st.toggle("Use Freedom Park default", value=True)
place = "" if use_default else st.text_input("Other location (optional)", placeholder="e.g., 'Ventura, CA'")

today = datetime.now().date()
max_day = today + timedelta(days=MAX_FORECAST_DAYS - 1)

if mode == "Now (arrival in X minutes)":
    arrival_min = st.slider("Arriving in (minutes)", 0, 180, 20, 5)
    target_dt = datetime.now() + timedelta(minutes=arrival_min)
else:
    d = st.date_input("Date", value=today, min_value=today, max_value=max_day)
    t = st.time_input(
        "Time",
        value=(datetime.now() + timedelta(hours=1)).time().replace(second=0, microsecond=0),
    )
    target_dt = datetime.combine(d, t)

st.caption(f"Target time: {target_dt.strftime('%Y-%m-%d %H:%M')} (local)")


# Keep these in session state so feedback can reference last result
if "last_result" not in st.session_state:
    st.session_state.last_result = None


if st.button("Check"):
    try:
        # Location selection
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
        times = hourly.get("time") or []

        # NOW (current)
        temp_c_now = current.get("temperature_2m")
        rh_now = current.get("relative_humidity_2m")
        wind_kmh_now = current.get("wind_speed_10m")
        precip_mm_now = current.get("precipitation")

        i_now = pick_hour_index(times, datetime.now()) if times else 0
        dew_c_now = hourly_at(hourly, "dew_point_2m", i_now)

        if dew_c_now is None and temp_c_now is not None and rh_now is not None:
            dew_c_now = compute_dewpoint_c(temp_c_now, rh_now)

        temp_f_now = c_to_f(temp_c_now)
        dew_f_now = c_to_f(dew_c_now)
        wind_mph_now = None if wind_kmh_now is None else wind_kmh_now * 0.621371

        if precip_mm_now is None:
            precip_mm_now = hourly_at(hourly, "precipitation", i_now)

        verdict_now, score_now, _ = wet_assess(temp_f_now, dew_f_now, rh_now, wind_mph_now, precip_mm_now)

        # TARGET (hourly)
        i = pick_hour_index(times, target_dt) if times else 0

        temp_c_t = hourly_at(hourly, "temperature_2m", i)
        rh_t = hourly_at(hourly, "relative_humidity_2m", i)
        wind_kmh_t = hourly_at(hourly, "wind_speed_10m", i)
        precip_mm_t = hourly_at(hourly, "precipitation", i)
        dew_c_t = hourly_at(hourly, "dew_point_2m", i)

        if dew_c_t is None and temp_c_t is not None and rh_t is not None:
            dew_c_t = compute_dewpoint_c(temp_c_t, rh_t)

        temp_f_t = c_to_f(temp_c_t)
        dew_f_t = c_to_f(dew_c_t)
        wind_mph_t = None if wind_kmh_t is None else wind_kmh_t * 0.621371

        verdict_t, score_t, reasons_t = wet_assess(temp_f_t, dew_f_t, rh_t, wind_mph_t, precip_mm_t)

        # Save for feedback
        st.session_state.last_result = {
            "label": label,
            "target_iso": target_dt.strftime("%Y-%m-%d %H:%M"),
            "verdict": verdict_t,
            "score": score_t,
        }

        # Display
        st.subheader(f"📍 {label}")

        st.write("**Now (current)**")
        cols = st.columns(4)
        cols[0].metric("Temp", "—" if temp_f_now is None else f"{temp_f_now:.1f}°F")
        cols[1].metric("Dew point", "—" if dew_f_now is None else f"{dew_f_now:.1f}°F")
        cols[2].metric("Humidity", "—" if rh_now is None else f"{rh_now:.0f}%")
        cols[3].metric("Wind", "—" if wind_mph_now is None else f"{wind_mph_now:.1f} mph")

        st.write(f"**Target ({target_dt.strftime('%a %b %d, %I:%M %p')}, nearest hourly forecast)**")
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

        if times:
            st.caption(f"Target forecast hour used: {times[i]} (index {i})")

    except requests.RequestException as e:
        st.error(f"Network/API error: {e}")
    except Exception as e:
        st.error(f"App error: {e}")


# ----------------------------
# Feedback UI (shows even without re-check)
# ----------------------------
st.divider()
st.subheader("✅ Was the prediction accurate?")

stats = load_feedback_stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total votes", stats["total"])
c2.metric("👍", stats["up"])
c3.metric("Accuracy", "—" if stats["accuracy"] is None else f"{stats['accuracy']:.0f}%")

last = st.session_state.last_result
if not last:
    st.info("Run a check first, then vote 👍 or 👎 based on what you actually saw at the rink.")
else:
    left, right = st.columns(2)

    if left.button("👍 Accurate"):
        save_feedback(last["verdict"], last["score"], last["label"], last["target_iso"], "up")
        st.success("Thanks — logged 👍")

    if right.button("👎 Not accurate"):
        save_feedback(last["verdict"], last["score"], last["label"], last["target_iso"], "down")
        st.warning("Logged 👎")

    st.caption(f"Last check: {last['label']} @ {last['target_iso']} → {last['verdict']} ({last['score']}/100)")


# Footer / disclaimer
st.caption("Disclaimer: Forecast-based estimate only. Outdoor surfaces can vary by shade, irrigation, and microclimate.")

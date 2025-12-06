
"""
climate_ai_agent.py

AI-style climate agent for OWID global monthly temperature anomalies.

Behavior:
- Downloads OWID monthly-temperature-anomalies.csv
- Filters for "World"
- Uses 'Day' as the date column
- Uses OWID's own anomaly (vs 1991–2020) for simplicity
- Checks if there is any new data since the last run (stored in agent_state.json)
- If new data exists:
    - Fits a linear trend vs fractional year
    - Forecasts next few years
    - Saves a plot, numeric summary, explanatory text
    - Updates the state file
- If no new data:
    - Prints a message and exits quickly

Hook:
- generate_explanation() is where you plug an LLM (ChatGPT, Gemini, etc.)
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
DATA_URL = "https://ourworldindata.org/grapher/monthly-temperature-anomalies.csv"
FORECAST_YEARS = 5
OUTPUT_DIR = "agent_outputs"
STATE_FILE = "agent_state.json"   # remembers last processed date


def load_state(state_file=STATE_FILE):
    """Load previous agent state (last processed date)."""
    if not os.path.exists(state_file):
        return {}
    with open(state_file, "r") as f:
        return json.load(f)


def save_state(state, state_file=STATE_FILE):
    """Save agent state to disk."""
    with open(state_file, "w") as f:
        json.dump(state, f)


def generate_explanation(numeric_summary: str, mode: str = "teacher") -> str:
    """
    Placeholder for AI explanation.

    This is where you plug in an LLM (ChatGPT, Gemini, etc.).
    For now it returns a simple, human-written template so the code runs
    without needing an API key.

    To use an LLM, you might:
    - Send `numeric_summary` plus instructions to the model
    - Get back a richer explanation
    - Return that text here
    """
    if mode == "student":
        return (
            "Student-friendly explanation:\n\n"
            "The summary you see above describes how Earth's average temperature "
            "has changed over time. Focus on three things:\n"
            "1) How warm it is now compared to the past.\n"
            "2) How quickly it is warming each decade.\n"
            "3) What the simple forecast suggests for the next few years.\n\n"
            "Use this summary as a starting point. What questions do you want "
            "to ask about the data or the model?"
        )
    else:
        return (
            "Teacher-friendly explanation:\n\n"
            "The numeric summary above gives the latest global temperature anomaly "
            "relative to the 1991–2020 average, the linear warming rate per decade, "
            "and a naive extrapolation a few years into the future. This can be used "
            "to: (a) illustrate trend vs noise, (b) discuss limitations of linear "
            "models for climate, and (c) invite students to critique assumptions, "
            "try different windows, or compare with scenario-based climate projections."
        )


def run_ai_climate_agent(
    forecast_years: int = FORECAST_YEARS,
    output_dir: str = OUTPUT_DIR,
    state_file: str = STATE_FILE,
) -> dict:
    """
    Main AI-style agent function.

    Returns a dict with:
        {
            "status": "no_new_data" or "updated",
            "numeric_summary": str,
            "explanation": str,
            "plot_path": str or None
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load previous state
    state = load_state(state_file)
    last_processed_str = state.get("last_processed_date", None)
    if last_processed_str is not None:
        last_processed_date = datetime.fromisoformat(last_processed_str).date()
    else:
        last_processed_date = None

    # 2. Download OWID data
    print("Downloading OWID data...")
    df = pd.read_csv(DATA_URL)

    world = df[df["Entity"] == "World"].copy()
    anom_col = "Temperature anomaly"  # OWID's baseline 1991–2020

    # Parse date and build fractional year
    world["date"] = pd.to_datetime(world["Day"])
    world["t_year"] = (
        world["date"].dt.year
        + (world["date"].dt.month - 0.5) / 12.0
    )
    world = world.dropna(subset=[anom_col, "t_year"]).sort_values("date")

    if world.empty:
        raise RuntimeError("No data available for 'World' after cleaning.")

    latest_date = world["date"].max().date()
    print(f"Latest date in OWID data: {latest_date}")

    # 3. Check if there is new data beyond what we already processed
    if last_processed_date is not None and latest_date <= last_processed_date:
        print("No new data since last run. Agent will exit.")
        return {
            "status": "no_new_data",
            "numeric_summary": "",
            "explanation": "",
            "plot_path": None,
        }

    # 4. Fit linear regression on all available data
    X = world[["t_year"]].values
    y = world[anom_col].values

    model = LinearRegression()
    model.fit(X, y)

    slope_per_year = model.coef_[0]
    slope_per_decade = slope_per_year * 10
    intercept = model.intercept_

    # 5. Build forecast
    last_t = world["t_year"].max()
    last_dt = world["date"].max()

    months = forecast_years * 12
    step = 1 / 12

    future_t_year = last_t + step * np.arange(1, months + 1)
    X_future = future_t_year.reshape(-1, 1)
    y_future_pred = model.predict(X_future)

    future_dates = pd.date_range(
        start=last_dt + pd.offsets.MonthBegin(1),
        periods=months,
        freq="MS"
    )

    # 6. Build numeric summary text
    latest_row = world.iloc[-1]
    latest_anom = latest_row[anom_col]
    target_year = last_dt.year + forecast_years
    # pick forecast near mid target_year
    idx_target = (np.abs(future_t_year - (target_year + 0.5))).argmin()
    target_anom = y_future_pred[idx_target]

    numeric_summary = (
        f"Numeric climate summary (OWID, baseline 1991–2020)\n"
        f"-------------------------------------------------\n"
        f"Latest month in dataset: {latest_date}\n"
        f"Latest anomaly: {latest_anom:.2f} °C above 1991–2020 average.\n\n"
        f"Linear warming rate: {slope_per_decade:.2f} °C per decade "
        f"(≈ {slope_per_year:.3f} °C per year).\n\n"
        f"Naive linear forecast around year {target_year}: "
        f"{target_anom:.2f} °C above 1991–2020 average.\n"
    )

    # 7. Generate human explanation (AI hook)
    explanation = generate_explanation(numeric_summary, mode="teacher")

    # 8. Plot and save
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"ai_agent_forecast_{timestamp}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(world["date"], world[anom_col], label="Historical anomaly")
    plt.plot(world["date"], model.predict(X), "--", label="Linear trend")
    plt.plot(future_dates, y_future_pred, ":", label=f"Forecast next {forecast_years} years")
    plt.axhline(0, linewidth=0.8)

    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C vs 1991–2020)")
    plt.title("Global Temperature Anomaly – AI Agent Trend & Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # 9. Update state
    state["last_processed_date"] = latest_date.isoformat()
    save_state(state, state_file)

    # 10. Print summary to console
    print("\n=== NUMERIC SUMMARY ===")
    print(numeric_summary)
    print("\n=== EXPLANATION (AI HOOK) ===")
    print(explanation)
    print(f"\nSaved plot → {plot_path}")
    print(f"State updated with last_processed_date = {latest_date}")

    return {
        "status": "updated",
        "numeric_summary": numeric_summary,
        "explanation": explanation,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_ai_climate_agent()

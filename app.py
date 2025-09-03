#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px


# Set page config
st.set_page_config(
    page_title="Fish Production Analysis",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)
#start by defining the utility functions
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip, collapse spaces, upper-case"""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Extract cage number (int) from mixed labels like 'CAGE 3 A' or 'C3A'"""
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
    """Find a column in df matching one of candidates"""
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

def to_number(x):
    """Convert messy numeric strings (with commas, text) into floats"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# Load data
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None, verbose=True):
    """
    Load + normalize the four input files and coerce key columns.
    Returns dict: feeding, harvest, sampling, transfers
    """
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Feeding
    c = find_col(feeding, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: feeding["CAGE NUMBER"] = to_int_cage(feeding[c])
    fa = find_col(feeding, ["FEED AMOUNT (KG)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED"], "FEED")
    if fa: feeding["FEED AMOUNT (KG)"] = feeding[fa].apply(to_number)
    # parse dates
    if "DATE" in feeding.columns:
        feeding["DATE"] = pd.to_datetime(feeding["DATE"], errors="coerce")

    # Harvest
    c = find_col(harvest, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: harvest["CAGE NUMBER"] = to_int_cage(harvest[c])
    hfish = find_col(harvest, ["NUMBER OF FISH"], "FISH")
    if hfish: harvest["NUMBER OF FISH"] = pd.to_numeric(harvest[hfish].map(to_number), errors="coerce")
    hkg = find_col(harvest, ["TOTAL WEIGHT (KG)", "TOTAL WEIGHT [KG]"], "WEIGHT")
    if hkg: harvest["TOTAL WEIGHT [KG]"] = pd.to_numeric(harvest[hkg].map(to_number), errors="coerce")
    habw = find_col(harvest, ["ABW (G)", "ABW [G]", "ABW(G)", "ABW"], "ABW")
    if habw: harvest["ABW (G)"] = pd.to_numeric(harvest[habw].map(to_number), errors="coerce")
    if "DATE" in harvest.columns:
        harvest["DATE"] = pd.to_datetime(harvest["DATE"], errors="coerce")

    # Sampling
    c = find_col(sampling, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: sampling["CAGE NUMBER"] = to_int_cage(sampling[c])
    sfish = find_col(sampling, ["NUMBER OF FISH"], "FISH")
    if sfish: sampling["NUMBER OF FISH"] = pd.to_numeric(sampling[sfish].map(to_number), errors="coerce")
    sabw = find_col(sampling, ["AVERAGE BODY WEIGHT (G)", "ABW (G)", "ABW [G]", "ABW(G)", "ABW"], "WEIGHT")
    if sabw: sampling["AVERAGE BODY WEIGHT (G)"] = pd.to_numeric(sampling[sabw].map(to_number), errors="coerce")
    if "DATE" in sampling.columns:
        sampling["DATE"] = pd.to_datetime(sampling["DATE"], errors="coerce")

    # Transfers
    if transfers is not None:
        oc = find_col(transfers, ["ORIGIN CAGE", "ORIGIN", "ORIGIN CAGE NUMBER"], "ORIGIN")
        dc = find_col(transfers, ["DESTINATION CAGE", "DESTINATION", "DESTINATION CAGE NUMBER"], "DEST")
        if oc: transfers["ORIGIN CAGE"] = to_int_cage(transfers[oc])
        if dc: transfers["DESTINATION CAGE"] = to_int_cage(transfers[dc])
        tfish = find_col(transfers, ["NUMBER OF FISH", "N_FISH"], "FISH")
        if tfish: transfers["NUMBER OF FISH"] = pd.to_numeric(transfers[tfish].map(to_number), errors="coerce")
        tkg = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")
        if tkg and tkg != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={tkg: "TOTAL WEIGHT [KG]"}, inplace=True)
        if "TOTAL WEIGHT [KG]" in transfers.columns:
            transfers["TOTAL WEIGHT [KG]"] = pd.to_numeric(transfers["TOTAL WEIGHT [KG]"].map(to_number), errors="coerce")
        tabw = find_col(transfers, ["ABW (G)", "ABW [G]", "ABW(G)"], "ABW")
        if tabw: transfers["ABW (G)"] = pd.to_numeric(transfers[tabw].map(to_number), errors="coerce")
        if "DATE" in transfers.columns:
            transfers["DATE"] = pd.to_datetime(transfers["DATE"], errors="coerce")

    data = {"feeding": feeding, "harvest": harvest, "sampling": sampling, "transfers": transfers}

    if verbose:
        print("=== Data Summary ===")
        for k, df in data.items():
            if df is not None and not df.empty:
                dmin = df["DATE"].min() if "DATE" in df.columns else None
                dmax = df["DATE"].max() if "DATE" in df.columns else None
                print(f"{k:<10} rows={len(df):>5} | {dmin} → {dmax}")
        print("====================")

    return data



# Preprocess Cage 2
def preprocess_cage2(feeding: pd.DataFrame,
                     harvest: pd.DataFrame,
                     sampling: pd.DataFrame,
                     transfers: pd.DataFrame | None = None):
    """
    Build Cage 2 timeline aligned to sampling dates, starting with the stocking
    on 2024-08-26 (7290 fish @ 11.9 g), and cutting off after the final harvest
    on 2025-07-09. Transfers (in/out) are included; the initial stocking inbound
    transfer on 2024-08-26 is excluded from transfer cumulatives to avoid double counting.
    Returns:
        feeding_c2 (within window),
        harvest_c2 (within window),
        base (sampling timeline with stock/flows and ABW),
        prod_summary (per-sampling production summary with growth & eFCRs)
    """
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # ---- helpers
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        out = df.dropna(subset=["DATE"]).copy()
        out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
        out = out.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    # Filter cage 2
    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in feeding.columns else _clip(feeding)
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in harvest.columns else _clip(harvest)
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in sampling.columns else _clip(sampling)

    # ---- manual stocking baseline
    stocked_fish = 7290
    initial_abw_g = 11.9
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT (G)": initial_abw_g
    }])

    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base[(base["DATE"] >= start_date) & (base["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = float(stocked_fish)

    # ---- ensure final harvest date appears with ABW if available/derivable
    final_h_date = harvest_c2["DATE"].max() if not harvest_c2.empty else pd.NaT
    if pd.notna(final_h_date):
        hh = harvest_c2[harvest_c2["DATE"] == final_h_date].copy()
        fish_col = "NUMBER OF FISH" if "NUMBER OF FISH" in hh.columns else None
        kg_col   = "TOTAL WEIGHT [KG]" if "TOTAL WEIGHT [KG]" in hh.columns else None
        abw_colh = "ABW (G)" if "ABW (G)" in hh.columns else None
        abw_final = np.nan
        if fish_col and kg_col and hh[fish_col].notna().any() and hh[kg_col].notna().any():
            tot_fish = pd.to_numeric(hh[fish_col], errors="coerce").fillna(0).sum()
            tot_kg   = pd.to_numeric(hh[kg_col],   errors="coerce").fillna(0).sum()
            if tot_fish > 0 and tot_kg > 0:
                abw_final = (tot_kg * 1000.0) / tot_fish
        if np.isnan(abw_final) and abw_colh and hh[abw_colh].notna().any():
            abw_final = pd.to_numeric(hh[abw_colh].map(to_number), errors="coerce").mean()
        if pd.notna(abw_final):
            if (base["DATE"] == final_h_date).any():
                base.loc[base["DATE"] == final_h_date, "AVERAGE BODY WEIGHT (G)"] = abw_final
            else:
                base = pd.concat([
                    base,
                    pd.DataFrame([{
                        "DATE": final_h_date,
                        "CAGE NUMBER": cage_number,
                        "AVERAGE BODY WEIGHT (G)": abw_final,
                        "STOCKED": float(stocked_fish)
                    }])
                ], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # ---- initialize cum columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ---- harvest cumulatives aligned to sampling timeline
    if not harvest_c2.empty:
        h = harvest_c2.sort_values("DATE").copy()
        if "NUMBER OF FISH" in h.columns:
            h["H_FISH"] = pd.to_numeric(h["NUMBER OF FISH"], errors="coerce").fillna(0.0)
        else:
            h["H_FISH"] = 0.0
        if "TOTAL WEIGHT [KG]" in h.columns:
            h["H_KG"] = pd.to_numeric(h["TOTAL WEIGHT [KG]"], errors="coerce").fillna(0.0)
        else:
            h["H_KG"] = 0.0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0.0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0.0)

    # ---- transfers cumulatives (exclude the initial inbound on start_date)
    first_inbound_idx = None
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            # Identify earliest inbound to cage 2 on/after start_date
            tin_all = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE") if "DESTINATION CAGE" in t.columns else pd.DataFrame()
            if not tin_all.empty and tin_all["DATE"].iloc[0].normalize() == start_date.normalize():
                first_inbound_idx = tin_all.index[0]

            # Drop the stocking inbound (avoid double counting vs STOCKED)
            if first_inbound_idx is not None and first_inbound_idx in t.index:
                t = t.drop(index=first_inbound_idx)

            # Prepare numeric flows
            fish_col = "NUMBER OF FISH" if "NUMBER OF FISH" in t.columns else None
            kg_col   = "TOTAL WEIGHT [KG]" if "TOTAL WEIGHT [KG]" in t.columns else None
            if fish_col:
                t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0.0)
            else:
                t["T_FISH"] = 0.0
            if kg_col:
                t["T_KG"]   = pd.to_numeric(t[kg_col], errors="coerce").fillna(0.0)
            else:
                t["T_KG"]   = 0.0

            # Outgoing from cage 2
            if "ORIGIN CAGE" in t.columns:
                tout = t[t["ORIGIN CAGE"] == cage_number].sort_values("DATE").copy()
                if not tout.empty:
                    tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                    tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
                    mo = pd.merge_asof(base[["DATE"]].sort_values("DATE"),
                                       tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]].sort_values("DATE"),
                                       on="DATE", direction="backward")
                    base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0.0)
                    base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0.0)

            # Incoming to cage 2
            if "DESTINATION CAGE" in t.columns:
                tin = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE").copy()
                if not tin.empty:
                    tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                    tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
                    mi = pd.merge_asof(base[["DATE"]].sort_values("DATE"),
                                       tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]].sort_values("DATE"),
                                       on="DATE", direction="backward")
                    base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0.0)
                    base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0.0)

    # ---- standing fish & biomass at each sampling point
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0.0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].round().astype(int)

    # Ensure ABW column name
    abw_col = "AVERAGE BODY WEIGHT (G)"
    if abw_col not in base.columns:
        alt = find_col(base, ["AVERAGE BODY WEIGHT (G)", "ABW (G)", "ABW [G]", "ABW(G)", "ABW"], "ABW")
        if alt:
            base[abw_col] = base[alt]
        else:
            base[abw_col] = np.nan

    base["BIOMASS (KG)"] = (base["NUMBER OF FISH"] * base[abw_col] / 1000.0).astype(float)

    # -----------------------
    # 3) Production summary
    # -----------------------
    # Aggregate feeding to daily
    if not feeding_c2.empty:
        feed_daily = feeding_c2.groupby("DATE", as_index=False)["FEED AMOUNT (KG)"].sum()
    else:
        # If feeding data is missing, create zero feed series (no synthetic here)
        feed_daily = pd.DataFrame({"DATE": base["DATE"].copy(), "FEED AMOUNT (KG)": 0.0})

    # Build summary rows at each sampling date in 'base'
    base = base.sort_values("DATE").reset_index(drop=True)

    # Map cumulative flows at each base date (already on base)
    # Prepare per-period sums for harvest/transfers (needed for biomass gain adjustment)
    def _window_sum(df, date_from, date_to, col):
        if df is None or df.empty: return 0.0
        mask = (df["DATE"] > date_from) & (df["DATE"] <= date_to)
        return pd.to_numeric(df.loc[mask, col], errors="coerce").fillna(0.0).sum()

    # Prepare lightweight event tables for period sums
    h_ev = pd.DataFrame()
    if not harvest_c2.empty:
        h_ev = harvest_c2.copy()
        if "TOTAL WEIGHT [KG]" not in h_ev.columns:
            h_ev["TOTAL WEIGHT [KG]"] = 0.0
        if "NUMBER OF FISH" not in h_ev.columns:
            h_ev["NUMBER OF FISH"] = 0.0

    t_ev = pd.DataFrame()
    if transfers is not None and not transfers.empty:
        # drop initial inbound row if it was on start_date (as above)
        t_ev = _clip(transfers)
        if first_inbound_idx is not None and first_inbound_idx in t_ev.index:
            t_ev = t_ev.drop(index=first_inbound_idx)
        # normalize helper cols
        if "NUMBER OF FISH" not in t_ev.columns:
            t_ev["NUMBER OF FISH"] = 0.0
        if "TOTAL WEIGHT [KG]" not in t_ev.columns:
            t_ev["TOTAL WEIGHT [KG]"] = 0.0

    # Iterate sampling points to compute period metrics
    records = []
    feed_daily = feed_daily.sort_values("DATE")
    for i, row in base.iterrows():
        cur_date = row["DATE"]
        prev_date = base.loc[i-1, "DATE"] if i > 0 else pd.NaT

        # Feed in period (prev_date, cur_date]
        if pd.isna(prev_date):
            feed_period = feed_daily.loc[feed_daily["DATE"] <= cur_date, "FEED AMOUNT (KG)"].sum()
        else:
            feed_period = feed_daily.loc[(feed_daily["DATE"] > prev_date) & (feed_daily["DATE"] <= cur_date), "FEED AMOUNT (KG)"].sum()

        # Harvest kg & fish during period
        harv_kg_period = _window_sum(h_ev, prev_date, cur_date, "TOTAL WEIGHT [KG]") if not h_ev.empty else 0.0
        harv_fish_period = _window_sum(h_ev, prev_date, cur_date, "NUMBER OF FISH") if not h_ev.empty else 0.0

        # Transfers in/out kg during period (only cage 2 relevant)
        tin_kg_period = 0.0
        tout_kg_period = 0.0
        if not t_ev.empty:
            if "DESTINATION CAGE" in t_ev.columns:
                tin_kg_period = _window_sum(t_ev[t_ev["DESTINATION CAGE"] == cage_number], prev_date, cur_date, "TOTAL WEIGHT [KG]")
            if "ORIGIN CAGE" in t_ev.columns:
                tout_kg_period = _window_sum(t_ev[t_ev["ORIGIN CAGE"] == cage_number], prev_date, cur_date, "TOTAL WEIGHT [KG]")

        # Biomass now and previous
        biomass_now = float(row["BIOMASS (KG)"]) if pd.notna(row["BIOMASS (KG)"]) else np.nan
        if i == 0:
            biomass_prev = 0.0  # before stocking we consider 0 biomass
        else:
            biomass_prev = float(base.loc[i-1, "BIOMASS (KG)"])

        # Biomass gain adjusted for removals/additions in the window:
        # gain = (biomass_now - biomass_prev) + harvest_kg + transfers_out_kg - transfers_in_kg
        biomass_gain_period = (biomass_now - biomass_prev) + harv_kg_period + tout_kg_period - tin_kg_period

        # eFCRs
        eFCR_period = (feed_period / biomass_gain_period) if biomass_gain_period and biomass_gain_period > 0 else np.nan

        # cumulative feed up to this date
        feed_cum = feed_daily.loc[feed_daily["DATE"] <= cur_date, "FEED AMOUNT (KG)"].sum()
        # cumulative biomass gain since start (sum of period gains)
        if i == 0:
            biomass_gain_cum = biomass_gain_period
        else:
            # accumulate from previous cum
            prev_cum = records[-1]["BIOMASS_GAIN_CUM (KG)"] if pd.notna(records[-1]["BIOMASS_GAIN_CUM (KG)"]) else 0.0
            biomass_gain_cum = (prev_cum if prev_cum else 0.0) + (biomass_gain_period if biomass_gain_period else 0.0)

        eFCR_cum = (feed_cum / biomass_gain_cum) if biomass_gain_cum and biomass_gain_cum > 0 else np.nan

        records.append({
            "DATE": cur_date,
            "CAGE NUMBER": cage_number,
            "ABW (G)": row["AVERAGE BODY WEIGHT (G)"],
            "FISH ALIVE": int(row["NUMBER OF FISH"]),
            "BIOMASS (KG)": biomass_now,
            "FEED_PERIOD (KG)": feed_period,
            "FEED_CUM (KG)": feed_cum,
            "HARVEST_PERIOD_FISH": harv_fish_period,
            "HARVEST_PERIOD_KG": harv_kg_period,
            "TRANSFERS_IN_PERIOD (KG)": tin_kg_period,
            "TRANSFERS_OUT_PERIOD (KG)": tout_kg_period,
            "BIOMASS_GAIN_PERIOD (KG)": biomass_gain_period,
            "BIOMASS_GAIN_CUM (KG)": biomass_gain_cum,
            "eFCR_PERIOD": eFCR_period,
            "eFCR_CUM": eFCR_cum
        })

    prod_summary = pd.DataFrame.from_records(records).sort_values("DATE").reset_index(drop=True)

    return feeding_c2, harvest_c2, base, prod_summary


# Mock Data Generator for Additional Cages
def generate_mock_cage_data(base_feeding, base_sampling, base_harvest, cage_ids=[3,4,5,6,7]):
    rng = np.random.default_rng(42)  # reproducibility
    feeding_all, sampling_all, harvest_all = [], [], []

    for cid in cage_ids:
        # Feeding: add random noise ±20%
        f = base_feeding.copy()
        f["CAGE NUMBER"] = cid
        f["FEED AMOUNT (Kg)"] *= rng.uniform(0.8, 1.2, size=len(f))
        feeding_all.append(f)

        # Sampling: jitter ABW by ±15%, fish numbers ±10%
        s = base_sampling.copy()
        s["CAGE NUMBER"] = cid
        if "AVERAGE BODY WEIGHT(G)" in s.columns:
            s["AVERAGE BODY WEIGHT(G)"] *= rng.uniform(0.85, 1.15, size=len(s))
        if "NUMBER OF FISH" in s.columns:
            s["NUMBER OF FISH"] = (s["NUMBER OF FISH"] * rng.uniform(0.9, 1.1, size=len(s))).astype(int)
        sampling_all.append(s)

        # Harvest: jitter weights ±20%, fish counts ±10%
        h = base_harvest.copy()
        if "CAGE NUMBER" in h.columns:
            h["CAGE NUMBER"] = cid
        elif "CAGE" in h.columns:
            h["CAGE"] = cid
        if "TOTAL WEIGHT [KG]" in h.columns:
            h["TOTAL WEIGHT [KG]"] *= rng.uniform(0.8, 1.2, size=len(h))
        if "NUMBER OF FISH" in h.columns:
            h["NUMBER OF FISH"] = (h["NUMBER OF FISH"] * rng.uniform(0.9, 1.1, size=len(h))).astype(int)
        harvest_all.append(h)

    return (
        pd.concat(feeding_all, ignore_index=True),
        pd.concat(sampling_all, ignore_index=True),
        pd.concat(harvest_all, ignore_index=True),
    )

# --------------------------------------------------
# Multi-Cage Preprocessing Wrapper
# --------------------------------------------------
def preprocess_multiple_cages(base_feeding, base_sampling, base_harvest, transfers=None):
    cages = [2,3,4,5,6,7]

    results = {}
    for cid in cages:
        if cid == 2:
            f, h, base = preprocess_cage2(base_feeding, base_harvest, base_sampling, transfers)
        else:
            # Extract cage-specific mock data
            f = base_feeding[base_feeding["CAGE NUMBER"] == cid]
            h = base_harvest[(base_harvest.get("CAGE NUMBER") == cid) | (base_harvest.get("CAGE") == cid)]
            s = base_sampling[base_sampling["CAGE NUMBER"] == cid]

            # Run cage2 logic (but skip transfers)
            f, h, base = preprocess_cage2(f, h, s, transfers=None)

        results[cid] = {
            "feeding": f,
            "harvest": h,
            "summary": generate_production_summary(base, f)  # summary per cage
        }
    return results

# app.py — Fish Cage Production Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================
# Streamlit Page Setup
# ============================
st.set_page_config(page_title="Fish Cage Production Dashboard",
                   layout="wide",
                   page_icon="🐟")

st.title("🐟 Fish Cage Production Analysis Dashboard")

# ============================
# Sidebar Controls
# ============================
cages = [2]  # extend this list if you want more cages later
selected_cage = st.sidebar.selectbox("Select Cage", cages)

kpi_options = ["Growth", "eFCR"]
selected_kpi = st.sidebar.selectbox("Select KPI", kpi_options)

# ============================
# Data Loading (placeholder)
# ============================
# Replace this with your preprocessing + compute_summary pipeline
@st.cache_data
def load_summary(cage_number: int) -> pd.DataFrame:
    # Example schema from compute_summary()
    # In real pipeline: preprocess_cage2 -> compute_summary
    df = pd.DataFrame({
        "DATE": pd.date_range("2024-08-26", periods=8, freq="M"),
        "CAGE NUMBER": cage_number,
        "BIOMASS (KG)": [87, 240, 520, 890, 1450, 2100, 3200, 4000],
        "eFCR_PERIOD": [None, 1.4, 1.2, 1.5, 1.3, 1.4, 1.25, 1.3],
        "eFCR_CUM": [None, 1.4, 1.3, 1.35, 1.32, 1.35, 1.3, 1.3]
    })
    return df

summary = load_summary(selected_cage)

# ============================
# KPI Visualizations
# ============================

if selected_kpi == "Growth":
    st.subheader(f"Cage {selected_cage} — Growth Curve")
    fig = px.line(summary, x="DATE", y="BIOMASS (KG)",
                  markers=True,
                  title="Biomass Growth Over Time",
                  labels={"BIOMASS (KG)": "Biomass (kg)", "DATE": "Date"})
    st.plotly_chart(fig, use_container_width=True)

elif selected_kpi == "eFCR":
    st.subheader(f"Cage {selected_cage} — eFCR Trends")
    fig = px.line(summary, x="DATE", y=["eFCR_CUM", "eFCR_PERIOD"],
                  markers=True,
                  title="Aggregated & Period eFCR",
                  labels={"value": "eFCR", "DATE": "Date", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Football Squad Predictor",
    page_icon="⚽",
    layout="wide",
)

# ================================
# LOAD MODEL BUNDLE
# ================================
@st.cache_resource
def load_bundle():
    try:
        with open("model_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/player_data.csv")
        return df
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")
        st.info("Make sure 'player_data.csv' is in the 'data/' folder")
        st.stop()

# Load everything
bundle = load_bundle()
xgb_starter = bundle["xgb_starter"]
xgb_sub = bundle["xgb_sub"]
scaler = bundle["scaler"]
label_encoders = bundle["encoders"]
top_features = bundle["top_features"]

# ================================
# FIXED COLUMN NAMES
# ================================
team_col = "C_name"
position_col = "PS"
player_col = "P_name"

df = load_data()

# ================================
# TEAM NAME NORMALIZATION
#  Dataset এ যেকোন ভ্যারিয়েন্ট → ক্যানোনিকাল নাম (তোমার SS এর মতো)
# ================================
TEAM_NORMALIZE = {
    # Exact canonical names (no change, শুধু completeness এর জন্য)
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Man City": "Man City",
    "Chelsea": "Chelsea",
    "Newcastle": "Newcastle",
    "Aston Villa": "Aston Villa",
    "Forest": "Forest",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "West Ham": "West Ham",
    "Man Utd": "Man Utd",
    "Wolves": "Wolves",
    "Tottenham": "Tottenham",
    "Leicester": "Leicester",
    "Ipswich": "Ipswich",
    "Southampton": "Southampton",

    # Common variants → canonical
    "Manchester City": "Man City",
    "ManCity": "Man City",

    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",

    "Nottingham Forest": "Forest",
    "Nottm Forest": "Forest",

    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",

    "AFC Bournemouth": "Bournemouth",

    "West Ham United": "West Ham",
    "West Ham Utd": "West Ham",

    "Manchester United": "Man Utd",
    "Man United": "Man Utd",

    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",

    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",

    "Leicester City": "Leicester",
    "Leicester City FC": "Leicester",

    "Ipswich Town": "Ipswich",

    # কিছু extra safety mapping
    "Southampton FC": "Southampton",
}

df[team_col] = df[team_col].replace(TEAM_NORMALIZE)

# প্রয়োজনীয় কলাম আছে কি না
required_cols = [team_col, position_col, player_col]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# ================================
# HELPER FUNCTIONS
# ================================
pos_map = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

def get_player_category(ps):
    return pos_map.get(str(ps).strip().upper(), "OTHER")

def get_unique_teams():
    teams = df[team_col].dropna().unique().tolist()
    return sorted([str(t).strip() for t in teams if str(t).strip() != ""])

# ================================
# 20 CANONICAL TEAMS + 20 MANAGERS
# (তোমার দেওয়া টিম লিস্ট অনুযায়ী)
# ================================
MANAGER_BY_TEAM = {
    "Liverpool": "Arne Slot",
    "Arsenal": "Mikel Arteta",
    "Man City": "Pep Guardiola",
    "Chelsea": "Mauricio Pochettino",
    "Newcastle": "Eddie Howe",
    "Aston Villa": "Unai Emery",
    "Forest": "Nuno Espirito Santo",
    "Brighton": "Roberto De Zerbi",
    "Bournemouth": "Andoni Iraola",
    "Brentford": "Thomas Frank",
    "Fulham": "Marco Silva",
    "Crystal Palace": "Oliver Glasner",
    "Everton": "Sean Dyche",
    "West Ham": "David Moyes",
    "Man Utd": "Erik ten Hag",
    "Wolves": "Gary O'Neil",
    "Tottenham": "Ange Postecoglou",
    "Leicester": "Enzo Maresca",
    "Ipswich": "Kieran McKenna",
    "Southampton": "Russell Martin",
}

# শুধু এই ২০টা canonical টিমই আমাদের ফাইনাল লিস্ট
CANONICAL_TEAMS = list(MANAGER_BY_TEAM.keys())

# reverse map: Manager -> [Teams]
MANAGER_TO_TEAMS = {}
for team_name, manager in MANAGER_BY_TEAM.items():
    MANAGER_TO_TEAMS.setdefault(manager, []).append(team_name)

# ================================
# SIDEBAR: INPUTS
# ================================
st.sidebar.title("⚽ Squad Builder")
st.sidebar.markdown("---")

st.sidebar.success(f"✅ Team Column: **{team_col}**")
st.sidebar.success(f"✅ Position Column: **{position_col}**")

st.sidebar.markdown("---")

# Team dropdown: শুধু canonical 20 team
teams = sorted(CANONICAL_TEAMS)

selected_team = st.sidebar.selectbox("🏟️ Select Team", teams)

# MANAGER SELECT (label = Manager (Team))
st.sidebar.markdown("### 👔 Manager")

all_managers = sorted(MANAGER_TO_TEAMS.keys())

manager_display_options = []
label_to_manager = {}

for manager in all_managers:
    teams_for_manager = ", ".join(MANAGER_TO_TEAMS.get(manager, []))
    label = f"{manager} ({teams_for_manager})"
    manager_display_options.append(label)
    label_to_manager[label] = manager

selected_manager_label = st.sidebar.selectbox(
    "Select / type manager",
    options=manager_display_options,
    index=0,
    help="Start typing to search manager name",
)

manager_name = label_to_manager[selected_manager_label]

# ================================
# FORMATION & SUBS
# ================================
st.sidebar.markdown("### 📊 Starting XI Formation")
st.sidebar.caption("Total must be 11 players")

col1, col2 = st.sidebar.columns(2)
with col1:
    sx_gk = st.number_input("🧤 GK", 0, 2, 1, key="sx_gk")
    sx_def = st.number_input("🛡️ DEF", 0, 6, 4, key="sx_def")
with col2:
    sx_mid = st.number_input("⚙️ MID", 0, 6, 4, key="sx_mid")
    sx_fwd = st.number_input("⚡ FWD", 0, 5, 2, key="sx_fwd")

total_xi = sx_gk + sx_def + sx_mid + sx_fwd

if total_xi == 11:
    st.sidebar.success(f"✅ Total: {total_xi} players")
else:
    st.sidebar.warning(f"⚠️ Total: {total_xi} / 11 players")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Substitutes")

col3, col4 = st.sidebar.columns(2)
with col3:
    sub_gk = st.number_input("🧤 Sub GK", 0, 2, 1, key="sub_gk")
    sub_def = st.number_input("🛡️ Sub DEF", 0, 4, 2, key="sub_def")
with col4:
    sub_mid = st.number_input("⚙️ Sub MID", 0, 4, 2, key="sub_mid")
    sub_fwd = st.number_input("⚡ Sub FWD", 0, 4, 2, key="sub_fwd")

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🎯 **PREDICT SQUAD**", type="primary", use_container_width=True)

# ================================
# MAIN AREA
# ================================
st.title("⚽ Football Starting XI & Substitute Predictor")
st.caption("AI-powered squad selection using Machine Learning")
st.markdown("---")

# ================================
# PREDICTION LOGIC
# ================================
if predict_btn:

    if total_xi != 11:
        st.error(f"⚠️ Starting XI must have exactly 11 players! (Current: {total_xi})")
        st.stop()

    with st.spinner("🔍 Analyzing players and generating predictions..."):

        try:
            # Filter team players (already normalized)
            team_players = df[df[team_col].str.strip() == selected_team.strip()].copy()
            team_players = team_players.drop_duplicates(subset=[player_col])

            if team_players.empty:
                st.error(f"❌ No players found for **{selected_team}** in the dataset!")
                st.info(
                    "Current unique teams in data (after normalization): "
                    + ", ".join(get_unique_teams())
                )
                st.stop()

            st.info(f"✅ Found **{len(team_players)}** players for {selected_team}")

            # Prepare input
            input_data = team_players.copy()
            input_data["Managers"] = manager_name

            # Label encoding
            for col in label_encoders:
                if col in input_data.columns:
                    le = label_encoders[col]
                    input_data[col] = (
                        input_data[col]
                        .fillna("missing")
                        .astype(str)
                        .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                    )

            # Add missing columns for scaler
            for col in scaler.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Scaling
            input_for_scaler = input_data[scaler.feature_names_in_]
            input_scaled = input_data.copy()
            input_scaled[scaler.feature_names_in_] = scaler.transform(input_for_scaler)

            # Feature selection
            final_input = input_scaled[top_features]

            # Prediction
            team_players["Starter_Score"] = xgb_starter.predict_proba(final_input)[:, 1]
            team_players["Sub_Score"] = xgb_sub.predict_proba(final_input)[:, 1]
            team_players["Category"] = team_players[position_col].apply(get_player_category)

            # Select Starting XI by position
            gks = team_players[team_players["Category"] == "GK"].sort_values(
                "Starter_Score", ascending=False
            )
            defs = team_players[team_players["Category"] == "DEF"].sort_values(
                "Starter_Score", ascending=False
            )
            mids = team_players[team_players["Category"] == "MID"].sort_values(
                "Starter_Score", ascending=False
            )
            fwds = team_players[team_players["Category"] == "FWD"].sort_values(
                "Starter_Score", ascending=False
            )

            start_xi = pd.concat(
                [
                    gks.head(sx_gk),
                    defs.head(sx_def),
                    mids.head(sx_mid),
                    fwds.head(sx_fwd),
                ]
            )

            # Display Starting XI
            st.success(f"✅ **Prediction Complete for {selected_team}!**")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("🏆 Starting XI", len(start_xi), "players")
            with col_m2:
                avg_conf = (start_xi["Starter_Score"].mean() * 100)
                st.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")
            with col_m3:
                if "Rating" in start_xi.columns:
                    avg_rating = start_xi["Rating"].mean()
                    st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")

            st.markdown("---")
            st.subheader(f"🏆 Starting XI ({sx_gk}-{sx_def}-{sx_mid}-{sx_fwd})")

            # Prepare display data
            xi_display_cols = [player_col, position_col, "Starter_Score"]
            if "Rating" in start_xi.columns:
                xi_display_cols.insert(2, "Rating")

            xi_df = start_xi[xi_display_cols].copy()

            xi_df["Confidence (%)"] = (
                xi_df["Starter_Score"] * 100
            ).round(0).astype(int)

            xi_df.insert(0, "No.", range(1, len(xi_df) + 1))

            rename_dict = {player_col: "Player", position_col: "Pos"}
            if "Rating" in xi_df.columns:
                rename_dict["Rating"] = "Avg Rating"

            xi_df = xi_df.rename(columns=rename_dict)

            final_cols = ["No.", "Pos", "Player"]
            if "Avg Rating" in xi_df.columns:
                final_cols.append("Avg Rating")
            final_cols.append("Confidence (%)")

            html_table = """
            <style>
                .dark-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 14px;
                    font-family: 'Source Sans Pro', sans-serif;
                    background-color: #0e1117;
                    color: #fafafa;
                }
                .dark-table thead tr {
                    background-color: #262730;
                    color: #fafafa;
                    text-align: center;
                    font-weight: 600;
                    border-bottom: 1px solid #3d3d4d;
                }
                .dark-table th {
                    padding: 12px 15px;
                    text-align: center !important;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .dark-table td {
                    padding: 12px 15px;
                    text-align: center !important;
                    border-bottom: 1px solid #262730;
                }
                .dark-table tbody tr {
                    background-color: #0e1117;
                }
                .dark-table tbody tr:hover {
                    background-color: #1a1d26;
                }
            </style>
            <table class="dark-table">
                <thead>
                    <tr>
            """

            for col in final_cols:
                html_table += f"<th>{col}</th>"
            html_table += "</tr></thead><tbody>"

            for _, row in xi_df[final_cols].iterrows():
                html_table += "<tr>"
                for col in final_cols:
                    html_table += f"<td>{row[col]}</td>"
                html_table += "</tr>"

            html_table += "</tbody></table>"

            st.markdown(html_table, unsafe_allow_html=True)

            # Substitutes
            remaining = team_players.drop(start_xi.index)

            rem_gks = remaining[remaining["Category"] == "GK"].sort_values(
                "Sub_Score", ascending=False
            )
            rem_defs = remaining[remaining["Category"] == "DEF"].sort_values(
                "Sub_Score", ascending=False
            )
            rem_mids = remaining[remaining["Category"] == "MID"].sort_values(
                "Sub_Score", ascending=False
            )
            rem_fwds = remaining[remaining["Category"] == "FWD"].sort_values(
                "Sub_Score", ascending=False
            )

            subs = pd.concat(
                [
                    rem_gks.head(sub_gk),
                    rem_defs.head(sub_def),
                    rem_mids.head(sub_mid),
                    rem_fwds.head(sub_fwd),
                ]
            )

            st.markdown("---")
            st.subheader("⚡ Substitutes Bench")

            if subs.empty:
                st.info("No substitutes selected with current settings.")
            else:
                sub_display_cols = [player_col, position_col, "Sub_Score"]
                if "Rating" in subs.columns:
                    sub_display_cols.insert(2, "Rating")

                sub_df = subs[sub_display_cols].copy()

                sub_df["Confidence (%)"] = (
                    sub_df["Sub_Score"] * 100
                ).round(0).astype(int)

                sub_df.insert(0, "No.", range(1, len(sub_df) + 1))

                sub_df = sub_df.rename(columns=rename_dict)

                html_table_sub = """
                <style>
                    .dark-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        font-size: 14px;
                        font-family: 'Source Sans Pro', sans-serif;
                        background-color: #0e1117;
                        color: #fafafa;
                    }
                    .dark-table thead tr {
                        background-color: #262730;
                        color: #fafafa;
                        text-align: center;
                        font-weight: 600;
                        border-bottom: 1px solid #3d3d4d;
                    }
                    .dark-table th {
                        padding: 12px 15px;
                        text-align: center !important;
                        font-size: 13px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    .dark-table td {
                        padding: 12px 15px;
                        text-align: center !important;
                        border-bottom: 1px solid #262730;
                    }
                    .dark-table tbody tr {
                        background-color: #0e1117;
                    }
                    .dark-table tbody tr:hover {
                        background-color: #1a1d26;
                    }
                </style>
                <table class="dark-table">
                    <thead>
                        <tr>
                """

                for col in final_cols:
                    html_table_sub += f"<th>{col}</th>"
                html_table_sub += "</tr></thead><tbody>"

                for _, row in sub_df[final_cols].iterrows():
                    html_table_sub += "<tr>"
                    for col in final_cols:
                        html_table_sub += f"<td>{row[col]}</td>"
                    html_table_sub += "</tr>"

                html_table_sub += "</tbody></table>"

                st.markdown(html_table_sub, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

else:
    st.info("👈 বামে সাইডবার থেকে টিম, ফরমেশন এবং ম্যানেজার সিলেক্ট করে **'PREDICT SQUAD'** বাটন চাপো।")

    st.markdown("### 📊 Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.caption("🤖 Powered by XGBoost Machine Learning | Built with Streamlit")


# -*- coding: utf-8 -*-
import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud

import requests
import plotly.express as px
import requests
import json

# -----------------------------
# Helpers
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces, fix known spellings, unify names."""
    mapping = {
        "s.no": "S.No",
        "course type": "Course Type",
        "name of the fund": "Name of the Fund",
        "duration": "Duration",
        "course title": "Course title",
        "maleforiegn": "MaleForeign",
        "femaleforiegn": "FemaleForeign",
        "totalforiegn": "TotalForeign",
        "malenational": "MaleNational",
        "femalenational": "FemaleNational",
        "totalnational": "TotalNational",
        "totalforiegn+national": "TotalAll",
        "year": "Year",
        "course tags": "Course Tags",
        "country name": "Country Name",
        "male": "Male",
        "female": "Female",
        "total": "Total",
    }
    cols = []
    for c in df.columns:
        c2 = str(c).replace("\n", " ").strip()
        c2 = c2.replace("Foriegn", "Foreign")
        c2 = c2.replace("  ", " ")
        c2_low = c2.lower()
        cols.append(mapping.get(c2_low, c2))
    df.columns = cols
    return df

def load_excel(excel_bytes_or_path):
    """Return sheet1, sheet2 as cleaned DataFrames."""
    try:
        if isinstance(excel_bytes_or_path, (bytes, bytearray)):
            s1 = pd.read_excel(io.BytesIO(excel_bytes_or_path), sheet_name=0, engine='openpyxl')
            s2 = pd.read_excel(io.BytesIO(excel_bytes_or_path), sheet_name=1, engine='openpyxl')
        else:
            s1 = pd.read_excel(excel_bytes_or_path, sheet_name=0, engine='openpyxl')
            s2 = pd.read_excel(excel_bytes_or_path, sheet_name=1, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        raise

    s1 = normalize_columns(s1)
    s2 = normalize_columns(s2)

    # Ensure numeric columns are numeric
    numeric_cols_s1 = ["MaleForeign", "FemaleForeign", "TotalForeign",
                      "MaleNational", "FemaleNational", "TotalNational",
                      "TotalAll"]
    
    numeric_cols_s2 = ["Male", "Female", "Total"]
    
    for col in numeric_cols_s1:
        if col in s1.columns:
            s1[col] = pd.to_numeric(s1[col], errors="coerce").fillna(0)

    for col in numeric_cols_s2:
        if col in s2.columns:
            s2[col] = pd.to_numeric(s2[col], errors="coerce").fillna(0)

    if "Year" in s1.columns:
        s1["Year"] = pd.to_numeric(s1["Year"], errors="coerce").fillna(0)

    # If TotalAll missing, compute it
    if "TotalAll" not in s1.columns:
        foreign_total = s1.get("TotalForeign", pd.Series([0] * len(s1)))
        national_total = s1.get("TotalNational", pd.Series([0] * len(s1)))
        s1["TotalAll"] = foreign_total.fillna(0) + national_total.fillna(0)

    return s1, s2

def kpi_card(label: str, value):
    """Display a KPI card."""
    if isinstance(value, (int, float, np.number)):
        display_value = f"{int(value):,}"
    else:
        display_value = str(value)
    st.metric(label, display_value)

def make_wordcloud(text_series):
    """Return base64 png of a wordcloud."""
    text = " ".join([str(x) for x in text_series.dropna().tolist()])
    if not text.strip():
        text = "No Tags"
    wc = WordCloud(width=1200, height=500, background_color="white").generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def create_quarterly_metrics_chart(df1):
    """Create quarterly metrics dynamically from data (Sheet1)."""
    if "Duration" not in df1.columns:
        st.warning("No 'Duration' column found to calculate quarterly metrics.")
        return go.Figure()

    # Map month abbreviations to quarter
    month_to_quarter = {
        "jan": "Q4", "feb": "Q4", "mar": "Q4",
        "apr": "Q1", "may": "Q1", "jun": "Q1",
        "jul": "Q2", "aug": "Q2", "sep": "Q2",
        "oct": "Q3", "nov": "Q3", "dec": "Q3"
    }

    # Try to extract month from Duration (e.g., "Apr 24", "July 2024", etc.)
    def extract_quarter(duration):
        if pd.isna(duration):
            return None
        d = str(duration).lower()
        for m, q in month_to_quarter.items():
            if m in d:
                return q
        return None

    df1["Quarter"] = df1["Duration"].apply(extract_quarter)

    # Aggregate by quarter
    quarterly = df1.groupby("Quarter")["TotalAll"].sum().reset_index()

    # Ensure all quarters appear
    all_quarters = ["Q1", "Q2", "Q3", "Q4"]
    quarterly = quarterly.set_index("Quarter").reindex(all_quarters, fill_value=0).reset_index()

    total = quarterly["TotalAll"].sum()
    if total > 0:
        quarterly["Percentage"] = (quarterly["TotalAll"] / total * 100).round(1)
    else:
        quarterly["Percentage"] = 0

    # Plot
    fig = go.Figure(go.Bar(
        x=quarterly["Quarter"],
        y=quarterly["Percentage"],
        text=[f"{p}%" for p in quarterly["Percentage"]],
        textposition="auto",
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ))

    fig.update_layout(
        title="Quarterly Course Metrics (Dynamic)",
        yaxis=dict(title="Percentage", range=[0, 100]),
        height=400
    )

    return fig

 
def create_thematic_area_chart(df1):
    """Create thematic area participation chart dynamically from Course Tags."""
    if "Course Tags" not in df1.columns or "TotalAll" not in df1.columns:
        st.warning("Thematic area chart requires 'Course Tags' and 'TotalAll' columns.")
        return go.Figure()

    # Expand Course Tags (if multiple tags are comma/semicolon separated)
    expanded = (
        df1.assign(**{
            "Course Tags": df1["Course Tags"].astype(str).str.split(r"[;,]")
        })
        .explode("Course Tags")
    )
    expanded["Course Tags"] = expanded["Course Tags"].str.strip()

    # Aggregate participants by thematic area
    thematic_agg = (
        expanded.groupby("Course Tags", as_index=False)["TotalAll"].sum()
        .sort_values("TotalAll", ascending=False)
    )

    total = thematic_agg["TotalAll"].sum()
    if total > 0:
        thematic_agg["Percentage"] = (thematic_agg["TotalAll"] / total * 100).round(1)
    else:
        thematic_agg["Percentage"] = 0

    # Plot
    fig = px.bar(
        thematic_agg,
        x="Percentage",
        y="Course Tags",
        orientation="h",
        text=thematic_agg["Percentage"].astype(str) + "%",
        title="Participant Turnout by Thematic Area"
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Percentage",
        showlegend=False,
        height=400
    )
    return fig
    

def create_gender_donut_chart(male_percentage=60, female_percentage=40):
    """Create gender distribution donut chart."""
    fig = go.Figure(data=[go.Pie(
        labels=['Male', 'Female'],
        values=[male_percentage, female_percentage],
        hole=0.6,
        marker_colors=['#1f77b4', '#ff7f0e']
    )])
    
    fig.update_layout(
        title="Gender Distribution",
        showlegend=True,
        height=300
    )
    
    return fig

def create_international_national_chart(international_percentage=50, national_percentage=50):
    """Create international vs national participation chart."""
    fig = go.Figure(data=[go.Pie(
        labels=['International', 'National'],
        values=[international_percentage, national_percentage],
        hole=0.6,
        marker_colors=['#2ca02c', '#d62728']
    )])
    
    fig.update_layout(
        title="Reach, Diversity & Engagement",
        showlegend=True,
        height=300
    )
    
    return fig

import circlify
import plotly.express as px


def create_funding_agency_circle_packing(df1):
    """Circle packing for participants by funding agencies."""
    if "Name of the Fund" not in df1.columns or "TotalAll" not in df1.columns:
        st.warning("Funding agency chart requires 'Name of the Fund' and 'TotalAll' columns.")
        return go.Figure()

    # Aggregate participants
    fund_agg = (
        df1.groupby("Name of the Fund", as_index=False)["TotalAll"].sum()
        .sort_values("TotalAll", ascending=False)
    )

    # Use circlify to compute circle packing
    circles = circlify.circlify(
        fund_agg["TotalAll"].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # Build dataframe with circle positions
    circle_data = []
    for circle, label, value in zip(circles, fund_agg["Name of the Fund"], fund_agg["TotalAll"]):
        circle_data.append({
            "x": circle.x,
            "y": circle.y,
            "r": circle.r,
            "label": label,
            "value": value
        })
    circle_df = pd.DataFrame(circle_data)

    # Plot with scatter
    fig = px.scatter(
        circle_df,
        x="x", y="y",
        size="r",
        size_max=200,
        hover_name="label",
        hover_data={"value": True, "x": False, "y": False, "r": False},
        text="label",
        title="Funding Agencies (Circle Packing)"
    )

    fig.update_traces(marker=dict(opacity=0.6, line=dict(width=2, color="black")), textposition="middle center")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False, height=600)

    return fig


#import pandas as pd
#import holoviews as hv
#from holoviews import opts
#import streamlit as st
#
#hv.extension("bokeh")
#
#def create_chord_diagram(df2):
#    # Aggregate participants by country
#    country_agg = df2.groupby("Country Name")["Total"].sum().reset_index()
#
#    # Create edge list (Country ? INCOIS with weight = Total)
#    links = []
#    for _, row in country_agg.iterrows():
#        links.append((row["Country Name"], "INCOIS", row["Total"]))
#
#    # Convert to DataFrame
#    links_df = pd.DataFrame(links, columns=["source", "target", "value"])
#
#    # Create chord
#    chord = hv.Chord(links_df)
#    chord = chord.opts(
#        opts.Chord(
#            cmap="Category20",
#            edge_cmap="Blues",
#            edge_color="source",
#            node_color="index",
#            labels="index",
#            edge_line_width=hv.dim("value") * 0.01,
#            width=800, height=800,
#            title="Foreign Participants Turnout to INCOIS"
#        )
#    )
#
#    return chord


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="ITCOO Course Analytics", layout="wide")
st.title("ITCOO Course Analytics")

with st.sidebar:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload Excel (two sheets: Sheet1 & Sheet2)", type=["xlsx"])
    default_path = "data.xlsx"
    use_default = st.checkbox(f"Use local file: {default_path}", value=False)

# Load data
try:
    if uploaded is not None:
        sheet1, sheet2 = load_excel(uploaded.getvalue())
    elif use_default:
        sheet1, sheet2 = load_excel(default_path)
    else:
        st.info("Upload your Excel file or tick 'Use local file' to proceed.")
        st.stop()
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")
    
    # Year range
    if "Year" in sheet1.columns and not sheet1["Year"].isna().all():
        valid_years = sheet1["Year"].dropna().unique()
        if len(valid_years) > 0:
            years_sorted = sorted(valid_years)
            yr_min, yr_max = int(min(years_sorted)), int(max(years_sorted))
            year_range = st.slider("Filter by Year", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
        else:
            year_range = (None, None)
    else:
        year_range = (None, None)

    # Course Type filter
    if "Course Type" in sheet1.columns:
        ctype_vals = sorted([str(x) for x in sheet1["Course Type"].dropna().unique()])
        selected_types = st.multiselect("Course Type", options=ctype_vals, default=ctype_vals)
    else:
        selected_types = []

    # Search in Course Tags
    tag_query = st.text_input("Search in Course Tags (optional)").strip()

# Apply filters
df1 = sheet1.copy()
if year_range[0] is not None:
    df1 = df1[(df1["Year"] >= year_range[0]) & (df1["Year"] <= year_range[1])]
if selected_types and "Course Type" in df1.columns:
    df1 = df1[df1["Course Type"].isin(selected_types)]
if tag_query and "Course Tags" in df1.columns:
    df1 = df1[df1["Course Tags"].astype(str).str.contains(tag_query, case=False, na=False)]

# Link Sheet2
df2 = sheet2.copy()
if "Course title" in df1.columns and "Course title" in df2.columns:
    df2 = df2[df2["Course title"].isin(df1["Course title"].dropna().unique())]

# KPIs
c1, c2, c3, c4 = st.columns(4)
total_courses = df1["Course title"].nunique() if "Course title" in df1.columns else len(df1)
total_foreign = df1.get("TotalForeign", pd.Series([0])).sum()
total_national = df1.get("TotalNational", pd.Series([0])).sum()
total_all = df1.get("TotalAll", pd.Series([0])).sum()

with c1: kpi_card("Total Courses", total_courses)
with c2: kpi_card("Foreign Participants", total_foreign)
with c3: kpi_card("National Participants", total_national)
with c4: kpi_card("Total Participants", total_all)

st.markdown("---")

# NEW: Add Quarterly Metrics and Thematic Area Charts
st.subheader("Quarterly Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    #st.plotly_chart(create_quarterly_metrics_chart(), use_container_width=True)
    st.plotly_chart(create_quarterly_metrics_chart(df1), use_container_width=True)

with col2:
    #st.plotly_chart(create_thematic_area_chart(), use_container_width=True)
    st.plotly_chart(create_thematic_area_chart(df1), use_container_width=True)


# NEW: Add Gender and International/National Charts
col3, col4 = st.columns(2)

with col3:
    # Calculate actual gender percentages from data
    male_total = df1.get("MaleForeign", pd.Series([0])).sum() + df1.get("MaleNational", pd.Series([0])).sum()
    female_total = df1.get("FemaleForeign", pd.Series([0])).sum() + df1.get("FemaleNational", pd.Series([0])).sum()
    total_gender = male_total + female_total
    
    if total_gender > 0:
        male_pct = (male_total / total_gender) * 100
        female_pct = (female_total / total_gender) * 100
        st.plotly_chart(create_gender_donut_chart(male_pct, female_pct), use_container_width=True)
    else:
        st.plotly_chart(create_gender_donut_chart(), use_container_width=True)

with col4:
    # Calculate actual international/national percentages
    total_foreign_val = df1.get("TotalForeign", pd.Series([0])).sum()
    total_national_val = df1.get("TotalNational", pd.Series([0])).sum()
    total_all_val = total_foreign_val + total_national_val
    
    if total_all_val > 0:
        intl_pct = (total_foreign_val / total_all_val) * 100
        natl_pct = (total_national_val / total_all_val) * 100
        st.plotly_chart(create_international_national_chart(intl_pct, natl_pct), use_container_width=True)
    else:
        st.plotly_chart(create_international_national_chart(), use_container_width=True)

st.markdown("---")

# Main Layout
left, right = st.columns([1, 1])


import plotly.graph_objects as go

# Left column
with left:
    st.subheader("Participants by Country")
    if not df2.empty and "Country Name" in df2.columns:
        country_agg = df2.groupby("Country Name", as_index=False)["Total"].sum().sort_values("Total", ascending=False)

        # --- Bar chart
        fig_bar = px.bar(
            country_agg.head(30),
            x="Total", y="Country Name",
            orientation="h", title="Top Countries by Participants"
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- World map
        fig_map = px.choropleth(
            country_agg, locations="Country Name", locationmode="country names", 
            color="Total", hover_name="Country Name", title="World Map: Participants by Country"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        
#        # --- Load detailed world GeoJSON (hosted)
#        url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
#        world_geo = requests.get(url).json()
        
#        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
#        world_geo = requests.get(url).json()
#
#        # Choropleth with GeoJSON
#        fig_map = px.choropleth(
#          country_agg,
#          geojson=world_geo,
#          locations="Country Name",
#          featureidkey="properties.name",
#          color="Total",
#          hover_name="Country Name",
#          title="World Map: Participants by Country"
#        )
#        fig_map.update_geos(fitbounds="locations", visible=True)
#        st.plotly_chart(fig_map, use_container_width=True)

        # --- Chord-like Sankey Diagram (Foreign ? INCOIS)
        source_nodes = list(country_agg["Country Name"])
        target_node = "INCOIS"

        labels = source_nodes + [target_node]
        sources = list(range(len(source_nodes)))
        targets = [len(source_nodes)] * len(source_nodes)
        values = list(country_agg["Total"])

        fig_chord = go.Figure(go.Sankey(
            node=dict(
                pad=20, thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        ))

        fig_chord.update_layout(title_text="Foreign Participants Turnout to INCOIS", font_size=12)
        st.plotly_chart(fig_chord, use_container_width=True)
        
#        chord = create_chord_diagram(df2)
#        st.bokeh_chart(hv.render(chord, backend="bokeh"), use_container_width=True)

    else:
        st.info("No country-level data found")



## Left column
#with left:
#    st.subheader("Participants by Country")
#    if not df2.empty and "Country Name" in df2.columns:
#        country_agg = df2.groupby("Country Name", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
#        fig_bar = px.bar(country_agg.head(30), x="Total", y="Country Name", orientation="h", title="Top Countries by Participants")
#        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
#        st.plotly_chart(fig_bar, use_container_width=True)
#        
#        fig_map = px.choropleth(country_agg, locations="Country Name", locationmode="country names", 
#                               color="Total", hover_name="Country Name", title="World Map: Participants by Country")
#        st.plotly_chart(fig_map, use_container_width=True)
#    else:
#        st.info("No country-level data found")

# Right column
with right:
    # Courses by year
    st.subheader("Courses by Year")
    if "Year" in df1.columns and "Course Type" in df1.columns:
        courses_by_year = df1.groupby(["Year", "Course Type"]).size().reset_index(name="Count")
        fig_area = px.area(courses_by_year, x="Year", y="Count", color="Course Type", title="Courses per Year by Type")
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Cannot build courses by year chart")

#    # Funding agencies
#    st.subheader("Funding Agencies")
#    if "Name of the Fund" in df1.columns:
#        fund_agg = df1.groupby("Name of the Fund")["TotalAll"].sum().reset_index().sort_values("TotalAll", ascending=False)
#        fig_fund = px.scatter(fund_agg, x="Name of the Fund", y="TotalAll", size="TotalAll", 
#                             title="Participants by Funding Agency")
#        fig_fund.update_layout(xaxis={"visible": False})
#        st.plotly_chart(fig_fund, use_container_width=True)
#    else:
#        st.info("No funding agency data")
    
    st.subheader("Funding Agencies")
    st.plotly_chart(create_funding_agency_circle_packing(df1), use_container_width=True)
    
    
    # Word cloud
    st.subheader("Course Tags")
    if "Course Tags" in df1.columns:
        b64_img = make_wordcloud(df1["Course Tags"])
        st.image(f"data:image/png;base64,{b64_img}", use_container_width=True)
    else:
        st.info("No course tags data")

# Data preview
with st.expander("Data Preview (Sheet1)"):
    st.dataframe(df1.head(20))
with st.expander("Data Preview (Sheet2)"):
    st.dataframe(df2.head(20))
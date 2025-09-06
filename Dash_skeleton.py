from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import io
import base64
from wordcloud import WordCloud

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
        # sheet2
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

def make_wordcloud(text_series: pd.Series) -> str:
    """Return base64 png of a wordcloud built from the 'Course Tags' column."""
    text = " ".join([str(x) for x in text_series.dropna().tolist()])
    if not text.strip():
        text = "No Tags"
    wc = WordCloud(width=1200, height=500, background_color="white").generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# -----------------------------
# Load and preprocess data
# -----------------------------
df1 = pd.read_excel("ITCOO_2024-25.xlsx", sheet_name="Sheet1")
df2 = pd.read_excel("ITCOO_2024-25.xlsx", sheet_name="Sheet2")

# Apply normalization and numeric conversion
df1 = normalize_columns(df1)
df2 = normalize_columns(df2)

for col in ["MaleForeign", "FemaleForeign", "TotalForeign", "MaleNational", "FemaleNational", "TotalNational", "TotalAll"]:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")
for col in ["Male", "Female", "Total"]:
    if col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
if "Year" in df1.columns:
    df1["Year"] = pd.to_numeric(df1["Year"], errors="coerce")

# -----------------------------
# Dash App Layout
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("ITCOO Course Analytics", style={"textAlign": "center"}),
    html.Div([
        html.Label("Filter by Year Range:"),
        dcc.RangeSlider(
            id="year-slider",
            min=df1["Year"].min(),
            max=df1["Year"].max(),
            value=[df1["Year"].min(), df1["Year"].max()],
            marks={str(y): str(y) for y in sorted(df1["Year"].unique())}
        ),
    ], style={"width": "90%", "margin": "auto", "padding": "20px"}),
    
    html.Div([
        # Left Column for Country and Gender plots
        html.Div([
            html.H3("Participants by Country"),
            dcc.Graph(id="country-bar"),
            dcc.Graph(id="world-map"),
            html.H3("ITCOO Training User Demographics"),
            dcc.Graph(id="gender-pie")
        ], style={"flex": "1", "padding": "10px"}),

        # Right Column for Course Trends and Funding/Tags
        html.Div([
            html.H3("Courses by Year (International vs National)"),
            dcc.Graph(id="course-area"),
            html.H3("Funding Agencies"),
            dcc.Graph(id="funding-bubble"),
            html.H3("Course Tags"),
            html.Img(id="word-cloud-img", style={"width": "100%", "height": "auto"})
        ], style={"flex": "1", "padding": "10px"})
    ], style={"display": "flex"})
])

# -----------------------------
# Dash App Callbacks
# -----------------------------
@app.callback(
    [Output("country-bar", "figure"),
     Output("world-map", "figure"),
     Output("gender-pie", "figure"),
     Output("course-area", "figure"),
     Output("funding-bubble", "figure"),
     Output("word-cloud-img", "src")],
    Input("year-slider", "value")
)
def update_dashboard(year_range):
    # Filter Sheet1 data
    dff1 = df1[(df1["Year"] >= year_range[0]) & (df1["Year"] <= year_range[1])]
    
    # Filter Sheet2 data based on Course titles from filtered Sheet1
    filtered_course_titles = dff1["Course title"].dropna().unique()
    dff2 = df2[df2["Course title"].isin(filtered_course_titles)]

    # 1. Country bar and map
    country_agg = dff2.groupby("Country Name", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
    fig_country = px.bar(
        country_agg.head(30),
        x="Total", y="Country Name",
        orientation="h", title="Top Countries by Participants",
    )
    fig_country.update_layout(yaxis={"categoryorder": "total ascending"})
    
    fig_map = px.choropleth(
        country_agg,
        locations="Country Name",
        locationmode="country names",
        color="Total",
        hover_name="Country Name",
        title="World Map: Participants by Country",
        color_continuous_scale="Blues",
    )

    # 2. Gender demographics
    male = dff1.get("MaleForeign", pd.Series(dtype=float)).sum() + dff1.get("MaleNational", pd.Series(dtype=float)).sum()
    female = dff1.get("FemaleForeign", pd.Series(dtype=float)).sum() + dff1.get("FemaleNational", pd.Series(dtype=float)).sum()
    fig_gender = px.pie(
        values=[male, female], names=["Male", "Female"],
        hole=0.35, title="Gender Distribution (Foreign + National)"
    )

    # 3. Courses by year (stacked area)
    courses_by_year = (
        dff1.groupby(["Year", "Course Type"], as_index=False)["Course title"]
        .nunique()
        .rename(columns={"Course title": "No of Courses"})
        .sort_values("Year")
    )
    fig_area = px.area(
        courses_by_year,
        x="Year", y="No of Courses", color="Course Type",
        title="No. of Courses per Year"
    )

    # 4. Funding agencies bubble chart
    fund_agg = (
        dff1.groupby("Name of the Fund", as_index=False)["TotalAll"]
        .sum()
        .sort_values("TotalAll", ascending=False)
    )
    fig_funding = px.scatter(
        fund_agg,
        x="Name of the Fund", y="TotalAll",
        size="TotalAll", color="Name of the Fund",
        title="Participants by Funding Agency"
    )
    fig_funding.update_layout(xaxis={"visible": False, "showticklabels": False}, showlegend=False)
    
    # 5. Word cloud
    b64_img = make_wordcloud(dff1["Course Tags"])
    
    return fig_country, fig_map, fig_gender, fig_area, fig_funding, f"data:image/png;base64,{b64_img}"

if __name__ == "__main__":
    app.run(debug=True)
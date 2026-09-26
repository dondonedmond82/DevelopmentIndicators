import io
import os
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Scikit-Learn for ML baseline & NLP feature extraction
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PyTorch Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim

# ReportLab imports for generating side-by-side Graph + Text PDF report
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Initialize Panel extension with Plotly support
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

# ---------------------------------------------------------
# 0. COLORBLIND-SAFE BLUE / RED / GREEN PALETTE
#    Plain #0000FF / #FF0000 / #00FF00 are notoriously hard
#    to tell apart for red-green color blindness (the most
#    common type). These three hex values are the Okabe-Ito
#    colorblind-safe versions of blue, red and green - they
#    stay distinguishable under deuteranopia, protanopia and
#    tritanopia.
# ---------------------------------------------------------
COLOR_BLUE = "#0072B2"    # colorblind-safe blue
COLOR_RED = "#D55E00"     # colorblind-safe red (vermillion)
COLOR_GREEN = "#009E73"   # colorblind-safe green (bluish-green)

PALETTE = [COLOR_BLUE, COLOR_RED, COLOR_GREEN]           # for discrete/categorical series
# Diverging scale for +/- values (correlation): red <-> white <-> blue.
# Blue/red stays distinguishable for red-green color blindness, unlike
# a red<->green diverging scale which would collapse for those users.
DIVERGING_SCALE = [COLOR_RED, "#ffffff", COLOR_BLUE]
# Sequential scale for single-direction intensity (rankings, map values):
# light gray -> blue, since blue remains reliably perceivable across all
# common types of color blindness.
SEQUENTIAL_SCALE = ["#f0f0f0", COLOR_BLUE]

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING (WITH TYPE FIXES)
# ---------------------------------------------------------
@pn.cache
def load_data():
    df_wide = pd.read_csv('wdi_wide.csv')
    df_ts = pd.read_csv('wdi_timeseries.csv')

    # Fix: Ensure all numeric indicator columns are explicitly converted to float
    ts_non_numeric = ['Year', 'Country Name', 'Country Code', 'Region', 'Income Group']
    ts_numeric_cols = [col for col in df_ts.columns if col not in ts_non_numeric]
    for col in ts_numeric_cols:
        df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')

    wide_non_numeric = ['Country Name', 'Country Code', 'Region', 'Income Group']
    wide_numeric_cols = [col for col in df_wide.columns if col not in wide_non_numeric]
    for col in wide_numeric_cols:
        df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

    # Feature engineering for NLP & text analysis
    df_wide['Country_Summary_Text'] = (
        "Country " + df_wide['Country Name'].fillna('') +
        " located in " + df_wide['Region'].fillna('') +
        " is classified as " + df_wide['Income Group'].fillna('') +
        " income level. GDP per capita: $" + df_wide['GDP_per_capita'].fillna(0).round(2).astype(str) +
        ", Life expectancy: " + df_wide['Life_expectancy'].fillna(0).round(1).astype(str) + " years."
    )
    return df_wide, df_ts

df_wide, df_ts = load_data()

numeric_cols = [
    'GDP_per_capita', 'Life_expectancy', 'Population', 'CO2_per_capita',
    'Unemployment_pct', 'Literacy_pct', 'Health_exp_pct_gdp', 'GDP_growth_pct',
    'Urban_pop_pct', 'Internet_users_pct', 'Gini_index', 'Infant_mortality',
    'Electricity_access_pct', 'Maternal_mortality'
]

# ---------------------------------------------------------
# 2. WIDGET CONTROLLERS
# ---------------------------------------------------------
region_select = pn.widgets.MultiSelect(
    name='Select Regions',
    options=list(df_wide['Region'].dropna().unique()),
    value=list(df_wide['Region'].dropna().unique())[:3]
)

income_select = pn.widgets.Select(
    name='Filter by Income Group',
    options=['All'] + list(df_wide['Income Group'].dropna().unique()),
    value='All'
)

map_metric_select = pn.widgets.Select(
    name='Map / Chart Indicator Variable',
    options=['Life_expectancy', 'GDP_per_capita', 'CO2_per_capita', 'Internet_users_pct', 'Infant_mortality'],
    value='Life_expectancy'
)

top_n_slider = pn.widgets.IntSlider(
    name='Top N Countries (Horizontal Bar)',
    start=5,
    end=30,
    step=1,
    value=10
)

# Deep Learning Hyperparameters
epochs_slider = pn.widgets.IntSlider(
    name='Deep Learning Epochs',
    start=50,
    end=500,
    step=50,
    value=200
)

lr_select = pn.widgets.Select(
    name='PyTorch Learning Rate',
    options=[0.001, 0.005, 0.01, 0.05],
    value=0.01
)

target_var_select = pn.widgets.Select(
    name='Target Variable (DL & ML)',
    options=['Life_expectancy', 'GDP_per_capita', 'CO2_per_capita', 'Infant_mortality'],
    value='Life_expectancy'
)

def filter_dataframe(regions, income):
    filtered = df_wide[df_wide['Region'].isin(regions)]
    if income != 'All':
        filtered = filtered[filtered['Income Group'] == income]
    return filtered

# ---------------------------------------------------------
# 3. VISUALIZATION GENERATORS (ALL 9 GRAPH TYPES)
#    Discrete/categorical charts cycle through PALETTE
#    (colorblind-safe blue/red/green). Diverging values
#    (correlation) use DIVERGING_SCALE; single-direction
#    intensity (rankings, map) uses SEQUENTIAL_SCALE.
# ---------------------------------------------------------

# 1. Pie Chart
@pn.depends(region_select.param.value, income_select.param.value)
def get_pie_chart(regions, income):
    filtered = filter_dataframe(regions, income)
    counts = filtered['Income Group'].value_counts().reset_index()
    fig = px.pie(
        counts, values='count', names='Income Group',
        title="1. Income Group Distribution (Pie Chart)",
        hole=0.4, color_discrete_sequence=PALETTE
    )
    fig.update_traces(marker=dict(line=dict(color="#ffffff", width=1.5)))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

# 2. Histogram
@pn.depends(region_select.param.value, income_select.param.value, map_metric_select.param.value)
def get_hist_chart(regions, income, metric):
    filtered = filter_dataframe(regions, income)
    fig = px.histogram(
        filtered, x=metric, nbins=20,
        title=f"2. Distribution of {metric} (Histogram)",
        color_discrete_sequence=[COLOR_BLUE], template="plotly_white"
    )
    return fig

# 3. Bar Chart
@pn.depends(region_select.param.value, income_select.param.value)
def get_bar_chart(regions, income):
    filtered = filter_dataframe(regions, income)
    avg_df = filtered.groupby('Region')['GDP_per_capita'].mean().reset_index()
    fig = px.bar(
        avg_df, x='Region', y='GDP_per_capita',
        title="3. Mean GDP per Capita by Region (Bar Chart)",
        color='Region', color_discrete_sequence=PALETTE, template="plotly_white"
    )
    return fig

# 4. Barh Chart (ranked -> blue sequential intensity)
@pn.depends(region_select.param.value, income_select.param.value, top_n_slider.param.value, map_metric_select.param.value)
def get_barh_chart(regions, income, top_n, metric):
    filtered = filter_dataframe(regions, income).sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(
        filtered, x=metric, y='Country Name', orientation='h',
        title=f"4. Top {top_n} Countries by {metric} (Horizontal Bar)",
        color=metric, color_continuous_scale=SEQUENTIAL_SCALE
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# 5. Scatter Plot
@pn.depends(region_select.param.value, income_select.param.value)
def get_scatter_plot(regions, income):
    filtered = filter_dataframe(regions, income)
    fig = px.scatter(
        filtered, x='GDP_per_capita', y='Life_expectancy',
        size='Population', color='Region', color_discrete_sequence=PALETTE,
        hover_name='Country Name',
        log_x=True, title="5. GDP per Capita vs. Life Expectancy (Scatter Plot)",
        template="plotly_white"
    )
    return fig

# 6. Heatmap (correlation is +/- -> red/white/blue diverging scale)
@pn.depends(region_select.param.value, income_select.param.value)
def get_heatmap(regions, income):
    filtered = filter_dataframe(regions, income)
    corr = filtered[numeric_cols[:8]].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        title="6. Indicator Correlation Matrix (Heatmap)",
        color_continuous_scale=DIVERGING_SCALE, zmin=-1, zmax=1
    )
    return fig

# 7. Map (Choropleth, single-direction intensity -> blue sequential)
@pn.depends(region_select.param.value, income_select.param.value, map_metric_select.param.value)
def get_geo_map(regions, income, metric):
    filtered = filter_dataframe(regions, income)
    fig = px.choropleth(
        filtered, locations="Country Code",
        color=metric, hover_name="Country Name",
        color_continuous_scale=SEQUENTIAL_SCALE,
        title=f"7. Global Distribution of {metric} (Map)"
    )
    fig.update_geos(showframe=False, showcoastlines=True, projection_type='natural earth')
    return fig

# 8. Line Chart (Time Series)
@pn.depends(region_select.param.value, map_metric_select.param.value)
def get_line_chart(regions, metric):
    if metric not in df_ts.columns:
        metric = 'GDP_per_capita' if 'GDP_per_capita' in df_ts.columns else df_ts.columns[2]

    ts_filtered = df_ts[df_ts['Region'].isin(regions)].copy() if 'Region' in df_ts.columns else df_ts.copy()
    ts_filtered[metric] = pd.to_numeric(ts_filtered[metric], errors='coerce')
    avg_ts = ts_filtered.groupby('Year')[metric].mean().reset_index()

    fig = px.line(
        avg_ts, x='Year', y=metric,
        title=f"8. Historical Trend of {metric} (Line Chart)",
        markers=True, color_discrete_sequence=[COLOR_BLUE], template="plotly_white"
    )
    return fig

# 9. Boxplot
@pn.depends(region_select.param.value, income_select.param.value)
def get_boxplot(regions, income):
    filtered = filter_dataframe(regions, income)
    fig = px.box(
        filtered, x='Income Group', y='CO2_per_capita',
        color='Income Group', color_discrete_sequence=PALETTE, points="all",
        title="9. CO2 Emissions per Capita by Income Group (Boxplot)"
    )
    return fig

# Data Table
@pn.depends(region_select.param.value, income_select.param.value)
def get_table(regions, income):
    filtered = filter_dataframe(regions, income)
    stats = filtered[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    stats = stats.reset_index().rename(columns={'index': 'Indicator', '50%': 'median'})
    return pn.widgets.Tabulator(stats, pagination='remote', page_size=10, height=300)

# ---------------------------------------------------------
# 4. DEEP LEARNING WORKFLOW (PYTORCH MLP NEURAL NETWORK)
# ---------------------------------------------------------
class DeepRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

@pn.depends(target_var_select.param.value, epochs_slider.param.value, lr_select.param.value)
def run_dl_pipeline(target_col, epochs, lr):
    ml_df = df_wide[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col]).values
    y = ml_df[target_col].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    model = DeepRegressor(X_train_s.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()

    preds = scaler_y.inverse_transform(preds_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    loss_df = pd.DataFrame({'Epoch': range(1, epochs + 1), 'MSE Loss': loss_history})
    fig_loss = px.line(
        loss_df, x='Epoch', y='MSE Loss',
        title=f"PyTorch Neural Network Training Loss Curve ({epochs} Epochs, lr={lr})",
        color_discrete_sequence=[COLOR_BLUE], template="plotly_white"
    )

    dl_card = pn.Column(
        pn.pane.Markdown(f"### 🧠 PyTorch Deep Learning Model Results for Target: `{target_col}`"),
        pn.Row(
            pn.indicators.Number(name='Deep Learning R² Score', value=r2, format='{value:.3f}',
                                  colors=[(1, COLOR_GREEN)]),
            pn.indicators.Number(name='Deep Learning RMSE', value=rmse, format='{value:.3f}',
                                  colors=[(1, COLOR_RED)])
        ),
        pn.pane.Plotly(fig_loss)
    )
    return dl_card

# ---------------------------------------------------------
# 5. NLP WORKFLOW
# ---------------------------------------------------------
nlp_country_select = pn.widgets.Select(
    name='Select Country for Synthetic RAG / Prompt Context',
    options=list(df_wide['Country Name'].dropna().unique())
)

prompt_input = pn.widgets.TextAreaInput(
    name='LLM User Query Prompt',
    value="Summarize the key economic and social challenges for this country based on the retrieved context.",
    height=80
)

# button_type='primary' renders in Panel's blue, matching COLOR_BLUE
nlp_run_btn = pn.widgets.Button(name='Run LLM / RAG Pipeline', button_type='primary')

@pn.depends(nlp_country_select.param.value, nlp_run_btn.param.clicks)
def run_nlp_llm_workflow(country_name, clicks):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_wide['Country_Summary_Text'])

    country_row = df_wide[df_wide['Country Name'] == country_name].iloc[0]
    retrieved_context = country_row['Country_Summary_Text']

    feature_names = np.array(tfidf.get_feature_names_out())
    doc_idx = country_row.name
    row_weights = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = row_weights.argsort()[-5:][::-1]
    top_keywords = ", ".join(feature_names[top_indices])

    simulated_llm_response = (
        f"**[Simulated LLM Agent Response via Context Augmented Prompting]**\n\n"
        f"**Target Entity:** {country_name}\n"
        f"**Context Vectors / Keywords:** `{top_keywords}`\n\n"
        f"**Analysis:**\n"
        f"{country_name} is located in the **{country_row['Region']}** region with an income classification of "
        f"**{country_row['Income Group']}**. Key statistical metrics include a GDP per capita of **${country_row['GDP_per_capita']:,.2f}** "
        f"and average life expectancy of **{country_row['Life_expectancy']} years**."
    )

    # alert_type='primary' keeps the callout in the blue family
    return pn.Column(
        pn.pane.Markdown("#### 🔍 1. Retrieved Document Context (Vector DB / RAG)"),
        pn.pane.Alert(retrieved_context, alert_type='primary'),
        pn.pane.Markdown("#### 🤖 2. Generated Response (LLM Engine)"),
        pn.pane.Markdown(simulated_llm_response)
    )

# ---------------------------------------------------------
# 6. PDF GENERATION ENGINE WITH SIDE-BY-SIDE GRAPH + TEXT
# ---------------------------------------------------------
def generate_pdf_report():
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(COLOR_BLUE),
        spaceAfter=15
    )
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2c2c2c")  # neutral dark gray for body readability
    )

    story = [
        Paragraph("World Development Indicators & Deep Learning Report", title_style),
        Paragraph(f"<b>Filter Context:</b> Regions: {', '.join(region_select.value)} | Income Group: {income_select.value}", body_style),
        Spacer(1, 15)
    ]

    items_to_export = [
        (
            get_pie_chart(region_select.value, income_select.value),
            "<b>1. Pie Chart: Income Group Distribution</b><br/><br/>Displays the proportional representation of global income tiers within the current dataset filter."
        ),
        (
            get_hist_chart(region_select.value, income_select.value, map_metric_select.value),
            f"<b>2. Histogram: {map_metric_select.value} Distribution</b><br/><br/>Shows the frequency distribution and spread for the selected indicator across countries."
        ),
        (
            get_bar_chart(region_select.value, income_select.value),
            "<b>3. Bar Chart: Mean GDP per Capita</b><br/><br/>Compares average national economic output across selected global regions."
        ),
        (
            get_barh_chart(region_select.value, income_select.value, top_n_slider.value, map_metric_select.value),
            f"<b>4. Horizontal Bar Chart: Top {top_n_slider.value} Ranking</b><br/><br/>Identifies leading countries ranked by the selected development metric."
        ),
        (
            get_scatter_plot(region_select.value, income_select.value),
            "<b>5. Scatter Plot: GDP vs. Life Expectancy</b><br/><br/>Visualizes relationship dynamics between economic capacity and health outcomes on a log scale."
        ),
        (
            get_heatmap(region_select.value, income_select.value),
            "<b>6. Heatmap: Indicator Correlations</b><br/><br/>Presents pairwise correlation coefficients between macro-economic and demographic metrics."
        ),
        (
            get_geo_map(region_select.value, income_select.value, map_metric_select.value),
            f"<b>7. Geospatial Map: Global {map_metric_select.value}</b><br/><br/>Choropleth mapping depicting geographical patterns for the selected variable."
        ),
        (
            get_line_chart(region_select.value, map_metric_select.value),
            f"<b>8. Line Chart: Historical Trend of {map_metric_select.value}</b><br/><br/>Displays aggregated longitudinal patterns over time."
        ),
        (
            get_boxplot(region_select.value, income_select.value),
            "<b>9. Boxplot: CO2 Emissions Dispersion</b><br/><br/>Illustrates median values, quartiles, and outlier data points for carbon footprints across income brackets."
        )
    ]

    table_data = []

    for fig, description in items_to_export:
        img_bytes = pio.to_image(fig, format='png', width=450, height=300, scale=2)
        img_buf = io.BytesIO(img_bytes)
        img = RLImage(img_buf, width=3.3 * inch, height=2.2 * inch)
        text_p = Paragraph(description, body_style)

        table_data.append([img, text_p])

    report_table = Table(table_data, colWidths=[3.5 * inch, 3.5 * inch])
    report_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
    ]))

    story.append(report_table)
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# button_type='primary' keeps this in the blue family
pdf_download_button = pn.widgets.FileDownload(
    callback=generate_pdf_report,
    filename="WDI_Comprehensive_Report.pdf",
    label="📄 Export Report to PDF",
    button_type="primary",
    sizing_mode="stretch_width"
)

# ---------------------------------------------------------
# 7. PANEL DASHBOARD LAYOUT
# ---------------------------------------------------------
sidebar = pn.Column(
    "## ⚙️ Controls",
    region_select,
    income_select,
    map_metric_select,
    top_n_slider,
    pn.layout.Divider(),
    "### 📄 Export Options",
    pdf_download_button,
    pn.layout.Divider(),
    "### 🧠 Deep Learning Parameters",
    target_var_select,
    epochs_slider,
    lr_select,
    width=300
)

tabs = pn.Tabs(
    ("📊 Visual Analytics (9 Plot Types)", pn.Column(
        pn.Row(pn.pane.Plotly(get_pie_chart), pn.pane.Plotly(get_hist_chart)),
        pn.Row(pn.pane.Plotly(get_bar_chart), pn.pane.Plotly(get_barh_chart)),
        pn.Row(pn.pane.Plotly(get_scatter_plot), pn.pane.Plotly(get_heatmap)),
        pn.Row(pn.pane.Plotly(get_geo_map), pn.pane.Plotly(get_line_chart)),
        pn.Row(pn.pane.Plotly(get_boxplot)),
        pn.pane.Markdown("### 📋 Descriptive Statistics Data Table"),
        get_table
    )),
    ("🧠 PyTorch Deep Learning Model", pn.Column(
        run_dl_pipeline
    )),
    ("💬 NLP & Modern LLM RAG Workflow", pn.Column(
        pn.pane.Markdown("### Modern RAG & Text Analytics Pipeline"),
        nlp_country_select,
        prompt_input,
        nlp_run_btn,
        pn.layout.Divider(),
        run_nlp_llm_workflow
    ))
)

# accent_base_color and header_background set to the colorblind-safe blue
template = pn.template.FastListTemplate(
    title="World Development Indicators - Data Science, PyTorch Deep Learning & LLM Portal",
    sidebar=[sidebar],
    main=[tabs],
    accent_base_color=COLOR_BLUE,
    header_background=COLOR_BLUE
)

template.servable()
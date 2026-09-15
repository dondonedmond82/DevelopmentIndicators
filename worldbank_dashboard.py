import io
import os
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@pn.cache
def load_data():
    df_wide = pd.read_csv('wdi_wide.csv')
    df_ts = pd.read_csv('wdi_timeseries.csv')

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

# 1. Dropdown Menu for filtering charts by Income Group
income_select = pn.widgets.Select(
    name='Filter by Income Group',
    options=['All'] + list(df_wide['Income Group'].dropna().unique()),
    value='All'
)

# 2. Dropdown Menu for choosing indicator in charts
map_metric_select = pn.widgets.Select(
    name='Map / Chart Indicator Variable',
    options=['Life_expectancy', 'GDP_per_capita', 'CO2_per_capita', 'Internet_users_pct', 'Infant_mortality'],
    value='Life_expectancy'
)

# 3. IntSlider Widget for controlling top N items in horizontal bar chart
top_n_slider = pn.widgets.IntSlider(
    name='Top N Countries (Horizontal Bar)',
    start=5,
    end=30,
    step=1,
    value=10
)

# 4. IntSlider Widget for Random Forest Estimators in ML model
n_estimators_slider = pn.widgets.IntSlider(
    name='ML Random Forest Estimators',
    start=10,
    end=200,
    step=10,
    value=100
)

target_var_select = pn.widgets.Select(
    name='ML Target Variable',
    options=['Life_expectancy', 'GDP_per_capita', 'CO2_per_capita', 'Infant_mortality'],
    value='Life_expectancy'
)

# Helper function to filter dataframe based on sidebar selections
def filter_dataframe(regions, income):
    filtered = df_wide[df_wide['Region'].isin(regions)]
    if income != 'All':
        filtered = filtered[filtered['Income Group'] == income]
    return filtered

# ---------------------------------------------------------
# 3. COMPONENT GENERATORS (VISUALIZATIONS)
# ---------------------------------------------------------

# 1. Table: Summary Statistics
@pn.depends(region_select.param.value, income_select.param.value)
def get_table(regions, income):
    filtered = filter_dataframe(regions, income)
    stats = filtered[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    stats = stats.reset_index().rename(columns={'index': 'Indicator', '50%': 'median'})
    return pn.widgets.Tabulator(stats, pagination='remote', page_size=10, height=300)

# 2. Pie Chart: Distribution by Income Group
@pn.depends(region_select.param.value, income_select.param.value)
def get_pie_chart(regions, income):
    filtered = filter_dataframe(regions, income)
    counts = filtered['Income Group'].value_counts().reset_index()
    fig = px.pie(
        counts, values='count', names='Income Group',
        title="Income Group Distribution",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

# 3. Bar Chart: Average Indicator by Region
@pn.depends(region_select.param.value, income_select.param.value)
def get_bar_chart(regions, income):
    filtered = filter_dataframe(regions, income)
    avg_df = filtered.groupby('Region')['GDP_per_capita'].mean().reset_index()
    fig = px.bar(
        avg_df, x='Region', y='GDP_per_capita',
        title="Mean GDP per Capita by Region ($)",
        color='Region', template="plotly_white"
    )
    return fig

# 4. Barh Chart: Top N Countries
@pn.depends(region_select.param.value, income_select.param.value, top_n_slider.param.value, map_metric_select.param.value)
def get_barh_chart(regions, income, top_n, metric):
    filtered = filter_dataframe(regions, income).sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(
        filtered, x=metric, y='Country Name', orientation='h',
        title=f"Top {top_n} Countries by {metric}",
        color=metric, color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# 5. Scatter Plot: GDP vs Life Expectancy
@pn.depends(region_select.param.value, income_select.param.value)
def get_scatter_plot(regions, income):
    filtered = filter_dataframe(regions, income)
    fig = px.scatter(
        filtered, x='GDP_per_capita', y='Life_expectancy',
        size='Population', color='Region', hover_name='Country Name',
        log_x=True, title="GDP per Capita vs. Life Expectancy (Log Scale)",
        template="plotly_white"
    )
    return fig

# 6. Heatmap: Correlation Matrix
@pn.depends(region_select.param.value, income_select.param.value)
def get_heatmap(regions, income):
    filtered = filter_dataframe(regions, income)
    corr = filtered[numeric_cols[:8]].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    return fig

# 7. Boxplot: Distribution Across Income Groups
@pn.depends(region_select.param.value, income_select.param.value)
def get_boxplot(regions, income):
    filtered = filter_dataframe(regions, income)
    fig = px.box(
        filtered, x='Income Group', y='CO2_per_capita',
        color='Income Group', points="all",
        title="CO2 Emissions per Capita by Income Level"
    )
    return fig

# 8. Geospatial Map: World Map Indicator Visualizer
@pn.depends(region_select.param.value, income_select.param.value, map_metric_select.param.value)
def get_geo_map(regions, income, metric):
    filtered = filter_dataframe(regions, income)
    fig = px.choropleth(
        filtered, locations="Country Code",
        color=metric,
        hover_name="Country Name",
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global {metric} Map"
    )
    fig.update_geos(showframe=False, showcoastlines=True, projection_type='natural earth')
    return fig

# ---------------------------------------------------------
# 4. ADVANCED MACHINE LEARNING WORKFLOW
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value, n_estimators_slider.param.value)
def run_ml_pipeline(target_col, n_estimators):
    ml_df = df_wide[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importance = importance.sort_values('Importance', ascending=True)

    fig = px.bar(
        importance, x='Importance', y='Feature', orientation='h',
        title=f"Feature Importances for Predicting {target_col} (n_estimators={n_estimators})",
        color='Importance', color_continuous_scale='Viridis'
    )

    metrics_card = pn.Column(
        pn.pane.Markdown(f"### ML Model Results: `{target_col}`"),
        pn.Row(
            pn.indicators.Number(name='R² Score', value=r2, format='{value:.3f}'),
            pn.indicators.Number(name='RMSE', value=rmse, format='{value:.3f}')
        ),
        pn.pane.Plotly(fig)
    )
    return metrics_card

# ---------------------------------------------------------
# 5. NLP & MODERN LLM APPLICATION WORKFLOW
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
        f"and average life expectancy of **{country_row['Life_expectancy']} years**.\n\n"
        f"*LLM Recommendation Engine Notice:* Based on context, target intervention should focus on "
        f"{'infrastructure and industrialization' if country_row['GDP_per_capita'] < 5000 else 'sustainable growth and high-tech workforce deployment'}."
    )

    return pn.Column(
        pn.pane.Markdown("#### 🔍 1. Retrieved Document Context (Vector DB / RAG)"),
        pn.pane.Alert(retrieved_context, alert_type='info'),
        pn.pane.Markdown("#### 🤖 2. Generated Response (LLM Engine)"),
        pn.pane.Markdown(simulated_llm_response)
    )

# ---------------------------------------------------------
# 6. DYNAMIC PDF GENERATION ENGINE
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
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=15
    )
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2c3e50')
    )

    story = [
        Paragraph("World Development Indicators Report", title_style),
        Paragraph(f"<b>Filter Context:</b> Regions: {', '.join(region_select.value)} | Income Group: {income_select.value}", body_style),
        Spacer(1, 15)
    ]

    items_to_export = [
        (
            get_pie_chart(region_select.value, income_select.value),
            "<b>Income Group Distribution</b><br/><br/>This pie chart displays the overall ratio of countries categorized by income levels within your active regional selection."
        ),
        (
            get_bar_chart(region_select.value, income_select.value),
            "<b>Mean GDP per Capita by Region</b><br/><br/>This chart highlights economic disparities across regions, demonstrating average gross domestic product values."
        ),
        (
            get_barh_chart(region_select.value, income_select.value, top_n_slider.value, map_metric_select.value),
            f"<b>Top {top_n_slider.value} Countries by {map_metric_select.value}</b><br/><br/>Horizontal ranking indicating top performing entities for the selected indicator variable."
        ),
        (
            get_scatter_plot(region_select.value, income_select.value),
            "<b>GDP per Capita vs. Life Expectancy</b><br/><br/>A log-scale scatter analysis illustrating health outcomes plotted against economic metrics."
        ),
        (
            get_heatmap(region_select.value, income_select.value),
            "<b>Feature Correlation Matrix</b><br/><br/>Linear correlations between primary macro-economic and social indicators."
        ),
        (
            get_boxplot(region_select.value, income_select.value),
            "<b>CO2 Emissions per Capita</b><br/><br/>Distribution and outliers of carbon footprints grouped by global income tiers."
        )
    ]

    table_data = []

    for fig, text in items_to_export:
        img_bytes = pio.to_image(fig, format='png', width=450, height=300, scale=2)
        img_buf = io.BytesIO(img_bytes)
        img = RLImage(img_buf, width=3.3 * inch, height=2.2 * inch)

        text_p = Paragraph(text, body_style)

        # Place image on left, text on right side
        table_data.append([img, text_p])

    report_table = Table(table_data, colWidths=[3.5 * inch, 3.5 * inch])
    report_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
    ]))

    story.append(report_table)
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# PDF File Download Widget for Left Sidebar
pdf_download_button = pn.widgets.FileDownload(
    callback=generate_pdf_report,
    filename="WDI_Report.pdf",
    label="📄 Export Report to PDF",
    button_type="success",
    sizing_mode="stretch_width"
)

# Needed by reportlab export helper to render Plotly figures
import plotly.io as pio

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
    "### 🤖 ML Parameters",
    target_var_select,
    n_estimators_slider,
    width=300
)

tabs = pn.Tabs(
    ("📊 Statistical Analytics", pn.Column(
        pn.Row(pn.pane.Plotly(get_pie_chart), pn.pane.Plotly(get_bar_chart)),
        pn.Row(pn.pane.Plotly(get_barh_chart), pn.pane.Plotly(get_scatter_plot)),
        pn.Row(pn.pane.Plotly(get_heatmap), pn.pane.Plotly(get_boxplot)),
        pn.Row(pn.pane.Plotly(get_geo_map)),
        pn.pane.Markdown("### 📋 Descriptive Statistics Data Table"),
        get_table
    )),
    ("🌲 Machine Learning Predictive Engine", pn.Column(
        run_ml_pipeline
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

template = pn.template.FastListTemplate(
    title="World Development Indicators - Data Science, ML & LLM Portal",
    sidebar=[sidebar],
    main=[tabs],
    accent_base_color="#1f77b4",
    header_background="#1f77b4"
)

# ==================================================================
# SERVE APPLICATION
# ==================================================================

template.servable()
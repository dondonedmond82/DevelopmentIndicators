# World Development Indicators: Data Science, ML & LLM Portal

An end-to-end interactive data science application built with **Python**, **Panel**, **Plotly**, **scikit-learn**, and **TF-IDF/RAG simulation workflows**. This dashboard processes global economic, social, and environmental indicators to deliver descriptive statistics, predictive machine learning, and context-augmented text analysis.

---

## 🌟 Overview & Features

* **Interactive Analytical Dashboard**: Built on Panel and Plotly, featuring dynamic filtering by regions and income groups.
* **8 Core Data Visualizations & Data Table**:
  * **Summary Statistics Table**: Interactive `Tabulator` providing count, mean, std, min, median, and max across indicators.
  * **Income Group Distribution**: Donut/Pie chart showing global economic class distributions.
  * **Mean GDP by Region**: Grouped bar chart comparing average GDP per capita.
  * **Top N Indicator Ranking**: Dynamic horizontal bar chart controlled by an `IntSlider`.
  * **GDP vs. Life Expectancy**: Logarithmic scatter plot with population sized nodes.
  * **Correlation Heatmap**: Pearson correlation matrix across numerical WDI indicators.
  * **CO2 Emissions Boxplot**: Distribution and outlier analysis across income levels.
  * **Geospatial World Map**: Choropleth visualization mapping key indicators globally.
* **Machine Learning Predictive Engine**: Random Forest Regression pipeline predicting key development metrics (`Life_expectancy`, `GDP_per_capita`, `CO2_per_capita`, `Infant_mortality`) with dynamic `n_estimators` control and real-time $R^2$ and RMSE metric evaluation.
* **NLP & Modern LLM RAG Workflow**: Synthetic Retrieval-Augmented Generation (RAG) pipeline utilizing TF-IDF vector retrieval and structured prompt augmentation.

---

## 🏗️ System Architecture
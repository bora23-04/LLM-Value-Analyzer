import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. PAGE CONFIGURATION ---
# Set the page to wide layout, add a title and icon
st.set_page_config(layout="wide", page_title="LLM Cost-Benefit Analysis")


# --- 2. DATA LOADING & FEATURE ENGINEERING ---
# Use @st.cache_data to load and process the data only once
@st.cache_data
def load_and_process_data():
    try:
        # Load the real dataset
        df = pd.read_csv('llm_comparison_dataset.csv')
    except FileNotFoundError:
        # If the file isn't found, stop the app and show an error
        st.error("FATAL ERROR: 'llm_comparison_dataset.csv' not found.")
        st.error("Please make sure your dataset file is in the same folder as the app.")
        return None  # Return None to stop processing

    # --- Feature Engineering ---
    # Create the 'value' columns
    df['performance_per_dollar'] = df['Benchmark (MMLU)'] / (df['Price / Million Tokens'] + 1e-6)
    df['speed_per_dollar'] = df['Speed (tokens/sec)'] / (df['Price / Million Tokens'] + 1e-6)

    # Create the categorical 'Model Type' column
    df['Model Type'] = df['Open-Source'].map({0: 'Proprietary (No)', 1: 'Open-Source (Yes)'})

    # Ensure ratings are integers for clean filtering
    rating_cols = ['Quality Rating', 'Speed Rating', 'Price Rating', 'Energy Efficiency']
    for col in rating_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# Load the data
df_full = load_and_process_data()

# Check if data loading failed
if df_full is None:
    st.stop()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("🤖 LLM Dashboard Filters")

# --- Provider and Model Type Filters ---
all_providers = sorted(df_full['Provider'].unique())
all_model_types = sorted(df_full['Model Type'].unique())

selected_providers = st.sidebar.multiselect(
    "Select Providers",
    options=all_providers,
    default=all_providers
)

selected_model_types = st.sidebar.multiselect(
    "Select Model Type",
    options=all_model_types,
    default=all_model_types
)

# --- Rating Filters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Filter by Simple Ratings (1-5)")

# Create lists for the 1-5 rating options
rating_options = sorted(df_full['Quality Rating'].unique())

selected_quality = st.sidebar.multiselect(
    "Select Quality Rating(s)",
    options=rating_options,
    default=rating_options  # Default to all ratings selected
)

selected_speed = st.sidebar.multiselect(
    "Select Speed Rating(s)",
    options=rating_options,
    default=rating_options
)

selected_price = st.sidebar.multiselect(
    "Select Price Rating(s)",
    options=rating_options,
    default=rating_options
)

# --- NEW: Energy Efficiency Filter ---
energy_options = sorted(df_full['Energy Efficiency'].unique())
selected_energy = st.sidebar.multiselect(
    "Select Energy Efficiency Rating(s)",
    options=energy_options,
    default=energy_options
)

# --- NEW: Context Window and Training Data Size Filters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Filter by Technical Specs")

# --- Context Window Filter ---
context_options = sorted(df_full['Context Window'].unique())
selected_context = st.sidebar.multiselect(
    "Select Context Window(s)",
    options=context_options,
    default=context_options
)

# --- Training Data Size Filter (Slider) ---
min_train_size = int(df_full['Training Dataset Size'].min())
max_train_size = int(df_full['Training Dataset Size'].max())

selected_train_size = st.sidebar.slider(
    "Select Training Data Size (Range)",
    min_value=min_train_size,
    max_value=max_train_size,
    value=(min_train_size, max_train_size)  # Default to full range
)

# --- UPDATED: Filter the dataframe based on all selections ---
if not selected_providers:
    selected_providers = all_providers
if not selected_model_types:
    selected_model_types = all_model_types
if not selected_quality:
    selected_quality = rating_options
if not selected_speed:
    selected_speed = rating_options
if not selected_price:
    selected_price = rating_options
if not selected_energy:
    selected_energy = energy_options
if not selected_context:
    selected_context = context_options

df = df_full[
    (df_full['Provider'].isin(selected_providers)) &
    (df_full['Model Type'].isin(selected_model_types)) &
    (df_full['Quality Rating'].isin(selected_quality)) &
    (df_full['Speed Rating'].isin(selected_speed)) &
    (df_full['Price Rating'].isin(selected_price)) &
    (df_full['Energy Efficiency'].isin(selected_energy)) &
    (df_full['Context Window'].isin(selected_context)) &
    (df_full['Training Dataset Size'] >= selected_train_size[0]) &
    (df_full['Training Dataset Size'] <= selected_train_size[1])
    ]

# Check if the dataframe is empty after filtering
if df.empty:
    st.warning("No models match your filter criteria. Please adjust your selection.")
    st.stop()  # Stop the script from running further

# --- 4. MAIN PAGE TITLE & RECOMMENDATIONS ---
st.title("🤖 LLM Cost-Benefit Analysis Dashboard")
st.markdown("""
This dashboard answers the key business question: **"Which LLM provides the best balance of performance, speed, and cost?"**
Use the filters on the left to narrow your search.
""")

st.header("🏆 Key Recommendations (Based on Filters)")

# Calculate key metrics from the *filtered* data
best_value_model = df.loc[df['performance_per_dollar'].idxmax()]
fastest_value_model = df.loc[df['speed_per_dollar'].idxmax()]
smartest_model = df.loc[df['Benchmark (Chatbot Arena)'].idxmax()]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Best Overall Value (Smartness per $)",
        value=best_value_model['Model'],
        help=f"This model has the highest 'performance_per_dollar' score ({best_value_model['performance_per_dollar']:.2f})."
    )
with col2:
    st.metric(
        label="Best Value for Speed (Speed per $)",
        value=fastest_value_model['Model'],
        help=f"This model has the highest 'speed_per_dollar' score ({fastest_value_model['speed_per_dollar']:.2f})."
    )
with col3:
    st.metric(
        label="Highest Human Preference",
        value=smartest_model['Model'],
        help=f"This model has the highest 'Chatbot Arena' score ({smartest_model['Benchmark (Chatbot Arena)']})."
    )

st.markdown("---")

# --- 5. MAIN DASHBOARD CHARTS ---
st.header("Main Dashboard: The Value Quadrant")

# Chart 1: The "Value Quadrant"
median_price = df['Price / Million Tokens'].median()
median_benchmark = df['Benchmark (MMLU)'].median()

fig_quadrant = px.scatter(
    df,
    x='Price / Million Tokens',
    y='Benchmark (MMLU)',
    color='Model Type',
    size='Speed (tokens/sec)',
    hover_data=['Model', 'Provider', 'Price / Million Tokens', 'Benchmark (MMLU)'],
    title='LLM Value Quadrant: Performance vs. Cost'
)
fig_quadrant.add_vline(x=median_price, line_dash="dash", line_color="red",
                       annotation_text=f"Median Price (${median_price:.2f})")
fig_quadrant.add_hline(y=median_benchmark, line_dash="dash", line_color="blue",
                       annotation_text=f"Median Benchmark ({median_benchmark})")
fig_quadrant.update_layout(xaxis_title='Price / Million Tokens (Cost)',
                           yaxis_title='Benchmark (MMLU) Score (Performance)')
st.plotly_chart(fig_quadrant, use_container_width=True)

# --- 6. TOP 15 CHARTS (Side-by-Side) ---
col_val, col_speed = st.columns(2)

with col_val:
    # Chart 2: The "Best Value" Winners
    df_top_value = df.sort_values(by='performance_per_dollar', ascending=False).head(15)
    fig_bar_value = px.bar(
        df_top_value.sort_values(by='performance_per_dollar', ascending=True),
        x='performance_per_dollar',
        y='Model',
        orientation='h',
        title='Top 15 "Best Value" Models (Smartness per $)',
        color='performance_per_dollar',
        color_continuous_scale='magma',
        hover_data=['Model', 'performance_per_dollar', 'Benchmark (MMLU)', 'Price / Million Tokens']
    )
    fig_bar_value.update_layout(xaxis_title='Performance per Dollar (Higher is Better)', yaxis_title='Model',
                                coloraxis_showscale=False)
    st.plotly_chart(fig_bar_value, use_container_width=True)

with col_speed:
    # Chart 3: The "Fastest Value" Winners
    df_top_speed = df.sort_values(by='speed_per_dollar', ascending=False).head(15)
    fig_bar_speed = px.bar(
        df_top_speed.sort_values(by='speed_per_dollar', ascending=True),
        x='speed_per_dollar',
        y='Model',
        orientation='h',
        title='Top 15 "Fastest Value" Models (Speed per $)',
        color='speed_per_dollar',
        color_continuous_scale='plasma',
        hover_data=['Model', 'speed_per_dollar', 'Speed (tokens/sec)', 'Price / Million Tokens']
    )
    fig_bar_speed.update_layout(xaxis_title='Speed per Dollar (Higher is Better)', yaxis_title='Model',
                                coloraxis_showscale=False)
    st.plotly_chart(fig_bar_speed, use_container_width=True)

st.markdown("---")

# --- 7. EXPLORATORY CHARTS (in an expander) ---
with st.expander("See all Exploratory Charts"):
    col_explore1, col_explore2 = st.columns(2)

    with col_explore1:
        # Chart 4: "Buy vs. Build"
        df_avg_value = df.groupby('Model Type')['performance_per_dollar'].mean().reset_index()
        fig_buy_build = px.bar(
            df_avg_value, x='Model Type', y='performance_per_dollar',
            title='"Buy vs. Build": Average Value (Performance per $1)',
            color='Model Type', labels={'performance_per_dollar': 'Average Performance per Dollar'}
        )
        st.plotly_chart(fig_buy_build, use_container_width=True)

        # Chart 5: Top 10 Smartest (Chatbot Arena)
        df_top_smart = df.sort_values(by='Benchmark (Chatbot Arena)', ascending=True).tail(10)
        fig_smart = px.bar(
            df_top_smart, x='Benchmark (Chatbot Arena)', y='Model', orientation='h',
            title='Top 10 "Smartest" Models (Human Preference)', color='Benchmark (Chatbot Arena)',
            color_continuous_scale='cividis_r', text_auto=True
        )
        fig_smart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_smart, use_container_width=True)

        # Chart 6: Top 10 Fastest (Raw Speed)
        df_top_fast = df.sort_values(by='Speed (tokens/sec)', ascending=True).tail(10)
        fig_fast = px.bar(
            df_top_fast, x='Speed (tokens/sec)', y='Model', orientation='h',
            title='Top 10 Fastest Models (Raw Speed)', color='Speed (tokens/sec)',
            color_continuous_scale='plasma', text_auto=True
        )
        fig_fast.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fast, use_container_width=True)

        # Chart 7: Top 10 Most Expensive
        df_top_price = df.sort_values(by='Price / Million Tokens', ascending=True).tail(10)
        fig_price = px.bar(
            df_top_price, x='Price / Million Tokens', y='Model', orientation='h',
            title='Top 10 Most Expensive Models', color='Price / Million Tokens',
            color_continuous_scale='hot_r', text_auto=True
        )
        fig_price.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_price, use_container_width=True)

    with col_explore2:
        # Chart 8: Models per Provider
        df_provider_count = df.groupby('Provider')['Model'].count().reset_index(name='Model Count').sort_values(
            by='Model Count', ascending=True)
        fig_provider = px.bar(
            df_provider_count,
            x='Model Count',
            y='Provider',
            orientation='h',
            title='Number of Models per Provider',
            color='Provider',
            text_auto=True
        )
        fig_provider.update_layout(legend_title='Provider', yaxis_title=None, showlegend=False)
        st.plotly_chart(fig_provider, use_container_width=True)

        # Chart 9: Top 10 Lowest Latency
        df_top_latency = df.sort_values(by='Latency (sec)', ascending=True).head(10)
        fig_latency = px.bar(
            df_top_latency.sort_values(by='Latency (sec)', ascending=False),
            x='Latency (sec)', y='Model', orientation='h',
            title='Top 10 Most Responsive Models (Lowest Latency)', color='Latency (sec)',
            color_continuous_scale='greens_r', text_auto=True
        )
        fig_latency.update_layout(xaxis_title='Latency (sec) (Lower is Better)', coloraxis_showscale=False)
        st.plotly_chart(fig_latency, use_container_width=True)

        # Chart 10: Price vs. Training Data Size
        fig_price_data = px.scatter(
            df, x='Training Dataset Size', y='Price / Million Tokens',
            color='Model Type', hover_data=['Model'],
            title='Exploratory: Price vs. Training Data Size'
        )
        st.plotly_chart(fig_price_data, use_container_width=True)

        # Chart 11: Context Window vs. Energy Efficiency
        fig_context_energy = px.scatter(
            df, x='Context Window', y='Energy Efficiency',
            color='Model Type', hover_data=['Model'],
            title='Exploratory: Context Window vs. Energy Efficiency'
        )
        st.plotly_chart(fig_context_energy, use_container_width=True)

# --- 8. SHOW THE RAW DATA ---
st.markdown("---")
# This section shows the full, original, unfiltered dataset
st.header("Full Project Dataset (with new 'Value' columns)")
st.dataframe(df_full)
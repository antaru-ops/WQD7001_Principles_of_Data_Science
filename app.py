import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SDG 7 Policy Simulator", layout="wide", page_icon="🌍")
st.title("🌍 SDG 7 Policy & Energy Simulator")
st.markdown("**Developed for Group Assignment 2 | Master of Data Science**")
st.markdown("---")

# --- DATA CACHING (Memoization for speed) ---
@st.cache_data
def load_data():
    # Make sure 'cleaned_global_energy.csv' is in the same folder as this script
    return pd.read_csv('cleaned_global_energy.csv')

try:
    df_long = load_data()
except FileNotFoundError:
    st.error("⚠️ Error: 'cleaned_global_energy.csv' not found. Please ensure it is in the same folder as app.py.")
    st.stop()

# --- TRAIN GLOBAL K-MEANS MODEL ONCE ---
cluster_data = df_long[(df_long['Entity_Type'] == 'Country') & (df_long['Year'] == 2022)]
pivot_df = cluster_data.pivot(index='Country_or_Region', columns='Energy_Type', values='Energy_Value').dropna(subset=['Share of renewables in electri', 'Average CO2 emission factor'])
X_train = pivot_df[['Share of renewables in electri', 'Average CO2 emission factor']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# FIX: Unbreakable Dynamic Cluster Mapping
# Find the actual mathematical centers of the 3 clusters
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centers, columns=['Renewables', 'CO2'])
centroid_df['Cluster_ID'] = centroid_df.index

# 1. Identify Green Pioneer (Highest Renewables)
green_id = centroid_df.loc[centroid_df['Renewables'].idxmax(), 'Cluster_ID']

# 2. Identify Fossil Dependent (Highest CO2)
fossil_id = centroid_df.loc[centroid_df['CO2'].idxmax(), 'Cluster_ID']

# 3. Identify Transitioning Nation (The one left over)
all_ids = {0, 1, 2}
transition_id = list(all_ids - {green_id, fossil_id})[0]

# Map them correctly
cluster_map = {
    green_id: 'Green Pioneer 🟢',
    transition_id: 'Transitioning Nation 🟡',
    fossil_id: 'Fossil Dependent 🔴'
}

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation Panel")
page = st.sidebar.radio("Select Analytical Module:", [
    "1. 'What-If' Policy Profiler", 
    "2. Policy Gap Forecaster (2030)"
])

# ==========================================
# MODULE 1: THE 'WHAT-IF' PROFILER
# ==========================================
if page == "1. 'What-If' Policy Profiler":
    st.header("🎛️ Module 1: Prescriptive Policy Profiler (K-Means)")
    st.write("Drafting a new energy bill? Enter your hypothetical targets below to instantly mathematically classify your proposed global standing.")
    
    col1, col2 = st.columns(2)
    with col1:
        user_renewable = st.slider("Target Share of Renewables (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
    with col2:
        user_co2 = st.slider("Target CO2 Emission Factor", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        
    st.markdown("### 🤖 Algorithmic Classification Result:")
    
    # ML Inference
    user_input_scaled = scaler.transform([[user_renewable, user_co2]])
    prediction = kmeans.predict(user_input_scaled)[0]
    predicted_name = cluster_map[prediction]
    
    # Text Output
    if "Green Pioneer" in predicted_name:
        st.success(f"**Classification: {predicted_name}**\n\nExcellent. This policy profile aligns with top-tier sustainable economies. No further developmental aid required; focus on technology export.")
    elif "Transitioning Nation" in predicted_name:
        st.warning(f"**Classification: {predicted_name}**\n\nModerate progress. This policy keeps you in the global middle-class. Consider applying for international green financing to accelerate infrastructure development.")
    else:
        st.error(f"**Classification: {predicted_name}**\n\nCRITICAL ALERT. This policy profile is highly reliant on pollutive fuels and fails SDG targets. Immediate structural intervention required.")

    # --- DYNAMIC DECISION BOUNDARY PLOT ---
    st.markdown("#### 🗺️ Visualizing the Policy's Mathematical Distance")
    
    # 1. Setup custom colors mapped to the dynamic IDs (0, 1, 2)
    light_colors = [''] * 3
    light_colors[green_id] = '#D4EFDF'       # Light Green Zone
    light_colors[transition_id] = '#FDEBD0'  # Light Yellow Zone
    light_colors[fossil_id] = '#FADBD8'      # Light Red Zone
    cmap_light = ListedColormap(light_colors)

    dark_colors = {
        green_id: '#228B22',      # Dark Forest Green
        transition_id: '#D68910', # Dark Orange/Yellow
        fossil_id: '#CB4335'      # Dark Red
    }

    # 2. Build the meshgrid for the background zones
    h = 0.05  # Step size (optimized for Streamlit speed)
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the zones
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 3. Draw the Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Paint the background zones
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    # Plot the historical country data points
    for cid in [green_id, transition_id, fossil_id]:
        idx = kmeans.labels_ == cid
        ax.scatter(X_scaled[idx, 0], X_scaled[idx, 1], 
                   c=dark_colors[cid], label=f'Historical {cluster_map[cid][:15]}', 
                   edgecolor='black', s=80, alpha=0.7)

    # Plot the Centroids (Centers)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Mathematical Centers')

    # Plot the User's Input
    user_x, user_y = user_input_scaled[0, 0], user_input_scaled[0, 1]
    assigned_center_x, assigned_center_y = centers[prediction, 0], centers[prediction, 1]
    
    # Draw the distance vector (Line)
    ax.plot([user_x, assigned_center_x], [user_y, assigned_center_y], 
            color='blue', linestyle='--', linewidth=2.5, label='Distance to Assigned Center')
    
    # Draw the User Point (Giant Star)
    ax.scatter(user_x, user_y, c='blue', s=600, marker='*', edgecolor='white', linewidth=1.5, label='Your Proposed Policy')

    ax.set_title('Live K-Means Decision Boundaries', fontweight='bold')
    ax.set_xlabel('Scaled Share of Renewables')
    ax.set_ylabel('Scaled CO2 Emission Factor')
    
    # Format the legend to be outside the plot so it doesn't cover data
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)

# ==========================================
# MODULE 2: POLICY GAP FORECASTER
# ==========================================
elif page == "2. Policy Gap Forecaster (2030)":
    st.header("📉 Module 2: 2030 Policy Gap Forecaster (Prophet)")
    st.write("Set a 2030 renewable energy target for a specific nation. The ML model will forecast their actual trajectory and calculate the resulting policy deficit.")
    
    countries = sorted(df_long[df_long['Entity_Type'] == 'Country']['Country_or_Region'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_country = st.selectbox("Select Sovereign Nation:", countries, index=countries.index("Malaysia") if "Malaysia" in countries else 0)
    with col2:
        target_2030 = st.slider(f"Aspirational 2030 Target for {selected_country} (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

    if st.button("Run Predictive Model"):
        with st.spinner(f'Executing Time-Series Regression for {selected_country}...'):
            country_data = df_long[(df_long['Country_or_Region'] == selected_country) & (df_long['Energy_Type'] == 'Share of renewables in electri')]
            
            if len(country_data) < 10:
                st.error("Insufficient historical data to generate a reliable forecast for this nation.")
            else:
                # Prepare and Train Prophet
                prophet_df = country_data[['Year', 'Energy_Value']].rename(columns={'Year': 'ds', 'Energy_Value': 'y'})
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'].astype(str) + '-01-01')
                
                m = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                m.fit(prophet_df)
                
                # Forecast
                future = m.make_future_dataframe(periods=8, freq='YS')
                forecast = m.predict(future)
                
                pred_2030 = forecast.iloc[-1]['yhat']
                gap = target_2030 - pred_2030
                
                st.markdown("### 📊 Forecasting Results")
                
                # Display Metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Your 2030 Target", f"{target_2030}%")
                metric_col2.metric("ML Predicted 2030", f"{pred_2030:.1f}%")
                
                # Calculate Deficit
                if gap > 0:
                    metric_col3.metric("Policy Deficit (Shortfall)", f"-{gap:.1f}%", delta_color="inverse")
                    st.error(f"**Insight:** {selected_country} will fail to meet your target by {gap:.1f}%. Policymakers must implement aggressive new green policies to bridge this deficit.")
                else:
                    metric_col3.metric("Target Status", "Exceeded!", delta_color="normal")
                    st.success(f"**Insight:** {selected_country} is mathematically on track to surpass your target without additional intervention.")
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 4))
                m.plot(forecast, ax=ax)
                plt.axhline(y=target_2030, color='red', linestyle='--', linewidth=2, label='Your Policy Target')
                plt.title(f"{selected_country}: Renewable Energy Forecast vs Target", fontweight='bold')
                plt.xlabel("Year")
                plt.ylabel("Share of Renewables (%)")
                plt.legend()
                st.pyplot(fig)

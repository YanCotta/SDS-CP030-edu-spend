import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EduSpend: Global Education Cost Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .page-header {
        font-size: 2.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .affordability-low {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .affordability-medium {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .affordability-high {
        background: linear-gradient(90deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .cluster-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2e86ab;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    """Load the education cost dataset."""
    try:
        # Try to load final labeled data first
        data_path = os.path.join(SCRIPT_DIR, 'data', 'final_labeled_data.csv')
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        try:
            # Fallback to original dataset
            data_path = os.path.join(SCRIPT_DIR, 'data', 'International_Education_Costs.csv')
            df = pd.read_csv(data_path)
            
            # Create TCA if not present
            if 'TCA' not in df.columns:
                df['TCA'] = df.get('Tuition_USD', 0) + (df.get('Rent_USD', 0) * 12) + df.get('Visa_Fee_USD', 0) + df.get('Insurance_USD', 0)
            
            # Create affordability tiers if not present
            if 'affordability_tier' not in df.columns and 'Affordability_Tier' not in df.columns:
                q33 = df['TCA'].quantile(0.33)
                q67 = df['TCA'].quantile(0.67)
                df['affordability_tier'] = df['TCA'].apply(
                    lambda x: 'Low' if x <= q33 else ('Medium' if x <= q67 else 'High')
                )
            elif 'Affordability_Tier' in df.columns:
                df['affordability_tier'] = df['Affordability_Tier']
            
            # Create dummy cost clusters if not present
            if 'cost_cluster' not in df.columns:
                np.random.seed(42)
                df['cost_cluster'] = np.random.randint(0, 5, len(df))
            
            return df
        except FileNotFoundError:
            st.error("No data file found. Please ensure the dataset is available.")
            return None

@st.cache_resource
def load_tca_model():
    """Load the trained TCA prediction model."""
    try:
        import joblib
        model_path = os.path.join(SCRIPT_DIR, 'tca_predictor.joblib')
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.warning("⚠️ TCA model file not found. Using placeholder prediction.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading TCA model: {e}")
        return None

@st.cache_resource 
def load_model_metadata():
    """Load model metadata for information display."""
    try:
        import json
        metadata_path = os.path.join(SCRIPT_DIR, 'tca_predictor_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return None

# Prediction functions
def predict_tca_placeholder(country, city, level, program, duration_years, living_cost_index, rent_usd, insurance_usd, visa_fee_usd):
    """
    Placeholder TCA prediction function.
    Used as fallback when trained model is not available.
    """
    # Country-based multipliers
    country_multipliers = {
        'USA': 1.3, 'United States': 1.3, 'UK': 1.25, 'United Kingdom': 1.25,
        'Canada': 1.2, 'Australia': 1.15, 'Germany': 0.8, 'France': 0.85,
        'Netherlands': 0.9, 'Sweden': 0.95, 'India': 0.3, 'China': 0.4,
        'Japan': 1.0, 'South Korea': 0.7
    }
    
    # Program-based multipliers
    program_multipliers = {
        'Medicine': 1.4, 'Business': 1.2, 'Engineering': 1.1, 'Law': 1.15,
        'Computer Science': 1.05, 'Arts': 0.9, 'Science': 0.95, 'Social Sciences': 0.85
    }
    
    # Level-based multipliers
    level_multipliers = {
        'PhD': 1.2, 'Masters': 1.0, 'Bachelors': 0.85, 'Diploma': 0.7
    }
    
    # Base costs
    base_housing = rent_usd * 12 * duration_years
    base_costs = base_housing + insurance_usd + visa_fee_usd
    
    # Apply multipliers
    country_mult = country_multipliers.get(country, 1.0)
    program_mult = program_multipliers.get(program, 1.0)
    level_mult = level_multipliers.get(level, 1.0)
    living_cost_adj = living_cost_index / 100
    
    # Estimated tuition
    estimated_tuition = 25000 * country_mult * program_mult * level_mult * living_cost_adj
    
    # Final TCA
    predicted_tca = base_costs + estimated_tuition
    return max(predicted_tca, 5000)

def predict_tca_with_model(model, country, city, level, program, duration_years, living_cost_index, rent_usd, insurance_usd, visa_fee_usd):
    """
    Use the trained model for TCA prediction if available, otherwise use placeholder logic.
    """
    if model is not None:
        try:
            # Prepare input data for the model
            input_data = pd.DataFrame({
                'Country': [country],
                'City': [city],
                'Program': [program],
                'Level': [level],
                'Duration_Years': [duration_years],
                'Living_Cost_Index': [living_cost_index],
                'Tuition_USD': [rent_usd * 12 * duration_years / 2],  # Estimate tuition
                'Rent_USD': [rent_usd],
                'Visa_Fee_USD': [visa_fee_usd],
                'Insurance_USD': [insurance_usd]
            })
            
            # Make prediction using the trained model
            prediction = model.predict(input_data)[0]
            return prediction
            
        except Exception as e:
            st.warning(f"⚠️ Error using trained model: {e}. Using fallback prediction.")
    
    # Fallback to placeholder logic if model not available
    return predict_tca_placeholder(country, city, level, program, duration_years, living_cost_index, rent_usd, insurance_usd, visa_fee_usd)

def predict_affordability_placeholder(country, city, level, program, duration_years, living_cost_index, rent_usd, insurance_usd, visa_fee_usd):
    """
    Placeholder affordability classification function.
    In production, this would use the loaded classifier model.
    """
    tca = predict_tca_placeholder(country, city, level, program, duration_years, living_cost_index, rent_usd, insurance_usd, visa_fee_usd)
    
    # Simple thresholds for demo (in production, use trained classifier)
    if tca <= 25000:
        return 'Low'
    elif tca <= 50000:
        return 'Medium'
    else:
        return 'High'

# Sidebar navigation
def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.markdown('<div class="sidebar-header">🎓 EduSpend Platform</div>', unsafe_allow_html=True)
    
    pages = {
        "TCA Budget Planner": "💰",
        "Affordability Explorer": "🌍", 
        "Market Cluster Analysis": "📊"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    return selected_page

# Page 1: TCA Budget Planner
def tca_budget_planner_page(df):
    """TCA Budget Planner page."""
    st.markdown('<h2 class="page-header">💰 Global Higher-Education Budget Planner</h2>', unsafe_allow_html=True)
    st.markdown("### Plan your international education budget with AI-powered cost predictions")
    
    # Input form
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌍 Destination Details")
            country = st.selectbox("Country", sorted(df['Country'].unique()) if df is not None else ['USA', 'UK', 'Canada'])
            city = st.selectbox("City", sorted(df['City'].unique()) if df is not None else ['New York', 'London', 'Toronto'])
            
        with col2:
            st.markdown("#### 🎓 Academic Details")
            level = st.selectbox("Degree Level", ['Bachelors', 'Masters', 'PhD', 'Diploma'])
            program = st.selectbox("Program", sorted(df['Program'].unique()) if df is not None else ['Computer Science', 'Business', 'Engineering'])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### ⏱️ Duration & Living")
            duration_years = st.number_input("Duration (Years)", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
            living_cost_index = st.slider("Living Cost Index", min_value=30, max_value=200, value=100)
            
        with col4:
            st.markdown("#### 💰 Cost Parameters")
            rent_usd = st.number_input("Monthly Rent (USD)", min_value=200, max_value=5000, value=1200, step=50)
            insurance_usd = st.number_input("Health Insurance (USD/year)", min_value=0, max_value=5000, value=1500, step=100)
            visa_fee_usd = st.number_input("Visa Fee (USD)", min_value=0, max_value=2000, value=500, step=50)
    
    # Calculate button
    if st.button("🔮 Calculate Total Cost of Attendance", type="primary", use_container_width=True):
        # Load the TCA model
        tca_model = load_tca_model()
        
        # Make prediction
        predicted_tca = predict_tca_with_model(
            tca_model, country, city, level, program, duration_years, 
            living_cost_index, rent_usd, insurance_usd, visa_fee_usd
        )
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Total Cost of Attendance</h2>
            <h1>${predicted_tca:,.0f}</h1>
            <p>For {level} in {program} at {city}, {country}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cost breakdown
        st.markdown("### 📊 Cost Breakdown Analysis")
        
        # Calculate components
        annual_rent = rent_usd * 12
        total_rent = annual_rent * duration_years
        estimated_tuition = predicted_tca - total_rent - insurance_usd - visa_fee_usd
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏠 Total Housing", f"${total_rent:,.0f}", f"{total_rent/predicted_tca*100:.1f}%")
        with col2:
            st.metric("📚 Estimated Tuition", f"${estimated_tuition:,.0f}", f"{estimated_tuition/predicted_tca*100:.1f}%")
        with col3:
            st.metric("🏥 Health Insurance", f"${insurance_usd:,.0f}", f"{insurance_usd/predicted_tca*100:.1f}%")
        with col4:
            st.metric("🛂 Visa Fee", f"${visa_fee_usd:,.0f}", f"{visa_fee_usd/predicted_tca*100:.1f}%")
        
        # Visualization
        st.markdown("### 📈 Cost Distribution")
        
        # Pie chart
        labels = ['Housing', 'Tuition', 'Insurance', 'Visa Fee']
        values = [total_rent, estimated_tuition, insurance_usd, visa_fee_usd]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Cost Distribution Breakdown",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Page 2: Affordability Explorer
def affordability_explorer_page(df):
    """Affordability Explorer page."""
    st.markdown('<h2 class="page-header">🌍 Destination Affordability Explorer</h2>', unsafe_allow_html=True)
    st.markdown("### Discover if your dream destination fits your budget")
    
    # Input form
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🌍 Location")
            country = st.selectbox("Country", sorted(df['Country'].unique()) if df is not None else ['USA', 'UK', 'Canada'], key="aff_country")
            city = st.selectbox("City", sorted(df['City'].unique()) if df is not None else ['New York', 'London', 'Toronto'], key="aff_city")
            
        with col2:
            st.markdown("#### 🎓 Program")
            level = st.selectbox("Degree Level", ['Bachelors', 'Masters', 'PhD', 'Diploma'], key="aff_level")
            program = st.selectbox("Program", sorted(df['Program'].unique()) if df is not None else ['Computer Science', 'Business', 'Engineering'], key="aff_program")
            
        with col3:
            st.markdown("#### 💰 Budget")
            duration_years = st.number_input("Duration (Years)", min_value=0.5, max_value=8.0, value=2.0, step=0.5, key="aff_duration")
            budget_range = st.selectbox("Your Budget Range", ['Under $30,000', '$30,000-$60,000', 'Over $60,000'])
    
    # Additional parameters in expander
    with st.expander("🔧 Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            living_cost_index = st.slider("Living Cost Index", min_value=30, max_value=200, value=100, key="aff_living")
            rent_usd = st.number_input("Monthly Rent (USD)", min_value=200, max_value=5000, value=1200, step=50, key="aff_rent")
        with col2:
            insurance_usd = st.number_input("Health Insurance (USD/year)", min_value=0, max_value=5000, value=1500, step=100, key="aff_insurance")
            visa_fee_usd = st.number_input("Visa Fee (USD)", min_value=0, max_value=2000, value=500, step=50, key="aff_visa")
    
    # Predict affordability
    if st.button("🔍 Check Affordability", type="primary", use_container_width=True):
        # Load the TCA model
        tca_model = load_tca_model()
        
        affordability = predict_affordability_placeholder(
            country, city, level, program, duration_years,
            living_cost_index, rent_usd, insurance_usd, visa_fee_usd
        )
        
        predicted_tca = predict_tca_with_model(
            tca_model, country, city, level, program, duration_years,
            living_cost_index, rent_usd, insurance_usd, visa_fee_usd
        )
        
        # Display result based on affordability
        if affordability == 'Low':
            st.markdown(f"""
            <div class="affordability-low">
                <h2>🟢 LOW COST DESTINATION</h2>
                <h3>Estimated TCA: ${predicted_tca:,.0f}</h3>
                <p>✅ Great choice! This destination is budget-friendly for {level} in {program}</p>
                <p>💡 Consider this affordable option for quality education</p>
            </div>
            """, unsafe_allow_html=True)
        elif affordability == 'Medium':
            st.markdown(f"""
            <div class="affordability-medium">
                <h2>🟡 MEDIUM COST DESTINATION</h2>
                <h3>Estimated TCA: ${predicted_tca:,.0f}</h3>
                <p>⚖️ Moderate costs - plan your budget carefully for {level} in {program}</p>
                <p>💡 Consider additional funding sources or scholarships</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="affordability-high">
                <h2>🔴 HIGH COST DESTINATION</h2>
                <h3>Estimated TCA: ${predicted_tca:,.0f}</h3>
                <p>💰 Premium destination with higher costs for {level} in {program}</p>
                <p>💡 Ensure adequate funding and explore scholarship opportunities</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### 📊 Affordability Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Affordability Tier", affordability)
        with col2:
            monthly_budget = predicted_tca / (duration_years * 12)
            st.metric("📅 Monthly Budget", f"${monthly_budget:,.0f}")
        with col3:
            annual_budget = predicted_tca / duration_years
            st.metric("📆 Annual Budget", f"${annual_budget:,.0f}")
        
        # Budget comparison
        if df is not None:
            st.markdown("### 🔍 Compare with Similar Destinations")
            
            # Filter similar programs
            similar_programs = df[
                (df['Program'] == program) & 
                (df['Level'] == level)
            ].copy()
            
            if len(similar_programs) > 0:
                # Calculate affordability distribution
                affordability_dist = similar_programs['affordability_tier'].value_counts() if 'affordability_tier' in similar_programs.columns else None
                
                if affordability_dist is not None:
                    fig = px.pie(
                        values=affordability_dist.values,
                        names=affordability_dist.index,
                        title=f"Affordability Distribution for {program} ({level})",
                        color_discrete_map={'Low': '#56ab2f', 'Medium': '#f7971e', 'High': '#fc4a1a'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Page 3: Market Cluster Analysis
def market_cluster_analysis_page(df):
    """Market Cluster Analysis page."""
    st.markdown('<h2 class="page-header">📊 University Cost Structure Analysis</h2>', unsafe_allow_html=True)
    st.markdown("### Explore global education market clusters and cost patterns")
    
    if df is None:
        st.error("No data available for cluster analysis.")
        return
    
    # Country filter
    st.markdown("#### 🌍 Filter by Country")
    selected_country = st.selectbox(
        "Select a country to analyze:",
        ['All Countries'] + sorted(df['Country'].unique()),
        key="cluster_country"
    )
    
    # Filter data
    if selected_country == 'All Countries':
        filtered_df = df.copy()
        display_title = "Global Market Analysis"
    else:
        filtered_df = df[df['Country'] == selected_country].copy()
        display_title = f"{selected_country} Market Analysis"
    
    if len(filtered_df) == 0:
        st.warning(f"No data available for {selected_country}")
        return
    
    # Display filtered data
    st.markdown(f"### 📋 {display_title}")
    st.markdown(f"**Found {len(filtered_df)} universities/programs**")
    
    # Cluster information
    if 'cost_cluster' in filtered_df.columns:
        st.markdown("#### 🏷️ Cost Cluster Distribution")
        
        cluster_counts = filtered_df['cost_cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution chart
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Universities by Cost Cluster",
                labels={'x': 'Cost Cluster', 'y': 'Number of Universities'},
                color=cluster_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average TCA by cluster
            if 'TCA' in filtered_df.columns:
                avg_tca_by_cluster = filtered_df.groupby('cost_cluster')['TCA'].mean().sort_index()
                
                fig = px.bar(
                    x=avg_tca_by_cluster.index,
                    y=avg_tca_by_cluster.values,
                    title="Average TCA by Cluster",
                    labels={'x': 'Cost Cluster', 'y': 'Average TCA (USD)'},
                    color=avg_tca_by_cluster.values,
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster details
        st.markdown("#### 🔍 Cluster Analysis Details")
        
        for cluster_id in sorted(filtered_df['cost_cluster'].unique()):
            cluster_data = filtered_df[filtered_df['cost_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            with st.expander(f"🏷️ Cluster {cluster_id} ({cluster_size} universities)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'TCA' in cluster_data.columns:
                        avg_tca = cluster_data['TCA'].mean()
                        st.metric("Average TCA", f"${avg_tca:,.0f}")
                    
                    if 'affordability_tier' in cluster_data.columns:
                        most_common_tier = cluster_data['affordability_tier'].mode().iloc[0] if len(cluster_data['affordability_tier'].mode()) > 0 else 'N/A'
                        st.metric("Most Common Tier", most_common_tier)
                
                with col2:
                    # Most common programs
                    if 'Program' in cluster_data.columns:
                        top_programs = cluster_data['Program'].value_counts().head(3)
                        st.markdown("**Top Programs:**")
                        for program, count in top_programs.items():
                            st.write(f"• {program} ({count})")
                
                with col3:
                    # Most common levels
                    if 'Level' in cluster_data.columns:
                        top_levels = cluster_data['Level'].value_counts().head(3)
                        st.markdown("**Education Levels:**")
                        for level, count in top_levels.items():
                            st.write(f"• {level} ({count})")
    
    # Display data table
    st.markdown("#### 📊 Detailed University Data")
    
    # Select columns to display
    display_columns = ['Country', 'City', 'Program', 'Level']
    if 'TCA' in filtered_df.columns:
        display_columns.append('TCA')
    if 'affordability_tier' in filtered_df.columns:
        display_columns.append('affordability_tier')
    if 'cost_cluster' in filtered_df.columns:
        display_columns.append('cost_cluster')
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    # Display with formatting
    display_df = filtered_df[available_columns].copy()
    
    if 'TCA' in display_df.columns:
        display_df['TCA'] = display_df['TCA'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name=f"eduSpend_{selected_country.replace(' ', '_')}_data.csv",
        mime="text/csv"
    )

# Main application
def main():
    """Main application function."""
    # Load data and model
    df = load_data()
    tca_model = load_tca_model()
    metadata = load_model_metadata()
    
    # Header
    st.markdown('<h1 class="main-header">🎓 EduSpend: Global Education Cost Platform</h1>', unsafe_allow_html=True)
    
    # Navigation
    selected_page = sidebar_navigation()
    
    # Page routing
    if selected_page == "TCA Budget Planner":
        tca_budget_planner_page(df)
    elif selected_page == "Affordability Explorer":
        affordability_explorer_page(df)
    elif selected_page == "Market Cluster Analysis":
        market_cluster_analysis_page(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>🎓 EduSpend Global Education Cost Platform | Multi-Page Streamlit Application</p>
        <p>🤖 AI-Powered TCA Predictions | 🌍 Affordability Classification | 📊 Market Cluster Analysis</p>
        <p>Developed by yan-cotta | Built with Streamlit, Scikit-learn & Plotly</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Status")
    
    if tca_model is not None:
        st.sidebar.success("✅ TCA Model: Loaded successfully")
        if metadata:
            st.sidebar.info(f"📊 Model R²: {metadata.get('performance_metrics', {}).get('cross_val_r2_mean', 'N/A'):.4f}")
    else:
        st.sidebar.warning("⚠️ TCA Model: Using placeholder")
    
    st.sidebar.markdown("### 📊 Dataset Info")
    if df is not None:
        st.sidebar.success(f"✅ Loaded {len(df):,} universities\n📍 {df['Country'].nunique()} countries\n🎓 {df['Program'].nunique()} programs")
    else:
        st.sidebar.error("❌ No dataset loaded")

if __name__ == "__main__":
    main()

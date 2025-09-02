import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(
    page_title="Fish Production Analysis",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FishProductionAnalyzer:
    def __init__(self):
        self.feeding_data = None
        self.transfer_data = None
        self.harvest_data = None
        self.sampling_data = None
        self.processed_data = {}
        
    def normalize_columns(self, df, expected_cols):
        """Normalize column names to expected format"""
        df.columns = df.columns.str.lower().str.strip()
        
        # Create mapping for common variations
        col_mapping = {
            'cage number': 'cage',
            'cage_number': 'cage',
            'feed amount (kg)': 'feed_amount_kg',
            'feed amount': 'feed_amount_kg',
            'origin cage': 'origin_cage',
            'destination cage': 'destination_cage',
            'number of fish': 'number_of_fish',
            'total weight': 'total_weight',
            'average body weight (g)': 'abw_g',
            'abw (g)': 'abw_g',
            'abw [g]': 'abw_g',
            'total weight [kg]': 'total_weight',
        }
        
        df = df.rename(columns=col_mapping)
        return df
    
    def validate_transfer_units(self, df):
        """Convert transfer weights from grams to kg if needed"""
        if 'total_weight' in df.columns and 'number_of_fish' in df.columns:
            # Calculate average weight per fish
            df['avg_weight_per_fish'] = df['total_weight'] / df['number_of_fish']
            
            # If average weight > 10 kg, likely entered in grams
            mask = df['avg_weight_per_fish'] > 10
            df.loc[mask, 'total_weight'] = df.loc[mask, 'total_weight'] / 1000
            df.loc[mask, 'avg_weight_per_fish'] = df.loc[mask, 'avg_weight_per_fish'] / 1000
            
        return df
    
    def load_and_process_data(self, feeding_file, transfer_file, harvest_file, sampling_file=None):
        """Load and process all data files"""
        try:
            # Load feeding data
            if feeding_file is not None:
                self.feeding_data = pd.read_excel(feeding_file)
                self.feeding_data = self.normalize_columns(self.feeding_data, ['date', 'cage', 'feed_amount_kg'])
                # Handle date parsing with error handling for malformed dates
                self.feeding_data['date'] = pd.to_datetime(self.feeding_data['date'], errors='coerce', dayfirst=True)
                # Remove rows with invalid dates
                self.feeding_data = self.feeding_data.dropna(subset=['date'])
                
            # Load transfer data
            if transfer_file is not None:
                self.transfer_data = pd.read_excel(transfer_file)
                self.transfer_data = self.normalize_columns(self.transfer_data, ['date', 'origin_cage', 'destination_cage', 'number_of_fish', 'total_weight'])
                self.transfer_data['date'] = pd.to_datetime(self.transfer_data['date'], errors='coerce', dayfirst=True)
                self.transfer_data = self.transfer_data.dropna(subset=['date'])
                self.transfer_data = self.validate_transfer_units(self.transfer_data)
                
            # Load harvest data
            if harvest_file is not None:
                self.harvest_data = pd.read_excel(harvest_file)
                self.harvest_data = self.normalize_columns(self.harvest_data, ['date', 'cage', 'number_of_fish', 'total_weight', 'abw_g'])
                self.harvest_data['date'] = pd.to_datetime(self.harvest_data['date'], errors='coerce', dayfirst=True)
                self.harvest_data = self.harvest_data.dropna(subset=['date'])
                
            # Load sampling data
            if sampling_file is not None:
                self.sampling_data = pd.read_excel(sampling_file)
                self.sampling_data = self.normalize_columns(self.sampling_data, ['date', 'cage', 'number_of_fish', 'abw_g'])
                self.sampling_data['date'] = pd.to_datetime(self.sampling_data['date'], errors='coerce', dayfirst=True)
                self.sampling_data = self.sampling_data.dropna(subset=['date'])
                
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Please check your date formats. Expected formats: DD/MM/YYYY, MM/DD/YYYY, or YYYY-MM-DD")
            return False
    
    def add_corrected_stocking_event(self, cage_num=2):
        """Add corrected stocking event for Cage 2"""
        stocking_event = {
            'date': pd.Timestamp('2024-08-26'),
            'cage': cage_num,
            'number_of_fish': 7290,
            'abw_g': 11.9,
            'event_type': 'stocking'
        }
        
        if self.sampling_data is None:
            self.sampling_data = pd.DataFrame([stocking_event])
        else:
            # Remove any existing stocking events for this cage
            self.sampling_data = self.sampling_data[~((self.sampling_data['cage'] == cage_num) & 
                                                     (self.sampling_data['date'] < pd.Timestamp('2024-08-27')))]
            # Add corrected stocking event
            stocking_df = pd.DataFrame([stocking_event])
            self.sampling_data = pd.concat([self.sampling_data, stocking_df], ignore_index=True)
            
        self.sampling_data = self.sampling_data.sort_values(['cage', 'date']).reset_index(drop=True)
    
    def generate_synthetic_sampling_data(self, cage_num=2):
        """Generate synthetic sampling data if none provided"""
        start_date = pd.Timestamp('2024-08-26')  # Corrected stocking date
        end_date = pd.Timestamp('2025-07-09')    # Analysis end date
        
        # Create sampling points every 3-4 weeks throughout the period
        sampling_dates = []
        current_date = start_date
        
        # Ensure we cover the full period from Aug 26, 2024 to July 9, 2025
        while current_date <= end_date:
            sampling_dates.append(current_date)
            # Variable interval between 21-28 days (3-4 weeks)
            current_date += timedelta(days=np.random.randint(21, 29))
        
        # Make sure we include the end date if close
        if sampling_dates[-1] < end_date - timedelta(days=14):
            sampling_dates.append(end_date)
        
        # Generate synthetic ABW data with realistic growth curve
        synthetic_data = []
        initial_abw = 11.9
        total_days = (end_date - start_date).days  # ~317 days
        
        for i, date in enumerate(sampling_dates):
            days_since_stock = (date - start_date).days
            
            # Realistic fish growth model over ~10.5 months
            # Target: 11.9g → ~400-500g over the period
            growth_factor = days_since_stock / total_days
            # Sigmoid growth curve with slower growth at end
            abw = initial_abw + (450 - initial_abw) * (growth_factor / (growth_factor + 0.3))
            abw += np.random.normal(0, abw * 0.04)  # 4% variation
            
            # Gradual mortality over the period (realistic for aquaculture)
            mortality_rate = 0.015  # 1.5% monthly mortality
            fish_count = int(7290 * ((1 - mortality_rate) ** (days_since_stock / 30.44)))
            
            synthetic_data.append({
                'date': date,
                'cage': cage_num,
                'number_of_fish': max(1000, fish_count),  # Minimum viable population
                'abw_g': max(initial_abw, abw),
                'event_type': 'sampling' if i > 0 else 'stocking'
            })
        
        return pd.DataFrame(synthetic_data)
    
    def calculate_production_metrics(self, cage_num):
        """Calculate production metrics for a specific cage"""
        # Filter data for specific cage
        cage_feeding = self.feeding_data[self.feeding_data['cage'] == cage_num].copy() if self.feeding_data is not None else pd.DataFrame()
        cage_sampling = self.sampling_data[self.sampling_data['cage'] == cage_num].copy() if self.sampling_data is not None else pd.DataFrame()
        cage_transfers_in = self.transfer_data[self.transfer_data['destination_cage'] == cage_num].copy() if self.transfer_data is not None else pd.DataFrame()
        cage_transfers_out = self.transfer_data[self.transfer_data['origin_cage'] == cage_num].copy() if self.transfer_data is not None else pd.DataFrame()
        
        if cage_sampling.empty:
            return pd.DataFrame()
        
        cage_sampling = cage_sampling.sort_values('date').reset_index(drop=True)
        
        # Calculate biomass for each sampling point
        cage_sampling['biomass_kg'] = cage_sampling['number_of_fish'] * (cage_sampling['abw_g'] / 1000)
        
        # Initialize metrics columns
        cage_sampling['feed_period_kg'] = 0.0
        cage_sampling['growth_period_kg'] = 0.0
        cage_sampling['period_efcr'] = np.nan
        cage_sampling['cumulative_feed_kg'] = 0.0
        cage_sampling['cumulative_growth_kg'] = 0.0
        cage_sampling['aggregated_efcr'] = np.nan
        
        for i in range(len(cage_sampling)):
            current_date = cage_sampling.iloc[i]['date']
            
            if i == 0:
                # First sampling point (stocking)
                cage_sampling.loc[i, 'growth_period_kg'] = cage_sampling.iloc[i]['biomass_kg']
                cage_sampling.loc[i, 'cumulative_growth_kg'] = cage_sampling.iloc[i]['biomass_kg']
            else:
                prev_date = cage_sampling.iloc[i-1]['date']
                
                # Calculate feed consumed in this period
                period_feed = cage_feeding[
                    (cage_feeding['date'] > prev_date) & 
                    (cage_feeding['date'] <= current_date)
                ]['feed_amount_kg'].sum()
                
                cage_sampling.loc[i, 'feed_period_kg'] = period_feed
                cage_sampling.loc[i, 'cumulative_feed_kg'] = cage_sampling.loc[i-1, 'cumulative_feed_kg'] + period_feed
                
                # Calculate growth in this period
                current_biomass = cage_sampling.iloc[i]['biomass_kg']
                prev_biomass = cage_sampling.iloc[i-1]['biomass_kg']
                
                # Account for transfers
                transfers_in = cage_transfers_in[
                    (cage_transfers_in['date'] > prev_date) & 
                    (cage_transfers_in['date'] <= current_date)
                ]['total_weight'].sum()
                
                transfers_out = cage_transfers_out[
                    (cage_transfers_out['date'] > prev_date) & 
                    (cage_transfers_out['date'] <= current_date)
                ]['total_weight'].sum()
                
                net_transfer = transfers_in - transfers_out
                growth_period = current_biomass - prev_biomass - net_transfer
                
                cage_sampling.loc[i, 'growth_period_kg'] = growth_period
                cage_sampling.loc[i, 'cumulative_growth_kg'] = cage_sampling.loc[i-1, 'cumulative_growth_kg'] + growth_period
                
                # Calculate eFCR
                if growth_period > 0:
                    cage_sampling.loc[i, 'period_efcr'] = period_feed / growth_period
                
                if cage_sampling.loc[i, 'cumulative_growth_kg'] > 0:
                    cage_sampling.loc[i, 'aggregated_efcr'] = cage_sampling.loc[i, 'cumulative_feed_kg'] / cage_sampling.loc[i, 'cumulative_growth_kg']
        
        return cage_sampling
    
    def generate_mock_cages(self, base_cage_num=2, num_mock_cages=5):
        """Generate synthetic cage data based on Cage 2 patterns"""
        np.random.seed(42)  # For reproducibility
        
        base_data = self.calculate_production_metrics(base_cage_num)
        if base_data.empty:
            return {}
        
        mock_cages = {}
        
        for mock_cage_id in range(100, 100 + num_mock_cages):  # Cage 100, 101, 102, etc.
            mock_data = base_data.copy()
            mock_data['cage'] = mock_cage_id
            
            # Randomize stocking
            stock_multiplier = 1 + np.random.normal(0, 0.05)
            mock_data['number_of_fish'] = (mock_data['number_of_fish'] * stock_multiplier).astype(int)
            
            # Randomize ABW with growth variation
            abw_multiplier = 1 + np.random.normal(0, 0.06, len(mock_data))
            mock_data['abw_g'] = mock_data['abw_g'] * abw_multiplier
            
            # Recalculate biomass
            mock_data['biomass_kg'] = mock_data['number_of_fish'] * (mock_data['abw_g'] / 1000)
            
            # Adjust feed amounts
            feed_multiplier = 1 + np.random.normal(0, 0.08, len(mock_data))
            mock_data['feed_period_kg'] = np.maximum(0, mock_data['feed_period_kg'] * feed_multiplier)
            mock_data['cumulative_feed_kg'] = mock_data['feed_period_kg'].cumsum()
            
            # Recalculate growth and eFCR
            for i in range(1, len(mock_data)):
                current_biomass = mock_data.iloc[i]['biomass_kg']
                prev_biomass = mock_data.iloc[i-1]['biomass_kg']
                growth_period = current_biomass - prev_biomass
                
                mock_data.loc[mock_data.index[i], 'growth_period_kg'] = growth_period
                
                if i == 1:
                    mock_data.loc[mock_data.index[i], 'cumulative_growth_kg'] = current_biomass
                else:
                    mock_data.loc[mock_data.index[i], 'cumulative_growth_kg'] = mock_data.iloc[i-1]['cumulative_growth_kg'] + growth_period
                
                # Recalculate eFCRs
                if growth_period > 0:
                    mock_data.loc[mock_data.index[i], 'period_efcr'] = mock_data.iloc[i]['feed_period_kg'] / growth_period
                
                if mock_data.iloc[i]['cumulative_growth_kg'] > 0:
                    mock_data.loc[mock_data.index[i], 'aggregated_efcr'] = mock_data.iloc[i]['cumulative_feed_kg'] / mock_data.iloc[i]['cumulative_growth_kg']
            
            # Add date jitter (±2 days)
            date_jitter = np.random.randint(-2, 3, len(mock_data))
            mock_data['date'] = mock_data['date'] + pd.to_timedelta(date_jitter, unit='D')
            
            mock_cages[mock_cage_id] = mock_data
        
        return mock_cages

def main():
    st.title("🐟 Fish Production Analysis Dashboard")
    st.markdown("---")
    
    analyzer = FishProductionAnalyzer()
    
    # Sidebar for file uploads and controls
    st.sidebar.header("📁 Data Upload")
    
    feeding_file = st.sidebar.file_uploader("Upload Feeding Record", type=['xlsx', 'xls'], key="feeding")
    harvest_file = st.sidebar.file_uploader("Upload Fish Harvest", type=['xlsx', 'xls'], key="harvest")
    sampling_file = st.sidebar.file_uploader("Upload Fish Sampling (Optional)", type=['xlsx', 'xls'], key="sampling")
    transfer_file = st.sidebar.file_uploader("Upload Fish Transfer (Optional)", type=['xlsx', 'xls'], key="transfer")
    
    use_synthetic_sampling = st.sidebar.checkbox("Use Synthetic Sampling Data", value=True, 
                                                help="Generate synthetic sampling data if no sampling file is uploaded")
    
    if st.sidebar.button("🔄 Process Data"):
        with st.spinner("Processing data..."):
            success = analyzer.load_and_process_data(feeding_file, transfer_file, harvest_file, sampling_file)
            
            if success:
                # Add corrected stocking event
                analyzer.add_corrected_stocking_event(cage_num=2)
                
                # Generate synthetic sampling data if needed
                if analyzer.sampling_data is None or use_synthetic_sampling:
                    st.info("📅 Generating synthetic sampling data for analysis period: 26 Aug 2024 - 09 Jul 2025")
                    synthetic_data = analyzer.generate_synthetic_sampling_data(cage_num=2)
                    if analyzer.sampling_data is None:
                        analyzer.sampling_data = synthetic_data
                    else:
                        # Replace any existing Cage 2 data with synthetic data
                        analyzer.sampling_data = analyzer.sampling_data[analyzer.sampling_data['cage'] != 2]
                        analyzer.sampling_data = pd.concat([analyzer.sampling_data, synthetic_data], ignore_index=True)
                        analyzer.sampling_data = analyzer.sampling_data.sort_values(['cage', 'date']).reset_index(drop=True)
                
                # Process main cage data
                cage_2_data = analyzer.calculate_production_metrics(cage_num=2)
                analyzer.processed_data[2] = cage_2_data
                
                # Generate mock cages
                mock_cages = analyzer.generate_mock_cages(base_cage_num=2, num_mock_cages=5)
                analyzer.processed_data.update(mock_cages)
                
                st.sidebar.success("✅ Data processed successfully!")
                st.rerun()
    
    # Main dashboard
    if analyzer.processed_data:
        st.sidebar.header("📊 Dashboard Controls")
        
        # Cage selection
        available_cages = list(analyzer.processed_data.keys())
        selected_cage = st.sidebar.selectbox("Select Cage", available_cages, format_func=lambda x: f"Cage {x}")
        
        # KPI selection
        kpi_options = ["Growth", "eFCR", "Biomass", "Feed Usage"]
        selected_kpi = st.sidebar.selectbox("Select KPI", kpi_options)
        
        if selected_cage in analyzer.processed_data:
            cage_data = analyzer.processed_data[selected_cage]
            
            # Display metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Analysis Period", 
                         "26 Aug 2024 - 09 Jul 2025")
            
            with col2:
                st.metric("Current Fish Count", f"{cage_data['number_of_fish'].iloc[-1]:,.0f}")
            
            with col3:
                st.metric("Current ABW (g)", f"{cage_data['abw_g'].iloc[-1]:.1f}")
            
            with col4:
                st.metric("Latest eFCR", f"{cage_data['aggregated_efcr'].iloc[-1]:.2f}" if not pd.isna(cage_data['aggregated_efcr'].iloc[-1]) else "N/A")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Growth Plot")
                
                if selected_kpi == "Growth":
                    fig = px.line(cage_data, x='date', y='biomass_kg', 
                                 title=f"Cage {selected_cage} - Biomass Over Time",
                                 labels={'biomass_kg': 'Biomass (kg)', 'date': 'Date'})
                elif selected_kpi == "Biomass":
                    fig = px.bar(cage_data, x='date', y='growth_period_kg', 
                                title=f"Cage {selected_cage} - Period Growth",
                                labels={'growth_period_kg': 'Growth (kg)', 'date': 'Date'})
                elif selected_kpi == "Feed Usage":
                    fig = px.bar(cage_data, x='date', y='feed_period_kg', 
                                title=f"Cage {selected_cage} - Feed Usage",
                                labels={'feed_period_kg': 'Feed (kg)', 'date': 'Date'})
                else:  # eFCR
                    fig = px.scatter(cage_data, x='date', y='abw_g', size='biomass_kg',
                                   title=f"Cage {selected_cage} - ABW vs Time",
                                   labels={'abw_g': 'ABW (g)', 'date': 'Date'})
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 eFCR Analysis")
                
                # Create dual-axis plot for eFCR
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add aggregated eFCR
                fig.add_trace(
                    go.Scatter(x=cage_data['date'], y=cage_data['aggregated_efcr'], 
                              name="Aggregated eFCR", line=dict(color="blue")),
                    secondary_y=False,
                )
                
                # Add period eFCR
                fig.add_trace(
                    go.Scatter(x=cage_data['date'], y=cage_data['period_efcr'], 
                              name="Period eFCR", line=dict(color="red", dash="dash")),
                    secondary_y=True,
                )
                
                fig.update_layout(title=f"Cage {selected_cage} - eFCR Combined Plot")
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Aggregated eFCR", secondary_y=False)
                fig.update_yaxes(title_text="Period eFCR", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Production Summary Table
            st.subheader(f"📋 Production Summary - Cage {selected_cage}")
            
            # Format the data for display
            display_data = cage_data.copy()
            display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')
            display_data = display_data.round(2)
            
            # Select relevant columns for display
            display_columns = ['date', 'number_of_fish', 'abw_g', 'biomass_kg', 'feed_period_kg', 
                             'growth_period_kg', 'period_efcr', 'cumulative_feed_kg', 
                             'cumulative_growth_kg', 'aggregated_efcr']
            
            st.dataframe(display_data[display_columns], use_container_width=True)
            
            # Export functionality
            st.subheader("💾 Export Data")
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            display_data[display_columns].to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label=f"📥 Download Cage {selected_cage} Summary (CSV)",
                data=csv_data,
                file_name=f"cage_{selected_cage}_production_summary.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Please upload data files and click 'Process Data' to begin analysis.")
        
        # Show sample data structure
        st.subheader("📖 Expected Data Structure")
        
        with st.expander("Sample Data Formats"):
            st.markdown("**Feeding Record:**")
            sample_feeding = pd.DataFrame({
                'DATE': ['2024-08-27', '2024-08-28'],
                'CAGE NUMBER': [2, 2],
                'FEED AMOUNT (Kg)': [25.5, 26.2]
            })
            st.dataframe(sample_feeding)
            
            st.markdown("**Fish Transfer:**")
            sample_transfer = pd.DataFrame({
                'DATE': ['2024-09-01'],
                'ORIGIN CAGE': [2],
                'DESTINATION CAGE': [3],
                'NUMBER OF FISH': [100],
                'Total weight': [2.5]
            })
            st.dataframe(sample_transfer)
            
            st.markdown("**Fish Harvest:**")
            sample_harvest = pd.DataFrame({
                'DATE': ['2025-07-09'],
                'CAGE': [2],
                'NUMBER OF FISH': [6500],
                'TOTAL WEIGHT [kg]': [3250],
                'ABW [g]': [500]
            })
            st.dataframe(sample_harvest)

if __name__ == "__main__":
    main()

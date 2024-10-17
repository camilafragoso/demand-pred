import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import pydeck as pdk

# Load the dataset
data = pd.read_csv('avg_demand_by_region.csv')


# Function to calculate median lat and lng for each region
def calculate_region_medians(df):
    region_medians = df.groupby('region')[['start_lat', 'start_lng']].median().reset_index()
    # Round the latitude and longitude to 2 decimal places
    region_medians['start_lat'] = region_medians['start_lat'].round(2)
    region_medians['start_lng'] = region_medians['start_lng'].round(2)
    region_medians.columns = ['region', 'median_lat', 'median_lng']
    return region_medians

# Function to calculate the distance between driver and each region
def calculate_distances(driver_lat, driver_lng, region_medians):
    region_medians['distance_km'] = region_medians.apply(
        lambda row: geodesic((driver_lat, driver_lng), (row['median_lat'], row['median_lng'])).km, axis=1)
    return region_medians

# Function to merge average demand with region medians
def merge_average_demand(df, region_medians):
    avg_demand = df.groupby('region')['actual_demand'].mean().reset_index()
    avg_demand.columns = ['region', 'avg_demand']
    return pd.merge(region_medians, avg_demand, on='region')

# Function to rank regions by demand
def rank_demand(region_data):
    region_data['demand_rank'] = region_data['avg_demand'].rank(ascending=False)
    return region_data

# Streamlit App
st.title('Rides Demand Map')

st.markdown("This app is a simplified batch implementation designed to help Bolt and ride-hailing platforms to optimize driver routes by showing their proximity to various regions. The tool uses historical demand data and a machine learning model to predict demand for each region considering the day of the week and time. The regions are represented on the map by the red dots by the median of their geographical points and are sized by the demand on that selected day and time. The driver location is represented by the blue dot.")

# Input for the user
st.sidebar.header('Input Driver Location and Time')
day_of_week = st.sidebar.selectbox('Select Day of the Week', options=data['day_of_week'].unique())
hour = st.sidebar.slider('Select Hour', min_value=0, max_value=23)
driver_lat = st.sidebar.number_input('Driver Latitude', value = 59.43, format="%.2f")
driver_lng = st.sidebar.number_input('Driver Longitude',value = 24.76, format="%.2f")

# Filter data based on user inputs
filtered_data = data[(data['day_of_week'] == day_of_week) & (data['hour'] == hour)]

# Calculate region medians
region_medians = calculate_region_medians(filtered_data)

# Calculate distances from driver to each region's median point
region_medians_with_distances = calculate_distances(driver_lat, driver_lng, region_medians)

# Merge average demand to region medians
final_data = merge_average_demand(filtered_data, region_medians_with_distances)

# Rank the regions by average demand
final_data = rank_demand(final_data)

# Display map with driver position and region median points
st.header('Map Visualization')

# Scale the point size based on demand rank
final_data['point_size'] = final_data['demand_rank'] * 100  # Scale factor for better visualization

# Create the layer for region median points
region_layer = pdk.Layer(
    'ScatterplotLayer',
    data=final_data,
    get_position=['median_lng', 'median_lat'],
    get_radius='point_size',  # Use the demand rank to adjust the point size
    get_fill_color=[255, 0, 0],
    pickable=True,
    extruded=True,
    get_line_width=0,
    get_elevation='avg_demand',
)

# Add driver position as a separate point
driver_layer = pdk.Layer(
    'ScatterplotLayer',
    data=pd.DataFrame({'lat': [driver_lat], 'lng': [driver_lng]}),
    get_position=['lng', 'lat'],
    get_radius=200,
    get_fill_color=[0, 0, 255],
    pickable=True,
)

# Set the view state for the map
view_state = pdk.ViewState(
    latitude=driver_lat,
    longitude=driver_lng,
    zoom=12,
    pitch=45,
)

# Render the map
st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=[region_layer, driver_layer],
))

# Display distance and demand rank information
st.write("Distances and Demand Rank for Each Region:")
st.dataframe(final_data[['region', 'distance_km', 'avg_demand', 'demand_rank']])

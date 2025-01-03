import random
import time
import pandas as pd
import hopsworks
import uuid

# Set to true initially
BACKFILL = False

def generate_aqi_data_dynamic():
    """
    Returns a single random AQI data row as a DataFrame, dynamically generated from the existing dataset.
    """
    # Load your dataset, so it can refresh and generate new values
    aqi_df = pd.read_csv("../aqi.csv")
    # Drop rows with any empty (NaN or None) values
    aqi_df = aqi_df.dropna()
    # Randomly select a row from the dataset for Country, City, lat, and lng
    location_data = aqi_df.sample(1).iloc[0]

    # Define ranges for AQI components
    aqi_ranges = {
        "aqi_value": (0, 500),
        "co_aqi_value": (0, 50),
        "ozone_aqi_value": (0, 200),
        "no2_aqi_value": (0, 100),
        "pm25_aqi_value": (0, 500),
    }

    # Generate random integer values for AQI components
    aqi_data = {key: random.randint(int(value[0]), int(value[1])) for key, value in aqi_ranges.items()}

    # Add location data
    aqi_data["country"] = location_data["Country"]
    aqi_data["city"] = location_data["City"]
    aqi_data["lat"] = location_data["lat"]
    aqi_data["lng"] = location_data["lng"]

    # Generate AQI Category
    aqi_data["aqi_category"] = (
        "Good" if aqi_data["aqi_value"] <= 50 else
        "Moderate" if aqi_data["aqi_value"] <= 100 else
        "Unhealthy for Sensitive Groups" if aqi_data["aqi_value"] <= 150 else
        "Unhealthy" if aqi_data["aqi_value"] <= 200 else
        "Very Unhealthy" if aqi_data["aqi_value"] <= 300 else
        "Hazardous" if aqi_data["aqi_value"] <= 500 else
        "Beyond Hazardous"
    )


    # Create DataFrame with the correct order and column names
    result_df = pd.DataFrame([aqi_data], columns=[
        "country", "city", "aqi_value", "aqi_category", "co_aqi_value", 
        "ozone_aqi_value", "no2_aqi_value", "pm25_aqi_value", "lat", "lng"
    ])
    
    # Ensure correct data types
    result_df = result_df.astype({
        'country': 'string', 
        'city': 'string', 
        'aqi_value': 'int64', 
        'aqi_category': 'string', 
        'co_aqi_value': 'int64', 
        'ozone_aqi_value': 'int64', 
        'no2_aqi_value': 'int64', 
        'pm25_aqi_value': 'int64', 
        'lat': 'float64', 
        'lng': 'float64', 
    })

    return result_df

def get_random_aqi_values():
    """
    Returns a DataFrame containing random AQI values using the existing dataset for location data.
    """
    return generate_aqi_data_dynamic()


if BACKFILL == True:
    aqi_df = pd.read_csv("../aqi.csv")
    # Example list of columns to drop
    columns_to_drop = ['CO AQI Category', 'Ozone AQI Category', 'NO2 AQI Category', 'PM2.5 AQI Category']

    # Drop all category columns except 'aqicategory'
    aqi_df = aqi_df.drop(columns=[col for col in columns_to_drop if col in aqi_df.columns])
else:
    aqi_df = get_random_aqi_values()
   
# Add UUID to each row
aqi_df["uuid"] = [str(uuid.uuid4()) for _ in range(len(aqi_df))]

# Drop rows with any empty (NaN or None) values
aqi_df = aqi_df.dropna()
# Refactoring column names
aqi_df.columns = [col.lower().replace(' ', '_').replace('.', '') for col in aqi_df.columns]
aqi_df



# Authenticate with Hopsworks using your API Key
project = hopsworks.login()
fs = project.get_feature_store()

aqi_fg = fs.get_or_create_feature_group(name="aqi",
                                  version=1,
                                  primary_key=["uuid"],
                                  description="Air Quality Prediction dataset project"
                                 )
aqi_fg.insert(aqi_df)
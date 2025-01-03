import pandas as pd
import hopsworks
import joblib
import datetime
from PIL import Image

# Login to Hopsworks
project = hopsworks.login()

# Access Feature Store and Model Registry
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load the aqi model
model = mr.get_model("aqi", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/aqi.pkl")

# Load feature view
feature_view = fs.get_feature_view(name="aqi", version=1)

# Perform batch inference
batch_data = feature_view.get_batch_data()
y_pred = model.predict(batch_data)


# Save prediction output as an image
category = y_pred[y_pred.size-1]
# flower_img = "assets/" + flower + ".png"
# img = Image.open(flower_img)
# img.save("../../assets/latest_iris.png")

# Retrieve actual label
aqi_fg = fs.get_feature_group(name="aqi", version=1)
df = aqi_fg.read()
label = df.iloc[-1]["aqi_category"]

# # Save actual label as an image
# label_flower = "assets/" + label + ".png"
# img = Image.open(label_flower)
# img.save("../../assets/actual_iris.png")

# Monitor predictions
monitor_fg = fs.get_or_create_feature_group(name="aqi_predictions",
                                  version=1,
                                  primary_key=["datetime"],
                                  description="Air Quality Prediction/Outcome Monitoring"
                                 )

now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
last_row = df.iloc[-1]
data = {
    'country': [last_row['country']],  # Convert to list for DataFrame compatibility
    'city': [last_row['city']],
    'aqi_value': [last_row['aqi_value']],
    'co_aqi_value': [last_row['co_aqi_value']],
    'ozone_aqi_value': [last_row['ozone_aqi_value']],
    'no2_aqi_value': [last_row['no2_aqi_value']],
    'pm25_aqi_value': [last_row['pm25_aqi_value']],
    'lat': [last_row['lat']],
    'lng': [last_row['lng']],
    'prediction': [category],  # Ensure 'flower' is a scalar value
    'label': [label],        # Ensure 'label' is a scalar value
    'datetime': [now],
}
monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df)

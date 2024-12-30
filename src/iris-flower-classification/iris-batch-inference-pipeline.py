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

# Load the Iris model
model = mr.get_model("iris", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/iris_model.pkl")

# Load feature view
feature_view = fs.get_feature_view(name="iris", version=1)

# Perform batch inference
batch_data = feature_view.get_batch_data()
y_pred = model.predict(batch_data)

# Save prediction output as an image
flower = y_pred[y_pred.size - 1]
# flower_img = "assets/" + flower + ".png"
# img = Image.open(flower_img)
# img.save("../../assets/latest_iris.png")

# Retrieve actual label
iris_fg = fs.get_feature_group(name="iris", version=1)
df = iris_fg.read()
label = df.iloc[-1]["variety"]

# # Save actual label as an image
# label_flower = "assets/" + label + ".png"
# img = Image.open(label_flower)
# img.save("../../assets/actual_iris.png")

# Monitor predictions
monitor_fg = fs.get_or_create_feature_group(
    name="iris_predictions",
    version=1,
    primary_key=["datetime"],
    description="Iris flower Prediction/Outcome Monitoring"
)

now = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
data = {
    's_length': [float(df.iloc[-1]['sepal_length'])],
    's_width': [float(df.iloc[-1]['sepal_width'])],
    'p_length': [float(df.iloc[-1]['petal_length'])],
    'p_width': [float(df.iloc[-1]['petal_width'])],
    'prediction': [flower],
    'label': [label],
    'datetime': [now],
}
monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df)

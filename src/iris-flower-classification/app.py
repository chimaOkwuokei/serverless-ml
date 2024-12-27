import gradio as gr
import numpy as np
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
from threading import Thread
import time

# Login to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load model
mr = project.get_model_registry()
model = mr.get_model("iris", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/iris_model.pkl")

# Define a function to make predictions
def iris(sepal_length, sepal_width, petal_length, petal_width):
    input_list = [sepal_length, sepal_width, petal_length, petal_width]
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    flower_label = res[0]

    # Fetch the flower image
    flower_url = f"https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/{flower_label}.png"
    img = Image.open(requests.get(flower_url, stream=True).raw)

    # Add prediction to monitor feature group
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        's_length': [sepal_length],
        's_width': [sepal_width],
        'p_length': [petal_length],
        'p_width': [petal_width],
        'prediction': [flower_label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg = fs.get_or_create_feature_group(
        name="iris_predictions",
        version=1,
        primary_key=["datetime"],
        description="Iris flower prediction monitoring"
    )
    monitor_fg.insert(monitor_df)

    return img, f"Predicted Flower: {flower_label}"

# Function to fetch and display real-time data
def fetch_monitor_data():
    monitor_fg = fs.get_feature_group(name="iris_predictions", version=1)
    fg_df = monitor_fg.read()
    return fg_df

# Set up a Gradio interface for predictions
demo = gr.Interface(
    fn=iris,
    title="Iris Flower Predictive Analytics",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="sepal length (cm)"),
        gr.inputs.Number(default=1.0, label="sepal width (cm)"),
        gr.inputs.Number(default=1.0, label="petal length (cm)"),
        gr.inputs.Number(default=1.0, label="petal width (cm)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Flower Image"),
        gr.Text(label="Prediction"),
    ],
)

# Set up a live DataFrame viewer
data_viewer = gr.Dataframe(
    label="Real-Time Predictions Monitor",
    value=fetch_monitor_data(),
    interactive=False
)

# Periodically update the DataFrame
def update_dataframe():
    while True:
        updated_df = fetch_monitor_data()
        data_viewer.update(value=updated_df)
        time.sleep(5)  # Update every 5 seconds

# Start the update thread
update_thread = Thread(target=update_dataframe, daemon=True)
update_thread.start()

# Launch both interfaces together
demo.launch(share=True)


# import gradio as gr
# import numpy as np
# from PIL import Image
# import requests

# import hopsworks
# import joblib

# project = hopsworks.login()
# fs = project.get_feature_store()


# mr = project.get_model_registry()
# model = mr.get_model("iris", version=1)
# model_dir = model.download()
# model = joblib.load(model_dir + "/iris_model.pkl")


# def iris(sepal_length, sepal_width, petal_length, petal_width):
#     input_list = []
#     input_list.append(sepal_length)
#     input_list.append(sepal_width)
#     input_list.append(petal_length)
#     input_list.append(petal_width)
#     # 'res' is a list of predictions returned as the label.
#     res = model.predict(np.asarray(input_list).reshape(1, -1)) 
#     # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
#     # the first element.
#     flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
#     img = Image.open(requests.get(flower_url, stream=True).raw)            
#     return img
        
# demo = gr.Interface(
#     fn=iris,
#     title="Iris Flower Predictive Analytics",
#     description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
#     allow_flagging="never",
#     inputs=[
#         gr.inputs.Number(default=1.0, label="sepal length (cm)"),
#         gr.inputs.Number(default=1.0, label="sepal width (cm)"),
#         gr.inputs.Number(default=1.0, label="petal length (cm)"),
#         gr.inputs.Number(default=1.0, label="petal width (cm)"),
#         ],
#     outputs=gr.Image(type="pil"))

# demo.launch()

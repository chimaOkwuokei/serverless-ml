# Air Quality Prediction Dashboard

Welcome to the Air Quality Prediction Dashboard! This project uses real-time data and predictive models to provide insights into air quality around the world.

## Overview

This dashboard showcases:

- **Real-time AQI Data**: Dynamically generated air quality index data based on historical records.
- **Predictive Modeling**: Predictions on air quality using machine learning models.
- **Interactive Visualizations**: User-friendly interface for exploring and understanding air quality data.

## Technologies Used

- **Hugging Face Spaces** - For hosting our web application in a serverless manner.
- **Gradio** - To build an interactive and intuitive user interface.
- **Hopsworks** - As our feature store and model registry, managing data and models efficiently.

## Features

### Data Generation
- **Dynamic AQI Generation**: 
  - Utilizes historical AQI data to generate new, random yet realistic air quality metrics for various locations around the globe.
  - Features like AQI values, CO, Ozone, NO2, and PM2.5 are dynamically calculated.

### Data Storage & Management
- **Hopsworks Feature Store**: 
  - Stores all generated AQI data with unique identifiers using UUIDs for traceability and data integrity.
  - Handles the backfill of historical data or the insertion of newly generated data.

### Model Registry
- **Hopsworks Model Registry**: 
  - Manages different versions of our air quality prediction models for deployment and experimentation.

### User Interface
- **Gradio Interface**:
  - Offers a user-friendly way to interact with the data, view predictions, and visualize air quality trends.
  - Provides controls for selecting different cities or countries to check their current and predicted air quality.

## Setup & Running the Application

### Prerequisites
- Python 3.8+
- Necessary Python libraries (pandas, hopsworks, uuid, etc.)
- An account on Hugging Face and Hopsworks for deployment and data management.

### Steps to Deploy

1. **Clone the Repository**:
   git clone (https://github.com/chimaOkwuokei/serverless-ml/)

Install Dependencies:
bash
pip install -r requirements.txt

Set Up Environment Variables:
HOPSWORKS_API_KEY: Your Hopsworks project API key.
HUGGING_FACE_TOKEN: Your Hugging Face token for Spaces.

Run Locally:
bash
python main.py  # Assuming your script is named main.py
Deploy to Hugging Face Spaces:
Create a new Space on Hugging Face with Gradio SDK.
Upload your project code, including the script and any static files like requirements.txt.
Configure your Space with the necessary environment variables.

Code Structure
main.py: Contains the logic for data generation, model interaction, and Gradio interface.
requirements.txt: Lists all Python packages required for the project.

Data Model
Columns: 
country, city, aqi_value, aqi_category, co_aqi_value, ozone_aqi_value, no2_aqi_value, pm25_aqi_value, lat, lng, uuid

License
MIT License (LICENSE) - See the LICENSE file for details.

Contributions
Contributions are welcome! Please fork the repository and submit pull requests.

Contact
For any queries or collaboration, please reach out via GitHub issues (https://github.com/chimaOkwuokei/serverless-ml/).

Thank you for exploring our Air Quality Prediction Dashboard! Let's work together to breathe cleaner air.

This README provides a clear, concise overview of the project, its technologies, and how to set it u

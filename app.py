import numpy as np
import gradio as gr
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")  # Ensure the file name is correct


def predict_bike_rental(season, hr, holiday, workingday, weathersit, temp, atemp, hum, windspeed, yr, mnth, 
                        weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, weekday_Thu, weekday_Tue, weekday_Wed):

    input_features = np.array([[season, hr, holiday, workingday, weathersit, temp, atemp, hum, windspeed, yr, mnth, 
                                weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, weekday_Thu, weekday_Tue, weekday_Wed]])

    print("Input shape:", input_features.shape)  # Debugging step

    # Make a prediction
    prediction = model.predict(input_features)
    return f"Predicted Bike Rentals: {int(prediction[0])}"

# Define Gradio UI
inputs = [
    gr.Number(label="Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)"),
    gr.Number(label="Hour (0-23)"),
    gr.Number(label="Holiday (0: No, 1: Yes)"),
    gr.Number(label="Working Day (0: No, 1: Yes)"),
    gr.Number(label="Weather (1-4)"),
    gr.Number(label="Temperature (normalized)"),
    gr.Number(label="Feels-like Temperature (normalized)"),
    gr.Number(label="Humidity (normalized)"),
    gr.Number(label="Wind Speed (normalized)"),
    gr.Number(label="Year (0: 2011, 1: 2012)"),
    gr.Number(label="Month (1-12)"),
    gr.Number(label="Friday (0 or 1)"),
    gr.Number(label="Monday (0 or 1)"),
    gr.Number(label="Saturday (0 or 1)"),
    gr.Number(label="Sunday (0 or 1)"),
    gr.Number(label="Thursday (0 or 1)"),
    gr.Number(label="Tuesday (0 or 1)"),
    gr.Number(label="Wednesday (0 or 1)")
]

app = gr.Interface(fn=predict_bike_rental, inputs=inputs, outputs="text", title="Bike Rental Prediction")
app.launch()

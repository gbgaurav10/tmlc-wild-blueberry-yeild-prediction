import streamlit as st
import pandas as pd 
import pickle
import joblib
from wild_blueberry.pipeline.predictions import PredictionPipeline
import warnings
warnings.filterwarnings("ignore")


# Load the model
model = joblib.load("final_model/model.joblib")

# Load the preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")



def main():
    st.title('Wild Blue-Berry Prediction WebApp')

    clonesize = st.slider('clonesize', 10.0, 40.0, step=1.0)
    honeybee = st.slider('Honeybee', 0.0, 19.0, step=0.1)
    bumbles = st.slider('Bumbles', 0.0, 0.60, step=0.01)
    andrena = st.slider('Andrena', 0.0, 0.75, step=0.1)
    osmia = st.slider('Osmia', 0.0, 0.75, step=0.1)
    MaxOfUpperTRange = st.slider('MaxOfUpperTRange', 69, 95, step=0.1)
    RainingDays = st.slider('RainingDays', 1.0, 35.0, step=0.1)
    fruitset = st.slider('Fruitset', 0.19, 0.68, step=0.1)

    # Define the form
    with st.form("Wild-Blue_Berry"):
        # Add submit button inside the form
        submit_button = st.form_submit_button("Submit")
    

    if submit_button:
        try:
            # Create a dataframe
            input_data = pd.DataFrame({
                "clonesize": [clonesize],
                "honeybee": [honeybee],
                "bumbles": [bumbles],
                "andrena": [andrena],
                "osmia": [osmia],
                "MaxOfUpperTRange": [MaxOfUpperTRange],
                "RainingDays": [RainingDays],
                "fruitset": [fruitset]
            })

            # Preprocess the input data
            X_transformed = preprocessor.transform(input_data)

            # Make the prediction
            prediction = model.predict(X_transformed)[0]  # Assuming you want a single prediction

            # Display the prediction to the user
            st.write(f"Predicted Wild Blue-Berry Yield: {prediction}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

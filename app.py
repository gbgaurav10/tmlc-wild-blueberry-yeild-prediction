import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the model
model = joblib.load("final_model/model.joblib")

# Load the preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")

# Define a function to display the image as a banner
def display_banner_image(image_path):
    st.image(image_path, use_column_width=True, output_format="auto")



def main():
    # Display the image at the top of the app
    display_banner_image("images/OIP.jpeg")

    st.title('Blue-Berry Yield Prediction App')

    # Set default values for sliders or use None for initial values
    clonesize = st.slider('Clonesize', 10, 40, step=1, value=None)
    honeybee = st.slider('Honeybee', 0.0, 19.0, step=0.1, value=None)
    bumbles = st.slider('Bumbles', 0.0, 0.60, step=0.01, value=None)
    andrena = st.slider('Andrena', 0.0, 0.75, step=0.1, value=None)
    osmia = st.slider('Osmia', 0.0, 0.75, step=0.1, value=None)
    MaxOfUpperTRange = st.slider('MaxOfUpperTRange', 69, 95, step=1, value=None)
    RainingDays = st.slider('RainingDays', 1.0, 35.0, step=0.1, value=None)
    fruitset = st.slider('Fruitset', 0.19, 0.68, step=0.1, value=None)

    # Define the form
    with st.form("Wild-Blue_Berry"):
        # Add submit button inside the form
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        try:
            # Check if any of the sliders are None (user hasn't selected a value)
            if any(value is None for value in [clonesize, honeybee, bumbles, andrena, osmia, MaxOfUpperTRange, RainingDays, fruitset]):
                st.warning("Please select values for all input parameters.")
            else:
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
                prediction = model.predict(X_transformed)[0]  

                # Display the prediction to the user
                st.write(f"Predicted Wild Blue-Berry Yield: {prediction}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

import streamlit as st 
import pandas as pd 
import joblib 
import warnings 
warnings.filterwarnings("ignore")


# Load the Model
model = joblib.load("final_model/wildblueberry_model.joblib")

# Load the Preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")


# Define the main function
def main():
    # Set page title and layout
    st.set_page_config(page_title="Wild Blue-Berry Yield Prediction", layout="wide")

    st.title("Wild Blue-Berry Yield Prediction App")

    # Define form inputs and initial values
    clonesize = st.slider("Clonesize", 10.0, 40.0, step=1)

    honeybee = st.slider("Honeybee", 0.0, 19.0, step=0.1)
    bumbles = st.slider("Bumbles", 0.0, 0.60, step=0.1)
    andrena = st.slider("Andrena", 0.0, 0.90, step=0.1)
    osmia = st.slider("Osmia", 0.0, 0.90, step=0.1)
    MaxOfUpperTRange = st.slider("MaxOfUpperTRange", 60.0, 100, step=0.10)
    Rainingdays = st.slider("Rainingdays", 1.0, 35.0, step=1)
    fruitset = st.slider("Fruitset", 0.10, 0.70, step=0.1)

    # Define the form
    with st.form("Income_inequality_form"):
        # Add submit button inside the form
        submit_button = st.form_submit_button("Submit")

        # If submit button is clicked
        if submit_button:
            try:
                # Create a DataFrame with the selected features
                input_data = pd.DataFrame({
                    "clonesize": [clonesize],
                    "honeybee": [honeybee],
                    "bumbles": [bumbles],
                    "andrena": [andrena],
                    "osmia": [osmia],
                    "MaxOfUpperTRange": [MaxOfUpperTRange],
                    "Rainingdays": [Rainingdays],
                    "fruitset": [fruitset]
                })

                # Ensure correct data types for the input features
                input_data = input_data.astype({
                    "clonesize": int,
                    "honeybee": int,
                    "bumbles": int,
                    "andrena": float,
                    "osmia": int,
                    "MaxOfUpperTRange": float,
                    "Rainingdays": float,
                    "fruitset": float

                })

                # Make the prediction
                X_transformed = preprocessor.transform(input_data)
                prediction = model.predict(X_transformed)

                # Map the prediction to human-readable labels
                Yield = income_mapping.get(prediction[0], "Unknown")

                # Show the prediction
                st.subheader("Prediction:")
                st.write("The predicted Yield is:", Yield)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Run the main function
if __name__ == "__main__":
    main()

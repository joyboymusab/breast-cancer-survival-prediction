import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle


# Function to scale values to range [0, 1]
# Get scaled values
def get_scaled_values(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


# Preprocess data function
def get_clean_data():
    data = pd.read_csv("/Users/Lenovo/Desktop/Projects Duc/Breast_Cancer.csv")

    data.rename(columns={"T Stage ": "T Stage"}, inplace=True)
    data["Grade"].replace({" anaplastic; Grade IV": "4"}, inplace=True)
    data["Grade"] = pd.to_numeric(data["Grade"], errors="coerce")
    data["T Stage"].replace({"T1": 1, "T2": 2, "T3": 3, "T4": 4}, inplace=True)
    data["T Stage"] = pd.to_numeric(data["T Stage"], errors="coerce")
    data["N Stage"].replace({"N1": 1, "N2": 2, "N3": 3}, inplace=True)
    data["N Stage"] = pd.to_numeric(data["N Stage"], errors="coerce")
    data["6th Stage"].replace(
        {"IIA": 1, "IIB": 2, "IIIA": 3, "IIIB": 4, "IIIC": 5}, inplace=True
    )
    data["6th Stage"] = pd.to_numeric(data["6th Stage"], errors="coerce")
    data["differentiate"].replace(
        {
            "Moderately differentiated": 2,
            "Poorly differentiated": 1,   
            "Well differentiated": 3,
            "Undifferentiated": 0,
        },
        inplace=True,
    )
    data["differentiate"] = pd.to_numeric(data["differentiate"], errors="coerce")
    data["A Stage"].replace({"Regional": 1, "Distant": 0}, inplace=True)
    data["A Stage"] = pd.to_numeric(data["A Stage"], errors="coerce")
    data["Estrogen Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
    data["Estrogen Status"] = pd.to_numeric(data["Estrogen Status"], errors="coerce")
    data["Progesterone Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
    data["Progesterone Status"] = pd.to_numeric(
        data["Progesterone Status"], errors="coerce"
    )
    data["Status"].replace({"Alive": 1, "Dead": 0}, inplace=True)
    data["Status"] = pd.to_numeric(data["Status"], errors="coerce")
    data["Marital Status"].replace(
        {"Married": 1, "Single": 2, "Divorced": 3}, inplace=True
    )
    data["Marital Status"] = pd.to_numeric(data["Marital Status"], errors="coerce")
    data["Race"].replace({"White": 1, "Black": 0}, inplace=True)
    data["Race"] = pd.to_numeric(data["Race"], errors="coerce")
    data = pd.get_dummies(data, columns=["Marital Status"], dtype=int)
    data.rename(columns={"Race": "Race_Black"}, inplace=True)
    data["Race_White"] = 1 - data["Race_Black"]
    data.dropna(inplace=True)
    return data


# Sidebar function
def add_sidebar():
    st.sidebar.header("Features Measurements")
    slider_labels = [
        ("Age", "Age", 30, 69),
        ("Tumor Size (mm)", "Tumor Size", 1, 140),
        ("ER Status (Estrogen Receptor)", "Estrogen Status", 0, 1),
        ("PR Status (Progesterone Receptor)", "Progesterone Status", 0, 1),
        ("Grade (IV Anaplastic to 4)", "Grade", 1, 4),
        ("T Stage", "T Stage", 1, 4),
        ("N Stage", "N Stage", 1, 3),
        ("6th Stage", "6th Stage", 1, 5),
        ("Differentiation", "differentiate", 0, 3),
        ("A Stage (Regional/Distant)", "A Stage", 0, 1),
        ("Regional Node Examined", "Regional Node Examined", 1, 61),
        ("Regional Node Positive", "Regional Node Positive", 1, 46),
        ("Survival Months", "Survival Months", 1, 107),
    ]
    data = {}
    for label, key, min_val, max_val in slider_labels:
        data[key] = st.sidebar.slider(
            label, min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2
        )
    race = st.sidebar.selectbox("Race", options=["White", "Black"])
    if race == "White":
        data["Race_White"], data["Race_Black"] = 1, 0
    else:
        data["Race_White"], data["Race_Black"] = 0, 1
    marital_status = st.sidebar.selectbox(
        "Marital Status", options=["Married", "Single", "Divorced", "Widowed"]
    )
    (
        data["Marital Status_Married"],
        data["Marital Status_Single"],
        data["Marital Status_Divorced"],
        data["Marital Status_Widowed"],
    ) = (0, 0, 0, 0)
    if marital_status == "Married":
        data["Marital Status_Married"] = 1
    elif marital_status == "Single":
        data["Marital Status_Single"] = 1
    elif marital_status == "Divorced":
        data["Marital Status_Divorced"] = 1
    else:
        data["Marital Status_Widowed"] = 1
    return data


# Radar chart function
# Radar chart function
def get_radar_chart(user_inputs):
    # Define categories (adjusting for marital status and race)
    categories = [
        "Age",
        "Tumor Size",
        "ER Status",
        "PR Status",
        "Grade",
        "T Stage",
        "N Stage",
        "6th Stage",
        "Differentiation",
        "A Stage",
        "Marital Status (One-Hot)",
        "Race (One-Hot)",
    ]
    categories = [
        *categories,
        categories[0],
    ]  # Close the loop by repeating the first category

    # Dynamically use slider inputs for the values
    marital_status_combined = (
        user_inputs["Marital Status_Married"]
        + user_inputs["Marital Status_Single"]
        + user_inputs["Marital Status_Divorced"]
    )
    race_combined = user_inputs["Race_White"] + user_inputs["Race_Black"]

    values = [
        user_inputs["Age"],
        user_inputs["Tumor Size"],
        user_inputs["Estrogen Status"],
        user_inputs["Progesterone Status"],
        user_inputs["Grade"],
        user_inputs["T Stage"],
        user_inputs["N Stage"],
        user_inputs["6th Stage"],
        user_inputs["differentiate"],
        user_inputs["A Stage"],
        marital_status_combined,
        race_combined,
    ]

    # Scale the values using the same min-max ranges from sliders
    scaled_values = [
        get_scaled_values(user_inputs["Age"], 30, 69),
        get_scaled_values(user_inputs["Tumor Size"], 1, 140),
        get_scaled_values(user_inputs["Estrogen Status"], 0, 1),
        get_scaled_values(user_inputs["Progesterone Status"], 0, 1),
        get_scaled_values(user_inputs["Grade"], 1, 4),
        get_scaled_values(user_inputs["T Stage"], 1, 4),
        get_scaled_values(user_inputs["N Stage"], 1, 3),
        get_scaled_values(user_inputs["6th Stage"], 1, 5),
        get_scaled_values(user_inputs["differentiate"], 0, 3),
        get_scaled_values(user_inputs["A Stage"], 0, 1),
        get_scaled_values(
            marital_status_combined, 0, 1
        ),  # One-hot encoded, so range 0-1
        get_scaled_values(race_combined, 0, 1),  # One-hot encoded, so range 0-1
    ]
    scaled_values = [*scaled_values, scaled_values[0]]  # Close the loop

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scaled_values, theta=categories, fill="toself", name="User Input Values"
        )
    )

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1]  # Values are scaled between 0 and 1
            )
        ),
        showlegend=True,
    )

    return fig


# Prediction function
# Prediction function
def add_predictions(user_inputs):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    columns_order = [
        "Age",
        "Race_Black",
        "T Stage",
        "N Stage",
        "6th Stage",
        "differentiate",
        "Grade",
        "A Stage",
        "Tumor Size",
        "Estrogen Status",
        "Progesterone Status",
        "Regional Node Examined",
        "Regional Node Positive",
        "Survival Months",
        "Marital Status_Single",
        "Marital Status_Married",  # Marital status columns (One-Hot)
        "Marital Status_Divorced",
        "Marital Status_Widowed",
        "Race_White"
    ]

    # Extract the values in the correct order and convert them to a list
    user_inputs = [user_inputs[column] for column in columns_order]
    input_array = np.array(user_inputs).reshape(1, -1)
    print("user_inputs", input_array)
    # Scale the input data
    input_array_scaled = scaler.transform(input_array)

    # Make the prediction
    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)

    st.subheader("Features cluster prediction")
    st.write("The features cluster is:")

    # Display prediction and probability
    st.write("Prediction: Dead" if prediction[0] == 0 else "Prediction: Alive")
    st.write("Probability of being Dead: {:.2f}".format(prediction_proba[0][0]))
    st.write("Probability of being Alive: {:.2f}".format(prediction_proba[0][1]))
    st.write(
        "This app can assist medical professionals in making a treatment plan, but should not be used as a substitute for a professional diagnosis."
    )


# Main function
def main():
    st.set_page_config(
        page_title="Breast Cancer Survivability Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    data = add_sidebar()
    with st.container():
        st.title("Breast Cancer Survivability Predictor")
        st.write(
            "This app predicts breast cancer survivability by analyzing patient features, aiding medical professionals in planning treatments."
        )
    col1, col2 = st.columns([4, 1])

    # Columns for radar chart and other content
    col1, col2 = st.columns([4, 1])

    with col1:
        # Get radar chart based on user inputs
        radar_chart = get_radar_chart(data)
        st.plotly_chart(radar_chart)

    with col2:
        # Pass the user inputs to the prediction function
        add_predictions(data)

if __name__ == "__main__":
    main()
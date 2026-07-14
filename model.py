import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data):
    # Drop the target column for feature scaling
    X = data.drop(["Status"], axis=1)
    y = data["Status"]

    # Scale the data
    scaler = StandardScaler()
    x = scaler.fit_transform(X)  # Apply scaling to X

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression(
        max_iter=1000
    )  # Set max_iter to avoid convergence issues
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv("C:/Users/smart/Desktop/Projects Duc/Breast_Cancer.csv")

    # Pre-processing steps

    data.rename(columns={"T Stage ": "T Stage"}, inplace=True)

    # Handle 'Grade' column
    data["Grade"].replace({" anaplastic; Grade IV": "4"}, inplace=True)
    data["Grade"] = pd.to_numeric(
        data["Grade"], errors="coerce"
    )  # Convert to numeric, set errors to NaN

    # Handle 'T Stage' column
    data["T Stage"].replace({"T1": 1, "T2": 2, "T3": 3, "T4": 4}, inplace=True)
    data["T Stage"] = pd.to_numeric(data["T Stage"], errors="coerce")

    # Handle 'N Stage' column
    data["N Stage"].replace({"N1": 1, "N2": 2, "N3": 3}, inplace=True)
    data["N Stage"] = pd.to_numeric(data["N Stage"], errors="coerce")

    # Handle '6th Stage' column
    data["6th Stage"].replace(
        {"IIA": 1, "IIB": 2, "IIIA": 3, "IIIB": 4, "IIIC": 5}, inplace=True
    )
    data["6th Stage"] = pd.to_numeric(data["6th Stage"], errors="coerce")

    # Handle 'differentiate' column
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

    # Handle 'A Stage' column
    data["A Stage"].replace({"Regional": 1, "Distant": 0}, inplace=True)
    data["A Stage"] = pd.to_numeric(data["A Stage"], errors="coerce")

    # Handle 'Estrogen Status' column
    data["Estrogen Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
    data["Estrogen Status"] = pd.to_numeric(data["Estrogen Status"], errors="coerce")

    # Handle 'Progesterone Status' column
    data["Progesterone Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
    data["Progesterone Status"] = pd.to_numeric(
        data["Progesterone Status"], errors="coerce"
    )

    # Handle 'Status' column
    data["Status"].replace({"Alive": 1, "Dead": 0}, inplace=True)
    data["Status"] = pd.to_numeric(data["Status"], errors="coerce")

    # Handle 'Marital Status' column
    data["Marital Status"] = data["Marital Status"].str.strip()
    data["Marital Status"].replace(
        {"Married": 1, "Single": 0, "Divorced": 2, "Widowed": 3}, inplace=True
    )
    data["Marital Status"] = pd.to_numeric(data["Marital Status"], errors="coerce")

    # Convert Race to only Black and White categories
    data["Race"].replace({"White": 1, "Black": 0, "other": 2}, inplace=True)
    data["Race"] = pd.to_numeric(data["Race"], errors="coerce")

    # Ensure Race has only two columns in the final dataset
    data = pd.get_dummies(data, columns=["Marital Status"], dtype=int)
    data.rename(columns={"Race": "Race_Black"}, inplace=True)
    data["Race_White"] = (
        1 - data["Race_Black"]
    )  # Create Race_White as the complement of Race_Black

    # Drop any remaining NaN values
    data.dropna(inplace=True)  # Drop rows with NaN values

    # Ensure the final dataset has the expected number of features
    print("Columns in final dataset:", data.columns)

    return data


def main():
    # Load and clean the data
    data = get_clean_data()

    # Create and train the model
    model, scaler = create_model(data)

    # Ensure the model folder exists
    if not os.path.exists("model"):
        os.makedirs("model")

    # Save the model and scaler
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()

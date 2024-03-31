import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load dataset
df = pd.read_csv("worklifebalance_new.csv")  # Load your dataset here

binary_map = {'Yes': 1, 'No': 0, }
df['Pressure'] = df['Pressure'].map(binary_map)
df['Burnout'] = df['Burnout'].map(binary_map)
binary_map2 ={'<8 hours': 0, '>8 but <12': 1, '>12 hours': 2}
df['Working hours'] = df['Working hours'].map(binary_map2)
df.to_csv('output_file.csv', index=False)

data2= pd.read_csv("output_file.csv")

# Convert categorical variables to numeric
label_encoder = LabelEncoder()
data2['stressed'] = label_encoder.fit_transform(data2['stressed'])  # Assuming 'stressed' column contains 'No' and 'Yes'
data2['stressed'] = data2['stressed'].astype(int)

# Preprocessing
X = data2[['Burnout', 'Pressure', 'Working hours']]  # Select relevant features
y = data2['stressed']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Prediction function
def predict_stress(burn_out, pressure, working_hours):
    input_data = [[burn_out, pressure, working_hours]]
    prediction = rf_classifier.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Employee Stress Predictor")
    st.write("Enter employee details to predict stress level.")

    burn_out = st.slider("Burn Out", min_value=0, max_value=10, step=1)
    pressure = st.slider("Pressure", min_value=0, max_value=10, step=1)
    working_hours = st.slider("Working Hours", min_value=0, max_value=24, step=1)

    if st.button("Predict"):
        prediction = predict_stress(burn_out, pressure, working_hours)
        if prediction == 1:
            st.error("Employee is stressed.")
        else:
            st.success("Employee is not stressed.")

if __name__ == "__main__":
    main()

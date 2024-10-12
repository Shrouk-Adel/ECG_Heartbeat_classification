import pickle
import streamlit as st
import pandas as pd


# Loading saved model 
logistic_model = pickle.load(open('logistc.sav','rb'))
xgboost_model = pickle.load(open('xgboost.sav','rb'))
scaler = pickle.load(open('scaler.sav','rb'))



def HearBeat_prediction(df):
    # drop for last 50 feature
    input= df.iloc[:, :-50]    # Drops the last 50 columns
    
    # scaling test data
    scaled_data = scaler.transform(input)
    
    # make prediction 
    predictions =[]   # list to store final prediction

    logistic_predictions = logistic_model.predict(scaled_data)
    
    for i in range(len(logistic_predictions)):
        lo_pred = logistic_predictions[i]
        if lo_pred == 0:
            predictions.append('N - Normal Beat')
        
        else: 
            xgboost_pred = (xgboost_model.predict(scaled_data[i].reshape(1,-1)))+1
            
            if xgboost_pred == 1:
                predictions.append('Q - Unclassified beat')
            elif xgboost_pred == 2:
                predictions.append('V - Premature ventricular contraction')
            elif xgboost_pred == 3:
                predictions.append( 'S - Supraventricular premature or ectopic beat')
            else: 
                predictions.append( 'F - Fusion of ventricular and normal beat')

    return predictions


def display_predictions(predictions):
    # Create DataFrame for predictions
    df_predictions = pd.DataFrame(predictions, columns=['Prediction'])  # Ensure correct column name

    # Add an index column for better reference
    df_predictions['Index'] = df_predictions.index + 1

    # Map colors based on predictions
    color_map = {
        'N - Normal Beat': 'green',
        'Q - Unclassified beat': 'red',
        'V - Premature ventricular contraction': 'orange',
        'S - Supraventricular premature or ectopic beat': 'gray',
        'F - Fusion of ventricular and normal beat': 'blue'
    }
    
    # Create a new column for colors based on predictions
    df_predictions['Color'] = df_predictions['Prediction'].map(color_map)

    # Function to apply color styling
    def highlight_row(row):
        return ['color: {}'.format(row['Color'])] * len(row)

    # Apply styling
    styled_df = df_predictions.style.apply(highlight_row, axis=1)

    # Display the predictions in an interactive table
    st.write("### Predictions:")
    st.dataframe(styled_df)  # Streamlit expects a Styler object



def main():

   # giving a title
   st.title('ECG Heartbeat prediction Web App')
   
   # write info about ECG Heartbeat 
   st.write("""
    ### What is an ECG?
    An Electrocardiogram (ECG) is a test that measures the electrical activity of the heart. It records the heart's rhythm and electrical signals, helping detect various heart conditions.

    ### How This App Works
    This app uses a machine learning model trained to classify different types of heartbeats based on ECG data. It first processes the input data, performs feature scaling, and then applies a trained model to predict the type of heartbeat.

    ### Types of Heartbeats Classified
    1. **Normal Beat**: A healthy heartbeat.
    2. **Premature Ventricular Contraction (PVC)**: An early beat originating from the ventricles.
    3. **Unclassified beat'**
    4. **Supraventricular premature or ectopic beat**
    5. **Fusion of ventricular and normal beat**

    ### Disclaimer
    This app is intended for educational purposes and should not be used for medical diagnosis. Always consult a healthcare professional for medical advice.
    """)
    
   st.image('Heartbeat.jpeg',caption="ECG Heartbeat classification",use_column_width=True)

    # File upload: User can upload an Excel file
   uploaded_file = st.file_uploader("Upload your data", type=["csv"])

   if uploaded_file is not None:
       # Read the uploaded Excel file into a pandas DataFrame
       df = pd.read_csv(uploaded_file)

        # Show the uploaded data
       st.write("Uploaded Data:")
       st.write(df)

   Diagnosis =''

   if st.button('ECG Heartbeat predict'):
        Diagnosis = HearBeat_prediction(df)
        display_predictions(Diagnosis)
    



if __name__ == '__main__':
    main()


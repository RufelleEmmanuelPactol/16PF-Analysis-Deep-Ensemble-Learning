from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pages_.loader_controller import retrieve_data_from_file




def data_shaping(file_handle):
    df = pd.read_excel(file_handle, sheet_name="Student Fit Template")

    def format_features(df):
        # Rename '16PF' column to 'attr_A'
        df.rename(columns={'16PF': 'attr_A'}, inplace=True)

        # List of attributes as per the image provided
        attributes = ['B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'Q1', 'Q2', 'Q3', 'Q4', 'EX', 'AX', 'TM',
                      'IN', 'SC']

        # Loop through columns from 'Unnamed: 8' onwards and rename them according to the attributes list
        for i, attr in enumerate(attributes, start=7):
            old_column_name = f'Unnamed: {i}'
            new_column_name = f'attr_{attr}'
            df.rename(columns={old_column_name: new_column_name}, inplace=True)

        # Drop the first row

        return df

    df = df.drop(0)
    df = df.rename(columns={'ID Number': 'IDNumber'})


    df = format_features(df)
    return df


def predict_underway(df: pd.DataFrame, model):
    CFIT_map = {
        'L': 2, 'BA': 4, 'A': 6, 'AA': 8, 'H': 10}

    valid_cfit = {'L', 'BA', 'A', 'AA', 'H'}

    df = df[df['CFIT'].isin(valid_cfit)]

    for cfit in df['CFIT']:
        if cfit not in valid_cfit:
            raise Exception(
                "Invalid CFIT input, should be within '{'L', 'BA', 'A', 'AA', 'H'}'. Got" + cfit + " instead.")

    df['CFIT'] = df['CFIT'].apply(lambda x: CFIT_map[x])

    input_features = ['attr_A', 'attr_B', 'attr_C', 'attr_E', 'attr_F', 'attr_G', 'attr_H', 'attr_I', 'attr_L',
                      'attr_M',
                      'attr_N'
        , 'attr_O', 'attr_Q1', 'attr_Q2',
                      'attr_Q3', 'attr_Q4', 'attr_EX', 'attr_AX', 'attr_TM', 'attr_IN', 'attr_SC', 'CFIT',]

    bscs_fit = []
    bsit_fit = []
    with st.spinner(text='Predicting...'):
        for index, item in df.iterrows():
            st.write("Go")
            input = item[input_features]
            input_bscs = item[input_features]
            input_bsit = item[input_features]
            input_bscs['Course'] = 1
            input_bsit['Course'] = 0
            st.write(input_bscs.astype(np.float32))
            prediction = model.predict(input_bscs.astype(np.float32))
            st.write(prediction)
            break










import tensorflow.keras as keras
def ModelPredictionComponent():
    model = keras.models.load_model('deep-learning-model.keras')
    st.header("Model Prediction 🪄")
    st.markdown("⚠️\t For this section, utilize the given template to generate a prediction.")
    with open('DataTrainingTemplate.xlsx', 'rb') as file:
        file_data = file.read()

    with open('acc.tmp') as accuracy:
        accuracy = float(accuracy.readline())

    # Create a download button for the Excel file
    st.download_button(
        label="Download Prediction Template",
        data=file_data,
        file_name='DataPredictionTemplate.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    st.info(f"The model's last recorded accuracy was {round(accuracy, 2)}%. "
            f"", icon="ℹ️")

    file = st.file_uploader("Please load the dataset provided that it follows the template standard.", type="xlsx")


    lower_bound = 50 - (accuracy/2)
    upper_bound = 50 + (accuracy/2)

    if file is not None:

        data = data_shaping(file)
        predict_underway(data, model)

        """
        data = retrieve_data_from_file(file)
        bscs, bsit, _, _ = data
        bscs['Course'] = bscs['Course'].apply(lambda x: 1)
        bsit['Course'] = bsit['Course'].apply(lambda x: 0)

        CFIT_map = {
            'L': 2, 'BA': 4, 'A': 6, 'AA': 8, 'H': 10}

        valid_cfit = {'L', 'BA', 'A', 'AA', 'H'}
        bscs = bscs[bscs['CFIT'].isin(valid_cfit)]
        bsit = bsit[bsit['CFIT'].isin(valid_cfit)]

        for cfit in bscs['CFIT']:
            if cfit not in valid_cfit:
                raise Exception(
                    "Invalid CFIT input, should be within '{'L', 'BA', 'A', 'AA', 'H'}'. Got" + cfit + " instead.")

            for cfit in bsit['CFIT']:
                if cfit not in valid_cfit:
                    raise Exception(
                        "Invalid CFIT input, should be within '{'L', 'BA', 'A', 'AA', 'H'}'. Got" + cfit + " instead.")

        bscs['CFIT'] = bscs['CFIT'].apply(lambda x: CFIT_map[x])
        bsit['CFIT'] = bsit['CFIT'].apply(lambda x: CFIT_map[x])

        # st.dataframe(bscs)

        input_features = ['attr_A', 'attr_B', 'attr_C', 'attr_E', 'attr_F', 'attr_G', 'attr_H', 'attr_I', 'attr_L', 'attr_M',
             'attr_N'
                , 'attr_O', 'attr_Q1', 'attr_Q2',
             'attr_Q3', 'attr_Q4', 'attr_EX', 'attr_AX', 'attr_TM', 'attr_IN', 'attr_SC', 'CFIT', 'Course']

        merged = pd.concat([bsit, bscs], ignore_index=True)
        results = []
        with st.spinner("Prediction Underway..."):
            for index, row in merged.iterrows():
                inputf = row[input_features]
                result = model.predict([inputf])
                results.append(result[0])

            merged['prediction_weight'] = results
            dsc = []
            for value in merged['prediction_weight']:
                verdict = ''
                if value <= 20:
                    verdict = 'Will Fail'
                elif value < lower_bound:
                    verdict = 'Unlikely to Pass'
                elif lower_bound <= value <= upper_bound:
                    verdict = 'Borderline / Unsure'
                elif value >= 80:
                    verdict = 'Will Excel'
                elif value >= 70:
                    verdict = 'High Chance to Excel'
                else:
                    verdict = 'Likely to Pass'
                dsc.append(verdict)
            merged['conclusion'] = dsc
            merged['predicted_rank'] = merged['prediction_weight'].rank(ascending=False)
            ideal = merged[['IDNumber', 'Previous School', 'prediction_weight', 'conclusion', 'predicted_rank']]
            st.dataframe(ideal)

            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.close()
                processed_data = output.getvalue()
                return processed_data

            # Convert DataFrame to Excel
            excel_data = to_excel(ideal)

            # Create download button
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name='prediction.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
            """


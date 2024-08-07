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


import tensorflow.keras as keras


def ModelPredictionComponent():
    st.header("Model Prediction 🪄")
    st.markdown("⚠️ For this section, utilize the given template to generate a prediction.")

    # Load and display the template file
    with open('TemplateBatchTraining.xlsx', 'rb') as file:
        file_data = file.read()

    with open('acc.tmp') as accuracy:
        accuracy = accuracy.readline()

    st.download_button(
        label="Download Prediction Template",
        data=file_data,
        file_name='TemplateBatchTraining.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    st.info(f"The model's last recorded accuracy was {accuracy}.", icon="ℹ️")

    file = st.file_uploader("Please load the dataset (must follow the template standard)", type="xlsx")

    if file is not None:
        attributes = [
            'attr_A', 'attr_AX', 'attr_B', 'attr_C', 'attr_E', 'attr_EX', 'attr_F',
            'attr_G', 'attr_H', 'attr_I', 'attr_IN', 'attr_L', 'attr_M', 'attr_N',
            'attr_O', 'attr_Q1', 'attr_Q2', 'attr_Q3', 'attr_Q4', 'attr_SC', 'attr_TM', 'CFIT_A',
            'CFIT_AA', 'CFIT_BA', 'CFIT_H', 'CFIT_L', 'CFIT_M'
        ]

        with st.spinner('Loading models...'):
            failure_model = keras.models.load_model("model_failure.keras")
            success_model = keras.models.load_model("model_success.keras")
            pass_model = keras.models.load_model("model_pass.keras")
            ensemble_model = keras.models.load_model("model_ensemble.keras")

        df = data_shaping(file)
        validate_data(df)

        from pages_.model_training import create_consistent_dummies
        df = create_consistent_dummies(df, column='CFIT')
        df.dropna(inplace=True)

        results = {
            'IDNumber': [], "BSCS-Pass Confidence": [], "BSCS-Failure Confidence": [],
            "BSCS-Excellence Confidence": [], "BSIT-Pass Confidence": [], "BSIT-Failure Confidence": [],
            "BSIT-Excellence Confidence": [], "BSCS-Verdict": [], "BSIT-Verdict": []
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (index, row) in enumerate(df.iterrows()):
            status_text.text(f"Processing record {i + 1} of {len(df)}")
            progress_bar.progress((i + 1) / len(df))

            results['IDNumber'].append(row['IDNumber'])

            for course in ['bscs', 'bsit']:
                input_vector = row[attributes].copy()
                input_vector['course_bscs'] = course == 'bscs'
                input_vector['course_bsit'] = course == 'bsit'
                input_vector = sort_series_custom(input_vector)

                input_array = np.asarray(input_vector, np.float64).reshape(1, -1)

                fail = failure_model.predict(input_array, verbose=0)
                success = success_model.predict(input_array, verbose=0)
                passing = pass_model.predict(input_array, verbose=0)

                input_vector['fail'] = fail[0][0]
                input_vector['pass'] = passing[0][0]
                input_vector['success'] = success[0][0]

                input_vector = sort_series_custom(input_vector)
                input_array = np.asarray(input_vector, np.float64).reshape(1, -1)

                final_out = ensemble_model.predict(input_array, verbose=0)[0]

                results[f'{course.upper()}-Failure Confidence'].append(f"{round(final_out[0] * 100, 2)}%")
                results[f'{course.upper()}-Pass Confidence'].append(f"{round((final_out[1] + final_out[2]) * 100, 2)}%")
                results[f'{course.upper()}-Excellence Confidence'].append(f"{round(final_out[2] * 100, 2)}%")
                results[f'{course.upper()}-Verdict'].append(verdict(final_out))

        status_text.text("Processing complete!")
        progress_bar.empty()

        result_df = pd.DataFrame(results)
        st.subheader("Prediction Results")
        st.dataframe(result_df)

        # Add download button for results
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
        )


def validate_data(df):
    valid_cfit = ['A', 'AA', 'BA', 'H', 'L', 'M']
    for item in df.CFIT:
        if item in valid_cfit:
            continue
        else:
            raise ValueError(f"The cfit {item} is not valid, must be in {valid_cfit}.")


def custom_sort_key(item):
    # Split the item into prefix and suffix
    parts = item.split('_')
    prefix = parts[0]
    suffix = '_'.join(parts[1:]) if len(parts) > 1 else ''

    # Define a custom order for prefixes
    prefix_order = {'attr': 0, 'CFIT': 1, 'course': 2, 'fail': 3, 'pass': 4, 'success': 5}

    # Get the order for this prefix, defaulting to a high number if not found
    prefix_value = prefix_order.get(prefix, 100)

    # Return a tuple that will be used for sorting
    return (prefix_value, prefix.lower(), suffix.lower())


def sort_series_custom(series):
    sorted_index = sorted(series.index, key=custom_sort_key)
    sorted_series = pd.Series(index=sorted_index, dtype=series.dtype)
    for idx in sorted_index:
        sorted_series[idx] = series[idx]
    return sorted_series


def verdict(triplet):
    # Extract values for readability
    failure, passing, success = triplet

    # Define thresholds for high, medium, and low confidence
    high_confidence_threshold = 0.6  # "High confidence" threshold
    medium_confidence_threshold = 0.3  # "Medium confidence" threshold

    # High confidence assessments
    if failure > high_confidence_threshold:
        return "High Confidence in Failure"
    elif success > high_confidence_threshold:
        return "High Confidence in Excellence"
    elif passing > high_confidence_threshold:
        return "High Confidence in Passing"

    # Medium confidence assessments
    if medium_confidence_threshold < failure <= high_confidence_threshold:
        return "Medium Confidence in Failure"
    elif medium_confidence_threshold < success <= high_confidence_threshold:
        return "Medium Confidence in Excellence"
    elif medium_confidence_threshold < passing <= high_confidence_threshold:
        return "Medium Confidence in Passing"

    # Low confidence assessments
    if failure > passing and failure > success:
        return "Low Confidence in Failure"
    elif success > passing and success > failure:
        return "Low Confidence in Excellence"
    elif passing > failure and passing > success:
        return "Low Confidence in Passing"

    # Close call or no clear winner
    return "No Clear Dominant Outcome - Close Call"

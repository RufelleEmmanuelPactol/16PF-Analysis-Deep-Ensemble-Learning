import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, Lasso
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CategoryEncoding
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.layers import Dense, InputLayer
from imblearn.over_sampling import SMOTE

from settings.connectivity import get_engine
import joblib
import pages_.yoy_view
from pages_.yoy_view import get_tags
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np






def ModelTrainingComponent():
    st.header("Model Training Pipeline 🚀")
    st.markdown("This section is for retraining and training models according to loaded datasets. [V-1]")
    tags = get_tags()

    selected = st.multiselect(label="Select datasets to include in training", options=tags)
    if st.button("Start Training! 🚀"):
        gen_set = str([item.id for item in selected]).replace("[", "(").replace("]", ")")
        query = f"""
                        SELECT  (grade/5)*100 as weighted, attr_A, attr_B, attr_C, attr_E, attr_F, attr_H, attr_G, attr_I, attr_L, attr_M, attr_N, attr_O, attr_Q1, attr_Q2,
                                attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, attr_IN, attr_SC, IF(cfit = 'L', 2,
                                IF(cfit = 'BA', 4,
                                IF(cfit = 'A', 6, 
                                IF(cfit = 'AA', 8,
                                IF(cfit = 'H', 10, NULL))))) as cfit, CASE when course = 'BSCS' then 1 else 0 end as course
                        FROM students
                        INNER JOIN assessments s on s.student_id = students.Id WHERE tagID in {gen_set};
                        """

        df = fetch_data_as_dataframe(get_engine(), query)
        with st.spinner("Training model"):
            model, result = train_nn(df)

            log_loss = result[0]
            accuracy = result[1]

            model.save('deep-learning-model.keras')
            st.info(f"Finished training model, with log loss of {log_loss:.4f}, accuracy of {round(accuracy * 100, 2)}%.")

def fetch_data_as_dataframe(connection, query: str) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute(query)
    result_set = cursor.fetchall()
    column_names = cursor.column_names
    cursor.close()
    df = pd.DataFrame(result_set, columns=column_names)
    return df.dropna()

def build_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape=(input_shape,)),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),
        Dense(160, activation='relu'),

        Dense(num_classes, activation='softmax')  # Output layer with softmax activation for classification
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_nn(df: pd.DataFrame, k=5):
    import tensorflow as tf
    old_grades = df.weighted
    df['weighted'] = df.weighted.apply(discretize_weights)
    df = pd.get_dummies(df, columns=['course'])

    num_classes = 3
    st.write(f"Classes: [0: 'Fail', 1: 'Pass', 2: 'Excel']")

    # smote = SMOTE(random_state=42)
    X_resampled, y_resampled = (df.drop(columns=['weighted']), df['weighted'])

    X_resampled = X_resampled.values
    y_resampled = y_resampled.values

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    all_histories = []
    all_evaluations = []

    for train_index, test_index in kf.split(X_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        model = build_model(X_train.shape[1], num_classes)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


        history = model.fit(X_train, y_train, epochs=300, batch_size=15, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
        evaluation = model.evaluate(X_test, y_test)

        all_histories.append(history)
        all_evaluations.append(evaluation)

        st.write(f"Validation Fold {fold} Evaluation: (Log Loss: {evaluation[0]}, Accuracy: {evaluation[1]})")
        fold += 1

    avg_evaluation = np.mean(all_evaluations, axis=0)
    st.write(f"Average Evaluation: {avg_evaluation}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (Last Fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)

    accuracy_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])

    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_diff, label='Train - Validation Accuracy Difference')
    plt.title('Train - Validation Accuracy Difference (Last Fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference')
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)

    w_df = pd.DataFrame(model.predict(pd.DataFrame(df.drop(columns=['weighted']))))
    w_df['original'] = (old_grades / 100) * 5
    w_df = w_df.rename(columns={0: 'Confidence to Fail', 1: 'Confidence to Pass', 2: 'Confidence to Excel'})
    st.markdown("#### Initial Data Valuation")
    st.write(w_df)
    st.info(f"Mean train-test difference for the last fold: {round(accuracy_diff.mean() * 100, 2)}%")

    return model, model.evaluate(X_test, y_test)

def discretize_weights(weight):
    if weight < 50:
        return 0
    if 50 <= weight < 85:
        return 1
    if 85 <= weight <= 100:
        return 2

# Note: Ensure that all required dependencies are installed and the Streamlit app is set up correctly.
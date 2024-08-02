import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.python import keras

from pages_.yoy_view import get_tags
from settings.connectivity import get_engine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import mysql.connector

class TagHolder:

    def __init__(self, _id, name):
        self.id = _id
        self.name = name

    def __str__(self):
        return self.name


def get_tags():
    connection = get_engine()
    if not connection:
        return []

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * from datasetTag")
            tags = cursor.fetchall()

            tag_holder = []
            for tag in tags:
                tag_holder.append(TagHolder(tag[0], tag[1]))

            return tag_holder
    except Exception as e:
        return []
    finally:
        if connection.is_connected():
            connection.close()


import mysql.connector


def get_engine():
    # Create a MySQL connection using mysql.connector
    connection = mysql.connector.connect(
        host='monorail.proxy.rlwy.net',
        port=45826,
        user='root',
        password='VoUeejgBIkMgYiPmYHxMFsIXffwxCKBK',
        database='railway'
    )
    return connection


def discretize_weights(weight):
    if weight < 50:
        return 0
    if 50 <= weight < 85:
        return 1
    if 85 <= weight <= 100:
        return 2


def fetch_data_as_dataframe(connection, query: str) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute(query)
    result_set = cursor.fetchall()
    column_names = cursor.column_names
    cursor.close()
    df = pd.DataFrame(result_set, columns=column_names)
    return df.dropna()

def ModelTrainingComponent():
    import pandas as pd
    import streamlit as st

    from pages_.yoy_view import get_tags
    from settings.connectivity import get_engine
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, train_test_split
    from tensorflow.keras.layers import Dense, InputLayer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from imblearn.over_sampling import SMOTE
    import mysql.connector
    st.header("Model Training Pipeline 🚀")
    st.markdown("This section is for retraining and training models according to loaded datasets. [V-2]")
    tags = get_tags()

    selected = st.multiselect(label="Select datasets to include in training", options=tags)
    if st.button("Start Training! 🚀"):
        with st.spinner('Fetching data from the database...'):
            gen_set = str([item.id for item in selected]).replace("[", "(").replace("]", ")")
            query = f"""
                            SELECT ((grade-1)/4)*100 as weighted, attr_A, attr_B, attr_C, attr_E, attr_F, attr_H, attr_G, attr_I, attr_L, attr_M, attr_N, attr_O, attr_Q1, attr_Q2,
                                    attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, attr_IN, attr_SC,
                                    cfit,
                                    CASE when course = 'BSCS' then 1 else 0 end as course_bscs,
                                    CASE when course = 'BSIT' then 1 else 0 end as course_bsit
                            FROM students
                            INNER JOIN assessments s on s.student_id = students.Id WHERE tagID in {gen_set};
                            """

            df = fetch_data_as_dataframe(get_engine(), query)
        with st.spinner("Training models..."):
            df = pd.get_dummies(df, columns=['cfit'])
            df.dropna(inplace=True)
            final_y = df['weighted'].apply(discretize_weights)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['weighted']), df['weighted'],
                                                                test_size=0.2)

            # success classifier
            from imblearn.over_sampling import SMOTE
            smote = SMOTE()
            success_y = df['weighted'].apply(discretize_weights).apply(lambda x: x == 2)
            x_s, y_s = smote.fit_resample(df.drop(columns=['weighted']), success_y)
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(x_s, y_s, test_size=0.2)
            final_df = df.copy()
            final_df_features = np.asarray(final_df.drop(columns=['weighted']), np.float64)
            from keras.src.optimizers import Adam
            from sklearn.metrics import accuracy_score

            # Convert data types
            X_train_s = np.asarray(X_train_s).astype(np.float64)
            y_train_s = np.asarray(y_train_s).astype(np.int16)
            X_test_s = np.asarray(X_test_s).astype(np.float64)
            y_test_s = np.asarray(y_test_s).astype(np.int16)

            # Create the model
            model = Sequential([
                InputLayer(shape=(X_train_s.shape[1],)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')  # Single output unit for binary classification
            ])

            # Compile the model
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Fit the model to the training data
            model.fit(X_train_s, y_train_s, epochs=30, batch_size=32, validation_split=0.2, verbose=1, )

            # Make predictions
            y_pred_prob = model.predict(X_test_s)
            y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels
            final_df['success'] = model.predict(final_df_features)

            # Calculate accuracy
            accuracy = accuracy_score(y_test_s, y_pred)
            st.info(f'Finished Ensemble-Member-Model 1 Accuracy (Success Predictor): {accuracy}')
            model.save('model_success.keras')

            # success classifier

            success_y = df['weighted'].apply(discretize_weights).apply(lambda x: x == 0)
            smote = SMOTE()
            x_f, y_f = smote.fit_resample(df.drop(columns=['weighted']), success_y)
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_f, y_f, test_size=0.2)
            from keras.src.optimizers import Adam
            from sklearn.metrics import accuracy_score

            # Convert data types
            X_train_f = np.asarray(X_train_f).astype(np.float64)
            y_train_f = np.asarray(y_train_f).astype(np.int16)
            X_test_f = np.asarray(X_test_f).astype(np.float64)
            y_test_f = np.asarray(y_test_f).astype(np.int16)

            # Create the model
            model = Sequential([
                InputLayer(shape=(X_train_s.shape[1],)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')  # Single output unit for binary classification
            ])

            # Compile the model
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Fit the model to the training data
            model.fit(X_train_f, y_train_f, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

            # Make predictions
            y_pred_prob = model.predict(X_test_f)
            y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels

            # Calculate accuracy
            accuracy = accuracy_score(y_test_f, y_pred)

            model.save('model_failure.keras')
            st.info(f'Finished Ensemble-Member-Model 2 Accuracy (Failure    Predictor): {accuracy}')

            final_df['failure'] = model.predict(final_df_features)
            # success classifier

            success_y = df['weighted'].apply(discretize_weights).apply(lambda x: x == 1)
            smote = SMOTE()
            x_b, y_b = smote.fit_resample(df.drop(columns=['weighted']), success_y)
            X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(x_b, y_b, test_size=0.2)
            from keras.src.optimizers import Adam
            from sklearn.metrics import accuracy_score

            # Convert data types
            X_train_b = np.asarray(X_train_f).astype(np.float64)
            y_train_b = np.asarray(y_train_f).astype(np.int16)
            X_test_b = np.asarray(X_test_f).astype(np.float64)
            y_test_b = np.asarray(y_test_f).astype(np.int16)

            # Create the model
            model = Sequential([
                InputLayer(shape=(X_train_s.shape[1],)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')  # Single output unit for binary classification
            ])

            # Compile the model
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Fit the model to the training data
            model.fit(X_train_f, y_train_f, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

            # Make predictions
            y_pred_prob = model.predict(X_test_f)
            y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class labels

            # Calculate accuracy
            accuracy = accuracy_score(y_test_f, y_pred)

            result = model.predict(final_df_features)
            df_corr = final_df[['success', 'failure']]
            df_corr['pass'] = result
            model.save('model_pass.keras')
            st.info(f'Finished Ensemble-Member-Model 3 Accuracy (Satisfactory Predictor): {accuracy}')

            merged_input_features = pd.concat([df_corr, pd.DataFrame(final_df_features)], axis=1)

            y_one_hot = pd.get_dummies(final_y)
            from sklearn.model_selection import KFold
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import InputLayer, Dense
            from tensorflow.keras.optimizers import Adam
            import numpy as np

            # Assuming final_y and merged_input_features are already defined

            # One-hot encode the target variable

            merged_input_features.columns = merged_input_features.columns.astype(str)
            merged_input_features, final_y = smote.fit_resample(merged_input_features, final_y)
            y_one_hot = pd.get_dummies(final_y)

            # Define the model creation function

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            def create_model(input_shape):
                model = Sequential([
                    InputLayer(shape=(input_shape,)),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(3, activation='softmax')
                ])
                model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'],)
                return model

            n_splits = 10
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Perform k-fold cross-validation
            fold_no = 1
            acc_per_fold = []
            loss_per_fold = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for train, test in kfold.split(merged_input_features, y_one_hot):
                status_text.text(f"Training fold {fold_no}/{n_splits}")
                model = create_model(merged_input_features.shape[1])

                history = model.fit(
                    merged_input_features.iloc[train],
                    y_one_hot.iloc[train],
                    epochs=100,
                    batch_size=15,
                    validation_data=(merged_input_features.iloc[test], y_one_hot.iloc[test]),
                    verbose=0,
                    callbacks=[early_stopping]
                )

                scores = model.evaluate(merged_input_features.iloc[test], y_one_hot.iloc[test], verbose=0)
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                progress_bar.progress(fold_no / n_splits)
                fold_no += 1

            status_text.text("Training complete!")

            st.markdown("## Final Model Statistics")

            # Visualize fold scores
            st.subheader("Performance per Fold")
            fold_df = pd.DataFrame({
                'Fold': range(1, n_splits + 1),
                'Accuracy': acc_per_fold,
                'Loss': loss_per_fold
            })

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            sns.barplot(x='Fold', y='Accuracy', data=fold_df, ax=ax1, palette='viridis')
            ax1.set_title('Accuracy per Fold')
            ax1.set_ylim(0, 100)

            sns.barplot(x='Fold', y='Loss', data=fold_df, ax=ax2, palette='viridis')
            ax2.set_title('Loss per Fold')

            st.pyplot(fig)

            # Display fold scores in a table
            st.subheader("Detailed Fold Scores")
            st.table(fold_df.style.format({'Accuracy': '{:.2f}%', 'Loss': '{:.4f}'}).highlight_max(axis=0))

            # Display average scores
            st.subheader("Average Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Accuracy", f"{np.mean(acc_per_fold):.2f}%", f"±{np.std(acc_per_fold):.2f}%")
            with col2:
                st.metric("Mean Loss", f"{np.mean(loss_per_fold):.4f}")

            # Visualize learning curves
            st.subheader("Learning Curves")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()

            st.pyplot(fig)

            # Train the final model on all data
            st.subheader("Final Model Training")
            final_model = create_model(merged_input_features.shape[1])
            final_model_progress = st.progress(0)
            final_model_status = st.empty()

            class CustomCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    final_model_progress.progress((epoch + 1) / 30)
                    final_model_status.text(f"Training epoch {epoch + 1}/30")

            final_model.fit(
                merged_input_features,
                y_one_hot,
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[CustomCallback()]
            )
            final_model.save('model_ensemble.keras')

            final_model_status.text("Final model training complete!")












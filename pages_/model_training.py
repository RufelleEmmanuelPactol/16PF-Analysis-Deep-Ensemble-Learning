from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from settings.connectivity import get_engine
import joblib
import pages_.yoy_view
from pages_.yoy_view import get_tags
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


def ModelTrainingComponent():
    st.header("Model Training Pipeline 🚀")
    st.markdown("This section is for retraining and training models according to loaded datasets.")
    tags = get_tags()

    selected = st.multiselect(label="Select datasets to include in training", options=tags)
    if st.button("Start Training! 🚀"):
        gen_set = str([item.id for item in selected]).replace("[", "(").replace("]", ")")
        query = f"""
                        SELECT  ((grade/5)*50) + ((c_cert/80)*50) as weighted
                , attr_A, attr_B, attr_C, attr_E, attr_F, attr_H, attr_G, attr_I, attr_L, attr_M, attr_N, attr_O, attr_Q1   , attr_Q2,
                  attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, attr_IN, attr_SC, IF(cfit = 'L', 2,
                           IF(cfit = 'BA', 4,
                              IF(cfit = 'A', 6,
                                 IF(cfit = 'AA', 8,
                                    IF(cfit = 'H', 10, NULL))))) as cfit, CASE when course = 'BSCS' then 1 else 0 end as course
                FROM students
                         INNER JOIN assessments s on s.student_id = students.Id WHERE tagID in {gen_set};
                        """

        df= fetch_data_as_dataframe(get_engine(), query)
        with st.spinner("Training model"):
            train_model(df)


import pandas as pd


def fetch_data_as_dataframe(connection, query: str) -> pd.DataFrame:
    """
    Execute a SQL query using an existing connection and return the result set as a pandas DataFrame.

    :param connection: MySQL connection object
    :param query: SQL query to execute
    :return: pandas DataFrame containing the result set
    """
    # Create a cursor to execute the query
    cursor = connection.cursor()

    # Execute the query
    cursor.execute(query)

    # Fetch all rows from the executed query
    result_set = cursor.fetchall()

    # Get column names from the cursor
    column_names = cursor.column_names

    # Close the cursor
    cursor.close()

    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(result_set, columns=column_names)
    return df.dropna()

from sklearn.metrics import mean_absolute_error
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test  = train_test_split(df.drop(columns=['weighted']), df['weighted'], test_size=0.2)
    model = RandomForestRegressor(n_estimators=2000, max_features=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = mean_absolute_error(y_test, y_pred)
    with open('acc.tmp', 'w') as f:
        f.write(str(accuracy))
    joblib.dump(model, 'model.joblib')
    st.info(f"Model has been trained, and has been saved. The model has an error of {round(accuracy, 2)} units, such that the weighted score is calculated through `(50*(FinalGrade/5)) + (50*(C_CERT/80))`.", icon="ℹ️",)
    st.info("When we discretize the results, we will take into consideration the mean absolute error.", icon="ℹ️")








import pandas as pd
import string

from pandasql import sqldf
import streamlit as st

from settings.connectivity import get_engine


def retrieve_data_from_file(df):
    # assume that df is a file handle

    bsit = pd.read_excel(df, sheet_name='BSIT')
    bscs = pd.read_excel(df, sheet_name='BSCS')

    def format_features(df):
        # Rename '16PF' column to 'attr_A'
        df.rename(columns={'16PF': 'attr_A'}, inplace=True)

        # List of attributes as per the image provided
        attributes = ['B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'Q1', 'Q2', 'Q3', 'Q4', 'EX', 'AX', 'TM',
                      'IN', 'SC']

        # Loop through columns from 'Unnamed: 8' onwards and rename them according to the attributes list
        for i, attr in enumerate(attributes, start=8):
            old_column_name = f'Unnamed: {i}'
            new_column_name = f'attr_{attr}'
            df.rename(columns={old_column_name: new_column_name}, inplace=True)

        # Drop the first row
        df = df.drop(0)

        return df

    bscs = bscs.rename(columns={'ID Number': 'IDNumber'})
    bsit = bsit.rename(columns={'ID Number': 'IDNumber'})
    bscs = bscs[bscs['Previous School'] != None]
    bsit = bsit[bsit['Previous School'] != None]
    return format_features(bscs), format_features(bsit), pd.read_excel('2324.xlsx',
                                                                       sheet_name='BSCS_FinalPerf'), pd.read_excel(
        '2324.xlsx', sheet_name='BSIT_FinalPerf'),


def load_schools(df: pd.DataFrame):
    schools = df['Previous School'].dropna().unique()
    try:
        # Get the MySQL connection
        conn = get_engine()

        if conn.is_connected():
            cursor = conn.cursor()

            # Fetch existing schools from the database
            cursor.execute("SELECT Name FROM Schools")
            existing_schools = set(name[0] for name in cursor.fetchall())

            # Start a transaction
            conn.start_transaction()

            # Loop through the schools and insert if not exists
            new_schools = [school for school in schools if school not in existing_schools]

            school_len = len(new_schools)
            inc = 1
            for school in new_schools:
                st.spinner(f"Loading schools progress: {(inc / school_len)*100}%")
                cursor.execute("INSERT INTO Schools (Name) VALUES (%s)", (school,))
                inc += 1

            # Commit the transaction
            conn.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        # Rollback the transaction in case of error
        if conn.is_connected():
            conn.rollback()

    finally:
        # Close the cursor and connection
        if conn.is_connected():
            cursor.close()
            conn.close()


def load_users(df: pd.DataFrame, tag, batch=1):
    users = df.iterrows()
    engine = get_engine()

    with engine as conn:
        cursor = conn.cursor()

        # Start transaction explicitly
        conn.start_transaction()

        # Retrieve or insert tag
        cursor.execute('SELECT id FROM datasetTag WHERE name = %s', (tag,))
        tag_result = cursor.fetchone()
        if tag_result is None:
            cursor.execute('INSERT INTO datasetTag (name) VALUES (%s)', (tag,))
            conn.commit()  # Commit to get the lastrowid
            tag_id = cursor.lastrowid
        else:
            tag_id = tag_result[0]

        # Cache existing student IDs
        cursor.execute('SELECT Student_ID FROM students')
        existing_students = set(row[0] for row in cursor.fetchall())

        incr = 0
        length = len(df)
        for index, user in users:
            incr += 1

            with st.spinner(f"Student loading progress: {(incr / length) * 100}%. Batch {batch}/2."):
                if user['IDNumber'] not in existing_students:
                    existing_students.add(user['IDNumber'])  # Add to the cache to avoid future checks
                    if user['Previous School'] is None or pd.isna(user['Previous School']):
                        school_id = None
                    # Get school_id
                    else:
                        cursor.execute('SELECT id FROM Schools WHERE Name = %s', (user['Previous School'],))
                        school_result = cursor.fetchone()
                        if school_result is not None:
                            school_id = school_result[0]
                        else:
                            school_id = None

                    # Insert the user
                    cursor.execute(
                        'INSERT INTO students (Student_id, previous_school_id, tagID, course) VALUES (%s, %s, %s, %s)',
                        (user['IDNumber'], school_id, tag_id, user['Course'])
                    )

        # Commit transaction
        conn.commit()

def clean_final_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[pd.notna(df['C Certification'])].copy()
    df = df[df['Final Grades'] != 'INC']
    df['Final Grades'] = df['Final Grades'].copy().astype(float)
    return df


def merge_and_clean(df16pf: pd.DataFrame, df_final: pd.DataFrame) -> pd.DataFrame:
    df = df16pf
    df16pf['Previous School'] = df['Previous School'].str.replace(r'(?i)^University of Cebu.*', 'UNIVERSITY OF CEBU', regex=True)
    df16pf['Previous School'] = df['Previous School'].str.replace(r'(?i)^University of San Carlos.*', 'UNIVERSITY OF '
                                                                                                      'SAN CARLOS',
                                                                  regex=True)
    df_final = clean_final_data(df_final)
    return pd.merge(df16pf, df_final, on='IDNumber', how='inner')



def load_and_merge_codechum(cs_codechum, it_codechum, cs16pf, it16pf):
    cs_codechum = pd.read_csv(cs_codechum)
    it_codechum = pd.read_excel(cs_codechum)
    cs_codechum = cs_codechum.rename(columns={'ID No.': 'IDNumber'})
    it_codechum = it_codechum.rename(columns={'ID No.': 'IDNumber'})
    return pd.merge(it16pf, it_codechum, on='IDNumber', how='inner'), pd.merge(cs16pf, cs_codechum, on='IDNumber',
                                                                               how='inner')


def load_merged_data_to_assessments(merged_df: pd.DataFrame):

    # Get the MySQL connection
    engine = get_engine()

    with engine.cursor() as conn:
        cursor = conn

        # Load existing student IDs into memory
        cursor.execute("SELECT id, student_id FROM students")
        existing_students = {row[1]: row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT s.student_id FROM assessments INNER JOIN railway.students s on assessments.student_id = s.Id;")
        existing_data = cursor.fetchall()

        already_added = set()
        for row in existing_data:
            already_added.add(row[0])

        # Prepare the insert statement
        insert_stmt = """
        INSERT INTO assessments (
            cfit, attr_A, attr_B, attr_C, attr_E, attr_F, attr_G,
            attr_H, attr_I,  attr_L, attr_M, attr_N, attr_O,
            attr_Q1, attr_Q2, attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM,
            attr_IN, attr_SC, c_cert, grade, student_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Start a transaction
        with conn:
            # Iterate over the merged dataframe and insert each row into the assessments table
            for index, row in merged_df.iterrows():
                student_id = existing_students.get(row['IDNumber'])

                if student_id in already_added:
                    continue

                if student_id:
                    cursor.execute(insert_stmt, (
                        row.get('CFIT', None),
                        row.get('attr_A', None),
                        row.get('attr_B', None),
                        row.get('attr_C', None),

                        row.get('attr_E', None),
                        row.get('attr_F', None),
                        row.get('attr_G', None),
                        row.get('attr_H', None),
                        row.get('attr_I', None),

                        row.get('attr_L', None),
                        row.get('attr_M', None),
                        row.get('attr_N', None),
                        row.get('attr_O', None),
                        row.get('attr_Q1', None),
                        row.get('attr_Q2', None),
                        row.get('attr_Q3', None),
                        row.get('attr_Q4', None),
                        row.get('attr_EX', None),
                        row.get('attr_AX', None),
                        row.get('attr_TM', None),
                        row.get('attr_IN', None),
                        row.get('attr_SC', None),
                        row.get('C Certification', None),
                        row.get('Final Grades', None),

                        student_id
                    ))

                    engine.commit()




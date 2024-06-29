import pandas as pd
import streamlit as st
from settings.connectivity import get_engine
import plotly.express as px
import plotly.figure_factory as ff


@st.cache_data
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
        st.error(f"Error while fetching tags: {e}")
        return []
    finally:
        if connection.is_connected():
            connection.close()


TAG_ID = None


class TagHolder:

    def __init__(self, _id, name):
        self.id = _id
        self.name = name

    def __str__(self):
        return self.name


def get_stat():
    STAT_QUERY = f"""
    WITH pass AS (
        SELECT
            COUNT(*) AS cnt,
            AVG(grade) AS avg_pass_grade
        FROM
            students
        INNER JOIN
            assessments a
        ON
            students.Id = a.student_id
        WHERE
            tagID = {TAG_ID}
            AND grade >= 3
    ),
    course_stats AS (
        SELECT
            course,
            COUNT(*) AS total,
            SUM(CASE WHEN grade >= 3 THEN 1 ELSE 0 END) AS pass_count,
            AVG(CASE WHEN grade >= 3 THEN c_cert END) AS avg_ccert,
            AVG(CASE WHEN grade >= 3 THEN grade END) AS avg_grade
        FROM
            students
        INNER JOIN
            assessments a
        ON
            students.Id = a.student_id
        WHERE
            tagID = {TAG_ID}
        GROUP BY
            course
    ),
    total_stats AS (
        SELECT
            COUNT(*) AS total,
            AVG(CASE WHEN grade >= 3 THEN c_cert END) AS avg_ccert
        FROM
            students
        INNER JOIN
            assessments a
        ON
            students.Id = a.student_id
        WHERE
            tagID = {TAG_ID}
    )
    SELECT
        ts.total,
        ROUND(ts.avg_ccert, 2) AS avg_ccert,
        ROUND((pass.cnt / ts.total) * 100, 2) AS pass_rate,
        ROUND(pass.avg_pass_grade, 2) AS avg_pass_grade,
        ROUND(cs_bscs.avg_ccert, 2) AS avg_ccert_bscs,
        ROUND(cs_bsits.avg_ccert, 2) AS avg_ccert_bsit,
        ROUND(cs_bscs.avg_grade, 2) AS avg_grade_bscs,
        ROUND(cs_bsits.avg_grade, 2) AS avg_grade_bsit,
        ROUND((cs_bscs.pass_count / cs_bscs.total) * 100, 2) AS pass_rate_bscs,
        ROUND((cs_bsits.pass_count / cs_bsits.total) * 100, 2) AS pass_rate_bsit
    FROM
        total_stats ts,
        pass,
        (SELECT * FROM course_stats WHERE course = 'BSCS') AS cs_bscs,
        (SELECT * FROM course_stats WHERE course = 'BSIT') AS cs_bsits;
    """

    conn = get_engine()
    cursor = conn.cursor()

    cursor.execute(STAT_QUERY)
    result = cursor.fetchone()

    columns = [
        "Total Students",
        "Average CCert",
        "Pass Rate (%)",
        "Average Passing Grade",
        "Average CCert (BSCS)",
        "Average CCert (BSIT)",
        "Average Grade (BSCS)",
        "Average Grade (BSIT)",
        "Pass Rate (BSCS) (%)",
        "Pass Rate (BSIT) (%)"
    ]

    # Create a DataFrame from the fetched result
    df = pd.DataFrame(result, index=columns, columns=["Value"])

    # Reset the index to remove it and make the labels a column

    # Rename the columns for clarity
    df.columns = ["Value"]

    return df.to_markdown()


class YearOnYearComponent:

    def __init__(self):
        self.display()

    def display(self):
        st.subheader("Year-on-Year Dashboard")
        st.markdown("View statistics, visualizations, and other statistics on a year-tag basis.")
        with st.spinner("Retrieving school tags..."):
            tags = get_tags()
            value = st.selectbox(label="Select SCHOOL-TAG (School Year) to introspect", options=tags)
            global TAG_ID
            TAG_ID = value.id
        clicked = st.button("View statistics")
        st.divider()
        if clicked:
            col1, col2 = st.columns(2)
            with col1:
                col1.markdown("#### General Statistics")
                with st.spinner("Loading general statistics..."):
                    col1.markdown(get_stat())
            with col2:
                col2.markdown("#### School Summary")
                with st.spinner("School Performance 1"):
                    best_school()
                st.divider()
                with st.spinner("School Performance 2"):
                    worst_school()

            st.divider()
            st.markdown("#### Correlation Analysis")
            with st.spinner("Loading correlation analysis..."):
                show_corr()

            with st.spinner("Loading best student statistics..."):
                st.markdown("#### Weighted Student Rankings (C-CERT + Prog2)")
                best_students()

            with st.spinner("Loading Correlation Analysis for Best and Worst 30 students..."):
                correlation_best_worst_students()


def best_school():
    query = f"""
    WITH school_stats AS (
        SELECT
            s.Name,
            COUNT(*) as student_count,
            AVG(CASE WHEN a.grade >= 3 THEN a.grade END) as avg_passing_grade,
            SUM(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) / COUNT(*) * 100 as passing_rate,
            AVG(a.c_cert) as avg_ccert_grade
        FROM
            students
        INNER JOIN
            Schools s
        ON
            students.previous_school_id = s.Id
        INNER JOIN
            assessments a
        ON
            students.Id = a.student_id
        WHERE students.tagID = {TAG_ID}
        GROUP BY
            s.Id
        HAVING
            student_count > 3
    )
    SELECT
        'Best' as Performance,
        Name,
        student_count,
        ROUND(avg_passing_grade, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade, 2) as avg_ccert_grade
    FROM
        school_stats
    ORDER BY
        avg_passing_grade DESC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    # Define the columns
    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate", "Average CCERT Grade"]

    # Create a DataFrame from the fetched result
    df = pd.DataFrame([result], columns=columns)

    # Transpose the DataFrame for vertical display
    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    # Convert the DataFrame to a Markdown table
    markdown_table = df.to_markdown(index=False)

    # Display the table using Streamlit
    st.markdown(markdown_table)


def worst_school():
    query = f"""
    WITH school_stats AS (
        SELECT
            s.Name,
            COUNT(*) as student_count,
            AVG(CASE WHEN a.grade >= 3 THEN a.grade END) as avg_passing_grade,
            SUM(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) / COUNT(*) * 100 as passing_rate,
            AVG(a.c_cert) as avg_ccert_grade
        FROM
            students
        INNER JOIN
            Schools s
        ON
            students.previous_school_id = s.Id
        INNER JOIN
            assessments a
        ON
            students.Id = a.student_id
        WHERE students.tagID = {TAG_ID}
        GROUP BY
            s.Id
        HAVING
            student_count > 3
    )
    SELECT
        'Worst' as Performance,
        Name,
        student_count,
        ROUND(avg_passing_grade, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade, 2) as avg_ccert_grade
    FROM
        school_stats
    ORDER BY
        avg_passing_grade ASC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    # Define the columns
    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate", "Average CCERT Grade"]

    # Create a DataFrame from the fetched result
    df = pd.DataFrame([result], columns=columns)

    # Transpose the DataFrame for vertical display
    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    # Convert the DataFrame to a Markdown table
    markdown_table = df.to_markdown(index=False)

    # Display the table using Streamlit
    st.markdown(markdown_table)


def show_corr():
    query = f"""
    SELECT 
        attr_A, attr_B, attr_C, attr_E, attr_F, attr_G, attr_H, attr_I, attr_L, attr_M, 
        attr_N, attr_O, attr_Q1, attr_Q2, attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, 
        attr_IN, attr_SC, c_cert, grade
    FROM 
        assessments 
    LEFT JOIN students s on s.id = assessments.student_id 
    WHERE tagID = {TAG_ID}
    """

    conn = get_engine()
    cursor = conn.cursor()

    cursor.execute(query)
    result = cursor.fetchall()

    # Get column names
    columns = [desc[0] for desc in cursor.description]

    # Create DataFrame
    df = pd.DataFrame(result, columns=columns)

    # Calculate correlations with the labels
    corr_matrix = df.corr()

    # Extract correlations with `c_cert` and `grade`
    corr_with_c_cert = corr_matrix['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
    corr_with_grade = corr_matrix['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

    # Create bar graphs using Plotly
    fig_c_cert = px.bar(
        x=corr_with_c_cert.index,
        y=corr_with_c_cert.values,
        title='Correlation with CCert Grades',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    fig_grade = px.bar(
        x=corr_with_grade.index,
        y=corr_with_grade.values,
        title='Correlation with Grade',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    # Display the bar graphs in Streamlit
    st.plotly_chart(fig_c_cert)
    st.plotly_chart(fig_grade)

    cursor.close()
    conn.close()


def best_students():
    query = """
    SELECT
        s.student_id,
        s.course,
        sch.Name as school,
        a.c_cert,
        a.grade,
        ROUND((0.5 * (a.grade / 5) + 0.5 * (a.c_cert / 80)), 2) as weighted_score
    FROM
        students s
    INNER JOIN
        assessments a
    ON
        s.Id = a.student_id
    INNER JOIN
        Schools sch
    ON
        s.previous_school_id = sch.Id
    ORDER BY
        weighted_score DESC
    LIMIT 10;
    """

    conn = get_engine()
    cursor = conn.cursor()


    cursor.execute(query)
    result = cursor.fetchall()

    # Define the columns
    columns = ["StudentID", "Course", "School", "CCert", "Final Grade", "Weighted Score"]

    # Create a DataFrame from the fetched result
    df = pd.DataFrame(result, columns=columns)

    # Convert the DataFrame to a Markdown table
    markdown_table = df.to_markdown(index=False)

    # Display the table using Streamlit
    st.markdown(markdown_table)


def correlation_best_worst_students():
    query_best = f"""
    SELECT 
        attr_A, attr_B, attr_C, attr_E, attr_F, attr_G, attr_H, attr_I, attr_L, attr_M, 
        attr_N, attr_O, attr_Q1, attr_Q2, attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, 
        attr_IN, attr_SC, c_cert, grade
    FROM 
        assessments 
    LEFT JOIN students s on s.id = assessments.student_id 
    WHERE tagID = {TAG_ID}
    ORDER BY 
        ROUND((0.5 * (grade / 5) + 0.5 * (c_cert / 80)), 2) DESC
    LIMIT 30
    """

    query_worst = f"""
    SELECT 
        attr_A, attr_B, attr_C, attr_E, attr_F, attr_G, attr_H, attr_I, attr_L, attr_M, 
        attr_N, attr_O, attr_Q1, attr_Q2, attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, 
        attr_IN, attr_SC, c_cert, grade
    FROM 
        assessments 
    LEFT JOIN students s on s.id = assessments.student_id 
    WHERE tagID = {TAG_ID}
    ORDER BY 
        ROUND((0.5 * (grade / 5) + 0.5 * (c_cert / 80)), 2) ASC
    LIMIT 30
    """

    conn = get_engine()
    cursor = conn.cursor()

    cursor.execute(query_best)
    result_best = cursor.fetchall()

    cursor.execute(query_worst)
    result_worst = cursor.fetchall()

    # Get column names
    columns = [desc[0] for desc in cursor.description]

    # Create DataFrames
    df_best = pd.DataFrame(result_best, columns=columns)
    df_worst = pd.DataFrame(result_worst, columns=columns)

    # Calculate correlations with the labels for best and worst students
    corr_matrix_best = df_best.corr()
    corr_matrix_worst = df_worst.corr()

    # Extract correlations with `c_cert` and `grade`
    corr_with_c_cert_best = corr_matrix_best['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
    corr_with_grade_best = corr_matrix_best['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

    corr_with_c_cert_worst = corr_matrix_worst['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
    corr_with_grade_worst = corr_matrix_worst['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

    # Create bar graphs using Plotly
    fig_c_cert_best = px.bar(
        x=corr_with_c_cert_best.index,
        y=corr_with_c_cert_best.values,
        title='Correlation with CCert (Best Students)',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    fig_grade_best = px.bar(
        x=corr_with_grade_best.index,
        y=corr_with_grade_best.values,
        title='Correlation with Grade (Best Students)',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    fig_c_cert_worst = px.bar(
        x=corr_with_c_cert_worst.index,
        y=corr_with_c_cert_worst.values,
        title='Correlation with CCert (Worst Students)',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    fig_grade_worst = px.bar(
        x=corr_with_grade_worst.index,
        y=corr_with_grade_worst.values,
        title='Correlation with Grade (Worst Students)',
        labels={'x': 'Features', 'y': 'Correlation'},
        text_auto=True
    )

    # Display the bar graphs in Streamlit
    st.plotly_chart(fig_c_cert_best)
    st.plotly_chart(fig_grade_best)
    st.plotly_chart(fig_c_cert_worst)
    st.plotly_chart(fig_grade_worst)



    cursor.close()
    conn.close()



    z
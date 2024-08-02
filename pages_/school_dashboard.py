import pandas as pd
import streamlit as st
from settings.connectivity import get_engine
import plotly.express as px

CURRENT_ID = None


def get_summary_query():
    return f"""
    SELECT
    COUNT(*) AS 'Total Count',
    AVG(CASE WHEN a.cfit = 'L' THEN 2 WHEN a.cfit = 'BA' THEN 4 WHEN a.cfit = 'A' THEN 6 WHEN a.cfit = 'AA' THEN 8 WHEN a.cfit = 'H' THEN 10 ELSE NULL END) AS `CFIT`,
    AVG(a.attr_A) AS `A`,
    AVG(a.attr_B) AS `B`,
    AVG(a.attr_C) AS `C`,
    AVG(a.attr_E) AS `E`,
    AVG(a.attr_F) AS `F`,
    AVG(a.attr_G) AS `G`,
    AVG(a.attr_H) AS `H`,
    AVG(a.attr_I) AS `I`,
    AVG(a.attr_L) AS `L`,
    AVG(a.attr_M) AS `M`,
    AVG(a.attr_N) AS `N`,
    AVG(a.attr_O) AS `O`,
    AVG(a.attr_Q1) AS `Q1`,
    AVG(a.attr_Q2) AS `Q2`,
    AVG(a.attr_Q3) AS `Q3`,
    AVG(a.attr_Q4) AS `Q4`,
    AVG(a.attr_EX) AS `EX`,
    AVG(a.attr_AX) AS `AX`,
    AVG(a.attr_TM) AS `TM`,
    AVG(a.attr_IN) AS `IN`,
    AVG(a.attr_SC) AS `SC`,
    AVG(a.c_cert) as `C Cert`,
    AVG(CASE WHEN a.grade >= 3 THEN a.grade ELSE NULL END) as `Average Passing Grade`,
    AVG(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) * 100 AS `Passing Rate`,
    SUM(CASE WHEN s.course = 'BSCS' THEN 1 ELSE 0 END) AS `BSCS Count`,
    SUM(CASE WHEN s.course = 'BSIT' THEN 1 ELSE 0 END) AS `BSIT Count`
FROM assessments a
INNER JOIN students s ON a.student_id = s.Id
INNER JOIN Schools ss ON s.previous_school_id = ss.Id
INNER JOIN datasetTag dt ON dt.id = s.tagID
WHERE ss.id = {CURRENT_ID} AND a.cfit IN ('L', 'BA', 'H', 'A', 'AA')
    """


def get_overall_averages_query():
    return """
    SELECT
    AVG(COUNT(*)) OVER () AS 'Total Count',
    AVG(AVG(CASE WHEN a.cfit = 'L' THEN 2 WHEN a.cfit = 'BA' THEN 4 WHEN a.cfit = 'A' THEN 6 WHEN a.cfit = 'AA' THEN 8 WHEN a.cfit = 'H' THEN 10 ELSE NULL END)) OVER () AS `CFIT`,
    AVG(AVG(a.attr_A)) OVER () AS `A`,
    AVG(AVG(a.attr_B)) OVER () AS `B`,
    AVG(AVG(a.attr_C)) OVER () AS `C`,
    AVG(AVG(a.attr_E)) OVER () AS `E`,
    AVG(AVG(a.attr_F)) OVER () AS `F`,
    AVG(AVG(a.attr_G)) OVER () AS `G`,
    AVG(AVG(a.attr_H)) OVER () AS `H`,
    AVG(AVG(a.attr_I)) OVER () AS `I`,
    AVG(AVG(a.attr_L)) OVER () AS `L`,
    AVG(AVG(a.attr_M)) OVER () AS `M`,
    AVG(AVG(a.attr_N)) OVER () AS `N`,
    AVG(AVG(a.attr_O)) OVER () AS `O`,
    AVG(AVG(a.attr_Q1)) OVER () AS `Q1`,
    AVG(AVG(a.attr_Q2)) OVER () AS `Q2`,
    AVG(AVG(a.attr_Q3)) OVER () AS `Q3`,
    AVG(AVG(a.attr_Q4)) OVER () AS `Q4`,
    AVG(AVG(a.attr_EX)) OVER () AS `EX`,
    AVG(AVG(a.attr_AX)) OVER () AS `AX`,
    AVG(AVG(a.attr_TM)) OVER () AS `TM`,
    AVG(AVG(a.attr_IN)) OVER () AS `IN`,
    AVG(AVG(a.attr_SC)) OVER () AS `SC`,
    AVG(AVG(a.c_cert)) OVER () as `C Cert`,
    AVG(AVG(CASE WHEN a.grade >= 3 THEN a.grade ELSE NULL END)) OVER () as `Average Passing Grade`,
    AVG(AVG(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) * 100) OVER () AS `Passing Rate`,
    AVG(SUM(CASE WHEN s.course = 'BSCS' THEN 1 ELSE 0 END)) OVER () AS `BSCS Count`,
    AVG(SUM(CASE WHEN s.course = 'BSIT' THEN 1 ELSE 0 END)) OVER () AS `BSIT Count`
FROM assessments a
INNER JOIN students s ON a.student_id = s.Id
INNER JOIN Schools ss ON s.previous_school_id = ss.Id
INNER JOIN datasetTag dt ON dt.id = s.tagID
WHERE a.cfit IN ('L', 'BA', 'H', 'A', 'AA')
    """


class SchoolHolder:
    def __init__(self, name, s_id):
        self.name = name
        self.id = s_id

    def __str__(self):
        return self.name



@st.cache_data
def get_schools():
    QUERY = """
    SELECT id, name from Schools
    """
    conn = get_engine()
    cur = conn.cursor()
    cur.execute(QUERY)
    schools = cur.fetchall()
    l_schools = [SchoolHolder("", None)]
    for school in schools:
        l_schools.append(SchoolHolder(school[1], school[0]))

    return l_schools


def get_stats():
    conn = get_engine()
    cursor = conn.cursor()

    # Fetch school-specific data
    cursor.execute(get_summary_query())
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=columns)

    # Fetch overall averages
    cursor.execute(get_overall_averages_query())
    overall_results = cursor.fetchall()
    overall_columns = [desc[0] for desc in cursor.description]
    overall_df = pd.DataFrame(overall_results, columns=overall_columns)
    conn.close()

    # Calculate percent differences and add overall averages to the DataFrame
    for column in df.columns:
        df[f'{column} - Average'] = overall_df[column][0]
        df[f'{column} - % Difference'] = (df[column] - overall_df[column][0]) / overall_df[column][0] * 100

    # Prepare basic stats DataFrames
    basic_stat = df[['Total Count', 'BSCS Count', 'BSIT Count']]
    basic_stat_2 = df[['C Cert', 'Average Passing Grade', 'Passing Rate']]

    # Add overall averages and percent differences to basic_stat_2
    basic_stat_2_extended = basic_stat_2.transpose().reset_index()
    basic_stat_2_extended.columns = ['Value', 'School Value']
    basic_stat_2_extended['Average Value'] = basic_stat_2_extended['Value'].map(lambda x: overall_df[x].values[0])
    basic_stat_2_extended['% Difference'] = (basic_stat_2_extended['School Value'] - basic_stat_2_extended[
        'Average Value']) / basic_stat_2_extended['Average Value'] * 100

    # Display the DataFrames




    return df, overall_df


def SchoolDashboardComponent():


    st.title("🏫 School Performance Dashboard")
    st.markdown("Analyze and compare school-based performance metrics.")

    with st.spinner("Loading schools..."):
        schools = get_schools()

    selected = st.selectbox("Select a school to analyze:", options=schools, format_func=lambda x: x.name)

    if selected.name:
        global CURRENT_ID
        CURRENT_ID = selected.id
        with st.spinner("Loading school information..."):
            display_school_stats(selected.name)


def display_school_stats(school_name):
    st.header(f"📊 Statistics for {school_name}")

    # Get the data
    df, overall_df = get_stats()

    # Display school rank
    rank = get_school_rank()
    st.metric("School Rank", value=int(rank) if isinstance(rank, (int, float)) else "N/A")

    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", int(df['Total Count'].values[0]))
    with col2:
        st.metric("BSCS Students", int(df['BSCS Count'].values[0]))
    with col3:
        st.metric("BSIT Students", int(df['BSIT Count'].values[0]))

    # Key performance indicators
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(3)
    kpis = ['C Cert', 'Average Passing Grade', 'Passing Rate']
    for i, kpi in enumerate(kpis):
        school_value = df[kpi].values[0]
        avg_value = overall_df[kpi].values[0]

        if school_value is not None and avg_value is not None and avg_value != 0:
            diff = ((school_value - avg_value) / avg_value) * 100
            kpi_cols[i].metric(
                kpi,
                f"{school_value:.2f}",
                f"{diff:+.2f}% vs Average",
                delta_color="normal" if diff >= 0 else "inverse"
            )
        else:
            if school_value is not None:
                kpi_cols[i].metric(kpi, f"{school_value:.2f}", "No comparison available")
            else:
                kpi_cols[i].metric(kpi, "N/A", "No data available")

    # Attribute comparison chart
    st.subheader("Attribute Comparison")
    attr_df = df.drop(
        columns=['Total Count', 'BSCS Count', 'BSIT Count', 'C Cert', 'Average Passing Grade', 'Passing Rate'])
    attr_df_melted = pd.melt(attr_df.reset_index(), id_vars=['index'], var_name='Attribute', value_name='Value')
    attr_df_melted = attr_df_melted[~attr_df_melted['Attribute'].str.contains('Average|Difference')]

    # Remove rows with NaN values
    attr_df_melted = attr_df_melted.dropna()

    if not attr_df_melted.empty:
        fig = px.bar(attr_df_melted, x='Attribute', y='Value', title='School vs Overall Average')

        # Create a list of overall average values
        overall_avg_values = [overall_df[attr].values[0] if attr in overall_df.columns else np.nan for attr in
                              attr_df_melted['Attribute']]

        fig.add_scatter(
            x=attr_df_melted['Attribute'],
            y=overall_avg_values,
            mode='lines',
            name='Overall Average'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No attribute data available for comparison.")

    # Detailed statistics in expander
    with st.expander("View Detailed Statistics"):
        st.dataframe(df.style.highlight_max(axis=0))




def get_school_rank():
    query = f"""
    WITH school_stats AS (
        SELECT
            s.Id,
            s.Name,
            COUNT(*) as student_count,
            AVG(CASE WHEN a.grade >= 3 THEN a.grade END) / 5 as avg_passing_grade,
            SUM(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) / COUNT(*) * 100 as passing_rate,
            AVG(a.c_cert) / 80 as avg_ccert_grade,
            (0.5 * (AVG(CASE WHEN a.grade >= 3 THEN a.grade END) / 5) +
             0.3 * (SUM(CASE WHEN a.grade >= 3 THEN 1 ELSE 0 END) / COUNT(*) * 100) / 100 +
             0.2 * (AVG(a.c_cert) / 80)) * 100 as weighted_score
        FROM
            students
        INNER JOIN
            Schools s ON students.previous_school_id = s.Id
        INNER JOIN
            assessments a ON students.Id = a.student_id
        GROUP BY
            s.Id
        HAVING
            student_count > 3
    )
    SELECT
        Id,
        Name,
        student_count,
        ROUND(avg_passing_grade * 100, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade * 100, 2) as avg_ccert_grade,
        ROUND(weighted_score, 2) as weighted_score
    FROM
        school_stats
    ORDER BY
        weighted_score DESC
    """
    engine = get_engine()
    df = pd.read_sql(query, engine)

    if CURRENT_ID not in df['Id'].values:
        return "School does not meet requirements to rank."

    df['Rank'] = df['weighted_score'].rank(method='dense', ascending=False)
    rank = df.loc[df['Id'] == CURRENT_ID, 'Rank'].values[0]

    return rank
import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from settings.connectivity import get_engine
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import streamlit as st
from settings.connectivity import get_engine
import plotly.express as px
import plotly.graph_objects as go



@st.cache_data
def display_stats():
    bscs_df, bsit_df = get_stat()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BSCS Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bscs_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bscs_df.Metric, bscs_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("BSIT Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bsit_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bsit_df.Metric, bsit_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

def best_school():
    query = f"""
    WITH school_stats AS (
        SELECT
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
        ROUND(avg_passing_grade * 100, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade * 100, 2) as avg_ccert_grade,
        ROUND(weighted_score, 2) as weighted_score
    FROM
        school_stats
    ORDER BY
        weighted_score DESC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate (%)",
               "Average CCERT Grade (%)", "Weighted Score (%)"]

    df = pd.DataFrame([result], columns=columns)
    df['Average Passing Grade'] = df["Average Passing Grade"].apply(lambda x: (x * 5) / 100)

    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Metric, df.Value],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def worst_school():
    query = f"""
    WITH school_stats AS (
        SELECT
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
        ROUND(avg_passing_grade * 100, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade * 100, 2) as avg_ccert_grade,
        ROUND(weighted_score, 2) as weighted_score
    FROM
        school_stats
    ORDER BY
        weighted_score ASC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate (%)",
               "Average CCERT Grade (%)", "Weighted Score (%)"]

    df = pd.DataFrame([result], columns=columns)
    df['Average Passing Grade'] = df["Average Passing Grade"].apply(lambda x: (x * 5) / 100)

    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Metric, df.Value],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

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

    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(result, columns=columns)

    corr_matrix = df.corr()

    fig_heatmap = px.imshow(corr_matrix,
                            title='Correlation Heatmap',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    corr_with_c_cert = corr_matrix['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
    corr_with_grade = corr_matrix['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

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

    st.plotly_chart(fig_c_cert, use_container_width=True)


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

    columns = ["StudentID", "Course", "School", "CCert", "Final Grade", "Weighted Score"]
    df = pd.DataFrame(result, columns=columns)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
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
    )
    SELECT
        pass.cnt,
        pass.avg_pass_grade,
        cs_bscs.total,
        cs_bscs.avg_ccert,
        cs_bscs.avg_grade,
        cs_bscs.pass_count,
        cs_bsits.total,
        cs_bsits.avg_ccert,
        cs_bsits.avg_grade,
        cs_bsits.pass_count
    FROM
        pass,
        (SELECT * FROM course_stats WHERE course = 'BSCS') AS cs_bscs,
        (SELECT * FROM course_stats WHERE course = 'BSIT') AS cs_bsits;
    """

    conn = get_engine()
    cursor = conn.cursor()

    cursor.execute(STAT_QUERY)
    result = cursor.fetchone()

    # Split the results for BSCS and BSIT
    total_pass = result[0]
    avg_pass_grade = result[1]
    total_bscs = result[2]
    avg_ccert_bscs = result[3]
    avg_grade_bscs = result[4]
    pass_count_bscs = result[5]
    total_bsit = result[6]
    avg_ccert_bsit = result[7]
    avg_grade_bsit = result[8]
    pass_count_bsit = result[9]

    # Create DataFrames for BSCS and BSIT
    bscs_data = {
        "Metric": ["Total Students", "Average CCert", "Pass Rate (%)", "Average Passing Grade", "Average Passing Grade (BSCS)", "Pass Rate (BSCS) (%)"],
        "Value": [total_bscs, avg_ccert_bscs, (pass_count_bscs / total_bscs) * 100, avg_pass_grade, avg_grade_bscs, (pass_count_bscs / total_bscs) * 100]
    }

    bsit_data = {
        "Metric": ["Total Students", "Average CCert", "Pass Rate (%)", "Average Passing Grade", "Average Passing Grade (BSIT)", "Pass Rate (BSIT) (%)"],
        "Value": [total_bsit, avg_ccert_bsit, (pass_count_bsit / total_bsit) * 100, avg_pass_grade, avg_grade_bsit, (pass_count_bsit / total_bsit) * 100]
    }

    bscs_df = pd.DataFrame(bscs_data)
    bsit_df = pd.DataFrame(bsit_data)

    return bscs_df, bsit_df

# Streamlit display function
def display_stats():
    bscs_df, bsit_df = get_stat()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BSCS Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bscs_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bscs_df.Metric, bscs_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("BSIT Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bsit_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bsit_df.Metric, bsit_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


class YearOnYearComponent:

    def __init__(self):
        self.display()

    def display(self):
        st.title("🎓 Year-on-Year Dashboard")
        st.markdown("Explore statistics, visualizations, and insights on a year-tag basis.")

        with st.spinner("Retrieving school tags..."):
            tags = get_tags()
            tags.insert(0, "")

        col1, col2 = st.columns([3, 1])
        with col1:
            value = st.selectbox(
                label="Select SCHOOL-TAG (School Year) to analyze",
                options=tags,
                format_func=lambda x: x.name if x else "Select a tag"
            )
        show_analysis = False
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📊 Analyze", use_container_width=True):
                if not value:
                    st.warning("Please select a tag to analyze.")
                    return
                show_analysis = True
                global TAG_ID
                TAG_ID = value.id
        if show_analysis:
            self.show_analysis()

    def show_analysis(self):

        sections = [
            "General Statistics",
            "School Summary",
            "Rankings",
            "Raw Correlation Analysis",
            "Derived Correlation Analysis"
        ]

        tabs = st.tabs(sections)

        with tabs[0]:
            self.show_general_statistics()

        with tabs[1]:
            self.show_school_summary()

        with tabs[2]:
            self.show_rankings()

        with tabs[3]:
            self.show_raw_correlation()

        with tabs[4]:
            self.show_derived_correlation()

    def show_general_statistics(self):
        st.header("📈 General Statistics")
        with st.spinner("Loading general statistics..."):
            display_stats()

    def show_school_summary(self):
        st.header("🏫 School Summary")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Loading best school performance..."):
                st.subheader("Top Performing School")
                best_school()
        with col2:
            with st.spinner("Loading worst school performance..."):
                st.subheader("School Needing Improvement")
                worst_school()

    def show_rankings(self):
        st.header("🏆 Rankings")
        with st.spinner("Loading rankings..."):
            st.subheader("Weighted Student Rankings (C-CERT + Prog2)")
            best_students()

            st.divider()

            st.subheader("TOP 10 Best Performing Schools")
            best_schools()

    def show_raw_correlation(self):
        st.header("🔗 Raw Correlation Analysis")
        with st.spinner("Loading correlation analysis..."):
            show_corr()
            with st.expander("About Correlation"):
                st.info(
                    "Pearson's r measures the linear relationship between two variables. "
                    "Here, we analyze correlations between 16PF variables and student performance "
                    "(final grades and CCERT grades)."
                )

            st.divider()

            with st.spinner("Analyzing top and bottom 30 students..."):
                correlation_best_worst_students()

    def show_derived_correlation(self):
        st.header("🧠 Derived Correlation Analysis")
        with st.spinner("Analyzing 16PF factors..."):
            show_common_16pf()
def display_stats():
    bscs_df, bsit_df = get_stat()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BSCS Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bscs_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bscs_df.Metric, bscs_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("BSIT Statistics")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(bsit_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[bsit_df.Metric, bsit_df.Value],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

def best_school():
    query = f"""
    WITH school_stats AS (
        SELECT
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
        ROUND(avg_passing_grade * 100, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade * 100, 2) as avg_ccert_grade,
        ROUND(weighted_score, 2) as weighted_score
    FROM
        school_stats
    ORDER BY
        weighted_score DESC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate (%)",
               "Average CCERT Grade (%)", "Weighted Score (%)"]

    df = pd.DataFrame([result], columns=columns)
    df['Average Passing Grade'] = df["Average Passing Grade"].apply(lambda x: (x * 5) / 100)

    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Metric, df.Value],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def worst_school():
    query = f"""
    WITH school_stats AS (
        SELECT
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
        ROUND(avg_passing_grade * 100, 2) as avg_passing_grade,
        ROUND(passing_rate, 2) as passing_rate,
        ROUND(avg_ccert_grade * 100, 2) as avg_ccert_grade,
        ROUND(weighted_score, 2) as weighted_score
    FROM
        school_stats
    ORDER BY
        weighted_score ASC
    LIMIT 1;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    columns = ["Performance", "Name", "Student Count", "Average Passing Grade", "Passing Rate (%)",
               "Average CCERT Grade (%)", "Weighted Score (%)"]

    df = pd.DataFrame([result], columns=columns)
    df['Average Passing Grade'] = df["Average Passing Grade"].apply(lambda x: (x * 5) / 100)

    df = df.T.reset_index()
    df.columns = ["Metric", "Value"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Metric, df.Value],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

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

    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(result, columns=columns)

    corr_matrix = df.corr()

    fig_heatmap = px.imshow(corr_matrix,
                            title='Correlation Heatmap',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    corr_with_c_cert = corr_matrix['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
    corr_with_grade = corr_matrix['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

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

    st.plotly_chart(fig_c_cert, use_container_width=True)
    st.plotly_chart(fig_grade, use_container_width=True)

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

    columns = ["StudentID", "Course", "School", "CCert", "Final Grade", "Weighted Score"]
    df = pd.DataFrame(result, columns=columns)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
def correlation_best_worst_students():
    query_template = """
    SELECT 
        attr_A, attr_B, attr_C, attr_E, attr_F, attr_G, attr_H, attr_I, attr_L, attr_M, 
        attr_N, attr_O, attr_Q1, attr_Q2, attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, 
        attr_IN, attr_SC, c_cert, grade
    FROM 
        assessments 
    LEFT JOIN students s on s.id = assessments.student_id 
    WHERE tagID = {TAG_ID}
    ORDER BY 
        ROUND((0.5 * (grade / 5) + 0.5 * (c_cert / 80)), 2) {order}
    LIMIT 30
    """

    conn = get_engine()
    cursor = conn.cursor()

    # Fetch data for best and worst students
    cursor.execute(query_template.format(TAG_ID=TAG_ID, order="DESC"))
    result_best = cursor.fetchall()
    cursor.execute(query_template.format(TAG_ID=TAG_ID, order="ASC"))
    result_worst = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    df_best = pd.DataFrame(result_best, columns=columns)
    df_worst = pd.DataFrame(result_worst, columns=columns)

    cursor.close()
    conn.close()

    # Function to create correlation plots
    def create_correlation_plot(df, title):
        corr_matrix = df.corr()
        corr_with_c_cert = corr_matrix['c_cert'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)
        corr_with_grade = corr_matrix['grade'].drop(['c_cert', 'grade']).sort_values(ascending=False, key=abs)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlation with CCert", "Correlation with Grade"))

        fig.add_trace(
            go.Bar(x=corr_with_c_cert.index, y=corr_with_c_cert.values, name="CCert"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=corr_with_grade.index, y=corr_with_grade.values, name="Grade"),
            row=1, col=2
        )

        fig.update_layout(
            title_text=title,
            height=600,
            showlegend=False,
            title_x=0.5
        )
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig.update_yaxes(range=[-1, 1])

        return fig

    st.subheader("Correlation Analysis for Top and Bottom Performers")

    tab1, tab2 = st.tabs(["Top 30 Students", "Bottom 30 Students"])

    with tab1:
        fig_best = create_correlation_plot(df_best, "Correlation Analysis for Top 30 Students")
        st.plotly_chart(fig_best, use_container_width=True)

    with tab2:
        fig_worst = create_correlation_plot(df_worst, "Correlation Analysis for Bottom 30 Students")
        st.plotly_chart(fig_worst, use_container_width=True)

    st.markdown("""
    ### Interpretation Guide
    - Bars represent the strength and direction of correlation between each factor and student performance (CCert and Grade).
    - Positive values indicate a positive correlation, while negative values indicate an inverse relationship.
    - Longer bars represent stronger correlations.
    - Compare the patterns between top and bottom performers to identify key differences in influential factors.
    """)


def best_schools():
    query = f"""
    WITH school_stats AS (
        SELECT
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
    LIMIT 10;
    """
    conn = get_engine()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    # Define the columns
    columns = ["School", "Student Count", "Average Passing Grade", "Passing Rate (%)", "Average CCERT Grade (%)",
               "Weighted Score (%)"]

    # Create a DataFrame from the fetched result
    df = pd.DataFrame(result, columns=columns)

    # Apply the transformation to Average Passing Grade
    df['Average Passing Grade'] = df["Average Passing Grade"].apply(lambda x: (x * 5) / 100)

    # Round numeric columns to 2 decimal places
    numeric_columns = ['Average Passing Grade', 'Passing Rate (%)', 'Average CCERT Grade (%)', 'Weighted Score (%)']
    df[numeric_columns] = df[numeric_columns].round(2)

    # Create the Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(color='black', size=12)),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left',
                   font=dict(color='black', size=11))
    )])

    # Update the layout for better appearance
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title_text="School Performance Overview",
        title_x=0.5,
        height=400  # Adjust this value based on your needs
    )

    # Display the table using Streamlit
    st.plotly_chart(fig, use_container_width=True)


def get_top_worst_students():
    query = """
    WITH student_scores AS (
        SELECT
            s.student_id,
            ROUND((0.5 * (a.grade / 5) + 0.5 * (a.c_cert / 80)), 2) as weighted_score,
            CASE
                WHEN a.cfit = 'L' THEN 2
                WHEN a.cfit = 'BA' THEN 4
                WHEN a.cfit = 'A' THEN 6
                WHEN a.cfit = 'AA' THEN 8
                WHEN a.cfit = 'H' THEN 10
                ELSE 0
            END AS cfit_rank,
            a.attr_A, a.attr_B, a.attr_C, a.attr_E, a.attr_F, a.attr_G, a.attr_H,
            a.attr_I, a.attr_L, a.attr_M, a.attr_N, a.attr_O, a.attr_Q1, a.attr_Q2,
            a.attr_Q3, a.attr_Q4, a.attr_EX, a.attr_AX, a.attr_TM, a.attr_IN, a.attr_SC
        FROM
            students s
        INNER JOIN
            assessments a
        ON
            s.Id = a.student_id
    )
    SELECT * FROM (
        SELECT * FROM student_scores
        ORDER BY weighted_score DESC
        LIMIT 30
    ) AS top_students
    UNION ALL
    SELECT * FROM (
        SELECT * FROM student_scores
        ORDER BY weighted_score ASC
        LIMIT 30
    ) AS bottom_students;
    """
    conn = get_engine()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


# Function to calculate the mean of each 16PF factor and sort them
def calculate_means(df):
    factor_columns = [
        'cfit_rank', 'attr_A', 'attr_B', 'attr_C', 'attr_E', 'attr_F', 'attr_G', 'attr_H',
        'attr_I', 'attr_L', 'attr_M', 'attr_N', 'attr_O', 'attr_Q1', 'attr_Q2',
        'attr_Q3', 'attr_Q4', 'attr_EX', 'attr_AX', 'attr_TM', 'attr_IN', 'attr_SC'
    ]
    factor_means = df[factor_columns].mean().sort_values(ascending=False)
    return factor_means


# Function to show common 16PF factors
def show_common_16pf():
    students = get_top_worst_students()
    df = pd.DataFrame(students)

    # Split into top 30 and worst 30
    top_30 = df.iloc[:30]
    worst_30 = df.iloc[30:]

    # Calculate means of 16PF factors
    top_30_means = calculate_means(top_30)
    worst_30_means = calculate_means(worst_30)

    # Create bar charts
    top_30_fig = px.bar(top_30_means, title='Mean 16PF Factors in Top 30 Students')
    worst_30_fig = px.bar(worst_30_means, title='Mean 16PF Factors in Worst 30 Students')

    # Show the charts in Streamlit
    st.plotly_chart(top_30_fig)
    st.plotly_chart(worst_30_fig)

    # Calculate squared mean differences with padding
    mean_diff = top_30_means - worst_30_means
    flat_padding = 0.1
    mean_diff_padded = mean_diff + flat_padding * np.sign(mean_diff)

    # Retain the sign and square the differences
    mean_diff = np.sign(mean_diff_padded) * (mean_diff_padded ** 2)

    mean_diff = mean_diff.sort_values(ascending=False)

    # Create bar chart for squared mean differences
    diff_fig = px.bar(mean_diff,
                      title='Squared Differences (with padding) in Mean 16PF Factors Between Top 30 and Worst 30 Students')

    # Show the difference chart in Streamlit
    st.plotly_chart(diff_fig)

    with st.expander("Rationale and How To Interpret This Chart"):
        st.markdown("""
        ## Interpretation Guide for This Section in The Dashboard
        
        The rationale behind the derived values in the charts is justified through the following key points:

        - Need for deriving differences
        - Squaring
        - Direction-Relative Padding
        - Calculation of CFIT values

        #### Need for Deriving Differences Between Best and Worst Students

        While the correlation between performance (grades, C-CERT) and features (16PF) is crucial in determining the attributes of a "good" computer studies student, it is equally important to identify the personality traits that distinguish "good" students from "bad" ones. By examining the differences in personality traits between high-performing and low-performing students, we can identify key attributes that significantly influence performance. This helps us hypothesize what "makes" a good student.

        #### Squaring

        Squaring values is a technique used to emphasize the magnitude of differences. Larger differences have a greater impact on the calculations, as demonstrated in established statistical models such as the sum of squared differences. This approach ensures that substantial differences are given more weight in the analysis.

        #### Direction-Relative Padding

        To prevent vanishing differences, a padding of 0.1 has been added to the values. Squaring values below 1 can cause them to diminish towards zero. To mitigate this, we apply a relative padding that maintains the direction of the difference. The padding is added based on the original direction of the difference: if the value is less than 0, we subtract 0.1; if the value is greater than 1, we add 0.1; otherwise, we add 0. 

        #### Calculation for CFIT Values

        The 16PF scores range from 1 to 10. To ensure consistency, we have assigned corresponding numerical values to CFIT rankings, which include:

        - L (Low) = 2
        - BA (Below Average) = 4
        - A (Average) = 6
        - AA (Above Average) = 8
        - H (High) = 10

        These values allow us to quantitatively assess CFIT rankings in relation to the 16PF scores.
        """)

class FeatureComponent:
    def __init__(self):
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        from pages_.model_training import fetch_data_as_dataframe, get_engine
        st.header("Features For Data Exploration and Engineering ⚙️")
        with st.spinner("Database query underway..."):
            query = f"""
                SELECT ((grade-1)/4)*100 as weighted, attr_A, attr_B, attr_C, attr_E, attr_F, attr_H, attr_G, attr_I, attr_L, attr_M, attr_N, attr_O, attr_Q1, attr_Q2,
                        attr_Q3, attr_Q4, attr_EX, attr_AX, attr_TM, attr_IN, attr_SC,
                        IF(cfit = 'L', 2,
                           IF(cfit = 'BA', 4,
                              IF(cfit = 'A', 6, 
                                 IF(cfit = 'AA', 8,
                                    IF(cfit = 'H', 10, NULL))))) as cfit, course,
                        CASE when course = 'BSCS' then 1 else 0 end as course_bscs,
                        CASE when course = 'BSIT' then 1 else 0 end as course_bsit
                FROM students
                INNER JOIN assessments s on s.student_id = students.Id WHERE cfit in ('L', 'BA', 'A', 'AA', 'H');
            """

            df = fetch_data_as_dataframe(get_engine(), query)

        # Cast the columns to the appropriate data types
        numeric_columns = ['weighted', 'attr_A', 'attr_B', 'attr_C', 'attr_E', 'attr_F', 'attr_H', 'attr_G', 'attr_I',
                           'attr_L', 'attr_M', 'attr_N', 'attr_O', 'attr_Q1', 'attr_Q2', 'attr_Q3', 'attr_Q4',
                           'attr_EX', 'attr_AX', 'attr_TM', 'attr_IN', 'attr_SC', 'cfit']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        plottable_attributes = ['attr_A', 'attr_B', 'attr_C', 'attr_E', 'attr_F', 'attr_H', 'attr_G', 'attr_I',
                                'attr_L', 'attr_M', 'attr_N', 'attr_O', 'attr_Q1', 'attr_Q2', 'attr_Q3', 'attr_Q4',
                                'attr_EX', 'attr_AX', 'attr_TM', 'attr_IN', 'attr_SC', 'cfit']

        with st.expander("Grade vs Attributes Scatter Plot", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                selected_attributes = st.multiselect(
                    "Select attributes to plot:",
                    options=plottable_attributes,
                    default=['attr_A', 'attr_B']
                )
            with col2:
                course_filter = st.multiselect(
                    "Filter by course:",
                    options=['BSCS', 'BSIT'],
                    default=['BSCS', 'BSIT']
                )

            if selected_attributes and course_filter:
                filtered_df = df[df['course'].isin(course_filter)]

                # Create a figure for each selected attribute
                for attr in selected_attributes:
                    # Sort the dataframe by the current attribute
                    sorted_df = filtered_df.sort_values(by=attr)

                    fig = px.scatter(sorted_df, x=attr, y='weighted',
                                     title=f'Grade vs {attr}',
                                     labels={attr: f'{attr} Score', 'weighted': 'Grade'},
                                     hover_data=['weighted', attr, 'course'],
                                     color='course')

                    fig.update_layout(
                        xaxis_title=f"{attr} Score",
                        yaxis_title="Grade",
                        xaxis=dict(range=[0, 10]),  # Set x-axis range to 0-10
                        yaxis=dict(range=[0, 100]),  # Set y-axis range to 0-100
                        template="plotly_white",
                    )

                    fig.update_traces(
                        hovertemplate=f"<b>Grade</b>: %{{y:.2f}}<br><b>{attr}</b>: %{{x:.2f}}<br><b>Course</b>: %{{customdata[2]}}<extra></extra>"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate and display correlation
                    correlation = sorted_df['weighted'].corr(sorted_df[attr])
                    st.write(f"Correlation between Grade and {attr}: {correlation:.4f}")

                st.subheader("Grade Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average Grade", f"{filtered_df['weighted'].mean():.2f}%")
                col2.metric("Median Grade", f"{filtered_df['weighted'].median():.2f}%")
                col3.metric("Minimum Grade", f"{filtered_df['weighted'].min():.2f}%")
                col4.metric("Maximum Grade", f"{filtered_df['weighted'].max():.2f}%")

            elif not selected_attributes:
                st.write("Please select at least one attribute to plot.")
            elif not course_filter:
                st.write("Please select at least one course to filter.")
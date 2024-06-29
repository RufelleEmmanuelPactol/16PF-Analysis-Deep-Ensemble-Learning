import streamlit as st
from pages_.loader_controller import retrieve_data_from_file, load_schools, load_users, merge_and_clean, load_merged_data_to_assessments


class LoaderComponent:

    def __init__(self):
        st.title("Data Loader Pipeline 📦")
        st.markdown("""
            🚀 Please utilize the provided template to ensure a seamless loading process.

            ⚠️ This page serves as the initiation point of the data pipeline. During the transformation stage, please be aware that some processing time is required. It is crucial that you DO NOT navigate away from this page while processing is underway.

            🔒 Rest assured, security measures have been implemented to allow the data loading process to be redone without the concern of duplicate data.
        """)

        with open('DataLoadingTemplate.xlsx', 'rb') as file:
            file_data = file.read()

        # Create a download button for the Excel file
        st.download_button(
            label="Download Template",
            data=file_data,
            file_name='DataLoadingTemplate.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        file = st.file_uploader("Please load the dataset provided that it follows the template standard.", type="xlsx")
        if file is not None:
            bscs_16pf, bsit_16pf, bscs_final, bsit_final = retrieve_data_from_file(file)
            data_mapper = {
                'BSCS 16 PF': bscs_16pf,
                'BSIT 16 PF': bsit_16pf,
                'BSCS Final Grades': bscs_final,
                'BSIT Final Grades': bsit_final

            }
            st.markdown("### View Data in Staging Area")
            returned = st.selectbox("Select Data to View:",
                                    ["BSCS 16 PF", "BSIT 16 PF", "BSCS Final Grades", "BSIT Final Grades"])
            st.dataframe(data_mapper[returned])
            tag = st.text_input("Input school year tag (ex: 2324)")
            if st.button("Confirm Accuracy and Load Data"):
                bscs = merge_and_clean(df_final=bscs_final, df16pf=bscs_16pf)
                bsit = merge_and_clean(df_final=bsit_final, df16pf=bsit_16pf)

                try:
                    tag = int(tag)
                except ValueError:
                    st.error("Please enter a year-tag using the prescribed format.")
                    return

                if not (2000 <= tag <= 3000):
                    st.error("Please enter a year-tag using the prescribed format.")
                    return

                st.warning("☕️ Grab a cup of coffee. This is gonna take a while...")

                debug_skip = False
                if not debug_skip:
                    with st.spinner("Loading Schools..."):
                        load_schools(bscs)
                        load_schools(bsit)

                with st.spinner("Loading Student Information..."):
                    load_users(bscs, tag)
                    load_users(bsit, tag, batch=2)

                with st.spinner("Loading assessment information to database..."):
                    load_merged_data_to_assessments(bscs)
                    load_merged_data_to_assessments(bsit)

                st.info("The loading process has finished successfully! Train the models or view the visualizations.")



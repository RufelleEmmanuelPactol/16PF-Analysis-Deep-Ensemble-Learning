import streamlit as st




def AuthComponent():
    st.header("Welcome to CCS Dashboard")
    st.markdown("To use this dashboard, please initiate the authentication process.")
    user = st.text_input("Enter username: ")
    password = st.text_input("Enter password: ", type="password")
    checked = st.checkbox("I accept CIT University's the privacy policy.")
    if st.button("Login"):
        if user == "CCS" and password == "CCS2024":
            if not checked:
                st.warning("Please agree to CIT University's privacy policy.")
        else:
            st.warning("Invalid username or password.")
            return False

        return checked


def MainComponent():
    st.sidebar.title("CCS Dashboard Apps")

    result = st.sidebar.selectbox(options=["Home 🏠", 'About 16 Personalities 🤼‍️',
                                           "Data Loading (ETL) 📦", "Model Training 🚀",
                                           "School Dashboards📓", "Year-on-Year Dashboards 💹",
                                           "Batch Prediction 🪄", "Analytics Report",],
                                  label="Select an app to explore.")

    from pages_.model_training import ModelTrainingComponent
    from pages_.personalities import PersonalitiesComponent
    from pages_.home import HomeComponent
    from pages_.loader_view import LoaderComponent
    from pages_.yoy_view import YearOnYearComponent
    from pages_.model_prediction import ModelPredictionComponent
    from pages_.school_dashboard import SchoolDashboardComponent

    page_map = {
        'About 16 Personalities 🤼‍️': PersonalitiesComponent,
        'Home 🏠': HomeComponent,
        "Data Loading (ETL) 📦": LoaderComponent,
        'Year-on-Year Dashboards 💹': YearOnYearComponent,
        'Model Training 🚀': ModelTrainingComponent,
        "Batch Prediction 🪄" : ModelPredictionComponent,
        "School Dashboards📓" : SchoolDashboardComponent

    }



    page_map.get(result)()



if __name__ == "__main__":
    if st.session_state.get("auth", False):
        MainComponent()

    else:
        if AuthComponent():
            st.session_state["auth"] = True
            st.rerun()

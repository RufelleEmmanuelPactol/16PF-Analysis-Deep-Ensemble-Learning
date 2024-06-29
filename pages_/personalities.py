
def PersonalitiesComponent():
    import streamlit as st
    import pandas as pd

    st.title("Extract-Transform-Load (ETL) Tool for 16PF ⚙️")
    st.markdown("Read more about this project, or learn about 16 Personalities.")

    st.info("Use the sidebar to select sections.")
    # Define the data⚙️
    primary_factors_data = [
        {"Factor": "A (Warmth)",
         "Low Factor": "The individual has the tendency to be socially reserved, impersonal, or distant towards other people",
         "High Factor": "The individual has the tendency to be warm, outgoing, and attentive towards others.",
         "Factor Description": "Tendency to be warmly involved with people vs. the tendency to be more reserved socially and interpersonally"},
        {"Factor": "B (Reasoning)",
         "Low Factor": "Low scorers may have difficulty in solving more complex items. They tend to be more hands-on in their learning style and may often need more time to reflect.",
         "High Factor": "High scorers can generally be seen as quick-learners and tend to be more adept at abstract thinking and problem solving.",
         "Factor Description": "This factor is a brief measure of reasoning or intelligence, although it is not intended as a replacement for more reliable, full-length measures of mental ability."},
        {"Factor": "C (Emotional Stability)",
         "Low Factor": "The individual tends to be reactive. They may feel a certain lack of control over life.",
         "High Factor": "The individual tends to manage events and emotions in a balanced, adaptive way. They may also make proactive choices in managing their lives.",
         "Factor Description": "Concerns feelings about coping with day to day life and its challenges"},
        {"Factor": "E (Dominance)",
         "Low Factor": "Tends to be deferential and cooperative, and may prefer to avoid conflict with others.",
         "High Factor": "Tends to be forceful, vocal in expressing their wishes and opinions even when not invited to do so, and pushy about obtaining what they want.",
         "Factor Description": "Tendency to exert one’s will over others vs. accommodating others’ wishes"},
        {"Factor": "F (Liveliness)",
         "Low Factor": "Low scorers tend to take life more seriously. They may be quieter and more cautious. They tend to inhibit their spontaneity, sometimes to the point of appearing constricted.",
         "High Factor": "High scorers are enthusiastic, spontaneous, and attention-seeking. They are lively and drawn to stimulating social situations.",
         "Factor Description": "Reflects enthusiasm, spontaneity, and attention-seeking behavior"},
        {"Factor": "G (Rule-Consciousness)",
         "Low Factor": "Tend to eschew the rules and regulations or they may have difficulty in conforming to strict rules.",
         "High Factor": "Tend to perceive themselves as strict follower of manners and principles, may depict themselves as rule-bound and conscientious.",
         "Factor Description": "Concerns adherence to rules and regulations"},
        {"Factor": "H (Social Boldness)",
         "Low Factor": "Tend to be socially timid, cautious, and shy. They may find speaking in front of a group of people to be a difficult experience.",
         "High Factor": "High scorers may consider themselves to be bold and adventurous in social groups, and show little fear of social situations.",
         "Factor Description": "Reflects social boldness and fearlessness in social situations"},
        {"Factor": "I (Sensitivity)",
         "Low Factor": "Low scorers tend to focus on utility and objectivity. Those with extreme scores may have trouble dealing with situations that demand sensitivity.",
         "High Factor": "High scorers tend to base judgements on personal taste and aesthetic values.",
         "Factor Description": "Reflects sensitivity to personal taste and aesthetic values"},
        {"Factor": "L (Vigilance)",
         "Low Factor": "Low scorers tend to be trusting and to expect fair treatment, loyalty, and good intentions from others.",
         "High Factor": "High scorers expect to be misunderstood or taken advantage of. They may be vigilant and suspicious towards other people’s intentions and motives.",
         "Factor Description": "Reflects vigilance and suspicion towards others' intentions"},
        {"Factor": "M (Abstractedness)",
         "Low Factor": "They focus more on their senses, observable data, and the realities and demands of their environment.",
         "High Factor": "They are more oriented to internal mental processes and ideas rather than practicalities.",
         "Factor Description": "Reflects orientation towards internal mental processes and ideas"},
        {"Factor": "N (Privateness)",
         "Low Factor": "Low scorers tend to talk about themselves readily. They are genuine, self-revealing, and forthright.",
         "High Factor": "High scorers tend to be private and non-disclosing when it comes to personal information.",
         "Factor Description": "Reflects the degree of privacy and non-disclosure of personal information"},
        {"Factor": "O (Apprehension)",
         "Low Factor": "Low scorers tend to be more self-assured, neither prone to apprehensiveness nor troubled about their sense of adequacy.",
         "High Factor": "High scorers tend to worry about things and to feel apprehensive and insecure.",
         "Factor Description": "Reflects apprehension and insecurity"},
        {"Factor": "Q1 (Openness to Change)",
         "Low Factor": "Low scorers tend to prefer traditional ways of looking at things. They don’t question the way things are done.",
         "High Factor": "High scorers tend to think of ways to improve things.",
         "Factor Description": "Reflects openness to change and improvement"},
        {"Factor": "Q2 (Self-Reliance)",
         "Low Factor": "They prefer to be around people and likes to do activities with them.",
         "High Factor": "They enjoy time alone and prefer to make decisions for themselves.",
         "Factor Description": "Reflects preference for self-reliance and independence"},
        {"Factor": "Q3 (Perfectionism)",
         "Low Factor": "Low scorers tend to leave more things to chance and tend to be more comfortable in a disorganized setting.",
         "High Factor": "High scorers tend to be organized, to keep things in their proper places, and to plan ahead.",
         "Factor Description": "Reflects the degree of perfectionism and organization"},
        {"Factor": "Q4 (Tension)",
         "Low Factor": "Low scorers tend to be relaxed and patient.",
         "High Factor": "High scorers tend to have a restless energy.",
         "Factor Description": "Reflects the degree of tension and restless energy"}
    ]

    # Re-defining data for global factors
    global_factors_data = [
        {"Factor": "EX (Extraversion)",
         "Low Factor": "Tend to value time spent alone or in solitary pursuits, being generally less inclined to seek out interaction with others.",
         "High Factor": "Tend to be people-oriented, to seek interaction with others, and to value time spent with others in social pursuits."},
        {"Factor": "AX (Anxiety)",
         "Low Factor": "Tend to be unperturbed by most events and less easily upset than most people.",
         "High Factor": "Tend to be more easily upset by events."},
        {"Factor": "TM (Tough-Mindedness)",
         "Low Factor": "Tend to be open to feelings, imagination, people, and new ideas.",
         "High Factor": "Tend to prefer logical, realistic solutions."},
        {"Factor": "IN (Independence)",
         "Low Factor": "Tend to be agreeable and accommodating to other people and external influences.",
         "High Factor": "Tend to take charge of situations and to influence others."},
        {"Factor": "SC (Self-Control)",
         "Low Factor": "Low scorers are unrestrained and tend to have fewer resources for controlling their behavior.",
         "High Factor": "High scorers are conscientious and have substantial resources for controlling their behavior and meeting their responsibilities."}
    ]

    # Re-defining data for Culture Fair Intelligence Test Scale 3
    cf_test_scale_3_data = [
        {"Classification": "LOW",
         "Interpretation": "An individual may have difficulty comprehending new and complicated concepts. It may take them a longer time to understand highly complicated new concepts and to reason deductively."},
        {"Classification": "BELOW AVERAGE",
         "Interpretation": "This may mean that an individual may have difficulty comprehending complicated concepts, and can study new programs or other concepts more effectively if given much longer time and practice."},
        {"Classification": "AVERAGE",
         "Interpretation": "This reflects a moderate ability to comprehend complex concepts presented. An individual’s mental abilities, such as memory, pattern-recognition, problem-solving, and learning are deemed to be adequately developed."},
        {"Classification": "ABOVE AVERAGE",
         "Interpretation": "This indicates a quick comprehension on the complicated concepts presented. The may easily recognize patterns, and may be able to learn new things quickly."},
        {"Classification": "HIGH",
         "Interpretation": "This indicates that an individual can perceive information accurately, recognize and recall what has been perceived, understand relationships, and apply the information to form a new and different concept. Such implies considerably strong analytical and problem- solving skills."}
    ]

    # Creating dataframes
    primary_factors_df = pd.DataFrame(primary_factors_data)
    global_factors_df = pd.DataFrame(global_factors_data)
    cf_test_scale_3_df = pd.DataFrame(cf_test_scale_3_data)

    # Create a Streamlit app

    # Display the DataFrame as a table

    section = st.sidebar.selectbox('Choose a section', ['Primary Factors', 'Global Factors', 'CF Test Scale'])

    def display_primary_factors():
        st.subheader("Primary Factors")
        st.markdown(primary_factors_df.to_markdown(index=False))

    def display_global_factors():
        st.subheader("Global Factors")
        st.markdown(global_factors_df.to_markdown(index=False))

    def display_cf_test_scale():
        st.subheader("CF Test Scale")
        st.markdown(cf_test_scale_3_df.to_markdown(index=False))

    # Main content based on the selected section
    if section == 'Primary Factors':
        display_primary_factors()
    elif section == 'Global Factors':
        display_global_factors()
    elif section == 'CF Test Scale':
        display_cf_test_scale()



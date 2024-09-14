import tempfile
import pandas as pd
import streamlit as st
from App import (recommend_jobs, read_data, job_by_source, job_by_type,rank_cvs,send_mail,
                 job_type_by_source, most_offered_company, top_skills,most_frequent_jobs
                 ,recommend_jobs_by_experience, recommend_by_skills_and_experience,rank_cvs_exp)
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util


st.set_page_config(page_title="Job Recommendation App", layout="wide")
st.markdown(
    """
        <style>
            .{
                margin:0;
                
            }
            .block-container {
                margin:0;
            }
            .st-emotion-cache-6qob1r {
                position: relative;
                height: 100%;
                width: 100%;
                overflow: overlay;
            }   
            .st-emotion-cache-1gv3huu {
                position: relative;
                top: 2px;
                background-color: rgb(240, 242, 246);
                z-index: 999991;
                min-width: 300px;
                max-width: 300px;
                transform: none;
                transition: transform 300ms ease 0s, min-width 300ms ease 0s, max-width 300ms ease 0s;
            }

            .st-emotion-cache-1whx7iy p {
                word-break: break-word;
                margin-bottom: 0px;
                font-size: 20px;
                font-weight : bold;
            }
            .st-am {
                margin-left: 15px;
                margin-bottom: 5px;
            }
            p, ol, ul, dl {
                margin: 0px 0px 1rem;
                padding: 0px;
                font-size: 1.1rem;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True
)

nav_options = {
    "Accueil": "🏠",
    "Job Matcher": "🔎",
    "Job Recommender": "🔍",
    "Dashboard": "📊"
}
nav_selection = st.sidebar.radio(
    "Navigation",
    options=list(nav_options.keys()),
    format_func=lambda x: f"{nav_options[x]} {x}"
)

def accueil():
    st.title("Welcome to the Job Recommendation App")

    # Create a centered layout using columns
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(
            "The **Job Recommendation App** is designed to help you find the most relevant job opportunities based on your skills and preferences. Here's a brief overview of what you can do with this app:"
        )

        st.header("Features:")
        st.write(
            "- **Job Recommender**: Upload your CV and get personalized job recommendations based on your skills and experience. Adjust filters to refine your search based on the time period and the number of job recommendations you want to see."
        )
        st.write(
            "- **Dashboard**: Visualize job data with interactive charts. Explore various metrics such as the most frequent jobs, top skills, job sources, job types, and more."
        )

        st.header("How it Works:")
        st.write(
            "1. **Upload Your CV**: Start by uploading your CV in PDF format. The app will analyze your document to understand your skills and experience."
        )
        st.write(
            "2. **Set Filters**: Choose a time period to filter job postings (e.g., Last Day, Last Week, Last Month) and specify the number of job recommendations you want."
        )
        st.write(
            "3. **Get Recommendations**: The app will provide you with a list of job opportunities that match your profile. You can view details and apply directly through provided links."
        )
        st.write(
            "4. **Explore Data**: Navigate to the Dashboard to see visualizations of job trends, skills, and sources based on the available job data."
        )

        st.header("Benefits:")
        st.write(
            "- **Personalized Recommendations**: Get job suggestions tailored to your skills and experience."
        )
        st.write(
            "- **Interactive Visualizations**: Analyze job market trends and data through interactive charts and graphs."
        )
        st.write(
            "- **Easy to Use**: Simple and intuitive interface to quickly find and explore job opportunities."
        )

        st.write(
            "If you have any questions or need help using the app, feel free to reach out to our support team."
        )


def matcher():
    st.title('Job Matcher')
    x, y = st.columns(2)

    with x:
        job_description_file = st.file_uploader('Upload the job description', type='pdf')
    with y:
        cv_files = st.file_uploader('Upload CVs', type='pdf', accept_multiple_files=True)

    if job_description_file is not None and cv_files:
        with x:
            a, b = st.columns([2, 1])
            # User input for number of CVs and matching criteria
            with a:
                radio_option = st.radio("Matching By:", ["Skills", "Experience and Project"], index=0, horizontal=True)

        
        if radio_option == "Skills":
            st.session_state.df = rank_cvs(job_description_file, cv_files)
        elif radio_option =="Experience and Project":
            st.session_state.df = rank_cvs_exp(job_description_file, cv_files)

        
        if 'df' in st.session_state:
            df = st.session_state.df
            st.subheader("Matching Results")
            email_password = 'kdbm reqo pfti tfnu'
            email_sender = 'amyne095@gmail.com'
            for index, row in df.iterrows():
                with st.expander(f"Details for {row['CV Name']} ({row['Similarity']:.2f} Match)"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Name:** {row['CV Name']}")
                        st.write(f"**Email:** {row['Email']}")
                        st.write(f"**Similarity:** {row['Similarity']:.2f}")
                    with col2:
                        if st.button(f"Send Email", key=f"send_email_{index}"):
                            send_mail(email_sender, email_password, row['Email'])
                            st.success(f"E-mail send to {row['Email']}")

def recommender():
    st.title("Job Recommender System")

    col1, _ = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose a CV (PDF only)", type="pdf")
    

    if uploaded_file is not None:


        
        x,y,z = st.columns([1,1,5])
        with x:
            time_filter = st.selectbox(
                'Select Time Period',
                options=['All Time', 'Last Day', 'Last Week', 'Last Month']
            )
        with y:
            # Number input field
            number = st.number_input(
                label="Enter a number",
                min_value=0,  # Minimum value
                max_value=100,  # Maximum value
                value=10,  # Default value
            )
        with z:
            radio_option = st.radio("Recommend By:", ["Skills", "Experience and Project", "Both"],index=0, horizontal=True)

        data = read_data()
        data = filter_data_by_time(data, time_filter)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.subheader("Best jobs found")
        if radio_option == "Skills":
            df = recommend_jobs(data, pdf_path, number)
        elif radio_option == "Experience and Project":
            df = recommend_jobs_by_experience(data, pdf_path, number)
        elif radio_option == "Both":
            df = recommend_by_skills_and_experience(data, pdf_path, number)

        num_columns = 3  # Adjust this number to change how many cards per row
        for i in range(0, len(df), num_columns):
            cols = st.columns(num_columns)
            for col, (_, row) in zip(cols, df.iloc[i:i + num_columns].iterrows()):
                with col:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #e6e6e6; 
                                    border-radius: 5px; padding: 15px; 
                                    margin-bottom: 10px; 
                                    box-shadow: 5px 8px 15px rgba(0,0,0,0.4);">
                            <h4 style="color:#4a90e2">{row['job_name']}</h4>
                            <p><b>Location:</b> {row['job_location']}</p>
                            <p>{row['job_text'][:150]}...</p>
                            <a href="{row['job_link']}">Read more</a>
                        </div>
                        """, unsafe_allow_html=True
                    )



# Filtering function based on selected time period
def filter_data_by_time(df, period):
    # Determine the start date based on the period
    if period == 'Last Day':
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    elif period == 'Last Week':
        start_date = (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')
    elif period == 'Last Month':
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        return df  # No filtering for 'All Time'

    # Ensure the 'job_date' column is in datetime format
    df['job_date'] = pd.to_datetime(df['job_date']).dt.strftime('%Y-%m-%d')

    # Filter the DataFrame
    return df[df['job_date'] >= start_date]

def dashbord_page():
    st.title("Dashboard")

    x,y,z = st.columns([1,3,3])
    with x:
        time_filter = st.selectbox(
            'Select Time Period',
            options=['All Time', 'Last Day', 'Last Week', 'Last Month']
        )

    df = read_data()

    # Apply time filter
    df = filter_data_by_time(df, time_filter)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = most_frequent_jobs(df, 'job_name')
        st.plotly_chart(fig1)

    with col2:
        fig2 = top_skills(df, 'skills')
        st.plotly_chart(fig2)

    with col1:
        fig3 = job_by_source(df)
        st.plotly_chart(fig3)
        
    with col2:
        fig4 = job_by_type(df)
        st.plotly_chart(fig4)        
        
    with col1:
        fig5 = most_offered_company(df)
        st.plotly_chart(fig5)

    with col2:
        fig6 = job_type_by_source(df)
        st.plotly_chart(fig6)

if nav_selection == "Accueil":
    accueil()
if nav_selection == "Job Matcher":
    matcher()
elif nav_selection == "Job Recommender":
    recommender()
elif nav_selection == "Dashboard":
    dashbord_page()


st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                text-align: center;
                padding: 10px;
                font-size: 0.9rem;
                color: #333;
                border-top: 1px solid #ddd;
            }
        </style>
        <div class="footer">
            &copy; 2024 Job Recommendation App. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

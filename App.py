import re
import os
import cv2
import ssl
import fitz 
import spacy
import smtplib
import tempfile
import psycopg2
import pytesseract
import pandas as pd
from PIL import Image
import plotly.express as px
from collections import Counter
from email.message import EmailMessage
from inference_sdk import InferenceHTTPClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

def get_skills(text):
    nlp = spacy.load("en_core_web_lg",disable = ['ner'])
    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk("/home/huser/app/jz_skill_patterns.jsonl")
    doc = nlp(text.lower())
    list_skills = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            list_skills.append(ent.text.lower())  
    
    list_skills = list(set(list_skills))
    return ' '.join(list_skills)

def pdf_to_jpg(pdf_path):
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = f"cv_{page_number + 1}.jpg"
        img.save(output_path, "JPEG")
        return output_path
    
# get skills
def get_skills_cv(path):
    image_path = pdf_to_jpg(path)
    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="BIrE2FC1DoCP2tdXB9MA"
    )
    
    # Define the model ID
    model_id = "resume-parser-bchlq/1"
    result = CLIENT.infer(image_path, model_id=model_id)
    
    image = cv2.imread(image_path)

    # Extract the 'Skills' section from the result
    skills_section = None
    for prediction in result['predictions']:
        if prediction['class'] == 'skills':
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            width = int(prediction['width'])
            height = int(prediction['height'])
            
            skills_image = image[y:y+height, x:x+width]            
            skills_image_rgb = cv2.cvtColor(skills_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(skills_image_rgb)
            skills_text = pytesseract.image_to_string(pil_image)
            
            return skills_text
    else:
        print("No 'Skills' section found in the image.")

# get skills from experience and project
def get_skills_experience_and_project_cv(path):
    image_path = pdf_to_jpg(path)
    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="BIrE2FC1DoCP2tdXB9MA"
    )
    
    # Define the model ID
    model_id = "resume-parser-bchlq/1"
    result = CLIENT.infer(image_path, model_id=model_id)
    
    image = cv2.imread(image_path)

    skills_text = ''
    for prediction in result['predictions']:
        if prediction['class'] == 'Experience':
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            width = int(prediction['width'])
            height = int(prediction['height'])
            
            skills_image = image[y:y+height, x:x+width]            
            skills_image_rgb = cv2.cvtColor(skills_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(skills_image_rgb)
            skills_text += pytesseract.image_to_string(pil_image)
        
        if prediction['class'] == 'Projects':
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            width = int(prediction['width'])
            height = int(prediction['height'])
            
            skills_image = image[y:y+height, x:x+width]            
            skills_image_rgb = cv2.cvtColor(skills_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(skills_image_rgb)
            skills_text += pytesseract.image_to_string(pil_image)
    
    return get_skills(skills_text)

def connect():
    try:
        conn = psycopg2.connect("host=127.0.0.1 dbname=postgres user=postgres password=huser")
        print('DB connected successfully')
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        raise
    return conn

def get_data():
    try:
        conn = connect()
        cursor = conn.cursor()

        fetch_data_sql = "SELECT job_link, job_name, job_text, job_company, job_location, job_type, job_date, skills FROM jobs_data;"

        try:
            cursor.execute(fetch_data_sql)
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=colnames)
            df.drop_duplicates(subset=['job_text'], inplace=True)
            return df

        except Exception as ex:
            print(f"An error occurred while fetching data: {ex}")

        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        print(f"Error connecting to the database: {e}")        

# Recommendation by skills
def recommend_jobs(df,path, top_n=15):
    skills_cv = get_skills_cv(path)
    job_skills_list = df['skills'].tolist()
    skills_to_exclude = {'documentation', 'support', 'business', 'collaboration'}

    # Filter the list to exclude specified skills
    skills_to_exclude = [skill for skill in job_skills_list if skill not in skills_to_exclude]


    skills_cv_list = [skills_cv]
    all_skills = skills_cv_list + job_skills_list

    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_skills)
    cv_vector = vectorizer.transform(skills_cv_list)
    job_vectors = vectorizer.transform(job_skills_list)

    similarity_scores = cosine_similarity(cv_vector, job_vectors).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['job_link', 'job_name', 'job_text','job_location','job_company']].copy()
    recommendations['job_text'] = recommendations['job_text'].str.replace('\n', ' ', regex=False)
    recommendations['score'] = similarity_scores[top_indices]
    
    return recommendations 




# Recommendation by Experience
def recommend_jobs_by_experience(df,path, top_n=15):
    skills_cv = get_skills_experience_and_project_cv(path)
    job_skills_list = df['skills'].tolist()
    skills_to_exclude = {'documentation', 'support', 'business', 'collaboration'}

    # Filter the list to exclude specified skills
    skills_to_exclude = [skill for skill in job_skills_list if skill not in skills_to_exclude]


    skills_cv_list = [skills_cv]
    all_skills = skills_cv_list + job_skills_list

    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_skills)
    cv_vector = vectorizer.transform(skills_cv_list)
    job_vectors = vectorizer.transform(job_skills_list)

    similarity_scores = cosine_similarity(cv_vector, job_vectors).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['job_link', 'job_name', 'job_text','job_location','job_company']].copy()
    recommendations['job_text'] = recommendations['job_text'].str.replace('\n', ' ', regex=False)
    recommendations['score'] = similarity_scores[top_indices]
    
    return recommendations      

def recommend_by_skills_and_experience(df, path, top_n=15):
    # Get skills-based recommendations
    skills_cv = get_skills_cv(path)
    experience_cv = get_skills_experience_and_project_cv(path)
    job_skills_list = df['skills'].tolist()
    
    vectorizer = TfidfVectorizer()
    all_skills = [skills_cv] + job_skills_list 
    all_skills = [experience_cv] + all_skills
    vectorizer.fit(all_skills)
    
    cv_vector = vectorizer.transform([skills_cv])
    job_vectors = vectorizer.transform(job_skills_list)

    # Get experience-based recommendations
    exp_vector = vectorizer.transform([experience_cv])
    experience_similarity_scores = cosine_similarity(exp_vector, job_vectors).flatten()
    skill_similarity_scores = cosine_similarity(cv_vector, job_vectors).flatten()
    
    # Combine the scores
    combined_scores = (skill_similarity_scores + experience_similarity_scores) / 2
    # Get top job recommendations
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['job_link', 'job_name', 'job_text', 'job_location', 'job_company']].copy()
    recommendations['job_text'] = recommendations['job_text'].str.replace('\n', ' ', regex=False)
    recommendations['score'] = combined_scores[top_indices]
    
    return recommendations


##### dashboard
def determine_source(url):
    if "linkedin" in url:
        return "LinkedIn"
    elif "indeed" in url:
        return "Indeed"
    elif "welcometothejungle" in url:
        return "Welcome to the Jungle"
    elif "glassdoor" in url:
        return "Glassdoor"
    else:
        return "Unknown"
    

def read_data():
    data = get_data()
    df = data.dropna()
    df['source'] = df['job_link'].apply(determine_source)
    df['job_type'] = df['job_type'].replace('AUTRE', 'Not Specified')
    return df

def job_by_source(df):
    fig = px.pie(df, 
                 names='source', 
                 title='Distribution of Jobs by Source',
                 color='source',  
                 color_discrete_sequence=px.colors.qualitative.Plotly, 
                 hole=0.3)  
    
    fig.update_layout(
        title_text='Distribution of Jobs by Source',
        title_font_size=24,
        title_x=0.5,
        title_xanchor='center',  
        legend_title_text='Source',
        legend=dict(
            orientation='v', 
            yanchor='bottom',
            y=-0.2,
            xanchor='right',
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)  
    )   
    
    fig.update_traces(
        textinfo='label+percent', 
        pull=[0.1] * len(df['source'])  
    )
    
    return fig
def cut_name(text):
    return text[:20]

def job_by_type(df):
    # data= df[df['job_type'] != 'Not Specified']
    fig = px.pie(df, 
                 names='job_type', 
                 title='Job Counts by Type',
                 color='job_type',  
                 color_discrete_sequence=px.colors.qualitative.Plotly, 
                 hole=0.3)  
    
    fig.update_layout(
        title_text='Job Counts by Type',
        title_font_size=24,
        title_x=0.5,
        title_xanchor='center',  
        legend_title_text='Job Type',
        legend=dict(
            orientation='v',  
            yanchor='bottom',
            y=-0.2,
            xanchor='left',
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)  
    )
    

    fig.update_traces(
        textinfo='label+percent',  
        pull=[0.05] * len(df['job_type'])
    )
    
    return fig


def most_offered_company(df,top_n=10):
    data= df[df['job_company'] != 'Not found']
    company_counts = data['job_company'].value_counts().reset_index()
    company_counts.columns = ['job_company', 'count']
    top_companies = company_counts.head(top_n)
    top_companies['job_company'] = top_companies['job_company'].apply(cut_name)

    fig = px.bar(top_companies, x='job_company', y='count', title='Most Frequently Offered Companies',color='job_company')
    fig.update_layout(
        title_font_size=24,
    )
    
    return fig


def job_type_by_source(df):
    fig = px.histogram(df, 
                    x='source', 
                    color='job_type', 
                    color_discrete_sequence=px.colors.qualitative.Plotly, 
                    title='Job Types by Source Link',
                    labels={'source_link': 'Source Link', 'count': 'Job Count'},
                    category_orders={'job_type': df['job_type'].value_counts().index.tolist()})

    fig.update_layout(
        barmode='stack', 
        title_font_size=24,
        xaxis_title='Source Link',
        yaxis_title='Count',
        legend_title='Job Type',
    )
    return fig


def top_skills(df, skills_column, top_n=15):
    
    all_skills = df[skills_column].dropna().str.split(',').explode().str.strip()
    skills_to_exclude = {'documentation', 'support', 'business', 'collaboration','commerce'}

# Filter the list to exclude specified skills
    all_skills = [skill for skill in all_skills if skill not in skills_to_exclude]
    skill_counts = Counter(all_skills)
    
    skill_df = pd.DataFrame(skill_counts.items(), columns=['Skill', 'Count'])
    skill_df = skill_df.sort_values(by='Count', ascending=False)
    top_skills_df = skill_df.head(top_n)
    
    fig = px.bar(top_skills_df, 
                 x='Skill', 
                 y='Count', 
                 title=f'Top {top_n} Most Common Job Skills',
                 labels={'Skill': 'Skill', 'Count': 'Count'},
                 color='Skill',
                 color_discrete_sequence=px.colors.qualitative.Plotly,)
    
    fig.update_layout(
        xaxis_title='Skill',
        yaxis_title='Count',
        title_font_size=24,
        xaxis_tickangle=-45,
        title_xanchor='center',
        title_x=0.5
    )
    
    return fig

def most_frequent_jobs(df, job_column, top_n=10):
    KEYWORD_SYNONYMS = {
    'data analyst': [
        'data analyst', 'analyse de données', 'analyst', 'quantitative analyst', 'data analytics', 'data investigator',
        'data examiner', 'report analyst', 'data consultant', 'data researcher', 'data specialist', 'data evaluator',
        'information analyst', 'analytics consultant', 'performance analyst','analyste de données'
    ],
    'data scientist': [
        'data scientist', 'scientifique des données', 'data science', 'data science specialist', 'machine learning engineer',
        'ML scientist', 'statistician', 'quantitative researcher', 'data modeler', 'AI engineer', 'artificial intelligence engineer',
        'data researcher', 'predictive analyst', 'data strategist', 'data developer','sciences de données','power bi','powerbi'
    ],
    'data engineer': [
        'data engineer', 'ingénieur des données','ingénieur data', 'data engineering', 'ETL developer', 'data architect', 'data systems engineer',
        'data pipeline engineer', 'data warehouse engineer', 'big data engineer', 'data infrastructure engineer', 'data integration engineer',
        'data operations engineer', 'data operations specialist', 'data platform engineer'
    ],
    'web developer': [
        'web developer', 'développeur web', 'full stack','fullstack', 'front-end', 'back-end ', 'web',
        'web designer', 'UI developer', 'UX developer', 'site developer', 'application developer', 'software developer',
        'web application developer', 'web systems developer', 'web consultant', 'web architect','devops'
    ],
    'mobile developer': [
        'mobile developer', 'développeur mobile', 'iOS developer', 'Android developer', 'mobile app developer', 'mobile application developer',
        'mobile software engineer', 'mobile UI/UX designer', 'app developer', 'cross-platform developer', 'mobile engineer',
        'flutter developer', 'react native developer', 'mobile systems developer', 'mobile technology specialist'
    ],
    'project manager': [
        'project manager', 'chef de projet', 'senior project manager', 'project coordinator', 'project leader', 'project director',
        'program manager', 'project supervisor', 'project planner', 'project administrator', 'project executive', 'project controller'
    ],
    'software engineer': [
        'software engineer', 'logiciel', 'software', 'software architect', 'programmer', 'application developer',
        'system software engineer', 'software designer', 'software consultant', 'software development engineer', 'code developer',
        'software development specialist'
    ],
    'data architect': [
        'data architect', 'architecte de données', 'data modeler', 'data systems architect', 'information architect', 'data engineer',
        'data structure engineer', 'data strategy consultant', 'data infrastructure architect', 'data integration architect'
    ]
    }
    def categorize_job_name(job_name):
        job_name_lower = job_name.lower()
        for category, synonyms in KEYWORD_SYNONYMS.items():
            if any(re.search(r'\b' + re.escape(synonym) + r'\b', job_name_lower) for synonym in synonyms):
                return category
        return job_name

    job_titles = df[job_column].dropna()
    job_titles = df['job_name'].apply(categorize_job_name)
    
    job_counts = Counter(job_titles)
    
    job_df = pd.DataFrame(job_counts.items(), columns=['Job Title', 'Count'])
    job_df = job_df.sort_values(by='Count', ascending=False)
    job_df['Job Title'] = job_df['Job Title'].apply(cut_name)

    
    # Select top N job titles
    top_jobs_df = job_df.head(top_n)
    
    # Create a bar chart for the most frequent job titles
    fig = px.bar(top_jobs_df, 
                 x='Job Title', 
                 y='Count', 
                 title=f'Top {top_n} Most Common Job Titles',
                 labels={'Job Title': 'Job Title', 'Count': 'Count'},
                 color='Job Title',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Update layout for better design
    fig.update_layout(
        xaxis_title='Job Title',
        yaxis_title='Count',
        title_font_size=24,
        title_xanchor='center',  
        xaxis_tickangle=-15,  
        title_x=0.5
    )
    
    return fig



def get_mail(path):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    pdf_document = fitz.open(path, filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    liste_emails = r.findall(text) 
    emails_valides = []
    for email in liste_emails:
        if email.endswith(('.com', '.tn','.org','.net')):
            emails_valides.append(email)
            
    if len(emails_valides)!=0:
        return emails_valides[0]
    else:
        return None

def rank_cvs(jd, cvs):
    ranked_cvs = []
    pdf_document = fitz.open(stream=jd.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    for cv in cvs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(cv.read())
            pdf_path = tmp_file.name

        job_skill = get_skills(text)
        skills_cv = get_skills_cv(pdf_path)
        skills_cv = get_skills(skills_cv)
        
        #Compute embedding for both lists
        embedding_1= model.encode(skills_cv, convert_to_tensor=True)
        embedding_2 = model.encode(job_skill, convert_to_tensor=True)

        similarity_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)
        similarity_value = similarity_tensor.item()

        mail = get_mail(pdf_path)

        ranked_cvs.append((cv.name, mail, similarity_value))
        
    df = pd.DataFrame(ranked_cvs, columns=['CV Name', 'Email', 'Similarity'])
    df = df.sort_values(by='Similarity', ascending=False)
    return df


def rank_cvs_exp(jd, cvs):
    ranked_cvs = []
    pdf_document = fitz.open(stream=jd.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for cv in cvs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(cv.read())
            pdf_path = tmp_file.name

        job_skill = get_skills(text)
        skills_cv = get_skills_experience_and_project_cv(pdf_path)
        skills_cv = get_skills(skills_cv)
        
        #Compute embedding for both lists
        embedding_1= model.encode(skills_cv, convert_to_tensor=True)
        embedding_2 = model.encode(job_skill, convert_to_tensor=True)

        similarity_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)
        similarity_value = similarity_tensor.item()

        mail = get_mail(pdf_path)

        ranked_cvs.append((cv.name, mail, similarity_value))
    df = pd.DataFrame(ranked_cvs, columns=['CV Name', 'Email', 'Similarity'])
    df = df.sort_values(by='Similarity', ascending=False)
    return df


def send_mail(email_sender, email_password,email_receiver):
    # Set the subject and body of the email
    subject = 'Merci pour votre participation au processus de recrutement'
    body = """
    Bonjour,

    Nous vous remercions sincèrement pour votre participation au processus de recrutement pour le poste de [Nom du poste] au sein de notre entreprise. Nous avons bien reçu votre candidature et tenons à vous exprimer notre gratitude pour l'intérêt que vous portez à [Nom de l'entreprise].

    Votre dossier est actuellement en cours d'évaluation, et nous ne manquerons pas de vous tenir informé(e) de la suite des événements dans les meilleurs délais.

    En attendant, nous vous souhaitons une excellente journée.

    Bien cordialement,
    """
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())

# def rank_cvs(jd, cvs):
#     ranked_cvs = []
#     pdf_document = fitz.open(stream=jd.read(), filetype="pdf")
#     text = ""
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text()

#     job_skill = get_skills(text)


#     vectorizer = TfidfVectorizer()
    
#     for cv in cvs:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(cv.read())
#             pdf_path = tmp_file.name

#         skills_cv = get_skills_cv(pdf_path)
#         # Convert skills_cv list to a single string
#         skills_cv = get_skills(skills_cv)
        
#         # Create a TF-IDF matrix with the job skills and CV skills
#         all_skills = [job_skill, skills_cv]
#         vectors = vectorizer.fit_transform(all_skills)
        
#         # Compute cosine similarity between job skills and CV skills
#         similarity_scores = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()
#         similarity_value = similarity_scores[0]
        
#         mail = get_mail(pdf_path)
#         ranked_cvs.append((cv.name, mail, similarity_value))
#         os.remove(pdf_path)
    
#     df_ranked = pd.DataFrame(ranked_cvs, columns=['CV Name', 'Email', 'Similarity'])
#     df_ranked = df_ranked.sort_values(by='Similarity', ascending=False)
#     return df_ranked
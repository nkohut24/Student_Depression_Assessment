import pandas as pd
import streamlit as st
import joblib
from google import genai
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
import os
import math
import requests
import streamlit as st
import random
from googleapiclient.discovery import build
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


GOOGLE_API_KEY = st.secrets["google"]["API_KEY"]

CSE_ID = st.secrets["google"]["CSE_ID"]

GENAI_API_KEY = st.secrets["genai"]["API_KEY"]
GENAI_SELECTED = random.choice(GENAI_API_KEY)
client = genai.Client(api_key=GENAI_SELECTED)

df = pd.read_csv('Student Depression Dataset.csv')
df = df.dropna()

df = pd.get_dummies(df, columns=['Gender', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'])
df = df.apply(pd.to_numeric, errors='coerce')  # converts everything to numeric, makes bad values NaN

x = df.drop(columns=['Depression', 'id', 'City', 'Profession', 'CGPA', 'Degree', 'Gender_Female','Sleep Duration_Others', 'Dietary Habits_Others','Have you ever had suicidal thoughts ?_No','Family History of Mental Illness_No'])
y = df['Depression']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                               test_size=0.2, # reserve 20% data for testing
                               random_state=365)
clf_lr = LogisticRegression(
    penalty= None, 
    max_iter=20000)
clf_lr.fit(x_train, y_train)

coef_lr = pd.DataFrame(clf_lr.coef_[0],index=x_train.columns,columns=['coefficient'])
print(coef_lr)

import statsmodels.api as sm
#logit_model=sm.Logit(y_train,sm.add_constant(x_train))
#result=logit_model.fit()
#print(result.summary())

y_predict = clf_lr.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))

# get coefficient of age
age_coef = coef_lr.loc['Age', 'coefficient']
# academic pressure
academic_coef = coef_lr.loc['Academic Pressure', 'coefficient']
# get coefficient of work pressure
work_coef = coef_lr.loc['Work Pressure', 'coefficient']
# get the coefficient of study satisfaction
study_coef = coef_lr.loc['Study Satisfaction', 'coefficient']
# get the coefficient of job satisifaction
job_coef = coef_lr.loc['Job Satisfaction', 'coefficient']
# work study hours
work_study_coef = coef_lr.loc['Work/Study Hours', 'coefficient']
# financial stress
financial_coef = coef_lr.loc['Financial Stress', 'coefficient']
# gender male
male_coef = coef_lr.loc['Gender_Male', 'coefficient']
# get the coefficient of sleep duration
sleep_lessthan5 = coef_lr.loc['Sleep Duration_Less than 5 hours', 'coefficient']
sleep_morethan8 = coef_lr.loc['Sleep Duration_More than 8 hours', 'coefficient']
sleep_coef_7to8 = coef_lr.loc['Sleep Duration_7-8 hours', 'coefficient']
sleep_coef_5to6 = coef_lr.loc['Sleep Duration_5-6 hours', 'coefficient']
# get the coefficient of family history of mental illness
family_coef = coef_lr.loc['Family History of Mental Illness_Yes', 'coefficient']
# dietary habits
diet_unhealthy_coef = coef_lr.loc['Dietary Habits_Unhealthy', 'coefficient']
diet_moderate_coef = coef_lr.loc['Dietary Habits_Moderate', 'coefficient']
diet_healthy_coef = coef_lr.loc['Dietary Habits_Healthy', 'coefficient']
# suicidal thoughtsS
suicidal_coef = coef_lr.loc['Have you ever had suicidal thoughts ?_Yes', 'coefficient']
# family history of mental illness
family_coef = coef_lr.loc['Family History of Mental Illness_Yes', 'coefficient']



# Initialize session state for tracking progress
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Assessment"

# Set up page configuration
st.set_page_config(page_title="Depression Risk Assessment", layout="wide")

primaryColor="#FF4B4B"

# set background color in streamlit
background_color = "#005249"
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    </style>
""", unsafe_allow_html=True)

# set font color in streamlit to white
st.markdown(f"""
    <style>
    .stApp {{
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# Create tabs for better organization
tabs = ["Home", "Assessment", "Visualizations", "Resources", "Track Progress", "Education"]
st.sidebar.title("Navigation")
st.session_state.current_tab = st.sidebar.radio("Go to", tabs)

def calculate_risk(inputs):
    risk_score = (
        inputs['Age'] * age_coef +
        inputs['Academic Pressure'] * academic_coef +
        inputs['Work Pressure'] * work_coef +
        inputs['Study Satisfaction'] * -study_coef +
        inputs['Job Satisfaction'] * job_coef +
        inputs['Work/Study Hours'] * work_study_coef +
        inputs['Financial Stress'] * financial_coef
    )

    # Gender adjustment
    if inputs['Gender'] == "Male":
        risk_score += male_coef

    # Sleep duration adjustment
    if inputs['Sleep Duration'] == "5-6 hours":
        risk_score += sleep_coef_5to6
    elif inputs['Sleep Duration'] == "7-8 hours":
        risk_score += sleep_coef_7to8
    elif inputs['Sleep Duration'] == "Less than 5 hours":
        risk_score += sleep_lessthan5
    elif inputs['Sleep Duration'] == "More than 8 hours":
        risk_score += sleep_morethan8

    # Dietary habits adjustment
    if inputs['Dietary Habits'] == "Healthy":
        risk_score += diet_healthy_coef
    elif inputs['Dietary Habits'] == "Unhealthy":
        risk_score += diet_unhealthy_coef
    elif inputs['Dietary Habits'] == "Moderate":
        risk_score += diet_moderate_coef

    # Suicidal thoughts and family history adjustment
    if inputs['Have you ever had suicidal thoughts ?'] == "Yes":
        risk_score += suicidal_coef
    if inputs['Family History of Mental Illness'] == "Yes":
        risk_score += family_coef

    risk_score = 1 / (1 + math.exp(-risk_score))  # Normalize score
    return risk_score

# Function to get risk category
def get_risk_category(score):
    if score < 0.4:
        return "low"
    elif score >= 0.4 and score < 0.7:
        return "moderate"
    else:
        return "high"

# Function to save user data
def save_assessment(inputs, score, category):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assessment = {
        "timestamp": timestamp,
        "inputs": inputs,
        "score": score,
        "category": category
    }
    st.session_state.history.append(assessment)
   
    # In a real app, you might want to save this to a file or database
    # For demo purposes, we'll just keep it in session state

df = pd.read_csv('Student Depression Dataset.csv')
df['risk_score'] = df.apply(lambda row: calculate_risk(row), axis=1)


# Function to get AI recommendations
def get_ai_recommendations(risk_category, specific_factors=None): 
    if specific_factors:
        prompt = f"""
        This user has a {risk_category} risk of depression.
        Their specific risk factors include: {', '.join(specific_factors)}.
        Please provide:
        1. 3-5 personalized coping strategies addressing these specific factors
        2. Resources they might find helpful
        3. A brief explanation of how these factors contribute to depression risk
        Keep your response compassionate, practical, and concise.
        """
    else:
        prompt = f"What can this user do about their {risk_category} risk of depression? Provide specific, actionable advice."
   
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

# ASSESSMENT TAB
if st.session_state.current_tab == "Assessment":
    st.title("Are you at Risk of Depression?")
    st.write("Please answer the following questions to find out if you are at risk of depression.")
   
    # Create sections for better organization
    st.header("Personal Information 🧍")
    age = st.number_input("What is your age?", min_value=0, max_value=100, value=25)
    gender = st.selectbox("What is your gender?", ["Male", "Female"])
   
    st.header("Work and Academic Life 🎒")
    academic_pressure = st.slider("How much academic pressure do you feel?", 0, 5, 0,
                                help="0 = No pressure, 5 = Extreme pressure")
    work_pressure = st.slider("How much work pressure do you feel?", 0, 5, 0,
                            help="0 = No pressure, 5 = Extreme pressure")
    study_satisfaction = st.slider("How satisfied are you with your studies?", 0, 5, 0,
                                help="0 = Not satisfied at all, 5 = Completely satisfied")
    job_satisfaction = st.slider("How satisfied are you with your job?", 0, 5, 0,
                                help="0 = Not satisfied at all, 5 = Completely satisfied")
    study_hours = st.number_input("How many hours do you study/work in a day?",
                                min_value=0, max_value=24, value=8)
    financial_stress = st.slider("How much financial stress do you feel?", 0, 5, 0,
                                help="0 = No stress, 5 = Extreme stress")
   
    st.header("Lifestyle Factors 🥗")
    sleep_options = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    sleep_duration = st.selectbox("How many hours do you sleep?", sleep_options)
    dietary_habits = st.selectbox("How would you describe your dietary habits?",
                                ["Healthy", "Moderate", "Unhealthy"])
   
    st.header("Mental Health History 📋")
    suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])
    family_history = st.radio("Is there a family history of mental illness?", ["Yes", "No"])
   
    # Add a button to submit the form
    if st.button("Calculate Risk"):
        # Collect all inputs
        inputs = {
            'Age': age,
            'Gender': gender,
            'Academic Pressure': academic_pressure,
            'Work Pressure': work_pressure,
            'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction,
            'Work/Study Hours': study_hours,
            'Financial Stress': financial_stress,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Family History of Mental Illness': family_history
        }
       
        # Calculate risk score
        risk_score = calculate_risk(inputs)
        risk_category = get_risk_category(risk_score)
       
        # Display results
        st.subheader("Your Results")
       
        # Create a visual indicator of risk
        col1, col2 = st.columns([1, 3])
        with col1:
            if risk_category == "low":
                st.markdown("### 🟢 Low Risk")
            elif risk_category == "moderate":
                st.markdown("### 🟠 Moderate Risk")
            else:
                st.markdown("### 🔴 High Risk")
       
        with col2:
            # Create a progress bar to visualize the risk score
            st.progress(max(0.0, min(risk_score, 1.0)))
            st.write(f"Risk Score: {risk_score:.2f}")
       
        # Identify specific risk factors
        risk_factors = []
        if academic_pressure > 3:
            risk_factors.append("high academic pressure")
        if work_pressure > 3:
            risk_factors.append("high work pressure")
        if financial_stress > 3:
            risk_factors.append("financial stress")
        if sleep_duration == "Less than 5 hours":
            risk_factors.append("insufficient sleep")
        if dietary_habits == "Unhealthy":
            risk_factors.append("poor dietary habits")
        if suicidal_thoughts == "Yes":
            risk_factors.append("suicidal thoughts")
            st.error("⚠️ If you're experiencing suicidal thoughts, please seek immediate help. Call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.")
       
        # Get personalized recommendations
        st.subheader("Personalized Recommendations")
        recommendations = get_ai_recommendations(risk_category, risk_factors)
        st.write(recommendations)
       
        # Save this assessment
        save_assessment(inputs, risk_score, risk_category)

# VISUALIZATIONS TAB
# add column with risk score to df

elif st.session_state.current_tab == "Visualizations":
    st.title("Understanding Depression Risk Factors")
   
    st.write("Explore the relationships between different factors and depression risk.")
   
    # Create visualization options
    viz_type = st.selectbox(
        "Select Visualization",
        ["Risk by Age Group", "Risk by Gender", "Risk by Sleep Duration", "Risk by Dietary Habits", "Risk by Suicidal Thoughts", "Risk by Family History of Mental Illness", "Risk Distribution", "Risk Factors Correlation"]
    )
   
    if viz_type == "Risk by Age Group":
        fig, ax = plt.subplots(figsize=(10, 6))
        df['age_group'] = pd.cut(df['Age'], bins=[15, 25, 35, 45, 55, 65])
        age_risk = df.groupby('age_group')['risk_score'].mean().reset_index()
        sns.barplot(x='age_group', y='risk_score', data=age_risk, ax=ax)
        ax.set_title("Average Depression Risk by Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
       
        st.write("""
        ### Key Insights:
        - Age is a significant factor in mental health risk
        - Mental health challenges can affect individuals of all ages
        - Young adults may face unique pressures from social media and academic expectations
        - Middle-aged adults may experience stress from career and family responsibilities
        - Older adults may face isolation and health-related challenges
        - Understanding age-related risk factors can help target interventions appropriately
        - Tailoring support to different age groups can enhance effectiveness
        - For example, young adults may benefit from peer support groups, while older adults may need more individualized care
        - Mental health is a lifelong journey, and support should be available at every stage
        """)
        # insert a picture of a family
        st.image("https://t4.ftcdn.net/jpg/02/58/17/33/360_F_258173340_wlszEvBmI5ubyRaQWF3JsIr57pjWasgN.jpg", width=400)
        

    elif viz_type == "Risk by Gender":
        fig, ax = plt.subplots(figsize=(10, 6))
        # make a risk by gender plot
        gender_risk = df.groupby("Gender")["risk_score"].mean().reset_index()
        sns.barplot(x="Gender", y="risk_score", data=gender_risk, ax=ax)
        ax.set_title("Average Depression Risk by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
        st.write("""
        ### Key Insights:
         - Depression can affect individuals of all genders, including males and females, regardless of age or background.
         ####@ Males:
         - Often exhibit symptoms such as irritability, anger, and risk-taking behaviors.
         - May be less likely to seek help due to societal expectations of masculinity.
         - Symptoms can be misinterpreted as signs of weakness or failure.
         ##### Females:
         - More likely to express feelings of sadness, anxiety, and emotional distress.
         - Tend to seek help more frequently and may be more open about their struggles.
         - Experience unique challenges related to hormonal changes and societal pressures.
         ##### Societal Influences:
         - Gender roles can shape how depression is perceived and expressed, leading to different coping mechanisms.
         - Males may feel pressured to appear strong and stoic, while females may face expectations to be nurturing and emotionally available.
         ##### Underdiagnosis and Misdiagnosis:
         - Males may be underdiagnosed due to atypical presentations of depression.
         - Females may be misdiagnosed or face stigma related to their emotional expressions.
         ##### Importance of Awareness:
         - Recognizing that depression does not discriminate is crucial for effective treatment and support.
         - Mental health resources should be accessible to everyone, encouraging open discussions about mental health.
         ##### Call to Action:
         - It’s vital for individuals of all genders to seek help and support when experiencing symptoms of depression.
         - Mental health awareness campaigns should promote understanding and empathy
         - Encourage individuals to reach out for help
         - Normalize conversations about mental health
         - Foster a supportive environment for all individuals
        """)
        st.image("https://www.socialnicole.com/wp-content/uploads/2015/02/youngsters.jpg", width=400)

    elif viz_type == "Risk by Suicidal Thoughts":
        fig, ax = plt.subplots(figsize=(10, 6))
        # make a risk by suicidal thoughts
        suicidal_risk = df.groupby("Have you ever had suicidal thoughts ?")["risk_score"].mean().reset_index()
        sns.barplot(x="Have you ever had suicidal thoughts ?", y="risk_score", data=suicidal_risk, ax=ax)
        ax.set_title("Average Depression Risk by Suicidal Thoughts")
        ax.set_xlabel("Suicidal Thoughts")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
        st.write("""
        ### Key Insights:
        - Individuals with suicidal thoughts have a significantly higher risk score
        - This highlights the importance of addressing suicidal ideation in mental health assessments
        - Early intervention and support can be crucial for those at risk
        - If you or someone you know is struggling, please seek help immediately
        - Resources are available, and you are not alone
        """)
        st.image("https://www.braintrainingaustralia.com/wp-content/uploads/2018/08/suicide.webp", width=400)

    elif viz_type == "Risk by Family History of Mental Illness":
        fig, ax = plt.subplots(figsize=(10, 6))
        # make a risk by family history of mental illness
        family_risk = df.groupby("Family History of Mental Illness")["risk_score"].mean().reset_index()
        sns.barplot(x="Family History of Mental Illness", y="risk_score", data=family_risk, ax=ax)
        ax.set_title("Average Depression Risk by Family History of Mental Illness")
        ax.set_xlabel("Family History of Mental Illness")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
        st.write("""
        ### Key Insights:
        - A family history of mental illness is associated with higher depression risk
        - Genetic and environmental factors can play a role in mental health
        - Understanding your family history can help in early detection and intervention
        """)
        st.image("https://www.pacificlife.com/content/pl-corp/insights-articles/supporting-multiple-generations/_jcr_content/root/responsivegrid/wpar/simplecolumns_copy_c/par1/image_660910426.img.jpg/1653586807451.jpg", width=400)

    elif viz_type == "Risk by Dietary Habits":
        df['Dietary Habits'] = pd.Categorical(df['Dietary Habits'],
                                                  categories=["Unhealthy", "Moderate", "Healthy", "Others"],
                                                  ordered=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        # make a risk by dietary habits plot
        dietary_risk = df.groupby("Dietary Habits")["risk_score"].mean().reset_index()
        sns.barplot(x="Dietary Habits", y="risk_score", data=dietary_risk, ax=ax)
        ax.set_title("Average Depression Risk by Dietary Habits")
        ax.set_xlabel("Dietary Habits")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
        st.write("""
        ### Key Insights:
        - Unhealthy dietary habits are associated with higher depression risk
        - A balanced diet can positively impact mental health
        - Nutrient-rich foods can support brain health and mood regulation
        - Consider consulting a nutritionist for personalized dietary advice
        - Small changes in diet can lead to significant improvements in mental well-being
        """)
        st.image("https://whatsupwellness.in/cdn/shop/articles/Healthy-Eating-2_1100x.jpg?v=1704693573", width=400)

    elif viz_type == "Risk by Sleep Duration":
        # order x axis by less than 5, 5-6, 7-8, more than 8
        df['Sleep Duration'] = pd.Categorical(df['Sleep Duration'],
                                                  categories=["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"],
                                                  ordered=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sleep_risk = df.groupby('Sleep Duration')['risk_score'].mean().reset_index() 
        sns.barplot(x='Sleep Duration', y='risk_score', data=sleep_risk, ax=ax)
        ax.set_title("Average Depression Risk by Sleep Duration")
        ax.set_xlabel("Sleep Duration")
        ax.set_ylabel("Average Risk Score")
        st.pyplot(fig)
        st.write("""
        ### Key Insights:
        - Sleep duration is a significant factor in mental health
        - Both insufficient and excessive sleep can be associated with depression
        - 7-8 hours of sleep is generally optimal for mental health
        - Poor sleep can exacerbate existing mental health issues
        - Prioritizing good sleep hygiene can improve overall well-being
        - Consider establishing a consistent sleep schedule
        - Limit screen time before bed
        - Create a relaxing bedtime routine
        - If sleep issues persist, consult a healthcare professional
        """)
        st.image("https://t3.ftcdn.net/jpg/04/09/83/06/360_F_409830661_pSWG1KUy0QW2W2rvT2W914k6l7t8YnpD.jpg", width=400)
       
    elif viz_type == "Risk Factors Correlation":
        # Convert Gender to numeric within the gender column
        # Data Encoding Functions
        def encode_categorical_columns(df):
        # Encoding categorical columns
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
            df['Sleep Duration'] = df['Sleep Duration'].map({
                'Less than 5 hours': 1,
                '5-6 hours': 2,
                '7-8 hours': 3,
                'More than 8 hours': 4
            })
        
            df['Dietary Habits'] = df['Dietary Habits'].map({
                'Healthy': 1,
                'Moderate': 2,
                'Unhealthy': 3
            })
        
            df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({
                'No': 0,
                'Yes': 1
            })
        
            df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({
                'No': 0,
                'Yes': 1
            })
            return df

        # Apply encoding
        df_encoded = encode_categorical_columns(df)

        # Correlation Analysis
        correlation_columns = [
            'Age', 'Gender', 'Academic Pressure', 'Work Pressure', 
            'Work/Study Hours', 'Financial Stress', 'Study Satisfaction', 
            'Sleep Duration', 'Dietary Habits', 
            'Have you ever had suicidal thoughts ?', 
            'Family History of Mental Illness'
        ]

        correlation_matrix = df_encoded[correlation_columns].corr()
        print("Correlation Matrix:\n", correlation_matrix)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Between Risk Factors")
        st.pyplot(fig)
            
        st.write("""
        ### Understanding Correlations:
        - Stronger correlations (closer to 1 or -1) indicate stronger relationships
        - Positive correlations mean factors increase together
        - Negative correlations mean as one factor increases, the other decreases
        """)
        
    elif viz_type == "Risk Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['risk_score'], kde=True, ax=ax)
        ax.set_title("Distribution of Depression Risk Scores")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Count")
    
        # Add lines for risk categories
        ax.axvline(x=0.4, color='orange', linestyle='--', label='Moderate Risk Threshold')
        ax.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
        ax.legend()
    
        st.pyplot(fig)

        st.write("""
        ### Key Insights:
        - The majority of individuals within the population fall within the low to moderate risk range
        - A smaller percentage may be at high risk, indicating a need for targeted interventions
        - Understanding the distribution can help individuals contextualize their own risk scores
        - This can also help in identifying trends or patterns in the population
        - For example, if a large number of individuals fall into the high-risk category, it may indicate a need for increased mental health resources or support in that population
        - Conversely, if most individuals fall into the low-risk category, it may suggest that current interventions and support systems are effective""")

# HOME TAB
elif st.session_state.current_tab == "Home":
    st.title("Welcome to the Depression Risk Assessment 🩺")
    st.write("This tool is designed to help you assess your risk of depression and provide personalized recommendations.")
    st.write("Please navigate through the tabs on the left to get started.")
    st.markdown("---")
    st.write("### How to Use This Tool:")
    st.write("- **Assessment**: Answer questions about your lifestyle and mental health.")
    st.write("- **Visualizations**: Explore data visualizations related to depression risk factors.")
    st.write("- **Resources**: Find helpful resources for mental health support.")
    st.write("- **Track Progress**: Monitor your mental health journey over time.")
    st.write("- **Education**: Learn more about depression, its risk factors, and warning signs.")
    st.image("https://wpvip.edutopia.org/wp-content/uploads/2023/09/feature_identify-emotions-feelings_all-grades_illustration_JakeOlimb_getty_1395986778.jpg?w=2880&quality=80&strip=all", width=300)
    st.markdown("---")
    st.write("### About the Developers")
    st.write("This tool was developed by Nancy Kohut, Dan Fox & Amanda Barry, students from Tulane University who are passionate about mental health awareness and support.")
    st.write("We believe that everyone deserves access to mental health resources and support.")
    st.markdown("---")
    st.write("### Disclaimer")
    st.write("This tool is not a substitute for professional medical advice, diagnosis, or treatment. If you are in crisis or need immediate help, please contact a mental health professional or call emergency services.")
    st.markdown("---")
    
# RESOURCES TAB
elif st.session_state.current_tab == "Resources":
    st.title("Mental Health Resources 📕")
   
    # Create categories of resources
    resource_type = st.radio(
        "Resource Type",
        ["Crisis Support", "Self-Help Tools", "Professional Help", "Campus Resources"]
    )
   
    if resource_type == "Crisis Support":
        st.header("Immediate Help")
        st.markdown("""
        ### 🚨 Crisis Resources
        - **988 Suicide & Crisis Lifeline**: Call or text 988 (24/7)
        - **Crisis Text Line**: Text HOME to 741741 (24/7)
        - **Emergency Services**: Call 911 if you or someone you know is in immediate danger
       
        ### 🌙 Warmlines (Non-Crisis Support)
        - **NAMI HelpLine**: 1-800-950-NAMI (6264)
        - **Peer Support Warmline**: [Find your state's warmline](https://warmline.org/warmdir.html)
        """)
       
    elif resource_type == "Self-Help Tools":
        st.header("Self-Help Strategies")
       
        self_help_category = st.selectbox(
            "Choose a category",
            ["Stress Management", "Sleep Improvement", "Mood Tracking", "Mindfulness"]
        )
       
        if self_help_category == "Stress Management":
            st.markdown("""
            ### Stress Reduction Techniques
           
            **Quick Stress Relievers (5 minutes or less):**
            - Deep breathing: 4-7-8 technique (inhale for 4, hold for 7, exhale for 8)
            - Progressive muscle relaxation: Tense and release each muscle group
            - Mindful observation: Focus completely on one object for 1 minute
           
            **Daily Practices:**
            - Schedule short breaks throughout your day
            - Limit caffeine and alcohol
            - Practice saying "no" to additional commitments
            - Create a prioritized to-do list
            """)
           
        elif self_help_category == "Sleep Improvement":
            st.markdown("""
            ### Better Sleep Habits
           
            **Evening Routine:**
            - Establish a consistent sleep schedule
            - Avoid screens 1 hour before bed
            - Create a cool, dark, quiet sleeping environment
            - Try a relaxation technique before bed
           
            **Daytime Habits for Better Sleep:**
            - Get natural sunlight during the day
            - Exercise regularly (but not right before bed)
            - Limit daytime naps to 20-30 minutes
            - Avoid caffeine after noon
            """)
           
        elif self_help_category == "Mood Tracking":
            st.markdown("""
            ### Tracking Your Mood
           
            **Benefits of Mood Tracking:**
            - Identifies patterns and triggers
            - Helps measure progress over time
            - Provides useful information for healthcare providers
           
            **Recommended Apps:**
            - Daylio
            - MoodKit
            - Moodpath
            - Youper
           
            **Simple Paper Method:**
            Rate your mood from 1-10 each morning and evening, noting major events or stressors
            """)
           
        elif self_help_category == "Mindfulness":
            st.markdown("""
            ### Mindfulness Practices
           
            **Beginning Mindfulness:**
            - Start with just 5 minutes daily
            - Focus on your breath, body sensations, or sounds around you
            - When your mind wanders, gently bring attention back
           
            **Free Resources:**
            - Insight Timer (app)
            - UCLA Mindful Awareness Research Center (free guided meditations)
            - Mindful.org (articles and practices)
           
            **Simple Grounding Exercise:**
            The 5-4-3-2-1 technique: Notice 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste
            """)
   
    elif resource_type == "Professional Help":
        st.header("Finding Professional Support")
       
        st.markdown("""
        ### Types of Mental Health Professionals
       
        **Psychiatrists**
        - Medical doctors who can prescribe medication
        - Specialize in diagnosis and treatment of mental health conditions
       
        **Psychologists**
        - Provide therapy and psychological testing
        - Cannot prescribe medication in most states
       
        **Licensed Counselors/Therapists**
        - Provide various types of therapy
        - May specialize in specific approaches or populations
       
        ### Finding Affordable Care
       
        - **Insurance**: Check your coverage for mental health services
        - **Community mental health centers**: Offer sliding scale fees
        - **University training clinics**: Reduced-cost services provided by supervised students
        - **Online therapy options**: Services like BetterHelp or Talkspace may be more affordable
        - **Open Path Psychotherapy Collective**: Network of therapists offering reduced rates
        """)

        st.markdown("### 📍 Find a Provider Near You")
        zip_code = st.text_input("Enter your ZIP code to find local providers")

        if zip_code:
            # Validate ZIP using geopy
            geolocator = Nominatim(user_agent="student-depression-support-app")
            location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})

            if location:
                st.success(f"Searching near: {location.address}")

                def google_search(query, api_key, cse_id, num=5):
                    service = build("customsearch", "v1", developerKey=api_key)
                    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
                    return res.get("items", [])

                query = f"Find a therapist near {location.address}"
                st.write(f"Looking for providers near **{location.address}**...")

                results = google_search(query, GOOGLE_API_KEY, CSE_ID)
                if results:
                    for result in results:
                        st.markdown(f"### [{result['title']}]({result['link']})")
                        st.write(result.get('snippet', ''))
                else:
                    st.warning("No results found. Try another ZIP code or be more specific.")
            else:
                st.error("Invalid ZIP code. Please enter a valid U.S. ZIP code.")

    elif resource_type == "Campus Resources":
        st.header("University Support Services🏫")
       
        st.markdown("""
        ### Common Campus Resources
       
        **Counseling Center**
        - Free or low-cost therapy sessions
        - Crisis intervention
        - Group therapy options
       
        **Student Health Services**
        - Medical care and sometimes psychiatric services
        - Health education and wellness programs
       
        **Academic Support**
        - Tutoring and academic coaching
        - Disability accommodations
       
        **Peer Support Programs**
        - Student-led support groups
        - Peer counseling services
       
        ### How to Access Services
       
        Most services can be accessed by:
        1. Visiting their office in person
        2. Calling their main number
        3. Emailing to schedule an appointment
        4. Using the university's online portal
       
        Many universities now offer same-day crisis appointments and after-hours support lines.
        """)
       
        university = st.text_input("Enter your university name to see specific resources")
        if university :

            def google_search(query, api_key, cse_id, num=5):
                service = build("customsearch", "v1", developerKey=api_key)
                res = service.cse().list(q=query, cx=cse_id, num=num).execute()
                return res.get("items", [])

            query = "Find campus mental health resources for " + university
            
            if query:
                results = google_search(query, GOOGLE_API_KEY, CSE_ID)
                for result in results:
                    st.markdown(f"### [{result['title']}]({result['link']})")
                    st.write(result['snippet'])
                    if results:
                        st.write(f"Finding campus mental health resources for {university}...")
                if not results:
                    st.warning("No results found. Try another university name or be more specific.")
                
        else:
            st.warning("Please enter your university name to find specific resources.")

        st.markdown("### Additional Resources")
        st.markdown("""
        - **[National Alliance on Mental Illness (NAMI)](https://nami.org/Home)**: Offers support groups, education, and advocacy
        - **[Mental Health America (MHA)](https://www.mhanational.org/)**: Provides resources and information on mental health
        - **[Substance Abuse and Mental Health Services Administration (SAMHSA)](https://www.samhsa.gov/)**: National helpline and treatment locator
        - **[American Psychological Association (APA)](https://www.apa.org/)**: Resources for finding a psychologist and understanding mental health
        - **[American Psychiatric Association (APA)](https://www.psychiatry.org/)**: Resources for finding a psychiatrist and understanding mental health
        - **[Psychology Today](https://www.psychologytoday.com/us)**: Directory of therapists, psychiatrists, and treatment centers
        - **[TherapyDen](https://www.therapyden.com/)**: Therapist directory with filters for various needs and preferences
        - **[GoodTherapy](https://www.goodtherapy.org/)**: Therapist directory with a focus on ethical practices
        - **[Open Path Psychotherapy Collective](https://openpathcollective.org/)**: Affordable therapy options
        - **[BetterHelp](https://www.betterhelp.com/)**: Online therapy platform with licensed therapists
        - **[Talkspace](https://www.talkspace.com/)**: Online therapy platform with licensed therapists
        - **[7 Cups](https://www.7cups.com/)**: Free online text chat with trained listeners
        - **[Crisis Text Line](https://www.crisistextline.org/)**: Free 24/7 text support for those in crisis
        """)

# TRACK PROGRESS TAB
elif st.session_state.current_tab == "Track Progress":
    st.title("Track Your Mental Health Journey")
   
    if not st.session_state.history:
        st.info("You haven't completed any assessments yet. Go to the Assessment tab to get started.")
    else:
        st.write(f"You have completed {len(st.session_state.history)} assessments.")
       
        # Display history in a table
        history_data = []
        for assessment in st.session_state.history:
            history_data.append({
                "Date": assessment["timestamp"],
                "Risk Score": f"{assessment['score']:.2f}",
                "Risk Level": assessment["category"].capitalize()
            })
       
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df)
       
        # Plot progress over time
        if len(st.session_state.history) > 1:
            st.subheader("Your Progress Over Time")
           
            fig, ax = plt.subplots(figsize=(10, 6))
            dates = [h["timestamp"] for h in st.session_state.history]
            scores = [h["score"] for h in st.session_state.history]
           
            ax.plot(dates, scores, marker='o', linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Risk Score")
            ax.set_title("Depression Risk Score Over Time")
           
            # Add threshold lines
            ax.axhline(y=0.4, color='orange', linestyle='--', label='Moderate Risk Threshold')
            ax.axhline(y=0.7, color='red', linestyle='--', label='High Risk Threshold')
            ax.legend()
           
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
           
            # Provide insights on progress
            first_score = st.session_state.history[0]["score"]
            latest_score = st.session_state.history[-1]["score"]
           
            if latest_score < first_score:
                st.success(f"Great progress! Your risk score has decreased by {first_score - latest_score:.2f} points since your first assessment.")
                st.write("Keep up the good work! Continue to practice self-care and reach out for support when needed.")
            elif latest_score > first_score:
                st.warning(f"Your risk score has increased by {latest_score - first_score:.2f} points since your first assessment.")
                st.write("This might be a good time to review your self-care strategies or consider reaching out for additional support.")
            else:
                st.info("Your risk score has remained stable since your first assessment.")
       
        
# EDUCATION TAB
elif st.session_state.current_tab == "Education":
    st.title("Learn About Depression")
   
    topics = ["What is Depression?", "Risk Factors", "Warning Signs"]
    selected_topic = st.selectbox("Select a topic to learn more", topics)
   
    if selected_topic == "What is Depression?":
        st.markdown("""
        ## Understanding Depression
       
        Depression (major depressive disorder) is a common and serious medical illness that negatively affects how you feel, think, and act. It is characterized by persistently low mood, loss of interest in activities, and various emotional and physical problems that can decrease a person's ability to function.
       
        ### Key Facts:
        - Depression affects approximately 280 million people worldwide
        - It's more than just feeling sad or going through a rough patch
        - It's a serious mental health condition that requires understanding and treatment
        - With proper diagnosis and treatment, the majority of people with depression can overcome it
       
        ### Common Symptoms:
        - Persistent sad, anxious, or "empty" mood
        - Loss of interest in hobbies and activities
        - Decreased energy, fatigue
        - Difficulty concentrating, remembering, making decisions
        - Sleep disturbances (insomnia or oversleeping)
        - Changes in appetite and weight
        - Thoughts of death or suicide
       
        
         """)
       
        # Embed a YouTube video
        st.subheader("Understanding Depression vs. Sadness")
        st.write("It's normal to feel sad sometimes, but depression is different. The video below explains the distinction:"
       )
        st.video("https://youtu.be/tNwRNmFT7-4?si=4iwZr7nwEJW55R3q")
       
    elif selected_topic == "Risk Factors":
        st.markdown("""
        ## Depression Risk Factors
       
        Depression can affect anyone—even a person who appears to live in relatively ideal circumstances. Several factors can play a role in depression:
       
        ### Biological Factors 👨‍👩‍👧
        - **Genetics**: Family history of depression increases risk
        - **Brain chemistry**: Imbalances in neurotransmitters like serotonin
        - **Physical health conditions**: Chronic illness, hormonal changes
       
        ### Psychological Factors 🧠
        - **Personality**: Tendencies toward pessimism, low self-esteem
        - **Early life experiences**: Childhood trauma or abuse
        - **Cognitive patterns**: Negative thinking styles
       
        ### Environmental Factors 🌺
        - **Chronic stress**: Work, academic, or financial pressure
        - **Major life changes**: Loss, relationship problems, moving
        - **Substance use**: Alcohol and drug misuse
        - **Social isolation**: Lack of support systems
       
        ### Student-Specific Risk Factors 🧑‍🎓
        - Academic pressure and competition
        - Financial stress and student debt
        - Sleep deprivation
        - Identity development challenges
        - Being away from established support networks
        """)
      
    elif selected_topic == "Warning Signs":
        st.markdown("""
        ## Warning Signs of Depression ⚠️
       
        Recognizing the warning signs of depression is crucial for early intervention. These signs may develop gradually and vary from person to person.
       
        ### Emotional Signs
        - Persistent sad, anxious, or "empty" feelings
        - Hopelessness or pessimism
        - Irritability or restlessness
        - Feelings of guilt, worthlessness, or helplessness
        - Loss of interest in previously enjoyed activities
       
        ### Physical Signs
        - Fatigue and decreased energy
        - Insomnia or oversleeping
        - Changes in appetite and weight
        - Aches, pains, headaches, or digestive problems without clear physical cause
        - Moving or talking more slowly
       
        ### Cognitive Signs
        - Difficulty concentrating or making decisions
        - Memory problems
        - Trouble thinking clearly
        - Thoughts of death or suicide
       
        ### Behavioral Signs
        - Withdrawing from friends and activities
        - Neglecting responsibilities
        - Increased use of alcohol or drugs
        - Decreased academic or work performance
        """)
       
        # Interactive self-check tool
        st.subheader("Depression Warning Signs Self-Check")
        st.write("This is not a diagnostic tool, but can help you recognize potential warning signs.")
       
        warning_signs = [
            "Persistent feelings of sadness or emptiness",
            "Loss of interest in activities once enjoyed",
            "Significant changes in appetite or weight",
            "Sleep disturbances (too much or too little)",
            "Fatigue or loss of energy",
            "Difficulty concentrating or making decisions",
            "Feelings of worthlessness or excessive guilt",
            "Thoughts of death or suicide"
        ]
       
        user_signs = []
        for sign in warning_signs:
            if st.checkbox(sign):
                user_signs.append(sign)
       
        if st.button("Check Results"):
            if len(user_signs) >= 5:
                st.warning("You've selected multiple warning signs of depression. Consider speaking with a mental health professional.")
            elif len(user_signs) >= 2:
                st.info("You've selected some warning signs that might warrant attention. Consider monitoring these feelings and reaching out for support if they persist.")
            else:
                st.success("You've selected few or no warning signs. Continue monitoring your mental health and practice self-care.")
           
            st.write("Remember: This is not a diagnosis. Only a qualified healthcare provider can diagnose depression.")
       
    # Similar content for other educational topics

# Add a footer with resources
st.sidebar.markdown("---")
st.sidebar.markdown("### Emergency Resources")
st.sidebar.markdown("**Crisis Text Line**: Text HOME to 741741")
st.sidebar.markdown("**National Suicide Prevention Lifeline**: 988")

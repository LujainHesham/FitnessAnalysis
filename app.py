"""
Fitness Workout Analysis: Gym vs Home Workouts
YouTube Comments Sentiment & Motivation Analysis
Author: Your Name
Project: Data Science Portfolio
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

import nltk
import os
import sys

# Download NLTK data to a persistent location
NLTK_DATA_PATH = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download required NLTK packages
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir=NLTK_DATA_PATH, quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH, quiet=True)

# Page configuration
st.set_page_config(
    page_title="Fitness Workout Analysis",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2C3E50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# Load and cache data
@st.cache_data
def load_data():
    """Load the validated fitness comments dataset"""
    try:
        df = pd.read_csv('fitness_comments_validated.csv')
        return df
    except FileNotFoundError:
        # If validated file doesn't exist, try enhanced
        try:
            df = pd.read_csv('fitness_comments_enhanced.csv')
            return df
        except FileNotFoundError:
            st.error("Data files not found. Please ensure CSV files are in the same directory.")
            return None

# Define motivation keywords (same as in notebook)
motivation_keywords = {
    'external': ['trainer', 'coach', 'class', 'group', 'spotter', 'gym bro', 'gym buddy',
                 'community', 'atmosphere', 'energy', 'people around', 'others', 'friends',
                 'together', 'team', 'competition', 'instructor', 'teacher', 'partner',
                 'social', 'crowd', 'everyone', 'them', 'they', 'audience'],
    'internal': ['discipline', 'self', 'myself', 'habit', 'routine', 'push myself',
                 'accountability', 'dedication', 'willpower', 'mental', 'mind', 'goal',
                 'purpose', 'reason', 'because i', 'want to', 'need to', 'must',
                 'determination', 'focus', 'mindset', 'personal', 'own', 'alone',
                 'self-motivated', 'self-discipline', 'inner', 'drive']
}

def classify_motivation(text):
    if pd.isna(text):
        return 'unknown'
    text_lower = str(text).lower()
    scores = {cat: sum(1 for kw in keywords if kw in text_lower) 
              for cat, keywords in motivation_keywords.items()}
    if sum(scores.values()) == 0:
        return 'not_detected'
    primary = max(scores, key=scores.get)
    max_score = scores[primary]
    tied = [cat for cat, score in scores.items() if score == max_score and score > 0]
    return 'mixed' if len(tied) > 1 else primary

# Sidebar navigation
st.sidebar.markdown("# 🏋️‍♂️ Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 Data Overview",
    "😊 Sentiment Analysis",
    "🎯 Motivation Analysis",
    "🔧 Equipment Analysis",
    "✅ Satisfaction Analysis",
    "📈 Validation Results",
    "🔍 Explore Comments"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Key Insights Quick Access")
st.sidebar.info(
    "💡 **Pro Tip:** Use the expanders and tooltips throughout "
    "the app to understand how each metric is calculated."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.markdown(
    "This analysis compares **Gym vs Home workout reviews** "
    "from YouTube comments, analyzing sentiment, motivation sources, "
    "equipment mentions, and satisfaction levels."
)

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
    </style>
    """, unsafe_allow_html=True)

# Load data
df = load_data()

if df is None:
    st.stop()

# ==================== OVERVIEW PAGE ====================
if page == "🏠 Overview":
    st.markdown('<div class="main-header">💪 Fitness Workout Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Gym vs Home Workouts: YouTube Comments Analysis</div>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Comments", f"{len(df):,}", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        gym_count = len(df[df['workout_type']=='gym'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Gym Comments", f"{gym_count:,}", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        home_count = len(df[df['workout_type']=='home'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Home Comments", f"{home_count:,}", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        avg_sentiment = df['sentiment_compound'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About this project
    with st.expander("📖 About This Project", expanded=True):
        st.markdown("""
        ### Problem Statement
        With the rise of home fitness (Peloton, Apple Fitness+, YouTube workouts) and traditional gym memberships, 
        **what drives people's satisfaction and motivation** in each environment?
        
        ### Approach
        1. **Data Collection**: Scraped 500+ YouTube comments from fitness videos using YouTube API
        2. **Text Cleaning**: Removed noise, URLs, special characters while preserving sentiment indicators
        3. **Sentiment Analysis**: Used VADER for compound sentiment scoring
        4. **Motivation Classification**: Categorized comments as externally or internally motivated
        5. **Equipment Analysis**: Identified gym-specific vs home-specific equipment mentions
        6. **Satisfaction Scoring**: Calculated net satisfaction (positive - negative indicators)
        7. **Validation**: Custom validation rules for data quality
        
        ### Key Findings
        - **Motivation**: Gym-goers are more externally motivated (trainers, community), while home exercisers rely on internal discipline
        - **Satisfaction**: Both groups show positive sentiment, but for different reasons
        - **Equipment**: Gym comments mention 3x more equipment items than home comments
        """)
    
    # Dataset composition pie chart
    st.markdown('<div class="sub-header">📊 Dataset Composition</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        workout_counts = df['workout_type'].value_counts()
        colors = ['#3498db', '#e67e22']
        ax.pie(workout_counts.values, labels=workout_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, explode=(0.05, 0.05))
        ax.set_title('Gym vs Home Comments Distribution', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>💡 How to Interpret This</h4>
        <p>The dataset contains YouTube comments from videos about gym workouts and home workouts. 
        Each comment was analyzed for:</p>
        <ul>
            <li><strong>Sentiment</strong> - Positive, negative, or neutral tone</li>
            <li><strong>Motivation Type</strong> - External (others/community) vs Internal (self-discipline)</li>
            <li><strong>Equipment Mentions</strong> - Specific gear referenced</li>
            <li><strong>Satisfaction Level</strong> - Overall expressed satisfaction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== DATA OVERVIEW ====================
elif page == "📊 Data Overview":
    st.markdown('<div class="main-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sample Comments")
        sample_size = st.slider("Number of samples", 5, 20, 10)
        st.dataframe(df[['workout_type', 'text', 'sentiment_category']].head(sample_size))
    
    with col2:
        st.subheader("Word Count Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column='word_count', by='workout_type', ax=ax)
        ax.set_title('Review Length Distribution by Workout Type')
        ax.set_ylabel('Word Count')
        ax.set_ylim(0, 150)
        st.pyplot(fig)
    
    st.subheader("Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Word Count (Gym)", f"{df[df['workout_type']=='gym']['word_count'].mean():.1f}")
    with col2:
        st.metric("Avg Word Count (Home)", f"{df[df['workout_type']=='home']['word_count'].mean():.1f}")
    with col3:
        st.metric("Validation Pass Rate", f"{df['validation_passed'].mean()*100:.1f}%")

# ==================== SENTIMENT ANALYSIS ====================
elif page == "😊 Sentiment Analysis":
    st.markdown('<div class="main-header">😊 Sentiment Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How Sentiment is Calculated", expanded=False):
        st.markdown("""
        **VADER (Valence Aware Dictionary and sEntiment Reasoner)** is used for sentiment analysis:
        - **Compound Score**: Ranges from -1 (most negative) to +1 (most positive)
        - **Positive**: Score ≥ 0.05
        - **Neutral**: -0.05 < Score < 0.05
        - **Negative**: Score ≤ -0.05
        """)
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_by_type = pd.crosstab(df['workout_type'], df['sentiment_category'], normalize='index')
        sentiment_by_type.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#95a5a6', '#e74c3c'])
        ax.set_title('Sentiment Distribution by Workout Type', fontweight='bold')
        ax.set_xlabel('Workout Type')
        ax.set_ylabel('Proportion')
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)
    
    with col2:
        # Violin plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df, x='workout_type', y='sentiment_compound', ax=ax, palette=['#3498db', '#e67e22'])
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Sentiment Score Distribution', fontweight='bold')
        ax.set_ylabel('Compound Sentiment Score')
        st.pyplot(fig)
    
    # Interactive Plotly chart
    st.subheader("Interactive Sentiment Explorer")
    fig = px.histogram(df, x='sentiment_compound', color='workout_type',
                       nbins=30, title='Sentiment Score Distribution',
                       labels={'sentiment_compound': 'Sentiment Score', 'count': 'Number of Comments'},
                       barmode='overlay')
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# ==================== MOTIVATION ANALYSIS ====================
elif page == "🎯 Motivation Analysis":
    st.markdown('<div class="main-header">🎯 Motivation Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How Motivation is Classified", expanded=False):
        st.markdown("""
        **Motivation Types:**
        - **External**: Driven by trainers, coaches, community, gym atmosphere, social factors
        - **Internal**: Driven by self-discipline, personal goals, willpower, habits
        - **Mixed**: Contains both external and internal motivation keywords
        """)
    
    # Apply motivation classification if not already present
    if 'motivation_type' not in df.columns:
        df['motivation_type'] = df['cleaned_text'].apply(classify_motivation)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        motivation_crosstab = pd.crosstab(df['workout_type'], df['motivation_type'], normalize='index')
        motivation_crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title('Motivation Source by Workout Type', fontweight='bold')
        ax.set_xlabel('Workout Type')
        ax.set_ylabel('Proportion')
        ax.legend(title='Motivation Type', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)
    
    with col2:
        st.subheader("Motivation Strength")
        fig, ax = plt.subplots(figsize=(8, 6))
        motivation_strength = df.groupby('workout_type')['motivation_strength'].mean()
        ax.bar(motivation_strength.index, motivation_strength.values, color=['#3498db', '#e67e22'])
        ax.set_title('Average Motivation Keywords per Comment', fontweight='bold')
        ax.set_ylabel('Number of Motivation Keywords')
        st.pyplot(fig)
    
    st.info(f"""
    **Key Insight:** 
    - Gym comments are {motivation_crosstab.loc['gym', 'external']*100:.1f}% externally motivated
    - Home comments are {motivation_crosstab.loc['home', 'internal']*100:.1f}% internally motivated
    """)

# ==================== EQUIPMENT ANALYSIS ====================
elif page == "🔧 Equipment Analysis":
    st.markdown('<div class="main-header">🔧 Equipment Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Equipment Categories", expanded=False):
        st.markdown("""
        **Equipment Categories Tracked:**
        - **Gym Specific**: Barbells, machines, cable systems, squat racks, etc.
        - **Home Specific**: Resistance bands, yoga mats, pull-up bars, bodyweight
        - **Free Weights**: Dumbbells, plates, kettlebells
        - **Cardio**: Treadmills, bikes, ellipticals, rowers
        """)
    
    # Equipment mentions by category
    equipment_categories = ['gym_specific', 'home_specific', 'free_weights', 'cardio']
    gym_means = [df[df['workout_type']=='gym'][f'equipment_{cat}'].mean() for cat in equipment_categories]
    home_means = [df[df['workout_type']=='home'][f'equipment_{cat}'].mean() for cat in equipment_categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(equipment_categories))
    width = 0.35
    ax.bar(x - width/2, gym_means, width, label='Gym', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, home_means, width, label='Home', color='#e67e22', alpha=0.8)
    ax.set_xlabel('Equipment Category')
    ax.set_ylabel('Average Mentions per Comment')
    ax.set_title('Equipment Mentions by Category', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Gym Specific', 'Home Specific', 'Free Weights', 'Cardio'])
    ax.legend()
    st.pyplot(fig)
    
    st.metric("Total Equipment Mentions (Gym)", f"{df[df['workout_type']=='gym']['total_equipment_mentions'].mean():.1f} avg")
    st.metric("Total Equipment Mentions (Home)", f"{df[df['workout_type']=='home']['total_equipment_mentions'].mean():.1f} avg")

# ==================== SATISFACTION ANALYSIS ====================
elif page == "✅ Satisfaction Analysis":
    st.markdown('<div class="main-header">✅ Satisfaction Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How Satisfaction is Calculated", expanded=False):
        st.markdown("""
        **Net Satisfaction Score** = Positive indicators - Negative indicators
        
        **Satisfaction Levels:**
        - Very Satisfied: Score ≥ 2
        - Satisfied: Score = 1
        - Neutral: Score = 0
        - Dissatisfied: Score = -1
        - Very Dissatisfied: Score ≤ -2
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        satisfaction_by_type = pd.crosstab(df['workout_type'], df['satisfaction_level'], normalize='index')
        satisfaction_by_type.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn')
        ax.set_title('Satisfaction Levels by Workout Type', fontweight='bold')
        ax.set_xlabel('Workout Type')
        ax.set_ylabel('Proportion')
        ax.legend(title='Satisfaction', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        satisfaction_scores = df.groupby('workout_type')['satisfaction_net'].mean()
        ax.bar(satisfaction_scores.index, satisfaction_scores.values, color=['#3498db', '#e67e22'])
        ax.set_title('Net Satisfaction Score', fontweight='bold')
        ax.set_ylabel('Net Satisfaction (Positive - Negative)')
        st.pyplot(fig)
    
    st.success(f"""
    **Overall Satisfaction:**
    - Gym: Net satisfaction of {df[df['workout_type']=='gym']['satisfaction_net'].mean():.2f}
    - Home: Net satisfaction of {df[df['workout_type']=='home']['satisfaction_net'].mean():.2f}
    """)

# ==================== VALIDATION RESULTS ====================
elif page == "📈 Validation Results":
    st.markdown('<div class="main-header">📈 TrustGuard Validation Results</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ About TrustGuard Validation", expanded=False):
        st.markdown("""
        **Validation Rules Applied:**
        1. **Text Length**: 5-500 words (meaningful content)
        2. **Motivation Content**: Contains motivation keywords
        3. **Equipment Mentions**: Equipment type matches workout type
        4. **Satisfaction Expression**: Contains satisfaction indicators
        5. **Spam Detection**: No promotional content or URLs
        """)
    
    # Validation pass rates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Validation Pass", f"{df['validation_passed'].mean()*100:.1f}%")
    with col2:
        st.metric("Gym Pass Rate", f"{df[df['workout_type']=='gym']['validation_passed'].mean()*100:.1f}%")
    with col3:
        st.metric("Home Pass Rate", f"{df[df['workout_type']=='home']['validation_passed'].mean()*100:.1f}%")
    
    # Validation by check
    checks = ['length', 'motivation', 'equipment', 'satisfaction', 'spam']
    gym_rates = [df[df['workout_type']=='gym'][f'valid_{check}'].mean()*100 for check in checks]
    home_rates = [df[df['workout_type']=='home'][f'valid_{check}'].mean()*100 for check in checks]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(checks))
    width = 0.35
    ax.bar(x - width/2, gym_rates, width, label='Gym', color='#3498db')
    ax.bar(x + width/2, home_rates, width, label='Home', color='#e67e22')
    ax.set_xlabel('Validation Check')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Validation Pass Rates by Check', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Length', 'Motivation', 'Equipment', 'Satisfaction', 'Spam'])
    ax.legend()
    st.pyplot(fig)

# ==================== EXPLORE COMMENTS ====================
elif page == "🔍 Explore Comments":
    st.markdown('<div class="main-header">🔍 Explore Comments</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        workout_filter = st.multiselect("Workout Type", options=df['workout_type'].unique(), default=df['workout_type'].unique())
    with col2:
        sentiment_filter = st.multiselect("Sentiment", options=df['sentiment_category'].unique(), default=df['sentiment_category'].unique())
    with col3:
        min_words = st.slider("Minimum Word Count", 1, 50, 5)
    
    # Apply filters
    filtered_df = df[
        (df['workout_type'].isin(workout_filter)) &
        (df['sentiment_category'].isin(sentiment_filter)) &
        (df['word_count'] >= min_words)
    ]
    
    st.write(f"Showing {len(filtered_df)} comments")
    
    # Display comments with expanders
    for idx, row in filtered_df.head(50).iterrows():
        with st.expander(f"💬 {row['workout_type'].upper()} | Sentiment: {row['sentiment_category']} | Words: {row['word_count']}"):
            st.write(f"**Comment:** {row['text']}")
            st.write(f"**Cleaned:** {row['cleaned_text']}")
            st.write(f"**Sentiment Score:** {row['sentiment_compound']:.3f}")
            if 'motivation_type' in row:
                st.write(f"**Motivation Type:** {row.get('motivation_type', 'N/A')}")
    
    # Download option
    if st.button("📥 Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_comments.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit | Data sourced from YouTube API | Fitness Workout Analysis Project"
    "</div>",
    unsafe_allow_html=True
)
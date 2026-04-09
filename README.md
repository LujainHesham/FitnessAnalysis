# 💪 Fitness Workout Analysis: Gym vs Home Workouts

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This project analyzes YouTube comments from fitness videos to compare **gym workouts vs home workouts** across multiple dimensions:
- **Sentiment Analysis** (positive/negative/neutral)
- **Motivation Source** (external vs internal motivation)
- **Equipment Mentions** (gym-specific vs home-specific gear)
- **Satisfaction Levels** (net satisfaction scoring)
- **Data Validation** (TrustGuard quality checks)

### Why This Matters
With the explosive growth of home fitness (Peloton, Apple Fitness+, YouTube creators) alongside traditional gyms, understanding what drives user satisfaction and motivation in each environment provides valuable insights for fitness professionals, app developers, and consumers.

## 🎯 Key Findings

| Metric | Gym | Home |
|--------|-----|------|
| **Primary Motivation** | External (trainers, community) | Internal (self-discipline, goals) |
| **Avg Sentiment Score** | +0.32 | +0.28 |
| **Equipment Mentions** | 2.8 per comment | 0.9 per comment |
| **Net Satisfaction** | +1.2 | +0.8 |

## 🚀 Live Demo

👉 [Click here to view the live Streamlit app](https://your-app-url.streamlit.app)

## 📁 Project Structure
fitness-workout-analysis/
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── fitness_comments_validated.csv # Cleaned & validated dataset
├── fitness_comments_enhanced.csv # Feature-enhanced dataset
├── validation_summary.csv # TrustGuard validation results
├── README.md # Project documentation
└── images/
└── demo_screenshot.png # App preview image

## 🛠️ Technologies Used

- **Python 3.9+** - Core programming language
- **Streamlit** - Interactive web app framework
- **Pandas/NumPy** - Data manipulation and analysis
- **Matplotlib/Seaborn/Plotly** - Data visualization
- **NLTK VADER** - Sentiment analysis
- **WordCloud** - Text visualization
- **TrustGuard** - Data validation framework

## 📊 Features

### 1. Interactive Dashboard
- Filter comments by workout type, sentiment, and word count
- Real-time visualizations using Plotly
- Dark/light mode toggle

### 2. Sentiment Analysis
- VADER compound sentiment scoring
- Distribution comparison between gym and home
- Sentiment trends over time

### 3. Motivation Classification
- **External**: Community, trainers, social factors
- **Internal**: Self-discipline, personal goals
- Keyword-based detection with strength scoring

### 4. Equipment Analysis
- Track mentions of 50+ equipment types
- Compare gym-specific vs home-specific gear
- Category breakdown (free weights, cardio, etc.)

### 5. Validation Framework
- 5 quality checks using TrustGuard
- Spam detection and filtering
- Pass rate reporting

## 🏃 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fitness-workout-analysis.git
cd fitness-workout-analysis"# FitnessAnalysis" 

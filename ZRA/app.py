import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime
import random
from textblob import TextBlob
from collections import defaultdict
import json

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Aspect keywords dictionary
ASPECTS = {
    'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'menu', 'cuisine',
             'biryani', 'pasta', 'pizza', 'burger', 'quality', 'portion'],
    'service': ['service', 'staff', 'waiter', 'waitress', 'manager',
                'delivery', 'time', 'speed', 'attitude', 'behavior'],
    'ambience': ['ambience', 'atmosphere', 'decor', 'lighting', 'music',
                 'noise', 'cleanliness', 'interior', 'space'],
    'price': ['price', 'cost', 'bill', 'value', 'expensive', 'cheap',
              'affordable', 'worth', 'budget']
}

# Dish keywords for analysis
DISHES = {
    'pasta': ['pasta', 'spaghetti', 'fettuccine'],
    'biryani': ['biryani', 'dum biryani'],
    'pizza': ['pizza', 'margherita'],
    'burger': ['burger', 'cheeseburger']
}

def preprocess_review(review_text):
    """Clean and preprocess a single review string."""
    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    return review

def load_or_train_model(df):
    """Load existing model or train new one"""
    model_path = 'naive_bayes_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            return model, tfidf_vectorizer
        except:
            pass
    
    # Train new model
    df['Review'] = df['Review'].apply(preprocess_review)
    tfidf_vectorizer = TfidfVectorizer(max_features=1500)
    X = tfidf_vectorizer.fit_transform(df['Review']).toarray()
    y = df['Liked'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    return model, tfidf_vectorizer

def analyze_aspect_sentiment(review, aspects):
    """Analyze sentiment for each aspect in a review"""
    try:
        # Simple keyword-based aspect analysis
        review_lower = review.lower()
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            matches = [keyword for keyword in keywords if keyword in review_lower]
            if matches:
                # Simple sentiment analysis using TextBlob
                polarity = TextBlob(review).sentiment.polarity
                aspect_sentiments[aspect] = polarity
        
        return aspect_sentiments
    except Exception as e:
        print(f"Error in aspect sentiment analysis: {e}")
        return {}

def process_reviews(df, aspects):
    """Process all reviews to extract aspect sentiments"""
    results = []
    
    for _, row in df.iterrows():
        review = row['Review']
        overall_sentiment = row['Liked']
        aspect_sentiments = analyze_aspect_sentiment(review, aspects)
        
        for aspect, polarity in aspect_sentiments.items():
            results.append({
                'review': review,
                'overall_sentiment': overall_sentiment,
                'aspect': aspect,
                'polarity': polarity,
                'sentiment_label': 1 if polarity > 0 else 0
            })
    
    return pd.DataFrame(results)

def generate_temporal_data(aspect_df):
    """Add simulated temporal data for trend analysis"""
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    date_range = (end_date - start_date).days
    aspect_df['date'] = [start_date + pd.Timedelta(days=random.randint(0, date_range))
                        for _ in range(len(aspect_df))]
    
    aspect_df['time_of_day'] = random.choices(
        ['morning', 'afternoon', 'evening', 'night'],
        weights=[0.2, 0.3, 0.4, 0.1],
        k=len(aspect_df)
    )
    
    return aspect_df

def analyze_dish_complaints(df, dishes):
    """Analyze dish complaints"""
    dish_complaints = []
    
    for dish, keywords in dishes.items():
        dish_reviews = df[df['review'].str.contains('|'.join(keywords), case=False)]
        
        if len(dish_reviews) > 0:
            total_mentions = len(dish_reviews)
            complaints = len(dish_reviews[dish_reviews['sentiment_label'] == 0])
            complaint_pct = (complaints / total_mentions) * 100
            
            dish_complaints.append({
                'dish': dish,
                'total_mentions': total_mentions,
                'complaints': complaints,
                'complaint_pct': complaint_pct,
                'examples': dish_reviews[dish_reviews['sentiment_label'] == 0]['review'].head(2).tolist()
            })
    
    return sorted(dish_complaints, key=lambda x: x['complaint_pct'], reverse=True)

def generate_competitor_data(your_metrics):
    """Simulate competitor data for benchmarking"""
    competitors = ['Competitor A', 'Competitor B', 'Competitor C']
    competitor_data = {}
    
    for competitor in competitors:
        noise = np.random.normal(0, 0.1, len(your_metrics))
        competitor_metrics = your_metrics * (1 + noise)
        competitor_metrics = competitor_metrics.clip(0, 1)
        competitor_data[competitor] = competitor_metrics
    
    comparison_df = pd.DataFrame(competitor_data)
    comparison_df['Your Restaurant'] = your_metrics
    return comparison_df

def generate_ai_suggestions(comparison_df):
    """Generate AI-powered improvement suggestions based on competitor benchmarking"""
    suggestions = []
    competitors = ['Competitor A', 'Competitor B', 'Competitor C']
    
    for aspect in comparison_df.index:
        your_score = comparison_df.loc[aspect, 'Your Restaurant']
        best_competitor = comparison_df.loc[aspect, competitors].idxmax()
        best_score = comparison_df.loc[aspect, best_competitor]
        
        # Convert to percentage
        your_score_pct = your_score * 100
        best_score_pct = best_score * 100
        
        # Generate specific recommendations based on aspect
        if aspect == 'food':
            recommendation = f"Improve food quality (your score: {your_score_pct:.0f}% vs {best_competitor}'s {best_score_pct:.0f}%). Consider: chef training, better ingredients."
        elif aspect == 'service':
            recommendation = f"Enhance service (your score: {your_score_pct:.0f}% vs {best_competitor}'s {best_score_pct:.0f}%). Consider: staff training, faster service."
        elif aspect == 'price':
            recommendation = f"Adjust pricing (your score: {your_score_pct:.0f}% vs {best_competitor}'s {best_score_pct:.0f}%). Consider: value meals, happy hours."
        elif aspect == 'ambience':
            recommendation = f"Improve ambience (your score: {your_score_pct:.0f}% vs {best_competitor}'s {best_score_pct:.0f}%). Consider: better lighting, music, decor."
        else:
            recommendation = f"Improve {aspect} (your score: {your_score_pct:.0f}% vs {best_competitor}'s {best_score_pct:.0f}%)"
        
        suggestions.append({
            'aspect': aspect,
            'your_score': your_score_pct,
            'best_competitor': best_competitor,
            'best_score': best_score_pct,
            'recommendation': recommendation
        })
    
    return suggestions

def create_chart_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Properly close the figure
    img.close()  # Close the BytesIO object
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process the data
            df = pd.read_csv(filepath)
            
            # Check if required columns exist
            if 'Review' not in df.columns or 'Liked' not in df.columns:
                return jsonify({'error': 'CSV must contain "Review" and "Liked" columns'}), 400
            
            # Load or train model
            model, tfidf_vectorizer = load_or_train_model(df.copy())
            
            # Basic analysis
            sentiment_dist = df['Liked'].value_counts()
            df['review_length'] = df['Review'].apply(len)
            
            # Aspect analysis
            aspect_df = process_reviews(df, ASPECTS)
            aspect_df = generate_temporal_data(aspect_df)
            
            # Dish complaint analysis
            dish_complaints = analyze_dish_complaints(aspect_df, DISHES)
            
            # Competitor benchmarking
            your_metrics = aspect_df.groupby('aspect')['polarity'].mean()
            comparison_df = generate_competitor_data(your_metrics)
            
            # Generate AI suggestions
            ai_suggestions = generate_ai_suggestions(comparison_df)
            
            # Create visualizations
            charts = {}
            
            try:
                # 1. Sentiment Distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                sentiment_dist.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
                ax.set_title('Distribution of Positive/Negative Reviews')
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Count')
                ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
                charts['sentiment_dist'] = create_chart_base64(fig)
                
                # 2. Review Lengths
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['review_length'], bins=30, color='#95a5a6', alpha=0.7)
                ax.set_title('Distribution of Review Lengths')
                ax.set_xlabel('Character Count')
                ax.set_ylabel('Frequency')
                charts['review_lengths'] = create_chart_base64(fig)
                
                # 3. Aspect Frequency (only if aspect analysis exists)
                if not aspect_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    aspect_counts = aspect_df['aspect'].value_counts()
                    aspect_counts.plot(kind='bar', ax=ax, color='#3498db')
                    ax.set_title('Frequency of Aspects Mentioned')
                    ax.set_xlabel('Aspect')
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    charts['aspect_frequency'] = create_chart_base64(fig)
                    
                    # 4. Aspect Sentiment Percentages
                    fig, ax = plt.subplots(figsize=(8, 6))
                    aspect_sentiment = aspect_df.groupby('aspect')['sentiment_label'].value_counts(normalize=True).unstack()
                    aspect_sentiment.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#27ae60'])
                    ax.set_title('Positive/Negative Sentiment Proportions by Aspect')
                    ax.set_ylabel('Proportion')
                    ax.tick_params(axis='x', rotation=45)
                    charts['aspect_sentiment'] = create_chart_base64(fig)
                    
                    # 5. Competitor Benchmarking
                    fig, ax = plt.subplots(figsize=(10, 6))
                    comparison_df.plot(kind='bar', ax=ax)
                    ax.set_title('Aspect Sentiment Benchmarking')
                    ax.set_ylabel('Positive Sentiment Ratio')
                    ax.tick_params(axis='x', rotation=45)
                    charts['benchmarking'] = create_chart_base64(fig)
                
                # 6. Dish Complaints
                if dish_complaints:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    dishes = [x['dish'] for x in dish_complaints]
                    percentages = [x['complaint_pct'] for x in dish_complaints]
                    colors = ['#ff6b6b' if pct > 30 else '#48dbfb' for pct in percentages]
                    
                    bars = ax.barh(dishes, percentages, color=colors)
                    ax.set_title('Dish Complaint Percentage')
                    ax.set_xlabel('Complaint Percentage')
                    ax.set_xlim(0, 100)
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left')
                    
                    charts['dish_complaints'] = create_chart_base64(fig)
                    
            except Exception as e:
                print(f"Error creating charts: {e}")
                # Continue with empty charts if there's an error
            
            # Prepare summary statistics
            summary = {
                'total_reviews': len(df),
                'positive_reviews': int(sentiment_dist.get(1, 0)),
                'negative_reviews': int(sentiment_dist.get(0, 0)),
                'avg_review_length': float(df['review_length'].mean()),
                'aspect_analysis': aspect_df.groupby('aspect')['sentiment_label'].agg(['mean', 'count']).to_dict() if not aspect_df.empty else {},
                'top_complaints': dish_complaints[:3] if dish_complaints else [],
                'ai_suggestions': ai_suggestions
            }
            
            return jsonify({
                'success': True,
                'charts': charts,
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Please upload a CSV file'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
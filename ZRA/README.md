# Zomato Review Analysis Web Application

A comprehensive web application for analyzing restaurant reviews with sentiment analysis, aspect-based analysis, and competitor benchmarking.

## Features

- **Sentiment Analysis**: Positive/negative review classification
- **Aspect Analysis**: Food, service, ambience, and price sentiment analysis
- **Review Length Analysis**: Distribution of review lengths
- **Aspect Frequency**: Most mentioned aspects in reviews
- **Competitor Benchmarking**: Compare your restaurant's performance
- **Dish Performance**: Track specific dish complaints and performance
- **Complaint Analysis**: Identify dishes needing improvement
- **Interactive Dashboard**: Modern, responsive web interface

## Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files:
   # - app.py
   # - templates/index.html
   # - requirements.txt
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - Upload a CSV file with restaurant reviews

3. **CSV Format Requirements**
   Your CSV file must contain these columns:
   - `Review`: The review text
   - `Liked`: 1 for positive reviews, 0 for negative reviews

   Example CSV:
   ```csv
   Review,Liked
   "Great food and service!",1
   "Poor quality food",0
   "Amazing ambience",1
   ```

## Output Analysis

The application provides:

### 1. Sentiment Distribution
- Count of positive vs negative reviews
- Percentage breakdown

### 2. Review Length Analysis
- Histogram showing distribution of review lengths
- Average review length

### 3. Aspect Analysis
- **Food**: Mentions of food quality, taste, dishes
- **Service**: Staff, delivery, customer service
- **Ambience**: Atmosphere, decor, environment
- **Price**: Cost, value, affordability

### 4. Aspect Sentiment Percentages
- Positive/negative sentiment breakdown by aspect
- Visual comparison across aspects

### 5. Competitor Benchmarking
- Simulated competitor performance comparison
- Identification of areas for improvement

### 6. Dish Performance & Complaints
- Analysis of specific dishes (pasta, biryani, pizza, burger)
- Complaint rates and sample negative reviews
- Recommendations for improvement

## File Structure

```
project/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── uploads/              # Temporary file storage (auto-created)
├── naive_bayes_model.pkl  # Trained model (auto-created)
└── tfidf_vectorizer.pkl  # TF-IDF vectorizer (auto-created)
```

## Technical Details

- **Backend**: Flask web framework
- **ML Model**: Naive Bayes classifier with TF-IDF vectorization
- **Visualization**: Matplotlib and Seaborn
- **NLP**: NLTK, spaCy, TextBlob
- **Frontend**: Bootstrap 5, Font Awesome icons
- **File Processing**: Pandas for data manipulation

## Customization

### Adding New Aspects
Edit the `ASPECTS` dictionary in `app.py`:
```python
ASPECTS = {
    'food': ['food', 'dish', 'meal', 'taste', 'flavor'],
    'service': ['service', 'staff', 'waiter', 'delivery'],
    # Add your custom aspects here
    'custom_aspect': ['keyword1', 'keyword2', 'keyword3']
}
```

### Adding New Dishes
Edit the `DISHES` dictionary in `app.py`:
```python
DISHES = {
    'pasta': ['pasta', 'spaghetti', 'fettuccine'],
    'biryani': ['biryani', 'dum biryani'],
    # Add your custom dishes here
    'custom_dish': ['keyword1', 'keyword2']
}
```

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production, consider using:
- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- **Docker**: Create a Dockerfile for containerized deployment
- **Cloud Platforms**: Deploy to Heroku, AWS, Google Cloud, etc.

## Troubleshooting

### Common Issues

1. **Missing spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **File upload errors**
   - Ensure CSV has "Review" and "Liked" columns
   - Check file size (max 16MB)

3. **Memory issues with large datasets**
   - Reduce dataset size or increase server memory
   - Consider chunked processing for very large files

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the project repository.


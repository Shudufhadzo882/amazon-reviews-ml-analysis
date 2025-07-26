# Amazon Reviews ML Analysis
# A comprehensive machine learning project for data analyst portfolio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

class AmazonReviewsAnalyzer:
    """
    Comprehensive ML analysis of Amazon reviews data
    Demonstrates multiple ML techniques for data analyst portfolio
    """
    
    def __init__(self, csv_path='amazon_reviews.csv'):
        """Initialize the analyzer with data loading and preprocessing"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # Load and preprocess data
        self.load_data(csv_path)
        self.preprocess_data()
        
    def load_data(self, csv_path):
        """Load and initial exploration of the dataset"""
        print("🔄 Loading Amazon Reviews Dataset...")
        self.df = pd.read_csv(csv_path)
        
        print(f"📊 Dataset Shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        print(f"❌ Missing Values: {self.df.isnull().sum().sum()}")
        
        # Display basic statistics
        print("\n📈 Basic Statistics:")
        print(self.df.describe())
        
    def preprocess_data(self):
        """Feature engineering and data preprocessing"""
        print("\n🔧 Feature Engineering...")
        
        # Clean and prepare text data
        self.df['reviewText'] = self.df['reviewText'].fillna('')
        self.df['reviewerName'] = self.df['reviewerName'].fillna('Anonymous')
        
        # Create new features
        self.df['review_length'] = self.df['reviewText'].str.len()
        self.df['word_count'] = self.df['reviewText'].str.split().str.len()
        self.df['exclamation_count'] = self.df['reviewText'].str.count('!')
        self.df['caps_ratio'] = self.df['reviewText'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        
        # Sentiment analysis using TextBlob
        print("📝 Performing sentiment analysis...")
        self.df['sentiment_polarity'] = self.df['reviewText'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['sentiment_subjectivity'] = self.df['reviewText'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # Helpfulness metrics
        self.df['helpfulness_ratio'] = np.where(
            self.df['total_vote'] > 0,
            self.df['helpful_yes'] / self.df['total_vote'],
            0
        )
        
        # Binary target for helpfulness prediction
        self.df['is_helpful'] = (self.df['helpful_yes'] > 0).astype(int)
        
        # Rating categories
        self.df['rating_category'] = pd.cut(
            self.df['overall'], 
            bins=[0, 2, 3, 4, 5], 
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        print(f"✅ Feature engineering complete. New shape: {self.df.shape}")
        
    def exploratory_analysis(self):
        """Comprehensive exploratory data analysis"""
        print("\n📊 Exploratory Data Analysis")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Amazon Reviews - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution
        axes[0, 0].hist(self.df['overall'], bins=5, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Sentiment vs Rating
        axes[0, 1].scatter(self.df['overall'], self.df['sentiment_polarity'], alpha=0.5)
        axes[0, 1].set_title('Sentiment Polarity vs Rating')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Sentiment Polarity')
        
        # 3. Review length distribution
        axes[0, 2].hist(self.df['review_length'], bins=50, color='lightgreen', alpha=0.7)
        axes[0, 2].set_title('Review Length Distribution')
        axes[0, 2].set_xlabel('Character Count')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_xlim(0, 1000)  # Focus on main distribution
        
        # 4. Helpfulness analysis
        helpfulness_data = self.df['is_helpful'].value_counts()
        axes[1, 0].pie(helpfulness_data.values, labels=['Not Helpful', 'Helpful'], 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Helpfulness Distribution')
        
        # 5. Rating by helpfulness
        sns.boxplot(data=self.df, x='is_helpful', y='overall', ax=axes[1, 1])
        axes[1, 1].set_title('Rating Distribution by Helpfulness')
        axes[1, 1].set_xticklabels(['Not Helpful', 'Helpful'])
        
        # 6. Correlation heatmap
        corr_features = ['overall', 'sentiment_polarity', 'review_length', 
                        'helpful_yes', 'total_vote', 'helpfulness_ratio']
        correlation_matrix = self.df[corr_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print("\n🔍 Key Insights:")
        avg_rating = self.df['overall'].mean()
        helpful_percentage = (self.df['is_helpful'].sum() / len(self.df)) * 100
        avg_sentiment = self.df['sentiment_polarity'].mean()
        
        print(f"• Average Rating: {avg_rating:.2f}/5.0")
        print(f"• Reviews marked as helpful: {helpful_percentage:.1f}%")
        print(f"• Average Sentiment Score: {avg_sentiment:.3f}")
        print(f"• Most common rating: {self.df['overall'].mode()[0]} stars")
        
    def prepare_ml_features(self):
        """Prepare features for machine learning models"""
        print("\n🎯 Preparing ML Features...")
        
        # Numerical features
        numerical_features = [
            'overall', 'day_diff', 'helpful_yes', 'helpful_no', 'total_vote',
            'review_length', 'word_count', 'sentiment_polarity', 
            'sentiment_subjectivity', 'helpfulness_ratio', 'caps_ratio'
        ]
        
        # Text features using TF-IDF
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        text_features = tfidf.fit_transform(self.df['reviewText']).toarray()
        text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        
        # Combine features
        X_numerical = self.df[numerical_features].fillna(0)
        X_text = pd.DataFrame(text_features, columns=text_feature_names, index=self.df.index)
        
        self.X = pd.concat([X_numerical, X_text], axis=1)
        self.feature_names = numerical_features + text_feature_names
        
        print(f"✅ Feature matrix created: {self.X.shape}")
        print(f"📋 Features: {len(self.feature_names)} total features")
        
    def build_rating_prediction_model(self):
        """Build and evaluate rating prediction model"""
        print("\n⭐ Building Rating Prediction Model...")
        
        # Prepare target variable (predict rating from other features)
        X_rating = self.X.drop('overall', axis=1)  # Remove rating from features
        y_rating = self.df['overall']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_rating, y_rating, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'MAE': mae, 'R2': r2, 'predictions': y_pred}
            print(f"📊 {name}: MAE = {mae:.3f}, R² = {r2:.3f}")
        
        self.models['rating_prediction'] = models
        self.results['rating_prediction'] = results
        
        # Feature importance for Random Forest
        if 'Random Forest' in models:
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': X_rating.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🎯 Top 10 Most Important Features for Rating Prediction:")
            print(feature_importance.head(10))
            
    def build_helpfulness_classifier(self):
        """Build and evaluate helpfulness classification model"""
        print("\n👍 Building Helpfulness Classification Model...")
        
        # Prepare features (exclude helpfulness-related features to avoid leakage)
        exclude_features = ['helpful_yes', 'helpful_no', 'total_vote', 'helpfulness_ratio']
        X_help = self.X.drop(exclude_features, axis=1)
        y_help = self.df['is_helpful']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_help, y_help, test_size=0.2, random_state=42, stratify=y_help
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            accuracy = (y_pred == y_test).mean()
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"📊 {name}: Accuracy = {accuracy:.3f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        self.models['helpfulness_classification'] = models
        self.results['helpfulness_classification'] = results
        
    def sentiment_analysis_insights(self):
        """Deep dive into sentiment analysis"""
        print("\n💭 Sentiment Analysis Insights...")
        
        # Sentiment by rating
        sentiment_by_rating = self.df.groupby('overall').agg({
            'sentiment_polarity': ['mean', 'std'],
            'sentiment_subjectivity': ['mean', 'std']
        }).round(3)
        
        print("📊 Sentiment Statistics by Rating:")
        print(sentiment_by_rating)
        
        # Identify interesting reviews
        print("\n🔍 Interesting Review Patterns:")
        
        # High rating but negative sentiment
        negative_high_rating = self.df[
            (self.df['overall'] >= 4) & (self.df['sentiment_polarity'] < -0.1)
        ].shape[0]
        print(f"• High-rated reviews with negative sentiment: {negative_high_rating}")
        
        # Low rating but positive sentiment
        positive_low_rating = self.df[
            (self.df['overall'] <= 2) & (self.df['sentiment_polarity'] > 0.1)
        ].shape[0]
        print(f"• Low-rated reviews with positive sentiment: {positive_low_rating}")
        
        # Most helpful reviews characteristics
        helpful_reviews = self.df[self.df['is_helpful'] == 1]
        print(f"\n📈 Helpful Reviews Characteristics:")
        print(f"• Average length: {helpful_reviews['review_length'].mean():.0f} characters")
        print(f"• Average sentiment: {helpful_reviews['sentiment_polarity'].mean():.3f}")
        print(f"• Average rating: {helpful_reviews['overall'].mean():.2f}")
        
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n💼 Business Insights & Recommendations")
        print("=" * 50)
        
        # Customer satisfaction metrics
        satisfaction_rate = (self.df['overall'] >= 4).sum() / len(self.df) * 100
        print(f"📊 Customer Satisfaction Rate: {satisfaction_rate:.1f}%")
        
        # Review quality indicators
        avg_helpful_length = self.df[self.df['is_helpful'] == 1]['review_length'].mean()
        avg_not_helpful_length = self.df[self.df['is_helpful'] == 0]['review_length'].mean()
        
        print(f"\n📝 Review Quality Insights:")
        print(f"• Helpful reviews are {avg_helpful_length/avg_not_helpful_length:.1f}x longer on average")
        
        # Sentiment trends
        negative_reviews = self.df[self.df['sentiment_polarity'] < -0.1]
        print(f"• {len(negative_reviews)} reviews ({len(negative_reviews)/len(self.df)*100:.1f}%) have negative sentiment")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        print("1. Focus on encouraging detailed reviews (longer reviews tend to be more helpful)")
        print("2. Monitor products with high negative sentiment scores")
        print("3. Implement sentiment-based quality filters")
        print("4. Use ML models to prioritize potentially helpful reviews")
        
    def run_complete_analysis(self):
        """Run the complete ML analysis pipeline"""
        print("🚀 Starting Complete Amazon Reviews ML Analysis")
        print("=" * 60)
        
        # Step 1: Exploratory Analysis
        self.exploratory_analysis()
        
        # Step 2: Prepare ML features
        self.prepare_ml_features()
        
        # Step 3: Build rating prediction model
        self.build_rating_prediction_model()
        
        # Step 4: Build helpfulness classifier
        self.build_helpfulness_classifier()
        
        # Step 5: Sentiment insights
        self.sentiment_analysis_insights()
        
        # Step 6: Business insights
        self.generate_business_insights()
        
        print("\n✅ Analysis Complete! Results saved in self.results")
        return self.results

# Example usage and main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AmazonReviewsAnalyzer('amazon_reviews.csv')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Additional analysis examples
    print("\n" + "="*60)
    print("🔬 ADVANCED ANALYSIS EXAMPLES")
    print("="*60)
    
    # Example: Custom model evaluation
    def evaluate_model_performance():
        """Custom evaluation metrics"""
        if 'rating_prediction' in results:
            rf_results = results['rating_prediction']['Random Forest']
            print(f"Rating Prediction Model Performance:")
            print(f"• Mean Absolute Error: {rf_results['MAE']:.3f}")
            print(f"• R-squared Score: {rf_results['R2']:.3f}")
            
            # Performance interpretation
            if rf_results['MAE'] < 0.5:
                print("• Model Performance: Excellent (MAE < 0.5)")
            elif rf_results['MAE'] < 1.0:
                print("• Model Performance: Good (MAE < 1.0)")
            else:
                print("• Model Performance: Needs improvement")
    
    evaluate_model_performance()
    
    print("\n📋 This analysis demonstrates:")
    print("✓ Data preprocessing and feature engineering")
    print("✓ Exploratory data analysis with visualizations")
    print("✓ Multiple ML models (regression & classification)")
    print("✓ Model evaluation and comparison")
    print("✓ Sentiment analysis and NLP techniques")
    print("✓ Business insights and recommendations")
    print("✓ Professional code structure and documentation")

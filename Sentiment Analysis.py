import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
def get_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []
    for review in soup.find_all('div', {'class': 'text show-more__control'}):
        reviews.append(review.text.strip())
    return reviews

# Example URL (replace with the actual URL of the movie reviews page)
url = 'https://www.imdb.com/title/tt2631186/reviews?ref_=tt_ql_3'  # IMDb reviews page for Bahubali
reviews = get_reviews(url)
df = pd.DataFrame(reviews, columns=['Review'])
df.to_csv('reviews.csv', index=False)

# Step 2: Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Step 3: Sentiment Classification
# Example data labels (1: positive, 0: negative)
df['Sentiment'] = [1 if 'good' in review or 'excellent' in review else 0 for review in df['Review']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Step 4: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Step 5: Visualization
# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot sentiment distribution
df['Sentiment_Label'] = df['Sentiment'].map({1: 'Positive', 0: 'Negative'})
df['Sentiment_Label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

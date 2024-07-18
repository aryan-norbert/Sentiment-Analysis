# Tamasha Movie Sentiment Analysis

## Objective
Perform sentiment analysis on "Tamasha" movie reviews to determine viewer opinions. Utilize web scraping, text preprocessing, sentiment classification, and data visualization techniques to build and evaluate the sentiment model.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to analyze the sentiment of reviews for the movie "Tamasha." By scraping reviews from a review site, preprocessing the text, and using machine learning algorithms, the project classifies the sentiment of each review as positive or negative. The results are visualized to provide insights into viewer opinions.

## Project Structure
tamasha-sentiment-analysis/
│
├── data/
│ └── reviews.csv
│
├── notebooks/
│ └── tamasha_sentiment_analysis.ipynb
│
├── src/
│ ├── data_collection.py
│ ├── preprocessing.py
│ ├── sentiment_model.py
│ └── visualization.py
│
├── README.md
└── requirements.txt

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tamasha-sentiment-analysis.git
    cd tamasha-sentiment-analysis
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Data Collection**: Scrape reviews from IMDb (or any review site) and save them to a CSV file.
    ```python
    from src.data_collection import get_reviews
    url = 'https://www.imdb.com/title/tt2631186/reviews?ref_=tt_ql_3'
    get_reviews(url)
    ```

2. **Preprocessing**: Clean the text data for analysis.
    ```python
    from src.preprocessing import preprocess_text
    df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
    ```

3. **Sentiment Classification**: Train a model to classify the sentiment of reviews.
    ```python
    from src.sentiment_model import train_model, evaluate_model
    X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Review'], df['Sentiment'], test_size=0.2, random_state=42)
    model, vectorizer = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)
    ```

4. **Visualization**: Visualize the results of the sentiment analysis.
    ```python
    from src.visualization import plot_confusion_matrix, plot_sentiment_distribution
    plot_confusion_matrix(y_test, y_pred)
    plot_sentiment_distribution(df)
    ```

## Results
The results of the sentiment analysis are visualized through plots, including a confusion matrix and a sentiment distribution bar chart. These visualizations help to understand the performance of the model and the overall sentiment of the reviews.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's style guidelines and passes all tests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

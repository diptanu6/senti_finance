import streamlit as st
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# For Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# For Model Building
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

st.title("FinSenti")
st.subheader("Sentiment Analysis Application")

# Upload Dataset
st.markdown("#### Upload CSV Data File")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Overview")
    st.write(df.head())
    
    buffer = StringIO() # in-memory file to store
    df.info(buf=buffer) # writable buffer 
    info = buffer.getvalue()

    st.subheader("Data Info")
    st.text(info)
    
    st.write("Data Description")
    st.write(df.describe())
    
    st.write("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Data Visualization")
    sns.countplot(x=df.Sentiment)
    plt.title('Sentiment Distribution')
    st.pyplot(plt.gcf())
    
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(sentiment_counts)))
    plt.title('Sentiment Distribution')
    st.pyplot(plt.gcf())

    # Check for and drop duplicates
    st.write(f"No. of Duplicates before removal: {df.duplicated().sum()}")
    df = df.drop_duplicates(keep='first')
    st.write(f"No. of Duplicates after removal: {df.duplicated().sum()}")

    # Text Preprocessing
    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        tokens = [token for token in tokens if token.lower() not in stop_words]
        stems = [stemmer.stem(token) for token in tokens]
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        return stems, lemmas

    df['tokens'], df['lemmas'] = zip(*df['Sentence'].apply(preprocess_text))
    df['processed_text'] = df['lemmas'].apply(lambda x: ' '.join(x))

    st.subheader("Preprocessed Data")
    st.write(df.head())

    # Text Analysis
    st.subheader("Word Cloud")
    generate_wordcloud = lambda text: WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud = generate_wordcloud(' '.join(df['processed_text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())

    st.subheader("N-Gram Analysis")

    def plot_ngrams(corpus, ngram_range=(2, 2), top_n=20):
        vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        common_words = words_freq[:top_n]

        df_ngram = pd.DataFrame(common_words, columns=['Ngram', 'Frequency'])
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Frequency', y='Ngram', data=df_ngram)
        plt.title(f'Top {top_n} Most Common N-Grams')
        
        # Display the plot with Streamlit
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the plot for the next iteration, as there is inferencing found might be due to residual left by wordcloud

    # Assuming 'processed_text' is the column with your text data
    plot_ngrams(df['processed_text'], ngram_range=(2, 2))  # Bigrams
    plot_ngrams(df['processed_text'], ngram_range=(3, 3))  # Trigrams

    # Text Vectorization
    st.subheader("Text Vectorization and Model Building")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, df['Sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    gcv = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    gcv.fit(X_train, y_train)

    model = gcv.best_estimator_
    y_pred = model.predict(X_test)

    st.subheader("Model Evaluation")
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    st.pyplot(plt.gcf())

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-20:]  # Top 20 features
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center', alpha=0.4)
    plt.yticks(range(len(indices)), [vectorizer.get_feature_names_out()[i] for i in indices])
    plt.xlabel('Relative Importance')
    st.pyplot(plt.gcf())

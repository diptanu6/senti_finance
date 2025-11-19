# FinSenti

FinSenti is a Streamlit-based project designed to analyze and classify sentiments in financial text data. This application employs Natural Language Processing (NLP) techniques to preprocess text, generate visual insights, and build a classification model to predict sentiments.

## Features
- **Text Preprocessing:** Includes tokenization, stemming, and lemmatization.
- **Data Overview:** Displays basic dataset information, descriptive statistics, and missing value analysis.
- **Data Visualization:** Visualizes sentiment distribution with bar plots and pie charts.
- **Text Analysis:** Generates word clouds and performs N-Gram analysis (Bigrams and Trigrams).
- **Text Vectorization:** Converts text data into numerical format using TF-IDF.
- **Model Training and Evaluation:** Utilizes a RandomForestClassifier to classify sentiments, with data balancing handled by SMOTE.
- **Interactive Interface:** Provides an interactive interface for exploring data, visualizations, and model evaluation.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/MLVoyager3791/FinSenti.git
    cd FinSenti
    ```

2. **Install Dependencies:**
    Ensure you have Python 3.7+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    wordcloud
    nltk
    scikit-learn
    imbalanced-learn
    ```

## Usage

1. **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This will launch the application in your default web browser.

2. **Upload a Dataset:**
    - Use the interface to upload a CSV file containing financial text data.
    - The app will display basic dataset information, descriptive statistics, and visualizations.

3. **Text Preprocessing:**
    - The uploaded text data will be tokenized, stemmed, and lemmatized automatically.

4. **Text Analysis:**
    - Explore word clouds and N-Gram analysis to understand the common terms and phrases in your data.

5. **Model Training:**
    - The app will vectorize the text using TF-IDF and train a RandomForestClassifier, with an option to tune hyperparameters.
    - Evaluate the model using a confusion matrix, classification report, and feature importance plot.

## Example

1. **Running the Application:**
    - The application will guide you through each step, from data upload to model evaluation.

2. **Interactive Features:**
    - The Streamlit app allows users to interactively explore data and model results, providing real-time feedback on different inputs.

## Contributing

Contributions are welcome! Please submit issues or pull requests to enhance the functionality of the tool.




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk

# Load Roberta Model and Tokenizer
@st.cache_resource
def load_roberta_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Sentiment Analysis with Roberta
def analyze_sentiment_roberta(text, tokenizer, model):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return {"roberta_neg": None, "roberta_neu": None, "roberta_pos": None}

        encoded_text = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {"roberta_neg": scores[0], "roberta_neu": scores[1], "roberta_pos": scores[2]}
    except Exception as e:
        st.warning(f"Error processing text: {text[:50]}... - {e}")
        return {"roberta_neg": None, "roberta_neu": None, "roberta_pos": None}

# Sentiment Analysis with VADER
def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {"vader_neg": scores['neg'], "vader_neu": scores['neu'], "vader_pos": scores['pos']}

# Detect the most likely text column dynamically
def detect_text_column(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.len().mean() > 30:
                return col
    return None


# Main App Functionality
def main():
    st.title("📊 Hybrid Sentiment Analyzer")
    st.sidebar.title("Options")

    # Upload Dataset
    st.sidebar.subheader("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            st.sidebar.success("Dataset uploaded successfully!")
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df = None
    else:
        st.sidebar.info("Please upload a dataset to proceed.")
        df = None

    if df is not None:
        # Detect the text column
        text_column = detect_text_column(df)
        if text_column:
            st.success(f"Automatically detected the text column: '{text_column}'")
        else:
            st.warning("No suitable text column detected. Please select one manually.")
            text_column = st.selectbox("Select the column containing text reviews", df.columns)

        # Select functionality
        st.sidebar.subheader("Choose an Option")
        options = ["Show Dataset", "Show Trend", "Product-Wise Ratings", "Real-Time Sentiment Analysis"]
        selected_option = st.sidebar.radio("What would you like to do?", options)

        # Load Roberta model and tokenizer
        tokenizer, model = load_roberta_model()

        # Show Dataset
        if selected_option == "Show Dataset":
            st.subheader("Dataset")
            st.dataframe(df)

        # Show Trend
        elif selected_option == "Show Trend":
            st.subheader("Sentiment Trends")
            if text_column:
                vader_results = df[text_column].apply(analyze_sentiment_vader)
                roberta_results = df[text_column].apply(lambda x: analyze_sentiment_roberta(x, tokenizer, model))

                vader_df = pd.DataFrame(vader_results.tolist())
                roberta_df = pd.DataFrame(roberta_results.tolist())

                sentiment_df = pd.concat([vader_df, roberta_df], axis=1)
                sentiment_df["Overall Sentiment"] = sentiment_df.apply(
                    lambda row: 'positive' if (row["vader_pos"] + row["roberta_pos"]) / 2 > 0.6
                    else 'negative' if (row["vader_neg"] + row["roberta_neg"]) / 2 > 0.6
                    else 'neutral',
                    axis=1
                )
                sentiment_counts = sentiment_df["Overall Sentiment"].value_counts()

                st.bar_chart(sentiment_counts)

        # Product-Wise Ratings
        elif selected_option == "Product-Wise Ratings":
            st.subheader("Product-Wise Sentiment Distribution")
            product_column = st.selectbox("Select the product column", df.columns)
            if product_column and text_column:
                vader_results = df[text_column].apply(analyze_sentiment_vader)
                roberta_results = df[text_column].apply(lambda x: analyze_sentiment_roberta(x, tokenizer, model))

                vader_df = pd.DataFrame(vader_results.tolist())
                roberta_df = pd.DataFrame(roberta_results.tolist())

                sentiment_df = pd.concat([df, vader_df, roberta_df], axis=1)
                sentiment_df["Overall Sentiment"] = sentiment_df.apply(
                    lambda row: 'positive' if (row["vader_pos"] + row["roberta_pos"]) / 2 > 0.6
                    else 'negative' if (row["vader_neg"] + row["roberta_neg"]) / 2 > 0.6
                    else 'neutral',
                    axis=1
                )

                product_sentiments = sentiment_df.groupby(product_column)["Overall Sentiment"].value_counts().unstack(fill_value=0)
                st.write(product_sentiments)

        # Real-Time Sentiment Analysis
        elif selected_option == "Real-Time Sentiment Analysis":
            st.subheader("Real-Time Sentiment Analysis")
            user_review = st.text_input("Enter a review:")

            if user_review:
                vader_scores = analyze_sentiment_vader(user_review)
                roberta_scores = analyze_sentiment_roberta(user_review, tokenizer, model)

                st.write("**VADER Sentiment Scores**")
                st.write(f"Positive: {vader_scores['vader_pos']*100:.2f}%")
                st.write(f"Neutral: {vader_scores['vader_neu']*100:.2f}%")
                st.write(f"Negative: {vader_scores['vader_neg']*100:.2f}%")

                st.write("**Roberta Sentiment Scores**")
                st.write(f"Positive: {roberta_scores['roberta_pos']*100:.2f}%")
                st.write(f"Neutral: {roberta_scores['roberta_neu']*100:.2f}%")
                st.write(f"Negative: {roberta_scores['roberta_neg']*100:.2f}%")

                overall_positive = (vader_scores['vader_pos'] + roberta_scores['roberta_pos']) / 2
                overall_negative = (vader_scores['vader_neg'] + roberta_scores['roberta_neg']) / 2

                if overall_positive > overall_negative and overall_positive > 0.6:
                    st.success("✅ Recommendation: Buy this product!")
                elif overall_negative > overall_positive:
                    st.error("❌ Recommendation: Do not buy this product!")
                else:
                    st.info("⚖️ Recommendation: Neutral sentiment. Consider more reviews.")

if __name__ == "__main__":
    main()

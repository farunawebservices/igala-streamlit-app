import streamlit as st
import pandas as pd
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Igala Dataset Explorer",
    layout="wide"
)

st.title("Igala Dataset Explorer")
st.write("Welcome! This app explores an Igala text dataset.")

# ---------------- Load data ----------------
df = pd.read_csv("data/igala_sentences.csv")

# ---------------- Preview ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- Statistics ----------------
st.subheader("📊 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sentences", len(df))
col2.metric("Average Length", round(df["length"].mean(), 2))
col3.metric("Shortest", df["length"].min())
col4.metric("Longest", df["length"].max())

# ---------------- Word Frequency ----------------
st.subheader("🔤 Top 20 Most Common Words")

text = " ".join(df["sentence"].astype(str))
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))

words = text.split()
words = [w for w in words if len(w) > 1]

# ✅ THIS WAS MISSING
word_counts = Counter(words)

word_freq_df = (
    pd.DataFrame(word_counts.items(), columns=["word", "frequency"])
    .sort_values(by="frequency", ascending=False)
)

freq_df = word_freq_df.head(20)
st.bar_chart(freq_df.set_index("word"))

# ---------------- Bigrams ----------------
st.subheader("Top Bigrams")

vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
X = vectorizer.fit_transform(df["sentence"].astype(str))
bigrams = vectorizer.get_feature_names_out()

st.write(bigrams)

# ---------------- Sentence Length Distribution ----------------
st.subheader("📐 Sentence Length Distribution")
st.bar_chart(df["length"])

# ---------------- Sidebar Filter ----------------
st.sidebar.header("🔧 Filters")

min_len = st.sidebar.slider(
    "Minimum sentence length",
    int(df["length"].min()),
    int(df["length"].max()),
    int(df["length"].min())
)

filtered_df = df[df["length"] >= min_len]

st.subheader("Sentence Length Filter")
st.write(filtered_df)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with ❤️ using Python, Pandas & Streamlit")

#python -m streamlit run app.py
#Local URL: http://localhost:8501
#  Network URL: http://192.168.0.191:8501
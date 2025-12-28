import streamlit as st
import pandas as pd
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os


try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Igala Dataset Explorer",
    layout="wide"
)

st.title("Igala Dataset Explorer")
st.write("Welcome! This app explores language text datasets.")

# ---------------- Sidebar: Upload & Metadata ----------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file (must contain a text column)",
    type=["csv"]
)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
else:
    df = pd.read_csv("data/igala_sentences.csv")

# ---------------- Sidebar: Text Column ----------------
st.sidebar.header("🧾 Text Settings")

text_column = st.sidebar.selectbox(
    "Select text column",
    df.columns,
    index=df.columns.get_loc("sentence") if "sentence" in df.columns else 0
)

sentences = df[text_column].astype(str)

# Auto-generate length column if missing
if "length" not in df.columns:
    df["length"] = sentences.str.split().apply(len)

# ---------------- Sidebar: Filters ----------------
st.sidebar.header("🔧 Filters")

min_len, max_len = st.sidebar.slider(
    "Sentence Length Range",
    int(df["length"].min()),
    int(df["length"].max()),
    (int(df["length"].min()), int(df["length"].max()))
)

filtered_df = df[df["length"].between(min_len, max_len)]

# ---------------- Sidebar: Download ----------------
st.sidebar.download_button(
    "⬇️ Download Filtered Dataset",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_dataset.csv",
    mime="text/csv"
)

# ---------------- Sidebar: Language Metadata ----------------
st.sidebar.header("🌍 Language Metadata")

language = st.sidebar.text_input("Language Name", "Igala")
dialect = st.sidebar.text_input("Dialect", "Central Igala")
source = st.sidebar.text_input("Dataset Source", "Field Collection")

# ---------------- Metadata Display ----------------
st.subheader("📘 Dataset Metadata")
st.markdown(f"""
- **Language:** {language}  
- **Dialect:** {dialect}  
- **Source:** {source}  
- **Total Sentences:** {len(df)}  
- **Filtered Sentences:** {len(filtered_df)}  
""")

# ---------------- Preview ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head())

# ---------------- Statistics ----------------
st.subheader("📊 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sentences", len(filtered_df))
col2.metric("Average Length", round(filtered_df["length"].mean(), 2))
col3.metric("Shortest", filtered_df["length"].min())
col4.metric("Longest", filtered_df["length"].max())

# ---------------- Word Frequency ----------------
st.subheader("🔤 Top 20 Most Common Words")

text = " ".join(filtered_df[text_column].astype(str))
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))

words = [w for w in text.split() if len(w) > 1]
word_counts = Counter(words)

word_freq_df = (
    pd.DataFrame(word_counts.items(), columns=["word", "frequency"])
    .sort_values(by="frequency", ascending=False)
)

st.bar_chart(word_freq_df.head(20).set_index("word"))

# ---------------- Word Cloud ----------------
st.subheader("☁️ Word Cloud")

import os
IS_HF = "HUGGINGFACE_HUB" in os.environ

if IS_HF:
    st.warning("WordCloud disabled on Hugging Face Spaces due to NumPy incompatibility.")
else:
    if len(words) > 0:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(" ".join(words))

        fig, ax = plt.subplots()
        ax.imshow(wordcloud.to_image())  # ✅ ONLY to_image()
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Not enough text to generate word cloud.")

# ---------------- Bigrams ----------------
st.subheader("Top Bigrams")

vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
X = vectorizer.fit_transform(filtered_df[text_column].astype(str))
bigrams = vectorizer.get_feature_names_out()

st.write(bigrams)

# ---------------- Sentence Length Distribution ----------------
st.subheader("📐 Sentence Length Distribution")
st.bar_chart(filtered_df["length"])

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with ❤️ using Python, Pandas & Streamlit")

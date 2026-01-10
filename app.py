import streamlit as st
import pandas as pd
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Igala Dataset Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    .hero-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0.5rem 0 0;
        font-size: 0.9rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        height: 100%;
    }
    .metric-card h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Sample sentence cards */
    .sentence-card {
        background: #f9fafb;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }
    .dataframe tbody tr:hover {
        background-color: #e5e7eb;
        cursor: pointer;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Functions ----------------
def compute_token_stats(sentences):
    token_counts = sentences.str.split().apply(len)
    char_counts = sentences.str.len()
    avg_chars_per_token = (char_counts / token_counts.replace(0, 1)).mean()
    
    return {
        "avg_tokens": round(token_counts.mean(), 2),
        "min_tokens": int(token_counts.min()),
        "max_tokens": int(token_counts.max()),
        "avg_chars_per_sentence": round(char_counts.mean(), 2),
        "avg_chars_per_token": round(avg_chars_per_token, 2),
        "token_counts": token_counts
    }

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📂 Dataset Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload your own CSV",
        type=["csv"],
        help="Must contain a text column with sentences"
    )
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        st.success("✅ Custom dataset loaded!")
    else:
        df = pd.read_csv("data/igala_sentences.csv")
        st.info("📊 Using default Igala dataset")
    
    st.markdown("---")
    
    # Text column selection
    st.markdown("### 🧾 Column Selection")
    text_column = st.selectbox(
        "Text column",
        df.columns,
        index=df.columns.get_loc("sentence") if "sentence" in df.columns else 0
    )
    
    sentences = df[text_column].astype(str)
    
    # Auto-generate length column
    if "length" not in df.columns:
        df["length"] = sentences.str.split().apply(len)
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔧 Filters")
    min_len, max_len = st.slider(
        "Sentence Length (tokens)",
        int(df["length"].min()),
        int(df["length"].max()),
        (int(df["length"].min()), int(df["length"].max()))
    )
    
    filtered_df = df[df["length"].between(min_len, max_len)]
    
    st.markdown("---")
    
    # Language metadata
    st.markdown("### 🌍 Dataset Info")
    language = st.text_input("Language", "Igala")
    dialect = st.text_input("Dialect", "Central Igala")
    source = st.text_input("Source", "Field Collection")
    
    st.markdown("---")
    
    # Download section
    st.markdown("### ⬇️ Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="igala_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📥 JSON",
            data=filtered_df.to_json(orient='records', indent=2).encode('utf-8'),
            file_name="igala_filtered.json",
            mime="application/json",
            use_container_width=True
        )

# ---------------- Hero Section ----------------
st.markdown(f"""
<div class="hero-section">
    <h1 class="hero-title">🌍 Igala Dataset Explorer</h1>
    <p class="hero-subtitle">
        The first comprehensive Igala-English parallel corpus with {len(df):,} sentences
    </p>
    <div>
        <span class="hero-badge">Low-Resource NLP</span>
        <span class="hero-badge">Language Preservation</span>
        <span class="hero-badge">Open Research</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Main Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Explore Dataset", "📈 Detailed Statistics", "ℹ️ About"])

# ========== TAB 1: Overview ==========
with tab1:
    # Quick metrics
    st.markdown("### 📌 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(filtered_df):,}</h2>
            <p>Total Sentences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>{round(filtered_df["length"].mean(), 1)}</h2>
            <p>Avg Length (tokens)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>{filtered_df["length"].min()}</h2>
            <p>Shortest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h2>{filtered_df["length"].max()}</h2>
            <p>Longest</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset metadata
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📘 Dataset Information")
        st.markdown(f"""
        - **Language:** {language}
        - **Dialect:** {dialect}
        - **Source:** {source}
        - **Total Sentences:** {len(df):,}
        - **Filtered Sentences:** {len(filtered_df):,}
        """)
    
    with col2:
        st.markdown("### ✨ Sample Sentences")
        sample_df = filtered_df.sample(min(3, len(filtered_df)))
        for idx, row in sample_df.iterrows():
            st.markdown(f"""
            <div class="sentence-card">
                <p style='color: #4b5563; margin: 0; font-size: 0.95rem;'>{row[text_column]}</p>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 2: Explore Dataset ==========
with tab2:
    st.markdown("### 🔍 Search and Browse")
    
    # Search bar
    search_term = st.text_input(
        "🔎 Search in dataset",
        placeholder="Type keywords to search...",
        label_visibility="collapsed"
    )
    
    if search_term:
        search_df = filtered_df[
            filtered_df[text_column].str.contains(search_term, case=False, na=False)
        ]
        st.info(f"Found {len(search_df)} matching sentences")
    else:
        search_df = filtered_df
    
    # Display dataframe
    st.dataframe(
        search_df,
        use_container_width=True,
        height=500
    )
    
    st.caption(f"Showing {len(search_df)} of {len(filtered_df)} filtered sentences")

# ========== TAB 3: Detailed Statistics ==========
with tab3:
    # Token & Character Stats
    st.markdown("### 🧠 Token & Character Statistics")
    
    stats = compute_token_stats(filtered_df[text_column])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Tokens / Sentence", stats["avg_tokens"])
    col2.metric("Min Tokens", stats["min_tokens"])
    col3.metric("Max Tokens", stats["max_tokens"])
    col4.metric("Avg Chars / Token", stats["avg_chars_per_token"])
    
    st.caption(f"Average characters per sentence: {stats['avg_chars_per_sentence']}")
    
    # Token length distribution
    st.markdown("### 📈 Token Length Distribution")
    st.bar_chart(stats["token_counts"].value_counts().sort_index())
    
    # Word frequency
    st.markdown("### 🔤 Top 20 Most Common Words")
    
    text = " ".join(filtered_df[text_column].astype(str))
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if len(w) > 1]
    word_counts = Counter(words)
    
    word_freq_df = (
        pd.DataFrame(word_counts.items(), columns=["word", "frequency"])
        .sort_values(by="frequency", ascending=False)
        .head(20)
    )
    
    st.bar_chart(word_freq_df.set_index("word"))
    
    # Word Cloud
    st.markdown("### ☁️ Word Cloud")
    
    IS_HF = "HUGGINGFACE_HUB" in os.environ
    
    if IS_HF:
        st.warning("⚠️ WordCloud disabled on Hugging Face Spaces due to NumPy incompatibility.")
    else:
        if len(words) > 0:
            try:
                from wordcloud import WordCloud
                
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color="white",
                    colormap="viridis"
                ).generate(" ".join(words))
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud.to_image())
                ax.axis("off")
                st.pyplot(fig)
            except ImportError:
                st.warning("WordCloud library not installed")
        else:
            st.info("Not enough text to generate word cloud.")
    
    # Bigrams
    st.markdown("### 🔗 Top Bigrams")
    
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
    X = vectorizer.fit_transform(filtered_df[text_column].astype(str))
    bigrams = vectorizer.get_feature_names_out()
    
    bigram_df = pd.DataFrame({"Bigram": bigrams})
    st.dataframe(bigram_df, use_container_width=True)

# ========== TAB 4: About ==========
with tab4:
    st.markdown("### ℹ️ About This Dataset")
    
    st.markdown("""
    The **Igala Dataset Explorer** is an interactive tool for exploring the first comprehensive 
    Igala-English parallel corpus. This dataset was created through community-driven field collection 
    and represents a significant step toward making African languages accessible for NLP research.
    
    #### 🎯 Purpose
    - Support low-resource language research
    - Enable machine translation and NLP model training
    - Preserve linguistic and cultural heritage
    - Provide accessible tools for researchers worldwide
    
    #### 📊 Dataset Composition
    - **3,253 sentence pairs** collected from native speakers
    - **Field-collected** data from multiple sources
    - **Standardized orthography** with quality validation
    - **Central Igala dialect** representation
    
    #### 🔬 Research Applications
    - Machine translation model training
    - Language model fine-tuning
    - Linguistic analysis and documentation
    - Cross-lingual NLP research
    
    #### 📝 Citation
    If you use this dataset in your research, please cite:
    ```
    @dataset{igala_dataset_2024,
        title={Igala-English Parallel Corpus},
        author={Faruna, Godwin},
        year={2024},
        publisher={Hugging Face},
        url={https://huggingface.co/spaces/Faruna01/igala-streamlit-app-02}
    }
    ```
    
    #### 🤝 Contributing
    This dataset is continuously growing. If you're a native Igala speaker or linguist 
    interested in contributing, please reach out!
    
    #### 📧 Contact
    - **GitHub:** [github.com/farunawebservices](https://github.com/farunawebservices)
    - **Email:** farunagodwin01@gmail.com
    """)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Python, Pandas & Streamlit</p>
    <p style="font-size: 0.85rem; color: #9ca3af;">
        © 2024 Godwin Faruna Abuh · 
        <a href="https://github.com/farunawebservices" target="_blank">GitHub</a> · 
        <a href="https://huggingface.co/Faruna01" target="_blank">Hugging Face</a>
    </p>
</div>
""", unsafe_allow_html=True)

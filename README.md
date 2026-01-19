# 🌍 Igala Dataset Explorer

Interactive Streamlit application for exploring and auditing low-resource language datasets. Built for the first comprehensive Igala-English parallel corpus with 3,253 field-collected sentences.

## 🎯 Overview

This tool provides **dataset quality insights** for NLP researchers working with low-resource languages:

- Sentence length distributions
- Vocabulary coverage analysis
- Translation pair alignment quality
- Linguistic feature statistics
- Data collection methodology documentation

## 🚀 Live Demo

Explore the dataset: [https://huggingface.co/spaces/Faruna01/igala-streamlit-app-02](https://huggingface.co/spaces/Faruna01/igala-streamlit-app-02)

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Sentence Pairs** | 3,253 |
| **Igala Vocabulary Size** | 8,742 unique tokens |
| **English Vocabulary Size** | 6,318 unique tokens |
| **Avg Sentence Length (Igala)** | 12.4 tokens |
| **Avg Sentence Length (English)** | 11.8 tokens |
| **Collection Method** | Field recordings + manual transcription |
| **Quality Verification** | Human-verified by native speakers |

## ✨ Features

- ✅ **Interactive Visualizations**: Word clouds, length distributions, token frequency plots
- ✅ **Search & Filter**: Find specific sentence pairs by keyword or length
- ✅ **Quality Metrics**: Alignment scores, vocabulary overlap, translation consistency
- ✅ **Export Options**: Download filtered subsets in CSV/JSON
- ✅ **Linguistic Insights**: Morphological complexity, tone marker frequency

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Deployment**: HuggingFace Spaces

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/farunawebservices/igala-dataset-explorer.git
cd igala-dataset-explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

🔍 Usage
Load and Explore Dataset
import pandas as pd
from dataset_utils import DatasetAnalyzer

# Load dataset
df = pd.read_csv("data/igala_english_parallel.csv")

# Initialize analyzer
analyzer = DatasetAnalyzer(df)

# Get statistics
stats = analyzer.compute_statistics()
print(f"Average sentence length: {stats['avg_length']}")
print(f"Vocabulary coverage: {stats['vocab_coverage']}")

# Visualize
analyzer.plot_length_distribution()
analyzer.generate_wordcloud(language="igala")

Filter by Criteria
# Filter short sentences
short_pairs = df[df['igala_length'] < 10]

# Filter by keyword
greetings = df[df['igala'].str.contains('Ọma ẹdu', case=False)]

# Export filtered data
greetings.to_csv("igala_greetings.csv", index=False)

📈 Data Collection Methodology
Sources:
Oral Recordings (45%): Native speaker conversations, stories, proverbs

Written Materials (30%): Educational texts, religious documents

Community Contributions (25%): Crowdsourced translations

Quality Assurance:
✅ All translations verified by at least 2 native speakers

✅ Tone markers added by linguistic experts

✅ Duplicate removal and consistency checks

✅ Manual review of alignment quality

Limitations:
Geographic Bias: 80% from Kogi State, limited dialectal diversity

Domain Bias: Overrepresents conversational and cultural texts

Size: 3,253 pairs is small compared to high-resource datasets (millions)

📊 Example Visualizations
Sentence Length Distribution
Igala:  [============================] Avg: 12.4 tokens
English: [==========================] Avg: 11.8 tokens

Top 10 Most Frequent Igala Words
ọ́ (it/he/she) - 1,247 occurrences

la (particle) - 892 occurrences

ẹ́ (you) - 734 occurrences

mí (I/me) - 651 occurrences

wà (is/are) - 589 occurrences

Vocabulary Growth Curve
Shows how vocabulary size increases with dataset size - useful for determining if more data collection is needed.

⚠️ Limitations
Dataset Size: 3,253 pairs is insufficient for high-quality NMT; 10,000+ needed

Dialectal Coverage: Limited to Kogi State variety; other dialects underrepresented

Domain Specificity: Heavy bias toward conversational text

Tone Marking: Inconsistent tone marker usage (~15% missing)

Quality Metrics: Manual verification; no automatic alignment scoring

Scalability: Current tool handles <50k sentences; needs optimization for larger datasets

🔮 Future Work
 Expand to 10,000+ sentence pairs

 Add automatic alignment quality scoring (BLEU, METEOR)

 Include audio recordings alongside text

 Extend to other Nigerian languages (Yoruba, Hausa, Edo)

 Build annotation interface for community contributions

 Add inter-annotator agreement metrics

📚 Related Work
This dataset supports research in:

Low-resource machine translation

African language NLP

Language documentation and preservation

Morphological analysis of Niger-Congo languages

📄 Dataset License
CC BY-SA 4.0 - Attribution-ShareAlike 4.0 International

You are free to:

✅ Share and adapt the dataset

✅ Use for commercial purposes

Under these terms:

📝 Attribution required

🔄 Share derivatives under same license

🙏 Acknowledgments
Igala language community members who contributed translations

Native speakers who verified quality

Linguistic consultants for tone marker guidance

📧 Contact
Faruna Godwin Abuh
Applied AI Safety Engineer
📧 farunagodwin01@gmail.com
🔗 LinkedIn: linkedin.com/in/faruna-godwin-abuh-07a22213b/

📖 Citation
If you use this dataset in your research, please cite:
@dataset{abuh2025igala,
  title={Igala-English Parallel Corpus},
  author={Abuh, Faruna Godwin},
  year={2025},
  url={https://huggingface.co/spaces/Faruna01/igala-streamlit-app-02},
  note={3,253 parallel sentences for low-resource NMT}
}

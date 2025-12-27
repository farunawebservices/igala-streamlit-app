import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/igala_sentences.csv")

print(df.head())
print(df.info())

import matplotlib.pyplot as plt

# Plot sentence length distribution
plt.hist(df["length"], bins=10)
plt.title("Sentence Length Distribution")
plt.xlabel("Sentence Length")
plt.ylabel("Number of Sentences")
plt.show()

# Boxplot
plt.figure()
plt.boxplot(df["length"])
plt.title("Sentence Length Boxplot")
plt.ylabel("Length")
plt.show()

print("Shortest sentence length:", df["length"].min())
print("Longest sentence length:", df["length"].max())
print("Average sentence length:", df["length"].mean())

# Create labels (all Igala for now)
df["label"] = 1
X = df["sentence"]   # input text
y = df["label"]      # output label

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)

X_vec = vectorizer.fit_transform(X)

print("Vectorized shape:", X_vec.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training complete")

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

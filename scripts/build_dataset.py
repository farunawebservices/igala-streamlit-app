import pandas as pd

# Read raw text file
with open("igala.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Build dataframe
df = pd.DataFrame({
    "sentence": sentences,
})

# Add length column
df["length"] = df["sentence"].str.split().apply(len)

# Save cleaned dataset
df.to_csv("data/igala_sentences.csv", index=False)

print(f"Saved {len(df)} sentences to data/igala_sentences.csv")

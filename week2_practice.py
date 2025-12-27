import pandas as pd
import string

# 1. Read raw text
with open("igala.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
# 2. Basic cleaning
text = text.lower()
text = text.replace("ʼ", "")
text = text.translate(str.maketrans("", "", string.punctuation))


# 3. Split into sentences (lines)
sentences = text.splitlines()

# 4. Remove empty lines
sentences = [s for s in sentences if s.strip() != ""]

# 5. Create DataFrame
df = pd.DataFrame({"sentence": sentences})

# 6. Add sentence length
df["length"] = df["sentence"].str.len()

# 7. Filter very short sentences
df = df[df["length"] > 3]

print(df.head())
print("Total sentences:", len(df))

#EDA (PRACTICAL,
print(df.describe())

df = df.reset_index(drop=True)

# 8. Save cleaned dataset
df.to_csv("data/igala_sentences.csv", index=False)
print("Saved to data/igala_sentences.csv")

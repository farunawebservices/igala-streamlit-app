import string
from collections import Counter

with open("igala.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()
text = text.replace("ʼ", "")
text = text.translate(str.maketrans("", "", string.punctuation))
words = text.split()
words = [w for w in words if len(w) > 1]

freq = Counter(words)
print(freq.most_common(20))
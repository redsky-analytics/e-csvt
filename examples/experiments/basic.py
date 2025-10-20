#%%

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2')

# Your names to compare
names = ["William", "Bill", "Margaret", "Maggie", "Robert", "Bob", "Elizabeth", "Liz"]

# Generate embeddings
embeddings = model.encode(names)

# Calculate cosine similarity between all pairs
similarity_matrix = cosine_similarity(embeddings)

# Find nickname matches (with threshold)
threshold = 0.5  # Adjust based on your needs

for i in range(len(names)):
    for j in range(i+1, len(names)):
        similarity = similarity_matrix[i][j]
        if similarity > threshold:
            print(f"{names[i]} ↔ {names[j]}: {similarity:.3f}")
# %%

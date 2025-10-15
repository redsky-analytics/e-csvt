#%%
from sentence_transformers import SentenceTransformer, util

# Test a few models
model = SentenceTransformer('all-mpnet-base-v2')

names = ["William", "Bill", "Margaret", "Maggie", "Robert", "Bob", "Elizabeth", "Liz", "Bisl"]
embeddings = model.encode(names)

# Compute similarities
similarities = util.cos_sim(embeddings, embeddings)

# Check William-Bill similarity
print(f"William-Bill: {similarities[0][1]:.3f}")
print(f"Margaret-Maggie: {similarities[2][3]:.3f}")
print(f"William-Margaret: {similarities[0][2]:.3f}")
print(f"Bisl-Bill: {similarities[8][1]:.3f}")

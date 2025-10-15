#%%
import jellyfish  # pip install jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_match(name1, name2, model, semantic_weight=0.7):
    # Semantic similarity
    embeddings = model.encode([name1, name2])
    semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Phonetic similarity
    phonetic_sim = jellyfish.jaro_winkler_similarity(name1.lower(), name2.lower())
    print(f"Phonetic similarity: {phonetic_sim:.3f}")
    print(f"Semantic similarity: {semantic_sim:.3f}")
    # Combined score
    return semantic_weight * semantic_sim + (1 - semantic_weight) * phonetic_sim

model = SentenceTransformer('all-MiniLM-L6-v2')
score = hybrid_match("William", "Bill", model)
print(f"Match score: {score:.3f}")
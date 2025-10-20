#%%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NicknameMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def find_nicknames(self, full_names, nicknames, threshold=0.5):
        """
        Find potential nickname matches
        
        Args:
            full_names: List of full names
            nicknames: List of potential nicknames
            threshold: Similarity threshold (0-1)
        
        Returns:
            Dictionary mapping nicknames to likely full names
        """
        # Encode both lists
        full_embeddings = self.model.encode(full_names)
        nick_embeddings = self.model.encode(nicknames)
        
        # Calculate similarities
        similarities = cosine_similarity(nick_embeddings, full_embeddings)
        
        matches = {}
        for i, nickname in enumerate(nicknames):
            best_match_idx = similarities[i].argmax()
            best_score = similarities[i][best_match_idx]
            
            if best_score > threshold:
                matches[nickname] = {
                    'full_name': full_names[best_match_idx],
                    'confidence': float(best_score)
                }
        
        return matches

# Usage
matcher = NicknameMatcher()

full_names = ["William", "Margaret", "Robert", "Elizabeth", "Christopher"]
nicknames = ["Bill", "Maggie", "Bob", "Liz", "Chris"]

results = matcher.find_nicknames(full_names, nicknames, threshold=0.4)

for nick, match in results.items():
    print(f"{nick} → {match['full_name']} (confidence: {match['confidence']:.3f})")
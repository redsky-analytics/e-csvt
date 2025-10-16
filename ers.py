"""
Hybrid Entity Resolution System
Combines blocking, fuzzy matching, and semantic similarity for maximum accuracy
Optimized for large datasets (1.5M+ records)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# Fix for multiprocessing issues with sentence-transformers on all platforms
# Prevents semaphore leaks and crashes on Windows/macOS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@dataclass
class MatchResult:
    """Store match results with detailed scores"""
    id1: int
    id2: int
    composite_score: float
    email_score: float
    first_name_score: float
    last_name_score: float
    company_score: float
    semantic_name_score: float
    match_type: str


class HybridEntityResolver:
    """
    Advanced entity resolution using multiple techniques:
    - Blocking to reduce search space
    - Fuzzy string matching
    - Semantic similarity via sentence transformers
    - Weighted scoring
    """
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_semantic: bool = True,
                 batch_size: int = 1000,
                 field_mapping: Dict[str, str] = None,
                 field_weights: Dict[str, float] = None,
                 use_embedding_blocking: bool = False,
                 embedding_block_k: int = 50,
                 embedding_similarity_threshold: float = 0.75):
        """
        Initialize the resolver

        Args:
            model_name: Sentence transformer model to use
            use_semantic: Whether to use semantic matching (slower but more accurate)
            batch_size: Batch size for semantic encoding
            field_mapping: Dict mapping field roles to CSV column names (None to skip field)
                          Example: {'email': 'email_address', 'first_name': 'fname',
                                   'last_name': None, 'company': 'org'}
                          Default: {'email': 'email', 'first_name': 'first_name',
                                   'last_name': 'last_name', 'company': 'company'}
            field_weights: Dict with weights for each field
                          Default: {'email': 0.35, 'last_name': 0.20, 'first_name': 0.15,
                                   'company': 0.15, 'semantic_name': 0.15}
            use_embedding_blocking: Whether to use embedding-based blocking (requires use_semantic=True)
            embedding_block_k: Number of nearest neighbors to include in embedding blocks
            embedding_similarity_threshold: Minimum cosine similarity for embedding blocking
        """
        self.use_semantic = use_semantic
        self.batch_size = batch_size
        self.use_embedding_blocking = use_embedding_blocking and use_semantic
        self.embedding_block_k = embedding_block_k
        self.embedding_similarity_threshold = embedding_similarity_threshold

        # Set up field mapping (None means field is not used)
        if field_mapping is None:
            self.field_mapping = {
                'email': 'email',
                'first_name': 'first_name',
                'last_name': 'last_name',
                'company': 'company'
            }
        else:
            self.field_mapping = field_mapping

        # Determine which fields are active (not None)
        self.active_fields = {k: v for k, v in self.field_mapping.items() if v is not None}

        if use_semantic:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(model_name)
            print(f"Model loaded: {model_name}")
        else:
            self.model = None

        # Set up scoring weights
        default_weights = {
            'email': 0.35,
            'last_name': 0.20,
            'first_name': 0.15,
            'company': 0.15,
            'semantic_name': 0.15  # Only used if use_semantic=True
        }

        if field_weights is None:
            self.base_weights = default_weights
        else:
            # Merge user weights with defaults
            self.base_weights = {**default_weights, **field_weights}

        # Calculate normalized weights based on active fields
        self._normalize_weights()

        # Precomputed embeddings cache (will store numpy arrays indexed by record index)
        self.embeddings_cache = None  # Initialized during find_duplicates()

    def _normalize_weights(self):
        """Normalize weights based on active fields and redistribute if fields are missing"""
        active_weight_fields = []

        # Determine which weight fields are active
        for field in ['email', 'first_name', 'last_name', 'company']:
            if field in self.active_fields:
                active_weight_fields.append(field)

        # Add semantic if enabled
        if self.use_semantic and ('first_name' in self.active_fields or 'last_name' in self.active_fields):
            active_weight_fields.append('semantic_name')

        if not active_weight_fields:
            raise ValueError("At least one field must be active for matching")

        # Extract weights for active fields
        active_weights = {field: self.base_weights.get(field, 0.0) for field in active_weight_fields}

        # Normalize to sum to 1.0
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            self.weights = {field: weight / total_weight for field, weight in active_weights.items()}
        else:
            # Equal weights if all are 0
            equal_weight = 1.0 / len(active_weight_fields)
            self.weights = {field: equal_weight for field in active_weight_fields}
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for matching"""
        df = df.copy()

        print("Preprocessing data...")

        # Create standardized fields for active fields only
        if 'first_name' in self.active_fields:
            col = self.active_fields['first_name']
            df['first_name_clean'] = df[col].fillna('').str.strip().str.lower()
        else:
            df['first_name_clean'] = ''

        if 'last_name' in self.active_fields:
            col = self.active_fields['last_name']
            df['last_name_clean'] = df[col].fillna('').str.strip().str.lower()
        else:
            df['last_name_clean'] = ''

        if 'email' in self.active_fields:
            col = self.active_fields['email']
            df['email_clean'] = df[col].fillna('').str.strip().str.lower()
        else:
            df['email_clean'] = ''

        if 'company' in self.active_fields:
            col = self.active_fields['company']
            df['company_clean'] = df[col].fillna('').str.strip().str.lower()
        else:
            df['company_clean'] = ''

        # Create full name if name fields exist
        df['full_name'] = df['first_name_clean'] + ' ' + df['last_name_clean']

        # Create blocking keys (only if relevant fields are active)
        if 'last_name' in self.active_fields:
            df['last_name_3char'] = df['last_name_clean'].str[:3]
        else:
            df['last_name_3char'] = ''

        if 'first_name' in self.active_fields:
            df['first_initial'] = df['first_name_clean'].str[0]
        else:
            df['first_initial'] = ''

        if 'email' in self.active_fields:
            df['email_domain'] = df['email_clean'].str.split('@').str[-1]
        else:
            df['email_domain'] = ''

        print(f"[OK] Preprocessed {len(df):,} records")
        print(f"  Active fields: {', '.join(self.active_fields.keys())}")
        return df
    
    def create_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create blocking keys to reduce comparison space
        Each block groups potentially similar records
        """
        blocks = defaultdict(list)

        print("Creating blocking keys...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building blocks", unit="records"):
            # Block 1: Exact email match (highest priority)
            if 'email' in self.active_fields and row['email_clean']:
                blocks[f"email_exact:{row['email_clean']}"].append(idx)

            # Block 2: Email domain (catches typos in name part)
            if 'email' in self.active_fields and row['email_domain']:
                blocks[f"email_domain:{row['email_domain']}"].append(idx)

            # Block 3: Last name first 3 chars + first initial
            if ('last_name' in self.active_fields and 'first_name' in self.active_fields and
                row['last_name_3char'] and row['first_initial']):
                blocks[f"name_key:{row['last_name_3char']}{row['first_initial']}"].append(idx)

            # Block 4: Soundex of last name
            if 'last_name' in self.active_fields and row['last_name_clean']:
                soundex = self._soundex(row['last_name_clean'])
                blocks[f"soundex:{soundex}"].append(idx)

            # Block 5: Company name first word
            if 'company' in self.active_fields and row['company_clean']:
                company_words = row['company_clean'].split()
                if company_words:
                    first_word = company_words[0]
                    if len(first_word) > 3:
                        blocks[f"company_word:{first_word}"].append(idx)

            # Block 6: First and last name tokens
            if ('first_name' in self.active_fields and 'last_name' in self.active_fields and
                row['first_name_clean'] and row['last_name_clean']):
                blocks[f"name_combo:{row['first_name_clean'][:2]}{row['last_name_clean'][:2]}"].append(idx)

        # Filter blocks: keep only blocks with 2-500 records
        # (too small = no matches, too large = too many comparisons)
        filtered_blocks = {
            k: v for k, v in blocks.items()
            if 2 <= len(v) <= 500
        }

        total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in filtered_blocks.values())
        print(f"[OK] Created {len(filtered_blocks):,} blocks with ~{total_pairs:,} potential comparisons")

        return filtered_blocks

    def create_embedding_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create blocks using embedding-based nearest neighbor search

        Args:
            df: Preprocessed dataframe

        Returns:
            Dictionary of embedding-based blocks
        """
        if not self.use_embedding_blocking or self.embeddings_cache is None:
            return {}

        try:
            import faiss
        except ImportError:
            print("Warning: FAISS not installed. Skipping embedding-based blocking.")
            print("Install with: pip install faiss-cpu  (or faiss-gpu for GPU support)")
            return {}

        print("\nCreating embedding-based blocks...")

        embeddings = self.embeddings_cache
        n_records = len(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (for normalized vectors = cosine similarity)

        # Add embeddings to index
        index.add(embeddings)

        print(f"  Built FAISS index with {n_records:,} vectors ({dimension} dimensions)")

        # Search for k nearest neighbors for each record
        k = min(self.embedding_block_k + 1, n_records)  # +1 because self is always nearest
        distances, indices = index.search(embeddings, k)

        print(f"  Finding {self.embedding_block_k} nearest neighbors per record...")

        # Create blocks based on nearest neighbors
        blocks = defaultdict(list)
        pairs_added = 0

        for record_idx in tqdm(range(n_records), desc="Building embedding blocks", unit="records"):
            # Get k nearest neighbors (excluding self at index 0)
            neighbor_indices = indices[record_idx][1:]  # Skip first (self)
            neighbor_distances = distances[record_idx][1:]

            # Only include neighbors above similarity threshold
            for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
                if distance >= self.embedding_similarity_threshold:
                    # Create a unique block key for this pair
                    # Use sorted indices to avoid duplicate blocks for (A,B) and (B,A)
                    block_key = f"emb_nn:{min(record_idx, neighbor_idx)}_{max(record_idx, neighbor_idx)}"

                    # Add both records to this block
                    if block_key not in blocks:
                        blocks[block_key] = [record_idx, int(neighbor_idx)]
                        pairs_added += 1

        print(f"[OK] Created {len(blocks):,} embedding-based blocks ({pairs_added:,} potential comparisons)")

        return blocks

    @staticmethod
    def _soundex(name: str) -> str:
        """Generate Soundex code for phonetic matching"""
        if not name or len(name) == 0:
            return "0000"
        
        name = name.upper()
        soundex = name[0]
        
        # Soundex mapping
        mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        prev_code = mapping.get(name[0], '0')
        
        for char in name[1:]:
            code = mapping.get(char, '0')
            if code != '0' and code != prev_code:
                soundex += code
                prev_code = code
            if len(soundex) >= 4:
                break
        
        return (soundex + '000')[:4]

    def _precompute_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Pre-compute embeddings for all records

        Args:
            df: Preprocessed dataframe with 'full_name' column

        Returns:
            numpy array of embeddings (shape: [n_records, embedding_dim])
        """
        if not self.use_semantic:
            return None

        if 'first_name' not in self.active_fields and 'last_name' not in self.active_fields:
            return None

        print("\nPre-computing embeddings for all records...")

        # Get full names
        full_names = df['full_name'].tolist()

        # Compute embeddings in batches
        embeddings = self.model.encode(
            full_names,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Pre-normalize for cosine similarity
        )

        # Convert to float32 for memory efficiency
        embeddings = embeddings.astype(np.float32)

        memory_mb = embeddings.nbytes / (1024 * 1024)
        print(f"[OK] Pre-computed {len(embeddings):,} embeddings ({embeddings.shape[1]} dimensions)")
        print(f"  Memory usage: {memory_mb:.1f} MB")

        return embeddings

    def _compute_fuzzy_scores(self, r1: pd.Series, r2: pd.Series) -> Dict[str, float]:
        """Compute fuzzy matching scores between two records"""
        scores = {}

        # Email matching (exact or fuzzy) - only if active
        if 'email' in self.active_fields:
            if r1['email_clean'] and r2['email_clean']:
                if r1['email_clean'] == r2['email_clean']:
                    scores['email'] = 1.0
                else:
                    scores['email'] = fuzz.ratio(r1['email_clean'], r2['email_clean']) / 100.0
            else:
                scores['email'] = 0.0
        else:
            scores['email'] = 0.0

        # First name - only if active
        if 'first_name' in self.active_fields:
            scores['first_name'] = fuzz.ratio(r1['first_name_clean'], r2['first_name_clean']) / 100.0
        else:
            scores['first_name'] = 0.0

        # Last name - only if active
        if 'last_name' in self.active_fields:
            scores['last_name'] = fuzz.ratio(r1['last_name_clean'], r2['last_name_clean']) / 100.0
        else:
            scores['last_name'] = 0.0

        # Company (token set ratio handles word order differences) - only if active
        if 'company' in self.active_fields:
            if r1['company_clean'] and r2['company_clean']:
                scores['company'] = fuzz.token_set_ratio(r1['company_clean'], r2['company_clean']) / 100.0
            else:
                scores['company'] = 0.0
        else:
            scores['company'] = 0.0

        return scores
    
    def _compute_semantic_score(self, idx1: int, idx2: int) -> float:
        """
        Compute semantic similarity between two records using pre-computed embeddings

        Args:
            idx1: Index of first record
            idx2: Index of second record

        Returns:
            Cosine similarity score (0-1)
        """
        if not self.use_semantic or self.embeddings_cache is None:
            return 0.0

        # Look up pre-computed embeddings
        emb1 = self.embeddings_cache[idx1]
        emb2 = self.embeddings_cache[idx2]

        # Compute cosine similarity (embeddings are already normalized)
        # For normalized vectors, cosine similarity = dot product
        similarity = np.dot(emb1, emb2)

        return float(similarity)
    
    def _batch_compute_semantic_scores(self, name_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute semantic scores in batches for efficiency"""
        if not self.use_semantic or not name_pairs:
            return [0.0] * len(name_pairs)
        
        scores = []
        
        # Process in batches
        for i in range(0, len(name_pairs), self.batch_size):
            batch = name_pairs[i:i + self.batch_size]
            
            # Flatten pairs for encoding
            all_names = []
            for n1, n2 in batch:
                all_names.extend([n1, n2])
            
            # Encode all names at once
            embeddings = self.model.encode(all_names)
            
            # Compute similarities for pairs
            for j in range(0, len(embeddings), 2):
                sim = cosine_similarity([embeddings[j]], [embeddings[j+1]])[0][0]
                scores.append(float(sim))
        
        return scores
    
    def compare_records(self, df: pd.DataFrame, idx1: int, idx2: int) -> MatchResult:
        """Compare two records and return detailed match result"""
        r1, r2 = df.loc[idx1], df.loc[idx2]

        # Compute fuzzy scores
        fuzzy_scores = self._compute_fuzzy_scores(r1, r2)

        # Compute semantic score for names (only if name fields are active)
        # Now uses pre-computed embeddings indexed by record index
        if self.use_semantic and ('first_name' in self.active_fields or 'last_name' in self.active_fields):
            semantic_score = self._compute_semantic_score(idx1, idx2)
        else:
            semantic_score = 0.0

        # Determine match type based on active fields
        match_type = "weak_match"  # default

        if 'email' in self.active_fields and fuzzy_scores['email'] == 1.0:
            match_type = "exact_email"
        elif 'email' in self.active_fields and fuzzy_scores['email'] > 0.9:
            match_type = "fuzzy_email"
        elif ('last_name' in self.active_fields and 'first_name' in self.active_fields and
              fuzzy_scores['last_name'] > 0.9 and fuzzy_scores['first_name'] > 0.8):
            match_type = "strong_name"
        elif self.use_semantic and semantic_score > 0.85:
            match_type = "semantic_name"

        # Compute weighted composite score using normalized weights
        composite = 0.0

        for field in ['email', 'first_name', 'last_name', 'company']:
            if field in self.weights:
                composite += self.weights[field] * fuzzy_scores[field]

        if 'semantic_name' in self.weights:
            composite += self.weights['semantic_name'] * semantic_score

        return MatchResult(
            id1=idx1,
            id2=idx2,
            composite_score=composite,
            email_score=fuzzy_scores['email'],
            first_name_score=fuzzy_scores['first_name'],
            last_name_score=fuzzy_scores['last_name'],
            company_score=fuzzy_scores['company'],
            semantic_name_score=semantic_score,
            match_type=match_type
        )

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file and validate that mapped columns exist

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with validated columns
        """
        print(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df):,} records")

        # Validate that all non-None mapped columns exist
        missing_cols = []
        for field_name, col_name in self.active_fields.items():
            if col_name not in df.columns:
                missing_cols.append(f"{field_name} -> '{col_name}'")

        if missing_cols:
            raise ValueError(
                f"Missing mapped columns in CSV:\n  " + "\n  ".join(missing_cols) +
                f"\n\nAvailable columns: {', '.join(df.columns)}"
            )

        return df

    def find_duplicates(self,
                       data,
                       threshold: float = 0.80,
                       max_comparisons: int = None) -> pd.DataFrame:
        """
        Find duplicate records in dataset

        Args:
            data: Either a DataFrame or path to CSV file
            threshold: Minimum composite score to consider a match (0-1)
            max_comparisons: Optional limit on comparisons (for testing)

        Returns:
            DataFrame of potential duplicates with scores
        """
        # Load data if it's a file path
        if isinstance(data, str):
            df = self.load_csv(data)
        else:
            df = data
        print(f"\n{'='*60}")
        print("HYBRID ENTITY RESOLUTION")
        print(f"{'='*60}")
        print(f"Records: {len(df):,}")
        print(f"Threshold: {threshold}")
        print(f"Semantic matching: {'Enabled' if self.use_semantic else 'Disabled'}")
        if self.use_embedding_blocking:
            print(f"Embedding blocking: Enabled (k={self.embedding_block_k}, threshold={self.embedding_similarity_threshold})")
        
        # Preprocess
        df_clean = self.preprocess_data(df)

        # Pre-compute embeddings for all records (if semantic matching is enabled)
        self.embeddings_cache = self._precompute_embeddings(df_clean)

        # Create string-based blocks
        blocks = self.create_blocks(df_clean)

        # Create embedding-based blocks if enabled
        if self.use_embedding_blocking:
            embedding_blocks = self.create_embedding_blocks(df_clean)
            # Merge embedding blocks with string-based blocks
            blocks.update(embedding_blocks)
            total_blocks = len(blocks)
            print(f"[OK] Total blocks after merging: {total_blocks:,}")
        
        # Calculate total potential comparisons for progress bar
        total_potential_comparisons = sum(len(v) * (len(v) - 1) // 2 for v in blocks.values())
        if max_comparisons:
            total_potential_comparisons = min(total_potential_comparisons, max_comparisons)
        
        # Find matches within blocks
        matches = []
        compared = set()
        comparison_count = 0
        
        print("\nComparing records within blocks...")
        
        # Create progress bar
        pbar = tqdm(total=total_potential_comparisons, desc="Finding matches", unit="comparisons")
        
        for block_key, indices in blocks.items():
            # Compare all pairs in block
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    
                    # Avoid duplicate comparisons across blocks
                    pair = tuple(sorted([idx1, idx2]))
                    if pair in compared:
                        continue
                    compared.add(pair)
                    
                    comparison_count += 1
                    pbar.update(1)
                    
                    # Apply max_comparisons limit if set (for testing)
                    if max_comparisons and comparison_count >= max_comparisons:
                        pbar.close()
                        print(f"\n[OK] Reached max comparisons limit: {max_comparisons:,}")
                        break
                    
                    # Compare records
                    result = self.compare_records(df_clean, idx1, idx2)
                    
                    if result.composite_score >= threshold:
                        matches.append(result)
                        pbar.set_postfix({'matches': len(matches)}, refresh=False)
                
                if max_comparisons and comparison_count >= max_comparisons:
                    break
            
            if max_comparisons and comparison_count >= max_comparisons:
                break
        
        pbar.close()
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total comparisons: {comparison_count:,}")
        print(f"Matches found: {len(matches)}")
        print(f"Reduction ratio: {len(df)*(len(df)-1)/2 / max(comparison_count, 1):.1f}x")
        
        # Convert to DataFrame
        if matches:
            # Build results with dynamic fields
            results_list = []
            for m in matches:
                result_row = {
                    'id1': df.loc[m.id1, 'id'] if 'id' in df.columns else m.id1,
                    'id2': df.loc[m.id2, 'id'] if 'id' in df.columns else m.id2,
                    'index1': m.id1,  # Keep internal index for clustering
                    'index2': m.id2,
                    'composite_score': round(m.composite_score, 4),
                    'email_score': round(m.email_score, 4),
                    'first_name_score': round(m.first_name_score, 4),
                    'last_name_score': round(m.last_name_score, 4),
                    'company_score': round(m.company_score, 4),
                    'semantic_score': round(m.semantic_name_score, 4),
                    'match_type': m.match_type,
                }

                # Include original data for review using actual column names
                for field_name in ['first_name', 'last_name', 'email', 'company']:
                    if field_name in self.active_fields:
                        col = self.active_fields[field_name]
                        result_row[f'{field_name}_1'] = df.loc[m.id1, col]
                        result_row[f'{field_name}_2'] = df.loc[m.id2, col]

                results_list.append(result_row)

            results_df = pd.DataFrame(results_list)
            
            # Sort by score
            results_df = results_df.sort_values('composite_score', ascending=False)
            
            # Print summary by match type
            print("\nMatches by type:")
            for match_type, count in results_df['match_type'].value_counts().items():
                print(f"  {match_type}: {count}")
            
            return results_df
        else:
            print("No matches found above threshold")
            return pd.DataFrame()
    
    def cluster_duplicates(self, matches_df: pd.DataFrame, original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Group duplicate pairs into clusters
        
        Args:
            matches_df: DataFrame of matches from find_duplicates()
            original_df: Original DataFrame with 'id' column (optional)
        
        Returns:
            DataFrame with cluster_id for each record
        """
        from collections import defaultdict
        
        print("\nClustering duplicates...")
        
        # Use index1/index2 if available (when id column exists), otherwise use id1/id2
        id_col1 = 'index1' if 'index1' in matches_df.columns else 'id1'
        id_col2 = 'index2' if 'index2' in matches_df.columns else 'id2'
        
        # Build graph of connections
        graph = defaultdict(set)
        for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Building graph", unit="pairs"):
            graph[row[id_col1]].add(row[id_col2])
            graph[row[id_col2]].add(row[id_col1])
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            visited.add(node)
            cluster.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for node in tqdm(list(graph.keys()), desc="Finding clusters", unit="nodes"):
            if node not in visited:
                cluster = set()
                dfs(node, cluster)
                clusters.append(cluster)
        
        # Create cluster mapping
        cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for record_id in cluster:
                cluster_map[record_id] = cluster_id
        
        print(f"[OK] Clustered {len(cluster_map):,} records into {len(clusters):,} groups")
        
        # Build result with original IDs if available
        result_data = []
        for record_index, cluster_id in cluster_map.items():
            row_data = {
                'record_index': record_index,
                'cluster_id': cluster_id
            }
            
            # Add original ID if available
            if original_df is not None and 'id' in original_df.columns:
                row_data['record_id'] = original_df.loc[record_index, 'id']
                # Also add the active fields for easy reference
                for field_name in ['first_name', 'last_name', 'email', 'company']:
                    if field_name in self.active_fields:
                        col = self.active_fields[field_name]
                        row_data[field_name] = original_df.loc[record_index, col]
            
            result_data.append(row_data)
        
        result_df = pd.DataFrame(result_data)
        
        # Sort by cluster_id and then by record_id if available
        sort_cols = ['cluster_id']
        if 'record_id' in result_df.columns:
            sort_cols.append('record_id')
        result_df = result_df.sort_values(sort_cols)
        
        return result_df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_sample_data(csv_path: str = 'sample_data.csv') -> pd.DataFrame:
    """
    Create sample dataset for testing and demonstration

    Args:
        csv_path: Optional path to save CSV file. If None, doesn't save to file.

    Returns:
        DataFrame with sample duplicate records
    """
    data = {
        'id': [
            'usr_8f3k2j9sd0fj23k4',
            'usr_2k3j4f9s0dfj2k3f',
            'usr_9f2k3j4s0dfj23k4',
            'usr_3k4j5f0s1dfj3k4f',
            'usr_0f3k4j5s1dfj34k5',
            'usr_4k5j6f1s2dfj4k5f',
            'usr_1f4k5j6s2dfj45k6',
            'usr_5k6j7f2s3dfj5k6f',
            'usr_2f5k6j7s3dfj56k7',
            'usr_6k7j8f3s4dfj6k7f',
            'usr_3f6k7j8s4dfj67k8',
            'usr_7k8j9f4s5dfj7k8f',
            'usr_4f7k8j9s5dfj78k9',
            'usr_8k9j0f5s6dfj8k9f',
            'usr_5f8k9j0s6dfj89k0',
        ],
        'first_name': [
            'William', 'Bill', 'Margaret', 'Maggie', 'Robert',
            'Bob', 'Elizabeth', 'Liz', 'Christopher', 'Chris',
            'William', 'William', 'John', 'Jane', 'Jane'
        ],
        'last_name': [
            'Smith', 'Smith', 'Johnson', 'Johnson', 'Williams',
            'Williams', 'Brown', 'Brown', 'Davis', 'Davis',
            'Smyth', 'Smith', 'Doe', 'Doe', 'Dowe'
        ],
        'email': [
            'william.smith@company.com', 'bill.smith@company.com',
            'margaret.j@company.com', 'maggie.j@company.com',
            'robert.w@company.com', 'bob.williams@company.com',
            'elizabeth.b@company.com', 'liz.brown@company.com',
            'chris.davis@company.com', 'christopher.davis@company.com',
            'w.smith@company.com', 'william.smith@company.com',
            'john.doe@company.com', 'jane.doe@otherco.com', 'jane.doe@company.com'
        ],
        'company': [
            'Acme Corp', 'Acme Corp', 'Acme Corporation', 'Acme Corp',
            'Beta Inc', 'Beta Inc', 'Gamma LLC', 'Gamma LLC',
            'Delta Co', 'Delta Co', 'Acme Corp', 'Acme Corporation',
            'Epsilon Ltd', 'Zeta Corp', 'Zeta Corporation'
        ]
    }

    df = pd.DataFrame(data)
    df.index = range(len(df))  # Ensure integer index

    # Save to CSV if path provided
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Sample data saved to: {csv_path}")

    return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create sample dataset
    df = make_sample_data(csv_path=None)  # Don't save yet

    print("Sample dataset:")
    print(df[['id', 'first_name', 'last_name', 'email', 'company']])

    # ========================================================================
    # EXAMPLE 1: Default usage (standard field names)
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: Default Configuration")
    print("="*70)

    resolver = HybridEntityResolver(
        model_name='all-MiniLM-L6-v2',
        use_semantic=True,  # Set to False for faster processing without ML
        batch_size=1000
    )

    # Find duplicates (accepts DataFrame or CSV path)
    duplicates = resolver.find_duplicates(
        df,
        threshold=0.75,  # Adjust threshold as needed
        max_comparisons=None  # Remove limit for production
    )

    if len(duplicates) > 0:
        print("\n" + "="*60)
        print("TOP MATCHES")
        print("="*60)
        print(duplicates[['id1', 'id2', 'first_name_1', 'last_name_1', 'first_name_2', 'last_name_2',
                         'composite_score', 'match_type']].head(10))

        # Save results
        duplicates.to_csv('duplicates_found.csv', index=False)
        print("\nFull results saved to: duplicates_found.csv")

        # Cluster duplicates (pass original df for id mapping)
        clusters = resolver.cluster_duplicates(duplicates, original_df=df)
        clusters.to_csv('clusters.csv', index=False)
        print("Clusters saved to: clusters.csv")

        print("\nSample cluster output:")
        print(clusters.head(10))

    # ========================================================================
    # EXAMPLE 2: Custom field mapping (with None to skip fields)
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Custom Field Mapping (Skip Company Field)")
    print("="*70)

    # Map custom column names, set company to None to skip it
    resolver_custom = HybridEntityResolver(
        model_name='all-MiniLM-L6-v2',
        use_semantic=True,
        field_mapping={
            'email': 'email',
            'first_name': 'first_name',
            'last_name': 'last_name',
            'company': None  # Skip company field in matching
        }
    )

    duplicates_custom = resolver_custom.find_duplicates(df, threshold=0.75)
    if len(duplicates_custom) > 0:
        print(f"\nFound {len(duplicates_custom)} matches without using company field")

    # ========================================================================
    # EXAMPLE 3: Custom field weights
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Custom Field Weights (Email-Heavy)")
    print("="*70)

    # Emphasize email more heavily
    resolver_weighted = HybridEntityResolver(
        model_name='all-MiniLM-L6-v2',
        use_semantic=True,
        field_weights={
            'email': 0.50,        # Increased from 0.35
            'last_name': 0.20,
            'first_name': 0.15,
            'company': 0.10,      # Decreased from 0.15
            'semantic_name': 0.05 # Decreased from 0.15
        }
    )

    print(f"Normalized weights: {resolver_weighted.weights}")

    # ========================================================================
    # EXAMPLE 4: Loading from CSV file
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Loading from CSV File")
    print("="*70)

    # Create sample CSV file using helper function
    make_sample_data(csv_path='sample_data.csv')

    # Load and process CSV directly
    resolver_csv = HybridEntityResolver(
        model_name='all-MiniLM-L6-v2',
        use_semantic=False  # Faster without semantic matching
    )

    # Pass CSV file path directly
    duplicates_from_csv = resolver_csv.find_duplicates(
        'sample_data.csv',  # Can pass file path instead of DataFrame
        threshold=0.75
    )

    print(f"\nProcessed CSV and found {len(duplicates_from_csv)} matches")
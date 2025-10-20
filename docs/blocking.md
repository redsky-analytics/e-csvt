# Blocking in Entity Resolution

## Overview

**Blocking** is a technique that dramatically reduces the number of record comparisons needed for entity resolution by partitioning records into smaller groups (blocks) and only comparing records within the same block.

**The Problem:**
- Naive approach: Compare every record to every other record = O(n²) comparisons
- For 100,000 records: 4,999,950,000 comparisons (5 billion!)
- Computationally infeasible for large datasets

**The Solution:**
- Blocking: Group similar records together, only compare within groups
- Reduces comparisons by 100-10,000x
- Makes entity resolution tractable for millions of records

## Why Blocking is Necessary

### Naive Approach: Cartesian Comparison

Without blocking, you must compare every record pair:

```
Records: A, B, C, D, E

All pairs:
(A,B), (A,C), (A,D), (A,E)
(B,C), (B,D), (B,E)
(C,D), (C,E)
(D,E)

Total: n × (n-1) / 2 = 5 × 4 / 2 = 10 comparisons
```

**Scaling:**
| Records | Comparisons | Approximate Time |
|---------|-------------|------------------|
| 1,000 | 499,500 | 1 minute |
| 10,000 | 49,995,000 | 2 hours |
| 100,000 | 4,999,950,000 | 2 weeks |
| 1,000,000 | 499,999,500,000 | 15 years |

### Blocking Approach: Divide and Conquer

With blocking, records are grouped by shared characteristics:

```
Records: A₁, A₂, B₁, B₂, C₁

Blocks:
- Block "A": {A₁, A₂}  → Compare A₁ to A₂ only (1 comparison)
- Block "B": {B₁, B₂}  → Compare B₁ to B₂ only (1 comparison)
- Block "C": {C₁}      → No comparisons (singleton)

Total: 2 comparisons instead of 10
```

**Key Reduction:** Only compare records that share at least one blocking key.

---

## How Blocking Works

### Step 1: Generate Blocking Keys

For each record, create one or more **blocking keys** - simplified representations of the record:

```python
Record: John Smith, john.smith@company.com, Acme Corp

Blocking keys generated:
1. email_exact:john.smith@company.com
2. email_domain:company.com
3. name_key:smij  (last 3 chars + first initial)
4. soundex:S530   (phonetic code for "Smith")
5. company_word:acme
6. name_combo:josm (first 2 + last 2 chars)
```

### Step 2: Group Records into Blocks

Records sharing a blocking key are placed in the same block:

```
Block "email_domain:company.com":
  - John Smith (john.smith@company.com)
  - Jane Smith (jane.smith@company.com)
  - Jon Smith (jon.smith@company.com)

Block "soundex:S530":
  - John Smith
  - Jon Smyth
  - Jane Smithe
```

### Step 3: Compare Within Blocks Only

The system only compares records in the same block:

```
Block "email_domain:company.com" has 3 records:
  Compare: (John, Jane), (John, Jon), (Jane, Jon)
  = 3 comparisons

Without blocking:
  If these were part of 1,000 records, naive approach would require
  1,000 × 999 / 2 = 499,500 comparisons
```

### Step 4: Deduplicate Comparisons

If two records appear in multiple blocks together, they're only compared once:

```
John Smith and Jon Smyth appear in:
  - Block "email_domain:company.com"
  - Block "soundex:S530"

They're only compared once, not twice.
```

---

## Blocking Strategies

The system uses **six different blocking strategies** to maximize recall while minimizing comparisons.

### Block 1: Exact Email Match

```python
blocks[f"email_exact:{row['email_clean']}"].append(idx)
```

**Purpose:** Highest confidence matches - identical emails

**Example:**
```
john.doe@company.com → Block "email_exact:john.doe@company.com"
```

**Catches:**
- Exact email duplicates
- Same person with consistent email address

**Misses:**
- Email typos
- Multiple email addresses for same person

---

### Block 2: Email Domain

```python
blocks[f"email_domain:{row['email_domain']}"].append(idx)
```

**Purpose:** Catch typos in email username, same organization

**Example:**
```
john.doe@company.com  → Block "email_domain:company.com"
jon.doe@company.com   → Block "email_domain:company.com"  (typo caught!)
```

**Catches:**
- Email username typos (john vs jon)
- Different people at same company (for later filtering)

**Misses:**
- Same person with different email domains (work vs personal)

---

### Block 3: Name Key (Last 3 Chars + First Initial)

```python
blocks[f"name_key:{row['last_name_3char']}{row['first_initial']}"].append(idx)
```

**Purpose:** Group people with similar names

**Example:**
```
William Smith   → Block "name_key:smiw"
Will Smith      → Block "name_key:smiw"  (same!)
Bill Smith      → Block "name_key:smib"  (different - missed)
```

**Catches:**
- Name variations with same first initial
- Spelling variations in last names

**Misses:**
- Nicknames with different first initials (William/Bill, Robert/Bob)
- Short last names with typos

---

### Block 4: Soundex (Phonetic Matching)

```python
soundex = self._soundex(row['last_name_clean'])
blocks[f"soundex:{soundex}"].append(idx)
```

**Purpose:** Catch phonetically similar last names

**Soundex Algorithm:**
- Keeps first letter
- Maps consonants to digits based on sound
- Results in 4-character code

**Examples:**
```
Smith   → S530
Smyth   → S530  (same code - caught!)
Smithe  → S530  (same code - caught!)
Schmidt → S253  (different code - missed)
```

**Catches:**
- Common spelling variations (Smith/Smyth)
- Phonetically similar names

**Misses:**
- Names that sound different despite being same person
- Married name changes

---

### Block 5: Company Name First Word

```python
first_word = row['company_clean'].split()[0]
if len(first_word) > 3:
    blocks[f"company_word:{first_word}"].append(idx)
```

**Purpose:** Group people from same organization

**Note:** Only uses words longer than 3 characters (skips "The", "Inc", etc.)

**Examples:**
```
Acme Corp        → Block "company_word:acme"
Acme Corporation → Block "company_word:acme"  (caught!)
The Acme Company → (skipped - "the" is ≤ 3 chars)
```

**Catches:**
- Company name variations
- Colleagues at same organization

**Misses:**
- Person changed companies
- Company names starting with short words
- Company names with different first words

---

### Block 6: Name Combo (First 2 + Last 2)

```python
blocks[f"name_combo:{row['first_name_clean'][:2]}{row['last_name_clean'][:2]}"].append(idx)
```

**Purpose:** Additional name-based grouping

**Examples:**
```
John Smith     → Block "name_combo:josm"
Jonathan Smith → Block "name_combo:josm"  (caught!)
Jane Smith     → Block "name_combo:jasm"  (different)
```

**Catches:**
- Name abbreviations (Jon/Jonathan)
- Similar name patterns

**Misses:**
- Very short names
- Different names with coincidental same pattern

---

## Block Filtering

After generating blocks, the system **filters** them based on size:

```python
filtered_blocks = {
    k: v for k, v in blocks.items()
    if 2 <= len(v) <= 500
}
```

### Why Filter?

**Blocks with < 2 records:**
- No comparisons possible (can't compare 1 record to itself)
- Singletons are removed

**Blocks with > 500 records:**
- Too many comparisons (common values like "John Smith")
- Block size 500 = 500 × 499 / 2 = 124,750 comparisons
- These blocks are removed to prevent explosion of comparisons

**Trade-off:** Filtering improves performance but may miss some matches in very common value scenarios.

---

## Embedding-Based Blocking (Optional)

**New Feature:** Use semantic similarity for blocking via FAISS nearest neighbor search.

### How It Works

1. **Pre-compute embeddings** for all records using SentenceTransformer
2. **Build FAISS index** with normalized embeddings
3. **Find K nearest neighbors** for each record in embedding space
4. **Create blocks** containing semantically similar records

### Configuration

```python
resolver = HybridEntityResolver(
    use_semantic=True,
    use_embedding_blocking=True,
    embedding_block_k=50,                 # Number of nearest neighbors
    embedding_similarity_threshold=0.75   # Minimum cosine similarity
)
```

### Implementation: `ers.py:254-317`

```python
def create_embedding_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
    # Build FAISS index
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(embeddings)

    # Find k nearest neighbors
    distances, indices = index.search(embeddings, k)

    # Create blocks for neighbors above threshold
    for record_idx in range(n_records):
        for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
            if distance >= self.embedding_similarity_threshold:
                # Create block containing this pair
                blocks[f"emb_nn:{min(record_idx, neighbor_idx)}_{max(record_idx, neighbor_idx)}"] = [record_idx, neighbor_idx]
```

### Advantages

✅ **Catches semantic matches** missed by string matching
- Example: "William" and "Bill" are far apart in string space but close in embedding space

✅ **Robust to typos** that change phonetics
- Example: "Schmidt" vs "Smith" (different soundex) but semantically similar

✅ **Language-agnostic** with multilingual models

### Disadvantages

❌ **Computationally expensive**
- Must compute embeddings for all records
- Requires FAISS library
- Best with GPU for large datasets

❌ **May miss exact matches** (rare)
- Relies on embedding space geometry

❌ **Additional memory**
- ~146 MB for 100K records (float32 embeddings)

### When to Use

**Use embedding blocking when:**
- Many nickname variations (William/Bill, Robert/Bob)
- International names with encoding issues
- High recall is critical
- Have computational resources (GPU recommended)

**Don't use when:**
- Email data is reliable (string matching sufficient)
- Speed is critical
- Limited memory/compute
- Precision more important than recall

---

## Performance: Naive vs Blocking

### Example: 100,000 Records

**Naive approach:**
```
100,000 × 99,999 / 2 = 4,999,950,000 comparisons (5 billion)
Estimated time: 2 weeks on single CPU
```

**String-based blocking:**
```
Assume:
- 50,000 blocks
- Average 20 records per block
- 20 × 19 / 2 = 190 comparisons per block

Total: 50,000 × 190 = 9,500,000 comparisons
Reduction: 4,999,950,000 / 9,500,000 = 526x faster!
Estimated time: 5 minutes
```

**Hybrid blocking (string + embedding):**
```
String blocks: 9,500,000 comparisons
Embedding blocks: +2,000,000 comparisons (k=50 neighbors)

Total: ~11,500,000 comparisons
Reduction: 4,999,950,000 / 11,500,000 = 435x faster!
Estimated time: 7 minutes
Still massive improvement, with better recall
```

---

## Risk of Missing Matches

### The Fundamental Rule

> **Two records will ONLY be compared if they share at least ONE blocking key.**

If two duplicate records share **ZERO blocking keys**, they will **NEVER be compared**, and the match will be **MISSED**.

### Scenario 1: No Shared Blocking Keys

**Example: Career change with nickname**

```
Record A:
  Name: William Thompson
  Email: will.thompson@oldcompany.com
  Company: OldCompany Inc

Record B:
  Name: Bill Thomas
  Email: b.thomas@newcompany.com
  Company: NewCompany LLC

Blocking key analysis:
✗ email_exact: Different emails
✗ email_domain: oldcompany.com vs newcompany.com
✗ name_key: "thow" vs "thob"
✗ soundex: T512 (Thompson) vs T520 (Thomas)
✗ company_word: "oldcompany" vs "newcompany"
✗ name_combo: "with" vs "bith"

Result: ZERO shared keys → NEVER COMPARED → MATCH MISSED
```

### Scenario 2: Common Value Blocks Filtered Out

**Example: John Smith (very common name)**

```
Suppose 600 people have last name "Smith":
- They all share soundex:S530
- Block size = 600 > 500 (filtered out!)

Result: Some Smith variations won't be compared
```

### Scenario 3: Missing Fields

```
Record A:
  Name: (empty)
  Email: contact@company.com
  Company: (empty)

Record B:
  Name: John Smith
  Email: john@company.com
  Company: Acme Corp

Blocking keys:
✗ name_key: Record A has no name
✗ soundex: Record A has no name
✗ company_word: Record A has no company
✓ email_domain: company.com (ONLY SHARED KEY)

If email_domain block is filtered (>500 records): MATCH MISSED
```

### How to Detect Missed Matches

1. **Analyze false negatives** from performance evaluation:
   ```bash
   uv run analyze_clustering_performance.py --clusters clusters.csv --ground-truth truth.csv
   ```

2. **Check for records with no blocking keys** (singletons)

3. **Review common value blocks** that were filtered out

### Mitigation Strategies

#### Strategy 1: Add More Blocking Keys

Current system uses 6 strategies. You could add:

**7. First Name Soundex**
```python
first_soundex = self._soundex(row['first_name_clean'])
blocks[f"first_soundex:{first_soundex}"].append(idx)
```
Catches nickname variations (Bill/Will same soundex)

**8. Email Username Only**
```python
email_user = row['email_clean'].split('@')[0]
blocks[f"email_user:{email_user}"].append(idx)
```
Catches same username with different domains

**9. Phone Number** (if available)
```python
blocks[f"phone:{row['phone_clean']}"].append(idx)
```

#### Strategy 2: Adjust Block Size Limits

Modify filtering thresholds in `ers.py:242-247`:

```python
# Current
filtered_blocks = {k: v for k, v in blocks.items() if 2 <= len(v) <= 500}

# More lenient (accept larger blocks)
filtered_blocks = {k: v for k, v in blocks.items() if 2 <= len(v) <= 1000}
```

**Trade-off:** More comparisons vs. fewer missed matches

#### Strategy 3: Use Embedding-Based Blocking

Enable FAISS blocking to catch semantic matches:

```python
resolver = HybridEntityResolver(
    use_embedding_blocking=True,
    embedding_block_k=50
)
```

Catches matches missed by string-based blocking (e.g., William/Bill, Thompson/Thomas if semantically similar)

#### Strategy 4: Two-Pass Approach

**Pass 1:** Use current blocking (fast)
**Pass 2:** For unmatched records, use more lenient blocking

#### Strategy 5: Hybrid String + Embedding Blocking

Combine all strategies for maximum recall:

```python
resolver = HybridEntityResolver(
    use_semantic=True,
    use_embedding_blocking=True,
    embedding_block_k=50,
    embedding_similarity_threshold=0.70  # Lower threshold for more coverage
)
```

---

## Monitoring Blocking Effectiveness

### Metrics to Track

**1. Coverage Rate**
```
Coverage = Records in at least one block / Total records
```
High coverage (>95%) is good. Low coverage means many singletons.

**2. Block Size Distribution**
```
- How many blocks have 2-10 records? (good)
- How many blocks have 100-500 records? (acceptable)
- How many blocks exceed 500 records? (filtered)
```

**3. Comparison Reduction Ratio**
```
Reduction = Naive comparisons / Actual comparisons

Example: 4,999,950,000 / 9,500,000 = 526x
```
Higher is better (more efficient)

**4. False Negative Rate**
```
FN Rate = Missed matches (from ground truth) / Total matches

Example: 8 missed / 50 total = 16% FN rate
```
Lower is better (fewer missed matches)

### Output from System

The system reports blocking effectiveness:

```
[OK] Created 50,234 blocks with ~9,500,000 potential comparisons

...

RESULTS
============================================================
Total comparisons: 9,456,234
Matches found: 1,234
Reduction ratio: 526.3x
```

**Interpreting:**
- **526.3x reduction:** Excellent efficiency
- **1,234 matches found:** Depends on your data, but check against expected duplicates
- If **expected 1,500 matches but found 1,234:** Investigate 266 false negatives

---

## Best Practices

### 1. Understand Your Data

Before choosing blocking strategies:
- **How complete are fields?** (90% have email vs 50%)
- **Are names standardized?** (consistent format vs messy)
- **Common values?** (many "John Smith" vs unique names)
- **International characters?** (accents, non-Latin scripts)

### 2. Start Simple, Add Complexity

```
Phase 1: Use default string-based blocking
Phase 2: Evaluate performance, identify gaps
Phase 3: Add targeted blocking strategies for identified gaps
Phase 4: Enable embedding blocking if needed
```

### 3. Monitor and Tune

- Run performance evaluation regularly
- Track false negative patterns
- Adjust block size limits based on data characteristics
- Add blocking strategies for identified gaps

### 4. Balance Efficiency and Recall

**High efficiency (low comparisons):**
- Strict blocking (more strategies, tighter thresholds)
- Risk: More missed matches

**High recall (find all matches):**
- Lenient blocking (fewer strategies, looser thresholds)
- Risk: More comparisons, slower

**Find the sweet spot for your use case.**

### 5. Use Embedding Blocking Selectively

Don't always enable embedding blocking:
- **Enable:** Nickname-heavy data, high recall requirements
- **Disable:** Email-reliable data, speed-critical applications

### 6. Document Your Blocking Strategy

For reproducibility:
```python
# Document which strategies you're using and why
resolver = HybridEntityResolver(
    use_semantic=True,
    use_embedding_blocking=True,  # Enabled because of nickname variations
    embedding_block_k=50,          # Tuned to balance recall and speed
    field_mapping={
        'email': 'email_address',
        'first_name': 'fname',
        'last_name': 'lname',
        'company': 'org'
    }
)
```

---

## Summary

**Blocking is essential** for entity resolution at scale:

✅ **Reduces comparisons** by 100-10,000x
✅ **Makes O(n²) problem tractable** for millions of records
✅ **Multiple strategies** catch different types of matches
✅ **Tunable trade-off** between efficiency and recall
✅ **Embedding-based blocking** available for maximum recall

**Key principle:** Records only compared if they share at least one blocking key.

**Trade-off:** Blocking improves efficiency but introduces risk of missing matches.

**Solution:** Use multiple diverse blocking strategies and monitor false negatives.

---

## Related Files

- `ers.py:201-252`: String-based blocking implementation (`create_blocks()`)
- `ers.py:254-317`: Embedding-based blocking implementation (`create_embedding_blocks()`)
- `clustering.md`: How matches are grouped after comparison
- `README.md`: Overall workflow and usage

# YouTube Audience Analytics: Identifying Pet Owners from Comments

Estimating pet ownership (cat vs. dog) among a YouTube creator's audience from comment text, using
PySpark for distributed processing and a weak-supervision approach to labeling.

## The problem

There's no ground-truth "does this user own a pet" label in a comments dataset — ownership has to be
inferred from what people write. This project uses **weak supervision**: users who explicitly self-declare
ownership (e.g. "my cat", "I have a dog") become a high-precision labeled set, a text classifier is trained
on it, and that classifier is then applied to the full user base to estimate ownership for users who never
said so explicitly.

## Approach

- Distributed data loading and cleaning with PySpark
- Regex-based weak-supervision labeling on explicit ownership statements
- Balanced negative sampling to build a labeled training set (~1:1 positive/negative)
- TF-IDF features via `CountVectorizer` + `IDF` — chosen over `HashingTF` specifically so that model
  features stay mapped to real vocabulary words, enabling interpretability later
- Model comparison: Logistic Regression, Random Forest, and Gradient-Boosted Trees, evaluated on accuracy,
  AUC, F1, precision, and recall
- Interpretability: extracting the words most predictive of pet ownership from the Logistic Regression
  coefficients
- Applying the best model to the full user base to estimate cat/dog ownership at scale
- Creator-level rollup: which creators have the most cat-owner or dog-owner concentrated audiences

## Results

Dataset: 5,820,035 comments from 2,537,174 users across 4,241 creators.

| | Value |
|---|---:|
| Total comments | 5,820,035 |
| Unique users | 2,537,174 |
| Unique creators | 4,241 |

| Model (cat ownership) | Accuracy | AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.844 | 0.920 | 0.791 |
| Random Forest | 0.774 | 0.907 | 0.770 |
| GBT | 0.964 | 0.982 | 0.959 |

The dog-ownership GBT model performed similarly: 0.960 accuracy, 0.983 AUC, 0.958 F1.

Applying the best model (GBT) to the full user base: **130,112 predicted cat owners (5.13%)** and
**135,044 predicted dog owners (5.32%)** — roughly 7x the number of users who explicitly declared
ownership, which is the core payoff of the weak-supervision approach.

## Limitations

Regex-matched self-declarations are a noisy proxy for true ownership, not verified ground truth — some
pet owners never say so explicitly (false negatives), and some matched phrases may not reflect genuine
ownership (false positives, e.g. sarcasm). Cat and dog ownership are modeled as two independent binary
classifiers rather than a joint multi-label problem, and the regex patterns only capture fairly literal
English-language phrasings of ownership.

## Repo structure

```
.
├── youtube_audience_analysis.ipynb   # main notebook: cleaning, weak-supervision labeling,
│                                      # feature engineering, modeling, creator-level analysis
├── requirements.txt
└── README.md
```

## Data

This notebook expects a compressed CSV of YouTube comments with at least `userid`, `comment`, and
`creator_name` columns, referenced via the `DATA_PATH` variable in the notebook. The raw data is not
committed to this repo.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook youtube_audience_analysis.ipynb
```

PySpark requires a Java runtime (JDK 8/11/17) available on your machine; see the
[PySpark installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
if `SparkSession.builder.getOrCreate()` fails.

## Key takeaway

The words most predictive of cat ownership were clearly meaningful and cat-specific — `catnip`,
`kneading`, `chattering`, `trilling` — recognizable cat behaviors and sounds, which is a strong sign the
classifier learned real, topically relevant patterns rather than spurious correlations. Interestingly, the
words pushing toward "not a cat owner" were largely uninterpretable and topic-unrelated, suggesting the
negative class is simply too broad and heterogeneous for the linear model to find a coherent negative
signal, unlike the positive class which has its own distinctive vocabulary.

At the creator level, large general-interest channels (e.g. Brave Wilderness) showed lower cat/dog
audience concentration (~3.5%), while creators with more specifically pet-focused content (e.g. Brian
Barczyk, The Dodo) showed noticeably higher concentration (9-11%) — consistent with niche content
attracting a more concentrated relevant audience.

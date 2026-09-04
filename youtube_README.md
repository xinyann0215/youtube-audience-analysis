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

*Fill in with your actual numbers after running the notebook.*

| | Value |
|---|---:|
| Total comments | _fill in_ |
| Unique users | _fill in_ |
| Unique creators | _fill in_ |

| Model (cat ownership) | Accuracy | AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | _fill in_ | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ | _fill in_ |
| GBT | _fill in_ | _fill in_ | _fill in_ |

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

*(Write 2-3 sentences here once you've run the notebook: which words were most predictive of cat vs. dog
ownership, and what the creator-level results suggest about audience targeting.)*

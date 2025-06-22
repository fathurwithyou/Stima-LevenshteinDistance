# %%
# !pip install gensim

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import time
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats


import string
from gensim.parsing.preprocessing import remove_stopwords


plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# %%
"""
## Indonesian Stopwords
"""

# %%
def get_comprehensive_indonesian_stopwords():
    """
    Get comprehensive Indonesian stop words categorized by type
    """
    stopwords_dict = {
        "pronouns": [
            "saya",
            "aku",
            "kamu",
            "anda",
            "dia",
            "ia",
            "kita",
            "kami",
            "kalian",
            "mereka",
            "ku",
            "mu",
            "nya",
            "sih",
            "dong",
        ],
        "conjunctions": [
            "dan",
            "atau",
            "tetapi",
            "namun",
            "serta",
            "bahkan",
            "sedangkan",
            "padahal",
            "karena",
            "sebab",
            "akibat",
            "hingga",
            "sampai",
            "supaya",
            "agar",
            "meski",
            "walaupun",
            "kendati",
            "biarpun",
        ],
        "prepositions": [
            "di",
            "ke",
            "dari",
            "pada",
            "untuk",
            "dengan",
            "oleh",
            "dalam",
            "atas",
            "bawah",
            "depan",
            "belakang",
            "samping",
            "antara",
            "diantara",
            "melalui",
            "menuju",
            "terhadap",
            "tentang",
            "mengenai",
        ],
        "determiners": [
            "ini",
            "itu",
            "yang",
            "para",
            "sang",
            "si",
            "sebuah",
            "suatu",
            "beberapa",
            "semua",
            "seluruh",
            "setiap",
            "masing",
            "tiap",
        ],
        "auxiliaries": [
            "adalah",
            "ialah",
            "yaitu",
            "yakni",
            "akan",
            "telah",
            "sudah",
            "sedang",
            "tengah",
            "pernah",
            "bisa",
            "dapat",
            "boleh",
            "harus",
            "wajib",
            "perlu",
            "mau",
            "ingin",
            "hendak",
        ],
        "common_words": [
            "ada",
            "tidak",
            "bukan",
            "belum",
            "juga",
            "saja",
            "hanya",
            "masih",
            "lagi",
            "pula",
            "pun",
            "lah",
            "kah",
            "tah",
            "deh",
            "kok",
            "sih",
            "ya",
            "iya",
            "oh",
            "eh",
            "ah",
            "wah",
            "aduh",
            "duh",
        ],
        "question_words": [
            "apa",
            "siapa",
            "kapan",
            "dimana",
            "kemana",
            "darimana",
            "mengapa",
            "kenapa",
            "bagaimana",
            "berapa",
            "mana",
        ],
        "time_indicators": [
            "hari",
            "bulan",
            "tahun",
            "minggu",
            "jam",
            "menit",
            "detik",
            "waktu",
            "masa",
            "saat",
            "ketika",
            "sewaktu",
            "semasa",
            "selama",
            "sambil",
        ],
    }

    all_stopwords = set()
    for category, words in stopwords_dict.items():
        all_stopwords.update(words)

    return all_stopwords, stopwords_dict

# %%
"""
## Algorithm
"""

# %%
class settings:
    LABEL = "sentiment"
    

# %%
df = pd.read_csv(r"data\INA_TweetsPPKM_Labeled_Pure.csv", sep="\t")
df = df[["Tweet", "sentiment"]]
df.rename(columns={"Tweet": "text", "sentiment": settings.LABEL}, inplace=True)
df.head()

# %%
def create_regex_patterns():
    """
    Create comprehensive regex patterns for Indonesian text cleaning
    """
    patterns = {
        "urls": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "mentions": r"@[A-Za-z0-9_]+",
        'hashtags': r'#',
        "phone_numbers": r"\b\d{4,}\b",
        "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "repeated_chars": r"(.)\1{2,}",  
        "numbers": r"\b\d+\b",
        "special_chars": r"[^a-zA-Z\s]",
        "extra_whitespace": r"\s+",
        "short_words": r"\b\w{1,2}\b",  
        "repeated_words": r"\b(\w+)(?:\s+\1\b)+",  
        "emoticons": r"[:-;=][oO\-]?[D\)\]\(\[\\OpP]",
        "single_chars": r"\b[a-zA-Z]\b",
    }

    return patterns

# %%
class AdvancedIndonesianTextPreprocessor:
    """
    Advanced text preprocessor for Indonesian social media text using regex patterns
    """

    def __init__(self):
        self.stopwords, self.stopwords_categories = (
            get_comprehensive_indonesian_stopwords()
        )
        self.regex_patterns = create_regex_patterns()
        self.negation_words = {
            "tidak",
            "bukan",
            "belum",
            "jangan",
            "tanpa",
            "gak",
            "nggak",
        }

        self.compiled_patterns = {}
        for name, pattern in self.regex_patterns.items():
            self.compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

    def clean_social_media_artifacts(self, text):
        """
        Remove social media specific artifacts using regex
        """

        text = self.compiled_patterns["urls"].sub("", text)
        text = self.compiled_patterns["mentions"].sub("", text)
        text = self.compiled_patterns["hashtags"].sub("", text) 
        text = self.compiled_patterns["phone_numbers"].sub("", text)
        text = self.compiled_patterns["emails"].sub("", text)
        text = self.compiled_patterns["emoticons"].sub("", text)

        return text

    def normalize_repeated_patterns(self, text):
        """
        Normalize repeated characters and words using regex
        """

        text = self.compiled_patterns["repeated_chars"].sub(r"\1\1", text)
        text = self.compiled_patterns["repeated_words"].sub(r"\1", text)

        return text

    def remove_noise_patterns(self, text):
        """
        Remove various noise patterns using regex
        """

        text = self.compiled_patterns["numbers"].sub("", text)
        text = self.compiled_patterns["special_chars"].sub(" ", text)
        text = self.compiled_patterns["single_chars"].sub("", text)
        text = self.compiled_patterns["extra_whitespace"].sub(" ", text)

        return text.strip()

    def advanced_stopwords_removal(self, text, strategy="standard"):
        """
        Advanced stopwords removal with different strategies
        """
        words = text.lower().split()

        if strategy == "standard":
            filtered_words = [
                word
                for word in words
                if word not in self.stopwords or word in self.negation_words
            ]

        elif strategy == "aggressive":
            filtered_words = []
            for word in words:
                if (
                    word not in self.stopwords
                    and len(word) > 2
                    and word not in self.negation_words
                ):
                    filtered_words.append(word)
                elif word in self.negation_words:
                    filtered_words.append(word)

        elif strategy == "conservative":
            common_stopwords = (
                self.stopwords_categories["common_words"]
                + self.stopwords_categories["prepositions"]
            )
            filtered_words = [
                word
                for word in words
                if word not in common_stopwords or word in self.negation_words
            ]

        elif strategy == "selective":
            preserve_categories = ["auxiliaries"]
            words_to_preserve = set()
            for cat in preserve_categories:
                words_to_preserve.update(self.stopwords_categories[cat])

            filtered_words = []
            for word in words:
                if (
                    word not in self.stopwords
                    or word in words_to_preserve
                    or word in self.negation_words
                ):
                    filtered_words.append(word)

        return " ".join(filtered_words)

    def preprocess_text(
        self, text, stopwords_strategy="standard", remove_short_words=True
    ):
        """
        Complete preprocessing pipeline
        """

        text = text.lower()
        text = self.clean_social_media_artifacts(text)
        text = self.normalize_repeated_patterns(text)
        text = self.remove_noise_patterns(text)
        text = self.advanced_stopwords_removal(text, strategy=stopwords_strategy)

        if remove_short_words:
            text = self.compiled_patterns["short_words"].sub("", text)
            text = self.compiled_patterns["extra_whitespace"].sub(" ", text).strip()

        return text


preprocessor = AdvancedIndonesianTextPreprocessor()

strategies = ["standard", "aggressive", "conservative", "selective"]
results_by_strategy = {}

for strategy in strategies:
    print(f"\nProcessing with strategy = {strategy}")

    processed_texts = []
    processing_times = []

    for text in df["text"]:
        start_time = time.time()
        processed_text = preprocessor.preprocess_text(text, stopwords_strategy=strategy)
        processing_times.append(time.time() - start_time)
        processed_texts.append(processed_text)

    results_by_strategy[strategy] = {
        "processed_texts": processed_texts,
        "processing_times": processing_times,
    }
    
    # Display head of processed data for each strategy
    print(f"\n--- Preview of {strategy} preprocessing results ---")
    temp_df = df.copy()
    temp_df[f'processed_text_{strategy}'] = processed_texts
    print(temp_df[['text', f'processed_text_{strategy}']].head())
    print("-" * 80)


print("\nProcessing with gensim baseline...")
baseline_processed_texts = []
baseline_times = []

for text in df["text"]:
    start_time = time.time()

    processed = remove_stopwords(text.lower())

    processed = re.sub(r"[^\w\s]", " ", processed)
    processed = re.sub(r"\s+", " ", processed).strip()
    baseline_times.append(time.time() - start_time)
    baseline_processed_texts.append(processed)

results_by_strategy["gensim_baseline"] = {
    "processed_texts": baseline_processed_texts,
    "processing_times": baseline_times,
}

# Display head of baseline preprocessing results
print("\n--- Preview of gensim baseline preprocessing results ---")
temp_df = df.copy()
temp_df['processed_text_baseline'] = baseline_processed_texts
print(temp_df[['text', 'processed_text_baseline']].head())
print("-" * 80)


def extract_tfidf_features(texts, max_features=10000, min_df=2):
    """
    Extract TF-IDF features from texts
    """

    non_empty_texts = [text if text.strip() else "empty" for text in texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=True,
    )

    features = vectorizer.fit_transform(non_empty_texts)
    return features, vectorizer


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name="SVM"):
    """
    Train SVM model and evaluate performance with balanced metrics
    """
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate multiple metrics for comprehensive evaluation
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Macro average (unweighted mean of per-class metrics)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    
    # Weighted average (weighted by support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "support": support,
        "predictions": y_pred,
    }


def cross_validate_model(features, labels, cv_folds=5):
    """
    Perform cross-validation with multiple scoring metrics
    """
    model = SVC(kernel="linear", random_state=42)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy', 
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = {}
    for metric_name, metric in scoring_metrics.items():
        try:
            scores = cross_val_score(
                model, features, labels, cv=cv_folds, scoring=metric
            )
            cv_results[metric_name] = scores
        except Exception as e:
            print(f"Warning: Could not compute {metric_name}: {e}")
            cv_results[metric_name] = None
    
    return cv_results


def analyze_preprocessing_effectiveness(original_texts, processed_texts, strategy_name):
    """
    Analyze the effectiveness of preprocessing strategy
    """
    analysis = {
        "strategy": strategy_name,
        "avg_length_reduction": 0,
        "words_removed_count": 0,
        "common_removed_words": [],
    }

    original_lengths = []
    processed_lengths = []
    all_removed_words = []

    for orig, proc in zip(original_texts, processed_texts):
        orig_words = orig.lower().split()
        proc_words = proc.split()

        original_lengths.append(len(orig_words))
        processed_lengths.append(len(proc_words))

        removed = set(orig_words) - set(proc_words)
        all_removed_words.extend(removed)

    analysis["avg_length_reduction"] = np.mean(original_lengths) - np.mean(
        processed_lengths
    )
    analysis["words_removed_count"] = len(all_removed_words)
    analysis["common_removed_words"] = [
        word for word, count in Counter(all_removed_words).most_common(10)
    ]

    return analysis


print("\n" + "=" * 70)
print("EXPERIMENTAL EVALUATION")
print("=" * 70)


y = df["sentiment"]

X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    range(len(df)), y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train_idx)}")
print(f"Test set size: {len(X_test_idx)}")


evaluation_results = {}
preprocessing_analyses = {}

all_strategies = ["gensim_baseline"] + strategies

for strategy in all_strategies:
    print(f"\nEvaluating preprocessing strategy: {strategy}")

    processed_texts = results_by_strategy[strategy]["processed_texts"]

    preprocessing_analyses[strategy] = analyze_preprocessing_effectiveness(
        df["text"].tolist(), processed_texts, strategy
    )

    X_train_texts = [processed_texts[i] for i in X_train_idx]
    X_test_texts = [processed_texts[i] for i in X_test_idx]

    try:
        features_train, vectorizer = extract_tfidf_features(X_train_texts)
        features_test = vectorizer.transform(X_test_texts)

        results = train_and_evaluate_model(
            features_train, features_test, y_train, y_test, strategy
        )

        all_features = vectorizer.transform(processed_texts)
        cv_results = cross_validate_model(all_features, y, cv_folds=5)

        evaluation_results[strategy] = {
            "results": results,
            "cv_results": cv_results,
            "features_train": features_train,
            "features_test": features_test,
            "vectorizer": vectorizer,
            "processing_times": results_by_strategy[strategy]["processing_times"],
        }

        print(f"{strategy} Accuracy: {results['accuracy']:.4f}")
        print(f"{strategy} Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"{strategy} F1-Macro: {results['f1_macro']:.4f}")
        print(f"{strategy} MCC: {results['mcc']:.4f}")
        if cv_results['balanced_accuracy'] is not None:
            print(f"{strategy} CV Balanced Accuracy: {cv_results['balanced_accuracy'].mean():.4f} (±{cv_results['balanced_accuracy'].std():.4f})")

    except Exception as e:
        print(f"Error processing strategy {strategy}: {e}")
        continue


def perform_statistical_test(baseline_scores, comparison_scores):
    """
    Perform paired t-test to determine statistical significance
    """
    statistic, p_value = stats.ttest_rel(comparison_scores, baseline_scores)
    return statistic, p_value


print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTING")
print("=" * 70)

if "gensim_baseline" in evaluation_results:
    baseline_cv = evaluation_results["gensim_baseline"]["cv_scores"]

    for strategy in strategies:
        if strategy in evaluation_results:
            strategy_cv = evaluation_results[strategy]["cv_scores"]
            statistic, p_value = perform_statistical_test(baseline_cv, strategy_cv)

            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"{strategy} vs Baseline:")
            print(f"  t-statistic: {statistic:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Result: {significance}")

plt.figure(figsize=(16, 12))


plt.subplot(3, 3, 1)
valid_strategies = [s for s in all_strategies if s in evaluation_results]
accuracies = [evaluation_results[s]["results"]["accuracy"] for s in valid_strategies]

bars = plt.bar(
    valid_strategies,
    accuracies,
    color=["red", "blue", "green", "orange", "purple"][: len(valid_strategies)],
)
plt.title("Preprocessing Strategy Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)


plt.subplot(3, 3, 2)
# Plot multiple CV metrics for comparison
metrics_to_plot = ['balanced_accuracy', 'f1_macro', 'accuracy']
colors = ['blue', 'green', 'red']

x_pos = np.arange(len(valid_strategies))
width = 0.25

for i, metric in enumerate(metrics_to_plot):
    means = []
    stds = []
    for strategy in valid_strategies:
        cv_data = evaluation_results[strategy]["cv_results"][metric]
        if cv_data is not None:
            means.append(cv_data.mean())
            stds.append(cv_data.std())
        else:
            means.append(0)
            stds.append(0)
    
    plt.bar(x_pos + i*width, means, width, label=metric, 
           color=colors[i], alpha=0.7, yerr=stds, capsize=3)

plt.title("Cross-Validation Performance (Multiple Metrics)")
plt.ylabel("CV Score")
plt.xticks(x_pos + width, valid_strategies, rotation=45, ha="right")
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(3, 3, 3)
avg_processing_times = [
    np.mean(evaluation_results[s]["processing_times"]) for s in valid_strategies
]

plt.bar(valid_strategies, avg_processing_times, color="purple", alpha=0.7)
plt.title("Average Processing Time per Text")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45, ha="right")


plt.subplot(3, 3, 4)
if preprocessing_analyses:
    strategies_with_analysis = [
        s for s in valid_strategies if s in preprocessing_analyses
    ]
    length_reductions = [
        preprocessing_analyses[s]["avg_length_reduction"]
        for s in strategies_with_analysis
    ]

    plt.bar(strategies_with_analysis, length_reductions, color="lightgreen", alpha=0.7)
    plt.title("Average Word Count Reduction")
    plt.ylabel("Words Removed")
    plt.xticks(rotation=45, ha="right")


plt.subplot(3, 3, 5)
if len(valid_strategies) >= 2:
    baseline_strategy = valid_strategies[0]
    # Use balanced accuracy as primary metric for finding best strategy
    best_strategy = max(
        valid_strategies, key=lambda s: evaluation_results[s]["results"]["balanced_accuracy"]
    )

    # Updated metrics to include balanced metrics
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"]
    baseline_metrics = []
    best_metrics = []
    
    for metric in metrics:
        if metric in evaluation_results[baseline_strategy]["results"]:
            baseline_metrics.append(evaluation_results[baseline_strategy]["results"][metric])
            best_metrics.append(evaluation_results[best_strategy]["results"][metric])

    x = np.arange(len(baseline_metrics))
    width = 0.35

    plt.bar(x - width / 2, baseline_metrics, width, label=baseline_strategy, alpha=0.7)
    plt.bar(x + width / 2, best_metrics, width, label=best_strategy, alpha=0.7)

    plt.title("Performance Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(x, metrics[:len(baseline_metrics)], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)


plt.subplot(3, 3, 6)
if valid_strategies:
    # Use balanced accuracy for finding best strategy
    best_strategy = max(
        valid_strategies, key=lambda s: evaluation_results[s]["results"]["balanced_accuracy"]
    )
    best_results = evaluation_results[best_strategy]["results"]

    cm = confusion_matrix(y_test, best_results["predictions"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"Confusion Matrix\n(Best: {best_strategy})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")


plt.subplot(3, 3, 7)
if "gensim_baseline" in evaluation_results:
    baseline_acc = evaluation_results["gensim_baseline"]["results"]["accuracy"]
    improvements = []
    strategy_names = []

    for strategy in valid_strategies:
        if strategy != "gensim_baseline":
            acc = evaluation_results[strategy]["results"]["accuracy"]
            improvement = ((acc - baseline_acc) / baseline_acc) * 100
            improvements.append(improvement)
            strategy_names.append(strategy)

    colors = ["green" if imp > 0 else "red" for imp in improvements]
    bars = plt.bar(strategy_names, improvements, color=colors, alpha=0.7)
    plt.title("Accuracy Improvement over Baseline")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    for bar, imp in zip(bars, improvements):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{imp:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


plt.subplot(3, 3, 8)
if preprocessing_analyses and valid_strategies:
    # Use balanced accuracy for finding best strategy
    best_strategy = max(
        valid_strategies, key=lambda s: evaluation_results[s]["results"]["balanced_accuracy"]
    )
    if best_strategy in preprocessing_analyses:
        common_words = preprocessing_analyses[best_strategy]["common_removed_words"][
            :10
        ]
        if common_words:
            word_counts = [1] * len(common_words)
            plt.barh(range(len(common_words)), word_counts, color="orange", alpha=0.7)
            plt.yticks(range(len(common_words)), common_words)
            plt.title(f"Common Removed Words\n({best_strategy})")
            plt.xlabel("Frequency")


plt.subplot(3, 3, 9)
if valid_strategies:
    effectiveness_scores = []
    for strategy in valid_strategies:
        # Use a composite score: balanced accuracy + f1_macro - time penalty
        balanced_acc = evaluation_results[strategy]["results"]["balanced_accuracy"]
        f1_macro = evaluation_results[strategy]["results"]["f1_macro"]
        time_penalty = np.mean(evaluation_results[strategy]["processing_times"]) * 0.1
        effectiveness = (balanced_acc + f1_macro) / 2 - time_penalty
        effectiveness_scores.append(effectiveness)

    plt.bar(valid_strategies, effectiveness_scores, color="skyblue", alpha=0.7)
    plt.title("Overall Strategy Effectiveness\n(Balanced Score - Time Penalty)")
    plt.ylabel("Effectiveness Score")
    plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

print(f"Dataset Size: {len(df)} tweets")
print(f"Training Size: {len(X_train_idx)} tweets")
print(f"Test Size: {len(X_test_idx)} tweets")
print()

if evaluation_results:
    print("Performance Results:")
    print("-" * 50)

    best_strategy = None
    best_accuracy = 0

    for strategy in valid_strategies:
        if strategy in evaluation_results:
            acc = evaluation_results[strategy]["results"]["accuracy"]
            if strategy == "gensim_baseline":
                print(f"Baseline (Gensim):           {acc:.4f}")
                baseline_acc = acc
            else:
                improvement = (
                    ((acc - baseline_acc) / baseline_acc) * 100
                    if "baseline_acc" in locals()
                    else 0
                )
                print(f"{strategy:25}: {acc:.4f} ({improvement:+.2f}%)")

            if acc > best_accuracy:
                best_accuracy = acc
                best_strategy = strategy

    print()
    print("Preprocessing Analysis:")
    print("-" * 50)
    for strategy in valid_strategies:
        if strategy in preprocessing_analyses:
            analysis = preprocessing_analyses[strategy]
            print(f"{strategy}:")
            print(f"  Average words removed: {analysis['avg_length_reduction']:.1f}")
            print(f"  Total unique words removed: {analysis['words_removed_count']}")

    if best_strategy:
        print()
        print("Best Configuration:")
        print("-" * 50)
        print(f"Best Strategy: {best_strategy}")

        best_results = evaluation_results[best_strategy]["results"]
        print(f"Accuracy: {best_results['accuracy']:.4f}")
        print(f"Precision (Macro): {best_results['precision_macro']:.4f}")
        print(f"Recall (Macro): {best_results['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {best_results['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {best_results['f1_weighted']:.4f}")
        print(f"Matthews Correlation Coefficient: {best_results['mcc']:.4f}")
        if best_results['auc_roc'] is not None:
            print(f"AUC-ROC: {best_results['auc_roc']:.4f}")
        
        print("\nPer-Class Performance:")
        for i, (prec, rec, f1, sup) in enumerate(zip(
            best_results['precision_per_class'],
            best_results['recall_per_class'], 
            best_results['f1_per_class'],
            best_results['support']
        )):
            class_name = "Negative" if i == 0 else "Positive"
            print(f"  {class_name}:")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
            print(f"    Support:   {sup}")

        print()
        print("Computational Efficiency:")
        print("-" * 50)
        for strategy in valid_strategies:
            if strategy in evaluation_results:
                avg_time = np.mean(evaluation_results[strategy]["processing_times"])
                total_time = np.sum(evaluation_results[strategy]["processing_times"])
                print(f"{strategy}: {avg_time:.6f}s/tweet, Total: {total_time:.4f}s")

        print()
        print("Cross-Validation Results (Best Strategy):")
        print("-" * 50)
        cv_results = evaluation_results[best_strategy]["cv_results"]
        for metric, scores in cv_results.items():
            if scores is not None:
                print(f"{metric}: {scores.mean():.4f} (±{scores.std():.4f})")

        print()
        print("Detailed Classification Report (Best Strategy):")
        print("-" * 50)
        print(
            classification_report(
                y_test,
                best_results["predictions"],
                target_names=["Negative", "Positive"],
            )
        )
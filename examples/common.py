import torch
from torch.nn.functional import cosine_similarity
import pandas as pd
from functools import cache
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from rouge_score import rouge_scorer
import statistics
import sacrebleu
import torch.nn.functional as F
from datasets import Dataset
import os
from sentence_transformers import SentenceTransformer

pio.renderers.default = "firefox"
FIGURE_LAYOUT = {
    "title_text": "",
    "xaxis_title": None,
    "yaxis_title": None,
    "template": "plotly_white"
}
FIGURE_COLORS = px.colors.qualitative.Light24

# Compare two pytorch matrices
@cache
def compare_matrices(matrix1, matrix2):
    # Calculate Pearson's correlation
    corr_matrix = torch.corrcoef(torch.stack([
        torch.flatten(matrix1), 
        torch.flatten(matrix2)
    ], dim=0))
    corr = corr_matrix[0][1].item()
    # Calculate cosine similarity
    sim_matrix = cosine_similarity(matrix1, matrix2, dim=0)
    sim = sim_matrix.mean().item()
    return corr, sim

# Convert PyTorch similarity/correlation matrix to dataframe
def pt_to_df(matrix: torch.Tensor, labels: list[str] = []) -> pd.DataFrame:
    df = pd.DataFrame(matrix)
    if labels:
        df.index = labels
        df.columns = labels
    return df

# Plot PCA using plotly
def pca_plot(points, color_ids=None, hide_labels=False, size=30):
    # Define labels and colors
    labels = ["<b>{}</b>".format(x) for x in points.index.tolist()]
    if color_ids:
        colors = [FIGURE_COLORS[i] for i in color_ids]
    else:
        colors = FIGURE_COLORS
    # Define plot
    plot = go.Scatter(
            x=points["x"],
            y=points["y"],
            text=labels,
            mode='markers',
            marker=dict(color=colors, size=size),
            showlegend=False
        )
    return plot, labels, colors

# Visualize PCA
def visualize_pca(points, title="", color_ids=None, hide_labels=False, textangle=0, plot_size=(800,1200), legend=[], legend_color_ids=[]):
    fig = go.Figure()
    plot, _, _ = pca_plot(points, color_ids, hide_labels)
    fig.add_trace(plot)
    if not hide_labels:
        fig.update_layout(annotations=[
            go.layout.Annotation(x=row["x"], y=row["y"],
            xref="x",
            yref="y",
            text=f"<b>{label}</b>",
            align='center',
            showarrow=False,
            yanchor='middle',
            textangle=textangle) for label, row in points.iterrows()])
    if legend and legend_color_ids:
        for index, label in enumerate(legend):
            fig.add_trace(go.Scatter(x=[None], y=[None], 
                showlegend=True,
                mode="markers",
                marker=dict(size=10, color=FIGURE_COLORS[legend_color_ids[index]]),
                name=label
            ))
    fig.update_layout(FIGURE_LAYOUT)
    fig.update_layout(title_text=title, height=plot_size[0], width=plot_size[1])
    fig.show()

# Visualize multiple PCAs
def visualize_pca_subplots(points_dfs, title="", titles=[], color_ids=None, hide_labels=True, size=20, rows=2, cols=3, plot_size=(800,1200)):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    # Add pca plots
    label_color_pairs = []
    for index, points in enumerate(points_dfs):
        plot, labels, colors = pca_plot(points, color_ids, hide_labels, size)
        label_color_pairs += [(labels[i], colors[i]) for i in range(len(labels))]
        row = math.floor(index/cols) + 1
        col = index - (row-1) * cols + 1
        fig.append_trace(plot, row=row, col=col)
    # Add legend
    label_color_pairs = list(set(label_color_pairs))
    for label, color in label_color_pairs:
        fig.append_trace(go.Scatter(x=[None], y=[None], 
            showlegend=True,
            mode="markers",
            marker=dict(size=10, color=color),
            name=label
        ), row=rows-1, col=cols-1)
    # Show figure
    fig.update_layout(FIGURE_LAYOUT)
    fig.update_layout(title_text=title, height=plot_size[0], width=plot_size[1])
    fig.show()

# Calculate Rouge-L score for specific model
def generate_rouge_scores(dataset: Dataset, ignored_features: list[str] = ["instruction", "index"], ref_feature: str = "golden_answer") -> list:
    features = [x for x in dataset.column_names if not x in ignored_features + [ref_feature]]
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    avg_scores = []
    for feature in features:
        feature_scores = []
        for item in dataset:
            ref = item[ref_feature] #type: ignore
            for hyp in item[feature]: #type: ignore
                score = rouge.score(ref, hyp)["rougeL"].fmeasure
                feature_scores.append(score)
        avg_scores.append(statistics.mean(feature_scores))
    return avg_scores

# Calculate similarity matrix between inverted Rouge-L scores
def generate_rouge_diff_matrix(rouge_scores: list) -> torch.Tensor:
    # Calculate similarity matrix based on inverted ROUGE-L scores
    rouge_diff = []
    for i in range(len(rouge_scores)):
        row = []
        for j in range(len(rouge_scores)):
            diff = abs(rouge_scores[i] - rouge_scores[j])
            inv_diff = 1.0 - diff
            row.append(inv_diff)
        rouge_diff.append(row)
    return torch.tensor(rouge_diff)

# Calculate BLEU scores
def generate_bleu_scores(dataset: Dataset, ignored_features: list[str] = ["instruction", "index"], ref_feature: str = "golden_answer") -> list:
    features = [x for x in dataset.column_names if not x in ignored_features + [ref_feature]]
    avg_scores = []
    for feature in features:
        feature_scores = []
        for item in dataset:
            for hyp in item[feature]: #type: ignore
                score = sacrebleu.sentence_bleu(hyp, [item[ref_feature]]).score / 100.0
                feature_scores.append(score)
        avg_scores.append(statistics.mean(feature_scores))
    return avg_scores

# Calculate BERT score
# WARNING: This implementation of BERTScore might be a bit clunky for production.
# ... It was originally designed for the experiment mentioned in paper.
# ... Please, do not mindlessly copy-paste it anywhere.
def generate_bert_scores(dataset: Dataset, ref_feature: str = "golden_answer", encoder_name: str = "sentence-transformers/all-MiniLM-L12-v2", batch_size=32, embeddings_path: str = "", ref_embeddings_path: str = "") -> list:
    assert os.path.exists(embeddings_path), "Sorry, only cached feature embeddings are suported. Specify embeddings_path"
    embeddings = torch.load(embeddings_path, weights_only=True)
    if not os.path.exists(ref_embeddings_path):
        encoder = SentenceTransformer(encoder_name)
        ref_texts = [x[ref_feature] for x in dataset] #type: ignore
        ref_embeddings = encoder.encode(ref_texts, batch_size=batch_size, convert_to_tensor=True)
        if ref_embeddings_path:
            torch.save(ref_embeddings, ref_embeddings_path)
    else:
        ref_embeddings = torch.load(ref_embeddings_path, weights_only=True)
    scores = []
    for i in range(embeddings.shape[0]):
        # Flatten
        hyps = embeddings[i].flatten(0, 1)
        refs = ref_embeddings.repeat_interleave(embeddings.shape[2], dim=0)
        # Normalize
        hyps = F.normalize(hyps, p=2, dim=1)
        refs = F.normalize(refs, p=2, dim=1)
        # Compute similarity
        sim = torch.matmul(hyps, refs.T)
        # Compute BERTScore
        precision = sim.max(dim=1).values.mean()
        recall = sim.max(dim=0).values.mean()
        f1_score = 2 * precision * recall / (precision + recall)
        scores.append(f1_score.item())
    return scores

# Benchmark models using BLEU, ROUGE and BERTScore
def benchmark(dataset: Dataset, ref_feature: str = "golden_answer", ignored_features: list[str] = ["index", "instruction"], encoder_name: str = "sentence-transformers/all-MiniLM-L12-v2", batch_size: int = 32, embeddings_path: str = "", ref_embeddings_path: str = ""):
    features = [x for x in dataset.column_names if not x in ignored_features + [ref_feature]]
    stats = {
        "BERTScore": generate_bert_scores(dataset=dataset, ref_feature=ref_feature, encoder_name=encoder_name, batch_size=batch_size, ref_embeddings_path=ref_embeddings_path, embeddings_path=embeddings_path),
        "BLEU": generate_bleu_scores(dataset=dataset, ignored_features=ignored_features, ref_feature=ref_feature),
        "ROUGE-L": generate_rouge_scores(dataset=dataset, ignored_features=ignored_features, ref_feature=ref_feature)
    }
    stats = pd.DataFrame(stats)
    stats.index = features
    return stats

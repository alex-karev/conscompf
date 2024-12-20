from datasets.utils.logging import set_verbosity_error
from datasets import  load_dataset, Dataset
from conscompf import ConSCompF
from common import benchmark, compare_matrices, generate_rouge_diff_matrix, pt_to_df, generate_rouge_scores, visualize_pca, visualize_pca_subplots
import pandas as pd
import json
import torch
set_verbosity_error()

# Load config
config = json.load(open("examples/common.json"))

# Initialize conscompt
conscompf = ConSCompF(quiet=False)
titles = ["Full"] 
titles += ["Few-shot ({})".format(key) for key in config["curated_instructions"].keys()]

# Full dataset
embeddings = [torch.load("__embeddings__/quantization.pt", weights_only=True)]

# Curated instructions
for index, instructions in enumerate(config["curated_instructions"].values()):
    title = titles[index+1]
    embeddings.append(torch.load(f"__embeddings__/quantization {title}.pt", weights_only=True))

# Compare wighted and unweighted similarity
columns = ["original", "q8", "q4", "q2", "pirate"]
stats = []
for i in range(len(embeddings)):
    data1=embeddings[i][0]
    row1 = {"Dataset": titles[i], "Method": "Simple"}
    row2 = {"Dataset": "", "Method": "Weighted"}
    for j in range(1, len(embeddings[i])):
        data2=embeddings[i][j]
        sim = conscompf.similarity(data1, data2)
        sim_w = conscompf.weighted_similarity(data1, data2)
        cons = conscompf.consistency(data2)
        sim_cons_corr = torch.corrcoef(torch.stack([sim, cons], dim=0))[0][1].item()
        sim_w_cons_corr = torch.corrcoef(torch.stack([sim_w, cons], dim=0))[0][1].item()
        row1[columns[j-1]] = sim_cons_corr
        row2[columns[j-1]] = sim_w_cons_corr
    stats.append(row1)
    stats.append(row2)
stats_df = pd.DataFrame(stats)
print(stats_df.to_latex(float_format="%.4f"))


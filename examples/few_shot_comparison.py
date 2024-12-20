from datasets.utils.logging import set_verbosity_error
from datasets import  load_dataset, Dataset
from conscompf import ConSCompF
from common import benchmark, compare_matrices, generate_rouge_diff_matrix, pt_to_df, generate_rouge_scores, visualize_pca, visualize_pca_subplots
import pandas as pd
import json
set_verbosity_error()

# Load config
config = json.load(open("examples/common.json"))

# Load dataset
dataset: Dataset = load_dataset(config["dataset_name"], config["subsets"]["comparison"], split="train") #type: ignore

# Initialize conscompt
conscompf = ConSCompF(quiet=True)

# Define titles
titles = ["Full"] 
titles += ["Few-shot ({})".format(key) for key in config["curated_instructions"].keys()]
titles += ["Few-shot ({})".format(n) for n in config["downsample"]]

# Generate rouge matrix
if config["run_benchmark"]:
    stats = benchmark(dataset, embeddings_path="__embeddings__/comparison.pt", ref_embeddings_path="__embeddings__/comparison_ref.pt")
    print("\n### Performance:\n\n{}".format(stats.to_latex(float_format="%.4f")))
    rouge_scores = stats["ROUGE-L"].tolist()
else:
    rouge_scores = generate_rouge_scores(dataset)
rouge_matrix = generate_rouge_diff_matrix(rouge_scores) 
if config["run_benchmark"]:
    rouge_matrix_df = pt_to_df(rouge_matrix, dataset.column_names[3:])
    print("\n### ROUGE-L similarity matrix:\n\n{}".format(rouge_matrix_df.to_latex(float_format="%.4f")))

# Pretty-print conscompf output with some statistics
def print_sim_matrix(sim_matrix, title):
    df_sim_matrix = pt_to_df(sim_matrix, labels=dataset.column_names[3:])
    print("\n### {}:\n\n{}".format(title, df_sim_matrix.to_latex(float_format="%.4f")))

# Full dataset
outs = [conscompf(dataset, ignored_features=["index", "instruction", "golden_answer"], embeddings_path="__embeddings__/comparison.pt", return_type="pt")]
print_sim_matrix(outs[0]["sim_matrix"], titles[0])

# Curated instructions
for index, instructions in enumerate(config["curated_instructions"].values()):
    samples = dataset.filter(lambda item: item["index"] in instructions)
    title = titles[index+1]
    out = conscompf(samples, ignored_features=["index", "instruction", "golden_answer"], embeddings_path=f"__embeddings__/comparison {title}.pt", return_type="pt")
    print_sim_matrix(out["sim_matrix"], title)
    outs.append(out)

# Downsample
for index, n_samples in enumerate(config["downsample"]):
    samples = dataset.take(n_samples)
    title = titles[1+len(config["curated_instructions"])+index]
    out = conscompf(samples, ignored_features=["index", "instruction", "golden_answer"], embeddings_path=f"__embeddings__/comparison {title}.pt", return_type="pt")
    print_sim_matrix(out["sim_matrix"], title)
    outs.append(out)

# Calculate overall statistics
stats = []
pca_dfs = []
for index, out in enumerate(outs):
    original_corr, original_sim = compare_matrices(out["sim_matrix"], outs[0]["sim_matrix"])
    rouge_corr, rouge_sim = compare_matrices(out["sim_matrix"], rouge_matrix)
    consistency = out["consistency"].mean(dim=-1).mean().item()
    stats.append({
        "Subset": titles[index], 
        "Consistency": consistency, 
        "ROUGE-L corr.": rouge_corr, 
        "ROUGE-L sim.": rouge_sim,
        "Original corr.": original_corr,
        "Original sim.": original_sim 
    })
    pca_df = pd.DataFrame(out["pca"]).set_axis(dataset.column_names[3:], axis=0).set_axis(["x", "y"], axis=1)
    pca_dfs.append(pca_df)
stats_df = pd.DataFrame(stats)
print(stats_df.to_latex(float_format="%.4f"))

# Visualize
visualize_pca(pca_dfs[0], titles[0])
visualize_pca_subplots(pca_dfs, title="Few-shot comparison", titles=titles)

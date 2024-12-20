from conscompf import ConSCompF
from common import visualize_pca
from datasets import  load_dataset, Dataset
import json

# Load config
config = json.load(open("examples/common.json"))

# Load dataset
dataset: Dataset = load_dataset(config["dataset_name"], config["subsets"]["prompts"], split="train") #type: ignore

# Initialize conscompt
conscompf = ConSCompF(quiet=True)

# Get output
out = conscompf(dataset, ignored_features=["index", "instruction", "golden_answer"], embeddings_path="__embeddings__/prompts.pt", return_type="df")

print("Similarity matrix:\n\n{}".format(out["sim_matrix"].round(4).to_latex()))
pca = out["pca"]
pca = pca.set_axis([x.split("-")[-1] for x in pca.index], axis=0) 
n_prompts = (len(dataset.features)-3) // 3
color_ids=[0] * n_prompts + [1] * n_prompts + [2] * n_prompts
visualize_pca(pca,  
    title="Prompt comparison", 
    color_ids=color_ids, 
    plot_size=(800,1000), 
    legend=["qwen2.5-3b", "phi3.5-mini", "gemma2-2b"],
    legend_color_ids=[0,1,2])

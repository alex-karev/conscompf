from conscompf import ConSCompF
from datasets import  load_dataset, Dataset
import pandas as pd
import json

# Load config, dataset and conscompd
config = json.load(open("examples/common.json"))
dataset: Dataset = load_dataset(config["dataset_name"], config["subsets"]["comparison"], split="train") #type: ignore
conscompf = ConSCompF(quiet=True)

# Calculate consistency scores
consistency = conscompf(dataset, ignored_features=["index", "instruction", "golden_answer"], embeddings_path="__embeddings__/comparison.pt", return_type="pt")["consistency"].mean(dim=0)

# Save as excel
indexes = [x["index"] for x in dataset] #type: ignore
instructions = [x["instruction"] for x in dataset] #type: ignore
consistency_df = pd.DataFrame({"instruction": instructions, "cons": consistency}).set_axis(indexes, axis=0)
consistency_df.to_excel("examples/instruction_consistency.xlsx")

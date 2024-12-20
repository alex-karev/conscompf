from collections.abc import Iterable
from typing import Literal, overload, TypedDict
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity
from functools import cache
from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
import math
import os
from tqdm import tqdm

ConSCompFInput = Dataset | DatasetDict | IterableDatasetDict | IterableDataset | pd.DataFrame | list[dict[str, list[str]]]

class ConSCompFOutputPt(TypedDict): 
    """ComSCompF output in PyTorch Tensor format."""
    sim_matrix: torch.Tensor 
    consistency: torch.Tensor
    pca: torch.Tensor

class ConSCompFOutputDf(TypedDict): 
    """ComSCompF output in pandas DataFrame format."""
    sim_matrix:  pd.DataFrame 
    consistency: pd.Series
    pca: pd.DataFrame

class ConSCompFOutputList(TypedDict): 
    """ComSCompF output in list format."""
    sim_matrix: list[list]
    consistency:  list 
    pca: list 

class ConSCompF:
    """
    ConSCompF: Consistency-focused Similarity Comparison Framework for generative large language models.

    Attributes
    ----------
    encoder : SentenceTransformer
        SentenceTranformer encoder object
    weighted : bool
        enables/disables similarity score weighting based on instruction consistency (default True)

    Methods
    -------
    __call__(*args, **kwargs)
        generates similarity matrix, pca components and instruction consistency scores
    consistency(data : torch.Tensor)
        generates instruction consistency scores for one model
    similarity(data1 : torch.Tensor, data2 : torch.Tensor)
        calculates unweighted similarity score between two models
    weighted_similarity(data1 : torch.Tensor, data2 : torch.Tensor)
        calculates weighted similarity score between two models

    """

    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L12-v2", max_batch_size=32, weighted: bool = True, quiet: bool = True) -> None:
        """
        Parameters
        ----------
        encoder_name : str
            SentenceTransformer encoder name/path (default "sentence-transformers/all-MiniLM-L12-v2")
        max_batch_size : int
            maximum number of strings in one batch when generating embeddings (default 32)
        weighted : bool
            enables/disables similarity score weighting based on instruction consistency (default True)
        quiet : bool
            enables/disables verbose output (default True)

        """
        self.weighted = weighted
        self.encoder = SentenceTransformer(encoder_name)
        self.max_batch_size = max_batch_size
        self.quiet = quiet 

    # Check input tensors
    @cache
    def check_data(self, *args) -> bool:
        for data in args:
            assert data.dim() == 3, "3D tensor of shape (n_samples, n_repeats, n_dim) is required!"
        return True
    
    # Calculate similarity score for each instruction
    @cache
    def similarity(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        Calculates unweighted similarity score between two models

        Parameters
        ----------
        data1 : torch.Tensor
            3d tensor of embeddings for the first model
        data2 : torch.Tensor
            3d tensor of embeddings for the second model
        """
        self.check_data(data1, data2)
        av1 = data1.mean(dim=1)
        av2 = data2.mean(dim=1)
        sim = cosine_similarity(av1, av2, dim=1)
        return sim

    # Calculate instruction consistency 
    @cache
    def consistency(self, data: torch.Tensor) -> torch.Tensor:
        """
        Calculates instruction consistency scores for one model

        Parameters
        ----------
        data : torch.Tensor
            3d tensor of embeddings for the model
        """

        self.check_data(data)
        comb = torch.combinations(torch.arange(data.shape[1]), 2)
        v = data[:,comb].permute(2,0,1,3)
        cons = cosine_similarity(v[0], v[1], dim=2)
        avg_cons = torch.mean(cons, dim=1)
        return avg_cons

   # Calculate weighted similarity scores between data1 and data2
    @cache
    def weighted_similarity(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        Calculates weighted similarity score between two models

        Parameters
        ----------
        data1 : torch.Tensor
            3d tensor of embeddings for the first model
        data2 : torch.Tensor
            3d tensor of embeddings for the second model
        """

        self.check_data(data1, data2)
        sim = self.similarity(data1, data2)
        cons1 = self.consistency(data1)
        cons2 = self.consistency(data2)    
        cons = (cons1 + cons2) / 2.0
        cons_n = (cons - cons.min()) / (cons.max() - cons.min())
        sim_w = sim + (1-sim) * (1-cons_n)
        return sim_w
   
    @overload
    def __call__(self, dataset: ConSCompFInput, return_type: Literal["pt"] = "pt", features: list[str] = [], ignored_features: list[str] = [], embeddings_path: str | None = None, weighted: bool | None = None, n_pca_dim: int = 2) -> ConSCompFOutputPt: ...
    
    @overload
    def __call__(self, dataset: ConSCompFInput, return_type: Literal["df"], features: list[str] = [], ignored_features: list[str] = [], embeddings_path: str | None = None, weighted: bool | None = None, n_pca_dim: int = 2) -> ConSCompFOutputDf: ...
    
    @overload
    def __call__(self, dataset: ConSCompFInput, return_type: Literal["list"], features: list[str] = [], ignored_features: list[str] = [], embeddings_path: str | None = None, weighted: bool | None = None, n_pca_dim: int = 2) -> ConSCompFOutputList: ...

    # Generate similarity matrix for multiple data points
    def __call__(self, dataset: ConSCompFInput, return_type: Literal["list", "pt", "df"] = "pt", features: list[str] = [], ignored_features: list[str] = [], embeddings_path: str | None = None, weighted: bool | None = None, n_pca_dim: int = 2) -> ConSCompFOutputPt | ConSCompFOutputList | ConSCompFOutputDf:
        """
        Generates similarity matrix, pca components and instruction consistency scores for a set of texts generated by multiple LLMs

        Parameters
        ----------
        dataset : Dataset | pd.DataFrame | list[dict[str, list[str]]]
            textual data to use for comparison
        return_type : Literal["list", "pt", "df"]
            type of the return output (default "pt")
        features : list[str]
            overrides feature names in dataset (default [])
        ignored_features : list[str]
            dataset features to be ignored during computation process (default [])
        embeddings_path : str
            path for caching embeddings. If specified, ConSCompF will try to load embeddings from path instead of generating them. If path does not exists, ConSCompF will generate embeddings and save them at this path.
        weighted : bool
            if specified, temporarilt overrides "weighted" attribute
        n_pca_dim : int
            number of dimensions for PCA (default 2)

        Returns
        -------
        sim_matrix : torch.Tensor | pandas.DataFrame | list
            LLM similarity matrix
        pca: torch.Tensor | pandas.DataFrame | list
            PCA representation of LLM similarity matrix
        consistency: torch.Tensor | pandas.Series | list
            instruction consistency scores for each instruction in a dataset according to each model
        """
        # Define parameters
        weighted = weighted if not weighted is None else self.weighted
        features = list(dataset[0].keys()) if isinstance(dataset, list) \
            else dataset.column_names if isinstance(dataset, Dataset) \
            else dataset.columns.tolist() if isinstance(dataset, pd.DataFrame) \
            else features
        if ignored_features:
            features = [x for x in features if not x in ignored_features]
        if isinstance(dataset, pd.DataFrame): dataset = Dataset.from_pandas(dataset)
        assert len(features) > 0, "Features are not defined"

        # Generate embeddings
        if not embeddings_path or not os.path.exists(embeddings_path):
            # Define parameters
            first_feature = dataset[0][features[0]]
            is_list = isinstance(first_feature, Iterable) and not isinstance(first_feature, str)
            n_answers = len(first_feature) if is_list else 1 #type: ignore
            n_instructions_per_batch = self.max_batch_size // n_answers
            n_batches = math.ceil(len(dataset)/n_instructions_per_batch)
       
            # Start generation
            embeddings = []
            if not self.quiet: print("Generating embeddings...")
            for feature in features:
                feature_embeddings = []
                for i in tqdm(range(n_batches), desc=f"\tProcessing feature '{feature}'", disable=self.quiet):
                    # Difine ids
                    start_index = i*n_instructions_per_batch
                    end_index = min((i+1)*n_instructions_per_batch, len(dataset))
                    i_samples = range(start_index, end_index)
                    # Extract texts
                    if is_list: 
                        texts = []
                        for i_sample in i_samples:
                            texts += [dataset[i_sample][feature][x] for x in range(n_answers)]
                    else:
                        texts = [dataset[x][feature] for x in i_samples]
                    # Generate embeddings
                    batch_embeddings = self.encoder.encode(texts, convert_to_tensor=True) #type: ignore
                    batch_embeddings = torch.tensor_split(batch_embeddings, batch_embeddings.shape[0] // n_answers)
                    feature_embeddings += batch_embeddings
                feature_embeddings = torch.stack(feature_embeddings, dim=0)
                embeddings.append(feature_embeddings)
            embeddings = torch.stack(embeddings, dim=0)

            # Save output
            if embeddings_path:
                torch.save(embeddings, embeddings_path)
                if not self.quiet: print(f"Embeddings were saved to '{embeddings_path}'.")
        # Load embeddings
        else:
            if not self.quiet: print(f"Loading embeddings from '{embeddings_path}'...")
            embeddings = torch.load(embeddings_path, weights_only=True)

        assert embeddings.dim() == 4, "4D tensor of shape (n_features, n_samples, n_repeats, n_dim) is required!"

        # Generate matrix
        sim_matrix = []
        for i in range(embeddings.shape[0]):
            row = []
            for j in range(embeddings.shape[0]):
                data1 = embeddings[i]
                data2 = embeddings[j]
                sim = self.weighted_similarity(data1, data2) if weighted else self.similarity(data1, data2)
                avg_sim = sim.mean(dim=0)
                row.append(avg_sim)
            row = torch.stack(row, dim=0)
            sim_matrix.append(row)
        sim_matrix = torch.stack(sim_matrix, dim=0)

        # Compute PCA
        _, _, v = torch.pca_lowrank(sim_matrix, q=n_pca_dim)
        pca_points = torch.matmul(sim_matrix, v)

        # Compute consistency scores
        consistency = []
        for i in range(embeddings.shape[0]):
            consistency.append(self.consistency(embeddings[i]))
        consistency = torch.stack(consistency, dim=0)
   
        # Return data
        if return_type == "pt": 
            return ConSCompFOutputPt(sim_matrix=sim_matrix, pca=pca_points, consistency=consistency)
        elif return_type == "list": 
            return ConSCompFOutputList(sim_matrix=sim_matrix.tolist(), pca=pca_points.tolist(), consistency=consistency.tolist())
        elif return_type == "df":
            return ConSCompFOutputDf(
                sim_matrix=pd.DataFrame(sim_matrix.numpy()).set_axis(features, axis=0).set_axis(features, axis=1), 
                pca=pd.DataFrame(pca_points.numpy()).set_axis(features, axis=0).set_axis(["x", "y", "z"][:n_pca_dim], axis=1), 
                consistency=pd.Series(consistency.tolist(), index=features)
            )

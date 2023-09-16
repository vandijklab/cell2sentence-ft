import os
import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import plotnine as pn
import scanpy as sc
import sklearn.linear_model as lm
from datasets import Dataset, load_dataset, concatenate_datasets
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from tqdm import tqdm

from src import utils

ROW_SUM = 10000


def normalize_and_rank_transform(data_matrix_X, normalize=True):
    """
    Helper function which accepts a data matrix, optionally row-normalizes it,
    and calculated a rank transformation of the data.

    Args:
        data_matrix_X:  numpy matrix of shape [num_cells, num_genes]
        normalize:      boolean flag for whether to normalize data

    Returns:
        data_matrix_X:  normalized data matrix
        rank_matrix_X:  matrix of rank values for each cell, shame shape as data_matrix_X
    """
    if normalize:
        normalized_data_matrix_X = (
            np.diag(ROW_SUM / np.ravel(np.sum(data_matrix_X, axis=1))) @ data_matrix_X
        )
        data_matrix_X = np.asarray(normalized_data_matrix_X)

    rank_matrix_X = np.zeros(shape=data_matrix_X.shape)
    for i in tqdm(range(data_matrix_X.shape[0])):
        cols = np.ravel(range(data_matrix_X.shape[1]))
        vals = np.ravel(data_matrix_X[i, :])
        cols, vals = shuffle(cols, vals)
        ranks = cols[np.argsort(-vals, kind="stable")]
        for j in range(len(ranks)):
            rank_matrix_X[i, ranks[j]] = j

    return data_matrix_X, rank_matrix_X


def evaluate_transformation(df, plotting_sample_size=10000):
    """
    Helper function which takes as input a pandas DataFrame of expression values and
    ranks, and fits a linear regression model to predict back expression value from
    log rank.

    Plots are created to show the relationship between log rank and log expression,
    as well as the performance of expression reconstruction by the linear model.
    Metrics for expression reconstruction, as well as the parameters of the linear
    model are saved in a CSV file.

    Args:
        df:                     pandas DataFrame with keys: 'preprocessed_transcript_count,
                                    'preprocessed_rank', 'log_preprocessed_transcript_count',
                                    and 'log_preprocessed_rank'
        plotting_sample_size:   how many values to sample for plotting
    """
    eval_output_dir = utils.DATA_DIR / "eval"
    eval_output_dir.mkdir(exist_ok=True, parents=True)

    # (1) Fit linear regression between log rank (x-axis) and log expression (y-axis)
    x_axis_name = "log_preprocessed_rank"
    y_axis_name = "log_preprocessed_transcript_count"
    x = np.array(df.loc[df[x_axis_name] < utils.BASE10_THRESHOLD, x_axis_name]).reshape(
        -1, 1
    )
    y = df.loc[df[x_axis_name] < utils.BASE10_THRESHOLD, y_axis_name]

    reg = lm.LinearRegression().fit(x, y)

    # Plot relationship
    plot = (
        pn.ggplot(
            df.sample(plotting_sample_size),
            pn.aes(x="log_preprocessed_rank", y="log_preprocessed_transcript_count"),
        )
        + pn.geom_abline(slope=reg.coef_, intercept=reg.intercept_, color="red")
        + pn.geom_point(color="blue", size=0.5)
        + pn.labs(
            x="Gene Log Rank",
            y="Gene Log Expression",
            title="Log Rank vs Log Expression",
        )
    )
    plot.save(os.path.join(eval_output_dir, "plot_log_rank_vs_log_expr.png"), dpi=300)

    # (2) Reconstruct expression from log rank, calculate reconstruction performance metrics
    rank_reconstructed_X = reg.predict(
        np.array(df["log_preprocessed_rank"]).reshape(-1, 1)
    )

    r_squared_score = r2_score(
        np.asarray(df["log_preprocessed_transcript_count"]),
        np.asarray(rank_reconstructed_X),
    )
    pearson_r_score = pearsonr(
        np.asarray(df["log_preprocessed_transcript_count"]),
        np.asarray(rank_reconstructed_X),
    )
    spearman_r_score = spearmanr(
        np.asarray(df["log_preprocessed_transcript_count"]),
        np.asarray(rank_reconstructed_X),
    )

    reconstructed_expr_values_df = pd.DataFrame(
        {
            "Ground Truth Expression": df["log_preprocessed_transcript_count"],
            "Reconstructed Expression from Log Rank": rank_reconstructed_X,
        }
    )
    plot = (
        pn.ggplot(
            reconstructed_expr_values_df.sample(plotting_sample_size),
            pn.aes(
                x="Ground Truth Expression", y="Reconstructed Expression from Log Rank"
            ),
        )
        + pn.geom_point(color="blue", size=0.5)
        + pn.geom_abline(slope=1, intercept=0, color="red")
        + pn.labs(
            x="Ground Truth Expression",
            y="Reconstructed Expression from Log Rank",
            title="Ground Truth Expression vs Reconstruction from Rank",
        )
    )
    plot.save(
        os.path.join(
            eval_output_dir, "plot_gt_expr_vs_reconstructed_expr_from_rank.png"
        ),
        dpi=300,
    )

    # 3. Create results dataframe and return
    metrics_df = pd.DataFrame(
        {
            "threshold": [utils.BASE10_THRESHOLD],
            "slope": [reg.coef_.item()],
            "intercept": [reg.intercept_.item()],
            "R^2": [r_squared_score.item()],
            "Pearson_R_statistic": [pearson_r_score.statistic.item()],
            "Pearson_R_p_value": [pearson_r_score.pvalue.item()],
            "Spearman_R_statistic": [spearman_r_score.statistic.item()],
            "Spearman_R_p_value": [spearman_r_score.pvalue.item()],
        }
    )
    metrics_df.to_csv(
        os.path.join(eval_output_dir, "transformation_metrics_and_parameters.csv")
    )


def main(data_filepath: Path, output_dir: Path):
    """Apply preprocessing steps and transform to cell sentences.

    Preprocessing follows https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html.
    """
    print(f"Loading data from {data_filepath}.")
    adata = anndata.read_h5ad(data_filepath)

    # reach for raw transcript counts in the .raw attribute
    if hasattr(adata, "raw") and adata.raw is not None:
        adata.X = adata.raw.X
    print(f"Done loading data for {len(adata)} cells.")

    # re-index gene names
    adata.var["feature_name"] = adata.var["feature_name"].astype(str)
    adata.var["ensembl_ids"] = adata.var.index
    adata.var_names = adata.var["feature_name"]
    adata.var_names_make_unique(join="_")

    # filter cells & genes below minimum occurence threshold
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # annotate the group of mitochondrial genes as 'mt'
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 200, :]
    print(f"Done filtering cells, remaining data of shape {adata.shape}.")

    raw_X = np.copy(adata.X.toarray())
    norm_X, rank_norm_X = normalize_and_rank_transform(
        np.copy(adata.X.todense()), normalize=True
    )
    # update adata object with normalized expression
    adata.X = np.log10(1 + norm_X)

    # create dataframe of ranks and expression values for plotting
    expr_and_rank_df = pd.DataFrame(
        {
            "raw_transcript_count": np.ravel(raw_X),
            "preprocessed_transcript_count": np.ravel(norm_X),
            "preprocessed_rank": np.ravel(rank_norm_X),
            "log_preprocessed_transcript_count": np.log10(1 + np.ravel(norm_X)),
            "log_preprocessed_rank": np.log10(1 + np.ravel(rank_norm_X)),
        }
    )
    # remove 0 expression entries in the cellxgene matrix
    expr_and_rank_df = expr_and_rank_df[expr_and_rank_df["raw_transcript_count"] != 0]
    print(f"Done normalizing data, {len(expr_and_rank_df)} data points remaining.")

    # compute metrics for transformation to cells and back
    evaluate_transformation(df=expr_and_rank_df, plotting_sample_size=10000)

    preprocessed_output_filepath = data_filepath.parent / (
        data_filepath.stem + data_filepath.suffix.replace(".h5ad", "_preprocessed.h5ad")
    )
    print(f"Saving preprocessed transcript counts to {preprocessed_output_filepath}.")
    adata.write_h5ad(preprocessed_output_filepath)

    # convert the adata into ranked sequences of gene names ("cell sentences")
    csdata = utils.csdata_from_adata(adata)

    # make text files containing the cell sentences
    txt_output_dir = output_dir / "cell_sentences"
    txt_output_dir.mkdir(exist_ok=True, parents=True)
    utils.xlm_prepare_outpath(csdata, txt_output_dir, species_tag="human")
    print(f"Done writing cell sentences to file.")

    # make arrow-formatted dataset compatible with HuggingFace's datasets
    hf_output_dir = output_dir / "cell_sentences_hf"
    hf_output_dir.mkdir(exist_ok=True, parents=True)
    data_splits = ["train", "valid", "test"]
    data_files = {
        data_split: str(txt_output_dir / f"{data_split}_human.txt")
        for data_split in data_splits
    }
    dataset = load_dataset("text", data_files=data_files)

    # load cell type labels if available with transcript counts
    for data_split in data_splits:
        dataset[data_split] = dataset[data_split].rename_column("text", "input_ids")
        # retrieve split chunk from preprocessed transcript counts
        dataset_split_sample_indices = np.load(
            txt_output_dir / f"{data_split}_partition_indices.npy"
        )
        adata_split = adata[dataset_split_sample_indices, :].copy()
        if "cell_type" in adata_split.obs.columns:
            cell_type_labels = {"cell_type": adata_split.obs["cell_type"].tolist()}
            cell_type_dataset = Dataset.from_dict(cell_type_labels)
            dataset[data_split] = concatenate_datasets(
                [dataset[data_split], cell_type_dataset], axis=1
            )

    dataset.save_to_disk(hf_output_dir)
    print(f"Done transforming data to cell sentences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filepath",
        type=Path,
        help="Input data filepath.",
        default=utils.DATA_DIR / "dominguez_sample.h5ad",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory filepath.",
        default=utils.DATA_DIR,
    )
    args = parser.parse_args()

    main(args.data_filepath, args.output_dir)

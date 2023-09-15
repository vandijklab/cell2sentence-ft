#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.utils import shuffle
from tqdm import tqdm

from src.csdata import CSData

DATA_DIR = Path("data/")
DATA_DIR.mkdir(exist_ok=True, parents=True)

BASE10_THRESHOLD = 3
SEED = 42


def generate_vocabulary(adata):
    """
    Create a vocabulary dictionary, where each key represents a single gene
    token and the value represents the number of non-zero cells in the provided
    count matrix.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
               `obs` correspond to cells and `vars` correspond to genes.
    Return:
        a dictionary of gene vocabulary
    """
    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({})... "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    vocabulary = OrderedDict()
    gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))

    for i, name in enumerate(adata.var_names):
        vocabulary[name] = gene_sums[i]

    return vocabulary


def generate_sentences(adata, prefix_len=None, random_state=42):
    """
    Transform expression matrix to sentences. Sentences contain gene "words"
    denoting genes with non-zero expression. Genes are ordered from highest
    expression to lowest expression.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
               `obs` correspond to cells and `vars` correspond to genes.
        random_state: sets the numpy random state for splitting ties
    Return:
        a `numpy.ndarray` of sentences, split by delimiter.
    """
    np.random.seed(random_state)

    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({}), "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    mat = sparse.csr_matrix(adata.X)
    sentences = []
    for i in tqdm(range(mat.shape[0])):
        cols = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i] : mat.indptr[i + 1]]

        cols, vals = shuffle(cols, vals)

        sentences.append(
            "".join([chr(x) for x in cols[np.argsort(-vals, kind="stable")]])
        )

    if prefix_len is not None:
        sentences = [s[:prefix_len] for s in sentences]

    return np.array(sentences, dtype=object)


def csdata_from_adata(adata, prefix_len=None, random_state=42):
    """
    Generate a CSData object from an AnnData object.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
               `obs` correspond to cells and `vars` correspond to genes.
        prefix_len: consider only rank substrings of length prefix_len
        random_state: sets the numpy random state for splitting ties
    Return:
        a CSData object containing a vocabulary, sentences, and associated name data.
    """
    return CSData(
        vocab=generate_vocabulary(adata),
        sentences=generate_sentences(
            adata, prefix_len=prefix_len, random_state=random_state
        ),
        cell_names=adata.obs_names,
        feature_names=adata.var_names,
    )


def xlm_prepare_outpath(csdata, outpath, species_tag, params=None):
    """
    Write formatted data to the outpath file location, for direct processing
    by the XLM monolinguistic translation model. If creating an outpath for
    multiple species, use the same `outpath` with different `species_tag`
    values. They will not conflict so long as species_tags are appropriately
    assigned.

    Note that XLM requires a dictionary sorted in order of increasing
    frequency of occurence.

    Arguments:
        csdata: a CSData object from a single species to be written.
        outpath: directory to write files to. Will create this directory
                 if it does not already exist.
        species_tag: a short string to be used as the species name in XLM.
                     Fulfills functions analaglous to language tags such as
                     'en', 'es', or 'zh'.
        delimiter: default = ' '. A token delimter for the generated sentences.
        params: a parameter object passed to train_test_validation_split:
    Return:
        None
    """

    if params is None:
        params = {}

    sentence_strings = csdata.create_sentence_strings(delimiter=" ")
    train, test, val = csdata.train_test_validation_split(**params)

    train_sentences = sentence_strings[train]
    test_sentences = sentence_strings[test]
    val_sentences = sentence_strings[val]

    os.makedirs(outpath, exist_ok=True)
    np.save(
        os.path.join(outpath, "train_partition_indices.npy"),
        np.array(train, dtype=np.int64),
    )
    np.save(
        os.path.join(outpath, "valid_partition_indices.npy"),
        np.array(val, dtype=np.int64),
    )
    np.save(
        os.path.join(outpath, "test_partition_indices.npy"),
        np.array(test, dtype=np.int64),
    )

    print("INFO: Writing Vocabulary File", file=sys.stderr)
    fn = "{}/vocab_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for k in tqdm(sorted(csdata.vocab, key=csdata.vocab.get, reverse=True)):
            if csdata.vocab[k] == 0:
                continue
            print("{} {}".format(k, csdata.vocab[k]), file=f)

    print("INFO: Writing Training Sentences", file=sys.stderr)
    fn = "{}/train_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(train_sentences):
            print(l, file=f)

    print("INFO: Writing Training Cell Barcodes", file=sys.stderr)
    fn = "{}/train_barcodes_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(csdata.cell_names[train]):
            print(l, file=f)

    print("INFO: Writing Testing Sentences", file=sys.stderr)
    fn = "{}/test_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(test_sentences):
            print(l, file=f)

    print("INFO: Writing Testing Cell Barcodes", file=sys.stderr)
    fn = "{}/train_barcodes_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(csdata.cell_names[test]):
            print(l, file=f)

    print("INFO: Writing Validation Sentences", file=sys.stderr)
    fn = "{}/valid_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(val_sentences):
            print(l, file=f)

    print("INFO: Writing Validation Cell Barcodes", file=sys.stderr)
    fn = "{}/valid_barcodes_{}.txt".format(outpath, species_tag)
    with open(fn, "w") as f:
        for l in tqdm(csdata.cell_names[val]):
            print(l, file=f)


def post_process_generated_cell_sentences(
    cell_sentence: str,
    global_dictionary: List,
    replace_nonsense_string: str = "NOT_A_GENE",
):
    """
    Post-processing function for generated cell sentences. Nonsense genes are replaced with 
    some string, e.g. 'NOT_A_GENE', so that ranks are not changed in generated output.

    Current assumptions in this function:
        - We replace nonsense genes with some string, e.g. 'NOT_A_GENE', so that ranks are not
            changed in generated output.

    Steps:
        1. Replace any nonsense genes with a specified token, e.g. 'NOT_A_GENE'
        2. Average the ranks of duplicated genes in generated sentence

    Arguments:
        cell_sentence:              generated cell sentence string
        global_dictionary:          list of global gene vocabulary (all uppercase)
        replace_nonsense_string:    string which will replace nonsense genes in generated output

    Returns:
        post_processed_sentence:    generated cell sentence after post processing steps
        num_nonsense_genes:         number of genes replaced with defined nonsense token
    """
    generated_gene_names = cell_sentence.split(" ")
    generated_gene_names = [generated_gene.upper() for generated_gene in generated_gene_names]

    # --- Replace nonsense genes ---#
    generated_gene_names = [
        gene_name if gene_name in global_dictionary else replace_nonsense_string
        for gene_name in generated_gene_names
    ]
    num_genes_replaced = generated_gene_names.count(replace_nonsense_string)

    # --- Average ranks ---#
    gene_name_to_occurrences = Counter(
        generated_gene_names
    )  # get mapping of gene name --> number of occurrences
    post_processed_sentence = generated_gene_names.copy()  # copy of generated gene list

    for gene_name in gene_name_to_occurrences:
        if (
            gene_name_to_occurrences[gene_name] > 1
            and gene_name != replace_nonsense_string
        ):
            # Find positions of all occurrences of duplicated generated gene in list
            # Note: using post_processed_sentence here; since duplicates are being removed, list will be
            #   getting shorter. Getting indices in original list will no longer be accurate positions
            occurrence_positions = [
                idx
                for idx, elem in enumerate(post_processed_sentence)
                if elem == gene_name
            ]
            average_position = int(
                sum(occurrence_positions) / len(occurrence_positions)
            )

            # Remove occurrences
            post_processed_sentence = [
                elem for elem in post_processed_sentence if elem != gene_name
            ]
            # Reinsert gene_name at average position
            post_processed_sentence.insert(
                average_position, gene_name
            )

    return post_processed_sentence, num_genes_replaced


def convert_cell_sentence_back_to_expression_vector(
    cell_sentence: List, global_dictionary: List, slope: float, intercept: float
):
    """
    Function to convert

    Current assumptions in this function:
        - We replace nonsense genes with some string, e.g. 'NOT_A_GENE', so that ranks are not
            changed in generated output.

    Steps:
        1. Replace any nonsense genes with a specified token, e.g. 'nan'
        2. Average the ranks of duplicated genes in generated sentence

    Arguments:
        cell_sentence:              generated cell sentence list, e.g. ['GENE1', 'GENE2']
        global_dictionary:          list of global gene vocabulary
        slope:                      slope value to use in inverse rank->expression transformation
        intercept:                  intercept value to use in inverse rank->expression transformation

    Returns:
        expression_vector:          expression vector for generated cell
    """
    expression_vector = np.zeros(len(global_dictionary), dtype=np.float32)
    for rank, gene_name in enumerate(cell_sentence):
        if gene_name in global_dictionary:
            log_rank = np.log10(1 + rank).item()
            gene_expr_val = intercept + (slope * log_rank)
            gene_idx_in_vector = global_dictionary.index(gene_name)
            expression_vector[gene_idx_in_vector] = gene_expr_val

    return expression_vector

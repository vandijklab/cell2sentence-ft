#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import os
import sys
from collections import OrderedDict
from pathlib import Path

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

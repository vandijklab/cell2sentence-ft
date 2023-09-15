"""
Serialize data for use with other external systems
"""

#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import os
import sys
import numpy as np
from tqdm import tqdm


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
        os.path.join(outpath, "val_partition_indices.npy"),
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

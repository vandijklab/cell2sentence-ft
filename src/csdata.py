#
# @author Rahul Dhodapkar
#
import zlib

import igraph as ig
import jellyfish
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import model_selection


def zlib_ncd(s1, s2):
    """
    Return the zlib normalized compression distance between two strings
    """
    bs1 = bytes(s1, "utf-8")
    bs2 = bytes(s2, "utf-8")

    comp_cat = zlib.compress(bs1 + bs2)
    comp_bs1 = zlib.compress(bs1)
    comp_bs2 = zlib.compress(bs2)

    return (len(comp_cat) - min(len(comp_bs1), len(comp_bs2))) / max(
        len(comp_bs1), len(comp_bs2)
    )


class CSData:
    """
    Lightweight wrapper class to wrap cell2sentence results.
    """

    def __init__(self, vocab, sentences, cell_names, feature_names):
        self.vocab = vocab  # Ordered Dictionary: {gene_name: num_expressed_cells}
        self.sentences = sentences  # list of sentences
        self.cell_names = cell_names  # list of cell names
        self.feature_names = feature_names  # list of gene names
        self.distance_matrix = None
        self.distance_params = None
        self.knn_graph = None

    def create_distance_matrix(self, dist_type="jaro", prefix_len=20):
        """
        Calculate the distance matrix for the CSData object with the specified
        edit distance method. Currently supported: ("levenshtein").

        Distance caculated as d = 1 / (1 + x) where x is the similarity score.
        """
        if self.distance_matrix is not None and (
            self.distance_params["dist_type"] == dist_type
            and self.distance_params["prefix_len"] == prefix_len
        ):
            return self.distance_matrix

        dist_funcs = {
            "levenshtein": jellyfish.levenshtein_distance,
            "damerau_levenshtein": jellyfish.damerau_levenshtein_distance,
            "jaro": lambda x, y: 1 - jellyfish.jaro_similarity(x, y),  # NOQA
            "jaro_winkler": lambda x, y: 1
            - jellyfish.jaro_winkler_similarity(x, y),  # NOQA
            "zlib_ncd": zlib_ncd,
        }

        is_symmetric = {
            "levenshtein": True,
            "damerau_levenshtein": True,
            "jaro": True,
            "jaro_winkler": True,
            "zlib_ncd": False,
        }

        mat = np.zeros(shape=(len(self.sentences), len(self.sentences)))

        for i, s_i in enumerate(self.sentences):
            for j, s_j in enumerate(self.sentences):
                if j < i and is_symmetric[dist_type]:
                    mat[i, j] = mat[j, i]
                    continue

                mat[i, j] = dist_funcs[dist_type](s_i[:prefix_len], s_j[:prefix_len])

        self.distance_params = {"dist_type": dist_type, "prefix_len": prefix_len}
        self.distance_matrix = mat
        # reset KNN graph if previously computed on old distance
        self.knn_graph = None

        return self.distance_matrix

    def create_knn_graph(self, k=15):
        """
        Create KNN graph
        """
        if self.distance_matrix is None:
            raise RuntimeError(
                'cannot "build_knn_graph" without running "create_distance_matrix" first'
            )

        adj_matrix = 1 / (1 + self.distance_matrix)
        knn_mask = np.zeros(shape=adj_matrix.shape)

        for i in range(adj_matrix.shape[0]):
            for j in np.argsort(-adj_matrix[i])[:k]:
                knn_mask[i, j] = 1

        masked_adj_matrix = knn_mask * adj_matrix

        self.knn_graph = ig.Graph.Weighted_Adjacency(masked_adj_matrix).as_undirected()
        return self.knn_graph

    def create_rank_matrix(self):
        """
        Generates a per-cell rank matrix for use with matrix-based tools. Features with zero
        expression are zero, while remaining features are ranked according to distance from
        the end of the rank list.
        """
        full_rank_matrix = np.zeros((len(self.cell_names), len(self.feature_names)))

        for i, s in enumerate((self.sentences)):
            for rank_position, c in enumerate(s):
                full_rank_matrix[i, ord(c)] = len(s) - rank_position

        return full_rank_matrix

    def find_differential_features(self, ident_1, ident_2=None, min_pct=0.1):
        """
        Perform differential feature rank testing given a set of sentence indexes.
        If only one group is given, the remaining sentences are automatically used
        as the comparator group.
        """

        if ident_2 is None:
            ident_2 = list(set(range(len(self.sentences))).difference(set(ident_1)))

        full_rank_matrix = self.create_rank_matrix()
        feature_ixs_to_test = np.array(
            np.sum(full_rank_matrix > 0, axis=0) > min_pct * len(self.cell_names)
        ).nonzero()[0]

        stats_results = []
        for f in feature_ixs_to_test:
            wilcox_stat, pval = stats.ranksums(
                x=full_rank_matrix[ident_1, f], y=full_rank_matrix[ident_2, f]
            )
            stats_results.append(
                {
                    "feature": self.feature_names[f],
                    "w_stat": wilcox_stat,
                    "p_val": pval,
                    "mean_rank_group_1": np.mean(full_rank_matrix[ident_1, f]),
                    "mean_rank_group_2": np.mean(full_rank_matrix[ident_2, f]),
                }
            )
        return pd.DataFrame(stats_results)

    def get_rank_data_for_feature(self, feature_name, invert=False):
        """
        Return an array of ranks corresponding to the prescence of a gene within
        each cell sentence. If a gene is not present in a cell sentence, np.nan
        is returned for that cell.

        Note that this returns rank (1-indexed), not position within the underlying
        gene rank list string (0-indexed).
        """
        feature_code = -1
        for i, k in enumerate(self.vocab.keys()):
            if k == feature_name:
                feature_code = i
                break

        if feature_code == -1:
            raise ValueError(
                "invalid feature {} not found in vocabulary".format(feature_name)
            )
        feature_enc = chr(feature_code)

        rank_data_vec = np.full((len(self.cell_names)), np.nan)
        for i, s in enumerate(self.sentences):
            ft_loc = s.find(feature_enc)
            if invert:
                rank_data_vec[i] = len(s) - ft_loc if ft_loc != -1 else np.nan
            else:
                rank_data_vec[i] = ft_loc + 1 if ft_loc != -1 else np.nan

        return rank_data_vec

    def create_sentence_strings(self, delimiter=" "):
        """
        Convert internal sentence representation (arrays of ints) to traditional
        delimited character strings for integration with text-processing utilities.
        """
        if np.any([delimiter in x for x in self.feature_names]):
            raise ValueError(
                (
                    'feature names cannot contain sentence delimiter "{}", '
                    + "please re-format and try again"
                ).format(delimiter)
            )

        enc_map = list(self.vocab.keys())

        joined_sentences = []
        for s in self.sentences:
            joined_sentences.append(delimiter.join([enc_map[ord(x)] for x in s]))

        return np.array(joined_sentences, dtype=object)

    def create_sentence_lists(self):
        """
        Convert internal sentence representation (arrays of ints) to
        sentence lists compatible with gensim
        """
        enc_map = list(self.vocab.keys())

        joined_sentences = []
        for s in self.sentences:
            joined_sentences.append([enc_map[ord(x)] for x in s])

        return np.array(joined_sentences, dtype=object)

    def train_test_validation_split(
        self, train_pct=0.8, test_pct=0.1, val_pct=0.1, random_state=42
    ):
        """
        Create train, test, and validation splits of the data given the supplied
        percentages with a specified random state for reproducibility.

        Arguments:
            sentences: an numpy.ndarray of sentences to be split.
            train_pct: Default = 0.6. the percentage of samples to assign to the training set.
            test_pct: Default = 0.2. the percentage of samples to assign to the test set.
            val_pct: Default = 0.2. the percentage of samples to assign to the validation set.
        Return:
            (train_sentences, test_sentences, val_sentences) split from the
            originally supplied sentences array.
        """
        if train_pct + test_pct + val_pct != 1:
            raise ValueError(
                "train_pct = {} + test_pct = {} + val_pct = {} do not sum to 1.".format(
                    train_pct, test_pct, val_pct
                )
            )

        s_1 = test_pct
        s_2 = val_pct / (1 - test_pct)

        X = range(len(self.sentences))
        X_train, X_test = model_selection.train_test_split(
            X, test_size=s_1, random_state=random_state
        )

        X_train, X_val = model_selection.train_test_split(
            X_train, test_size=s_2, random_state=random_state
        )

        return (X_train, X_test, X_val)

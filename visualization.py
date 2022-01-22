import pandas as pd
import matplotlib.pyploat as plt
import seaborn as sns
from typing import Union, Iterable, Optional


def plot_features_distribution(features: pd.DataFrame,
                               label: pd.DataFrame):
    r"""

    Args:
        features (pd.DataFrame):
            Should have index level "Features" for feature names and "FeatureGroup" for type of features.features. The
            columns should be patients IDs.
        label:
            Index should be patients ID and a column "Status" should contain the +ve/-ve status of the patient

    Returns:

    """
    import matplotlib.pyplot as plt
    import seaborn as sns



def plot_features_frequency(features_list: Iterable[pd.DataFrame]):
    features_indices_sets = [set(['__'.join(j) for j in i.index]) for i in features_list]
    all_features = union.set(*features_indices_sets)

    features_frequencies = {i: 0 for i in all_features}
    for i in all_features:
        for j in features_indices_sets:
            if i in j:
                features_frequencies[i] += 1
    features_frequencies = pd.Series(features_frequencies, name="frequencies")

    # Plot a bar chart

    return features_frequencies
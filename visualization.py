import pandas as pd

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


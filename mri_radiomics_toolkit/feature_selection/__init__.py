"""Feature selection subpackage.

This subpackage provides modular components for building feature selection pipelines.
Each component is designed to handle a specific feature selection task and can be
combined with others to create custom feature selection workflows.
"""

from .base import FeatureSelectionStep, FeatureSelectionPipeline
from .variance_filter import VarianceFilterStep
from .icc_filter import ICCFilterStep
from .statistical_filter import TTestFilterStep, ANOVAFilterStep
from .supervised_selection import SupervisedSelectionStep
from .normalization import NormalizationStep
from .bootstrapped_selection import BootstrappedSelectionStep
from .sklearn_compat import ElasticNetSelector
from .bbrent import BBRENTStep

__all__ = [
    # New modular implementation
    'FeatureSelectionStep',
    'FeatureSelectionPipeline',
    'VarianceFilterStep',
    'ICCFilterStep',
    'TTestFilterStep',
    'ANOVAFilterStep',
    'SupervisedSelectionStep',
    'NormalizationStep',
    'BootstrappedSelectionStep',
    'ElasticNetSelector',
    'BBRENTStep',
] 
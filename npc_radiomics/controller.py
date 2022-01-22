from . import FeatureExtractor, FeatureSelector, ModelBuilder

from typing import Optional
from pathlib import Path
from mnts.scripts.normalization import *
from mnts.mnts_logger import MNTSLogger

import yaml
import pprint
import pandas as pd

__all__ = ['Controller']

class Controller(object):
    r"""
    Args:
        with_norm (bool, Optional):
            If specified, the images and segmentation are normalized using the yaml file specified by
            the option `

    Attributes:
        extractor (FeatureExtractor):
            Module for extracting feature, `with_norm` option is passed to this.
        selector (FeatureSelector):
            Module for selecting features.
        model_builder (ModelBuilder):
            Module for model building.

    """
    def __init__(self,
                 *args,
                 setting = None,
                 with_norm = False,
                 **kwargs):
        super(Controller, self).__init__()

        self.extractor = FeatureExtractor(*args, **kwargs)
        self.selector = FeatureSelector(*args, **kwargs)
        self.model_builder = ModelBuilder(*args, **kwargs)
        self._receved_params = kwargs
        self._logger = MNTSLogger[__class__.__name__]
        self._with_norm = with_norm
        self._setting = setting

        self.saved_state = {
            'extractor': self.extractor,
            'selector': self.selector,
            'model': self.model_builder,
            'setting': self._setting,
            'norm_ready': False,
            'predict_ready': False,
        }

        if self._setting is not None:
            self.read_setting(Path(self._setting))

    def read_setting(self, f: Path) -> dict:
        r"""
        Settings Available:
            Extractor:
                id_globber

            Selector:
                criteria_threshold
                n_trials
                boot_runs
                boot_ratio
                thres_percentage
                return_freq
                boosting
        """
        with f.open('r') as stream:
            self._logger.info(f"Reading from file: {str(f)}")
            data_loaded = yaml.safe_load(stream)

        if 'Selector' in data_loaded.keys():
            _s = data_loaded['Selector']
            s = self.selector.saved_state['setting']
            s.update((k, _s[k]) for k in set(_s).intersection(s))
            self.selector.saved_state['setting'] = s
            self._logger.info(f"Updated selector setting: {pprint.pformat(s)}")

        if 'Extractor' in data_loaded.keys():
            _s = data_loaded['Extractor']
            id_globber = _s.get('id_globber', "^\w*")
            self.extractor.id_globber = id_globber
            self._logger.info(f"Updating extractor setting: {id_globber}")
            param_file = _s.get('pyrad_param_file', None)
            self._logger.info(f">>>>{param_file}")
            if not param_file is None:
                self.extractor.param_file = param_file  # the extractor handles it itself
                self.saved_state['pyrad_param_file'] = self.extractor.param_file

        if 'Controller' in data_loaded.keys():
            _s = data_loaded['Controller']


    def load_norm_settings(self,
                           norm_state_file: Optional[Path] = None,
                           norm_graph: Optional[Path] = None,
                           fe_state: Optional[Path] = None):
        r"""
        If `self.with_norm` is True, the normalization states and graph should be loaded ahead if this is a new
        instance. This will either call `self.extractor.load` if `fe_state` is specified, or put the two options into
        the fe_state manually. If you save the state of this class after calling this function, the normalization
        settings will automatically be stored, you don't need call this again after loading the saved states.
        """
        if not fe_state is None:
            self.extractor.load(fe_state)
        else:
            self.extractor.load_norm_graph(norm_graph)
            self.extractor.load_norm_state(norm_state_file)
        self.saved_state['norm_ready'] = True
        return 0

    def extract_feature(self,
                        img_path,
                        seg_path,
                        py_rad_param_file=None) -> pd.DataFrame:
        if not py_rad_param_file is None:
            self._logger.info("Overriding and updating pyradiomics param file.")
            self.extractor.param_file = py_rad_param_file
            self.saved_state['pyrad_param_file'] = self.extractor.param_file
        if self._with_norm:
            # Check if the norm_graph has been loaded
            if not self.saved_state.get('norm_ready', False):
                raise AttributeError(f"Feature extractor norm is not ready, please run `load_norm_setting` first.")
            self._logger.info("Extracting feature with normalization")
            df = self.extractor.extract_features_with_norm(img_path, seg_path, param_file=py_rad_param_file)
        else:
            self._logger.info("Extracting feature without normalization.")
            df = self.extractor.extract_features(img_path, seg_path, param_file=py_rad_param_file)
        return df

    def fit(self,
            img_path: Path,
            seg_path: Path,
            gt_path: Path,
            seg_b_path: Optional[Path] = None,
            with_normalization: Optional[bool] = None):
        r"""
        Note that while the `predict` method offers the option to do normalization, some of the normalization
        requires training and this `fit` method does not train the normalizer for you.
        """
        if self.extractor.param_file is None:
            raise AttributeError("Please set the pyradiomics parameter file first.")
        if with_normalization is not None:
            self._with_norm = bool(with_normalization)

        df_a = self.extract_feature(img_path, seg_path=seg_path)
        df_b = self.extract_feature(img_path, seg_path=seg_b_path) if seg_b_path is not None else None
        if gt_path.suffix == '.csv':
            gt_df = pd.read_csv(str(gt_path), index_col=0)
        elif gt_path.suffix == '.xlsx':
            gt_df = pd.read_excel(str(gt_path, index_col=0))
        overlap_index = set(df_a.index) & set(gt_df.index)
        if df_b is not None:
            overlap_index = overlap_index & set(df_b.index)
            df_b = df_b.loc[overlap_index]

        self._logger.debug(f"df_a:\n {df_a.to_string()}")
        self._logger.debug(f"gt_df:\n {gt_df.to_string()}")
        feats_a, feats_b = self.selector.fit(df_a.loc[overlap_index],
                                             gt_df.loc[overlap_index],
                                             X_b=df_b)

        results, predict_table = self.model_builder.fit(feats_a, gt_df)
        self.saved_state['predict_ready'] = True
        return 0

    def predict(self,
                img_path: Path,
                seg_path: Path,
                with_normalization: Optional[bool] = False):
        if with_normalization is not None:
            self._with_norm = bool(with_normalization)
        df = self.extract_feature(img_path, seg_path=seg_path)
        return self.model_builder.predict(df)

    def predict_df(self,
                   img_feat: pd.DataFrame):
        return self.model_builder.predict(df)

    def save(self, f):
        if any([v is None for v in self.saved_state.values()]):
            raise ArithmeticError("There are nothing to save.")
        joblib.dump(self.saved_state, filename=f.with_suffix('.ctl'))

    def load(self, f: Path):
        r"""
        Load att `self.saved_state`. The file saved should be a dictionary containing key 'selected_features', which
        points to a list of features in the format of pd.MultiIndex or tuple
        """
        assert Path(f).is_file(), f"Cannot open file {f}"
        d = joblib.load(f)
        if not isinstance(d, dict):
            raise TypeError("State loaded is incorrect!")
        self.saved_state.update(d)
        self.read_setting(self.saved_state['setting'])

        #TODO: write checks for each of the modules and make sure all are ready for inference.
        # note that for FE `with_norm` more checks are needed.
        self.saved_state['predict_ready'] = True
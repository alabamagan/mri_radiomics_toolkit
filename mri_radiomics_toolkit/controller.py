import pprint
import warnings
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
import yaml
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs
from typing import Union

from . import FeatureExtractor, FeatureSelector, ModelBuilder

__all__ = ['Controller']

class Controller(object):
    r"""A controller class for managing feature extraction, selection, and model building.

    This class provides a high-level API to manage and control the feature extraction, selection,
    and model building instances.

    Args:
        *args:
            Variable length argument list.
        setting (str, Optional):
            Path to a YAML file containing settings for the feature extractor, feature selector,
            and model builder modules. If this is provided, the settings are read from the file
            during initialization.
        with_norm (bool, Optional):
            If set to True, normalization is performed using settings from the YAML file specified
            in `setting`. Default is False.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        extractor (FeatureExtractor):
            The feature extractor module. It is responsible for extracting features from the data.
        selector (FeatureSelector):
            The feature selector module. It is responsible for selecting the most relevant features
            from the extracted features.
        model_builder (ModelBuilder):
            The model builder module. It is responsible for building a predictive model using the
            selected features.
        _with_norm (bool):
            Determines if normalization should be performed based on the value passed to `with_norm`
            during initialization.
        _setting (str):
            Holds the path to the YAML settings file if provided during initialization.
        _logger (MNTSLogger):
            Logger for Controller class.
        _receved_params (dict):
            Dictionary to store any additional keyword arguments received during initialization.
        saved_state (dict):
            Dictionary to store the current state of the pipeline. It includes:
             - 'extractor': Instance of the FeatureExtractor.
             - 'selector': Instance of the FeatureSelector.
             - 'model': Instance of the ModelBuilder.
             - 'setting': Current setting.
             - 'norm_ready': A boolean indicating whether the normalization is ready.
             - 'predict_ready': A boolean indicating whether the model is ready for prediction.

    """
    def __init__(self,
                 *args,
                 setting = None,
                 with_norm = False,
                 **kwargs):
        super(Controller, self).__init__()

        self._receved_params = kwargs
        self._logger = MNTSLogger[__class__.__name__]
        self._with_norm = with_norm
        self._setting = setting

        self.saved_state = {
            'extractor': FeatureExtractor(*args, **kwargs),
            'selector': FeatureSelector(*args, **kwargs),
            'model': ModelBuilder(*args, **kwargs),
            'setting': self._setting,
            'norm_ready': False,
            'predict_ready': False,
        }
        if self._setting is not None:
            self.read_setting(Path(self._setting))

    @property
    def extractor(self):
        return self.saved_state['extractor']

    @extractor.setter
    def extractor(self, val):
        self.saved_state['extractor'] = val

    @property
    def selector(self):
        return self.saved_state['selector']

    @selector.setter
    def selector(self, val):
        self.saved_state['selector'] = val

    @property
    def model_builder(self):
        return self.saved_state['model']

    @model_builder.setter
    def model_builder(self, val):
        self.saved_state['model'] = val

    def read_setting(self, f: Path) -> dict:
        r"""Reads and updates the settings for the feature extractor, selector, and controller.

        This method reads the settings from a YAML file and updates the settings of the feature
        extractor, feature selector, and the controller. The method also handles the case where
        the settings are already loaded and stored in the 'setting_file_stream' of the 'saved_state'
        attribute.

        Settings Available:
        - **Extractor**:
            - `id_globber` (str)
        - **Selector**:
            - `criteria_threshold` (float)
            - `n_trials` (int)
            - `boot_runs` (int)
            - `boot_ratio` (float)
            - `thres_percentage` (float)
            - `return_freq` (float)
            - `boosting` (bool)

        Args:
            f (Path):
                The path to a YAML file containing the settings.

        Returns:
            dict:
                A dictionary containing the loaded settings.

        Raises:
            FileNotFoundError:
                Raised when no file or saved settings are available to read from.

        .. See also::
            - :class:`FeatureExtractor`
            - :class:`FeatureSelector`

        """
        if not f is None:
            f = Path(f)
            with f.open('r') as stream:
                self._logger.info(f"Reading from file: {str(f)}")
                settings_loaded = yaml.safe_load(stream)
                # Read it to saved_state
                stream.seek(0)
                self.saved_state['setting_file_stream'] = stream.read()
        else:
            if self.saved_state.get('setting_file_stream', None) is None:
                msg = "Nothing is provided to read."
                raise FileNotFoundError(msg)
            else:
                with StringIO(self.saved_state['setting_file_stream']) as stream:
                    settings_loaded = yaml.safe_load(stream)

        if 'Selector' in settings_loaded.keys():
            _s = settings_loaded['Selector']
            s = self.selector.saved_state['setting']
            s.update((k, _s[k]) for k in set(_s).intersection(s))
            self.selector.saved_state['setting'] = s
            self._logger.info(f"Updated selector setting: {pprint.pformat(s)}")

        if 'Extractor' in settings_loaded.keys():
            _s = settings_loaded['Extractor']
            id_globber = _s.get('id_globber', "^\w*")
            self.extractor.id_globber = id_globber
            param_file = _s.get('pyrad_param_file', None)
            if not param_file is None:
                self.extractor.param_file = param_file  # the extractor handles it itself
                self.saved_state['pyrad_param_file'] = self.extractor.param_file
            else:
                msg = "No pyrad param file was found in saved state."
                self._logger.warning(msg)
            self._logger.info(f"Updating extractor setting: {pprint.pformat(_s)}")

        if 'Controller' in settings_loaded.keys():
            _s = settings_loaded['Controller']

    def set_pyrad_param_file(self,
                             param_file_path: Union[Path, str]) -> None:
        r"""Set the configuration file for feature extraction. This is optional if you only want prediction from already
        extracted features. However, if you want to perform prediction from images directly, you must set the param file
        using either this file or the `pyrad_param_file` attribute of the controller setting.

        Args:
            param_file_path (Path or str):
                Path to the yml Pyradiomics setting file.

        Raises:
            FileNotFoundError
                Error is raised if the input file cannot be opened

        """
        param_file_path = Path(param_file_path)
        if not param_file_path.is_file():
            raise FileNotFoundError(f"Cannot open file: {str(param_file_path)}")



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

        # Align the ids first
        id_globber = self.extractor.id_globber
        matched_ids, missing_ids, duplicates = self.validate_ids(img_path, seg_path)
        if any(len(mids) for mids in missing_ids.values()):
            self._logger.warning(f"Find ID mismatch.")
            self._logger.debug(f"{matched_ids = }\n{missing_ids = }")
            raise AttributeError(f"ID globber ({id_globber}) is not set properly.")

        df_a = self.extract_feature(img_path, seg_path=seg_path) # rows are features columns are datapoints
        df_b = self.extract_feature(img_path, seg_path=seg_b_path) if seg_b_path is not None else None
        if gt_path.suffix == '.csv':
            gt_df = pd.read_csv(str(gt_path), index_col=0)
        elif gt_path.suffix == '.xlsx':
            gt_df = pd.read_excel(str(gt_path, index_col=0))
        overlap_index = df_a.index.intersection(gt_df.index)
        if df_b is not None:
            overlap_index = overlap_index.intersection(df_b.index)
            df_b = df_b.loc[list(overlap_index)] # pandas doesn't allow using set as indexers
        overlap_index = list(overlap_index)

        self._logger.debug(f"df_a:\n {df_a.to_string()}")
        self._logger.debug(f"gt_df:\n {gt_df.to_string()}")
        feats_a, feats_b = self.selector.fit(df_a.loc[overlap_index],
                                             gt_df.loc[overlap_index],
                                             X_b=df_b)

        results, predict_table = self.model_builder.fit(feats_a, gt_df)
        self.saved_state['predict_ready'] = True
        return 0

    def fit_df(self,
               df_a: pd.DataFrame,
               gt_df: pd.DataFrame,
               df_b: Optional[pd.DataFrame] = None,
               **kwargs) -> Tuple[pd.DataFrame]:
        r"""
        This is a direct port of `selector.fit()`, and then `model_builder.fit()` The features should have the patients
        as rows and feature as columns. In other words, this function skips the extraction step.

        Args:
            df_a (pd.DataFrame):
                Dataframe with patients as rows and features as columns.
            gt_df (pd.DataFrame):
                Dataframe with patients as row and at least a column "Status"
            df_b (pd.DataFrame, Optional):
                If provided and features have not been selected, feature selection would be carried out based on the
                setting in the controller setting yml file. Note that it could take hours to finish the feature
                selection. The dataframe should have patients as rows and features as columns.
            **kwargs:
                These will be passed eventually to :func:`model_building.cv_grid_search`. See that function for more.

        """
        if (self.extractor.param_file is None) and (self.selected_features is None):
            msg = f"Param file for Pyradiomics feature extraction is not provided. The controller fitted without " \
                  f"this setting file cannot perform predictions directly on input images."
            self._logger.warning(msg)

        overlap_index = df_a.index.intersection(gt_df.index)
        if df_b is not None:
            overlap_index = overlap_index.intersection(df_b.index)
            df_b = df_b.loc[overlap_index]

        self._logger.debug(f"df_a:\n {df_a.to_string()}")
        self._logger.debug(f"gt_df:\n {gt_df.to_string()}")

        # if features are not selected yet
        if self.selected_features is None:
            feats_a, feats_b = self.selector.fit(df_a.loc[overlap_index],
                                                 gt_df.loc[overlap_index],
                                                 X_b=df_b)
        else:
            if not df_b is None:
                self._logger.warning("Received `df_b` input but features are already selected. Ignoring `df_b`")
            try:
                feats_a = df_a[self.selected_features]
            except KeyError as e:
                msg = f"Some keys of the selected features is not in `df_a` or `df_b`: \n" \
                      f"{self.selected_features.difference(df_a.columns)}"
                raise KeyError(msg)

        results, predict_table = self.model_builder.fit(feats_a.loc[overlap_index], gt_df.loc[overlap_index],
                                                        **kwargs)
        self.saved_state['predict_ready'] = True
        return results, predict_table


    def predict(self,
                img_path: Path,
                seg_path: Path,
                with_normalization: Optional[bool] = False) -> pd.DataFrame:
        if self.extractor.param_file is None:
            msg = f"Pyradiomics settings was not loaded. Cannot perform direct prediction on images because the " \
                  f"controller doesn't know how to extract the features. Perform this setting using the method " \
                  f"`set_pyrad_param_file(file_dir)`."
            raise AttributeError
        if with_normalization is not None:
            self._with_norm = bool(with_normalization)
        df = self.extract_feature(img_path, seg_path=seg_path)
        return self.model_builder.predict(df)

    def predict_df(self,
                   img_feat: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        r"""Directly perform prediction on input dataframe.

        Args:
            img_feat (pd.DataFrame):
                Input features. The rows should be datapoints and the columns should be features.
        """
        return self.model_builder.predict(self.selector.predict(img_feat))

    def save(self, f: Path) -> int:
        if any([v is None for v in self.saved_state.values()]):
            raise ArithmeticError("There are nothing to save.")
        f = Path(f)
        _logger_store = {}
        for key in ['selector', 'extractor', 'model']:
            # temporally remove loggers because it will block joblib.dump from working properly
            _logger_store[key] = self.saved_state[key]._logger
            self.saved_state[key]._logger = None

        joblib.dump(self.saved_state, filename=f.with_suffix('.ctl'))
        # Return the logger
        for key in ['selector', 'extractor', 'model']:
            self.saved_state[key]._logger = _logger_store[key]
        del _logger_store # Manually delete the reference to ensure it will release memory after running
        return 0

    def load(self, f: Path) -> None:
        r"""
        Load att `self.saved_state`. The file saved should be a dictionary containing key 'selected_features', which
        points to a list of features in the format of pd.MultiIndex or tuple
        """
        assert Path(f).is_file(), f"Cannot open file {f}"
        d = joblib.load(f)
        if not isinstance(d, dict):
            raise TypeError("State loaded is incorrect!")
        self.saved_state.update(d)
        if not 'setting_file_stream' in self.saved_state:
            self.read_setting(Path(self.saved_state['setting']))
        else:
            # Read from saved file stream if possible
            self.read_setting(None)

        #TODO: write checks for each of the modules and make sure all are ready for inference.
        # note that for FE `with_norm` more checks are needed.

        # Set up loggers
        for key in ['selector', 'extractor', 'model']:
            self.saved_state[key]._logger = MNTSLogger[type(self.saved_state[key]).__name__]
        self.saved_state['predict_ready'] = True

    @property
    def selected_features(self):
        if not self.saved_state['predict_ready']:
            self._logger.error("Features was not selected yet! Call fit() or load() first!")
            return None
        else:
            return self.selector.selected_features

    @selected_features.setter
    def selected_features(self, v):
        raise ArithmeticError("Selected features should not be manually assigned.")

    def validate_ids(self, img_dir: Path, seg_dir: Path):
        """
        Validates and aligns image and segmentation IDs using the ID globber.

        Args:
            img_dir (Path):
                Directory containing the image files.
            seg_dir (Path):
                Directory containing the segmentation files.

        Returns:
            tuple:
                A tuple containing:
                    - matched_ids (set):
                        Set of IDs present in both images and segmentations.
                    - missing_ids (dict):
                        Dictionary with keys 'images' and 'segmentations' containing
                        lists of missing IDs.
                    - duplicates (dict):
                        Dictionary with keys 'images' and 'segmentations' containing
                        dictionaries of IDs with multiple files.

        Raises:
            AttributeError:
                If the ID globber is not properly set.
        """
        # Only run after initialization
        assert self.extractor is not None, "This instance has not been normalized"


        # Get unique IDs with their associated files
        image_id_dict = get_unique_IDs(img_dir.iterdir(),
                                       self.extractor.id_globber,
                                       return_dict=True)
        seg_id_dict = get_unique_IDs(seg_dir.iterdir(),
                                     self.extractor.id_globber,
                                     return_dict=True)

        # Find overlapping and missing IDs
        image_ids = set(image_id_dict.keys())
        seg_ids = set(seg_id_dict.keys())
        matched_ids = image_ids.intersection(seg_ids)

        missing_ids = {
            'images': list(seg_ids - image_ids),
            'segmentations': list(image_ids - seg_ids)
        }

        # Check for duplicates
        duplicates = {
            'images': {id_: files for id_, files in image_id_dict.items()
                       if len(files) > 1 and id_ in matched_ids},
            'segmentations': {id_: files for id_, files in seg_id_dict.items()
                              if len(files) > 1 and id_ in matched_ids}
        }

        return matched_ids, missing_ids, duplicates

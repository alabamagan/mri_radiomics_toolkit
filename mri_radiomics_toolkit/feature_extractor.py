import logging
import multiprocessing as mpi
import os
import pprint
import re
import tempfile
# import traceback
import time
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Callable, Tuple, Any, List
from zipfile import ZIP_DEFLATED, ZipFile

import SimpleITK as sitk
import joblib
import numpy as np
import pandas as pd
import radiomics
from tqdm import tqdm
from mnts.filters import MNTSFilterGraph
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.normalization import run_graph_inference
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from radiomics import featureextractor
from .utils import zipdir, compress, decompress, is_compressed, ExcelWriterProcess, unify_dataframe_levels


# Fix logger

__all__ = ['FeatureExtractor']

normalization_yaml =\
"""
SpatialNorm:
    out_spacing: [0.45, 0.45, 0]

HuangThresholding:
    closing_kernel_size: 10
    _ext:
        upstream: 0 
        is_exit: True

N4ITKBiasFieldCorrection:
    _ext:
        upstream: [0, 1]
    
NyulNormalizer:
    _ext:
        upstream: [2, 1]
        is_exit: True

"""

def progress_bar_helper(_, pbar: Callable = None) -> None:
    r"""Helper function to update a progress bar for mpi progress monitoring."""
    if pbar is None:
        raise ValueError("No progress bar.")
    pbar.update()

def check_images_matches(image1: sitk.Image, image2: sitk.Image) -> bool:
    r"""Checks if two images are in the same space"""
    return (image1.GetDimension() == image2.GetDimension() and
            image1.GetSize() == image2.GetSize() and
            np.all(np.isclose(image1.GetOrigin(), image2.GetOrigin(), atol=1E-4)) and
            np.all(np.isclose(image1.GetSpacing(), image2.GetSpacing(), atol=1E-4)) and
            np.all(np.isclose(np.array(image1.GetDirection()).flatten(),
                              np.array(image2.GetDirection()).flatten(), atol=1E-4)))

def get_radiomics_features(fn: Path,
                           mn: Path,
                           param_file: Path,
                           *args,
                           id_globber: str = "^[0-9a-zA-Z]+",
                           by_slice: int = -1,
                           connected_components: Optional[bool] = False,
                           writer_func: Optional[Callable] = None,
                           mpi_progress: Optional[Tuple[Any, Any]] = None,
                           resample: Optional[bool] = False) -> pd.DataFrame:
    r"""
    Return the features computed by pyramdiomics in a `pd.DataFrame` structure. This data
    output will at most has three column levels. The primary level is `Study number`,
    which are globbed by the `id_globber` from the `fn`. The secondary level is `Slice
    number`, which will only be added if you set `by_slice` to be >= 0. The tertiary level
    is the `Class code`, which will only be added if the segmentation `mn` consist of more
    than one class value.

    Args:
        fn (Path):
            Image file directory (needs to be .nii.gz).
        mn (Path):
            Segmentation file directory.
        param_file (Path):
            Path to the pyradiomics setting.
        *args (Sequence of Path):
            If this is provided, they directory is treated as extra segmentation
        id_globber (str, Optional):
            Regex pattern for globbing the study ID from `fn`. Default to "^[0-9a-zA-Z]"
        by_slice (int, Optional):
            If an integer >= 0 is specified, the slices along the specified axis will be
            scanned one-by-one and features will be extracted from the slices with
            segmentation. The output `pd.DataFrame` will have an additional column level
            called `Slice number`, which will mark the number of slice that set of
            features were extracted. Default to be -1.
        connected_components (bool, Optional):
            If True, convert the msk into binary mask and then perform connected body
            filter to separate the mask into node island before extraction. Default to
            `False.
        writer_func (Callable, Optional):
            If not `None`, the data will be streamed to the specified file after each data
            is processed. The callable provided is called after the calculations are done
            with syntax `writer_func(df)`. Default to `None`
        mpi_progress:
        resample (bool, Optional):
            If true, resmpale segmentation to image if there's a mismatch found.
    Returns:
        pd.DataFrame:
            DataFrame of features. Rows are features and columns are data points.
    """
    assert fn.is_file() and mn.is_file(), f"Cannot open images or mask at: {fn} and {mn}"

    try:
        logger = MNTSLogger['radiomics_features']
        logger.info(f"Current thread: {mpi.current_process().name}")
        logger.debug(pprint.pformat({'Input': fn, 'Mask': mn, 'args': args}))
        im = sitk.ReadImage(str(fn))
        msk = sitk.ReadImage(str(mn))
        logger.debug(f"Finishing reading...")
        if len(args) > 0:
            logger.info(f"Get multiple segmentations, got {1 + len(args)}")
            msk = [msk] + [sitk.ReadImage(str(a)) for a in args]
        else:
            msk = [msk]
        # check if they have same spacing
        for i, _msk in enumerate(msk):
            if not check_images_matches(_msk, im):
                logger.warning(f"Detected differences in spacing! Resampling "
                               f"Spacing: {_msk.GetSpacing()} -> {im.GetSpacing()} "
                               f"Direction: {_msk.GetDirection()} -> {im.GetDirection()} "
                               f"Origin: {_msk.GetOrigin()} -> {im.GetOrigin()} ")
                filt = sitk.ResampleImageFilter()
                filt.SetInterpolator(sitk.sitkNearestNeighbor)
                filt.SetReferenceImage(im)
                msk[i] = filt.Execute(_msk)

            if not (np.isfinite(sitk.GetArrayFromImage(im)).all() and np.isfinite(sitk.GetArrayFromImage(_msk)).all()):
                logger.warning(f"Detected NAN in image {str(fn)}")

        #╔═════════════════════╗
        #║▐ Extract features   ║
        #╚═════════════════════╝
        _start_time = time.time()
        logger.info(f"Extracting feature...")
        feature_extractor = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))
        dfs = []
        for i, _msk in enumerate(msk):
            # cast mask as uint
            _msk = sitk.Cast(_msk, sitk.sitkUInt8)

            if connected_components:
                # convert msk to binary and then rearrange ids
                _msk = _msk != 0
                _msk = sitk.ConnectedComponent(_msk)

            # check if the mask has more than one classes
            label_stat = sitk.LabelShapeStatisticsImageFilter()
            label_stat.Execute(_msk)
            val = set(label_stat.GetLabels())
            
            if len(val) == 0:
                raise ValueError("The label file seems to be empty!")

            # Create a new binary image for each non-zero classes
            _msk = {v: _msk == v for v in val}

            cols = []
            for val, _binmsk in _msk.items():
                if by_slice < 0:
                    _x = feature_extractor.execute(im, _binmsk)
                    logger.debug(f"finished extraction for mask {i} value {val}")
                    _df = pd.DataFrame.from_dict({k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                                                 for k, v in _x.items()}, orient='index')

                    if len(_msk) == 1:
                        _col_name = [f'{re.search(id_globber, fn.name).group()}']
                    else:
                        _col_name = pd.MultiIndex.from_tuples([[f'{re.search(id_globber, fn.name).group()}',
                                                                f'C{val}']],
                                                              name = ('Study number', 'Class code'))
                    _df.columns = _col_name
                    _df.columns.name = 'Study number'

                else:
                    # if features are to be extracted slice-by-slice
                    assert by_slice <= _binmsk.GetDimension(), \
                        f"by_slice is larger than dimension: {_binmsk.GetDimension()}"
                    logger.info("Extracting features by slice...")
                    # try to get worker thread number
                    slice_cols = []
                    case_number = re.search(id_globber, fn.name).group()
                    for j in tqdm(range(_binmsk.GetSize()[by_slice]), position=1):
                        _index = [slice(None)] * by_slice + [j]
                        _slice_im = sitk.JoinSeries(im[_index])
                        _slice_seg = sitk.JoinSeries(_binmsk[_index])
                        # skip if slice is empty
                        if sitk.GetArrayFromImage(_slice_seg).sum() == 0:
                            continue

                        _x = feature_extractor.execute(_slice_im, _slice_seg)
                        logger.debug(f"finished extraction for mask {i}, slice {j} value {val}")
                        slice_df = pd.DataFrame.from_dict({k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                                                     for k, v in _x.items()}, orient='index')

                        _col_name = pd.MultiIndex.from_tuples([[f'{case_number}',
                                                                f'S{j}',
                                                                f'C{val}']],
                                                              names=('Study number', 'Slice number', 'Class code'))
                        slice_df.columns = _col_name
                        slice_cols.append(slice_df)
                        pass
                    _df = pd.concat(slice_cols, axis=1)
                cols.append(_df)
            df = pd.concat(cols, axis=1)
            if len(msk) > 1:
                df.columns = pd.MultiIndex.from_product([df.columns, ['Segment' + chr(ord('A') + i)]])
            dfs.append(df)
        if len(dfs) == 1:
            out = dfs[0]
        else:
            out = pd.concat(dfs, axis=1)
        logger.debug(f"Intermediate result:\n {out}")
        _end_time = time.time()
        logger.info(f"Finsihed extracting features, time took: {_end_time - _start_time}s")

        # If writer is provided, write it
        if writer_func is not None:
            writer_func(out)

        # Update progress
        if isinstance(mpi_progress, tuple):
            lock, progress = mpi_progress
            # prevent simultaneous access to it the process
            with lock:
                progress.value += 1
        return out

    except Exception as e:
        MNTSLogger['radiomics_features'].error(f"Error during get_radiomics_features for image and mask {fn} and {mn}!")
        MNTSLogger['radiomics_features'].exception(e)

def get_radiomics_features_from_folder(im_dir: Path,
                                       seg_dir: Path,
                                       param_file: Path,
                                       *args,
                                       id_globber: str = "^[0-9a-zA-Z]+",
                                       idlist: Iterable[str] = None,
                                       num_worker: Optional[int] = 1,
                                       **kwargs) -> pd.DataFrame:
    r"""
    This pairs up the image and the segmentation files using the global regex globber

    Args:
        im_dir (Path):
            Folder that contains the image files
        seg_dir (Path):
            Folder that contains the segmentation files
        param_file (Path):
            Path to the pyradiomics setting.
        idlist (list):
            A list of IDs for extraction.

        *args:
            If this exist, arguments will be attached to `mask_dir`, additional features will be extracted
            if there are more than one masks


    Return:
        pd.DataFrame
    """
    logger = MNTSLogger['radiomics_features']
    ids = get_unique_IDs(list([str(i.name) for i in im_dir.iterdir()]), id_globber)

    # Check if provided ID list can be found in the specified folder
    if idlist is not None:
        ids_set = set(ids)
        idlist_set = set(idlist)

        # Find elements present in both sets
        overlap = ids_set & idlist_set

        # Find missing elements from both sets
        missing_from_files = idlist_set - ids_set
        missing_from_idlist = ids_set - idlist_set

        if missing_from_files:
            logger.warning(f"IDs specified in idlist but missing from target folder: \n{missing_from_files}")

        if missing_from_idlist:
            logger.warning(f"IDs found in target folder but not specified in idlist: \n{missing_from_idlist}")

        ids = overlap

    # Load the pairs
    if not args is None:
        mask_dir = [seg_dir] + list(args)
    else:
        mask_dir = [seg_dir]

    mask = []
    source = None
    for i, msk in enumerate(mask_dir):
        # For each mask dir, load the IDs that matches with the input dir
        _source, _mask = load_supervised_pair_by_IDs(im_dir, msk, ids,
                                                    globber=id_globber, return_pairs=False)
        logger.debug(f"Find {len(_source)} pairs.")
        if len(_source) != len(ids):
            # This shouldn't happen.
            raise RunTimeError("Something went wrong when pairing the images and masks")
        mask.append(_mask)
        if i == 0:
            source = _source # First is already a list
        else:
            logger.info("Multiple masks detected.")
            if not len(source) == len(_source):
                # If multiple masks are used, the IDs must perfectly align between all mask files and image files
                raise IndexError(f"Length of the inputs are incorrect, somethings is wrong with index globbing for "
                                 f"{str(msk)}.")

    r"""
    Multi-thread
    """
    if len(mask) == 1:
        z = repeat_zip(source, mask[0], [param_file])
    else:
        logger.info("Using multiple mask for extraction!")
        z = repeat_zip(source, mask[0], [param_file], *mask[1:])
        
    if num_worker > 1:
        with mpi.Manager() as manager:
            # configure the progress bar
            progress = manager.Value('i', 0)
            pbar = tqdm(total=len(source), desc=f"Feature extraction", leave=True)

            # create the worker pool
            pool = mpi.Pool(num_worker) # pyradiomics also runs in multi-thread
            logger.info(f"Multi-thread with {num_worker} workers.")
            func = partial(get_radiomics_features,
                        id_globber=id_globber,
                        mpi_progress=(manager.Lock(), progress), **kwargs)

            res = pool.starmap_async(func, z)
            # Update progress bar
            while not res.ready():
                pbar.n = progress.value
                pbar.refresh(nolock=False)
                time.sleep(0.1)

            # let the pbar finish last refresh
            time.sleep(0.1)
            pbar.n = progress.value
            pbar.refresh(nolock=False)

            # close the pool
            pool.close()
            pool.join()
            res = res.get()

            # closs the pbar
            pbar.close()
    else:
        res = []
        for zz in z:
            res.append(get_radiomics_features(*zz, id_globber=id_globber, **kwargs))

    df = unify_dataframe_levels(pd.concat(res, axis=1), axis=1)
    new_index = [o.split('_') for o in df.index]
    new_index = pd.MultiIndex.from_tuples(new_index, names=('Pre-processing', 'Feature_Group', 'Feature_Name'))
    df.index = new_index
    return df




class FeatureExtractor(object):
    r"""`FeatureExtractor` is a class for extracting features from image data based on given parameters.
    Features extracted are stored in a pandas DataFrame.

    Attributes:
        DEFAULT_NORM_STATE (pathlib.Path):
            Default path for normalization state, located in "assets/t2wfs".
        id_globber (str):
            A regex pattern to be used for globbing IDs. Defaults to "^[0-9a-zA-Z]+".
        idlist (list):
            A list of IDs to be used. Defaults to None.
        param_file (str or pathlib.Path):
            Path to a file containing parameters for feature extraction. Can also be a YAML-formatted string.
            Defaults to None.
        by_slice (int):
            The slice number to be used. Defaults to -1.
        _extracted_features (pd.DataFrame
            ): A DataFrame storing the features extracted.
        _logger (Logger
            ): A logging object.

    Example:
        >>> fe = FeatureExtractor()
        >>> fe.extract_features(im_path, seg_path, idlist=id_list, param_file=param_file_path)
        >>> fe.save_features(output_path)

    .. notes::
        The `param_file` attribute can be a path pointing to a YAML file containing extraction parameters.
        Alternatively, it can be a string formatted in YAML, which directly specifies the parameters.
        The `idlist` attribute should be a list of identifiers corresponding to the images from which features
        will be extracted.
    """
    DEFAULT_NORM_STATE = Path(__file__).parent.joinpath("assets/t2wfs").resolve()

    def __init__(self, *, id_globber = "^[0-9a-zA-Z]+", idlist = None, param_file = None, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.saved_state = {
            'param_file': None, # Path to param file
            'norm_state_file': FeatureExtractor.DEFAULT_NORM_STATE,    # Override in `extract_features_with_norm` if specified
            'norm_graph': None,
        }
        self.idlist = idlist
        self.param_file = param_file
        self.id_globber = id_globber
        self.by_slice = -1
        self._extracted_features = None
        self._logger = MNTSLogger[__class__.__name__]


    @property
    def param_file(self):
        out = self.saved_state['param_file']
        if is_compressed(out):
            out = decompress(out)
        return out

    @param_file.setter
    def param_file(self, v: Union[str, Path]):
        if str(v).find('.yml') > -1:
            assert Path(v).is_file(), f"Cannot open {str(v)}"
            self.saved_state['param_file'] = ''.join(Path(v).open('r').readlines())
        else:
            self.saved_state['param_file'] = v

    def load(self, f: Path):
        r"""Load att `self.save_state`. The file saved should be a dictionary containing key 'selected_features', which
        points to a list of features in the format of pd.MultiIndex or tuple

        Args:
            f (pathlib.Path): The path to the file from which the state should be loaded.

        Raises:
            AssertionError: If the file does not exist.
            TypeError: If the loaded state is not a dictionary.
        """
        assert Path(f).is_file(), f"Cannot open file {f}"
        d = joblib.load(f)
        if not isinstance(d, dict):
            raise TypeError("State loaded is incorrect!")
        self.saved_state.update(d)

    def save(self, f: Path):
        r"""Save the current state to a file.

        Args:
            f (pathlib.Path): The path to the file where the state should be saved.
        """
        if any([v is None for v in self.saved_state.values()]):
            self._logger.warning("Some saved state components are None.")
        if f.is_dir(): # default name
            f = f.joinpath('saved_state.fe')
        # Always save the compressed version of the param file.
        if not is_compressed(self.saved_state['param_file']):
            self.saved_state['param_file'] = compress(self.param_file)
        joblib.dump(self.saved_state, filename=f.with_suffix('.fe'))


    def save_features(self, outpath: Path):
        """Save the extracted features to an Excel or CSV file.

        Args:
            outpath (pathlib.Path):
                The path where the features should be saved. The file type is determined by the suffix.

        Raises:
            AssertionError: If the parent directory does not exist.
        """
        outpath = Path(outpath).resolve()
        assert outpath.parent.is_dir(), f"Cannot save to: {outpath}"
        if outpath.suffix == '':
            outpath = outpath.with_suffix('.xlsx')

        if outpath.suffix == '.xlsx' or outpath.suffix is None:
            self._extracted_features.to_excel(str(outpath.with_suffix('.xlsx').resolve()))
        elif outpath.suffix == '.csv':
            self._extracted_features.to_csv(str(outpath.resolve()))

    def extract_features(self,
                         im_path: Path,
                         seg_path: Path,
                         *args,
                         idlist: Optional[Iterable[str]] = None,
                         param_file: Optional[Path] = None,
                         by_slice: Optional[int] = -1,
                         stream_output: Optional[bool] = None,
                         connected_components: Optional[bool] = False,
                         num_workers: Optional[int] = 8) -> pd.DataFrame:
        """Extract features from image data based on the given parameters.

        Args:
            im_path (pathlib.Path):
                The path to the image data.
            seg_path (pathlib.Path):
                The path to the segmentation data.
            idlist (Iterable[str], optional):
                A list of IDs to be used. Defaults to None.
            param_file (str or pathlib.Path, optional):
                Path to a file containing parameters for feature extraction.
                Can also be a YAML-formatted string. Defaults to None.
            by_slice (int, optional):
                Whether to extract the features slice-by-slice and which axis to step. See
                :func:`get_radiomics_features` for details. Defaults to -1.
            stream_output (bool, optional):
                If this is `True`, program will try to stream output using :class:`ExcelWriterProcess`.
                Defaults to `False`.
            num_workers (int, optional):
                Number of workers for threaded feature extraction. Default to 1.

        Returns:
            pd.DataFrame: The DataFrame containing the extracted features. Dimensions are (n_samples, n_features).

        Raises:
            ArithmeticError: If no param file is specified.
            FileNotFoundError: If the param file does not exist.
        """
        if param_file is None and self.saved_state['param_file'] is None:
            raise ArithmeticError("Please specify param file.")
        if param_file is None:
            param_file = self.saved_state['param_file']
        if idlist is None:
            idlist = self.idlist
        if by_slice is None:
            by_slice = self.by_slice

        # Handle param_file in different formats (Path, string filepath, or raw string content)
        try:
            if isinstance(param_file, Path):
                # If it's a Path object, read its content
                if param_file.is_file():
                    self.param_file = param_file.read_text()
                else:
                    raise FileNotFoundError(f"Cannot find pyrad param file: {param_file}")
            elif isinstance(param_file, str):
                # Try to interpret string as a file path first
                path_obj = Path(param_file)
                if path_obj.is_file():
                    self.param_file = path_obj.read_text()
                else:
                    # Not a valid file path, check if it's compressed content or raw YAML
                    if is_compressed(param_file):
                        self.param_file = decompress(param_file)
                    else:
                        self.param_file = param_file  # Use as raw string
            elif param_file is None:
                raise ValueError("Parameter file is required but was None")
            else:
                raise TypeError(f"param_file must be a string or Path object, got {type(param_file)}")
        except OSError as e:
            # Properly handle OS errors including "filename too long"
            raise OSError(f"Error accessing parameter file: {str(e)}")

        with tempfile.NamedTemporaryFile('w', suffix='.yml') as tmp_param_file:
            tmp_param_file.write(str(self.param_file))
            tmp_param_file.flush()
            if stream_output:
                # make sure writer is read
                if ExcelWriterProcess._instance is None:
                    raise ArithmeticError("Stream output is ON but writer was not prepared")

            df = get_radiomics_features_from_folder(im_path, seg_path, Path(tmp_param_file.name), *args,
                                                    id_globber=self.id_globber,
                                                    idlist=idlist,
                                                    by_slice=by_slice,
                                                    writer_func=ExcelWriterProcess.write if stream_output else None,
                                                    connected_components=connected_components,
                                                    num_worker=num_workers)

        self._extracted_features = df.T
        return self._extracted_features

    def extract_features_with_norm(self,
                                   im_path: Path,
                                   seg_path: Path,
                                   *args,
                                   norm_state_file: Optional[Path] = None,
                                   param_file: Optional[Path] = None,
                                   by_slice: Optional[int] = -1,
                                   **kwargs) -> pd.DataFrame:
        r"""
        This function normalize the image and segmentation and put them into a temporary folder, then call
        `extract_features()` to extract the radiomics feature.

        Args:
            norm_state_file:
                If specified, overrides `self.saved_state['norm_state_file'] option.
            param_file:
                If specified, use this as the params for pyradiomics feature extraction and override the
                `self.saved_state['param_file']` option.
            by_slice:
                See :func:`get_radiomics_features`

        Returns:
            pd.DataFrame
                The rows

        """
        with tempfile.TemporaryDirectory() as temp_root_dir:
            f = tempfile.NamedTemporaryFile('w', suffix='.yaml', dir=temp_root_dir)
            temp_dir_im = tempfile.TemporaryDirectory(prefix=temp_root_dir + os.sep).name
            temp_dir_seg = [tempfile.TemporaryDirectory(prefix=temp_root_dir + os.sep).name for i in range(1 + len(args))]
            if args is not None and len(args) > 0:
                # self._logger.warning("The function extract_features_with_norm() does not support multiple segmentation "
                #                      "yet! Ignoring additional segmentation input. ")
                self._logger.info(f"Receive multiple segmentations: {len(args) + 1}")
                seg_path = [seg_path] + args
            else:
                seg_path = [seg_path]

            self._logger.info(f"Created temp directory: ({temp_dir_im}, {temp_dir_seg})")

            # The graph yml must exist, use default if not properly specified. Once load, put text into saved_state
            graph_yml = self.load_norm_graph(self.saved_state['norm_graph'])
            f.write(graph_yml)
            f.flush()
            G = MNTSFilterGraph.CreateGraphFromYAML(f.name)
            self._logger.info(f"Using normalization graph:\n{G}")


            # Override instance attribute if specified in arguments
            norm_state = self.saved_state['norm_state_file'] if norm_state_file is None else norm_state_file
            norm_state, norm_state_temp_dir = self.load_norm_state(norm_state)
            # Raise error if state continues to be nothing
            if norm_state is None:
                raise FileNotFoundError(f"Cannot open normaliation graph state!")

            # normalize the images
            command = f"--input {im_path} --state-dir {str(norm_state)} --output {temp_dir_im} --file {f.name} --verbose " \
                      f"-n 12".split(' ')
            self._logger.debug(f"Command for image normalization: {command}")
            self._logger.info("Start normalization...")
            _start_time = time.time()
            run_graph_inference(command)

            # normalize the segmentation for spatial transforms
            # For each segmentation input run once
            for _seg_path, _temp_seg_out in zip(seg_path, temp_dir_seg):
                command = f"--input {_seg_path} --state-dir {str(norm_state)} --output {str(_temp_seg_out)} --file {f.name} --verbose " \
                          f"-n 12 --force-segment".split(' ')
                self._logger.debug(f"Command for segment normalization: {command}")
                run_graph_inference(command)
            _end_time = time.time()
            self._logger.info(f"Finished normalization, time took: {_end_time - _start_time:.01f}s")
            # Get the name of the last output nodes !!! Only working on last output !!!
            ext_node_name = G.nodes[G._exits[-1]]['filter'].get_name()

            # run py radiomics
            if param_file is None:
                param_file = self.saved_state['param_file']
                if param_file is None:
                    raise AttributeError("Param file is not properly specified!")
            elif isinstance(param_file, Path):
                self.saved_state['param_file'] = param_file.read_text()
            elif isinstance(param_file, str):
                self.saved_state['param_file'] = param_file
                _f = tempfile.NamedTemporaryFile('w')
                _f.write(param_file)
                _f.flush()

            _start_time = time.time()
            self._logger.info("Start running pyradiomics...")
            df = self.extract_features(Path(temp_dir_im).joinpath(ext_node_name),
                                       Path(temp_dir_seg[0]).joinpath(ext_node_name),
                                       *[Path(_p).joinpath(ext_node_name) for _p in temp_dir_seg[1:]],
                                       param_file=param_file,
                                       *kwargs)
            _end_time = time.time()
            self._logger.info(f"Finished pyradiomics, time took: {_end_time - _start_time:.01f}s")

            # Cleanup
            try:
                _f.close()
                norm_state_temp_dir.cleanup() # Should have write an escape but I am lazy
            except:
                pass
        return df

    def load_norm_state(self, norm_state):
        norm_state_temp_dir = None
        if isinstance(norm_state, bytes):  # if it is a bytestring
            # unzip to a temp directory
            bio = BytesIO()
            bio.write(norm_state)
            bio.seek(0)
            with ZipFile(bio, 'r', ZIP_DEFLATED) as zf:
                self._logger.info(f"Unzipping norm_state_files...")
                norm_state_temp_dir = tempfile.TemporaryDirectory()
                zf.extractall(norm_state_temp_dir.name)
                norm_state = norm_state_temp_dir.name
                self._logger.info(f"Unzipped sucessful: {pprint.pformat(list(Path(norm_state).iterdir()))}")
            bio.close()
        elif isinstance(norm_state, (str, Path)):
            if Path(norm_state).is_file():
                self._logger.warning(f"{str(norm_state)} is a file, `norm_state_file` should be a directory "
                                     f"instead. Using the parent directory {str(norm_state.parent)} instead.")
                norm_state = norm_state.parent
            # store the whole directory as a byte string in saved state
            bio = BytesIO()
            zf = ZipFile(bio, 'w', ZIP_DEFLATED)
            zipdir(str(norm_state), zf)
            zf.close()
            bio.seek(0)
            bs = b''.join(bio.readlines())
            self.saved_state['norm_state_file'] = bs
            self._logger.info(f"Capturing norm_state_file as byte string, zipping {norm_state}")
        return norm_state, norm_state_temp_dir

    def load_norm_graph(self, graph: Union[str, Path]) -> str:
        r"""
        Return
        """
        if graph is None:
            graph_yml = normalization_yaml # Load from global scope variable specified at top of this file
            self.saved_state['norm_graph'] = graph_yml # save it into saved state
        elif isinstance(graph, (str, Path)):
            graph = str(graph)
            if graph.find('.yml') > -1:  # If its a .yml file path
                # Load the file as string
                self.saved_state['norm_graph'] = ''.join(open(graph, 'r').readlines())
            else:  # Other wise its a long string, load as is
                self.saved_state['norm_graph'] = graph
            graph_yml = self.saved_state['norm_graph']
        return graph_yml

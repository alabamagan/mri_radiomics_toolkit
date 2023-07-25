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
from typing import Iterable, Optional, Sequence, Union
from zipfile import ZIP_DEFLATED, ZipFile

import SimpleITK as sitk
import joblib
import numpy as np
import pandas as pd
from mnts.filters import MNTSFilterGraph
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.normalization import run_graph_inference
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from radiomics import featureextractor
from .utils import zipdir, compress, decompress, is_compressed, ExcelWriterProcess


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

def get_radiomics_features(fn: Path,
                           mn: Path,
                           param_file: Path,
                           *args,
                           id_globber: str = "^[0-9a-zA-Z]+",
                           by_slice: int = -1,
                           connected_components = False,
                           writer: ExcelWriterProcess = None) -> pd.DataFrame:
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
    Returns:
        pd.DataFrame
            Dataframe of featurse. Rows are features and columns are data points.
    """
    assert fn.is_file() and mn.is_file(), f"Cannot open images or mask at: {fn} and {mn}"

    try:
        logger = MNTSLogger['radiomics_features']
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
            if not all(np.isclose(im.GetSpacing(), _msk.GetSpacing(), atol=1E-4)):
                logger.warning(f"Detected differences in spacing! Resampling "
                                                         f"{_msk.GetSpacing()} -> {im.GetSpacing()}...")
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
                    slice_cols = []
                    case_number = re.search(id_globber, fn.name).group()
                    for j in range(_binmsk.GetSize()[by_slice]):
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
        logger.debug(f"Finsihed extracting features, time took: {_end_time - _start_time}s")

        # If writer is provided, write it
        if writer is not None:
            writer.write

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

        *args:
            If this exist, arguments will be attached to `mask_dir`, additional features will be extracted
            if there are more than one masks

    Return:
        pd.DataFrame
    """
    logger = MNTSLogger['radiomics_features']
    ids = get_unique_IDs(list([str(i.name) for i in im_dir.iterdir()]), id_globber)
    if not idlist is None:
        overlap = set.intersection(set(ids), set(idlist))
        missing_idlist = set(idlist) - set(ids)
        missing_files = set(ids) - set(idlist)
        if len(missing_idlist):
            logger.warning(f"Some of the specified ids cannot be founded: \n {missing_idlist}")
        if len(missing_files):
            logger.info(f"IDs filtered away: {missing_files}")
        ids = overlap

    # Load the pairs
    if not args is None:
        mask_dir = [seg_dir] + list(args)
    else:
        mask_dir = [seg_dir]

    mask = []
    for i, msk in enumerate(mask_dir):
        _source, _mask = load_supervised_pair_by_IDs(str(im_dir), str(msk), ids,
                                                    globber=id_globber, return_pairs=False)
        mask.append(_mask)
        if i == 0:
            source = _source
        else:
            if not len(source) == len(_source):
                raise IndexError(f"Length of the inputs are incorrect, somethings is wrong with index globbing for "
                                 f"{str(msk)}.")

    source = [im_dir.joinpath(s) for s in source]
    mask = [[mask_dir[i].joinpath(s) for s in mask[i]] for i in range(len(mask))]
    func = partial(get_radiomics_features, id_globber=id_globber, **kwargs)

    r"""
    Multi-thread
    """
    if len(mask) == 1:
        z = repeat_zip(source, mask[0], [param_file])
    else:
        z = repeat_zip(source, mask[0], [param_file], *mask[1:])
    pool = mpi.Pool(mpi.cpu_count())
    res = pool.starmap_async(func, z)
    pool.close()
    pool.join()
    res = res.get()
    df = pd.concat(res, axis=1)
    new_index = [o.split('_') for o in df.index]
    new_index = pd.MultiIndex.from_tuples(new_index, names=('Pre-processing', 'Feature_Group', 'Feature_Name'))
    df.index = new_index
    return df

class FeatureExtractor(object):
    r"""
    Attributes:
        _extracted_features (pd.DataFrame):
            Features extracted using the parameter file

    Examples:
        >>> fe = FeatureExtractor()

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
        r"""
        Load att `self.save_state`. The file saved should be a dictionary containing key 'selected_features', which
        points to a list of features in the format of pd.MultiIndex or tuple
        """
        assert Path(f).is_file(), f"Cannot open file {f}"
        d = joblib.load(f)
        if not isinstance(d, dict):
            raise TypeError("State loaded is incorrect!")
        self.saved_state.update(d)

    def save(self, f: Path):
        if any([v is None for v in self.saved_state.values()]):
            self._logger.warning("Some saved state components are None.")
        if f.is_dir(): # default name
            f = f.joinpath('saved_state.fe')
        # Always save the compressed version of the param file.
        if not is_compressed(self.saved_state['param_file']):
            self.saved_state['param_file'] = compressed(self.param_file)
        joblib.dump(self.saved_state, filename=f.with_suffix('.fe'))


    def save_features(self, outpath: Path):
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
                         by_slice: Optional[int] = None) -> pd.DataFrame:
        if param_file is None and self.saved_state['param_file'] is None:
            raise ArithmeticError("Please specify param file.")
        if param_file is None:
            param_file = self.saved_state['param_file']
        if idlist is None:
            idlist = self.idlist
        if by_slice is None:
            by_slice = self.by_slice

        # if param_file is not Path and a string, write contents to tempfile
        try:
            if Path(param_file).is_file():
                self.param_file = param_file.open('r').read() # this will read param file to str and store it in
                                                              # saved_state
            elif isinstance(param_file, str):
                # if the param_file is a string, test if its compressed
                if is_compressed(param_file):
                    self.param_file = decompress(param_file)
                else:
                    self.param_file = param_file
            elif param_file is None:
                raise FileNotFoundError(f"Cannot open pyrad param file: {param_file}")
        except OSError: # Prevent returning file name too long error
            pass

        with tempfile.NamedTemporaryFile('w', suffix='.yml') as tmp_param_file:
            tmp_param_file.write(self.param_file)
            tmp_param_file.flush()
            df = get_radiomics_features_from_folder(im_path, seg_path, Path(tmp_param_file.name), *args,
                                                    id_globber=self.id_globber,
                                                    idlist=idlist,
                                                    by_slice=by_slice)

        self._extracted_features = df.T
        if self._extracted_features.index.nlevels > 1:
            self._extracted_features.sort_index(level=0, inplace=True)
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

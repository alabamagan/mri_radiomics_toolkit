import multiprocessing as mpi
import re
import os
import tempfile
import zipfile
import pprint
from pathlib import Path
from typing import Optional, Union, Sequence, Iterable
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
from functools import partial

import SimpleITK as sitk
import numpy as np
import pandas as pd
import joblib
import torchio as tio
import torch

from mnts.scripts.normalization import run_graph_inference
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from mnts.filters import MNTSFilterGraph
from radiomics import featureextractor

# Fix logger
global logger

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
                           augmentor: tio.Compose = None) -> pd.DataFrame:
    r"""
    Return the features computed by pyramdiomics in a `pd.DataFrame` structure.

    Args:
        fn (Path):
            Image file directory (needs to be .nii.gz).
        mn (Path):
            Segmentation file directory.
        param_file (Path):
            Path to the pyradiomics setting.
        *args (Sequence of Path):
            If this is provided, they directory is treated as extra segmentation
    Returns:
        pd.DataFrame
    """
    assert fn.is_file() and mn.is_file(), f"Cannot open images or mask at: {fn} and {mn}"

    try:
        logger = MNTSLogger['radiomics_features']
        im = sitk.ReadImage(str(fn))
        msk = sitk.ReadImage(str(mn))
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
                filt.SetReferenceImage(im)
                msk[i] = filt.Execute(_msk)

            if not (np.isfinite(sitk.GetArrayFromImage(im)).all() and np.isfinite(sitk.GetArrayFromImage(_msk)).all()):
                logger.warning(f"Detected NAN in image {str(fn    )}")

        # If an augmentor is given, do augmentation first
        if not augmentor is None:
            # Reseed becaus multi-processing thread might fork the same random state
            np.random.seed()
            torch.random.seed()

            logger.info(f"Augmentor was given: {augmentor}")
            _ = {f'mask_{i}': tio.LabelMap.from_sitk(_msk) for i, _msk in enumerate(msk)}
            subject = tio.Subject(image=tio.ScalarImage.from_sitk(im), **_)
            logger.debug(f"Original subject: {subject}")
            subject = augmentor.apply_transform(subject)
            im = subject['image'].as_sitk()
            msk = [subject[f'mask_{i}'].as_sitk() for i in range(len(msk))]

        feature_extractor = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))
        dfs = []
        for i, _msk in enumerate(msk):
            X = feature_extractor.execute(im, sitk.Cast(_msk, sitk.sitkUInt8))
            df = pd.DataFrame.from_dict({k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                                         for k, v in X.items()}, orient='index')
            df.columns = [f'{re.search(id_globber, fn.name).group()}']
            if len(msk) > 1:
                df.columns = pd.MultiIndex.from_product([['Segment_' + chr(ord('A') + i)], df.columns])
            dfs.append(df)
        if len(dfs) == 1:
            out = dfs[0]
        else:
            out = pd.concat(dfs, axis=1)
        logger.debug(f"Intermediate result:\n {out}")
        return out

    except Exception as e:
        MNTSLogger['radiomics_features'].error("Error during get_radiomics_features!")
        MNTSLogger['radiomics_features'].exception(e)

def get_radiomics_features_from_folder(im_dir: Path,
                                       seg_dir: Path,
                                       param_file: Path,
                                       *args,
                                       id_globber: str = "^[0-9a-zA-Z]+",
                                       idlist: Iterable[str] = None,
                                       augmentor: tio.Compose = None) -> pd.DataFrame:
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
    if not augmentor is None:
        func = partial(get_radiomics_features, id_globber=id_globber, augmentor=augmentor)
    else:
        func = get_radiomics_features

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
    def __init__(self, *, id_globber = "^[0-9a-zA-Z]+", idlist = None, param_file = None, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.saved_state = {
            'param_file': None, # Path to param file
            'norm_state_file': Path('../assets/t2wfs/'),    # Override in `extract_features_with_norm` if specified
            'norm_graph': None,
        }
        self.idlist = idlist
        self.param_file = param_file
        self.id_globber = id_globber
        self._extracted_features = None
        self._logger = MNTSLogger[__class__.__name__]

    @property
    def param_file(self):
        return self.saved_state['param_file']

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
        joblib.dump(self.saved_state, filename=f.with_suffix('.fe'))

    def save_features(self, outpath: Path):
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
                         id_globber: Optional[str] = "^[0-9a-zA-Z]+",
                         idlist: Optional[Iterable[str]] = None,
                         param_file: Optional[Path] = None,
                         augmentor: Optional[Union[tio.Compose, Path]] = None) -> pd.DataFrame:
        if param_file is None and self.saved_state['param_file'] is None:
            raise ArithmeticError("Please specify param file.")
        if param_file is None:
            param_file = self.saved_state['param_file']
        if idlist is None:
            idlist = self.idlist

        # if param_file is not Path and a string, write contents to tempfile
        try:
            if Path(param_file).is_file():
                self.param_file = param_file # this will read param file to str and store it in saved_state
        except OSError: # Prevent returning file name too long error
            pass

        with tempfile.NamedTemporaryFile('w', suffix='.yml') as tmp_param_file:
            tmp_param_file.write(self.param_file)
            tmp_param_file.flush()
            df = get_radiomics_features_from_folder(im_path, seg_path, Path(tmp_param_file.name), *args,
                                                    id_globber=self.id_globber,
                                                    augmentor=augmentor, idlist=idlist)

        self._extracted_features = df.T
        if self._extracted_features.index.nlevels > 1:
            self._extracted_features.sort_index(level=0, inplace=True)
        return self._extracted_features

    def extract_features_with_norm(self,
                                   im_path: Path,
                                   seg_path: Path,
                                   *args,
                                   norm_state_file: Optional[Path] = None,
                                   param_file: Optional[Path] = None) -> pd.DataFrame:
        r"""
        This function normalize the image and segmentation and put them into a temporary folder, then call
        `extract_features()` to extract the radiomics feature.

        Args:
            norm_state_file:
                If specified, overrides `self.saved_state['norm_state_file'] option.
            param_file:
                If specified, use this as the params for pyradiomics feature extraction and override the
                `self.saved_state['param_file']` option.

        """
        #TODO: make this work for multiple segmentations
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
            run_graph_inference(command)

            # normalize the segmentation for spatial transforms
            # For each segmentation input run once
            for _seg_path, _temp_seg_out in zip(seg_path, temp_dir_seg):
                command = f"--input {_seg_path} --state-dir {str(norm_state)} --output {str(_temp_seg_out)} --file {f.name} --verbose " \
                          f"-n 12 --force-segment".split(' ')
                self._logger.debug(f"Command for segment normalization: {command}")
                run_graph_inference(command)

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

            df = self.extract_features(Path(temp_dir_im).joinpath(ext_node_name),
                                       Path(temp_dir_seg[0]).joinpath(ext_node_name),
                                       *[Path(_p).joinpath(ext_node_name) for _p in temp_dir_seg[1:]],
                                       param_file=param_file)

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


def zipdir(path, ziph):
    r"""
    Helper function to zip the norm state directory
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(root, file).replace(path, ''))
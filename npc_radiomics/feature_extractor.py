import multiprocessing as mpi
import re
import os
import tempfile
import zipfile
import pprint
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO

import SimpleITK as sitk
import numpy as np
import pandas as pd
import joblib
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
                           id_globber ="^[0-9a-zA-Z]+") -> pd.DataFrame:
    r"""
    Return the featuers computed by pyramdiomics in a `pd.DataFrame` structure.

    Args:
        fn (Path):
            Image file directory (needs to be .nii.gz).
        mn (Path):
            Segmentation file directory.
        param_file (Path):
            Path to the pyradiomics setting.
    Returns:
        pd.DataFrame
    """
    assert fn.is_file() and mn.is_file(), f"Cannot open images or mask at: {fn} and {mn}"

    try:
        im = sitk.ReadImage(str(fn))
        msk = sitk.ReadImage(str(mn))
        # check if they have same spacing
        if not all(np.isclose(im.GetSpacing(), msk.GetSpacing(), atol=1E-4)):
            MNTSLogger['radiomics_features'].warning(f"Detected differences in spacing! Resampling "
                                                     f"{im.GetSpacing()} -> {msk.GetSpacing()}...")
            filt = sitk.ResampleImageFilter()
            filt.SetReferenceImage(im)
            msk = filt.Execute(msk)
        # Check if image has nan
        if not (np.isfinite(sitk.GetArrayFromImage(im)).all() and np.isfinite(sitk.GetArrayFromImage(msk)).all()):
            MNTSLogger['radiomics_features'].warning(f"Detected NAN in image {str(fn    )}")


        feature_extractor = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))

        features = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))
        X = feature_extractor.execute(im, sitk.Cast(msk, sitk.sitkUInt8))
        df = pd.DataFrame.from_dict({k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                                     for k, v in X.items()}, orient='index')
        df.columns = [f'{re.search(id_globber, fn.name).group()}']
        return df
    except Exception as e:
        MNTSLogger['radiomics_features'].error("Error during get_radiomics_features!")
        MNTSLogger['radiomics_features'].exception(e)

def get_radiomics_features_from_folder(im_dir: Path,
                                       seg_dir: Path,
                                       param_file: Path,
                                       id_globber ="^[0-9a-zA-Z]+") -> pd.DataFrame:
    r"""
    This pairs up the image and the segmentation files using the global regex globber

    Args:
        im_dir (Path):
            Folder that contains the image files
        seg_dir (Path):
            Folder that contians the segmentation files
        param_file (Path):
            Path to the pyradiomics setting.

    Return:
        pd.DataFrame
    """
    ids = get_unique_IDs(list([str(i.name) for i in im_dir.iterdir()]), id_globber)

    # Load the pairs
    source, mask = load_supervised_pair_by_IDs(str(im_dir), str(seg_dir), ids,
                                               globber=id_globber, return_pairs=False)
    source = [im_dir.joinpath(s) for s in source]
    mask = [seg_dir.joinpath(s) for s in mask]

    r"""
    Multi-thread
    """
    z = repeat_zip(source, mask, [param_file])
    pool = mpi.Pool(mpi.cpu_count())
    res = pool.starmap_async(get_radiomics_features, z)
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
    def __init__(self, id_globber = "^[0-9a-zA-Z]+", param_file=None):
        super(FeatureExtractor, self).__init__()
        self.saved_state = {
            'param_file': None, # Path to param file
            'norm_state_file': Path('../assets/t2wfs/'),    # Override in `extract_features_with_norm` if specified
            'norm_graph': None,
        }
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

    def extract_features(self, im_path: Path, seg_path: Path, param_file: Optional[Path] = None) -> pd.DataFrame:
        if param_file is None and self.saved_state['param_file'] is None:
            raise ArithmeticError("Please specify param file.")
        if param_file is None:
            param_file = self.saved_state['param_file']
        df = get_radiomics_features_from_folder(im_path, seg_path, param_file, id_globber=self.id_globber)
        self.saved_state['param_file'] = param_file
        self._extracted_features = df.T
        return self._extracted_features

    def extract_features_with_norm(self,
                                   im_path: Path,
                                   seg_path: Path,
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
        with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f, \
             tempfile.TemporaryDirectory() as temp_dir_im, \
             tempfile.TemporaryDirectory() as temp_dir_seg:
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
            command = f"--input {seg_path} --state-dir {str(norm_state)} --output {temp_dir_seg} --file {f.name} --verbose " \
                      f"-n 12 --force-segment".split(' ')
            self._logger.debug(f"Command for segment normalization: {command}")
            run_graph_inference(command)

            # Get the name of the last output nodes
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
                                       Path(temp_dir_seg).joinpath(ext_node_name),
                                       param_file)

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
import re, os
import radiomics
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
from pathlib import Path
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from pprint import *

import multiprocessing as mpi
import argparse

# Fix logger
global logger
global globber

def get_radiomics_features(fn: Path,
                           mn: Path,
                           param_file: Path) -> pd.DataFrame:
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


    im = sitk.ReadImage(str(fn))
    msk = sitk.ReadImage(str(mn))
    feature_extractor = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))

    features = featureextractor.RadiomicsFeatureExtractor(str(param_file.resolve()))
    X = feature_extractor.execute(im, sitk.Cast(msk, sitk.sitkUInt8))
    df = pd.DataFrame.from_dict({k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                                 for k, v in X.items()}, orient='index')
    df.columns = [f'{re.search(globber, fn.name).group()}']
    return df

def get_radiomics_features_from_folder(im_dir: Path,
                                       seg_dir: Path,
                                       param_file: Path) -> pd.DataFrame:
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
    ids = get_unique_IDs(list([str(i.name) for i in im_dir.iterdir()]), globber)

    # Load the pairs
    source, mask = load_supervised_pair_by_IDs(str(im_dir), str(seg_dir), ids,
                                               globber=globber, return_pairs=False)
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
    new_index = pd.MultiIndex.from_tuples(new_index)
    df.index = new_index
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-dir', action='store', type=Path,
                        help="Directory to image files.")
    parser.add_argument('-s', '--seg-dir', action='store', type=Path,
                        help='Directory to segmentation files.')
    parser.add_argument('-p', '--param-file', action='store', type=Path, default=Path('./pyradiomics_setting.yml'),
                        help='Path to the pyradiomics settings.')
    parser.add_argument('-g', '--id-globber', action='store', type=str, default=r"^(NPC|T1rhoNPC|K|P|RHO)?[0-9]{2,4}",
                        help='Regex ID globber for pairing images and segmentation.')
    parser.add_argument('-o', '--output', action='store', type=Path,
                        help='Where to output the computed features as excel.')
    parser.add_argument('-v' '--verbose', action='store_true',
                        help='Verbosity option.')
    parser.add_argument('--keep-log', action='store_true',
                        help='If true, the log file is saved to "pyradiomics.log"')
    args = parser.parse_args()

    global logger
    global globber

    im_path, seg_path, param_file = [Path(getattr(args, p)) for p in ['img_dir', 'seg_dir', 'param_file']]
    for p in [im_path, seg_path]:
        if not p.is_dir():
            raise IOError(f"Cannot open file: {str(p.resolve())}")
    if not param_file.is_file():
        raise IOError(f"Cannot open param file: {str(param_file.resolve())}")

    # Prepare logger config
    i = 0
    basename = 'pyradiomics'
    log_file = Path('Log')
    while log_file.with_suffix('.log').exists():
        i += 1
        log_file = log_file.joinpath(f'{basename}-{i}')
    logger = MNTSLogger(log_dir=str(log_file.with_suffix('.log')), logger_name='pyradiomics', verbose=True)
    radiomics.logger = logger


    globber = args.id_globber

    # Run main code
    outpath = Path(args.output)
    if not outpath.parent.is_dir():
        outpath.parent.mkdir(exist_ok=False)
    if outpath.is_dir():
        outpath = outpath.join('extracted_features.xlsx')

    df = get_radiomics_features_from_folder(im_path, seg_path, param_file)
    if outpath.suffix == '.xlsx' or outpath.suffix is None:
        df.to_excel(str(outpath.with_suffix('.xlsx').resolve()))
    elif outpath.suffix == '.csv':
        df.to_csv(str(outpath.resolve()))
    else:
        raise IOError(f"Inccorect suffix, only '.csv' or '.xlsx' accepted, got {outpath.suffix} instead.")

    # Remove log file if not needed
    if not args.keep_log:
        del logger
        log_file.with_suffix('.log').unlink()
    return df

if __name__ == '__main__':
    main()
import re, os, sys
import radiomics
import SimpleITK as sitk
import pandas as pd
import logging
import numpy as np
from radiomics import featureextractor
from pathlib import Path
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from pprint import *

from mri_radiomics_toolkit.feature_extractor import FeatureExtractor
import multiprocessing as mpi
import argparse

# Fix logger
global id_globber

# Defaults
default_pyrad_paramfile = Path(__file__).joinpath('../../pyradiomics_setting/pyradiomics_setting-v3.yml')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-dir', action='store', type=Path,
                        help="Directory to image files.")
    parser.add_argument('-s', '--seg-dir', action='store', type=Path,
                        help='Directory to segmentation files.')
    parser.add_argument('-p', '--param-file', action='store', type=Path, default=default_pyrad_paramfile,
                        help='Path to the pyradiomics settings.')
    parser.add_argument('-f', '--with-norm', action='store', type=Path, default=None,
                        help="If specified path to a normalization state-dir, extract_feature_with_norm will be called.")
    parser.add_argument('-g', '--id-globber', action='store', type=str, default=r"^(NPC|T1rhoNPC|K|P|RHO)?[0-9]{2,4}",
                        help='Regex ID globber for pairing images and segmentation.')
    parser.add_argument('-o', '--output', action='store', type=Path,
                        help='Where to output the computed features as excel.')
    parser.add_argument('-v' '--verbose', action='store_true',
                        help='Verbosity option.')
    parser.add_argument('--id-list', action='store', default=None,
                        help='If specificied pass this to feature extractor.')
    parser.add_argument('--keep-log', action='store_true',
                        help='If true, the log file is saved to "pyradiomics.log"')
    args = parser.parse_args()

    try:
        img_path, seg_path, param_file = [Path(getattr(args, p)).resolve()
                                          for p in ['img_dir', 'seg_dir', 'param_file']]
        for p in [img_path, seg_path]:
            if not p.is_dir():
                raise IOError(f"Cannot open file: {str(p.resolve())}")
        if not param_file.resolve().is_file():
            raise IOError(f"Cannot open param file: {str(param_file.resolve())}")

        NORM_FLAG = args.with_norm is not None
        if NORM_FLAG:
            norm_file_path = Path(args.with_norm)
            assert norm_file_path.is_dir(), f"Norm path specified but cannot be openned: {args.with_norm}"
    except Exception as e:
        parser.print_usage()
        raise AttributeError("Error during argument check")

    # Prepare logger config
    i = 0
    basename = 'pyradiomics'
    log_file_base = Path('../Log')
    log_file = log_file_base.joinpath(basename)
    while log_file.with_suffix('.log').exists():
        i += 1
        log_file = log_file_base.joinpath(f'{basename}-{i}')


    with MNTSLogger(log_dir=str(log_file.with_suffix('.log')), log_level='debug',
                    logger_name='pyradiomics', verbose=True, keep_file=args.keep_log) as logger:
        if str(args.id_list).count(os.sep):
            idlist_file = Path(args.id_list)
            if idlist_file.suffix == '.xlsx':
                idlist = [str(r) for r in pd.read_excel(idlist_file, index_col=0).index]
            elif idlist_file.suffix == ".csv":
                idlist = [str(r) for r in pd.read_csv(idlist_file, index_col=0).index]
            else:
                idlist = [r.rstrip() for r in idlist_file.open('r').readlines()]
        elif isinstance(args.id_list, (list, tuple)):
            idlist = [str(x) for x in args.id_list]
        elif args.id_list is None:
            # Matches image and seg files, use glob instead of rglob
            seg_files = seg_path.glob("*nii.gz")
            img_files = img_path.glob("*nii.gz")
            seg_ids = [re.search(args.id_globber, f.name) for f in seg_files]
            seg_ids = [mo.group() for mo in seg_ids if mo is not None]
            img_ids = [re.search(args.id_globber, f.name) for f in img_files]
            img_ids = [mo.group() for mo in img_ids if mo is not None]

            seg_missing_ids = list(set(img_ids) - set(seg_ids))
            img_missing_ids = list(set(seg_ids) - set(img_ids))
            if not len(seg_missing_ids) == 0:
                seg_missing_ids.sort()
                msg = f"IDs missing in segmentation dir: {pformat(','.join(seg_missing_ids), width=80)}"
                logger.warning(msg)
            if not len(seg_missing_ids) == 0:
                img_missing_ids.sort()

                msg = f"IDs missing in image dir: {pformat(','.join(img_missing_ids), width=80)}"
                logger.warning(msg)
            idlist = list(set(img_ids).intersection(set(seg_ids)))
        else:
            idlist = [r for r in args.id_list.split(',')]

        fe = FeatureExtractor(id_globber=args.id_globber, idlist=idlist)

        # Run main code
        outpath = Path(args.output)
        if not outpath.parent.is_dir():
            outpath.parent.mkdir(exist_ok=False)
        if outpath.is_dir():
            outpath = outpath.join('extracted_features.xlsx')

        if NORM_FLAG:
            df = fe.extract_features_with_norm(img_path, seg_path, param_file=param_file, norm_state_file=norm_file_path)
        else:
            df = fe.extract_features(img_path, seg_path, param_file=param_file)

        if outpath.suffix == '.xlsx' or outpath.suffix is None:
            df.to_excel(str(outpath.with_suffix('.xlsx').resolve()))
        elif outpath.suffix == '.csv':
            df.to_csv(str(outpath.resolve()))
        else:
            raise IOError(f"Inccorect suffix, only '.csv' or '.xlsx' accepted, got {outpath.suffix} instead.")


    return df


if __name__ == '__main__':
    main()
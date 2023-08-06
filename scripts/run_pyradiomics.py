import re, os, sys
import radiomics
import SimpleITK as sitk
import pandas as pd
import logging
import numpy as np
import radiomics
from radiomics import featureextractor
from pathlib import Path
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs, load_supervised_pair_by_IDs, repeat_zip
from pprint import *

from mri_radiomics_toolkit.feature_extractor import FeatureExtractor
from mri_radiomics_toolkit.utils import ExcelWriterProcess
import multiprocessing as mpi
import argparse

# Fix logger
global id_globber

# Defaults
default_pyrad_paramfile = Path(__file__).joinpath('../../pyradiomics_setting/pyradiomics_setting-v3.yml')

def take_over_logger(pyrad_logger: logging.Logger):
    r"""Quikc snippet to take over the logger used by pyradiomics"""
    handlers = pyrad_logger.handlers
    for h in handlers:
        pyrad_logger.removeHandler(h)

    # add back MNTS created handlers
    if MNTSLogger.global_logger is None:
        msg = 'Only call this snippet after the global logger was created.'
        raise ArithmeticError(msg)

    mnts_logger = MNTSLogger['pyradiomics']
    mnts_handlers = mnts_logger._logger.handlers
    for f in mnts_handlers:
        pyrad_logger.addHandler(f)


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
    parser.add_argument('-k', '--stream', action='store_true',
                         help="Stream output to an excel file.")
    parser.add_argument('-v' '--verbose', action='store_true',
                        help='Verbosity option.')
    parser.add_argument('--id-list', action='store', default=None,
                        help='If specificied pass this to feature extractor.')
    parser.add_argument('--keep-log', action='store_true',
                        help='If true, the log file is saved to "pyradiomics.log"')
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode.")
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
        # Take over radiomics logger
        rad_logger = radiomics.logger
        radiomics.setVerbosity(10 if args.debug else 40) # DEBUG
        take_over_logger(rad_logger)

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

        # Check if output file already exist
        outpath = Path(args.output)
        if outpath.is_file():
            logger.warning("Found existing rad file. Appending newly calculated features here.")
            if outpath.suffix == '.xlsx' or outpath.suffix is None:
                df_existing = pd.read_excel(str(outpath.with_suffix('.xlsx').resolve()), index_col=[0, 1, 2])
            elif outpath.suffix == '.csv':
                df_existing = pd.read_csv.to_csv(str(outpath.resolve()), index_col=[0, 1, 2])

            # find if there's any ID that has already been calculated. Skip them.
            overlapping_ids = set(df_existing.columns).intersection(set(idlist))
            if not len(overlapping_ids) == 0:
                logger.warning(f"Some of the patients has already been processed. Removing them"
                               f"from the existing idlist.")
                idlist = set(idlist) - overlapping_ids
                idlist = list(idlist)
                logger.info(f"ID removed: {pformat(','.join(idlist))}")


        if args.debug:
            logger.warning("Entering debug mode.")
            idlist = idlist[:5]

        fe = FeatureExtractor(id_globber=args.id_globber, idlist=idlist)

        # Run main code
        if not outpath.parent.is_dir():
            outpath.parent.mkdir(exist_ok=False)
        if outpath.is_dir():
            outpath = outpath.join('extracted_features.xlsx')
        if args.stream:
            stream_writer = ExcelWriterProcess(output_file=str(outpath.with_suffix(".xlsx").resolve()))
            stream_writer.start()

        if NORM_FLAG:
            df = fe.extract_features_with_norm(img_path, seg_path, param_file=param_file, norm_state_file=norm_file_path)
        else:
            df = fe.extract_features(img_path, seg_path, param_file=param_file, stream_output=args.stream)

        if args.stream:
            # Elegantly close writer
            stream_writer.stop()

            # Change outpath to save a copy in case something went wrong
            outpath.with_name(outpath.stem + "_bak" + outpath.suffix)

        if outpath.suffix == '.xlsx' or outpath.suffix is None:
            df.to_excel(str(outpath.with_suffix('.xlsx').resolve()))
        elif outpath.suffix == '.csv':
            df.to_csv(str(outpath.resolve()))
        else:
            # Force saving as xlsx
            df.to_excel(str(outpath.with_suffix('.xlsx').resolve()))
            raise RuntimeWarning(f"Inccorect suffix, only '.csv' or '.xlsx' accepted, got {outpath.suffix} instead."
                                 f"Saving as Excel anyway.")

    return df


if __name__ == '__main__':
    main()
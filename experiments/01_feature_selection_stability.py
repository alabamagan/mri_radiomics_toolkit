import torchio as tio
import SimpleITK as sitk
import time
import re
import pandas as pd
import numpy as np
import sklearn

from pathlib import Path
from typing import Union, Any
from npc_radiomics.controller import Controller
from npc_radiomics.feature_selection import FeatureSelector
from npc_radiomics.feature_extractor import FeatureExtractor
from npc_radiomics.perf_metric import *
from mnts.mnts_logger import MNTSLogger

from tqdm import auto

def build_augmentation() -> tio.Compose:
    r"""This builds the augmentator"""

    out = tio.Compose([
        tio.ToCanonical(),
        tio.RandomAffine(scales=[0.95, 1.05],
                         degrees=10),
        tio.RandomFlip(axes='lr'),
        tio.RandomNoise(mean=0, std=[0, 1])
    ])
    return out

def time_feature_selection():
    startitme = time.time()
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        logger.info("{:=^100s}".format(f" Testing time required for feature selection "))
        p_setting = Path("01_configs/C3_BRENT.yml")
        p_feature_1 = Path("../samples/samples_feat_1st.xlsx")
        p_feature_2 = Path("../samples/samples_feat_2nd.xlsx")
        p_gt = Path("../samples/sample_datasheet.csv")

        feat_set_1 = pd.read_excel(str(p_feature_1), index_col=[0, 1, 2]).T
        feat_set_2 = pd.read_excel(str(p_feature_2), index_col=[0, 1, 2]).T
        gt = pd.read_csv(str(p_gt), index_col=0)
        overlap_index = set(feat_set_1.index) & set(feat_set_2.index) & set(gt.index)
        feat_set_1 = feat_set_1.loc[overlap_index]
        feat_set_2 = feat_set_2.loc[overlap_index]
        gt = gt.loc[overlap_index]

        ctl = Controller(setting=p_setting, with_norm=False)
        ctl.selector.fit(feat_set_1, gt, feat_set_2)

        endtime = time.time()
        time_used = (endtime - startitme)/60 # in minute
        logger.info("{:=^100s}".format(f" Time used {time_used:.02f}min ")) # 165 min on 212, 134 on A6000

def time_augment_feature_extraction():
    r"""
    Estimate time required to extract features with augmentation
    """
    startitme = time.time()
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        logger.info("{:=^130s}".format(f" Testing time required for feature extract with augment "))

        p_setting = Path("01_configs/C3_BRENT.yml")
        p_pyrad_setting = Path('01_configs/pyrad_settings.yml')
        p_img = Path('../NPC_Radiomics/10.Pre-processed-v2/01.NyulNormalized/')
        p_seg_a = Path('../NPC_Radiomics/0B.Segmentation/01.First/')
        p_seg_b = Path('../NPC_Radiomics/0B.Segmentation/02.Second/')

        clt = Controller(setting=p_setting, with_norm=False)
        df = clt.extractor.extract_features(p_img, p_seg_a, p_seg_b, param_file=p_pyrad_setting,
                                            augmentor=build_augmentation())
        df.to_excel('./output/time_augment_feature.xlsx')
        logger.debug(f"Final output: {df.to_string}")
        endtime = time.time()
        time_used = (endtime - startitme)/60 # in minute
        logger.info("{:=^130s}".format(f" Time used {time_used:.02f}min ")) # 24min on 212
    pass

def generate_augmented_features():
    import gc
    num_of_trials = 100
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        p_setting = Path("01_configs/C3_BRENT.yml")
        p_pyrad_setting = Path('01_configs/pyrad_settings.yml')
        logger.warning(f"Random_seed: {np.random.get_state()[1][0]}")
        p_img = Path('../NPC_Radiomics/10.Pre-processed-v2/01.NyulNormalized/')
        p_seg_a = Path('../NPC_Radiomics/0B.Segmentation/01.First/')
        p_seg_b = Path('../NPC_Radiomics/0B.Segmentation/02.Second/')
        for i in auto.trange(num_of_trials):
            logger.warning(f"Random_seed loop {i}: {np.random.get_state()[1][0]}")
            logger.info("{:-^130}".format(f" Trial {i:03d} "))
            transform = build_augmentation()
            ctl = Controller(setting=p_setting, with_norm=False)
            df = ctl.extractor.extract_features(p_img, p_seg_a, p_seg_b, param_file=p_pyrad_setting,
                                                augmentor=transform)
            if df.index.nlevels > 1:
                _ = df.index.levels[0].to_list()
                df = [df.loc[key] for key in _]
            else:
                df = [df]

            for j, dff in enumerate(df):
                out_name = Path(f"./_exclude_output/trial-{i:03d}_feature-{chr(ord('A') + j)}.xlsx")
                if not out_name.parent.is_dir():
                    out_name.parent.mkdir(exist_ok=True)
                logger.info(f"Saving to: {str(out_name)}")
                dff.to_excel(str(out_name))
            del df, ctl, transform
            gc.collect()
        pass

def get_stability(selected_features: Union[Path, str],
                  all_features: Union[Path, str]) -> float:
    with MNTSLogger('./01_feature_selection_stability.py', keep_file=True, verbose=True, log_level='debug') as logger:
        selected_features = Path(selected_features)
        full_feat_list = Path(all_features)

        sel = pd.read_excel(selected_features, index_col=0).fillna('').astype(str)
        feats = pd.read_excel(full_feat_list, index_col=0, header=[0, 1, 2]).T
        # discard diagnosis
        try:
            feats.drop('diagnostics', level=0, axis=0, inplace=True) # Assume columns are features, rows are patients
        except:
            logger.warning("Cannot drop 'diagnostics' column.")
            pass

        feats = [str(s) for s in feats.index.to_list()]
        Z = feat_list_to_binary_mat(sel, feats)
        return getStability(Z)

def validate_feature_selection_stability(clt_setting_path: Path,
                                         out_path: Path):
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        logger.info("{:=^100s}".format(" Validation - feature selection stability "))

        p_setting = Path(clt_setting_path)
        p_gt = Path("../samples/sample_datasheet.csv")
        p_out = Path(out_path)
        logger.info(f"Results are written to {str(p_out)}")

        features_folder = Path('./output/')
        features_excel_regexp = r".*trial-(?P<trial_number>\d+)_feature-(?P<feature_set_code>\w+).xlsx"

        # Build a pd.Table from this
        map_dict = {re.search(features_excel_regexp, str(ff)).groups(): str(ff.resolve()) for ff in features_folder.iterdir()}
        df = pd.Series(map_dict, name='FileNames').to_frame()

        for ids in df.index.levels[0].to_list():
            # Skip if trial already ran
            if p_out.is_file():
                _df = pd.read_excel(str(p_out), index_col=0)
                if f"Trial-{ids}" in _df.columns:
                    logger.warning(f"Find column 'Trial-{ids}', skipping...")
                    continue
            # Augmented
            p_feature_a = Path(df.loc[ids, 'A']['FileNames'])
            p_feature_b = Path(df.loc[ids, 'B']['FileNames'])

            # Orginal features
            p_feature_a = Path('./output/original_feature-A.xlsx')
            p_feature_b = Path('./output/original_feature-B.xlsx')

            gt = pd.read_csv(p_gt, index_col=0)
            df_feat_a = pd.read_excel(str(p_feature_a), index_col=0, header=[0, 1, 2])
            df_feat_b = pd.read_excel(str(p_feature_b), index_col=0, header=[0, 1, 2])
            overlap_index = set(gt.index) & set(df_feat_a.index) & set(df_feat_b.index)
            df_feat_a.index = df_feat_a.index.astype('str') # Make-sure the index are of same datatype
            df_feat_b.index = df_feat_b.index.astype('str')
            gt.index = gt.index.astype('str')
            df_feat_a = df_feat_a.loc[overlap_index]
            df_feat_b = df_feat_b.loc[overlap_index]
            gt = gt.loc[overlap_index]

            # Bootstrap (no replacement)
            boot_index = sklearn.utils.resample(df_feat_a.index, replace=False, n_samples = int(0.8 * len(df_feat_a)))
            logger.info(f"bootstrapped index: {','.join(boot_index)}")
            df_feat_a = df_feat_a.loc[boot_index]
            df_feat_b = df_feat_b.loc[df_feat_a.index]
            gt = gt.loc[df_feat_a.index]
            df_feat_a.sort_index(inplace=True)
            df_feat_b.sort_index(inplace=True)
            gt.sort_index(inplace=True)

            logger.debug(f"\n{df_feat_a}")
            logger.debug(f"\n{df_feat_b}")

            ctl = Controller(setting=p_setting, with_norm=False)
            ctl.selector.fit(df_feat_a, gt, df_feat_b)

            # Save the selected features to the excel file
            s = ctl.selector.selected_features

            # Append if exist
            if p_out.is_file():
                reader = pd.ExcelFile(str(p_out))
                ori_df = reader.parse(reader.sheet_names[0], index_col=0)
                reader.close()
                new_df = ori_df.join(pd.Series([str(i) for i in s],
                                               name=f'Trial-{ids}'), how='outer')
                new_df.fillna("")
                new_df.to_excel(str(p_out))
            else:
                pd.Series([str(i) for i in s],
                          name=f'Trial-{ids}').to_frame().to_excel(str(p_out))

def stability_summary():
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:

        # for p in Path('./01_configs/').glob("C*yml"):
        #     validate_feature_selection_stability(p, str(p.with_suffix('_exclude_.xlsx')))
        # validate_feature_selection_stability(Path('./01_configs/C1_elastic_net_only.yml'),
        #                                      '_exclude_C1_elastic_net_only.xlsx')
        validate_feature_selection_stability(Path('./01_configs/C2_RENT.yml'),
                                             '_exclude_C2_RENT.xlsx')
        # validate_feature_selection_stability(Path('./01_configs/C3_BRENT.yml'),
        #                                      '_exclude_C3_BRENT.xlsx')

        # p_feat_list = Path("./output/trial-000_feature-A.xlsx")
        # select_features = Path('./').glob("*xlsx")
        #
        # for i in select_features:
        #     s = get_stability(i, p_feat_list)
        #     print(f"{i}: {s}")

if __name__ == '__main__':
    # generate_augmented_features()
    stability_summary()
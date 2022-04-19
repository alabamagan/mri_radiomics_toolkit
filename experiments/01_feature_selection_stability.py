r"""
This file contains the experiments ran to evaluate the feature selection stability of various configuration.
This study proposed to use boosting in combination with bagging to improve the algorithm RENT. The configurations
used are as follow:
  1. Ordinary elastic net
  2. RENT
  3. boosted RENT
  4. bagged RENT
  5. bagged-boosted RENT
"""

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


def time_feature_selection():
    startitme = time.time()
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        logger.info("{:=^100s}".format(f" Testing time required for feature selection "))
        p_setting = Path("01_configs/C3_BBRENT.yml")
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


def get_stability(selected_features: Union[Path, str],
                  all_features: Union[Path, str]) -> [np.ndarray, pd.Series]:
    r"""
    This function convert the features selected in each trial into a one-hot sparse matrix with rows as trials
    and columns as features. The Nogeauria
    """
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
        nog_score = getStability(Z)
        jac, jac_mean, jac_sd = jaccard_mean(Z)

        return Z, jac, pd.Series([nog_score, jac_mean, jac_sd], index=['Nogueira', 'JAC Mean', 'JAC SD'])

def validate_feature_selection_stability(clt_setting_path: Path,
                                         out_path: Path):
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        logger.info("{:=^100s}".format(" Validation - feature selection stability "))

        p_setting = Path(clt_setting_path)
        p_gt = Path("../samples/sample_datasheet.csv")
        p_out = Path(out_path)
        logger.info(f"Results are written to {str(p_out)}")

        # for ids in df.index.levels[0].to_list():
        for ids in range(100):
            # Skip if trial already ran
            if p_out.is_file():
                _df = pd.read_excel(str(p_out), index_col=0)
                if f"Trial-{ids}" in _df.columns:
                    logger.warning(f"Find column 'Trial-{ids}', skipping...")
                    continue

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

            # Resample (no replacement)
            np.random.seed() # reset the random seed
            boot_index = sklearn.utils.resample(df_feat_a.index, replace=False, n_samples = int(0.7 * len(df_feat_a)))
            logger.info(f"bootstrapped index: {','.join(boot_index)}")
            df_feat_a = df_feat_a.loc[boot_index]
            df_feat_b = df_feat_b.loc[df_feat_a.index]
            gt = gt.loc[df_feat_a.index]
            df_feat_a.sort_index(inplace=True)
            df_feat_b.sort_index(inplace=True)
            gt.sort_index(inplace=True)

            logger.debug(f"\n{df_feat_a}")
            logger.debug(f"\n{df_feat_b}")
            logger.debug(f"\n{gt}")

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

def feature_extraction():
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:
        p_im = Path("../NPC_Radiomics/10.Pre-processed-v2/01.NyulNormalized")
        p_seg_A = Path("../NPC_Radiomics/0B.Segmentation/01.First/")
        p_seg_B = Path("../NPC_Radiomics/0B.Segmentation/02.Second/")
        p_setting = Path("../pyradiomics_setting/pyradiomics_setting-v3.yml")

        extractor = FeatureExtractor(id_globber="^(NPC|T1rhoNPC|K|P|RHO)?[0-9]{2,4}")
        df = extractor.extract_features(p_im, p_seg_A, p_seg_B, param_file=p_setting)
        df['Segment_A'].to_excel("../NPC_Radiomics/10.Pre-processed-v2/features-v3repeat.xlsx")

def stability_summary():
    with MNTSLogger('./01_feature_selection_stability.log', keep_file=True, verbose=True, log_level='debug') as logger:

        # validate_feature_selection_stability(Path('./01_configs/C1_elastic_net_only.yml'),
        #                                      '_exclude_C1_elastic_net_only.xlsx')
        # validate_feature_selection_stability(Path('./01_configs/C2_RENT.yml'),
        #                                      '_exclude_C2_RENT.xlsx')
        # validate_feature_selection_stability(Path('./01_configs/C3_BBRENT.yml'),
        #                                      '_exclude_C3_BRENT.xlsx')
        # validate_feature_selection_stability(Path('./01_configs/C4_BoostingRENT.yml'),
        #                                      '_exclude_C4_BoostingRENT.xlsx')
        # validate_feature_selection_stability(Path('./01_configs/C5_BaggingRENT.yml'),
        #                                       '_exclude_C5_BootstrappingRENT.xlsx')
        p_feat_list = Path("./output/trial-000_feature-A.xlsx")
        select_features = Path('./').glob("_exclude_C*xlsx")

        p_out = Path('./01_feature_selection_stability.xlsx')
        writer = pd.ExcelWriter(p_out)
        Z = {}
        df = []
        jac_mat = []
        for i in select_features:
            name = str(i).replace('.xlsx', '')
            z, jac, s = get_stability(i, p_feat_list)
            s.name = str(name)
            df.append(s)
            Z[name] = z
            jac_mat.append(pd.Series(jac, name=name))
        # Compute hypothesis test
        ttest_df = []
        for i in Z:
            row = pd.Series([], name=i)
            for j in Z:
                if i == j:
                    continue
                zi = Z[i]
                zj = Z[j]
                t = hypothesisTestT(zi, zj, 0.05)
                row[j] = t['p-value']
            ttest_df.append(row)

        df = pd.concat(df, axis=1)
        ttest_df = pd.concat(ttest_df, axis=1).fillna('-')
        ttest_df.sort_index(inplace=True)
        ttest_df.sort_index(1, inplace=True)
        jac_df = pd.concat(jac_mat, axis=1)
        df.to_excel(writer, sheet_name="Stability Score")
        jac_df.to_excel(writer, sheet_name="JAC")
        ttest_df.to_excel(writer, sheet_name="Hypothesis Test")
        writer.save()
        writer.close()
        print(df.to_string())
        print(ttest_df.to_string())

if __name__ == '__main__':
    # generate_augmented_features()
    # stability_summary()
    feature_extraction()
"""
Purpose of this script is to study the feature selection stability of BRENT
"""

import pandas as pd

from pathlib import Path

from mnts.mnts_logger import MNTSLogger

import joblib
from sklearn import *

import numpy as np

from mri_radiomics_toolkit.controller import Controller


def main():
    with MNTSLogger("./02_pipeline_performance.log", logger_name="main", log_level='debug', keep_file=True) as logger:
        # Load features & Config
        #-----------------------
        p_feature_a = Path('./output/original_feature-A.xlsx')
        p_feature_b = Path('./output/original_feature-B.xlsx')
        features_a = pd.read_excel(str(p_feature_a), header=(0, 1, 2), index_col=0)
        features_b = pd.read_excel(str(p_feature_b), header=(0, 1, 2), index_col=0)
        p_ctl_setting = Path('./02_configs/BRENT.yml')
        p_pyrad_setting = Path('./02_configs/pyrad_settings.yml')

        # Load target
        #------------
        status = Path("./output/gt-datasheet.csv")
        status = pd.read_csv(str(status), index_col=0)
        status.columns = ['Status']

        # match target and feature, use the datasheet
        features_a = features_a.loc[status.index]
        features_b = features_b.loc[status.index]

        # K-fold options
        n_fold = 5
        splitter = model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True)
        splits = splitter.split(status.index, status[status.columns[0]])

        # # Load splits if exist
        # p_existing_split = Path("../output/20220324/predict_tables.xlsx")
        # if p_existing_split.is_file():
        #     excel_file = pd.ExcelFile(p_existing_split)
        #     excel_dfs = {i: pd.read_excel(excel_file, index_col=0, sheet_name=i) for i in excel_file.sheet_names}
        #     all_ids = list(status.index)
        #     splits = []
        #     for i in excel_dfs:
        #         logger.info(f"{i}")
        #         test_index = [all_ids.index(case_id) for case_id in excel_dfs[i].index]
        #         train_index = [all_ids.index(ti) for ti in all_ids if not ti in excel_dfs[i].index]
        #         splits.append((train_index, test_index))
        #         logger.info(f"{excel_dfs[i].index}")

        # Storage variables
        fold_freq = {}
        performance = []
        features_rate_table = []
        selected_features_list = []
        output_root = Path('../output/20220412(v2)')
        output_root.mkdir(exist_ok=True)
        summary_file = output_root.joinpath('output_summary.xlsx')
        excel_writter = pd.ExcelWriter(str(summary_file))
        pdt_writter = pd.ExcelWriter(str(output_root.joinpath('predict_tables.xlsx')))

        # Training K-Fold
        #----------------
        for fold, (train_index, test_index) in enumerate(splits):
            train_ids, test_ids = [str(status.index[i]) for i in train_index], \
                                  [str(status.index[i]) for i in test_index]
            train_ids.sort()
            test_ids.sort()

            # Create controller
            controller = Controller(setting=str(p_ctl_setting), param_file=str(p_pyrad_setting))

            # Seperate traing and test features
            train_feat_a = features_a.loc[train_ids]
            train_feat_b = features_b.loc[train_ids]
            test_feat_a = features_a.loc[test_ids]
            test_feat_b = features_b.loc[test_ids]
            train_targets = status.loc[train_ids]
            test_targets = status.loc[test_ids]

            # No need to normalize for feature selection, its included inside
            load_features = False # Skip the feature selection process
            if not summary_file.is_file() or not load_features:
                controller.fit_df(train_feat_a,
                                  train_targets,
                                  train_feat_b)
                features_freq = controller.selector.saved_state['feat_freq']
                features_rate_table.append(pd.Series(features_freq['Rate'], name=f"freqrate_fold_{fold}"))
                selected_features = controller.selected_features
                selected_features_list.append(pd.Series([str(i) for i in selected_features],
                                                        name=f"selected_features_fold_{fold}"))
            else:
                features_freq = pd.read_excel(str(summary_file), sheet_name=f"features_freq_fold_{fold}",
                                              index_col=[0, 1, 2])
                raise NotImplemented
            fold_freq[fold] = features_freq

            # Isolate and normalize the features, try to drop useless metadata first
            results, predict_table = controller.model_builder.fit(train_feat_a[selected_features],
                                                                  train_targets,
                                                                  test_feat_a[selected_features],
                                                                  test_targets)
            best_params = controller.model_builder.saved_state['best_params']
            estimators = controller.model_builder.saved_state['estimators']

            # Save results
            performance.append(pd.Series(results, name=f'fold_{fold}'))
            predict_table.to_excel(pdt_writter, sheet_name=f'fold_{fold}')
            logger.info(f"Best_params: \n{best_params}")
            logger.info(f"Trying to dump models to {output_root}")
            controller.save(output_root.joinpath(f'fold_{fold}_models.ctl'))

        # Save results
        #-------------
        pd.concat(selected_features_list, axis=1).to_excel(excel_writter, sheet_name="Selected_Features")
        pd.concat(features_rate_table, axis=1).to_excel(excel_writter, sheet_name="Feat_Summary")
        pd.concat(performance, axis=1).to_excel(excel_writter, sheet_name="Performance")
        excel_writter.close()
        pdt_writter.close()

if __name__ == '__main__':
    main()


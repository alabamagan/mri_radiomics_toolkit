import pandas as pd

from pathlib import Path
from mnts.mnts_logger import MNTSLogger

import joblib
from sklearn import *
import numpy as np

from npc_radiomics.controller import Controller


def main():
    with MNTSLogger("./03_building_final_model.log", logger_name="main", log_level='debug', keep_file=True) as logger:
        # Load features & Config
        #-----------------------
        p_feature_a = Path('./output/original_feature-A.xlsx')
        p_feature_b = Path('./output/original_feature-B.xlsx')
        features_a = pd.read_excel(str(p_feature_a), header=(0, 1, 2), index_col=0)
        features_b = pd.read_excel(str(p_feature_b), header=(0, 1, 2), index_col=0)
        p_ctl_setting = Path('./02_configs/BRENT.yml')
        p_pyrad_setting = Path('./02_configs/pyrad_settings.yml')
        p_save_dir = Path('./output/03_final_model.ctl')
        p_out_excel = pd.ExcelWriter("./output/03_output_summary.xlsx")

        # Load target
        #------------
        status = Path("./output/gt-datasheet.csv")
        status = pd.read_csv(str(status), index_col=0)
        status.columns = ['Status']

        # match target and feature, use the datasheet
        features_a = features_a.loc[status.index]
        features_b = features_b.loc[status.index]

        # Setup controller
        controller = Controller(setting=str(p_ctl_setting), param_file=str(p_pyrad_setting))

        # Fit models
        results, predict_table = controller.fit_df(features_a,
                                                   status,
                                                   features_b)
        logger.info(f"Saving to {str(p_save_dir)}")
        controller.save(p_save_dir)

        # Save properties
        logger.info(f"Results: {results}")
        features_freq = controller.selector.saved_state['feat_freq']
        features_freq = pd.Series(features_freq['Rate'], name=f"freqrate_fold").to_frame()
        selected_features = controller.selected_features
        selected_features = pd.Series([str(i) for i in selected_features],name=f"selected_features").to_frame()
        features_freq.to_excel(p_out_excel, sheet_name="Feature Frequency")
        selected_features.to_excel(p_out_excel, sheet_name="Selected Features")
        p_out_excel.close()




if __name__ == '__main__':
    main()

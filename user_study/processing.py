#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing user study data.

@author: Robin Roussel
"""
import json
import pandas as pd
#import _context

# Import external data.
sheet = pd.read_excel("data/user_study-20170113.xlsx", sheetname="Sheet1")
sheet.drop(["Age", "Sex", "Familiarity"], 1, inplace=True)

# Create data frame.
colnames = [name[:-1] + ("BAS" if name.endswith("A") else "INV")
            for name in sheet.columns.values[1:]]
rownames = ["TotalTime",
            "TotalSliderMoves",
            "S1Moves",
            "S2Moves",
            "S3Moves",
            "S4Moves",
            "S5Moves",
            "HelperCalls",
            "InitDistParamSpace",
            "InitDistCurveSpace",
            "InitDistFeatureSpace",
            "FinalDistParamSpace",
            "FinalDistCurveSpace",
            "FinalDistFeatureSpace",
            "FinalDistPerceived",
            ]
stats = pd.DataFrame(columns=colnames, index=rownames)

#from _config import tasks, cand_name
#
#for i, task_data in enumerate(tasks):
#    taskid = task_data['taskid']
#    mecha = task_data['mecha_type'](*task_data['target_props'])
#    dp = list(mecha.props[:mecha.constraint_solver.nb_dprops])
#    nb = task_data['pt_density']
#    # Load candidate data
#    filename = "data/" + cand_name + "_task_" + str(taskid) + ".json"
#    with open(filename, "r") as file:
#        for subtask, data in json.load(file)['subtask_data'].items():
#            final_props = dp+data['cont_props'][-1][0]
#            mecha.reset(*final_props)



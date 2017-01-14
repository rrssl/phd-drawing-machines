#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing user study data.

@author: Robin Roussel
"""
import json
import pandas as pd
import scipy.spatial.distance as dist

from _config import tasks
#import _context

# Import spreadsheet.
sheet = pd.read_excel("data/user_study-20170113.xlsx", sheetname="Sheet1")
sheet.drop(["Age", "Sex", "Familiarity"], 1, inplace=True)
sheet.drop([0, 1], 0, inplace=True)
nb_candidates = len(sheet.index)

# Create data frames.
rownames = [name[:-1] + ("BAS" if name.endswith("A") else "INV")
            for name in sheet.columns.values[1:]]
colnames = ["TotalTime",
            "TotalSliderMoves",
#            "S1Moves",
#            "S2Moves",
#            "S3Moves",
#            "S4Moves",
#            "S5Moves",
#            "HelperCalls",
#            "InitDistParamSpace",
#            "InitDistCurveSpace",
#            "InitDistFeatureSpace",
#            "FinalDistParamSpace",
#            "FinalDistCurveSpace",
#            "FinalDistFeatureSpace",
            "FinalDistPerceived",
            "DistTravelled",
            ]
stats = pd.DataFrame(0., columns=colnames, index=rownames)

# Aggregate candidate results.
for cand_name in sheet['Name'].values:
    # Iterate over tasks.
    for i, task_data in enumerate(tasks):
        taskid = task_data['taskid']
        mecha = task_data['mecha_type'](*task_data['target_props'])
        dp = list(mecha.props[:mecha.constraint_solver.nb_dprops])
        nb = task_data['pt_density']
        # Load session data for this task.
        filename = "data/" + cand_name + "_task_" + str(taskid) + ".json"
        with open(filename, "r") as file:
            session_data = json.load(file)
            # Compute InitDist...
            init_props = session_data['init_props']

            # Iterate over subtasks.
            for subtask, sub_data in session_data['subtask_data'].items():
                sub_name = "T" + str(taskid) + subtask[7:10].upper()

                stats.at[sub_name, "TotalTime"] += sub_data['tot_time']

                nb_slider_moves = 0
                nb_helper_calls = 0
                dist_travelled = 0.

                positions = [[init_props[len(dp):], ""]] + sub_data['cont_props']
                for i, (pos, type_) in enumerate(positions[1:]):
                    if type_ == "s":
                        nb_slider_moves += 1
                    elif type_ == "p":
                        nb_helper_calls += 1
                    dist_travelled += dist.euclidean(positions[i][0], pos)
                    # TODO: add slider-specific measurements here

                stats.at[sub_name, "TotalSliderMoves"] += nb_slider_moves
#                stats.at[sub_name, "HelperCalls"] += nb_helper_calls
                stats.at[sub_name, "DistTravelled"] += dist_travelled

                # Compute FinalDist...
                final_props = dp+sub_data['cont_props'][-1][0]
                mecha.reset(*final_props)

                score_label = "T" + str(taskid) + (
                    "A" if sub_data['order'] == 0 else "B")
                score = float(
                    sheet.loc[sheet['Name'] == cand_name, score_label])
                stats.at[sub_name, "FinalDistPerceived"] += 5. - score

stats /= nb_candidates



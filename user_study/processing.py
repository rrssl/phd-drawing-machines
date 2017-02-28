#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing user study data.

@author: Robin Roussel
"""
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
colnames = ["Total time",
            "Total slider moves",
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
            "Final distance perceived",
            "Distance travelled",
            ]
means = pd.DataFrame(0., columns=colnames, index=rownames)

# Aggregate average candidate results.
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

                means.at[sub_name, "Total time"] += sub_data['tot_time']

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

                means.at[sub_name, "Total slider moves"] += nb_slider_moves
#                means.at[sub_name, "HelperCalls"] += nb_helper_calls
                means.at[sub_name, "Distance travelled"] += dist_travelled

                # Compute FinalDist...
                final_props = dp+sub_data['cont_props'][-1][0]
                mecha.reset(*final_props)

                score_label = "T" + str(taskid) + (
                    "A" if sub_data['order'] == 0 else "B")
                score = float(
                    sheet.loc[sheet['Name'] == cand_name, score_label])
                means.at[sub_name, "Final distance perceived"] += 5. - score

means /= nb_candidates

#------------------------------------------------------------------------------

errors = pd.DataFrame(0., columns=colnames, index=rownames)

# Aggregate variance
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

                errors.at[sub_name, "Total time"] += (
                        means.at[sub_name, "Total time"] - sub_data['tot_time'])**2

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

                errors.at[sub_name, "Total slider moves"] += (
                        means.at[sub_name, "Total slider moves"] - nb_slider_moves)**2
#                errors.at[sub_name, "HelperCalls"] += nb_helper_calls
                errors.at[sub_name, "Distance travelled"] += (
                        means.at[sub_name, "Distance travelled"] - dist_travelled)**2

                # Compute FinalDist...
                final_props = dp+sub_data['cont_props'][-1][0]
                mecha.reset(*final_props)

                score_label = "T" + str(taskid) + (
                    "A" if sub_data['order'] == 0 else "B")
                score = float(
                    sheet.loc[sheet['Name'] == cand_name, score_label])
                errors.at[sub_name, "Final distance perceived"] += (
                        means.at[sub_name, "Final distance perceived"] - (5. - score))**2

errors /= nb_candidates
errors = np.sqrt(errors)

#------------------------------------------------------------------------------

mpl.style.use('ggplot')

fig, ax = plt.subplots(2, 2, figsize=(16,9))
labels = ["T{}".format(i+1) for i in range(len(means.index)//2)]

def extract_subframe(frame, metric_name):
    return pd.DataFrame(
        data={'BAS': frame[metric_name][::2].values,
              'INV': frame[metric_name][1::2].values},
        index=labels)

for i, metric_name in enumerate(colnames):
    m = extract_subframe(means, metric_name)
    e = extract_subframe(errors, metric_name)
    m.plot.bar(yerr=e, rot=0, ax=ax.flat[i], title=metric_name, ylim=(0, None))

plt.ioff()
plt.show()



import sys, os, pdb
from os.path import join as ospj
import json
from collections import *

import numpy as np
import pandas as pd
from pandas.core.common import flatten

import pprint
pp = pprint.PrettyPrinter()

#%% md

d_folder = '/home/zhongchongyang/BABEL-main/data/babel_v1.0_release' # Data folder
l_babel_dense_files = ['train', 'val']
l_babel_extra_files = ['extra_train', 'extra_val']

# BABEL Dataset
babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(ospj(d_folder, file+'.json')))

for file in l_babel_extra_files:
    babel[file] = json.load(open(ospj(d_folder, file+'.json')))

#%% md

### Duration of mocap for which BABEL action labels are available

#%%

# for babel_set in [l_babel_dense_files, l_babel_dense_files+l_babel_extra_files]:
#     dur = 0.0
#     list_sids = []
#     for spl in babel_set:
#         for sid in babel[spl]:
#             if sid not in list_sids:
#                 list_sids.append(sid)
#                 dur += babel[spl][sid]['dur']
#
#                 # Duration of each set
#     minutes = dur//60
#     print('Total duration = {0} hours {1} min. {2:.0f} sec.'.format(
#         minutes//60, minutes%60, dur%60))
#     print('Total # seqs. = ', len(list_sids))
#     print('-'*30)

#%% md

### Search BABEL for action

#%%

def get_cats(ann, file):
    # Get sequence labels and frame labels if they exist
    seq_l, frame_l = [], []
    if 'extra' not in file:
        if ann['seq_ann'] is not None:
            seq_l = flatten([seg['act_cat'] for seg in ann['seq_ann']['labels']])
        if ann['frame_ann'] is not None:
            frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = flatten([seg['act_cat'] for seq_ann in ann['seq_anns'] for seg in seq_ann['labels']])
        if ann['frame_anns'] is not None:
            frame_l = flatten([seg['act_cat'] for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']])

    return list(seq_l), list(frame_l)

#%%

action = 'transition'
act_anns = defaultdict(list) # { seq_id_1: [ann_1_1, ann_1_2], seq_id_2: [ann_2_1], ...}
n_act_spans = 0

for spl in babel:
    for sid in babel[spl]:

        seq_l, frame_l = get_cats(babel[spl][sid], spl)
        # print(seq_l + frame_l)

        if action in seq_l + frame_l:

            # Store all relevant mocap sequence annotations
            act_anns[sid].append(babel[spl][sid])

            # # Individual spans of the action in the sequence
            n_act_spans += Counter(seq_l+frame_l)[action]

print('# Seqs. containing action {0} = {1}'.format(action, len(act_anns)))
print('# Segments containing action {0} = {1}'.format(action, n_act_spans))

#%%

# View a random annotation
key = np.random.choice(list(act_anns.keys()))
pp.pprint(act_anns[key])
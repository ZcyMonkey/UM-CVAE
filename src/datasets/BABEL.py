#!/usr/bin/env python
  
# -*- coding: utf-8 -*-
#
# Adapted from https://github.com/lshiwjx/2s-AGCN for BABEL (https://babel.is.tue.mpg.de/)

import pickle as pkl

import numpy
import numpy as np
import os
from .dataset import Dataset
from collections import *

class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self, datapath="/data_1/zhongchongyang/babel_dataset", **kargs):
        self.datapath = datapath
        spl = 'train'
        super().__init__(**kargs)
        total_num_actions = 20
        self.num_classes = total_num_actions
        act2idx_150 = BABEL_action_enumerator20
        act2idx = {k: act2idx_150[k] for k in act2idx_150 if act2idx_150[k] < self.num_classes}
        pkldatafilepath = os.path.join(datapath, "babel_v1.0_train_samples.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))
        split_idxs = defaultdict(list)
        for i, y1 in enumerate(data['Y1']):

            # Check if action category in list of classes
            if y1 not in act2idx:
                continue

            sid = data['sid'][i]
            split_idxs[spl].append(i)
        ar_idxs = np.array(split_idxs[spl])
        data_used = []
        actions = []
        for idx in ar_idxs:
            # for i in range(len(data["X"][idx])-1,-1,-1):
            #     if data["X"][idx][i].all() == 0:
            #         data["X"][idx] = numpy.delete(data["X"][idx],i,axis=0)
            #     else:
            #         continue
            data_remove0 = data["X"][idx][[not np.all(data["X"][idx][i] == 0) for i in range(data["X"][idx].shape[0])], :]
            if (data_remove0.shape[0]>60):
                data_used.append(data_remove0)
                actions.append(data["Y1"][idx])
        pose=[x[:,:22,:] for x in data_used]
        pos3d=[x[:,22:,:]for x in data_used]

        self._pose = pose
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = pos3d

        self._actions = [x for x in actions]




        self._train = list(range(len(self._pose)))

        #keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(act2idx)}
        self._label_to_action = {i: x for i, x in enumerate(act2idx)}

        self._action_classes = act2idx

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose


BABEL_action_enumerator150 = {
    "walk": 0,
    "stand": 1,
    "hand movements": 2,
    "turn": 3,
    "interact with/use object": 4,
    "arm movements": 5,
    "t pose": 6,
    "step": 7,
    "backwards movement": 8,
    "raising body part": 9,
    "look": 10,
    "touch object": 11,
    "leg movements": 12,
    "forward movement": 13,
    "circular movement": 14,
    "stretch": 15,
    "jump": 16,
    "touching body part": 17,
    "sit": 18,
    "place-something": 19,
    "take/pick something up": 20,
    "run": 21,
    "bend": 22,
    "throw": 23,
    "foot movements": 24,
    "a pose": 25,
    "stand up": 26,
    "lowering body part": 27,
    "sideways movement": 28,
    "move up/down incline": 29,
    "action with ball": 30,
    "kick": 31,
    "gesture": 32,
    "head movements": 33,
    "jog": 34,
    "grasp object": 35,
    "waist movements": 36,
    "lift something": 37,
    "knee movement": 38,
    "wave": 39,
    "move something": 40,
    "swing body part": 41,
    "catch": 42,
    "dance": 43,
    "lean": 44,
    "greet": 45,
    "poses": 46,
    "touching face": 47,
    "sports move": 48,
    "exercise/training": 49,
    "clean something": 50,
    "punch": 51,
    "squat": 52,
    "scratch": 53,
    "hop": 54,
    "play sport": 55,
    "stumble": 56,
    "crossing limbs": 57,
    "perform": 58,
    "martial art": 59,
    "balance": 60,
    "kneel": 61,
    "shake": 62,
    "grab body part": 63,
    "clap": 64,
    "crouch": 65,
    "spin": 66,
    "upper body movements": 67,
    "knock": 68,
    "adjust": 69,
    "crawl": 70,
    "twist": 71,
    "move back to original position": 72,
    "bow": 73,
    "hit": 74,
    "touch ground": 75,
    "shoulder movements": 76,
    "telephone call": 77,
    "grab person": 78,
    "play instrument": 79,
    "tap": 80,
    "spread": 81,
    "skip": 82,
    "rolling movement": 83,
    "jump rope": 84,
    "play catch": 85,
    "drink": 86,
    "evade": 87,
    "support": 88,
    "point": 89,
    "side to side movement": 90,
    "stop": 91,
    "protect": 92,
    "wrist movements": 93,
    "stances": 94,
    "wait": 95,
    "shuffle": 96,
    "lunge": 97,
    "communicate (vocalise)": 98,
    "jumping jacks": 99,
    "rub": 100,
    "dribble": 101,
    "swim": 102,
    "sneak": 103,
    "to lower a body part": 104,
    "misc. abstract action": 105,
    "mix": 106,
    "limp": 107,
    "sway": 108,
    "slide": 109,
    "cartwheel": 110,
    "press something": 111,
    "shrug": 112,
    "open something": 113,
    "leap": 114,
    "trip": 115,
    "golf": 116,
    "move misc. body part": 117,
    "get injured": 118,
    "sudden movement": 119,
    "duck": 120,
    "flap": 121,
    "salute": 122,
    "stagger": 123,
    "draw": 124,
    "tie": 125,
    "eat": 126,
    "style hair": 127,
    "relax": 128,
    "pray": 129,
    "flip": 130,
    "shivering": 131,
    "interact with rope": 132,
    "march": 133,
    "zombie": 134,
    "check": 135,
    "wiggle": 136,
    "bump": 137,
    "give something": 138,
    "yoga": 139,
    "mime": 140,
    "wobble": 141,
    "release": 142,
    "wash": 143,
    "stroke": 144,
    "rocking movement": 145,
    "swipe": 146,
    "strafe": 147,
    "hang": 148,
    "flail arms": 149
}
BABEL_action_enumerator21 = {
    "walk": 0,
    "turn": 1,
    "arm movements": 2,
    "stretch": 3,
    "jump": 4,
    "sit": 5,
    "run": 6,
    "bend": 7,
    "throw": 8,
    "sideways movement": 9,
    "wave": 10,
    "martial art": 11,
    "kneel": 12,
    "knock": 13,
    "twist": 14,
    "bow": 15,
    "play instrument": 16,
    "slide": 17,
    "tie": 18,
    "eat": 19,
    "zombie": 20
}
BABEL_action_enumerator20 = {"walk":0,"backwards movement": 1,"kick": 2,"jump":3,"sit":4,"run":5,"dance":6,"throw":7,"sideways movement":8,
                             "punch": 9,"drink": 10,"twist":11,"zombie": 12,"swim":13,"sneak": 14,"golf": 15,"lift something": 16,"salute": 17,"crouch": 18,"cartwheel": 19}
















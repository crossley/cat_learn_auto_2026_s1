import math
import random
from datetime import datetime
import numpy as np
import pandas as pd
from psychopy import core, event, visual

# define function to create df of numerical stroop number pairs
def make_stroop_pairs(n_total, p_incongruent, random_seed=None):
    rng = np.random.default_rng(random_seed)

    ds_ns_rec = []

    # create master list of numbers
    num_list = [2, 3, 4, 5, 6, 7, 8]

    # number of incongruent and congruent trials
    n_incongruent = int(np.ceil(p_incongruent * n_total))
    n_congruent = n_total - n_incongruent

    # split each condition into ~half size cue, ~half value cue
    # if odd number, extra trial goes to "Value"
    half_incon = n_incongruent // 2
    half_con = n_congruent // 2

    # running counters inside each condition
    incon_count = 0
    con_count = 0

    # sample from list randomly without replacement -- ensures none are equal
    for trl in range(n_total):
        num_left, num_right = rng.choice(num_list, size=2, replace=False)

        # set congruency
        if trl < n_incongruent:
            congruency = "incongruent"

            # font is bigger for smaller number
            if num_left > num_right:
                # small font for left number
                size_left = "small"
                size_right = "big"
            else: 
                # small font for right number
                size_left = "big"
                size_right = "small"

            if incon_count < half_incon:
                cue = "Size"
            else: 
                cue = "Value"

            incon_count += 1

        else: 
            congruency = "congruent"

            if num_left > num_right:
                # font is bigger for left number
                size_left = "big"
                size_right = "small"
            else: 
                # font is bigger for right number
                size_right = "big"
                size_left = "small"
            
            if con_count < half_con:
                cue = "Size"
            else:
                cue = "Value"

            con_count += 1
        
        ds_ns_rec.append({"value_left": num_left, 
                          "size_left": size_left, 
                          "value_right": num_right,
                          "size_right": size_right,
                          "congruency": congruency,
                          "cue": cue})

    ds_ns = pd.DataFrame(ds_ns_rec)

    if random_seed is None:
        ds_ns = ds_ns.sample(frac=1).reset_index(drop=True)
    else:
        ds_ns = ds_ns.sample(frac=1, random_state=int(rng.integers(
            0, 2**32 - 1))).reset_index(drop=True)

    return ds_ns

# check = ds_ns["value_left"] == ds_ns["value_right"]


import numpy as np
import pandas as pd


def read_data():
    dataset = np.genfromtxt('dermatology.data',delimiter=',')
    mean = np.nanmean(dataset,axis=0)
    inds = np.where(np.isnan(dataset))
    dataset[inds] = np.take(mean,inds[1])
    return dataset

def variance_threshold(dataset):
    variance = np.var(dataset,axis=0)
    rows = len(variance)
    variance = np.reshape(variance,(rows,1))
    col = []
    for i in range(rows):
        col.append(i)
    col = np.reshape(col,(rows,1))
    # print(col)
    new_arr = np.c_[variance,col]
    new_arr = sorted(new_arr, key = lambda x:x[0])
    new_arr = np.asarray(new_arr)
    
    return new_arr

def get_feature_name(val):
    feature_name = {
        0: 'erythema',
        1: 'scaling',
        2: 'definite_borders',
        3: 'itching',
        4: 'koebner_phenomenon',
        5: 'polygonal_papules',
        6: 'follicular_papules',
        7: 'oral_mucosal_involvement',
        8: 'knee_and_elbow_involvement',
        9: 'scalp_involvement',
        10: 'family_history',
        11: 'melanin_incontinence',
        12: 'eosinophils_in_the_infiltrate',
        13: 'PNL_infiltrate',
        14: 'fibrosis_of_the_papillary_dermis',
        15: 'exocytosis',
        16: 'acanthosis',
        17: 'hyperkeratosis',
        18: 'parakeratosis',
        19: 'clubbing_of_the_rete_ridges',
        20: 'elongation_of_the_rete_ridges',
        21: 'thinning_of_the_suprapapillary_epidermis',
        22: 'spongiform_pustule',
        23: 'munro_microabcess',
        24: 'focal_hypergranulosis',
        25: 'disappearance_of_the_granular_layer',
        26: 'vacuolisation_damage_of_basal_layer',
        27: 'spongiosis',
        28: 'saw_tooth_appearance_of_retes',
        29: 'follicular_horn_plug',
        30: 'perifollicular_parakeratosis',
        31: 'inflammatory_monoluclear_inflitrate',
        32: 'band_like_infiltrate',
        33: 'Age'
    }

    return feature_name.get(val)

def print_feature_ranking(arr):
    for i in range(len(arr[:,1])):
        val = arr[i][1]
        if(val != 34):
            print(get_feature_name(val))

if(__name__ == "__main__"):
    dataset = read_data()
    feature_ranking = variance_threshold(dataset)
    print_feature_ranking(feature_ranking)
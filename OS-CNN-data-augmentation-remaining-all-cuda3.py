from os.path import dirname

import pandas as pd
import numpy as np
from sklearn import preprocessing
from Classifiers.OS_CNN.OS_CNN_easy_use import OS_CNN_easy_use
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataloader.TSC_data_loader import TSC_data_loader
from dataset_types import all_up_to_2Mb_dataset_list_remaining

import torch
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

# Jitter, scaling, window_slice, window_warp, time_warp, rotation, magnitude_warp, spawner, wdba
https://github.com/uchidalab/time_series_augmentation
def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import utils.dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag = dtw.RETURN_ALL, slope_constraint=slope_constraint, window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jitter(ret, sigma=sigma)


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret

def pipeline(dataset_name_to_save, X_train, y_train, X_test, y_test, epochs):
    Result_log_folder = './Example_Results_of_OS_CNN/OS_CNN_result_iter_0/'

    model = OS_CNN_easy_use(
        Result_log_folder = Result_log_folder, # the Result_log_folder,
        dataset_name = dataset_name_to_save,           # dataset_name_to_save for creat log under Result_log_folder,
        device = "cuda:3",                # Gpu 
        max_epoch = epochs,                        # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough,
        paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128],
        )

    model.fit(X_train, y_train, X_test, y_test)

    y_predict = model.predict(X_test)

    acc = accuracy_score(y_predict, y_test)
    print(acc)
    return acc


if __name__ == "__main__":
    NUM_AUG_RUNS = 5
    AUG_PERCENTAGE = 0.2 # Increase dataset by 20 percent
    results_df = pd.DataFrame()

    augmentations = [(jitter, "jitter"), (scaling, "scaling"), (window_slice, "window_slice"),
                    (window_warp, "window_warp"), (time_warp, "time_warp"), (rotation, "rotation"),
                    (magnitude_warp, "magnitude_warp")]
    augmentations_need_labels = [(spawner, "spawner"), (wdba, "wdba")]

    for dataset_name in all_up_to_2Mb_dataset_list_remaining[30:]:
        dataset_path = dirname("./Example_Datasets/UCRArchive_2018/")
        X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)
        for augmentation in augmentations:
            fun = augmentation[0]
            augmentation_name = augmentation[1]
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_train_funced = fun(X_train_reshaped)
            X_train_funced = X_train_funced.reshape((X_train_funced.shape[0], X_train_funced.shape[1]))
            X_train_funced = X_train_funced.astype(np.float32)
            dataset_name_to_save = f"{dataset_name}_{augmentation_name}_aug"
            aug_length = int(AUG_PERCENTAGE*len(X_train_funced))

            X_train_aug = np.vstack((X_train, X_train_funced[:aug_length]))
            y_train_aug = np.concatenate((y_train, y_train[:aug_length]))
            accs = []
            for i in range(NUM_AUG_RUNS):
                acc = pipeline(dataset_name_to_save, X_train_aug, y_train_aug, X_test, y_test, epochs=2000)
                accs.append(acc)
                results_df.loc[dataset_name_to_save, f"acc{i+1}"] = acc
            results_df.loc[dataset_name_to_save, "mean_acc"] = np.mean(accs)
            results_df.to_csv("results_aug_remaining_cuda3.csv", index=True)
        
        for augmentation in augmentations_need_labels:
            fun = augmentation[0]
            augmentation_name = augmentation[1]
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_train_funced = fun(X_train_reshaped, y_train)
            X_train_funced = X_train_funced.reshape((X_train_funced.shape[0], X_train_funced.shape[1]))
            X_train_funced = X_train_funced.astype(np.float32)
            dataset_name_to_save = f"{dataset_name}_{augmentation_name}_aug"
            aug_length = int(AUG_PERCENTAGE*len(X_train_funced))

            X_train_aug = np.vstack((X_train, X_train_funced[:aug_length]))
            y_train_aug = np.concatenate((y_train, y_train[:aug_length]))
            accs = []
            for i in range(NUM_AUG_RUNS):
                acc = pipeline(dataset_name_to_save, X_train_aug, y_train_aug, X_test, y_test, epochs=2000)
                accs.append(acc)
                results_df.loc[dataset_name_to_save, f"acc{i+1}"] = acc
            results_df.loc[dataset_name_to_save, "mean_acc"] = np.mean(accs)
            results_df.to_csv("results_aug_remaining_cuda3.csv", index=True)

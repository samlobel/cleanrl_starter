import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

def gather_pkl_files_from_base_dir(base_dir, selected_run_names=None):
    """
    base dir has directories with log files inside of them

    Args:
        base_dir (_type_): acme results dir, if None use the default acme setting
        selected_run_names (_type_): if None, use all run_names, otherwise use the selected ones
        
    Returns:
        a dict of {run_name: pkl_path}
    """
    # find all subdir and get the run_name of all
    id_to_pkl = {}
    for run_name in os.listdir(base_dir):
        if selected_run_names is not None and run_name not in selected_run_names:
            continue
        exp_dir = os.path.join(base_dir, run_name)
        if not os.path.isdir(exp_dir):
            continue
        pkl_path = os.path.join(exp_dir, 'log_dict.pkl')
        id_to_pkl[run_name] = pkl_path

    return id_to_pkl

def interpolate_xys(scores):
    all_xs = [frames for (frames, returns) in scores]
    flattened_unique = np.sort(np.unique(np.concatenate(all_xs)))
    all_returns = []
    for frames, returns in scores:
        f = scipy.interpolate.interp1d(frames, returns, kind='nearest', fill_value='extrapolate')
        interpolated_returns = f(flattened_unique)
        all_returns.append(interpolated_returns)

    return np.array(flattened_unique), np.array(all_returns)

def truncate_and_interpolate(scores, max_frames=-1, min_frames=-1):
    filtered_scores = [(frames, returns) for (frames, returns) in scores if max(frames) > min_frames]
    if not filtered_scores:
        return np.array([]), np.array([])
    assert all(all(frames[i] <= frames[i+1] for i in range(len(frames) - 1)) for (frames, returns) in filtered_scores), "needs to be sorted for what follows"
    min_max_frames = min([max(frames) for (frames, returns) in filtered_scores])
    min_max_frames = min(min_max_frames, max_frames)
    shortened_scores = []
    for frames, returns in filtered_scores:
        if min_max_frames > 0:
            # import ipdb; ipdb.set_trace()
            frames = np.array(frames) # sure
            frames = frames[frames <= max_frames]
            returns = returns[:len(frames)]
        shortened_scores.append((frames, returns))
    
    xs, all_ys = interpolate_xys(shortened_scores)
    return xs, all_ys


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=25):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def smoothen_data(scores, n=10):
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data

def generate_plot(xs, all_ys, label, smoothen=0, linewidth=2, all_seeds=False, log_scale=False):
    # smoothen is a number of iterations to average over. Much less meaningful now that we have weird x axis.
    if smoothen > 0:
        xs = moving_average(xs, n=smoothen)
        all_ys = smoothen_data(all_ys, n=smoothen)

    median, mean, top, bottom = get_plot_params(all_ys)
    # 
    plt.plot(xs, mean, linewidth=linewidth, label=label, alpha=0.9)
    # plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )
    plt.fill_between( xs, top, bottom, alpha=0.2 )
    if all_seeds:
        for i, score in enumerate(all_ys):
            plt.plot(xs, score, linewidth=linewidth, label=label+f"_{i+1}", alpha=0.6)

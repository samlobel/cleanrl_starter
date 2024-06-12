import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from comparison_plotting_utils import *
from plotting_utils import get_summary_data, extract_exploration_amounts, load_count_dict, get_true_vs_approx


def get_config(run_name, group_key):
    try:
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)_.*", run_name).group(1)
    except: # if its at the end of the id name
        try:
            return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)$", run_name).group(1)
        except Exception as e:
            print(f"Failed on {run_name}, {group_key}")
            raise e


def default_make_key(log_dir_name, group_keys):
    keys = [get_config(log_dir_name, group_key) for group_key in group_keys]
    key = "_".join(keys)
    return key

def extract_log_dirs_filter(id_to_pkl, group_keys=("rewardscale",), field_to_plot='episodic_return', filter_func=None):
    # Filter only includes ones that are true

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        if filter_func and not filter_func(run_name):
            continue
        try:
            keys = [get_config(run_name, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(run_name, group_keys)
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {run_name}")
            print(e)

    return log_dir_map

def extract_log_dirs(id_to_pkl, group_keys=("rewardscale",), field_to_plot='episodic_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        try:
            keys = [get_config(run_name, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(run_name, group_keys)
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {run_name}")
            print(e)

    return log_dir_map

def extract_log_dirs_group_func(id_to_pkl, group_func=lambda x: x, field_to_plot='episodic_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        try:
            key = group_func(run_name)
            if key is None:
                continue
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except:
            print(f"Could not extract from {run_name}")

    return log_dir_map


def plot_comparison_learning_curves(
    # id_to_pkl, # dict
    base_dir, #str
    selected_run_names=None,
    # experiment_name=None,
    # stat='eval_episode_lengths',
    group_keys=("constraintlossscale",),
    group_func=None,
    filter_func=None, # Only include things that are "true" in filter. At the moment this is on parsed config name.
    run_name_filter_func=None,
    save_path=None,
    show=True,
    smoothen=10,
    log_dir_path_map=None,
    uniform_truncate=False,
    truncate_max_frames=-1,
    truncate_min_frames=-1,
    ylabel=False,
    legend_loc=None,
    linewidth=2,
    min_seeds=1,
    all_seeds=False,
    title=None,
    min_final_val=None,
    max_final_val=None,
    field_to_plot='episodic_return',
    log_scale=False,
    include_legend=True,
    include_y_label=True,
    include_x_label=True,
    ):

    # import seaborn as sns
    # NUM_COLORS=100
    # clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    # sns.set_palette(clrs)
    id_to_pkl = gather_pkl_files_from_base_dir(base_dir=base_dir, selected_run_names=selected_run_names)

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"
    if save_path is not None:
        plt.figure(figsize=(24,12))


    # ylabel = ylabel or "Average Return"
    ylabel = ylabel or field_to_plot

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(id_to_pkl=id_to_pkl, group_func=group_func, field_to_plot=field_to_plot)
        else:
            # log_dir_path_map = extract_log_dirs(id_to_pkl=id_to_pkl, group_keys=group_keys, field_to_plot=field_to_plot)
            log_dir_path_map = extract_log_dirs_filter(id_to_pkl=id_to_pkl, group_keys=group_keys, field_to_plot=field_to_plot, filter_func=run_name_filter_func)

    # for config in log_dir_path_map:
    for config in sorted(log_dir_path_map.keys()):
        if config is None:
            continue
        if filter_func and not filter_func(config):
            continue
        curves = log_dir_path_map[config]
        print(config)
        for curve in curves:
            print(f"\t{len(curve[0])}")
        truncated_xs, truncated_all_ys = truncate_and_interpolate(curves, max_frames=truncate_max_frames, min_frames=truncate_min_frames)
        print(truncated_xs.shape)
        # 
        if len(truncated_all_ys) < min_seeds:
            continue
        if smoothen and smoothen > 0 and len(truncated_xs) < smoothen:
            continue

        if min_final_val is not None:
            # 
            if np.array(truncated_all_ys)[:, -1].mean().item() <= min_final_val:
                print('skipping because min_final_val violated')
                continue
            # else:
            #     print("not skipping")
            #     import ipdb; ipdb.set_trace()
            #     # print("val", np.array(truncated_all_ys)[:, -1].mean().item())
            #     print("vals", np.array(truncated_all_ys)[:, -1].tolist())
        if max_final_val is not None:
            # 
            if np.array(truncated_all_ys)[:, -1].mean().item() >= max_final_val:
                print('skipping because max_final_val violated')
                continue
        # score_array = np.array(truncated_all_ys)
        print(np.max(truncated_all_ys))
        generate_plot(
            # score_array,
            truncated_xs,
            truncated_all_ys,
            label=config,
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds,
            log_scale=log_scale)
    
    # plt.grid()
    if include_x_label:
        plt.xlabel("Environment Steps")
    if include_y_label:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if log_scale:
        plt.yscale('log')
    if include_legend:
        if legend_loc:
            plt.legend(loc=legend_loc)
        else:
            plt.legend()

    if show:
        plt.show()

    # if show:
    #     if legend_loc:
    #         plt.legend(loc=legend_loc)
    #     else:
    #         plt.legend()
    #     plt.show()
    
    if save_path is not None:
        # plt.legend()
        plt.savefig(save_path)
        plt.close()


def get_rmse_for_each_iteration(count_dict):
    exact, approx = get_true_vs_approx(count_dict, "bonus")
    assert len(exact) == len(approx)
    exact = np.asarray(exact)
    approx = np.asarray(approx)
    sq_errors = (exact-approx) ** 2
    root_mean_sq_errors = np.mean(sq_errors) ** 0.5
    return root_mean_sq_errors



if __name__ == "__main__":

    group_func = None
    run_name_filter_func = None
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/tabular/ring_sgd_sweep_1"
    base_dir = "/Users/slobal1/Code/ML/cleanrl_sample/runs/onager_testing_1"

    # def run_name_filter_func(run_name):
    #     fields_to_check = [
    #         "amountnoiseprob_0.1",
    #         "maingammaindex_-3",
    #     ]
    #     return all([field in run_name for field in fields_to_check])

    group_keys = [
        "totaltimesteps",
    ]

    plot_comparison_learning_curves(
        base_dir=base_dir,
        save_path=None,
        show=True,
        # save_path="/Users/slobal1/Downloads/matplotlib_plots/r2d2/visgrid/cfn_tau_1.png",
        group_keys=group_keys,
        group_func=group_func,
        run_name_filter_func=run_name_filter_func,
        smoothen=100,
        field_to_plot="episodic_return", # td_loss, total_loss, SPS, constraint_loss, q_values, episodic_return, episodic_length, 
        # truncate_min_frames=400000,
        # truncate_max_frames=11000,
        # min_seeds=5,
        # all_seeds=True,
        # title="R2D2 RND sweep",
        # min_final_val=0.01,
        # max_final_val=1e4,
        # log_scale=True,
        # include_legend=False,
        )

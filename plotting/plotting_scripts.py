from comparison_plotting import *
import os
from functools import partial

this_dir = os.path.dirname(os.path.realpath(__file__))
above_this_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
PLOT_DIR = os.path.join(above_this_dir, "plots")

# This is just an example of what could be done.

log_scale_fields = (
    "td_loss",
    "constraint_loss",
    "pairwise_violation_mse",
    "tabular_total_mse_from_optimal",
    "tabular_total_l1_from_optimal",
)

def make_minatar_sweep_general(base_dir, filters, group_keys, save_dir, not_filters=(),
                               save_file="minatar_sweep.png", field_to_plot="episodic_return",
                               log_scale=False, smoothen=100):

    def run_name_filter_func_pre(run_name, env_id="RandomTabularEnv-v0"):
        # mode_match= f"tabularinitializationmode_{mode}" in run_name
        env_match = f"envid_{env_id}" in run_name
        filters_match = all([field in run_name for field in filters])
        not_filters_match = all([field not in run_name for field in not_filters])

        return filters_match and not_filters_match and env_match

    plt.rcParams["figure.figsize"] = (20,12)
    for i, env_id in enumerate(["Asterix-v1", "Breakout-v1", "Freeway-v1", "Seaquest-v1", "SpaceInvaders-v1"]):
        plt.subplot(2, 3, i+1)
        run_name_filter_func = partial(run_name_filter_func_pre, env_id=env_id)
        plot_comparison_learning_curves(
            base_dir=base_dir,
            save_path=None,
            show=False,
            group_keys=group_keys,
            group_func=None,
            run_name_filter_func=run_name_filter_func,
            smoothen=smoothen,
            field_to_plot=field_to_plot,
            # field_to_plot="tabular_total_l1_from_optimal",
            log_scale=log_scale,
            # include_x_label=(i == ),
            # include_y_label=(i % 4 == 0),
            include_x_label=True,
            include_y_label=True,
            # include_legend=(i==4),
            include_legend=True,
            )
        plt.title(env_id)
    # plt.suptitle(env_id)
    plt.savefig(os.path.join(save_dir, save_file))
    plt.close()


def make_first_minatar_sweep():
    print("make_first_minatar_sweep")
    base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/minatar/all_envs_first_sweep_target_toggle"
    group_keys = ["applyconstrainttotarget"]
    filters = []
    save_dir = os.path.join(PLOT_DIR, "minatar", "first_sweep_target_toggle")

    for field_to_plot in ["episodic_return", "td_loss", "constraint_loss", "pairwise_violation_mse"]:
        print(field_to_plot)
        # field_to_plot = "pairwise_violation_mse"
        log_scale = field_to_plot in log_scale_fields
        save_file = field_to_plot.replace("_", " ").capitalize().replace(" ", "") + ".png"
        make_minatar_sweep_general(base_dir, filters, group_keys, save_dir,
                                save_file=save_file,
                                field_to_plot=field_to_plot,
                                log_scale=log_scale,
                                smoothen=1000)
    
    base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/minatar/all_envs_target_toggle_less_regularization"
    save_dir = os.path.join(PLOT_DIR, "minatar", "first_sweep_target_toggle_smaller_constraint")

    for field_to_plot in ["episodic_return", "td_loss", "constraint_loss", "pairwise_violation_mse"]:
        print(field_to_plot)
        # field_to_plot = "pairwise_violation_mse"
        log_scale = field_to_plot in log_scale_fields
        save_file = field_to_plot.replace("_", " ").capitalize().replace(" ", "") + ".png"
        make_minatar_sweep_general(base_dir, filters, group_keys, save_dir,
                                save_file=save_file,
                                field_to_plot=field_to_plot,
                                log_scale=log_scale,
                                smoothen=1000)

    # make_minatar_sweep_general(base_dir, filters, group_keys, save_dir, save_file="minatar_first_sweep_target_toggle.png",
    #                            field_to_plot=field_to_plot, log_scale=log_scale)



if __name__ == '__main__':
    make_first_minatar_sweep()

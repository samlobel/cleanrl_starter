from comparison_plotting_utils import *
from plotting_utils import *


def plot_single_run(run_directory, field_to_plot, smoothen=10, log_scale=False):
    pkl_path = os.path.join(run_directory, 'log_dict.pkl')
    frames, quantities_to_plot = get_summary_data(pkl_path, field_to_plot)
    print(quantities_to_plot)
    # Might need to add dummy dimension
    frames = np.array(frames)
    quantities_to_plot = np.array(quantities_to_plot)
    quantities_to_plot = quantities_to_plot[None, :]
    generate_plot(frames, quantities_to_plot, field_to_plot, smoothen=smoothen)
    if log_scale:
        plt.yscale('log')

    plt.legend()
    plt.xlabel("Environment Steps")
    plt.ylabel(field_to_plot.replace("_", " "))
    plt.show()


if __name__ == '__main__':
    """Set up so that we can do a single run plot"""
    experiment_name = "/Users/slobal1/Code/ML/cleanrl_sample/runs/testing"
    run_title = "heelo"
    run_directory = os.path.join(experiment_name, run_title)
    field_to_plot = "episodic_return"

    plot_single_run(run_directory, field_to_plot=field_to_plot, smoothen=10, log_scale=False)


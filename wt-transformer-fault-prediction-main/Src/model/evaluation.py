import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def find_kappa(
    train_results: pd.DataFrame,
    median_error: float,
    std_error: float,
) -> tuple:
    """
    Find the optimal kappa value for the model.
    """
    optimal_kappa = 0.0
    for k in np.arange(3, 6.2, 0.2):
        k = round(k, 2)
        threshold = median_error + k*std_error
        outliers = train_results[train_results['Residual'] > threshold].shape[0]

        print(f'Number of outliers found for threshold k={k}: {outliers}')
        if outliers == 0:
            print(f'Optimal kappa value found: {k}')
            optimal_kappa = k
            break
    else:
        print('No optimal kappa value found within the range. Maximum kappa value used (6.0).')
        optimal_kappa = 6.0
        
    return optimal_kappa

def plot_results(
    train_results: pd.DataFrame,
    test_result: pd.DataFrame,
    alarm_df: pd.DataFrame,
    training_split: float,
    threshold: float,
    project_folder: str,
    wind_turbine: str,
    WT_name: str,
    selected_kappa: float,
    MA_window: int,
    colors: dict,
    alarm_mode: str = 'dynamic',
    month_interval: int = 4,
    fontsize: int = 22,
    figsize: tuple = (12, 6),
    ax1_y_range: np.array = np.arange(0.0, 0.018, 0.002),
    ax2_y_range: np.array = np.arange(-5, 4, 1)
) -> None:
    """
    Plot the results of the model training and testing.
    """

    train_size = int(train_results.shape[0] * training_split)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left y-axis (Residual)
    sns.lineplot(data=train_results[:train_size], x='date', y='Residual', ax=ax1, color=colors['train'])
    sns.lineplot(data=train_results[train_size:], x='date', y='Residual', ax=ax1, color=colors['validation'])
    sns.lineplot(data=test_result, x='date', y='Residual', ax=ax1, color=colors['test'])

    ax1.axhline(y=threshold, color=colors['threshold'], linestyle='--', linewidth=2, label=f'threshold \u03BA={selected_kappa}')

    if WT_name == 'WT8':
        ax1.axvline(x=pd.to_datetime('2021-10-18'), color='black', linestyle='--', linewidth=2)
        ax1.axvline(x=pd.to_datetime('2022-02-23'), color='brown', linestyle='--', linewidth=2)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot on the second y-axis
    if alarm_mode == 'constant':
        ax2.axhline(y=0, color='red', linewidth=2)
    elif alarm_mode == 'dynamic' and WT_name == 'WT8':
        sns.lineplot(data=alarm_df, x='date', y='Alarm', ax=ax2, color=colors['alarm'], label='FPI')
    elif alarm_mode == 'dynamic' and WT_name != 'WT8':
        sns.lineplot(data=alarm_df, x='date', y='Alarm', ax=ax2, color=colors['alarm'], label='FPI')
    elif alarm_mode == 'none':
        pass

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=fontsize, loc='upper left')

    # Remove the default legend from the second axis
    if ax2.get_legend():
        ax2.get_legend().remove()

    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)  # Left y-axis ticks
    ax2.tick_params(axis='y', labelsize=fontsize)  # Right y-axis ticks
    plt.title(WT_name, fontsize=fontsize)

    # Change the range of the y_axis in axis 1, with a fixed step
    ax1.set_yticks(ax1_y_range)
    ax2.set_yticks(ax2_y_range)

    # Save the img in Testing_plots folder
    fig.savefig(f'{project_folder}/Testing_plots/{wind_turbine}_{MA_window}_residuals.png')
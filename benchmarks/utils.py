import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch

from network import SdfNet

DEFAULT_SDF_ARGS = {
    'L': 16,
    'hidden_sizes': [256, 256, 256, 256]
}

def load_neural_sdf(path: Path, dim: int, sdf_args: dict = DEFAULT_SDF_ARGS):
    state_dict= torch.load(path)
    sdf_net = SdfNet(dim, **sdf_args)
    result = sdf_net.load_state_dict(state_dict, strict=False)
    if len(result.missing_keys): print(f"Warning: Missing keys in model!")
    if len(result.unexpected_keys): print(f"Warning: Unexpected keys in model!")
    return sdf_net

def benchmark_single_call(fn, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    has_failed = False
    with torch.no_grad():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            has_failed = True
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_time.elapsed_time(end_time)
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MiB

    return elapsed_time_ms, peak_memory_mb, not has_failed

def call_script(script_path: Path, *args, **kwargs):
    import subprocess
    command = ["python", str(script_path)] + list(map(str, args)) + [f"--{key}={value}" for key, value in kwargs.items()]
    
    # Start the subprocess and pipe the output to the console
    proc = subprocess.Popen(command, 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1) # line buffered

    # for line in proc.stdout:
    #     print(line, end="")      # Echo to console immediately

    proc.wait()  # Wait for the process to complete

    if proc.returncode != 0:
        print(f"Error calling script {script_path}: {proc.stderr} {proc.stdout.read()}")

def get_machine_specs():
    import platform
    import subprocess
    os_name = platform.system()
    if platform.system() in ["Linux", "Darwin"]:
        cpu_name = subprocess.getoutput("lscpu | grep 'Model name' | awk -F: '{print $2}'").strip()
    else:
        cpu_name = subprocess.getoutput("wmic cpu get name").strip().split("\n")[1].strip()
    gpu_name = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").strip()
    return os_name, cpu_name, gpu_name

def set_figure_title_with_specs(fig: plt.Figure, benchmark_name: str):
    os_info, cpu_info, gpu_info = get_machine_specs()
    fig.suptitle(f"Benchmark: {benchmark_name}\n" + f"OS: {os_info}    CPU: {cpu_info}    GPU: {gpu_info}", fontsize=12)

def plot_bar_chart(ax: plt.Axes, df: pd.DataFrame, metric: str, metric_label: str, method_to_label: dict, method_key: str = "method", reference_method: str = None, y_quantile: float = 0.95, y_scale: str = "linear", legend: bool = True, x_label: bool = True, ignored_methods: list = []):
    cycler = plt.style.library["seaborn-v0_8"]['axes.prop_cycle']
    colors = cycler.by_key()['color']

    # Width of one bar in log10 units
    bar_width = 0.31

    all_methods = list(df[method_key].unique())
    methods = [method for method in all_methods if method not in ignored_methods]
    
    # Determine the upper bound
    y_upper_bound = df[f"{metric}_avg"].quantile(y_quantile) # Quantile is robust against outliers
    y_map_to_linear = lambda x: np.log10(x) if y_scale == "log" else x

    background_color = '#EAEAF2' 
    ax.set_facecolor(background_color)
    for method in methods:
        if method in ignored_methods:
            continue

        # Get the method's data (copy because it is modified in-place)
        df_method = df[df[method_key] == method].copy()

        # Determine NaN entries and set metric to 1 for plotting (bar remains still visible)
        valid = ~df_method[f"{metric}_avg"].isna()
        df_method[f"{metric}_avg"] = df_method[f"{metric}_avg"].fillna(1)

        offset = (methods.index(method) + 0.5 - len(methods) / 2) * bar_width  # Offset for each method to avoid overlap

        logx = np.log10(df_method["num_rays"])
        left = 10**(logx + offset - bar_width/2)
        right = 10**(logx + offset + bar_width/2)

        color = mpl.colors.to_rgb(colors[methods.index(method) % len(colors)])
        edgecolor = 'black'
        hatch = None
        if method == "ling_pytorch":
            # Get the same color as ling but pre-multiplied alpha to make it less saturated
            color = mpl.colors.to_rgb(colors[all_methods.index("ling") % len(colors)])
            # Pre-multiplied alpha (0.5) to make it less saturated
            alpha = 0.5
            color = alpha * np.array(color) + (1-alpha) * np.array(mpl.colors.to_rgba(background_color)[:3])
            hatch = '///'
            edgecolor = (0.4, 0.4, 0.4) 

        ax.bar(left[valid], df_method[f"{metric}_avg"][valid], width=(right-left)[valid], align="edge", capsize=5, label=method_to_label[method], color=color, edgecolor=edgecolor, hatch=hatch)

        if (~valid).any():
            # Plot NaN bars in gray
            ax.bar(left[~valid], df_method[f"{metric}_avg"][~valid], width=(right-left)[~valid], align="edge", capsize=5, label=None, color='lightgray', edgecolor='gray')

        # Plot the value and the improvement over a reference method on top of the bar
        if reference_method is not None and reference_method in methods:
            df_reference = df[df[method_key] == reference_method]
            for x, y_method, y_reference, valid_method in zip((0.58*left + 0.42*right), df_method[f"{metric}_avg"], df_reference[f"{metric}_avg"], valid):
                label_str = (f"{y_method:.2f}" if y_method < 100 else f"{y_method:.1f}" if y_method < 1000 else f"{y_method:.0f}") if valid_method else "N/A"

                if valid_method and not np.isnan(y_reference):
                    factor = y_method / y_reference
                    factor_str = f"{factor:.2f}x" if factor < 1 else (f"{factor:.1f}x" if factor < 10 else f"{factor:.0f}x")
                    label_str += f"\n({factor_str})"

                y_text_anchor = y_method
                text_va = 'bottom'
                text_color = 'black'
                path_effects = None

                # If the text is out of bounds, move it inside the plot
                is_out_of_bounds = y_map_to_linear(y_method) > y_map_to_linear(y_upper_bound) * 0.95
                if is_out_of_bounds:
                    label_str = "↑\n" + label_str
                    y_text_anchor = min(y_upper_bound, y_method)
                    text_va = 'top'

                    # Choose white if the color is visible (dark), otherwise black
                    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                    text_color = 'white' if luminance < 0.7 else 'black'
                    
                    # Increase text visibility on hatch
                    if hatch is not None:
                        path_effects = [mpe.withStroke(linewidth=3, foreground=color)]

                ax.text(x, y_text_anchor, label_str, ha='center', va=text_va, fontsize=7.5, rotation=0, color=text_color, path_effects=path_effects)

    ax.set_xscale("log")
    ax.set_yscale(y_scale)
    if legend:
        ax.legend(loc="upper left")
    if x_label:
        ax.set_xlabel("Number of rays", fontsize=12)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_ybound(upper=y_upper_bound)
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statannotations.Annotator import Annotator
import ramanspy as rp
from pathlib import Path 
from raman_helper import *

import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self, channels_in, channels_out, batch_norm):
        super(BasicConv, self).__init__()
        basic_conv = [nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True)]
        basic_conv.append(nn.PReLU())
        if batch_norm:
            basic_conv.append(nn.BatchNorm1d(channels_out))

        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)

class ResUNetConv(nn.Module):
    def __init__(self, num_convs, channels, batch_norm):
        super(ResUNetConv, self).__init__()
        unet_conv = []
        for _ in range(num_convs):
            unet_conv.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
            unet_conv.append(nn.PReLU())
            if batch_norm:
                unet_conv.append(nn.BatchNorm1d(channels))

        self.body = nn.Sequential(*unet_conv)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class UNetLinear(nn.Module):
    def __init__(self, repeats, channels_in, channels_out):
        super().__init__()
        modules = []
        for i in range(repeats):
            modules.append(nn.Linear(channels_in, channels_out))
            modules.append(nn.PReLU())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, num_convs, batch_norm):
        super(ResUNet, self).__init__()
        res_conv1 = [BasicConv(1, 64, batch_norm)]
        res_conv1.append(ResUNetConv(num_convs, 64, batch_norm))
        self.conv1 = nn.Sequential(*res_conv1)
        self.pool1 = nn.MaxPool1d(2)

        res_conv2 = [BasicConv(64, 128, batch_norm)]
        res_conv2.append(ResUNetConv(num_convs, 128, batch_norm))
        self.conv2 = nn.Sequential(*res_conv2)
        self.pool2 = nn.MaxPool1d(2)

        res_conv3 = [BasicConv(128, 256, batch_norm)]
        res_conv3.append(ResUNetConv(num_convs, 256, batch_norm))
        res_conv3.append(BasicConv(256, 128, batch_norm))
        self.conv3 = nn.Sequential(*res_conv3)
        self.up3 = nn.Upsample(scale_factor=2)

        res_conv4 = [BasicConv(256, 128, batch_norm)]
        res_conv4.append(ResUNetConv(num_convs, 128, batch_norm))
        res_conv4.append(BasicConv(128, 64, batch_norm))
        self.conv4 = nn.Sequential(*res_conv4)
        self.up4 = nn.Upsample(scale_factor=2)

        res_conv5 = [BasicConv(128, 64, batch_norm)]
        res_conv5.append(ResUNetConv(num_convs, 64, batch_norm))
        self.conv5 = nn.Sequential(*res_conv5)
        res_conv6 = [BasicConv(64, 1, batch_norm)]
        self.conv6 = nn.Sequential(*res_conv6)

        self.linear7 = UNetLinear(3, 500, 500)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)

        x2 = self.conv2(x1)
        x3 = self.pool1(x2)

        x3 = self.conv3(x3)
        x3 = self.up3(x3)

        x4 = torch.cat((x2, x3), dim=1)
        x4 = self.conv4(x4)
        x5 = self.up4(x4)

        x6 = torch.cat((x, x5), dim=1)
        x6 = self.conv5(x6)
        x7 = self.conv6(x6)

        out = self.linear7(x7)

        return out

SEED = 19

matplotlib.rc_file_defaults()
plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

METRICS = ['MSE', 'SAD', 'SID']
colors = list(plt.cm.get_cmap()(np.linspace(0, 1, 4)))

# Load pretrained model
net = ResUNet(3, False).float()
net.load_state_dict(torch.load(r"ResUNet.pt", map_location=torch.device('cpu')))

def nn_preprocesing(spectral_data, wavenumber_axis):
    flat_spectral_data = spectral_data.reshape(-1, spectral_data.shape[-1])

    output = net(torch.Tensor(flat_spectral_data).unsqueeze(1)).cpu().detach().numpy()
    output = np.squeeze(output)

    output = output.reshape(spectral_data.shape)

    return output, wavenumber_axis

nn_denoiser = rp.preprocessing.PreprocessingStep(nn_preprocesing)

def get_results(spectrum_to_denoise, target, denoiser):

    # Normalise input and output to 0-1
    spectrum_to_denoise = minmax.apply(spectrum_to_denoise)
    target = minmax.apply(target)

    output = denoiser.apply(spectrum_to_denoise)

    metrics_result = {metric: getattr(rp.metrics, metric)(output, target) for metric in METRICS}

    return output, metrics_result

baseliners = {
    'SG (2, 5)': rp.preprocessing.denoise.SavGol(window_length=5, polyorder=2),
    'SG (2, 7)': rp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
    'SG (2, 9)': rp.preprocessing.denoise.SavGol(window_length=9, polyorder=2),
    'SG (3, 5)': rp.preprocessing.denoise.SavGol(window_length=5, polyorder=3),
    'SG (3, 7)': rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    'SG (3, 9)': rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
}

minmax = rp.preprocessing.normalise.MinMax()

# e.g. use

def add_normal_noise(spectrum, std=0.15):
    spectrum = rp.preprocessing.normalise.MinMax().apply(spectrum)

    # add noise
    noise = np.random.normal(0, std, len(spectrum.spectral_data))
    noisy_spectrum = rp.Spectrum(spectrum.spectral_data + noise, spectrum.spectral_axis)

    return noisy_spectrum

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()

raman_visual = Raman_Data(path, 20, 20)
raman_slice = raman_visual.get_slice500(3)

raman_slice = raman_slice.flat

np.random.seed(SEED)

selected_index = np.random.randint(0, raman_slice.shape[0])
selected_target = raman_slice[selected_index]
selected_input = add_normal_noise(selected_target)

nn_results = get_results(selected_input, selected_target, nn_denoiser)[0]
baseline_results = get_results(selected_input, selected_target, baseliners['SG (3, 9)'])[0]

results = minmax.apply([selected_input, baseline_results, selected_target, nn_results])
labels = ['Input (data with noise)', 'Savitzky-Golay (3, 9)', 'Target (authentic data)', 'Neural network']

plt.figure(figsize=(10, 4), tight_layout=True)
ax = rp.plot.spectra(results, plot_type='single', ylabel='Normalised intensity', title='Transfer dataset')
ax.legend(labels)

plt.show()
"""
def show_results(nn_results_df, baseline_results_dfs):
    for metric in METRICS:
        plt.figure(figsize=(4, 6), tight_layout=True)

        bar_kwargs = {'linewidth': 2, 'zorder': 5}
        err_kwargs = {'zorder': 0, 'fmt': 'none', 'linewidth': 2, 'ecolor': 'k', 'capsize': 5}

        combined_df = pd.concat([nn_results_df[metric], *[df[metric] for df in baseline_results_dfs.values()]], axis=1,
                                ignore_index=True)
        combined_df.columns = ['NN'] + list(baseline_results_dfs.keys())

        # Plot
        means = combined_df.mean()
        stds = combined_df.std()
        labels = combined_df.columns

        sg_cmap = LinearSegmentedColormap.from_list('', [colors[1], [1, 1, 1, 1]])
        colors_to_use = list(sg_cmap(np.linspace(0, 1, len(baseliners.keys()) + 2)))[:-2]

        ax = plt.gca()
        ax.bar(labels, means, color=[colors[3]] + colors_to_use[::-1], **bar_kwargs)
        ax.errorbar(labels, means, yerr=[[0] * len(stds), stds], **err_kwargs)

        # Significance tests
        combined_df_ = combined_df.melt(var_name='Denoiser', value_name=metric)
        box_pairs = [('NN', base) for base in baseliners.keys()]
        annotator = Annotator(ax, box_pairs, data=combined_df_, x="Denoiser", y=metric)
        annotator.configure(test='Wilcoxon', text_format='star', loc='inside', comparisons_correction='fdr_bh')
        annotator.apply_and_annotate()

        ax.set_title(metric)
        plt.xticks(rotation=45, ha='right')
        plt.show()

transfer_baseline_results_dfs = {k: pd.DataFrame(columns=METRICS) for k in baseliners.keys()}
transfer_nn_results_df = pd.DataFrame(columns=METRICS)
for spectrum in raman_slice:
    spectrum_with_noise = add_normal_noise(spectrum)
    transfer_nn_results_df = pd.concat([transfer_nn_results_df, pd.DataFrame([get_results(spectrum_with_noise, spectrum, nn_denoiser)[1]])], ignore_index=True)

    for name, denoiser in baseliners.items():
        transfer_baseline_results_dfs[name] = pd.concat([transfer_baseline_results_dfs[name], pd.DataFrame([get_results(spectrum_with_noise, spectrum, denoiser)[1]])], ignore_index=True)

show_results(transfer_nn_results_df, transfer_baseline_results_dfs)
"""
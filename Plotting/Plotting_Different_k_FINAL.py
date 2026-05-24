import torch
import matplotlib.pyplot as plt

synthetics = ['12DAckley', '12DRosen', '12DStyTang']
titles = ['Ackley 12D', 'Rosenbrock 12D', 'Styblinski-Tang 12D']

k_list = [1, 5, 10, 20]
markers = ['X', 'o', '^', 's']
labels = ['k=1', 'k=5', 'k=10', 'k=20']

path = '/fs/ess/PAS2983/jontwt/BEACON/Continuous_SingleOutcome/Results/Different k/'

marker_interval = 7
text_size = 16
marker_size = 5
linewidth = 2
alpha = 0.3

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)

for idx, synthetic in enumerate(synthetics):

    ax = axes[idx]

    for k, marker, label in zip(k_list, markers, labels):

        cost = torch.load(
            f'{path}{synthetic}/{synthetic}_cost_list_BEACON_k{k}.pt'
        )

        coverage = torch.load(
            f'{path}{synthetic}/{synthetic}_coverage_list_BEACON_k{k}.pt'
        )

        coverage_mean = torch.mean(coverage, dim=0)
        coverage_std = torch.std(coverage, dim=0)
        cost_mean = torch.mean(cost, dim=0)

        ax.plot(
            cost_mean[::marker_interval],
            coverage_mean[::marker_interval],
            label=label,
            marker=marker,
            markersize=marker_size,
            linewidth=linewidth,
        )

        ax.fill_between(
            cost_mean,
            coverage_mean - coverage_std,
            coverage_mean + coverage_std,
            alpha=alpha,
        )

    ax.set_title(titles[idx], fontsize=text_size)
    ax.grid(alpha=0.5)

    ax.tick_params(axis='both', labelsize=text_size-2)

    ax.set_ylim(0.0, 1.05)

    if idx == 0:
        ax.set_ylabel('Reachability', fontsize=text_size)

    ax.set_xlabel('Number of evaluations', fontsize=text_size)

# shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.06),
    ncol=4,
    fontsize=16,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

plt.savefig('Different_k_1x3.png', dpi=300, bbox_inches='tight')
# plt.show()
import matplotlib.pyplot as plt


def main():
    baseline = 88.31

    dense_test_mean = [89.36, 89.66, 90.02, 89.61]
    dense_test_var = [0.14, 0.52, 0.82, 0.54]
    dense_test_unseen_mean = [88.93, 89.26, 89.63, 89.19]
    dense_test_unseen_var = [0.22, 0.51, 0.91, 0.52]

    sparse_test_mean = [88.94, 88.68, 88.68, 88.35]
    sparse_test_var = [0.59, 0.41, 0.75, 0.38]
    sparse_test_unseen_mean = [88.63, 88.27, 88.03, 88.14]
    sparse_test_unseen_var = [0.69, 0.44, 0.77, 0.37]

    x = [2, 4, 8, 16]

    plt.figure(figsize=(4, 3))
    plt.errorbar(x, dense_test_mean, yerr=dense_test_var, fmt='o-', capsize=3, label='Test')
    plt.errorbar(x, dense_test_unseen_mean, yerr=dense_test_unseen_var, fmt='o-', capsize=3, label='Unseen Test')
    plt.hlines(baseline, xmin=2, xmax=16, color='gray', linestyle='--', label='DPO')
    plt.grid(alpha=0.6, linestyle='-', linewidth=0.5)
    plt.xticks(x)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Ranking Accuracy', fontsize=12)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('./analysis/figures/dense.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(4, 3))
    plt.errorbar(x, sparse_test_mean, yerr=sparse_test_var, fmt='o-', capsize=3, label='Test')
    plt.errorbar(x, sparse_test_unseen_mean, yerr=sparse_test_unseen_var, fmt='o-', capsize=3, label='Unseen Test')
    plt.hlines(baseline, xmin=2, xmax=16, color='gray', linestyle='--', label='DPO')
    plt.grid(alpha=0.6, linestyle='-', linewidth=0.5)
    plt.xticks(x)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Ranking Accuracy', fontsize=12)

    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/figures/sparse.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()

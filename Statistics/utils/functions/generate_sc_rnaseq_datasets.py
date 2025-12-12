import numpy as np
from tqdm import tqdm


def generate_scrnaseq_data(
    n_genes=1000,
    n_cells=1500,
    params_sequencing_depth={"mean": 1, "sd": 0.01},
    params_gene_means={"shape_gamma": 0.28, "rate_gamma": 0.26},
    params_gene_dispersion={"mean_norm": 1.2, "sd_norm": 1.15},
    percent_bimodal=5,
):

    # Create gene-specific mean expressions and dispersions
    gene_means = np.random.gamma(
        params_gene_means["shape_gamma"], 1 / params_gene_means["rate_gamma"], n_genes
    )
    gene_means_2 = np.random.gamma(
        params_gene_means["shape_gamma"], 1 / params_gene_means["rate_gamma"], n_genes
    )
    gene_dispersion = 1 / (
        10
        ** np.random.normal(
            params_gene_dispersion["mean_norm"],
            params_gene_dispersion["sd_norm"],
            n_genes,
        )
    )

    # Get log-normally distributed total number of RNA molecules sequenced per cell
    mean_size = params_sequencing_depth["mean"]
    sd_size = params_sequencing_depth["sd"]
    sequencing_depth = np.round(
        10 ** np.random.normal(np.log10(mean_size), sd_size, n_cells)
    ).astype(int)

    n_bimodal = round(n_genes * percent_bimodal / 100)

    # Initialize counts matrix
    counts = np.zeros((n_cells, n_genes))

    # Initialize data structures to store important variables for each gene
    gene_data = {
        "Gene": np.arange(1, n_genes + 1),
        "GeneDispersion": np.zeros(n_genes),
        "GeneMean": np.zeros(n_genes),
        "GeneMean2": np.zeros(n_genes),
        "BimodalProb": np.zeros(n_genes),
        "TrueMuSequencingDepth": [None] * n_genes,
    }

    gene_sd = 0.4
    for gene in tqdm(range(n_genes)):
        # Simulate the true expression rate for each cell
        meanlog = np.log10(gene_means[gene]) - (gene_sd / 2)
        if gene < n_bimodal:
            # Create bimodal distribution for the first n_bimodal genes
            bimodal_prob = np.random.uniform(0.2, 0.8)
            meanlog_2 = np.log10(gene_means_2[gene]) - (gene_sd / 2)
            component = np.random.uniform(size=n_cells) < bimodal_prob
            true_mu = 10 ** np.where(
                component,
                np.random.normal(meanlog, gene_sd, n_cells),
                np.random.normal(meanlog_2, gene_sd, n_cells),
            )
            gene_data["GeneMean2"][gene] = gene_means_2[gene]
            gene_data["BimodalProb"][gene] = bimodal_prob
        else:
            # Create unimodal distribution for the remaining genes
            true_mu = 10 ** np.random.normal(meanlog, gene_sd, n_cells)

        # Simulate the observed read counts per cell as a negative binomial process
        # The expected count for each cell is proportional to the true expression rate and the sequencing depth
        # In numpy, negative_binomial uses n (number of successes) and p (probability)
        # R's rnbinom with size and mu: size = 1/dispersion, p = size/(size+mu)
        size = 1 / gene_dispersion[gene]
        mu = true_mu * sequencing_depth
        p = size / (size + mu)
        counts[:, gene] = np.random.negative_binomial(size, p)

        # Store important variables for the current gene
        gene_data["GeneMean"][gene] = gene_means[gene]
        gene_data["GeneDispersion"][gene] = gene_dispersion[gene]
        gene_data["TrueMuSequencingDepth"][gene] = true_mu * sequencing_depth

    # Normalize the expression counts per cell (plus pseudocount) and log-transform them
    counts_normalized = counts / sequencing_depth[:, np.newaxis] + 1e-4
    counts_lognormalized = np.log10(counts_normalized)

    # Create a data frame with important variables for each cell
    cell_data = {
        "Cell": np.arange(1, n_cells + 1),
        "SequencingDepth": sequencing_depth,
        "SumCounts": np.sum(counts, axis=1),
        "SumCountsNormalized": np.sum(counts_lognormalized, axis=1),
        "NonZeroCounts": np.sum(counts > 0, axis=1),
    }

    return {
        "counts": counts,
        "counts_lognormalized": counts_lognormalized,
        "cell_data": cell_data,
        "gene_data": gene_data,
    }

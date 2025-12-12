generate_scrnaseq_data <- function(
    n_genes = 1000,
    n_cells = 1500,
    params_sequencing_depth = list(mean = 1, sd = 0.01),
    params_gene_means = list(shape_gamma = 0.28, rate_gamma = 0.26),
    params_gene_dispersion = list(mean_norm = 1.2, sd_norm = 1.15),
    percent_bimodal = 5) {
    # Create gene-specific mean expressions and dispersions
    gene_means <- rgamma(n_genes, shape = params_gene_means$shape_gamma, params_gene_means$rate_gamma)
    gene_means_2 <- rgamma(n_genes, shape = params_gene_means$shape_gamma, params_gene_means$rate_gamma)
    gene_dispersion <- 1 / (10^rnorm(n_genes, params_gene_dispersion$mean_norm, params_gene_dispersion$sd_norm)) # Different dispersions per gene

    # Get log-normally distributed total number of RNA molecules sequenced per cell (library size or sequencing depth)
    # Log-normal guarantees non-negative counts and a tapering down in density for higher counts
    # We first set average library size and standard deviation
    mean_size <- params_sequencing_depth$mean
    sd_size <- params_sequencing_depth$sd
    sequencing_depth <- round(10^rnorm(n_cells, log10(mean_size), sd_size))

    n_bimodal <- round(n_genes * percent_bimodal / 100) # Calculate the number of bimodal genes

    # Initialize counts matrix
    counts <- matrix(0, nrow = n_cells, ncol = n_genes)

    # Initialize a data frame to store important variables for each gene
    gene_data <- data.frame(
        Gene = 1:n_genes,
        GeneDispersion = rep(0, n_genes),
        GeneMean = rep(0, n_genes),
        GeneMean2 = rep(0, n_genes),
        BimodalProb = rep(0, n_genes),
        TrueMuSequencingDepth = I(vector("list", n_genes))
    )

    gene_sd <- 0.4
    pb <- txtProgressBar(min = 0, max = n_genes, initial = 0)
    for (gene in 1:n_genes) {
        setTxtProgressBar(pb, gene) # Set a progress bar
        # Simulate the true expression rate for each cell
        meanlog <- log10(gene_means[gene]) - (gene_sd / 2)
        if (gene <= n_bimodal) {
            # Create bimodal distribution for the first n_bimodal genes
            bimodal_prob <- runif(1, 0.2, 0.8) # Draw the probability of each cell being from each bimodal from a uniform distribution
            meanlog_2 <- log10(gene_means_2[gene]) - (gene_sd / 2)
            true_mu <- 10^ifelse(runif(n_cells) < bimodal_prob,
                rnorm(n_cells, meanlog, gene_sd),
                rnorm(n_cells, meanlog_2, gene_sd)
            )
            gene_data$GeneMean2[gene] <- gene_means_2[gene]
            gene_data$BimodalProb[gene] <- bimodal_prob
        } else {
            # Create unimodal distribution for the remaining genes
            true_mu <- 10^rnorm(n_cells, meanlog, gene_sd)
        }

        # Simulate the observed read counts per cell as a negative binomial process
        # The expected count for each cell is proportional to the true expression rate and the sequencing depth
        counts[, gene] <- rnbinom(n_cells, size = 1 / gene_dispersion[gene], mu = true_mu * sequencing_depth)

        # Store important variables for the current gene
        gene_data$GeneMean[gene] <- gene_means[gene]
        gene_data$GeneDispersion[gene] <- gene_dispersion[gene]
        gene_data$TrueMuSequencingDepth[[gene]] <- true_mu * sequencing_depth
    }
    close(pb) # Close the progress bar

    # Normalize the expression counts per cell (plus pseudocount) and log-transform them
    counts_normalized <- counts / sequencing_depth + 1e-4
    counts_lognormalized <- log10(counts_normalized)

    # Create a data frame with important variables for each cell
    cell_data <- data.frame(
        Cell = 1:n_cells,
        SequencingDepth = sequencing_depth,
        SumCounts = rowSums(counts),
        SumCountsNormalized = rowSums(counts_lognormalized),
        NonZeroCounts = rowSums(counts > 0)
    )

    return(list(
        counts = counts,
        counts_lognormalized = counts_lognormalized,
        cell_data = cell_data,
        gene_data = gene_data
    ))
}
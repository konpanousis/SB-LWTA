# Nonparametric Bayesian Deep Networks with Local Competition
Code Implementation for Nonparametric Bayesian Deep Networks with Local Competition published in ICML 2019. 
The code provides functions for constructing networks with normal priors and posteriors over the network's weights alongside an IBP prior for component omission; variational approximation is used for training. The supported activations are: ReLU, MaxOut and Local Winner-Takes-All.

Abstract

Local competition among neighboring neurons is
a common procedure taking place in biological
systems. This finding has inspired research on
more biologically plausible deep networks that
comprise competing linear units, as opposed to
nonlinear units that do not entail any form of (lo-
cal) competition. This paper revisits this mod-
eling paradigm, with the aim of enabling infer-
ence of networks that retain high accuracy for the
least possible model complexity; this includes the
needed number of connections or locally compet-
ing sets of units, as well as the required floating-
point precision for storing the network parameters.
To this end, we leverage solid arguments from the
field of Bayesian nonparametrics. Specifically,
we introduce auxiliary discrete latent variables
representing which initial network components
are actually needed for modeling the data at hand,
and perform Bayesian inference over them. Then,
we impose appropriate stick-breaking priors over
the introduced discrete latent variables; these give
rise to a well-established structure inference mech-
anism. As we experimentally show using bench-
mark datasets, our approach yields networks with
less memory footprint than the state-of-the-art,
and with no compromises in predictive accuracy.

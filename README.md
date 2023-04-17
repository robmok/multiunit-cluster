# multiunit-cluster

Python code for "A multi-level account of hippocampal function from behavior to neurons".

## Software and packages

Python version: 3.8.2

Python packages:
- pytorch 1.7.1
- numpy
- matplotlib
- scipy
- pandas
- itertools
- imageio (optional: plotting gifs)

## Scripts

Models
- MultiUnitCluster.py - main model class for concept and spatial learning that is called by other functions
- MultiUnitClusterNBanks.py - main nbanks model class (2 banks)

Simulations, exploration
- shj-multiunit.py - basic script to run Shepard et al.'s (SHJ) 6 problems
- shj-single-problems-plot.py - run SHJ single problems, 3d plots and gifs
- shj-multiunit-bigsim.py - run SHJ big sim with hpc scale (same as shj-multiunit but large number of units)
- shj-multiunit-lesions.py - run SHJ problems with lesions and plotting
- shj-multiunit-noise.py - run SHJ problems with update/learning noise and plotting
- shj-multiunitnbank.py - nbanks - run SHJ problems
- shj-multiunitnbank-bigsim.py - big sim
- shj-multiunit-gridsearch.py - gridsearch for fitting SHJ behavioural data
- shj-multiunit-nbanks-gridsearch.py - grid search
- spatial-multiunit-k.py - run spatial simulations 

Plotting
- shj-single-problems-plot.py - sample script to run SHJ single problems and plotting
- gridsearch_analysis.py - fit, get best parameters and plot
- plot-spatial-multiunit-k.py - plot spatial results

Other
- scores.py - grid score computations, from Banino et al., 2018, Nature
- demos-doubleupd-catlearn.py - plotting demos of double update for figures
- plot-place-field-examples.py


# Data on OSF
To load up gridsearch results, download directory "muc-shj-gridsearch".

To load up and explore spatial results, download directory "muc-spatial-results"

To see some gifs of units learning over time in the concept learning (SHJ) problems and spatial problems, see directory "gifs".

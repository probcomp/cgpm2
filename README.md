# CGPM2

Minimal implementation of composable generative population models for Bayesian
synthesis of probabilistic programs.


### Installation

Please see the instructions in the [dockerfile](./docker/ubuntu1604). Aside from
the python dependencies listed in the dockerfile, this software depends on the
following repositories:

- [cgpm](https://github.com/probcomp/cgpm)
- [crosscat](https://github.com/probcomp/crosscat)

### Tests

Before making any push to master, please run the following command in the shell:

    $ ./check.sh



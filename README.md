# Green Tea

Green tea is an constraint optimizer based on SHAC [[1]](https://arxiv.org/abs/1805.10255). Its intended use is for
minimizing noisy, and slow evaluation functions, such as:

- Minimize the runtime of mixed integer solvers based on hyper parameter tuning
- Tuning of game-playing parameters for chess / go engines
- Neural architecture search

[1] Manoj Kumar, George E. Dahl, Vijay Vasudevan, Mohammad Norouzi, _Parallel Architecture and Hyperparameter Search via Successive Halving and Classification_, https://arxiv.org/abs/1805.10255

## Dependencies

- [Python 3.6](https://www.python.org/downloads/)
- [SciPy 1.2.1](https://www.scipy.org/) (and [SKLearn](https://scikit-learn.org/))
- [XGBoost 0.82](https://xgboost.readthedocs.io/)

## Usage

```bash
./green-tea.py < [control file]
```

### Control file syntax

The control files is a [YAML](https://yaml.org/) file containing directives of what executable to minimize, and what
parameters to minimize it over:

- `exec` (required) - Path to an executable that execute the process to be minimized. It will be provided a YAML file
  containing the parameter values from standard input.
- `params` (required) - The parameters to optimize over, each parameter has a few properties:
  - `type` (required) - `uniform`, `normal`, or `integer`.
  - `shape` (optional) - The shape of the parameter, for example `[2, 8]`.
  - `range` (required) - The `lower` (inclusive) and `upper` (inclusive) bound of the parameter. For `normal` parameters one can also set `mean` and `std`.
- `constraints` (optional) - The global constraints to apply, each constraint is a python expression that should return a boolean.

#### Example

```yaml
exec: ./examples/rosenbrock.py
params:
  x:
    type: uniform
    range:
      lower: -2.0
      upper: 2.0
  y:
    type: uniform
    range:
      lower: -2.0
      upper: 2.0
constraints:
  - x + y > -3
  - x + y <  3
```

## How it works

The algorithm used is a variation of _random sampling_, where after evaluation a _batch_ of random samples we
build a model that cut the _batch_ into two sets of _good_ and _bad_ samples. This model is then used to reject any
future random samples that would get classified as _bad_. This is reminiscent of the [cutting-plane method](https://en.wikipedia.org/wiki/Cutting-plane_method)
for linear programming, but since we are optimizing a black-box function each cut is a classification problem.

Simplified steps, for the full details see [the paper](https://arxiv.org/abs/1805.10255): 

1. Generate, and evaluate, a batch of _N_ random samples.
2. Classify each sample in the batch as _Good_ if it better than the median evaluation value of the batch, otherwise the
   sample is classified as _Bad_.
3. Train a model to predict whether a sample is _Good_ or _Bad_ based on the data-set generated in step 2.
4. Generate, and evaluate, a new batch of _N_ samples where each sample must be classified as _Good_ by all of the
   classifier(s) generated from step 3.
5. Repeat from step 2.

### Classifier Cut

The quality of the classifier used to reduce the feasible set after each batch is paramount to the quality of the final
solution. If the classifier is bad, then this method is, at best, no better than random sampling. Any classifier used
should have the following properties:

- Robust to over fitting for very small data-sets, a batch is normally 12 to 20 samples.
- Fast to evaluate, since we will be generating and reject a lot of random samples.
- Produce conservative classifications, since we want to avoid removing regions that has not been properly explored yet.

These properties limits us to relatively simple algorithms. Currently the following seems feasible:

1. Gradient Boosting Decision Trees
2. Logistic Regression - _This may produce too aggressive cuts, consider increasing `-p` for this_

#### Classification Classes

The percentage of samples that are classified as _Good_ and _Bad_ can be controlled with the `-p` argument, by default
is 50:

```bash
./green-tea.py -p 75 < [control file]
```

It may be useful to tune this parameter for certain problems where, for example the initial feasible set is very
large, the evaluation function looks like a parabolic function, or has many highs and valleys.

## TODO

- [x] Add support for constraints.
- [x] Add support for additional parameter types, for example `integer`.
- [ ] Investigate fuzzy classification to help with noisy evaluation functions.
- [ ] Optimize random sample generation by tightening bounds based on classifier fit.

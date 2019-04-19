# Green Tea

Green tea is an constraint optimizer based on SHAC [[1]](https://arxiv.org/abs/1805.10255). Its intended use is for
minimizing noisy, and slow evaluation functions, such as:

- Minimize the runtime of mixed integer solvers based on hyper parameters
- Tuning of game-playing parameters for chess / go engines
- Neural architecture search

[1] Manoj Kumar, George E. Dahl, Vijay Vasudevan, Mohammad Norouzi, _Parallel Architecture and Hyperparameter Search via Successive Halving and Classification_, https://arxiv.org/abs/1805.10255

## Dependencies

- [Python 3.6](https://www.python.org/downloads/)
- [SkLearn 0.21](https://scikit-learn.org/stable/)

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
  - `type` (required) - `uniform` or `real`
  - `range` (required) - The `lower` (inclusive) and `upper` (inclusive) bound of the parameter. For `normal` parameters one can also set `mean` and `std`.

#### Example

```yaml
exec: ./examples/rosenbrock.py
params:
  x:
    type: uniform
    range:
      lower: -10.0
      upper: 10.0
  y:
    type: uniform
    range:
      lower: -10.0
      upper: 10.0
```

## How it works

The algorithm used is a minor variation of _random sampling_, where instead of immediately evaluating our expensive
black-box function on the sample we pre-screen each generated sample using a model that has been trained to recognize
_good_ samples:

1. Generate, and evaluate, a population of _N_ random samples.
2. Classify each sample in the population as _Good_ if it better than the median evaluation value, otherwise the sample
   is classified as _Bad_.
3. Train a classifier to predict whether a sample is _Good_ or _Bad_ based on the dataset generated in step 2.
4. Generate, and evaluate, a new population of _N_ samples where each sample must be classified as _Good_ by the
   classifiers from step 3.
5. Repeat from step 2.

See [the paper](https://arxiv.org/abs/1805.10255) for full details.

## TODO

- [ ] Add support for constraints.
- [x] Add support for additional parameter types, for example `integer`.
- [ ] Investigate fuzzy classification to help with noisy evaluation functions.

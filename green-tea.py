#!/usr/bin/env python3
#
# Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import numpy as np
import sys
import yaml

from sklearn.ensemble import GradientBoostingClassifier
from subprocess import Popen, DEVNULL, PIPE


class FeatureGenerator:
    def __init__(self, config, range_):
        self.lower = range_.get('lower', -math.inf)
        self.upper = range_.get('upper',  math.inf)
        self.shape = config.get('shape', [1])
        self.flat_shape = int(np.prod(self.shape))


class FeatureUniformGenerator(FeatureGenerator):
    def __init__(self, config, range_):
        FeatureGenerator.__init__(self, config, range_)

        from scipy.stats import uniform
        self._generator = uniform(loc=self.lower, scale=self.upper - self.lower)

    def __call__(self, n=1):
        return self._generator.rvs(size=n * self.flat_shape).reshape([n] + self.shape)


class FeatureNormalGenerator(FeatureGenerator):
    def __init__(self, config, range_):
        FeatureGenerator.__init__(self, config, range_)

        self.mean = range_.get('mean', (self.upper + self.lower) / 2.0)
        self.std = range_.get('std', 1.0)

        if not math.isfinite(self.mean):
            self.mean = 0.0

        from scipy.stats import norm
        self._generator = norm(loc=self.mean, scale=self.std)

    def __call__(self, n=1):
        return self._generator.rvs(size=n * self.flat_shape).reshape([n] + self.shape)


class FeatureIntegerGenerator(FeatureGenerator):
    def __init__(self, config, range_):
        FeatureGenerator.__init__(self, config, range_)

    def __call__(self, n=1):
        return np.random.randint(self.lower, self.upper + 1, size=n * self.flat_shape).reshape([n] + self.shape)


class Feature:
    def __init__(self, name, config):
        self.name = name
        self.type = config['type'].lower()

        if self.type == 'uniform':
            self._generator = FeatureUniformGenerator(config, config['range'])
        elif self.type == 'normal':
            self._generator = FeatureNormalGenerator(config, config['range'])
        elif self.type == 'integer':
            self._generator = FeatureIntegerGenerator(config, config['range'])
        else:
            raise ValueError('Unsupported parameter type -- ' + self.type)

    @property
    def flat_shape(self):
        return self._generator.flat_shape

    def __call__(self, n=1):
        return self._generator(n=n)


class Constraint:
    def __init__(self, code):
        self._code = compile(code, code, 'eval', dont_inherit=True)

    def __call__(self, sample):
        return eval(self._code, {}, sample.todict())


class Sample:
    def __init__(self, features):
        self._features = features
        self._values = {}

    def todict(self):
        return self._values

    def tolist(self):
        return np.asarray(
            [element for value in self._values.values() for element in value],
            np.float32
        )

    def safe_dump(self, dump_to):
        def tolist_or_scalar(arr):
            if arr.size == 1:
                return float(arr)
            return arr.tolist()

        yaml.safe_dump(
            {key: tolist_or_scalar(value) for key, value in self._values.items()},
            dump_to,
            explicit_start=True
        )

    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, key, value):
        self._values[key] = value


class Problem:
    def __init__(self, config):
        self.exec_path = config['exec']
        self.constraints = list(map(
            lambda constraint: Constraint(constraint),
            config.get('constraints', [])
        ))
        self.features = list(map(
            lambda feature: Feature(feature[0], feature[1]),
            config['params'].items()
        ))

    def evaluate(self, sample):
        with Popen(self.exec_path, shell=True, encoding='utf8', stdin=PIPE, stdout=PIPE, stderr=PIPE) as program:
            sample.safe_dump(program.stdin)
            stdout, stderr = program.communicate()

            try:
                return float(stdout)
            except ValueError:
                print('Could not parse problem output.', file=sys.stderr)
                print('--- Output ---', file=sys.stderr)
                print(stdout, file=sys.stderr)
                print('--- Error ---', file=sys.stderr)
                print(stderr, file=sys.stderr)
                exit(1)

    def sample(self, up_to_n=1):
        num_features = len(self.features)
        samples = list([Sample(self.features) for _ in range(up_to_n)])

        for i in range(num_features):
            samples_i = self.features[i](n=up_to_n)
            for j in range(up_to_n):
                samples[j][self.features[i].name] = samples_i[j, :]

        # prune samples that does not satisfy all constraints
        for c in self.constraints:
            samples = list([s for s in samples if c(s)])

        return samples


def generate_sample(problem, trained_classifiers):
    while True:
        samples = problem.sample(up_to_n=64)

        # prune samples that any classifier considers bad in reverse order since more
        # recent classifiers _should_ be more strict, and therefore fail faster.
        for c in reversed(trained_classifiers):
            p = c.predict([s.tolist() for s in samples])
            samples = list([samples[i] for i, p_ in enumerate(p) if p_])

            if not samples:
                break  # early exit

        # if there are any samples training after pruning then they passed training
        if samples:
            return samples[0]


def fit_classifier(points, values):
    def fit_one_classifier(x, y):
        c = GradientBoostingClassifier(n_estimators=200)
        try:
            c.fit(x, y)
            return c
        except ValueError:
            return None

    num_folds = 5
    num_samples = len(points)
    fold_size = num_samples // num_folds
    if fold_size == 0:
        num_folds = 1

    classifiers = []
    classifier_scores = []

    # iterate over each fold of the data, and train one classifier over all of the
    # training samples that are not in that fold. Then calculate the accuracy based
    # on the _training data_ in the current fold.
    points = np.asarray(points)
    values = np.asarray(values)

    fold_indices = np.arange(0, num_samples)
    np.random.shuffle(fold_indices)

    for i in range(num_folds):
        current_fold = np.arange(i * fold_size, (i + 1) * fold_size)
        classifier = fit_one_classifier(
            points[np.isin(fold_indices, current_fold, invert=True), :],
            values[np.isin(fold_indices, current_fold, invert=True)]
        )

        if classifier is not None:
            if current_fold.size > 0:
                classifier_score = classifier.score(
                    points[np.isin(fold_indices, current_fold), :],
                    values[np.isin(fold_indices, current_fold)]
                )
            else:
                classifier_score = classifier.score(points, values)

            classifiers.append(classifier)
            classifier_scores.append(classifier_score)

    # return the classifier with the best _validation score_
    try:
        best_classifier_index = max(range(num_folds), key=lambda j: classifier_scores[j])

        return classifiers[best_classifier_index], classifier_scores[best_classifier_index]
    except IndexError:
        return None, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='total_sample_budget', type=int)
    parser.add_argument('-b', dest='num_batches', type=int)

    args = parser.parse_args()
    problem = Problem(yaml.safe_load(sys.stdin))
    total_sample_budget = args.total_sample_budget or 200
    num_batches = args.num_batches or 18
    classifier_budget = math.ceil(total_sample_budget / num_batches)

    trained_classifiers = []
    batch_points = []
    batch_values = []

    global_min_point = None
    global_min_value = math.inf

    try:
        for t in range(total_sample_budget):
            x = generate_sample(problem, trained_classifiers)
            y = problem.evaluate(x)

            batch_points.append(x)
            batch_values.append(y)

            global_min_value = np.min([global_min_value, y])
            if global_min_value == y:
                global_min_point = x
                global_min_point.safe_dump(sys.stdout)

            # train and add one additional classifier if we have reaches the threshold
            is_last_sample = t == (total_sample_budget - 1)

            if len(batch_points) >= classifier_budget or is_last_sample:
                y_median = np.median(batch_values)
                y_min = np.min(batch_values)

                classifier, classifier_score = fit_classifier(
                    list(map(lambda x: x.tolist(), batch_points)),
                    list(map(lambda y: 1 if y < y_median else 0, batch_values))
                )
                accepted_classifier = classifier_score >= 0.51

                if accepted_classifier:
                    trained_classifiers.append(classifier)
                    batch_points.clear()
                    batch_values.clear()

                print(
                    '{:5} -- global_min {:.6e}, local_min {:.6e}, local_median {:.6e}, accuracy {:.3f} ({})'.format(
                        t + 1,
                        global_min_value,
                        y_min,
                        y_median,
                        classifier_score,
                        'ok' if accepted_classifier else 'no'
                    ),
                    file=sys.stderr
                )
    finally:
        if global_min_point:
            global_min_point.safe_dump(sys.stdout)


if __name__ == '__main__':
    main()

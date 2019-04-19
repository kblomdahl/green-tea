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

import yaml
import sys
import math
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from subprocess import Popen, DEVNULL, PIPE


class FeatureGenerator:
    def __init__(self, config):
        self.lower = config.get('lower', -math.inf)
        self.upper = config.get('upper',  math.inf)


class FeatureUniformGenerator(FeatureGenerator):
    def __init__(self, config):
        FeatureGenerator.__init__(self, config)

        from scipy.stats import uniform
        self._generator = uniform(loc=self.lower, scale=self.upper - self.lower)

    def __call__(self, n=1):
        return self._generator.rvs(size=n)


class FeatureNormalGenerator(FeatureGenerator):
    def __init__(self, config):
        FeatureGenerator.__init__(self, config)

        self.mean = config.get('mean', (self.upper + self.lower) / 2.0)
        self.std = config.get('std', 1.0)

        if not math.isfinite(self.mean):
            self.mean = 0.0

        from scipy.stats import norm
        self._generator = norm(loc=self.mean, scale=self.std)

    def __call__(self, n=1):
        return self._generator.rvs(size=n)


class FeatureIntegerGenerator(FeatureGenerator):
    def __init__(self, config):
        FeatureGenerator.__init__(self, config)

    def __call__(self, n=1):
        return np.random.randint(self.lower, self.upper + 1, size=n)


class Feature:
    def __init__(self, name, config):
        self.name = name
        self.type = config['type'].lower()

        if self.type == 'uniform':
            self._generator = FeatureUniformGenerator(config['range'])
        elif self.type == 'normal':
            self._generator = FeatureNormalGenerator(config['range'])
        elif self.type == 'integer':
            self._generator = FeatureIntegerGenerator(config['range'])
        else:
            raise ValueError('Unsupported parameter type -- ' + self.type)

    def __call__(self, n=1):
        return self._generator(n=n)


class Problem:
    def __init__(self, config):
        self.exec_path = config['exec']
        self.features = list(map(
            lambda feature: Feature(feature[0], feature[1]),
            config['params'].items()
        ))

    def dump_sample(self, x, dump_to):
        yaml.safe_dump(
            {feature.name: float(x[i]) for i, feature in enumerate(self.features)},
            dump_to
        )

    def evaluate(self, sample):
        with Popen(self.exec_path, shell=True, encoding='utf8', stdin=PIPE, stdout=PIPE, stderr=DEVNULL) as program:
            self.dump_sample(sample, program.stdin)
            stdout, _ = program.communicate()

            return float(stdout)

    def sample(self, n=1):
        k = len(self.features)
        samples = np.zeros((n, k), np.float32)

        for i in range(k):
            samples[:, i] = self.features[i](n=n)
        return samples


def generate_sample(problem, trained_classifiers):
    while True:
        samples = problem.sample(n=32)

        for c in trained_classifiers:
            p = c.predict(samples)
            samples = samples[p > 0, :]

            if samples.size == 0:
                break  # early exit

        # if there are any samples training after pruning then they passed training
        if samples.size > 0:
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
    problem = Problem(yaml.safe_load(sys.stdin))
    total_sample_budget = 200
    classifier_budget = math.ceil(total_sample_budget / 18)

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

                # train and add one additional classifier if we have reaches the threshold
            is_last_sample = t == (total_sample_budget - 1)

            if len(batch_points) >= classifier_budget or is_last_sample:
                y_median = np.median(batch_values)
                y_min = np.min(batch_values)

                classifier, classifier_score = fit_classifier(
                    batch_points,
                    list(map(lambda y: 1 if y < y_median else 0, batch_values))
                )
                accepted_classifier = classifier_score >= 0.51

                if accepted_classifier:
                    trained_classifiers.append(classifier)
                    batch_points.clear()
                    batch_values.clear()

                print('{: 4} -- global_min {:.6e}, local_min {:.6e}, local_median {:.6e}, accuracy {:.3f} ({})'.format(
                    t + 1,
                    global_min_value,
                    y_min,
                    y_median,
                    classifier_score,
                    'ok' if accepted_classifier else 'no',
                    file=sys.stderr
                ))
    finally:
        problem.dump_sample(global_min_point, sys.stdout)


if __name__ == '__main__':
    main()

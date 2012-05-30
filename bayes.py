#!/usr/bin/python
"""Naive Bayes Classifier
Author: Advait Shinde (advait.shinde@gmail.com)
"""

from __future__ import division  # Always use float division
import re
import sys
from collections import defaultdict
from operator import mul
from string import ascii_uppercase


class Feature(object):
  """A Feature represents a boolean assessment of an input string.
  The judge(s) method determines whether this feature is present in s.

  This feature base class only tests for the presence of a substring in s.
  """
  def __init__(self, base):
    self.base = sanitize(base)
    # Counts is a 2-level dict that keeps a count of our training strings.
    # The first level index is the class_number.
    # The second level index is a bool indicating the presence of this feature.
    # The values are initialized to one for smoothing
    self.counts = defaultdict(lambda: {True: 1, False: 1})

  def judge(self, target):
    """Judges the target against base.
    Override this method for more complicated Features (regex, etc.).

    Args:
      target: String the test string to judge against.
    Returns:
      Returns True iff base is a substring within target.
    """
    return self.base in target

  def train(self, target, class_number):
    """Trains the system against the target string and class_number.

    Args:
      target: String the training string to train with.
      class_number: Int the authoritative class the target is a part of.
    """
    presence = self.judge(target)
    self.counts[class_number][presence] += 1

  def test(self, target, class_number):
    """Determines the probability of this feature's presence for the class.

    Returns:
      Float probability that the feature is present for the class_number.
    """
    presence = self.judge(target)
    return (self.counts[class_number][presence] /
      (self.counts[class_number][False] + self.counts[class_number][True]))


class RegexFeature(Feature):
  """Regex-based Features."""
  def __init__(self, r):
    self.r = re.compile(r, re.IGNORECASE)
    super(RegexFeature, self).__init__(r)

  def judge(self, target):
    return (bool)(self.r.match(target))


class Classifier(object):
  """Naive Bayes Classifer.

  Useful methods:
    addFeature(f): Adds a feature f to the Classifier.
    train(s, c): Trains the classifier according to string s and class c.
    classify(s): Classifies string s in a class based on MAP.
    test(s, c): Returns the probability string s belongs to class c.
  """
  def __init__(self):
    self.features = []  # A list of all Feature objects
    self.total_classes = 2
    self.class_counts = defaultdict(lambda: 0)

  def priorProbability(self, class_number):
    """Returns the prior probability of class_number."""
    return (self.class_counts[class_number] /
      (self.class_counts[0] + self.class_counts[1]))

  def likelihood(self, target, class_number):
    """Returns the likelihood of the target given the class_number.
    This is P(F1=f1 ... Fn=fn | C=class_number).
    """
    likelihoods = [f.test(target, class_number) for f in self.features]
    return reduce(mul, likelihoods)  # Product of the list items

  def addFeature(self, f):
    """Adds a feature f to our Classifier."""
    self.features.append(f)

  def train(self, s, class_number):
    """Trains the classifier according to string s and class class_number.

    (1) Updates the counts that are used to compute the likelihoods.
    (2) Updates the counts that are used to compute the prior probabilities.
    """
    for f in self.features:
      f.train(s, class_number)            # (1)
    self.class_counts[class_number] += 1  # (2)

  def classify(self, s):
    """Returns the class_number that string s most likely belongs to."""
    probabilities = [self.test(s, c) for c in range(self.total_classes)]
    # Return the class_number/index of the largest probability
    return max(enumerate(probabilities), key=lambda x, y: cmp(x[1], y[1]))[0]

  def test(self, s, class_number):
    """Returns the probability s belongs to class_number."""
    classes = xrange(self.total_classes)
    # Prior probabilities
    # P(C=class_number)
    priors = [self.priorProbability(c) for c in classes]
    # Likelihoods
    # P(F1=f1 ... Fn=fn | C=classNumber)
    likelihoods = [self.likelihood(s, c) for c in classes]
    # Intermediate Probabilities
    # P(C=classNumber) * P(F1=f1 ... Fn=fn | C=classNumber)
    intermediates = [priors[c] * likelihoods[i] for i in classes]
    # Posterior Probability
    # P(C=classNumber | F1=f1 and F2=f2 and ... Fn=fn)
    return intermediates[class_number] / sum(intermediates)


def sanitize(s):
  """Sanitize input string s and return it."""
  _ret = s.encode('ascii', 'ignore')  # Remove unicode characters
  _ret = ''.join(_ret.split())  # Remove whitespace
  _ret = _ret.upper()  # Capitalize
  return _ret


def main():
  if len(sys.argv) != 5:
    print """\nUsage:
    $ bayes.py class1-trainfile class2-trainfile class1-testfile class2-testfile
    """
    return -1
  trainFile1, trainFile2 = sys.argv[1:3]
  testFile1, testFile2 = sys.argv[3:5]

  # Populate Classifier
  classifier = Classifier()
  # Add features for all uppercase characters
  for c in ascii_uppercase:
    classifier.addFeature(Feature(c))
  # Add features for common english digrams
  digrams = ["th", "he", "in", "en", "nt", "re", "er", "an",
      "ti", "es", "on", "at"]
  for digram in digrams:
    classifier.addFeature(Feature(digram))
  # Add features for common english trigrams
  trigrams = ["the", "and", "tha", "ent", "ing", "ion", "tio", "for", "nde",
      "has", "nce", "edt", "tis", "oft", "sth", "men"]
  for trigram in trigrams:
    classifier.addFeature(Feature(trigram))
  # Add some regex features
  regexes = [r"^.*ville$", r"^.*sk$"]
  for r in regexes:
    classifier.addFeature(RegexFeature(r))

  # Train!
  with open(trainFile1, 'r') as f:
    for line in f:
      classifier.train(sanitize(line), 0)  # Train class_number=0
  with open(trainFile2, 'r') as f:
    for line in f:
      classifier.train(sanitize(line), 1)  # Train class_number=1

  # Test!
  results = []
  with open(testFile1, 'r') as f:
    actual_class_number = 0
    for line in f:
      target = sanitize(line)
      probability = classifier.test(target, 0)
      predicted_class_number = (int)(probability <= 0.5)
      results.append((target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number))
  with open(testFile2, 'r') as f:
    actual_class_number = 1
    for line in f:
      target = sanitize(line)
      probability = classifier.test(target, 0)
      predicted_class_number = (int)(probability <= 0.5)
      results.append((target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number))

  # Print results
  print ""
  print "string                          true class  pred.class   postr C=1  correct?"
  print "------                          ----------  ----------   ---------  --------"
  print ""
  for r in results:
    if len(r[0]) >= 40:
      r[0] = r[0][:36] + "... "
    sys.stdout.write(r[0])
    sys.stdout.write(' ' * (41 - len(r[0])))  # Print spaces
    sys.stdout.write("{:d}".format(1 - r[1]))
    sys.stdout.write(' ' * 11)
    sys.stdout.write("{:d}".format(1 - r[2]))
    sys.stdout.write(' ' * 4)
    sys.stdout.write("{0:.6f}".format(r[3]))
    sys.stdout.write(' ' * 9)
    if r[4]:
      sys.stdout.write('Y')
    else:
      sys.stdout.write('N')
    sys.stdout.write('\n')

  # Compute and print accuracy and mean squared error
  n_results = len(results)
  n_correct = len([r[4] for r in results if r[4]])
  accuracy = n_correct / n_results
  error = sum([((1 - r[1]) - r[3]) ** 2 for r in results]) / n_results
  print ""
  print "Summary of {:d} test cases, {:d} correct; accuracy = {:.2f}".format(n_results, n_correct, accuracy)
  print "Mean squared error: {:.6f}".format(error)


if __name__ == "__main__":
  main()

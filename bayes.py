#!/usr/bin/python
"""Naive Bayes Classifier
Author: Advait Shinde (advait.shinde@gmail.com)
"""

from __future__ import division  # Always use float division
import sys
from collections import defaultdict


class Feature:
  """A Feature represents a boolean assessment of an input string.
  The judge(s) method determines whether this feature is present in s.

  This feature base class only tests for the presence of a substring in s.
  """
  def __init__(self, base):
    self.base = base
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

  def test(self, presence, class_number):
    """Determines the probability of this feature's presence for the class.

    Returns:
      Float probability that the feature is present for the class_number.
    """
    return (self.counts[class_number][presence] /
      (self.counts[class_number][False] + self.counts[class_number][True]))


class Classifier:
  """Naive Bayes Classifer.

  Useful methods:
    addFeature(f): Adds a feature f to the Classifier.
    train(s, c): Trains the classifier according to string s and class c.
  """
  pass


def main():
  if len(sys.argv) != 5:
    print """\nUsage:
    $ bayes.py class1-trainfile class2-trainfile class1-testfile class2-testfile
    """
    return -1
  trainFile1, trainFile2 = sys.argv[1:3]
  testFile1, testFile2 = sys.argv[3:5]
  print trainFile1, trainFile2, testFile1, testFile2

if __name__ == "__main__":
  main()

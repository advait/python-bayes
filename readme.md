Naive Bayes Classifier
======================

Author: Advait Shinde (advait.shinde@gmail.com)

This is a Naive Bayes Classifier implementation in Python.
See http://en.wikipedia.org/wiki/Naive_Bayes_classifier for more information.

This is a supervised learning algorithm. Use the provided train methods to
supervisedly teach the classifier which strings belong in which "class". Use
the classify() or the test() methods to predict which class a test string
falls into.

Usage:
------
  $ bayes.py class1-trainfile class2-trainfile class1-testfile class2-testfile

  class1-trainfile: A new-line separated list of strings that fall into class 1
  class2-trainfile: A new-line separated list of strings that fall into class 2
  class1-testfile: A new-line separated list of strings that incidentally
      fall into class 1. We will predict these strings' class and use this
      information to generate an accuracy/error report.
  class2-testfile: A new-line separated list of strings that incidentally
      fall into class 2. We will predict these strings' class and use this
      information to generate an accuracy/error report.

Concepts:
---------

### Feature
A feature is a simple function whose input is a string and whose output is a
boolean. More specifically, a feature categorizes a string based on simple
criteria (does the string contain a specific character or substring? or does
the string match a given regular expression). Several features are used in
conjunction to determine which class a test string falls into.


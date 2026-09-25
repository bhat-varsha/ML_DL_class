"""
A Decision Tree makes predictions by asking a sequence of questions about the features.
So the model learns a set of if-else rules from the training data.

Root
 ↓
Internal nodes
 ↓
Branches
 ↓
Leaf

Root → first question/split
Node → a decision/question
Branch → outcome of a decision
Leaf → final prediction

tree searche for splits that reduces impurity 

to measure impurity we use :
1. Gini=1-i=1∑C​pi2​

"""
# Range Minimum Query with Segment Trees and MNIST input images

Problem: given a sequence input find the minimum value within a certain range

Input: [digit 0, digit 1, ..., digit L], query_low, query_high

Output: digit X (where X is the smallest digit between digit query_low and digit query_high)

Here, all digit inputs are MNIST images and query_low and query_high are int arguments. 
Output is a probability distribution over the 10 MNIST classes.
This code doesn't allow for tree updates but such an extension would be straight forward.

Trained model generalises to longer sequences perfectly once trained. See output.log for example output. 

Reproduce results with ```python3 main.py```

Inspired by Heiko Strathmann, et al. "Persistent Message Passing." ICLR 2021 Workshop on Geometrical and Topological Representation Learning.


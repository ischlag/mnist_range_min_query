# Range Minimum Query with Segment Trees and MNIST input images

Problem: given a sequence input find the minimum value within a certain range

Input: [digit 0, digit 1, ..., digit L], query_low, query_high

Output: digit X (where X is the smallest digit between digit query_low and digit query_high)

Here, all digit inputs are MNIST images and query_low and query_high are int arguments. 
Output is a probability distribution over the 10 MNIST classes.
This implementation doesn't come with an update to the tree but such an extension would be straight forward. 



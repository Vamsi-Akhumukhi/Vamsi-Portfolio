# 100. Same Tree

## Description
>Given the roots of two binary trees p and q, write a function to check if they are the same or not.

>Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
## Test cases


**Example 1**

    Input: p = [1,2,3] and q = [1,2,3]
    Output: true


**Example 2**

    Input: p = [1,2], q = [1,null,2]
    Output: false
## Solution:
We need to do recursion to check if two binary trees are structurally identical and nodes has the same value

We consider that we are allowed to visit Null nodes.
P and Q can have 3 cases:
1. p exist but q doesn't , return false
1. both doesn't exist, i.e both are same. Return true
1. if none of the first two conditions meet, we look at children and return True only if values of nodes are equal and if True holds for left and right subtrees.

![Default Case](https://raw.githubusercontent.com/Vamsi-Akhumukhi/LeetCode/master/100.%20Same%20Tree/Same%20tree%20-%202.jpg "Default Case")

![Edge Case](https://raw.githubusercontent.com/Vamsi-Akhumukhi/LeetCode/master/100.%20Same%20Tree/Same%20tree%20Edge%20Case.png.jpg "Edge Case")
## Code

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        #we are allowed to visit the empty nodes
        if p and not q or q and not p:  #if any one of them is empty
            return False
        if not p and not q:
            return True

        left = self.isSameTree(p.left,q.left)
        right = self.isSameTree(p.right,q.right)

        return p.val == q.val and left and right
```

**Time Complexity = O(N), we visit all the nodes in the worst case scenario**

**Space Complexity = O(H),H is the height of the tree**


# 1047. Remove All Adjacent Duplicates In String

## Description
> Given a string s containing only lowercase letters, continuously remove adjacent characters that are the same and return the result.
## Test cases
**Example 1**

    Input: s = "abbcccccaa"
    Output: "ca"

**Example 2**

    Input: s = "abccba"
    Output: ""

**Example 3**

    Input: s = "mbccbefddfe"
    Output: "m"
## Solution

 We start with a result stack, go through all the characters in the string S one by one.

 If the current character and the previous character in the result are the same. We pop them as they are adjacent duplicate pair.

 If the next character is different, append it to the end of the result

## Code

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:

        res = []
        for char in S:
            if res and res[-1] == char:
                res.pop()
            else:
                res.append(char)
        return "".join(res)


```
**Time Complexity = O(N)**

**Space Complexity = O(N)**


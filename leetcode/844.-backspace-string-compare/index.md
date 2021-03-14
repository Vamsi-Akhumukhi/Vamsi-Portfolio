# 844. Backspace String Compare

> Given two strings s and t, which represents a sequence of keystrokes, where # denotes a backspace, return whether or not the sequences produce the same result.

**Example 1**

    Input: s = "ABC#", t = "CD##AB"
    Output: true

**Example 2**

    Input: s = "como#pur#ter", t = "computer"
    Output: true

**Example 3**

    Input: "cof#dim#ng", t = "coding"
    Output: false

```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def build(S):
            res = []
            for char in S:
                if char != '#':
                    res.append(char)
                # pop only if the result is not empty
                elif res:
                    res.pop()

            return ''.join(res)
            #if we dont join the result will be in a list.
        return build(S) == build(T)

```
**Time Complexity = O(m+n), where m and n represents the length of the strings S and T respectively**

**Space Complexity = O(1)**


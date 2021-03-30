# 771. Jewels and Stones

## Description
> You're given strings **jewels** representing the types of stones that are jewels, and **stones** representing the stones you have. Each character in stones is a type of stone you have. You want to know how many of the stones you have are also jewels.

>Letters are case sensitive, so "a" is considered a different type of stone from "A".
## Test cases

**Example 1**

    Input: jewels = "aA", stones = "aAAbbbb"
    Output: 3


**Example 2**

    Input: jewels = "z", stones = "ZZ"
    Output: 0
## Solution 1: Two For loops

We check compare each character in Jewels in each character in Stones

## Code

```python
class Solution:
  res = []

      for i in range(len(jewels)):
          for j in range(len(stones)):
              if jewels[i] == stones[j]:
                  res.append(jewels[i])
```
solution works. But, the problem with this approach is it's very time consuming
i.e O(J*S)
**Time Complexity = O(J*S)**

**Space Complexity = O(J)**

So, we try another approach using Dictionaries

## Solution 2 : Using Dictionaries
We create a <segment>Dictionary</segment> to store each character of Jewels.
> Dictionary = {'a','A'}

Then, we compare each character of <segment>stones</segment> with the characters present in the <segment>Dictionary</segment>
Increment the counter whenever there is a match and return the counter

```python
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:

        dictionary = set(jewels)

        count = 0
        for char in stones:
            if char in dictionary:
                count = count + 1
        return count
```
With this approach we reduced the time Complexity to
**Time Complexity = O(J+S)**

**Space Complexity = O(J)**


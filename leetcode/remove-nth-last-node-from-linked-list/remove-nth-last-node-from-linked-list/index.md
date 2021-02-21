# 19. Remove Nth Node From End of List

>Given the head of a linked list, remove the nth node from the end of the list and return its head.

>**Follow up**: Could you do this in one pass?




![Removing nth node from end in Linked List](https://raw.githubusercontent.com/Vamsi-Akhumukhi/LeetCode/master/Remove%20nth%20last%20node%20from%20Linked%20List-02.jpg)

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/Vamsi-Akhumukhi/LeetCode/master/Remove%20nth%20last%20node%20from%20Linked%20List-02.jpg">
</p>

**Example 1**

    Input: head = [1,2,3,4,5,6], n = 2
    Output: [1,2,3,4,6]

**Example 2**

    Input: head = [2], n = 1
    Output: []

**Example 3**

    Input: head = [1,2,3], n = 1
    Output: [1,2]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        fast = slow = head

        for i in range(n):
            fast = fast.next

        if not fast:
            return head.next

        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```
**Time Complexity = O(m+n), where m and n represents the length of the list L1 and L2 respectively**

**Space Complexity = O(1)**


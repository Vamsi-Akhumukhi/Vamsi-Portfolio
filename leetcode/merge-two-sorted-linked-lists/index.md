# 21. Merge Two Sorted Lists

>Merge two sorted linked lists and return it as a **sorted** list. The list should be made by splicing together the nodes of the first two lists.

**Example 1**

    Input: l1 = [-1,2,4], l2 = [1,3,4]
    Output: [-1,1,2,3,4,4]

**Example 2**

    Input: l1 = [-1,4,5,7,8,9,10], l2 = [1,3,4]
    Output: [-1,1,3,4,4,5,7,8,9,10]

**Example 3**

    Input: l1 = [], l2 = [1,3,4]
    Output: [1,3,4]

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        dummy = curr = ListNode(0)
        #current always points to the tail of the list

        if l1:
            curr.next = l1
        elif l2:
            curr.next = l2

        # 2 edge cases where any of them could be null

        while l1 and l2:

            if l1.val < l2.val:
                # i need my curr to point to l1
                curr.next = l1

                #now that my curr is at l1, i will move l1 forward
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next

            curr = curr.next

            #when both are equal
            curr.next = l1 or l2

        return dummy.next
```
**Time Complexity = O(m+n), where m and n represents the length of the list L1 and L2 respectively**

**Space Complexity = O(1)**


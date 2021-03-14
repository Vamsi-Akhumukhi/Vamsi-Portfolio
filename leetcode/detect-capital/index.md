# 520. Detect Capital

>Given a word, you need to judge whether the usage of capitals in it is right or not. We define the usage of capitals in a word to be right when one of the following cases holds:

1. All letters in this word are capitals, like **"AMERICA".**
2. All letters in this word are not capitals, like **"vamsi".**
3. Only the first letter in this word is capital, like **"Predator".**

```python
class Solution:
    def detectCapitalUse(self, word: str) -> bool:

        return (word[0] == word[0].upper() and (word[1:] == word[1:].lower() or word[1:] == word[1:].upper())) or (word == word.lower())
```


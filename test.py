class Solution(object):
    def lengthOfLongestSubstring(self, s):
        start = 0
        current = 0
        list_length = []

        while current < len(s):
            count = 1
            is_double = False
            i = start # 0
            while i < current:
                if s[i] == s[current]:
                    is_double = True
                    break
                count += 1
                i += 1

            if is_double:
                list_length.append(count)
                start = i + 1

            current += 1

        return max(list_length)


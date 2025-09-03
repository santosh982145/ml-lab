# q1: Finding pairs with sum equal to tot
a = [2, 7, 4, 1, 3, 6]
tot = 10
c = 0
for i in range(len(a)):
    for j in range(i + 1, len(a)):
        if a[i] + a[j] == tot:
            c += 1
print("q1:", c)

# q2: Range of a list
def f_r(a):
    if len(a) < 3:
        return "Range determination not possible"
    return max(a) - min(a)

a2 = [5, 3, 8, 1, 0, 4]
print("q2:", f_r(a2))

# q3: Matrix multiplication power
def mp(a, b):
    # Fix: Use len(b) for cols of b and len(a[0]) for proper k
    return [
        [
            sum(
                a[i][k] * b[k][j]
                for k in range(len(a[0]))
            )
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]

def mm(a, m):
    # Identity matrix
    ans = [
        [int(i == j) for j in range(len(a))]
        for i in range(len(a))
    ]
    for _ in range(m):
        ans = mp(ans, a)
    return ans

a3 = [[1, 2], [3, 4]]
print("q3:", mm(a3, 2))

# q4: Most frequent character in text
def chance(text):
    f = {}
    for charcter in text:
        charcter_lower = charcter.lower()
        if 'a' <= charcter_lower <= 'z':
            f[charcter_lower] = f.get(charcter_lower, 0) + 1
    max_charcter = max(f, key=f.get)
    return max_charcter, f[max_charcter]

in_str = "hippopotamus"
charcter, count = chance(in_str)
print("q4:", charcter)
print("q4:", count)

# q5: Random list, mean, median, mode
import random
import statistics

n = [random.randint(1, 10) for _ in range(25)]
print("q5 n:", n)

mean = statistics.mean(n)
median = statistics.median(n)
try:
    mode = statistics.mode(n)
except statistics.StatisticsError:
    mode = "No unique mode"
print("q5 mean:", mean)
print("q5 median:", median)
print("q5 mode:", mode)

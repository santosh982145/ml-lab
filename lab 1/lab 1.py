q1
a = [2, 7, 4, 1, 3, 6]
tot = 10
c = 0

for i in range(len(a)):
    for j in range(i + 1, len(a)):
        if a[i] + a[j] == tot:
            c += 1

print(c)

q2
def f_r(a):
    if len(a) < 3:
        return "Range determination not possible"
    return max(a) - min(a)

a = [5, 3, 8, 1, 0, 4]
print(f_r(a))

q3
def mp(a, b):
    return [
        [
            sum(
                a[i][k] * b[k][j]
                for k in range(len(a))
            )
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]

def mm(a, m):
    ans = [
        [int(i == j) for j in range(len(a))]
        for i in range(len(a))
    ]  # identity matrix
    for _ in range(m):
        ans = mp(ans, a)
    return ans

a = [[1, 2], [3, 4]]
print(mm(a, 2))

q4
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
print(charcter)
print(count)

q5
import random
import statistics

n = [random.randint(1, 10) for _ in range(25)]

print("n:", n)

mean = statistics.mean(n)
median = statistics.median(n)
mode = statistics.mode(n)

print(mean)
print(median)
print(mode)
x = [[2,2] for i in range(30)]

t = [[1,1] for _ in range(5)]

x[:len(t)] = t
print(x)
from sklearn.metrics.pairwise import cosine_similarity

x = [[0.1,0.9]]
y = [[0.3,0.5]  ]
ret =cosine_similarity(x, y)
print(ret)
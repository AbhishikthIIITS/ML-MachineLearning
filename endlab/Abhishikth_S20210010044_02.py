import numpy as np
import matplotlib.pyplot as plt

print("Name :- Abhishikth Boda")
print("Roll Number :- S20210010044")
print("Date :- 1st December 2023")
print("Machine Learning Lab Exam")
print("Set Number :- 02")

#function for generating the dataset
def generate_dataset(n):
    np.random.seed(42)  # For reproducibility
    m1, m2, m3, m4 = [0, 0], [10, 0], [0, 9], [9, 8]
    s1, s2, s3, s4 = np.eye(2), np.array([1, 0.2, 0.2, 1.5]).reshape(2, 2), \
                     np.array([1, 0.4, 0.4, 1.1]).reshape(2, 2), np.array([0.3, 0.2, 0.2, 0.5]).reshape(2, 2)
    

    #initializing a variable stack to create a dataset
    c_x = np.vstack([
        np.random.multivariate_normal(m1, s1, n // 4),
        np.random.multivariate_normal(m2, s2, n // 4),
        np.random.multivariate_normal(m3, s3, n // 4),
        np.random.multivariate_normal(m4, s4, n // 4)
    ])

    return c_x

#function for calculating euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

#function for plotting clusters
#this function give detailed information for each attribute
def plot_clusters(data, centroids, k):
    plt.scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'K-Means Clustering (k={k})')
    plt.legend()
    plt.show()

#function for initializing centroids
#we are initializing the centroid randomly for first iteration
def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

#function for performing k-means clustering
def kmeans(data, k, max_iters=100):
    #here we added an variable max_iters. It means that if no of iterations cross a particular value,
    #but do not yield a result then there cluster would be the final one
    centroids = initialize_centroids(data, k)
    prev_centroids = centroids.copy()

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        #Here we are allocating every data point to its nearest cluster
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        #Here the clusters would be updated after every iteration is completed
        for i in range(k):
            if clusters[i]:
                centroids[i] = np.mean(clusters[i], axis=0)

        #Here we will check for the base condition of K means clustering algorithm
        #That is if for two consecutive iteration the clusters are same then we will conclude the algorithm
        #this is what the logic below implies and also give output
        if np.all(prev_centroids == centroids):
            break

        prev_centroids = centroids.copy()

    return np.array(centroids)

#Here we are calling the function to generate dataset c_x
n = 400
c_x = generate_dataset(n)

#Here we are applying k means algorithm for each value of starting from 2 ranging to 5
for k in range(2, 6):
    centroids = kmeans(c_x, k)
    plot_clusters(c_x, centroids, k)


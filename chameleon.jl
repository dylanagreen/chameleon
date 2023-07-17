using FileIO
using Images

k = 5

### K-Means ###

"""
    euclidean_distance(data::Matrix, centroids::Matrix)
Compute the Euclidean Distance between all points in the data matrix and
all the points in the centroids matrix.
"""
function euclidean_distance(data::Matrix, centroids::Matrix)
    c = transpose(centroids[1, :])
    dists = sqrt.(sum((data .- c) .^ 2, dims=2))
    for i in 2:size(centroids)[1]
        c = transpose(centroids[i, :])
        d = sqrt.(sum((data .- c) .^ 2, dims=2))
        dists = hcat(dists, d)
    end
    return dists
end

"""
    k_means_step(data::Matrix, K::Int)
Run a single (full) iteration of the K-means algorithm for finding `K` centroids
in the `data` matrix.

An iterations consists of:
- Find the category assignment of each point using Euclidean Distance
- Compute updated centroid locations as a mean of each category
- Repeat until convergence

Starting centroids are picked as `K` random points in the `data` matrix.
Once convergence is achieved, return the centroids, as well as each
points classification, their Euclidean distance to their classifying centroid
and finally the history of their Euclidean distance to their classifying centroid.
"""
function k_means_iter(data::Matrix, K::Int)
    # Pick two random rows as starting points
    start_idcs = rand(rng, 1:size(data)[1], K)
    centroids = data[start_idcs, :]

    # Current distance and category
    # We will terminate when no more elements change categories
    cur_dist = euclidean_distance(data, centroids)
    cur_cat = map(ele -> ele[2], argmin(cur_dist, dims=2))
    changes = ones(length(cur_cat))

    hist = [sum(cur_dist .^2 )]

    while sum(changes) > 0
        centroids = mean(data[vec(cur_cat .== 1), :], dims=1)
        for i in 2:K
            c = mean(data[vec(cur_cat .== i), :], dims=1)
            centroids = vcat(centroids, c)
        end

        new_dist = euclidean_distance(data, centroids)
        new_cat = map(ele -> ele[2], argmin(new_dist, dims=2))

        changes = cur_cat .!= new_cat

        # Saving these to return them and for use in next loop
        cur_cat = new_cat
        cur_dist = new_dist
        push!(hist, sum(cur_dist .^ 2))

    end
    return centroids, cur_cat, cur_dist, hist

end

"""
    k_means(data::Matrix, K::Int, N::Int)
Run `N` iterations of the K-means algorithm, and return the one with
the lowest sum of Euclidiean distances for all points.

Once all iterations are complete, return the centroids, as well as each
points classification and the history of their Euclidean distance to
their classifying centroid.
"""
function k_means(data::Matrix, K::Int, N::Int)
    # N = number of times to "restart". Needs to be >= 1
    # Do once to get starting values
    centroids, best_cat, best_dist, best_hist = k_means_iter(data, K)

    # Loop N times, and if we get a better classification
    # save that one
    for i in 2:N
        c, cat, dist, hist = k_means_iter(data, K)

        # We will need to square the distances to correctly
        # compare the "sum of squares" score instead of the
        # "sum of euclidiean distance"
        if sum(dist .^ 2) < sum(best_dist .^2)
            centroids = c
            best_cat = cat
            best_dist = dist
            best_hist = hist
        end
    end

    # Don't need to return the distance here
    return centroids, best_cat, best_hist
end
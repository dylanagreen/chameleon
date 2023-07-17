using LinearAlgebra
using Random
using Statistics

using FileIO
using Images
using Distances

# """
#     euclidean_distance(data::Matrix, centroids::Matrix)
# Compute the Euclidean Distance between all points in the data matrix and
# all the points in the centroids matrix.
# """
# function euclidean_distance(data::Matrix, centroids::Matrix)
#     c = transpose(centroids[1, :])
#     dists = sqrt.(sum((data .- c) .^ 2, dims=2))

#     f = pairwise(euclidean, transpose(data), transpose(centroids), dims=2)
#     for i in 2:size(centroids)[1]
#         c = transpose(centroids[i, :])
#         d = sqrt.(sum((data .- c) .^ 2, dims=2))
#         dists = hcat(dists, d)
#     end

#     println(size(dists), dists[1:5, :])
#     println(size(f), f[1:5, :])
#     return dists
# end

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
function k_means_iter(data::Matrix, K::Int, max_iter::Int)
    # Pick two random rows as starting points
    start_idcs = rand(rng, 1:size(data)[1], K)
    centroids = data[start_idcs, :]

    # Current distance and category
    # We will terminate when no more elements change categories
    cur_dist = pairwise(euclidean, transpose(data), transpose(centroids), dims=2)
    cur_cat = map(ele -> ele[2], argmin(cur_dist, dims=2))
    changes = ones(length(cur_cat))

    # Counter acts as a hard stop
    counter = 0
    while (sum(changes) > 0) && (counter < max_iter)
        centroids = mean(data[vec(cur_cat .== 1), :], dims=1)
        for i in 2:K
            c = mean(data[vec(cur_cat .== i), :], dims=1)
            centroids = vcat(centroids, c)
        end

        new_dist = pairwise(euclidean, transpose(data), transpose(centroids), dims=2)
        new_cat = map(ele -> ele[2], argmin(new_dist, dims=2))

        changes = cur_cat .!= new_cat

        # Saving these to return them and for use in next loop
        cur_cat = new_cat
        cur_dist = new_dist
        counter += 1
    end
    return centroids, cur_cat, cur_dist

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
    max_iter = 50
    centroids, best_cat, best_dist = k_means_iter(data, K, max_iter)

    # Loop N times, and if we get a better classification
    # save that one
    for i in 2:N
        c, cat, dist = k_means_iter(data, K, max_iter)

        # We will need to square the distances to correctly
        # compare the "sum of squares" score instead of the
        # "sum of euclidiean distance"
        if sum(dist .^ 2) < sum(best_dist .^2)
            centroids = c
            best_cat = cat
            best_dist = dist
        end
    end
    return centroids, best_cat
end


# Set seed for reproducibility
rng = Xoshiro(91701)
K = 4

# Load the image and permute the dimensions to get it in (x, y, rgb) order
img_1 = FileIO.load("./dial_of_destiny.jpg")
img_1 = float64.(PermutedDimsArray(channelview(img_1), [2, 3, 1]))
img_1_shape = size(img_1)

# Reshape is necessary to turn this into a single list of 3-dimensional points
# we don't care about x/y position when determinig the color space centroids
img_1 = reshape(img_1, :, 3)

println("k means 1...")
centroids_1, _= k_means(img_1, K, 3)

img_2 = FileIO.load("./bullet_train.jpg")
img_2 = float64.(PermutedDimsArray(channelview(img_2), [2, 3, 1]))
img_2_shape = size(img_2)
img_2 = reshape(img_2, :, 3)

println("k means 2...")
centroids_2, cat_2 = k_means(img_2, K, 3)

old_centroids = dropdims(centroids_2[cat_2, :], dims=2)
final = img_2 .- old_centroids

new_centroids = dropdims(centroids_1[cat_2, :], dims=2)
final = final .+ new_centroids

println("saving...")
# Make sure we only have valid RGB colors
# There's probably a better way to handle this
clamp!(final, 0, 1.0)
final = PermutedDimsArray(reshape(final, img_2_shape), [3, 1, 2])
save("test.png", colorview(RGB, final))

using Random, CSV, DataFrames, Printf

# Abstract classifier 
abstract type AbstractClassifier end


# Structures
# Binary Tree
mutable struct BinaryTree
    children_left::Vector{Int}
    children_right::Vector{Int}
    BinaryTree() = new([], [])
end

# Decision Tree
mutable struct DecisionTreeClassifier{T} <: AbstractClassifier
    #internal variables
    num_nodes::Int
    binarytree::BinaryTree
    n_samples::Vector{Int} # total samples per each node
    values::Vector{Vector{Float64}} # samples per class per each node. Float64 to speed up calculations
    impurities::Vector{Float64}
    split_features::Vector{Union{Int, Nothing}}
    split_values::Vector{Union{T, Nothing}} #Note: T is the same for all values
    n_features::Union{Int, Nothing}
    n_classes::Union{Int, Nothing}
    features::Vector{String}
    feature_importances::Union{Vector{Float64}, Nothing}

    # external parameters
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing} # sets n_features_split
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}

    DecisionTreeClassifier{T}(;
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=Random.GLOBAL_RNG
        ) where T = new(
            0, BinaryTree(), [], [], [], [], [], nothing, nothing, [], nothing,
            max_depth, max_features, min_samples_leaf, check_random_state(random_state)
            )
end

# As with RandomForestClassifier, we can set the default type to Float64
DecisionTreeClassifier(; max_depth=nothing, max_features=nothing, min_samples_leaf=1, random_state=Random.GLOBAL_RNG
	) = DecisionTreeClassifier{Float64}(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf,
		random_state=check_random_state(random_state)
    )


# Functions
# Binary Tree functions
function add_node!(tree::BinaryTree)
    push!(tree.children_left, -1)
    push!(tree.children_right, -1)
    return
end

function set_left_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_left[node_id] = child_id
    return
end

function set_right_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_right[node_id] = child_id
    return
end

function get_children(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id], tree.children_right[node_id]
end

function is_leaf(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id] == tree.children_right[node_id] == -1
end

# Decision Tree functions
function fit!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame)
    @assert size(Y, 2) == 1 "Y must be a vector"
    @assert size(X, 1) == size(Y, 1) "X and Y must have the same number of rows"

    # Set initial variables
    tree.n_features = size(X, 2)
    tree.n_classes = size(unique(Y), 1)
    tree.features = names(X)

    # Fit
    split_node!(tree, X, Y, 0)

    # Set attributes
    tree.feature_importances = feature_importance_impurity(tree)
end


# Split node is where we get closer to the ML magic
# First some functions for split_node!
gini_score(counts) = 1 - sum([c*2 for c in counts])/(sum(counts)^2)

function count_classes(Y, n::Int)
    # This is the slow way
    counts = zeros(n)
    for entry in eachrow(Y)
        counts[entry[1]] += 1
    end
    return counts
end


function set_defaults!(tree::DecisionTreeClassifier, Y::DataFrame)
    values = count_classes(Y, tree.n_classes)
    push!(tree.values, values)
    push!(tree.impurities, gini_score(values))
    push!(tree.split_features, nothing)
    push!(tree.split_values, nothing)
    push!(tree.n_samples, size(Y, 1))
    add_node!(tree.binarytree)
end


function split_node!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame, depth::Int=0)
    tree.num_nodes += 1
    node_id = tree.num_nodes
    set_defaults!(tree, Y)

    if tree.impurities[node_id] == 0
        return #only one class in this node (measured by Gini). No more splitting
    end
    
    n_features_split = isnothing(tree.max_features) ? tree.n_features : min(tree.n_features, tree.max_features)
    features = randperm(tree.random_state, tree.n_features)[1:n_features_split]

    # Make the split
    best_score = Inf
    for i in features
        best_score = find_better_split(i, X, Y, node_id, best_score, tree)
    end
    if best_score == Inf
        return # no split found
    end

    # Make children (Where we keep slitting)
    if isnothing(tree.max_depth) || (depth < tree.max_depth)
        x_split = X[:, tree.split_features[node_id]]
        lhs = x_split .<= tree.split_values[node_id]
        rhs = .!lhs
        set_left_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[lhs, :], Y[lhs, :], depth + 1)
        set_right_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[rhs, :], Y[rhs, :], depth + 1)
    end
end

# And now we get to the real ML magic (find_better_split)
function find_better_split(feature_idx, X::DataFrame, Y::DataFrame, node_id::Int,
                            best_score::AbstractFloat, tree::DecisionTreeClassifier)
    x = X[:, feature_idx]

    n_samples = length(x)

    order = sortperm(x)
    x_sort, y_sort = x[order], Y[order, 1]

    rhs_count = count_classes(y_sort, tree.n_classes)
    lhs_count = zeros(Int, tree.n_classes)

    xi, yi = zero(x_sort[1]), zero(y_sort[1])
    for i in 1:(n_samples-1)
        global xi = x_sort[i]
        global yi = y_sort[i]
        lhs_count[yi] += 1; rhs_count[yi] -= 1
        if (xi == x_sort[i+1]) || (sum(lhs_count) < tree.min_samples_leaf)
            continue
        end
        if sum(rhs_count) < tree.min_samples_leaf
            break
        end
        curr_score = gini_score(lhs_count) * sum(lhs_count)/n_samples + gini_score(rhs_count) * sum(rhs_count)/n_samples
        if curr_score < best_score
            best_score = curr_score
            tree.split_features[node_id] = feature_idx
            tree.split_values[node_id] = (xi + x_sort[i+1])/2
        end
    end
    return best_score
end

function predict(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs = predict_prob(tree, X)
    return mapslices(argmax, probs, dims=2)[:, 1]
end

function predict_prob(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs  = zeros(nrow(X), tree.n_classes)
    for (i, xi) in enumerate(eachrow(X))
        counts = predict_row(tree, xi)
        probs[i, :] .= counts/sum(counts)
    end
    return probs
end


function predict_row(tree::DecisionTreeClassifier, xi::T ) where T <: DataFrameRow
    next_node = 1
    while !is_leaf(tree.binarytree, next_node)
        left, right = get_children(tree.binarytree, next_node)
        next_node = xi[tree.split_features[next_node]] <= tree.split_values[next_node] ? left : right
    end
    return tree.values[next_node]
end


function feature_importance_impurity(tree::DecisionTreeClassifier)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    feature_importances = zeros(tree.n_features)
    total_samples = tree.n_samples[1]
    for node in 1:length(tree.impurities)
        if is_leaf(tree.binarytree, node)
            continue
        end
        spit_feature = tree.split_features[node]
        impurity = tree.impurities[node]
        n_samples = tree.n_samples[node]
        # calculate score
        left, right = get_children(tree.binarytree, node)
        lhs_gini = tree.impurities[left]
        rhs_gini = tree.impurities[right]
        lhs_count = tree.n_samples[left]
        rhs_count = tree.n_samples[right]
        score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
        # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
        feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    end
    # normalise
    feature_importances = feature_importances/sum(feature_importances)
    return feature_importances
end


# Some utilty functions
check_random_state(seed::Int) = MersenneTwister(seed)
check_random_state(rng::AbstractRNG) = rng

# And now we can fit the model
# We will use the iris dataset
using RDatasets
iris = dataset("datasets", "iris")
X = iris[:, 1:4]
Y = iris[:, 5] 
# put Y in a DataFrame and convert to Int
Y = DataFrame(Y = map(x -> x == "setosa" ? 1 : x == "versicolor" ? 2 : 3, Y))

# We will use the DecisionTreeClassifier
tree = DecisionTreeClassifier()

# And now we can fit the model
fit!(tree, X, Y)

# And now we can predict
predictions = predict(tree, X)
# Accruacy
println("Accuracy: ", sum(predictions .== Y[:, 1])/length(predictions))

# And we can get the feature importance
fi = feature_importance_impurity(tree)
println("Feature importance: ", fi)

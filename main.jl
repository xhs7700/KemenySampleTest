using Statistics
using GeneralGraphs
using GraphDatasets
using LinearAlgebra
using Arpack
using LinearAlgebraUtils
using SparseArrays
using ProgressBars
using Random
using Base.Threads

function bernstein_error(n, var_val, max_val, delta)
    ratio = log(3 / delta)
    return sqrt(2 * var_val * var_val * ratio / n) + 3 * max_val * ratio / n
end

function get_maxlen(g::NormalUnweightedGraph, tol::Float64)
    d, sp_A = diagadj(g)
    N = length(d)
    P = spdiagm(inv.(d)) * sp_A
    lam = 0.995
    max_len = ceil(Int, log(tol * (1 - lam) / (2 * N)) / log(lam))
    @show max_len
    return max_len
end

function approx_kem_sqrt(g::NormalUnweightedGraph, eps::Float64, delta::Float64, alpha::Union{Int,Float64}, c::Int, max_len::Int)
    N, _ = size(g)
    max_sample = ceil(Int, alpha * N)

    sample_nodes = randsubseq(1:N, min(c * sqrt(1 / N), 1.0))
    ratio = N / length(sample_nodes)
    sample_num = Dict(src => max_sample for src in sample_nodes)
    test_kem = Threads.Atomic{Float64}(-max_len)

    @threads for src in ProgressBar(sample_nodes)
        ans_sum = 0
        ans_sqsum = 0
        sample_num = max_sample
        for r in 1:max_sample
            ans = 0
            u = src
            for l in 1:max_len
                if u == src
                    ans += 1
                end
                u = rand(g.adjs[u])
            end
            ans_sum += ans
            ans_sqsum += ans * ans
            ans_mean = ans_sum / r
            ans_var = ans_sqsum / r - ans_mean * ans_mean
            approx_err = bernstein_error(r, ans_var, max_len ÷ 2, delta)
            if approx_err < eps
                sample_num = r
                break
            end
        end
        atomic_add!(test_kem, ans_sum * ratio / sample_num)
    end
    # open(io -> join(io, values(sample_num), '\n'), "tmp.txt", "w")
    return test_kem[]
end

function approx_kem_le(g::NormalUnweightedGraph, max_len::Int, max_sample::Int)
    test_trace = Threads.Atomic{Float64}(0)
    test_pinv = Threads.Atomic{Float64}(0)
    d, _ = diagadj(g)
    N = length(d)
    v = sortperm(d; rev=true)[begin]
    @threads for _ in ProgressBar(1:max_sample)
        inforest = falses(N)
        fa = zeros(Int, N)
        inforest[v] = true
        test_trace_internal = zero(Float64)
        for src in 1:N
            u = src
            while inforest[u] == false
                test_trace_internal += 1
                fa[u] = rand(g.adjs[u])
                u = fa[u]
            end
            u = src
            while inforest[u] == false
                inforest[u] = true
                u = fa[u]
            end
        end
        atomic_add!(test_trace, test_trace_internal)
    end
    test_trace[] /= max_sample
    @threads for _ in ProgressBar(1:max_sample)
        u = v
        test_pinv_internal = zero(Float64)
        for _ in 1:max_len
            if u == v
                test_pinv_internal += 1
            end
            u = rand(g.adjs[u])
        end
        atomic_add!(test_pinv, test_pinv_internal)
    end
    test_pinv[] = (test_pinv[] * sum(d)) / (max_sample * d[v]) - max_len
    return test_trace[] - test_pinv[]
end

function test_model(g::NormalUnweightedGraph, std_kem::Float64, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    @show g.name
    @show size(g)
    N, _ = size(g)
    eps = tol * N
    len=ceil(Int,c*sqrt(N))
    println("num_sample = $len")
    println("length_sample = $max_len")
    get_maxlen(g, tol)
    stats_sqrt = @timed approx_kem_sqrt(g, eps, delta, alpha, c, max_len)
    test_kem_sqrt = stats_sqrt.value
    println("Running time of approx_kem_sqrt is $(stats_sqrt.time)s")
    println("GC time of approx_kem_sqrt is $(stats_sqrt.gctime)s")
    ratio_sqrt = getratio(std_kem, test_kem_sqrt)
    @show std_kem, test_kem_sqrt, ratio_sqrt
    stats_le = @timed approx_kem_le(g, max_len, max_sample)
    test_kem_le = stats_le.value
    println("Running time of approx_kem_le is $(stats_le.time)s")
    println("GC time of approx_kem_le is $(stats_le.gctime)s")
    ratio_le = getratio(std_kem, test_kem_le)
    @show std_kem, test_kem_le, ratio_le
end

function test_pseudo(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    std_kem = 2.5 * 3^x - (5 / 3) * 2^x + 0.5
    g = loadPseudofractal(x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

function test_koch(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    std_kem = (2 * x + 1) * 4^x + 1 / 3
    g = loadKoch(x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

function test_cayley(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    std_kem = (3 * x * 4^(x + 1) - 13 * 2^(2 * x + 1) + 35 * 2^x - 9) / (2 * (2^x - 1))
    g = load3CayleyTree(x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

function test_hanoiext(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    std_kem = (32 * 5^x * 3^(x - 1) - 64 * 3^(2 * x - 2) - 2 * 3^x) / (10 * (3^x + 3^(x - 1) - 1))
    g = loadHanoiExt(x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

function test_apollo(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    # 1+\frac{1}{12}\left( 32\times 3^g-16\left(\frac{9}{5}\right)^g +11 \right)
    std_kem = 1 + (32 * 3^x - 16 * (9 / 5)^x + 11) / 12
    g = loadApollo(x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

function test_corona(x::Int, tol::Float64, delta::Float64, alpha::Int, c::Int, max_sample::Int, max_len::Int)
    std_kem = (16 / 5 - 6) * 4^x + (16 / 3) * 10^x + 2 / 3
    g = loadCorona(3, x) |> NormalUnweightedGraph
    test_model(g, std_kem, tol, delta, alpha, c, max_sample, max_len)
end

const tol = 0.01
const delta = 0.1
const alpha = 10
const c = 100
const max_sample = 128
const max_len = 40000

# test_pseudo(5, tol, delta, alpha, c, max_sample, max_len)
# test_cayley(7, tol, delta, alpha, c, max_sample, max_len)
# test_hanoiext(5, tol, delta, alpha, c, max_sample, max_len)
# test_apollo(5, tol, delta, alpha, c, max_sample, max_len)
# test_corona(2, tol, delta, alpha, c, max_sample, max_len)

using LinearAlgebra
using SparseArrays
using BandedMatrices

import Base: show

struct Basis
    t::AbstractKnotSet
    x::AbstractVector
    w::AbstractVector
    B::Vector{AbstractMatrix}
    ∂B::Vector{AbstractMatrix}
    bl
    br
end

# See http://pages.cs.wisc.edu/~deboor/pgs/bsplvb.f
function evaluate!(Bᵢ, t::AbstractKnotSet{T}, x::AbstractVector, bl::T, br::T) where T
    #=
    \[B_{i,1,t}(x)=\begin{cases}
    1, & t_i\leq x < t_{i+1}\\
    0, & \textrm{else}
    \end{cases}\]
    =#
    for i = 1:length(t)-1
        Bᵢ[1][:,i] = (t[i] .<= x .< t[i+1])
    end

    #=
    \[B_{i,k,t}(x)=\frac{x-t_i}{t_{i+k-1}-t_i}B_{i,k-1,t}(x)-
    \frac{t_{i+k}-x}{t_{i+k}-t_{i+1}}B_{i+1,k-1,t}(x)\]
    =#
    for kk = 2:order(t)
        for i = 1:length(t) - kk
            d₁ = t[i+kk-1]-t[i]
            d₂ = t[i+kk]-t[i+1]
            f₁ = d₁ == 0 ? 0 : (x .- t[i])/d₁
            f₂ = d₂ == 0 ? 0 : (t[i+kk] .- x)/d₂
            Bᵢ[kk][:,i] = f₁.*Bᵢ[kk-1][:,i] + f₂.*Bᵢ[kk-1][:,i+1]
        end
    end
    Bᵢ[end][:,1] *= bl
    Bᵢ[end][:,end] *= br
end

function evaluate!(∂Bᵢ, Bᵢ, t::BSplines.AbstractKnotSet{T}, x::AbstractVector,
                   bl::T, br::T) where T
    #=
    \[\partial B_{i,k,t}(x) = (k-1)
    \left[
    \frac{B_{i,k-1,t}(x)}{t_{i+k-1}-t_i}-
    \frac{B_{i+1,k-1,t}(x)}{t_{i+k}-t_{i+1}}
    \right]\]
    \[\begin{aligned}
    \partial^{(n)} B_{i,k,t}(x)
    &= (k-1)
    \left[
    \frac{\partial^{(n-1)} B_{i,k-1,t}(x)}{t_{i+k-1}-t_i}-
    \frac{\partial^{(n-1)} B_{i+1,k-1,t}(x)}{t_{i+k}-t_{i+1}}
    \right] \\
    &=
    (k-1)(k-2)
    \left[
    \frac{
    \frac{\partial^{(n-2)}B_{i,k-2,t}(x)}{t_{i+k-2}-t_i}-
    \frac{\partial^{(n-2)}B_{i+1,k-2,t}(x)}{t_{i+k-1}-t_{i+1}}
    }{t_{i+k-1}-t_i}-
    \frac{
    \frac{\partial^{(n-2)}B_{i+1,k-2,t}(x)}{t_{i+k-1}-t_{i+1}}-
    \frac{\partial^{(n-2)}B_{i+2,k-2,t}(x)}{t_{i+k}-t_{i+2}}
    }{t_{i+k}-t_{i+1}}
    \right]
    \end{aligned}\]


    de Boor X(15):
    \[ \partial^{(m)}\sum_j \alpha_j B_{j,k,t}
    =\sum_j \alpha_j^{(m+1)} B_{j,k-m,t}\]

    where [de Boor X(16)]

    \[\alpha_r^{(m+1)}=
    \begin{cases}
    \alpha_r, & m=0,\\
    \frac{\alpha_r^{(m)}-\alpha_{r-1}^{(m)}}{(t_{r+k-m}-t_r)/(k-m)}, & m>0.
    \end{cases}\]

    To calculate \(\partial^{(m)}B_{i,k,t}\), \(\forall m\in[1,k-1]\), we do

    - \(\forall i\in[1,N]\), set \(\alpha^{(1)}_j = \delta_{ij}\)
      - \(\forall m\in[1,k-1]\)
        - Compute \(\alpha^{(m+1)}\) using X(16)
        - Compute \(\partial^{(m)}B_{i,k,t}\) using X(15)
    =#

    evaluate!(Bᵢ, t, x, bl, br)

    k = order(t)
    N = length(t) - k
    α = Matrix{T}(undef, N, k)
    for i ∈ 1:N
        α[:,1] .= zero(T)
        α[i,1] = if i == 1
            bl
        elseif i == N
            br
        else
            one(T)
        end
        for m ∈ 1:k-1
            ∂Bᵢ[m][:,i] .= zero(T)
            for j ∈ 1:N
                δt = t[j+k-m] - t[j]
                if δt == 0
                    α[j,m+1] = 0
                    continue
                end
                α[j,m+1] = (k-m)*(α[j,m]-(j>1 ? α[j-1,m] : 0))/δt
                ∂Bᵢ[m][:,i] += α[j,m+1]*Bᵢ[k-m][:,j]
            end
        end
    end
end

function Basis(t::AbstractKnotSet{T}, bl::T=one(T), br::T=one(T)) where T
    x,w = lgwt(t)
    B = [spzeros(eltype(x), length(x), length(t)-kk)
         for kk = 1:order(t)]
    ∂B = [spzeros(eltype(x), length(x), length(t)-kk)
          for kk = 1:order(t)]
    evaluate!(∂B, B, t, x, bl, br)
    Basis(t, x, w, B, ∂B, bl, br)
end
Basis(t::AbstractKnotSet{T}, bl, br) where T = Basis(t, T(bl), T(br))

function (basis::Basis)(x::AbstractVector)
    Bᵢ = [spzeros(eltype(x), length(x), length(basis.t)-kk)
          for kk = 1:order(basis.t)]
    evaluate!(Bᵢ, basis.t, x, basis.bl, basis.br)
    Bᵢ[end]
end

"""
    basis(f)

Generate the scalar operator corresponding to `f(x)` on the BSpline
`basis`.
"""
function (basis::Basis)(f::Function)
    m = size(basis.B[end],2)
    k = order(basis.t)
    B = BandedMatrix{eltype(basis.t)}(undef, m,m, 0, k-1)
    fwx = weights(basis) .* f.(locs(basis))
    for j = 1:m
        for i = max(1,j-k):j
            B[i,j] = dot(basis.B[end][:,i], (fwx .* basis.B[end][:,j]))
        end
    end
    Symmetric(B)
end

"""
    basis(I)

Return the overlap matrix of `basis`.
"""
(basis::Basis)(::UniformScaling) = basis(one)

function derop(::Type{T}, basis::Basis, o::Integer) where T
    m = size(basis.B[end],2)
    k = order(basis.t)
    D = BandedMatrix{T}(undef, m,m, k-1, k-1)
    symmetric = o == 2 && basis.bl == 0 && basis.br == 0
    for j = 1:m
        for i = max(1,j-k):(symmetric ? j : min(j+k,m))
            D[i,j] = dot(basis.B[end][:,i], (weights(basis) .* basis.∂B[o][:,j]))
        end
    end
    symmetric ? Symmetric(D) : D
end
derop(basis::Basis, o::Integer) = derop(eltype(basis.t), basis, o)

locs(basis::Basis) = basis.x
weights(basis::Basis) = basis.w

function show(io::IO, basis::Basis)
    write(io, "BSpline basis with $(basis.t)")
end

@recipe function plot(basis::Basis, n=100)
    x = range(first(basis.t), stop=last(basis.t),
              length=n*numintervals(basis.t))
    x, Matrix(basis(x))
end

export locs, weights, derop

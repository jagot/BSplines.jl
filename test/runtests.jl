using BSplines
using Test
using LinearAlgebra
using SparseArrays

function vecdist(a::AbstractVector, b::AbstractVector,
                 ϵ = eps(eltype(a)))
    δ = √(sum(abs2, a-b))
    δ, δ/√(sum(abs2, a .+ ϵ))
end

@testset "Knot sets" begin
    @testset "Linear" begin
        t = LinearKnotSet(7, 0, 1, 10)
        @test order(t) == 7
        @test numintervals(t) == 10
        @test length(t) == 23
        @test first(t) == 0
        @test last(t) == 1
        for i = 1:7
            @test t[i] == 0
            @test t[end-i+1] == 1
        end
        tt = range(0, stop=1, length=11)
        for i = 7:17
            @test t[i] == tt[i-6]
        end

        ttt = collect(t)
        @test length(t) == length(ttt)
        for i ∈ eachindex(t)
            @test t[i] == ttt[i]
        end

        @test eltype(t) <: Real
    end

    @testset "Exponential" begin
        t = ExpKnotSet(7, -8, 1, 31)
        @test order(t) == 7
        @test numintervals(t) == 31
        @test first(t) == 0
        @test last(t) == 10

        t′ = ExpKnotSet(7, -8, 1, 31, include0=false)
        @test order(t′) == 7
        @test numintervals(t′) == 31
        @test first(t′) == 1e-8
        @test last(t′) == 10

        t′′ = ExpKnotSet(7, -8, 1, 31, base=2, include0=false)
        @test order(t′′) == 7
        @test numintervals(t′′) == 31
        @test first(t′′) == 1/256
        @test last(t′′) == 2
    end
end

@testset "Quadrature" begin
    t = LinearKnotSet(1, 0, 1, 2)
    x,w = BSplines.lgwt(t)
    @test all(w .== 1/4)
    @test x == [-1,1,-1,1]/(4*√3) + [1,1,3,3]/4
end

@testset "Basis" begin
    k = 3
    t = LinearKnotSet(k, 0, 1, 2)
    basis = BSplines.Basis(t)
    @testset "Eval on subintervals" begin
        x₁ = range(-1,stop=-0.5,length=10)
        Bᵢ₁ = basis(x₁)
        @test norm(Bᵢ₁) == 0

        function testbasis(x)
            Bᵢ = basis(x)
            B̃ = spzeros(Float64, length(x), 4)
            B̃[:,1] = (x .>= 0) .* (x .< 0.5) .* ((2x .- 1).^2)
            B̃[:,2] = (x .>= 0) .* (x .< 0.5) .* (2/3*(1 .- (3x .- 1).^2)) +
                (x .>= 0.5) .* (x .< 1)  .* (2*(x .- 1).^2)
            B̃[:,3] = (x .>= 0) .* (x .< 0.5) .* (2*x.^2) +
                (x .>= 0.5) .* (x .< 1)  .* (2/3*(1 .- (3x .- 2).^2))
            B̃[:,4] = (x .>= 0.5) .* (x .< 1) .* ((2x .- 1).^2)
            for j = 1:4
                δ,δr = vecdist(Bᵢ[:,j],B̃[:,j])
                @test δ < 1e-15
                @test δr < 1e-15
            end
        end
        testbasis(range(0,stop=1,length=50))
        testbasis(range(-0.5,stop=0.6,length=40))
        testbasis(range(0.5,stop=1.6,length=40))
    end

    @testset "Scalar operators" begin
        @testset "Overlap matrix" begin
            B = basis(I)
            Br = [ 1/10  7/120  1/120      0
                  7/120    1/6   1/10  1/120
                  1/120   1/10    1/6  7/120
                      0  1/120  7/120   1/10]
            @test norm(B-Br) < 1e-15
        end
        @testset "Position operator" begin
            B = basis(x->x)
            Br = [ 1/120   1/96   1/480       0
                    1/96  7/120    1/20   1/160
                   1/480   1/20  13/120  23/480
                       0  1/160  23/480  11/120]
            @test norm(B-Br) < 1e-15
        end
    end

    @testset "Derivative operators" begin
        # References generated using AnalyticBSplines.jl
        ∂n = [derop(basis, n) for n ∈ 1:k-1]
        ∂ref = [[ -1//2    5//12   1//12   0
                  -5//12   0//1    1//3   1//12
                  -1//12  -1//3    0//1   5//12
                    0     -1//12  -5//12  1//2],
                [ 4//3  -2//1   2//3   0
                  2//1  -8//3   0//1  2//3
                  2//3   0//1  -8//3  2//1
                   0     2//3  -2//1  4//3]]
        for n ∈ 1:k-1
            @test norm(∂n[n]-∂ref[n]) < 1e-15
        end

        k′ = 7
        t′ = LinearKnotSet(k′, 0, 1, 3)
        basis′ = BSplines.Basis(t′)

        ∂′n = [derop(basis′, n) for n ∈ 1:k′-1]
        ∂′ref = [
 [    -1//2          10625//29568     283021//2395008    23827//1197504     479//199584        5//27216          1//149688         0                0
  -10625//29568          0//1          35867//177408    118891//1064448   26167//709632    11017//1419264     7141//8515584       1//946176         0
 -283021//2395008   -35867//177408         0//1           5129//39424     11645//101376   245743//4257792    20455//1216512    7141//8515584       1//149688
  -23827//1197504  -118891//1064448    -5129//39424          0//1         11035//118272   218689//2128896   245743//4257792   11017//1419264       5//27216
    -479//199584    -26167//709632    -11645//101376    -11035//118272        0//1         11035//118272     11645//101376    26167//709632      479//199584
      -5//27216     -11017//1419264  -245743//4257792  -218689//2128896  -11035//118272        0//1           5129//39424    118891//1064448   23827//1197504
      -1//149688     -7141//8515584   -20455//1216512  -245743//4257792  -11645//101376    -5129//39424          0//1         35867//177408   283021//2395008
        0               -1//946176     -7141//8515584   -11017//1419264  -26167//709632  -118891//1064448   -35867//177408        0//1         10625//29568
        0                 0               -1//149688        -5//27216      -479//199584   -23827//1197504  -283021//2395008  -10625//29568         1//2],
[    90//11      -27843//2464      169651//66528      2347//4752        377//5544         7//1188           1//4158            0                0
  16509//2464     -4413//616      -104639//133056   191011//266112    72047//177408  118771//1064448    31501//2128896        3//78848          0
 169651//66528  -104639//133056    -46861//22176    -16405//29568     58403//177408   45935//118272    121843//709632     31501//2128896       1//4158
   2347//4752    191011//266112    -16405//29568     -3083//3168     -36415//88704    39203//177408     45935//118272    118771//1064448       7//1188
    377//5544     72047//177408     58403//177408   -36415//88704     -5807//7392    -36415//88704      58403//177408     72047//177408      377//5544
      7//1188    118771//1064448    45935//118272    39203//177408   -36415//88704    -3083//3168      -16405//29568     191011//266112     2347//4752
      1//4158     31501//2128896   121843//709632    45935//118272    58403//177408  -16405//29568     -46861//22176    -104639//133056   169651//66528
       0              3//78848      31501//2128896  118771//1064448   72047//177408  191011//266112   -104639//133056     -4413//616       16509//2464
       0               0                1//4158          7//1188        377//5544      2347//4752      169651//66528     -27843//2464         90//11],
[  -108//1     41193//224    -178259//2016    10723//1008     41//24        43//252        1//126          0             0
 -23049//224     162//1       -78221//1344   -13759//2688   4773//1792    2023//1536    5039//21504       9//7168        0
 -93901//2016  78221//1344         0//1       -9741//896    -905//256     4079//3584   10065//7168     5039//21504      1//126
 -10723//1008  13759//2688      9741//896         0//1     -4621//896    -5055//1792    4079//3584     2023//1536      43//252
    -41//24    -4773//1792       905//256      4621//896       0//1      -4621//896     -905//256      4773//1792      41//24
    -43//252   -2023//1536     -4079//3584     5055//1792   4621//896        0//1      -9741//896    -13759//2688   10723//1008
     -1//126   -5039//21504   -10065//7168    -4079//3584    905//256     9741//896        0//1      -78221//1344   93901//2016
       0          -9//7168     -5039//21504   -2023//1536  -4773//1792   13759//2688   78221//1344     -162//1      23049//224
       0            0             -1//126       -43//252     -41//24    -10723//1008  178259//2016   -41193//224      108//1],
[  1080//1    -224775//112    413855//336     -58255//168     1045//28       185//42         5//21            0              0
 138105//112   -62505//28     851525//672    -354385//1344  -16085//896    56255//5376   34625//10752      135//3584         0
 232415//336  -781435//672     57055//112       7365//448   -45905//896   -25215//1792   25535//3584     34625//10752       5//21
  32465//168  -354385//1344     7365//448       8255//112      595//64     -3815//128   -25215//1792     56255//5376      185//42
   1045//28    -16085//896    -45905//896        595//64      5055//112      595//64    -45905//896     -16085//896      1045//28
    185//42     56255//5376   -25215//1792     -3815//128      595//64      8255//112     7365//448    -354385//1344    32465//168
      5//21     34625//10752   25535//3584    -25215//1792  -45905//896     7365//448    57055//112    -781435//672    232415//336
       0          135//3584    34625//10752    56255//5376  -16085//896  -354385//1344  851525//672     -62505//28     138105//112
       0             0             5//21         185//42      1045//28    -58255//168   413855//336    -224775//112      1080//1],
[   -7290//1    1578285//112   -1077255//112    203895//56     -26055//28       1395//14         45//7             0              0
 -1170045//112    40095//2     -3001995//224   2125575//448   -866295//896     -3375//256    127305//3584      3645//3584         0
  -873225//112  3274155//224     -18225//2     1109295//448     26325//896   -421605//1792   -10125//512     127305//3584       45//7
  -158535//56   2228985//448   -1109295//448         0//1      202095//448     26325//896   -421605//1792     -3375//256      1395//14
   -19305//28    866295//896     -26325//896   -202095//448         0//1      202095//448     26325//896    -866295//896     19305//28
    -1395//14      3375//256     421605//1792   -26325//896   -202095//448         0//1     1109295//448   -2228985//448    158535//56
      -45//7    -127305//3584     10125//512    421605//1792   -26325//896  -1109295//448     18225//2     -3274155//224    873225//112
         0        -3645//3584   -127305//3584     3375//256    866295//896  -2125575//448   3001995//224     -40095//2     1170045//112
         0             0            -45//7       -1395//14      26055//28    -203895//56    1077255//112   -1578285//112      7290//1],
[174960//7    -98415//2      70605//2     -107595//7      38070//7       -9180//7       1080//7            0            0
  98415//2   -677970//7    1944135//28    -241245//8    1185435//112    -79245//32    105435//448     10935//448        0
 115155//2  -3158865//28   1117395//14   -1852875//56    157545//16    -176175//224  -419985//448    105435//448    1080//7
 215595//7   -474525//8    2229525//56    -186705//14     26325//56     325215//112  -176175//224    -79245//32    13500//7
  72090//7  -2080485//112   157545//16      26325//56    -57105//14      26325//56    157545//16   -2080485//112   72090//7
  13500//7    -79245//32   -176175//224    325215//112    26325//56    -186705//14   2229525//56    -474525//8    215595//7
   1080//7    105435//448  -419985//448   -176175//224   157545//16   -1852875//56   1117395//14   -3158865//28   115155//2
       0       10935//448   105435//448    -79245//32   1185435//112   -241245//8    1944135//28    -677970//7     98415//2
       0            0         1080//7       -9180//7      38070//7     -107595//7      70605//2      -98415//2    174960//7]]

        for n ∈ 1:k′-1
            @test norm(∂′n[n]-∂′ref[n])/abs(1e-16 + norm(∂′ref[n])) < 1e-14
        end
    end

    @testset "Dirichlet0 conditions" begin
        k = 3
        t = LinearKnotSet(k, 0, 2, 4)
        basis = BSplines.Basis(t, 0, 0)

        ∂n = [derop(basis, n) for n ∈ 1:k-1]
        ∂ref = [[ 0    0       0       0      0     0
                  0   0//1    3//8    1//24   0     0
                  0  -3//8    0//1    5//12  1//24  0
                  0  -1//24  -5//12   0//1   3//8   0
                  0    0     -1//24  -3//8   0//1   0
                  0    0       0       0      0     0],
                [ 0    0      0      0      0    0
                  0  -8//3   1//3   1//3    0    0
                  0   1//3  -2//1   2//3   1//3  0
                  0   1//3   2//3  -2//1   1//3  0
                  0    0     1//3   1//3  -8//3  0
                  0    0      0      0      0    0]]
        for n ∈ 1:k-1
            @test norm(∂n[n]-∂ref[n]) < 1e-15
        end
    end
end

@testset "Splines" begin
    k = 7
    t = LinearKnotSet(k, 0, 1, 10)
    basis = BSplines.Basis(t)
    x = range(first(t), stop=last(t), length=101)[1:end-1] # Due to issue #2
    B = basis(x)
    for n = 1:k-1
        c = rand(n)
        f = x -> sum(c[i]*x^(i-1) for i in eachindex(c))
        S = Spline(f, basis)
        @test norm(S(B) - f.(x))/abs(1e-16 + norm(f.(x))) < 1e-14
    end
end

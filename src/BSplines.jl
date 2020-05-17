module BSplines

function __init__()
    @warn "The BSplines.jl package has been deprecated in favour of JuliaApproximation/CompactBases.jl"
    nothing
end

using RecipesBase

include("knot_sets.jl")
include("quadrature.jl")
include("basis.jl")
include("splines.jl")

end

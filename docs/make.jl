using SpikingRNNs
using Documenter

DocMeta.setdocmeta!(SpikingRNNs, :DocTestSetup, :(using SpikingRNNs); recursive=true)

makedocs(;
    modules=[SpikingRNNs],
    authors="Dylan Festa",
    repo="https://github.com/dylanfesta/SpikingRNNs.jl/blob/{commit}{path}#{line}",
    sitename="SpikingRNNs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/SpikingRNNs.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/SpikingRNNs.jl",
)

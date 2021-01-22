using CompressingSolvers
using Documenter

makedocs(;
    modules=[CompressingSolvers],
    authors="Florian Schaefer <flotosch@gmail.com> and contributors",
    repo="https://github.com/f-t-s/CompressingSolvers.jl/blob/{commit}{path}#L{line}",
    sitename="CompressingSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://f-t-s.github.io/CompressingSolvers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/f-t-s/CompressingSolvers.jl",
)

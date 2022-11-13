


using BenchmarkTools

const n = 1_000_000

function thefunction(v)
  Threads.@threads for i in eachindex(v)
      v[i] = randn() .* 33.4
  end
  return nothing
end


println("\n")
@info "Benchmark 1 : \n"
b1 = @benchmark thefunction(x) setup=(x=zeros(n));
show(stdout, MIME("text/plain"), b1)
println("\n")
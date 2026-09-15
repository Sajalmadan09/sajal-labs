CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DACCELERATE_NEW_LAPACK -framework Accelerate

all: mlp transformer gemm_bench

mlp: mlp.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o mlp mlp.cpp

transformer: transformer.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o transformer transformer.cpp

gemm_bench: gemm_bench.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o gemm_bench gemm_bench.cpp

clean:
	rm -f mlp transformer gemm_bench

.PHONY: clean all

CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DACCELERATE_NEW_LAPACK -framework Accelerate

all: mlp transformer

mlp: mlp.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o mlp mlp.cpp

transformer: transformer.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o transformer transformer.cpp

clean:
	rm -f mlp transformer

.PHONY: clean all

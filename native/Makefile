CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DACCELERATE_NEW_LAPACK -framework Accelerate

mlp: mlp.cpp
	$(CXX) $(CXXFLAGS) -o mlp mlp.cpp

clean:
	rm -f mlp

.PHONY: clean

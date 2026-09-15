CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DACCELERATE_NEW_LAPACK -framework Accelerate

all: mlp transformer gemm_bench gender_predict sajal bert

mlp: mlp.cpp mlp_model.hpp common.hpp
	$(CXX) $(CXXFLAGS) -o mlp mlp.cpp

transformer: transformer.cpp transformer_model.hpp common.hpp
	$(CXX) $(CXXFLAGS) -o transformer transformer.cpp

gemm_bench: gemm_bench.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o gemm_bench gemm_bench.cpp

gender_predict: gender_predict.cpp mlp_model.hpp text_features.hpp common.hpp
	$(CXX) $(CXXFLAGS) -o gender_predict gender_predict.cpp

sajal: sajal.cpp mlp_model.hpp transformer_model.hpp text_features.hpp bert_model.hpp wordpiece.hpp common.hpp
	$(CXX) $(CXXFLAGS) -o sajal sajal.cpp

bert: bert.cpp bert_model.hpp wordpiece.hpp common.hpp
	$(CXX) $(CXXFLAGS) -o bert bert.cpp

clean:
	rm -f mlp transformer gemm_bench gender_predict sajal bert

.PHONY: clean all

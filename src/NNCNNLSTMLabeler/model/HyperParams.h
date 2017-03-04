#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int hiddenSize;
	int biRNNHiddenSize;
	int rnnHiddenSize;
	int wordContext;
	int wordWindow;
	int windowOutput;
	dtype dropProb;


	//auto generated
	int wordDim;
	int extWordDim;
	int inputSize;
	int labelSize;

	unordered_map<string, int>* hyper_word_stats;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		hiddenSize = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		biRNNHiddenSize = opt.biRNNHiddenSize;
		wordContext = opt.wordcontext;
		dropProb = opt.dropProb;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */
#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet extWordAlpha; // should be initialized outside
	LookupTable extWords; // should be initialized outside
	LSTMParams lstm_project;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.extWordDim = extWords.nDim;
		opts.labelSize = labelAlpha.size();
		lstm_project.initial(opts.rnnHiddenSize, opts.wordDim + opts.extWordDim, mem);
		opts.inputSize = opts.rnnHiddenSize * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		extWords.exportAdaParams(ada);
		lstm_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		//checkgrad.add(&(olayer_linear.b), "olayer_linear.b");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */
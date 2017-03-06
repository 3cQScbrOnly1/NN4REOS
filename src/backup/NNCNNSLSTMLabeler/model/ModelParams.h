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
	UniParams hidden_linear;
	LSTMParams lstm_params; 
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
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = (opts.wordDim + opts.extWordDim)* opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.windowOutput;
		hidden_linear.initial(opts.hiddenSize, opts.inputSize, true, mem);
		lstm_params.initial(opts.rnnHiddenSize, opts.hiddenSize, mem);
		olayer_linear.initial(opts.labelSize, opts.rnnHiddenSize * 3, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		extWords.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		lstm_params.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&hidden_linear.W, "hidden_linear.W");
		checkgrad.add(&hidden_linear.b, "hidden_linear.b");
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
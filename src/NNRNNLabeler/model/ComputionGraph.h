#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	vector<LookupNode> _ext_word_inputs;
	vector<ConcatNode> _word_represents;
	RNNBuilder _rnn;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat_pool;

	LinearNode _output;

	unordered_map<string, int>* p_word_stats;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
		_ext_word_inputs.resize(sent_length);
		_word_represents.resize(sent_length);
		_rnn.resize(sent_length);
		_avg_pooling.setParam(sent_length);
		_max_pooling.setParam(sent_length);
		_min_pooling.setParam(sent_length);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_ext_word_inputs.clear();
		_word_represents.clear();
		_rnn.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
			_ext_word_inputs[idx].setParam(&model.extWords);
			_ext_word_inputs[idx].init(opts.extWordDim, opts.dropProb, mem);
			_word_represents[idx].init(opts.wordDim + opts.extWordDim, -1, mem);
		}
		_rnn.init(&model.rnn_project, opts.dropProb, true, mem);
		_avg_pooling.init(opts.rnnHiddenSize, -1, mem);
		_max_pooling.init(opts.rnnHiddenSize, -1, mem);
		_min_pooling.init(opts.rnnHiddenSize, -1, mem);
		_concat_pool.init(opts.rnnHiddenSize * 3, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);

		p_word_stats = opts.hyper_word_stats;
	}

public:
	string p_change_word(const string& word) {
		double p = 0.5;
		unordered_map<string, int>::iterator it;
		it = p_word_stats->find(word);
		if (it != p_word_stats->end() && it->second == 1)
		{
			double x = rand() / double(RAND_MAX);
			if (x > p)
				return unknownkey;
			else
				return word;
		}
		else
			return word;
	}

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = feature.m_tweet_words.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			string word;
			if (bTrain)
				word = p_change_word(feature.m_tweet_words[i]);
			else
				word = feature.m_tweet_words[i];
			_word_inputs[i].forward(this, word);
			_ext_word_inputs[i].forward(this, word);
			_word_represents[i].forward(this, &_word_inputs[i], &_ext_word_inputs[i]);
		}

		_rnn.forward(this, getPNodes(_word_represents, words_num));
		_max_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_min_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_avg_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_concat_pool.forward(this, &_max_pooling, &_min_pooling, &_avg_pooling);
		_output.forward(this, &_concat_pool);
	}
};

#endif /* SRC_ComputionGraph_H_ */
#pragma once
#define NEED_MOMENTUM

typedef double WEIGHT_TYPE;

struct SNeuron  //神经细胞，神经元
{
	int m_nInput;
	WEIGHT_TYPE *m_pWeights;//对应输入的权值
#ifdef NEED_MOMENTUM
	WEIGHT_TYPE *m_pPrevUpdate;//在引入冲量项时用于记录上一次的权值更新
#endif // NEED_MOMENTUM
	double m_dActivation;//激励值 输出值 经过Sigmoid函数之后的值
	double m_dError;//误差值
	void Init(int nInput)
	{
		m_nInput = nInput + 1;//由于有一个偏置项，输入数目是实际输入数目+1
		m_pWeights = new WEIGHT_TYPE[m_nInput];//权值数组分配空间
#ifdef NEED_MOMENTUM
		m_pPrevUpdate = new WEIGHT_TYPE[m_nInput];//为上一次权值数组分配空间
#endif // NEED_MOMENTUM
		m_dActivation = 0;
		m_dError = 0;
	}

	~SNeuron()
	{
		delete[]m_pWeights;
#ifdef NEED_MOMENTUM
		delete[]m_pPrevUpdate;
#endif // NEED_MOMENTUM
	}
};

struct SNeuronLayer  //神经网络层
{
	int m_nNeuron;//该层神经元数目
	SNeuron *m_pNeurons;//神经元数组
	SNeuronLayer(int nNeuron, int nInputsPerNeuron)
	{
		m_nNeuron = nNeuron;
		m_pNeurons = new SNeuron[nNeuron];//分配nNeuron个神经元的数组空间
		for (int i = 0; i < nNeuron; i++)
		{
			m_pNeurons[i].Init(nInputsPerNeuron);//神经元初始化
		}
	}
	~SNeuronLayer()
	{
		delete[]m_pNeurons;
	}
};

struct NEURALNET_HEADER
{
	double dwVersion;
	int m_nInput;
	int m_nOutput;
	int m_nHiddenLayer;
	int m_nNeuronsPlayer;
	int m_nEpoches;
};



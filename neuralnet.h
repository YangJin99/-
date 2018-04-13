#pragma once

#include "neuron.h"
#include <vector>
#include <cstring>
#include <afx.h>
#include <time.h>
#include <stdlib.h>
#include <afxwin.h>

#define WEIGHT_FACTOR 0.1//一个大于0小于1的浮点数，用来限定初始权值的范围
#define BIAS 1//偏置项w0的系数

typedef std::vector<double> iovector;

//返回一个0，1之间的随机浮点数
inline double RandFloat()
{
	return (rand()) / (RAND_MAX + 1.0);
}
//返回一个大于-1小于1的随机浮点数
inline double RandomClamped()
{
	return WEIGHT_FACTOR*(RandFloat() - RandFloat());
}

//神经网络类定义
class CNeuralNet
{
private:
	//初始化参数，不可更改

	int m_nInput;//输入单元数目
	int m_nOutput;//输出单元数目
	int m_nNeuronsPerLyr;//隐藏层单元数目
	//隐藏层数目，不包含输出层
	int m_nHiddenLayer;
	//训练配置信息
	int m_nMaxEpoch;//最大训练时代数目
	double m_dMinError;//误差阈值
	//动态参数

	int m_nEpochs;
	double m_dLearningRate;
	double m_dErrorSum;//一个时代的累计误差
	double m_dErr;//一个时代的平均到每一次训练、每个输出的误差
	bool m_bStop;//控制训练过程是否中途停止
	SNeuronLayer *m_pHiddenLyr;//隐藏层
	SNeuronLayer *m_pOutLyr;//输出层
	std::vector<double> m_vecError;//训练过程中对应于给个时代的训练误差
public:

	CNeuralNet(int nInput, int nOutput, int nNeuronsPerLyr);
	~CNeuralNet();
	void InitializeNetwork();//初始化网络
	bool CalculateOutput(std::vector<double> intput, std::vector<double> &output);//计算网络输出，前向传播
	bool TrainingEpoch(std::vector<double> &intputs, std::vector<double> &outputs);//训练一个时代，反向调整
	bool Train(std::vector<double> &SetIn, std::vector<double> &SetOut);//整个反向传播训练过程
	//识别某一个未知类别样本，返回类别标号
	int Recognize(CString strPathName, CRect rt, double &dConfidence);
	//获取参数
	double GetErrorSum() { return m_dErrorSum; }
	double GetError() { return m_dErr; }
	int GetEpoch() { return m_nEpochs; }
	int GetNumOutput() { return m_nOutput; }
	int GetNumInput() { return m_nInput; }
	int GetNumNeuronsPerLyr() { return m_nNeuronsPerLyr; }
	//设定训练配置信息
	void SetMaxEpoch(int nMaxEpoch)
	{
		m_nMaxEpoch = nMaxEpoch;
	}
	void SetMinError(int dMinError)
	{
		m_dMinError = dMinError;
	}
	void SetLeraningRate(double dLearningRate)
	{
		m_dLearningRate = dLearningRate;
	}
	void SetStopFlag(BOOL bStop)
	{
		m_bStop = bStop;
	}
	//保存和装载训练文件
	bool SaveToFile(const char* lpszFileName, bool bCreate = true);//保存训练结果
	bool LoadFromFile(const char* lpszFileName, DWORD dwStartPos = 0);//装载训练结果
protected:

	void CreateNetwork();//建立网络，为各层单元分配空间
	//Sigmoid激励函数
	double Sigmoid(double netinput)
	{
		double response = 1.0;//控制函数陡峭程度的参数
		return (1 / (1 + exp(-netinput / response)));
	}
};

CNeuralNet::CNeuralNet(int nInput, int nOutput, int nNeuronsPerLyr)
{
	m_nHiddenLayer = 1;//暂时只支持一个隐藏层网络
	m_nInput = nInput;
	m_nOutput = nOutput;
	m_nNeuronsPerLyr = nNeuronsPerLyr;

	m_pHiddenLyr = nullptr;
	m_pOutLyr = nullptr;

	CreateNetwork();//为网络各层分配空间
	InitializeNetwork();//初始化整个网络
}

void CNeuralNet::CreateNetwork()
{
	m_pHiddenLyr = new SNeuronLayer(m_nNeuronsPerLyr, m_nInput);
	m_pOutLyr = new SNeuronLayer(m_nOutput, m_nNeuronsPerLyr);
}

void CNeuralNet::InitializeNetwork()
{
	int i, j;

	srand((unsigned)time(NULL));
	//初始化隐藏层网络
	for (i = 0; i < m_pHiddenLyr->m_nNeuron; i++)
	{
		for (j = 0; j < m_pHiddenLyr->m_pNeurons[i].m_nInput; j++)
		{
			m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] = RandomClamped();
#ifdef NEED_MOMENTUM
			//第1个时代的训练开始之前，还没有上一次的权值更新信息
			m_pHiddenLyr->m_pNeurons[i].m_pPrevUpdate[j] = 0;
#endif // NEED_MOMENTUM
		}// j
	}// i
	//初始化网络层权值
	for (i = 0; i < m_pOutLyr->m_nNeuron; i++)
	{
		for (j = 0; j < m_pOutLyr->m_pNeurons[i].m_nInput; j++)
		{
			m_pOutLyr->m_pNeurons[i].m_pWeights[j] = RandomClamped();
#ifdef NEED_MOMENTUM
			//第1个时代的训练开始之前，还没有上一次的权值更新信息
			m_pOutLyr->m_pNeurons[i].m_pPrevUpdate[j] = 0;
#endif // NEED_MOMENTUM
		}// j
	}//i
	m_dErrorSum = 999.0;//初始化为一个很大的训练误差，将随着训练进行逐渐减小
	m_nEpochs = 0;//当前训练时代数目
}

bool CNeuralNet::CalculateOutput(std::vector<double>input, std::vector<double>&output)
{
	if (input.size() != m_nInput)//输入特征向量维数与网络输入不等
		return false;

	int i, j;
	double nInputSum;//求和项

	//计算隐藏层输出
	for (i = 0; i < m_pHiddenLyr->m_nNeuron; i++)
	{
		nInputSum = 0;
		for (j = 0; j < m_pHiddenLyr->m_pNeurons[i].m_nInput - 1; j++)  //点乘计算
		{
			nInputSum += m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * (input[j]);
		}// for j

		//加上偏移项
		nInputSum += m_pHiddenLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
		//计算S函数的输出
		m_pHiddenLyr->m_pNeurons[i].m_dActivation = Sigmoid(nInputSum);
	}//for i

	for (i = 0; i < m_pOutLyr->m_nNeuron; i++)
	{
		nInputSum = 0;
		//点乘计算
		for (j = 0; j < m_pOutLyr->m_pNeurons[i].m_nInput - 1; j++)
		{
			nInputSum += m_pOutLyr->m_pNeurons[i].m_pWeights[j] * m_pHiddenLyr->m_pNeurons[j].m_dActivation;
		}//for j

		//加上偏移项
		nInputSum += m_pOutLyr->m_pNeurons[i].m_pWeights[j] * BIAS;
		//计算S函数的输出
		m_pOutLyr->m_pNeurons[i].m_dActivation = Sigmoid(nInputSum);
		//存入输出向量
		output.push_back(m_pOutLyr->m_pNeurons[i].m_dActivation);
	}//for i
	return true;
}

bool CNeuralNet::TrainingEpoch(std::vector<double>& SetIn, std::vector<double>& SetOut)
{
	int i, j, k;
	double WeightUpdate;//权值更新量
	double err;//误差项

	m_dErrorSum = 0;//累积误差
	for (i = 0; i < SetIn.size(); i++)//增量的梯度下降（针对每个训练样本更新权值
	{
		iovector vecOutputs;
		if (!CalculateOutput(SetIn[i], vecOutputs))
		{
			return false;
		}
		//更新输出层权重
		for (j = 0; j < m_pOutLyr->m_nNeuron; j++)
		{
			//计算误差项
			err = ((double)SetOut[i][j] - vecOutputs[j])*vecOutputs[j] * (1 - vecOutputs[j]);
			m_pOutLyr->m_pNeurons[j].m_dError = err;

			//更新累计误差
			m_dErrorSum += ((double)SetOut[i][j] - vecOutputs[j])*((double)SetOut[i][j] - vecOutputs[j]);

			//更新每个输入的权重
			for (k = 0; k < m_pHiddenLyr->m_nNeuron; k++)
			{
				WeightUpdate = err*m_dLearningRate*m_pHiddenLyr->m_pNeurons[k].m_dActivation;
#ifdef NEED_MOMENTUM
				//带有冲量项的权值更新
				m_pOutLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate + m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
				m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] = WeightUpdate;
#else
				//跟新单元权值
				m_pOutLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate;
#endif // NEED_MOMENTUM
			}//for k  
			 //////////////////////////////////////////////////////////////////////////////////////////这里的k可能出错
			WeightUpdate = err*m_dLearningRate*BIAS;//偏置更新量
#ifdef NEED_MOMENTUM
			//带有冲量权的权值更新量
			m_pOutLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate + m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
			m_pOutLyr->m_pNeurons[j].m_pPrevUpdate[k] = WeightUpdate;
#else
			//偏置项的更新
			m_pOutLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate;
#endif // NEED_MOMENTUM
		}// for j

		//更新隐藏层权重
		for (j = 0; j < m_pHiddenLyr->m_nNeuron; j++)
		{
			err = 0;
			for (k = 0; k < m_pOutLyr->m_nNeuron; k++)
			{
				err += m_pOutLyr->m_pNeurons[k].m_dError*m_pOutLyr->m_pNeurons[k].m_pWeights[j];
			}//for k

			err *= m_pHiddenLyr->m_pNeurons[j].m_dActivation*(1 - m_pHiddenLyr->m_pNeurons[j].m_dActivation);

			//更新每个输入的权重
			for (k = 0; k < m_pHiddenLyr->m_pNeurons[j].m_nInput - 1; k++)
			{
				WeightUpdate = err*m_dLearningRate*SetIn[i][k];
#ifdef NEED_MOMENTUM
				//带有冲量项的权值更新量
				m_pHiddenLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate + m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
				m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] = WeightUpdate;
#else
				m_pHiddenLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate;
#endif // NEED_MOMENTUM
			}//for k

			//配置更新量
			WeightUpdate = err*m_dLearningRate*BIAS;
#ifdef NEED_MOMENTUM
			//带有冲量项的权值更新量
			m_pHiddenLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate + m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] * MOMENTUM;
			m_pHiddenLyr->m_pNeurons[j].m_pPrevUpdate[k] = WeightUpdate;
#else
			//偏置项的更新
			m_pHiddenLyr->m_pNeurons[j].m_pWeights[k] += WeightUpdate;
#endif // NEED_MOMENTUM
		}//for j
	}//for i
	m_nEpochs++;//时代计数+1
	return true;
}

bool CNeuralNet::Train(std::vector<double>&SetIn, std::vector<double>&SetOut)
{
	m_bStop = false;//是否要中途停止训练
	CString strOutMsg;//输出信息
	do
	{
		//训练一个时代
		if (!TrainingEpoch(SetIn, SetOut))
		{
			strOutMsg.Format("训练在第%d个时代出现错误",GetEpoch());
			AfxMessageBox(strOutMsg);
			return false;
		}

		//计算1个时代的平均到每1次训练、每个输出的错误
		m_dErr = GetErrorSum() / (GetNumOutput()*SetIn.size());
		if (m_dErr < m_dMinError)
			break;
		m_vecError.push_back(m_dErr);//记录各个时代的错误，以备训练结束后绘制训练误差曲线
		WaitForInputIdle();//在循环中暂停下来以检查是否有用户动作和消息，主要是为了让训练可以在中途停止
		if (m_bStop)
			break;
	} while (m_nMaxEpoch -- > 0);
	return true;
}

int CNeuralNet::Recognize(CString strPathName, CRect rt, double &dConfidence)
{






}




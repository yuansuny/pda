#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <map>
#include <set>
#include <assert.h>
#include <boost/python.hpp>
#include <Python.h>
using namespace std;

double const reload_threshold = 1e-2;
int Nsamples;
int Nfeatures;
int Nclasses;
int selected_features;
string file_name;

typedef struct jpState
{
  double *jointProbabilityVector;
  int numJointStates;
  double *firstProbabilityVector;
  int numFirstStates;
  double *secondProbabilityVector;
  int numSecondStates;
  map<double, int> firstStateMap;
  map<double, int> secondStateMap;
} JointProbabilityState;

int **data;
int *label;
double *nominator;
double **x_matrix;
double ***prob_ratio;
double ***memory_store;
double **featureMergeArray;
double **data_inverse;
JointProbabilityState *featureDistribution;

inline float fast_log2 (float val)
{
   int * const    exp_ptr = reinterpret_cast <int *> (&val);
   int            x = *exp_ptr;
   const int      log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

   return (val + log_2);
} 

int normaliseArray(double *inputVector, int *outputVector, int vectorLength, map<double, int>& stateMap)
{
  int minVal = 0;
  int maxVal = 0;
  int currentValue;
  int i;
  stateMap.clear();
  
  if (vectorLength > 0)
  {
    minVal = (int) floor(inputVector[0]);
    maxVal = (int) floor(inputVector[0]);
  
    for (i = 0; i < vectorLength; i++)
    {
      currentValue = (int) floor(inputVector[i]);
      outputVector[i] = currentValue;
      
      if (currentValue < minVal)
      {
        minVal = currentValue;
      }
      else if (currentValue > maxVal)
      {
        maxVal = currentValue;
      }
    }/*for loop over vector*/
    
    for (i = 0; i < vectorLength; i++)
    {
      outputVector[i] = outputVector[i] - minVal;
      stateMap[inputVector[i]] = outputVector[i];
    }

    maxVal = (maxVal - minVal) + 1;
  }
  
  return maxVal;
}/*normaliseArray(double*,double*,int)*/
int mergeArrays(double *firstVector, double *secondVector, double *outputVector, int vectorLength)
{
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int firstNumStates;
  int secondNumStates;
  int i;
  int *stateMap;
  int stateCount;
  int curIndex;
  
  firstNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  secondNormalisedVector = (int *) calloc(vectorLength,sizeof(int));

  map<double, int> state_map;
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength,state_map);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength,state_map);
  
  /*
  ** printVector(firstNormalisedVector,vectorLength);
  ** printVector(secondNormalisedVector,vectorLength);
  */
  stateMap = (int *) calloc(firstNumStates*secondNumStates,sizeof(int));
  stateCount = 1;
  for (i = 0; i < vectorLength; i++)
  {
    curIndex = firstNormalisedVector[i] + (secondNormalisedVector[i] * firstNumStates);
   /*
    if (stateMap[curIndex] == 0)
    {
      stateMap[curIndex] = stateCount;
      stateCount++;
    }
*/
    outputVector[i] = curIndex;
  }
    
  free(firstNormalisedVector);
  free(secondNormalisedVector);
  free(stateMap);
  
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  stateMap = NULL;
  
  /*printVector(outputVector,vectorLength);*/
  return stateCount;
}/*mergeArrays(double *,double *,double *, int, bool)*/

JointProbabilityState calculateJointProbability(double *firstVector, double *secondVector, int vectorLength)
{
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int *firstStateCounts;
  int *secondStateCounts;
  int *jointStateCounts;
  double *firstStateProbs;
  double *secondStateProbs;
  double *jointStateProbs;
  int firstNumStates;
  int secondNumStates;
  int jointNumStates;
  int i;
  double length = vectorLength;
  map<double, int> firstStateMap;
  map<double, int> secondStateMap;

  JointProbabilityState state;
  firstNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  secondNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength, firstStateMap);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength, secondStateMap);
  jointNumStates = firstNumStates * secondNumStates;
  
  firstStateCounts = (int *) calloc(firstNumStates,sizeof(int));
  secondStateCounts = (int *) calloc(secondNumStates,sizeof(int));
  jointStateCounts = (int *) calloc(jointNumStates,sizeof(int));
  
  firstStateProbs = (double *) calloc(firstNumStates,sizeof(double));
  secondStateProbs = (double *) calloc(secondNumStates,sizeof(double));
  jointStateProbs = (double *) calloc(jointNumStates,sizeof(double));
    
  for (i = 0; i < vectorLength; i++)
  {
    firstStateCounts[firstNormalisedVector[i]] += 1;
    secondStateCounts[secondNormalisedVector[i]] += 1;
    jointStateCounts[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += 1;
  }
  
  for (i = 0; i < firstNumStates; i++)
  {
    firstStateProbs[i] = firstStateCounts[i] / length;
  }
  
  for (i = 0; i < secondNumStates; i++)
  {
    secondStateProbs[i] = secondStateCounts[i] / length;
  }
  
  for (i = 0; i < jointNumStates; i++)
  {
    jointStateProbs[i] = jointStateCounts[i] / length;
  }

  free(firstNormalisedVector);
  free(secondNormalisedVector);
  free(firstStateCounts);
  free(secondStateCounts);
  free(jointStateCounts);
    
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  firstStateCounts = NULL;
  secondStateCounts = NULL;
  jointStateCounts = NULL;
  
  
  state.jointProbabilityVector = jointStateProbs;
  state.numJointStates = jointNumStates;
  state.firstProbabilityVector = firstStateProbs;
  state.numFirstStates = firstNumStates;
  state.secondProbabilityVector = secondStateProbs;
  state.numSecondStates = secondNumStates;
  state.firstStateMap = firstStateMap;
  state.secondStateMap = secondStateMap;
  return state;
}/*calculateJointProbability(double *,double *, int)*/


void readData(const string& train_file_name, const string& train_label_file_name) {
	string line;
	ifstream train_file(train_file_name), label_file(train_label_file_name);
	int i = 0, j = 0;
	while (getline(train_file, line)) {
		stringstream ss(line);
		string tem;
		j = 0;
		while (getline(ss, tem, ' ')) {
			data[i][j++] = stoi(tem);
		}
		++i;
	}
	i = 0;
	while (getline(label_file, line)) {
		label[i++] = stoi(line);	
	}
}

//log of conditional probability p(fi|se)=p(fi,se)/p(se)
double log_probability_con(JointProbabilityState& s, int &fi, int &se) 
{
	if (s.secondProbabilityVector[se] < 1e-10) 
		return 0.0;
	double value = s.jointProbabilityVector[se*s.numFirstStates+fi] / s.secondProbabilityVector[se];
	if (value < 1e-10) {
		return -1e10;
	} else {
		return log(value);
	}
}
inline double probability_con(JointProbabilityState& s, int &fi, int &se) 
{
	if (s.secondProbabilityVector[se] < 1e-10) 
		return 1.0;
	double value = s.jointProbabilityVector[se*s.numFirstStates+fi] / s.secondProbabilityVector[se];
	return value;
}

//logsumexp
double logsumexp(double nums[], int ct) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < ct ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < ct ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}

void feature_selection_discrete_greedy(int n_features) {

	double *label_vector = (double*) calloc(Nsamples,sizeof(double));
	for (int i = 0; i < Nsamples; ++i) {
		label_vector[i] = label[i];
	}

	double *cur_feature = (double*) calloc(Nsamples, sizeof(double));
	for (int i = 0; i < Nfeatures; ++i) {
		for (int j = 0; j < Nsamples; ++j) {
			cur_feature[j] = data[j][i];
			data_inverse[i][j] = data[j][i];
		}
		featureDistribution[i] = calculateJointProbability(cur_feature, label_vector, Nsamples);
		mergeArrays(cur_feature, label_vector, featureMergeArray[i], Nsamples);
	}

	//p(y') for each sample
	for (int i = 0; i < Nsamples; ++i) {
		for (int j = 0; j < featureDistribution[0].numSecondStates; ++j) {
			x_matrix[i][j] = featureDistribution[0].secondProbabilityVector[j];
		}
	}

	//start feature selection
	int feature_array[Nfeatures];	
	set<int> feature_set;
	double last_MI_learned = 0.0;
	int cur_step = 0;
	int initial_step = 0;
	while (cur_step < n_features) {
		double max_MI_learned = -1e10;
		int selected_feature = -1;
		if (initial_step == 0) { //first step: select maximum feature
		    for (int i = 0; i < Nfeatures; ++i){
                for (int j = 0; j < Nsamples; ++j){
                    for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
                        int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
                        memory_store[i][j][k] = probability_con(featureDistribution[i], firstValue, k);
                    }
                }
            }
        }
        for (int i = 0; i < Nfeatures; ++i) {
            if (feature_set.find(i) == feature_set.end()){
                double new_MI_learned = 0.0;
                double nominator_sum = 0.0;
                for (int j = 0; j < Nsamples; ++j) {
                    double tem = 0.0;
                    for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
                        tem += x_matrix[j][k]*memory_store[i][j][k];
                        if (featureDistribution[i].secondStateMap[label_vector[j]] == k){
                            assert(memory_store[i][j][k] != 0);
                            nominator_sum += log(memory_store[i][j][k]);
                        }
                    }
                    assert(tem != 0);
                    new_MI_learned -= log(tem);
                }
                new_MI_learned /= Nsamples;
                new_MI_learned += nominator_sum/Nsamples;
                if (new_MI_learned > max_MI_learned) {
                    max_MI_learned = new_MI_learned;
                    selected_feature = i;
                }
            }
        }
        if (max_MI_learned - last_MI_learned < reload_threshold && initial_step != 0) {
			cout << "WARNING: cannot increase the lower bound, clear everything" << endl;
			initial_step = 0;
			last_MI_learned = 0.0;
			for (int i = 0; i < Nfeatures; ++i)
				for (int j = 0; j < Nsamples; ++j)
					for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
						int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
						memory_store[i][j][k] = probability_con(featureDistribution[i], firstValue, k);
					}
			continue;
		} else{
		    feature_set.insert(selected_feature);
		    for (int i = 0; i < Nfeatures; ++i) {
                JointProbabilityState distribution = calculateJointProbability(data_inverse[selected_feature], featureMergeArray[i], Nsamples);
                for (int j = 0; j < Nsamples; ++j) {
                    for (int k = 0; k < featureDistribution[selected_feature].numSecondStates; ++k) {
                        int firstValue = featureDistribution[selected_feature].firstStateMap[data[j][selected_feature]];
                        int secondValue = distribution.secondStateMap[featureDistribution[i].firstStateMap[data[j][i]]+k*featureDistribution[i].numFirstStates];
                        double conditional_prob = probability_con(distribution, firstValue, secondValue);
                        memory_store[i][j][k] *= conditional_prob;
                    }
                }
            }
            last_MI_learned = max_MI_learned;
            feature_array[cur_step] = selected_feature;
            ++initial_step;
            cout << "step:" << cur_step++ << ": selected feature " << selected_feature << ", MI lower bound " << max_MI_learned << endl;
		}
	}
	cout << file_name << endl;	
	for (int i = 0; i < n_features; ++i) {
		if (i != n_features-1) 
			cout << feature_array[i] << " ";
		else
			cout << feature_array[i] << endl;
	}

	char results[100];
	FILE *fp;
    strcpy(results,"results/");
    strcat(results,"sel_features/");
    strcat(results,"selFeatures_VMIin_dataset_");
    strcat(results,file_name.c_str());
    strcat(results,".csv");
    fp = fopen(results, "w");

	for (int i = 0; i < n_features; ++i) {
		if (i != n_features-1)
		    fprintf(fp, "%d,", feature_array[i]);
		else
			fprintf(fp, "%d\n", feature_array[i]);
	}
	fclose(fp);


}

void VMIin(int nsF=100, string filename="wine", int nS = 10, int nF = 10, int nC = 10){
	Nsamples =  nS;
    Nfeatures = nF;
    Nclasses = nC;
    selected_features = nsF;
    printf("Nsamples is %d\n", Nsamples);
    printf("Nfeatures is %d\n", Nfeatures);
    printf("Nclasses is %d\n", Nclasses);
    file_name = filename;
    printf("filename is %s\n", file_name.c_str());

    data = (int **) malloc(Nsamples * sizeof(int *));
    for (int i = 0; i < Nsamples; i++){
        data[i] = new int[Nfeatures];
    }

    x_matrix = (double **)malloc(Nsamples * sizeof(double *));
    for (int i = 0; i < Nsamples; i++){
        x_matrix[i] = new double[Nclasses];
    }

    label = new int[Nsamples];
    nominator = new double[Nfeatures];

    prob_ratio = (double ***)malloc(Nfeatures * sizeof(double **));
    memory_store = (double ***)malloc(Nfeatures * sizeof(double **));
    for (int i = 0; i < Nfeatures; i++){
        prob_ratio[i] = (double **)malloc(Nsamples * sizeof(double *));
        memory_store[i] = (double **)malloc(Nsamples * sizeof(double *));
        for (int j = 0; j < Nsamples; j++){
            prob_ratio[i][j] = new double[Nclasses];
            memory_store[i][j] = new double[Nclasses];
        }
    }

    featureMergeArray = (double **)malloc(Nfeatures * sizeof(double *));
    data_inverse = (double **)malloc(Nfeatures * sizeof(double *));
    for (int i = 0; i < Nfeatures; i++){
        featureMergeArray[i] = new double[Nsamples];
        data_inverse[i] = new double[Nsamples];
    }
    featureDistribution = new JointProbabilityState[Nfeatures];

	readData("datasets/"+file_name+".txt", "datasets/"+file_name+"_labels.txt");
	feature_selection_discrete_greedy(selected_features);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(VMIin_overloads, VMIin, 1, 5)

BOOST_PYTHON_MODULE(VMI_in){
    boost::python::def("VMIin", &VMIin, VMIin_overloads(boost::python::args("nsF", "filename", "nS", "nF", "nC")));
}

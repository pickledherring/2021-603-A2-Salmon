#include <stdio.h>
#include <cuda_runtime.h>
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

__global__ void KNN(float* test, float* train, int* predictions, float* candidates, int* classCounts,
					int k,  int n_test, int n_train, int n_att, int n_classes) {
    // Implements a parallel kNN where for each candidate query an
	// in-place priority queue is maintained to

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < (2 * k); i++) {
        		candidates[i] = FLT_MAX;
    }

    if (tid < n_test) {
    	if (tid == 64) {
    		for (int i = 0; i < n_train * n_att; i++) {
    			printf("train[%d]: %f ", i, train[i]);
    		}
    	}
        for (int keyIndex = 0; keyIndex < n_train; keyIndex++) {
//        	if (tid == 64) {printf("\tkey index is %d\n", keyIndex);}
        	float dist = 0;
			for (int i = 0; i < n_att - 1; i++) {
//				if (tid == 64) {printf("\ti is %d\n", i);}

				float diff = test[tid * n_att + i] - train[keyIndex * n_att + i];
				dist += diff * diff;
//				if (tid == 64) {printf("\t\tdist = %f \n", dist);}
			}
            // Add to our candidates
            for (int c = 0; c < k; c++){
//            	if (tid == 64) {printf("\t\t%f < %f?: \n", dist, candidates[2 * c]);}
                if (dist < candidates[2 * c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                        if (tid == 64) {
                        	printf("candidates[2 * %d + 2] = %f\n", x, candidates[2 * x]);
                        	printf("candidates[2 * %d + 3] = %f\n", x, candidates[2 * x + 1]);
                        }
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    // class value
                    candidates[2 * c + 1] = train[keyIndex * n_att - 1];
                    if (tid == 64) {printf("candidates[%d] = %f\n", 2 * c + 1,
                    		train[keyIndex * n_att - 1]);}
                    break;
                }
            }
            printf("made it to d! rank %d", tid);
        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++) {
        	if (tid == 64) {printf("(int)candidates[2 * %d + 1]: %d\n", i, candidates[2 * i + 1]);}
            classCounts[(int)candidates[2 * i + 1]] += 1;
        }
        printf("made it to e! rank %d", tid);

        
        int max = -1;
        int max_index = 0;
        for (int i = 0; i < n_classes; i++) {
            if (classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }
//        if (tid == 0 && keyIndex == n_train - 1) {printf("made it to e!\n");}
        predictions[tid] = max_index;
        for (int i = 0; i < 2 * k; i++) {candidates[i] = FLT_MAX;}
        memset(classCounts, 0, n_classes * sizeof(int));
        }
    }
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset) {
    // matrix size numberClasses x numberClasses
    int* confusionMatrix = (int*)malloc(dataset->num_classes() *
    		dataset->num_classes() * sizeof(int));
    printf("made it to f!\n");
    for (int i = 0; i < dataset->num_instances(); i++) {
        // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(
        		dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    printf("made it to g!\n");
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset) {
    int successfulPredictions = 0;
    printf("made it to h!\n");
    for (int i = 0; i < dataset->num_classes(); i++) {
        // elements in the diagonal are correct predictions
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i];
    }
    printf("made it to i!\n");
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: ./KNN datasets/train.arff datasets/test.arff k\n");
        exit(0);
    }
    int k = strtol(argv[3], NULL, 10);
    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData* train = parserTrain.parse();
    ArffData* test = parserTest.parse();
    int test_num = test->num_instances();
    int train_num = train->num_instances();
    int att_num = train->num_attributes();
    int n_classes = train->num_classes();
    printf("made it to k!\n");
    // predictions is the array where you have to return the class
    	// predicted (integer) for the test dataset instances
    int* h_predictions = (int*)malloc(test_num * sizeof(int));
    float* h_test_floats = (float*)malloc(test_num * att_num * sizeof(float));
    float* h_train_floats = (float*)malloc(train_num * att_num * sizeof(float));
    for (int i = 0; i < test_num; i++) {
        for (int j = 0; j < att_num; j++) {
            h_test_floats[i * att_num + j] = test->get_instance(i)->get(j)
            										->operator float();
        }
    }
    printf("made it to l!\n");
    for (int i = 0; i < train_num; i++) {
        for (int j = 0; j < att_num; j++) {
           h_train_floats[i * att_num + j] = train->get_instance(i)->get(j)
        											->operator float();
        }
    }
    printf("made it to m!\n");
    float* d_test_floats, * d_train_floats, * candidates;
    int* classCounts, * d_predictions;

    cudaMalloc(&d_train_floats, train_num * att_num * sizeof(float));
    cudaMalloc(&d_test_floats, test_num * att_num * sizeof(float));
    cudaMalloc(&d_predictions, test_num * sizeof(int));
    cudaMalloc(&candidates, k * 2 * sizeof(float));
    cudaMalloc(&classCounts, test_num * sizeof(int));
    printf("made it to n!\n");
    cudaMemcpy(d_test_floats, h_test_floats, att_num * test_num * sizeof(float),
    		cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_floats, h_train_floats, att_num * train_num * sizeof(float),
    		cudaMemcpyHostToDevice);
    int threads_per_block = 128;
    int grid_size = (test_num + threads_per_block - 1) / threads_per_block;
    printf("made it to o!\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    KNN<<<grid_size, threads_per_block>>>(d_test_floats, d_train_floats, d_predictions,
    		candidates, classCounts, k, test_num, train_num, att_num, n_classes);
    cudaMemcpy(h_predictions, d_predictions, test_num * sizeof(int),
    		cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("made it to o!\n");

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(h_predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);
    cudaFree(d_test_floats);
    cudaFree(d_train_floats);
    cudaFree(d_predictions);
    cudaFree(d_train_floats);

    free(h_test_floats);
    free(h_train_floats);
    free(h_predictions);
    cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError,
				cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) +
    		end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %d test instances on"
    		" %d train instances required %llu ms CPU time. Accuracy was %.4f\n",
            k, test_num, train_num, (long long unsigned int)diff, accuracy);
}

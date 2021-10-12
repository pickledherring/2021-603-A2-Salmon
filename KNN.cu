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

__global__ void KNN(float** test, float** train, float* predictions,
					int k,  int n_test, int n_train, int n_classes) {
    // Implements a parallel kNN where for each candidate query an
	// in-place priority queue is maintained to identify the kNN's.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 1) {
        printf("got to g");
    }
    // stores k-NN candidates for a query vector as a sorted 2d array.
    // First element is inner product, second is class.
    float* candidates = (float*)malloc(k * 2 * sizeof(float));
    for (int i = 0; i < 2 * k; i++) {candidates[i] = 99999999.9;}
    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)malloc(n_classes * sizeof(int));
    if (tid == 1) {
        printf("got to h");
    }
    if (tid < n_test) {
        for (int keyIndex = 0; keyIndex < n_train; keyIndex++) {
        	float dist = 0;
			for (int i = 0; i < n_classes - 1; i++) {
				float diff = test[tid][i] - train[keyIndex][i];
				dist += diff * diff;
			};
            // Add to our candidates
            for (int c = 0; c < k; c++){
                if (dist < candidates[2 * c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    // class value
                    candidates[2 * c + 1] = train[keyIndex][n_classes - 1];
                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for (int i = 0; i < k; i++) {
            classCounts[(int)candidates[2 * i + 1]] += 1;
        }
        
        int max = -1;
        int max_index = 0;
        for (int i = 0; i < n_classes; i++) {
            if (classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[tid] = max_index;
        for (int i = 0; i < 2 * k; i++) {candidates[i] = 99999999.9;}
        memset(classCounts, 0, n_classes * sizeof(int));
    }
    if (tid == 1) {
        printf("got to i");
    }
}

int* computeConfusionMatrix(float* predictions, ArffData* dataset) {
    // matrix size numberClasses x numberClasses
    int* confusionMatrix = (int*)malloc(dataset->num_classes() *
    		dataset->num_classes() * sizeof(int));
    
    for (int i = 0; i < dataset->num_instances(); i++) {
        // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(
        		dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset) {
    int successfulPredictions = 0;
    
    for (int i = 0; i < dataset->num_classes(); i++) {
        // elements in the diagonal are correct predictions
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i];
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char* argv[]) {
	printf("got to a!\n");
    if (argc != 4) {
        printf("Usage: ./KNN datasets/train.arff datasets/test.arff k\n");
        exit(0);
    }
    printf("got to b!\n");
    int k = strtol(argv[3], NULL, 10);
    printf("k: %d\n", k);
    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData* train = parserTrain.parse();
    ArffData* test = parserTest.parse();
    printf("got to c!\n");
    int test_num = test->num_instances();
    int train_num = train->num_instances();
    int att_num = train->num_attributes();
    printf("got to d!\n");
    printf("test_num: %d\ntrain_num: %d\natt_num: %d\n", test_num,
    		train_num, att_num);
    // predictions is the array where you have to return the class
    	// predicted (integer) for the test dataset instances
    float* h_predictions = (float*)malloc(test_num * sizeof(float*));
    float* h_test_floats = (float*)malloc(test_num * att_num * sizeof(float*));
    float* h_train_floats = (float*)malloc(train_num * att_num * sizeof(float*));
    for (int i = 0; i < test_num; i++) {
        for (int j = 0; j < att_num; j++) {
            h_test_floats[i * att_num + j] = test->get_instance(i)->get(j)
            										->operator float();
        }
    }
    for (int i = 0; i < train_num; i++) {
        for (int j = 0; j < att_num; j++) {
           h_train_floats[i * att_num + j] = train->get_instance(i)->get(j)
        											->operator float();
        }
    }
    printf("got to e!\n");
    float* d_test_floats, * d_train_floats, * d_predictions;

    cudaMalloc(&d_train_floats, train_num * att_num * sizeof(float));
    cudaMalloc(&d_test_floats, test_num * att_num * sizeof(float));
    cudaMalloc(&d_predictions, test_num * sizeof(float));

    cudaMemcpy(d_test_floats, h_test_floats, att_num * test_num * sizeof(
    		float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_floats, h_train_floats, att_num * train_num * sizeof(
    		float), cudaMemcpyHostToDevice);
    printf("got to f!\n");
    int threads_per_block = 64;
    int grid_size = (test_num + threads_per_block + 1) / threads_per_block;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    printf("got to g!\n");
    KNN<<<grid_size, threads_per_block>>>(d_test_floats, d_train_floats,
    		d_predictions, k, test_num, train_num, att_num);
    cudaMemcpy(h_predictions, d_predictions, test_num * sizeof(float),
    		cudaMemcpyDeviceToHost);
    printf("got to h!\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError,
        		cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(h_predictions, test);
    printf("got to i!\n");
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);
    printf("got to j!\n");
    cudaFree(d_test_floats);
    cudaFree(d_train_floats);
    cudaFree(d_predictions);

    free(h_test_floats);
    free(h_train_floats);
    free(h_predictions);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) +
    		end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %lu test instances on"
    		"%lu train instances required %llu ms CPU time. Accuracy was %.4f\n",
            k, test_num, train_num, (long long unsigned int) diff, accuracy);
}

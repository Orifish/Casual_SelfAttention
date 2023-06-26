#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void CasualSA_Kernal(float* QKV,int N,int C,float* output,int* class_index,int* index_num,int class_num,int* Origin,float* row_sum,int* row_max,int N_max){
    
// we treat each row as a block. Each Thread may solve more than one element if the row size of the Softmax Matrix is larger than 1024.  
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    unsigned int stride = blockDim.x;       // if each Thread need to solve more than one element, the stride should be the blockDim
    int row_NMax = blk_idx;         // The blk_idx represents this thread is which row in the Softmax Matrix.
    int class_id = row_NMax / N_max;        // We use the max number of all group to build the grid, so the each class have the same number of blocks: N_max.
    int row_thisclass = row_NMax%N_max;
    int col_thisclass = thd_idx;        // each thread present one element, the col of the element is just the threadIdx of the thread. Additionally, if a thread need to solve more than one element, this varible will be update during the thread.

    if(row_thisclass>=index_num[class_id])      // The thread of a block may not be used, because the block number is ruled by the max number of all groups.
        return;

    // The code below is to get the virtual row of this thread.
    // To understand this, consider the Softmax Matrix, we want to get this element is got by the multiply of which row of Query and which col of Key.
    // So we need to know if the Matrix is just the N*N, which row the thread should belong to.
    // And we just need to know which class(group number) it is, which row it is in his class. And then add it of all class in front of it.(because we have sorted the input Tensor)
    int row_NC = row_thisclass;
    for(int i=0;i<class_id;i++){
        row_NC += index_num[i];
    }

    // Calculate the element and get the max&sum value of each row.
    while(col_thisclass<index_num[class_id]){
        int col_NC = col_thisclass;
        for(int i=0;i<class_id;i++){
            col_NC += index_num[i];
        }

        float mid=0;        // varible which restore the result of Q*K' .
        for(int i=0;i<C;i++){
            mid  += QKV[row_NC*C + i]*QKV[col_NC*C + i];
        }
        int Compare = mid * 10000000;           // To restore the max value, we can just get the int type by "atomicMax", so we make it float type manually.
        atomicMax(&row_max[row_NC],Compare);            // save the max of every element without exp
        atomicAdd(&row_sum[row_NC],exp(mid));           // save the sum of e^x
        col_thisclass += stride;        // add the stride, to solve more than one element.
    }
    __syncthreads();        // wait one line(block) OK

    col_thisclass = thd_idx;        // refresh the col id
    int row_target = Origin[row_NC];        // get the original row before sorted.
    while(col_thisclass<index_num[class_id]){
        int col_NC = col_thisclass;
        for(int i=0;i<class_id;i++){
            col_NC += index_num[i];
        }
        float mid=0;        // Calculate the Q*K' again.
        for(int i=0;i<C;i++){
            mid  += QKV[row_NC*C + i]*QKV[col_NC*C + i];
        }

        // Calculate the Softmax and softmax(Q*K')*V
        for(int i=0;i<C;i++){
            float row_max_float_pointfront = row_max[row_NC]/10000000;      // get the header of float
            float row_max_float_back = row_max[row_NC]%10000000;        // get the back of float
            row_max_float_back =row_max_float_back/10000000.0;     
            float row_max_float = row_max_float_pointfront +  row_max_float_back;        // add them to get the float max value
            float to_add = QKV[col_NC*C+i]*(exp(mid-row_max_float)/(row_sum[row_NC]/exp(row_max_float)));       // this part direct the "e^(x-max)/(e^sum/e^max)"
                                                                                                                // the reasong why we use sum/e^max is that we don't want to sync twice.
            atomicAdd(&output[row_target*C+i],to_add);          // Put the result to the target_row directly. Get the result.


        }
        col_thisclass += stride;
    }
}


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,int* row_max,int N_max) {
    cudaError_t err;

    dim3 blocks(class_num*N_max); 
    dim3 threads(THREADS_PER_BLOCK);

    CasualSA_Kernal<<<blocks,threads>>>(QKV,N,C,output,class_index,index_num,class_num,Origin,row_sum,row_max,N_max);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}

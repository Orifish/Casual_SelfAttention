#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ int wait_num;
__global__ void CasualSA_Kernal(float* QKV,int N,int C,float* output,int* class_index,int* index_num,int class_num,int* Origin,float* row_sum,int* row_max,int* index_num_2,int Thread_num,int N_max){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    unsigned int stride = blockDim.x;
    int row_NMax = blk_idx;
    int class_id = row_NMax / N_max;
    int row_thisclass = row_NMax%N_max;
    int col_thisclass = thd_idx;

    if(row_thisclass>=index_num[class_id])
        return;
    int row_NC = row_thisclass;
    for(int i=0;i<class_id;i++){
        row_NC += index_num[i];
    }
    while(col_thisclass<index_num[class_id]){
        int col_NC = col_thisclass;
        for(int i=0;i<class_id;i++){
            col_NC += index_num[i];
        }
        float mid=0;
        for(int i=0;i<C;i++){
            mid  += QKV[row_NC*C + i]*QKV[col_NC*C + i];
        }
        int Compare = mid * 10000000;
        atomicMax(&row_max[row_NC],Compare);            // save the max of every element without exp
        atomicAdd(&row_sum[row_NC],exp(mid));
        col_thisclass += stride;
    }
    __syncthreads();        // wait one line(block) OK
    col_thisclass = thd_idx;        // refresh the col id
    int row_target = Origin[row_NC];
    while(col_thisclass<index_num[class_id]){
        int col_NC = col_thisclass;
        for(int i=0;i<class_id;i++){
            col_NC += index_num[i];
        }
        float mid=0;
        for(int i=0;i<C;i++){
            mid  += QKV[row_NC*C + i]*QKV[col_NC*C + i];
        }
        for(int i=0;i<C;i++){
            float row_max_float_pointfront = row_max[row_NC]/10000000;
            float row_max_float_back = row_max[row_NC]%10000000;
            row_max_float_back =row_max_float_back/10000000.0;
            float row_max_float = row_max_float_pointfront +  row_max_float_back;
            float to_add = QKV[col_NC*C+i]*(exp(mid-row_max_float)/(row_sum[row_NC]/exp(row_max_float)));
            atomicAdd(&output[row_target*C+i],to_add);


        }
        col_thisclass += stride;
    }
}


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,int* row_max,int* index_num_2,int Thread_num,int N_max) {
    cudaError_t err;



    dim3 blocks(class_num*N_max); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    int value=0;
    cudaMemcpyToSymbol(wait_num, &value, sizeof(int));     // initialize the wait_num
    CasualSA_Kernal<<<blocks,threads>>>(QKV,N,C,output,class_index,index_num,class_num,Origin,row_sum,row_max,index_num_2,Thread_num,N_max);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}

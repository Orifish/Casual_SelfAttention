#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ int wait_num;
__global__ void CasualSA_Kernal(float* QKV,int N,int C,float* output,int* class_index,int* index_num,int class_num,int* Origin,float* row_sum,float* row_max,int* index_num_2,int Thread_num){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if(idx>=Thread_num){
        return;
    }else{
        int class_id = 0;
        int sum = 0;
        for(int i=0;i<class_num;i++){
            if(idx<index_num_2[i]){
                class_id = i-1;
                sum = index_num_2[class_id];
                break;
            }
            class_id = class_num-1;
            sum = index_num_2[class_id];
        }
        int idx_N_row = 0;
        for(int i=0;i<class_id;i++){
            idx_N_row += index_num[i];
        }
        int idx_N_col = idx_N_row + (idx-sum)%index_num[class_id];
        idx_N_row += (idx-sum)/index_num[class_id];
        float mid_NN = 0;
        for(int i=0;i<C;i++){
            mid_NN += QKV[idx_N_row*C + i] * QKV[idx_N_col*C + i];
        }
        


        atomicAdd(&row_sum[idx_N_row],exp(mid_NN));
        atomicAdd(&wait_num,1);       // 开始等待
        while(wait_num!=Thread_num);     // 同步整个grid

        int row_target = Origin[idx_N_row];

        for(int i=0;i<C;i++){
            float to_add = QKV[idx_N_row*C+i]*(exp(mid_NN)/row_sum[idx_N_row]);
            atomicAdd(&output[row_target*C+i],to_add);
        }
            /*
            for(int i=0;i<C;i++){
                atomicAdd(&output[row_target*C+col],QKV[row*C+i]*((mid_NN-row_max[row])/row_sum[row]));
            }*/
    }
}


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,float* row_max,int* index_num_2,int Thread_num) {
    cudaError_t err;

    dim3 blocks(DIVUP(Thread_num, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    int value=0;
    cudaMemcpyToSymbol(wait_num, &value, sizeof(int));     // 初始化wai_num
    CasualSA_Kernal<<<blocks,threads>>>(QKV,N,C,output,class_index,index_num,class_num,Origin,row_sum,row_max,index_num_2,Thread_num);     
    // cudaFree(Sort_Matrix); 
    // cudaFree(Origin);
    // SA_forward_kernel<<<blocks, threads>>>(batch_size, class_num,N,C,class_index,index_num,Q,K,V,next_index,output);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    //cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}

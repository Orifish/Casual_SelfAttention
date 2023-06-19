#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ long wait_num;
__global__ void CasualSA_Kernal(float* QKV,int N,int C,float* output,int* class_index,int* index_num,int class_num,int* Origin,float* row_sum，float* row_max){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if(idx>=N*N){
        return;
    }else{
        int row = idx/N;
        int col = idx%N;
        int this_class = class_index[row];
        int this_start_row = 0;
        int next_start_row = 0;
        if(this_class+1>class_num){
            for (int i=this_class;i>0;i--){
                this_start_row += index_num[this_class];
            }
            next_start_row = 0;
        }
        else{
            for (int i=this_class;i>0;i--){
                this_start_row += index_num[this_class];
            }
            for (int i=this_class;i>0;i--){
                next_start_row += index_num[this_class+1];
            }
        }
            if(row<this_start_row){
                return;
            }  
            if(row>=next_start_row){
                return;             
            }           
            float mid_NN = 0;
            for(int i=0;i<C;i++){
                mid_NN += QKV[row*C + i] * QKV[row*C + i];
            }
            atomicMax(&row_max[row],mid_NN);
            wait_num += 1;       // 开始等待
            while(wait_num!=N*N);     // 同步整个grid
            atomicAdd(&row_sum[row],exp(mid_NN-row_max[row]));
            wait_num += 1;
            while(wait_num!=2*N*N);

            output[row*C+col] = 0;
            int row_target = Origin[row];
            for(int i=0;i<C;i++){
                atomicAdd(&output[row_target*C+col],QKV[row*C+i]*((mid_NN-row_max[row])/row_sum[row]));
            }
    }
}

// sort_num是每一个行对应类，在自己类中的排序，在pytorch中给出，可以用==index来得到。
// index_num是已经降序排序完的索引对应的数量
void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,float* row_max) {
    cudaError_t err;

    dim3 blocks(DIVUP(N * N, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    long value=0;
    cudaMemcpyToSymbol(wait_num, &value, sizeof(long));     // 初始化wai_num
    CasualSA_Kernal<<<blocks,threads>>>(QKV,N,C,output,class_index,index_num,class_num,Origin,row_sum,row_max);     
    // cudaFree(Sort_Matrix); 
    // cudaFree(Origin);
    // SA_forward_kernel<<<blocks, threads>>>(batch_size, class_num,N,C,class_index,index_num,Q,K,V,next_index,output);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ long wait_num;
/*
__global__ void SortCopy_Kernal(float* Sort_Matrix,int N,int C,float* QKV,int* index_num,int* class_index,int* sort_num,int* Orign){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if (idx >= N * C) {
    return;
    }else{
        int row = idx/C;
        int true_class = class_index[row];      // 知道这行是什么class
        int start_row = 0;
        for (int i=true_class;i>0;i--){
            start_row += index_num[true_class];
        }       // 定这种类别的初始行
        int row_target = start_row + sort_num[row];       // 得到我应该把这一行给复制到哪一行

        for (int i = 0;i<C;i++){
            Sort_Matrix[row_target*C + i] = QKV[row*C+i]       // copy过去
        }
        Orign[row_target] = row;        // 把原本的位置保存
    }
}*/




__global__ void CasualSA_Kernal(float* QKV,int N,int C,float* output,int* class_index,int* index_num,int class_num,int* Origin){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if(idx>=N*C){
        return;
    }else{
        int row = idx/C;
        int col = idx%C;
        int next_class = class_index[row]+1;      // 知道这行是什么class，并且因为要计算下一个index的起始行，所以要+1
        int this_class = class_index[row];

        if(next_claas>class_num){       // 如果已经是最后一类了
            int this_start_row = 0;
            int next_start_row = N;
            for (int i=true_class;i>0;i--){
                this_start_row += index_num[this_class];
            }       // 定这种类别和下个类别的初始行
        }     
        else{
            int this_start_row = 0;
            int next_start_row = 0;
            for (int i=true_class;i>0;i--){
                this_start_row += index_num[this_class];
                next_start_row += ineex_num[next_class];
            }       // 定这种类别和下个类别的初始行
        }
            if(row<this_start_row)  return;
            if(row>=next_start_row)     return;         // 注意等于号，此处欠推导
            float mid_NN = 0;
            for(int i=0;i<C;i++){
                mid_NN += QKV[row*C + i] * QKV[row*C + i];       // 不同于广义的矩阵乘法，我们是能避免转置的，这里的QKV应该是Q和K
            }
            // 得到了矩阵成积结果了，开始Softmax过程，每个Thread负责一个输出元素。到这里的已经是在需要输出的范围内的数据了。
            atomicAdd(&Block_sum,exp(mid_NN));
            waitnum += 1;       // 开始等待
            while(waitnum<N*N);     // 同步整个块
            // __syncthreads();        // 此处需要保证一个block一定是一行，这块应该不行。
            // Softmax完成，注意，这里感觉不行，因为默认了一个Block对应一行，具体分配规则待考虑。
            output[row*C+col] = 0;
            int row_target = Orign[row];
            for(int i=0;i<C;i++){
                atomicAdd(&output[row_target*C+col],QKV[row*C+i]*(mid_NN/Block_sum));       
            }   // 此外，这里的QKV应该是V，并且这里直接还原回了原行   
    }
}

// sort_num是每一个行对应类，在自己类中的排序，在pytorch中给出，可以用==index来得到。
// index_num是已经降序排序完的索引对应的数量
void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin) {
    cudaError_t err;

    // dim3 blocks(DIVUP(N * C, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(DIVUP(m, n)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    /*
    float* Sort_Matrix;
    int* Orign;       // 记录原本是哪行的

    cudaMallocManaged(&Sort_Matrix, N*C * sizeof(float));
    cudaMallocManaged(&Orign, N * sizeof(int));

    SortCopy_Kernal<<<blocks,threads>>>(Sort_Matrix,N,C,QKV,index_num,class_index,sort_num,Orign);        // 注意index_num必须要已经降序排序完了。
    cudaDeviceSynchronize();
    */
    long value=0;
    cudaMemcpyToSymbol(wait_num, &value, sizeof(long));     // 初始化wai_num
    CasualSA_Kernal<<<blocks,threads>>>(QKV,N,C,output,class_index,index_num,class_num,Origin);     
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

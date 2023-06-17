#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>
#include <cublas_v2.h>


# define THREADS_PER_BLOCK 1024
# define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))



__global__ void casualSA_forward_kernel(int class_num,int N,int C, int* class_index,int* index_num, float* Q,float* K,float* V,int* next_index) {
    // 每个线程完成一个
    // 输入格式，Q、K、V为N*C的Tensor，next_index为N*1的tensor，对应每一行的类对应下一个自己类的索引，获得方法是在pytorch中索引==然后左移一格，class_index为N*1的tensor，对应每一行的类
    // output要预先在pytorch中建立好，大小与QKV相同
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if (idx >= batch_size * N * C) {
    return;
    } else {
    // 均假设是先横着走再竖着走
    int row = idx/C;
    int col = idx%C;
    int N_class = class_index[row];     // 确认第几行，是哪一类
    int next_pos = next_index[row];     // 确定下一个自己类是哪一行
    for (int i=0;i<N;i++){
        Q[row*C+col]*K[col*C+row];
    }
    } 
}


__global__ void SortCopy_Kernal(float* Sort_Matrix,int N,int C,float* QKV,int* index_num,int* class_index,int* sort_num,int* Orign){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    if (idx >= batch_size * N * C) {
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
}


__global__ void Matrix_Mul(float* QKV,int N,int C,float* output){
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int idx = blk_idx * blockDim.x + thd_idx;
    int row = idx/C;
    int col = idx%C;
    output[row][col] = 0;
    for(int i=0;i<C;i++){
        output[row][col] = QKV[row][i] * QKV[row][i];       // 不同于广义的矩阵乘法，我们是能避免转置的

    }
    
}


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,int* next_index,float* output,int* sort_num) {
    cudaError_t err;

    dim3 blocks(DIVUP(N * C, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    float* Sort_Matrix[N][C];
    int* Orign[N][1];       // 记录原本是哪行的

    cudaMallocManaged(&Sort_Matrix, N*C * sizeof(float));
    cudaMallocManaged(&Orign, N * sizeof(int));

    SortCopy_Kernal<<<blocks,threads>>>(Sort_Matrix,N,C,QKV,index_num,class_index,sort_num,Orign);        // 注意index_num必须要已经降序排序完了。

    SA_forward_kernel<<<blocks, threads>>>(batch_size, class_num,N,C,class_index,index_num,Q,K,V,next_index,output);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}


// 输入N行C列的Tensor Q、K、V
// sort_num是每一个行对应类，在自己类中的排序，在pytorch中给出，可以用[,:这一行]==index来得到。
// index_num是已经降序排序完的索引对应的数量
int casualSA_forward_wrapper(int class_num,int N,int C,at::Tensor class_index_tensor,at::Tensor index_num_tensor, at::Tensor QKV_tensor,at::Tensor Next_Index,at::Tensor output_tensor,at::Tensor sort_num_tensor) {
    int *class_index = level_end_index_tensor.data_ptr<int>();
    int *index_num = input_features_tensor.data_ptr<float>();
    float *QKV = QKV_tensor.data_ptr<float>();
    int *next_index = Next_Index.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    int* sort_num = sort_num_tensor.data_ptr<int>();

    // cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    casualSA_kernel_forward_launcher(class_num,N,C,class_index,index_num,QKV,next_index,output,sort_num);
    return 1;
}




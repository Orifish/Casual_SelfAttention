#include <torch/extension.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include <cublas_v2.h>   // 思考要不要用这个做矩阵乘法


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,int* next_index,float* output,int* sort_num);


// 输入N行C列的Tensor Q、K、V
// sort_num是每一个行对应类，在自己类中的排序，在pytorch中给出，可以用[,:这一行]==index来得到。
// index_num是已经降序排序完的索引对应的数量
int casualSA_forward_wrapper(int class_num,int N,int C,at::Tensor class_index_tensor,at::Tensor index_num_tensor, at::Tensor QKV_tensor,at::Tensor Next_Index,at::Tensor output_tensor,at::Tensor sort_num_tensor) {
    CHECK_INPUT(class_index_tensor);
    CHECK_INPUT(index_num_tensor);
    CHECK_INPUT(QKV_tensor);
    CHECK_INPUT(Next_Index);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(sort_num_tensor);

    
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



// To do
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("casualSA_forward_wrapper", &casualSA_forward_wrapper, "casualSA_forward_wrapper");
}

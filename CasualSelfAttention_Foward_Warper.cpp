#include <torch/extension.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>
// #include <cublas_v2.h>   // 思考要不要用这个做矩阵乘法


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,float* row_max);


int casualSA_forward_wrapper(int class_num,int N,int C,at::Tensor class_index_tensor,at::Tensor index_num_tensor, at::Tensor QKV_tensor,at::Tensor output_tensor,at::Tensor Origin_tensor,at::Tensor row_sum_tensor,at::Tensor row_max_tensor) {
    CHECK_INPUT(class_index_tensor);
    CHECK_INPUT(index_num_tensor);
    CHECK_INPUT(QKV_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(Origin_tensor);
    CHECK_INPUT(row_sum_tensor);
    CHECK_INPUT(row_max_tensor);

    
    int *class_index = class_index_tensor.data_ptr<int>();    // N*1的组别tensor
    int *index_num = index_num_tensor.data_ptr<int>();     // 组数*1的tensor
    float *QKV = QKV_tensor.data_ptr<float>();      // 使用应分别为QKV，N*C tensor
    float *output = output_tensor.data_ptr<float>();      // N*C tensor
    int* Origin = Origin_tensor.data_ptr<int>();   // 输入的原index，N*1
    float* row_sum = row_sum_tensor.data_ptr<float>();    // 输入全0的row_sum;
    float* row_max = row_max_tensor.data_ptr<float>();    // 输入全0的row_sum;

    // cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    casualSA_kernel_forward_launcher(class_num,N,C,class_index,index_num,QKV,output,Origin,row_sum,row_max);
    return 1;
}



// To do
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("casualSA_forward_wrapper", &casualSA_forward_wrapper, "casualSA_forward_wrapper");
}
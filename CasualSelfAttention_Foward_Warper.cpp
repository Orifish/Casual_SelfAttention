#include <torch/extension.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void casualSA_kernel_forward_launcher(int class_num,int N,int C, int* class_index,int* index_num, float* QKV,float* output,int* Origin,float* row_sum,int* row_max,int N_max);


int casualSA_forward_wrapper(int class_num,int N,int C,at::Tensor class_index_tensor,at::Tensor index_num_tensor, at::Tensor QKV_tensor,at::Tensor output_tensor,at::Tensor Origin_tensor,at::Tensor row_sum_tensor,at::Tensor row_max_tensor,int N_max) {
    CHECK_INPUT(class_index_tensor);
    CHECK_INPUT(index_num_tensor);
    CHECK_INPUT(QKV_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(Origin_tensor);
    CHECK_INPUT(row_sum_tensor);
    CHECK_INPUT(row_max_tensor);


    int *class_index = class_index_tensor.data_ptr<int>();    // group number tensor, size: N*1, which N represents the total row of the input Query
    int *index_num = index_num_tensor.data_ptr<int>();     // tensor, size: class_num*1.
    float *QKV = QKV_tensor.data_ptr<float>();      // input Query,Key,Value. Now they are represented by one varible "QKV"
    float *output = output_tensor.data_ptr<float>();      // the output tensor which is prepared in advance, size: N*C.
    int* Origin = Origin_tensor.data_ptr<int>();   // The original index before the tensor was sorted by the group number, size: N*1.
    float* row_sum = row_sum_tensor.data_ptr<float>();    // zeros tensor, which is prepared in advance to get the sum value of each row during Softmax, size: N*1.
    int* row_max = row_max_tensor.data_ptr<int>();    // zeros tensor, which is prepared in advance to get the max value of each row during Softmax, size N*1.

    casualSA_kernel_forward_launcher(class_num,N,C,class_index,index_num,QKV,output,Origin,row_sum,row_max,N_max);
    return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("casualSA_forward_wrapper", &casualSA_forward_wrapper, "casualSA_forward_wrapper");
}
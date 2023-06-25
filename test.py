import torch
import CasualSelfAttention
import torch.nn.functional as F
import time


def SA(QKV):
    QKV_ = torch.matmul(QKV,QKV.transpose(0,1))        # A*A矩阵
    QKV_ = torch.nn.functional.softmax(QKV_,dim=1)      # 对行求softmax
    QKV_ = torch.matmul(QKV_,QKV)

    return QKV_

def CasualSelfAttention_(QKV,class_index):
    N,C = QKV.size()
    _,Origin_index = torch.sort(class_index,dim=0)
    input_Query = QKV[Origin_index]     # 排序后的输入
    output = torch.zeros(input_Query.size())
    class_num = class_index.unique().size()[0]
    _,index_num = torch.unique(torch.sort(class_index,dim=0)[0],dim=0,return_counts=True)
    class_index = torch.sort(class_index,dim=0)[0]
    row_sum = torch.zeros(N,1)
    row_max = torch.zeros(N,1)
    index_num_2 = torch.cumsum(index_num * index_num, dim=0)
    N_max =  int(torch.max(index_num))
    Thread_num = int(class_num * N_max *N_max)
    index_num_2[1:int(index_num_2.size()[0])] = index_num_2.clone()[0:int(index_num_2.size()[0] - 1)]
    index_num_2[0] = 0

    Index = torch.cumsum(index_num,dim=0)
    Index[1:int(Index.size()[0])] = Index.clone()[0:int(Index.size()[0] - 1)]
    Index[0] = 0
    input_Query = torch.transpose(input_Query,0,1)[0]
    output = torch.transpose(output,0,1)[0]
    class_index = class_index.int().cuda()
    index_num = index_num.int().cuda()
    input_Query = input_Query.cuda()
    output = output.cuda()
    output_Pytorch = output.clone()
    Origin_index = Origin_index.int().cuda()
    row_sum = row_sum.cuda()
    row_max =row_max.int().cuda()
    index_num_2 = index_num_2.int().cuda()
    begin = time.time_ns()
    CasualSelfAttention.casualSA_forward_wrapper(class_num,N,C,class_index,index_num,input_Query,output,Origin_index,row_sum,row_max,index_num_2,Thread_num,N_max)
    end = time.time_ns()
    print('kernel takes:',(end-begin)/1000000.0,'ms')
    input_Query_pytorch = input_Query.clone()

    begin = time.time_ns()
    for i in torch.unique(class_index):
        if i < class_num-1:
            input_Query_pytorch[Index[i]:Index[i + 1] , :] = SA(
                input_Query_pytorch[Index[i]:Index[i + 1], :])
        else:
            input_Query_pytorch[Index[i]:N, :] = SA(input_Query_pytorch[Index[i]:N, :])
    a = 0
    for origin in Origin_index:
        output_Pytorch[int(origin), :] = input_Query_pytorch[a, :]
        a += 1
    end = time.time_ns()
    print('Pytorch takes:',(end-begin)/1000000.0,'ms')
    error = torch.sqrt(sum(sum((output - output_Pytorch) * (output - output_Pytorch))) / (N * C))
    error_percent = error/(sum(sum(output))/(N*C))
    print('absolute error is:',float(error.cpu().detach()),'\n')
    print('relative error is:',float(error_percent.cpu().detach()*100),'%\n')
    return output


if __name__ == '__main__':
    N = 10000
    C = 100
    QKV = torch.rand([N, C])
    class_index = torch.randint(0, 20, [N, 1])

    output = CasualSelfAttention_(QKV,class_index)

    # print(output)
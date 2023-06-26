import torch
import CasualSelfAttention
import torch.nn.functional as F
import time


def SA(QKV):
    QKV_ = torch.matmul(QKV,QKV.transpose(0,1))        # matmul to get A*A Matrix, 'A' represents how many rows are in this group.
    QKV_ = torch.nn.functional.softmax(QKV_,dim=1)      # Softmax to each row
    QKV_ = torch.matmul(QKV_,QKV)       # matmul to get A*C Matrix

    return QKV_

def CasualSelfAttention_(QKV,class_index):
    N,C = QKV.size()
    _,Origin_index = torch.sort(class_index,dim=0)      # get the Origin_index of each row
    input_Query = QKV[Origin_index]     # sort the input by group
    output = torch.zeros(input_Query.size())        # prepare the output tensor 
    class_num = class_index.unique().size()[0]      # get the number of class
    _,index_num = torch.unique(torch.sort(class_index,dim=0)[0],dim=0,return_counts=True)       # get the number of each group
    class_index = torch.sort(class_index,dim=0)[0]      # get the sorted class tensor, size: N*1
    row_sum = torch.zeros(N,1)      # prepare the zeros tensor to restore sum value of each row.
    row_max = torch.zeros(N,1)      # prepare the zeros tensor to restore max value of each row.
    N_max =  int(torch.max(index_num))      # get the max number of groups, to build the block.
    Index = torch.cumsum(index_num,dim=0)       # to get the cumsum of the index number, used in Pytorch Method.
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
    input_Query_pytorch = input_Query.clone()


    # Calculate by CUDA
    begin = time.time_ns()
    CasualSelfAttention.casualSA_forward_wrapper(class_num,N,C,class_index,index_num,input_Query,output,Origin_index,row_sum,row_max,N_max)
    end = time.time_ns()
    print('kernel takes:',(end-begin)/1000000.0,'ms')

    # Calculate by Pytorch
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

    # show the result
    error = torch.sqrt(sum(sum((output - output_Pytorch) * (output - output_Pytorch))) / (N * C))
    Max_error = torch.max(torch.max(abs(output - output_Pytorch)))
    error_percent = error/(sum(sum(output))/(N*C))
    error_Maxpercent = Max_error/(sum(sum(output))/(N*C))
    print('absolute error is:',float(error.cpu().detach()),'\n')
    print('relative error is:',float(error_percent.cpu().detach()*100),'%\n')
    print('absolute Max error is:',float(Max_error.cpu().detach()),'\n')
    print('relative Max error is:',float(error_Maxpercent.cpu().detach()*100),'%\n')

    return output


if __name__ == '__main__':
    N = 10000
    C = 100
    QKV = torch.rand([N, C])
    class_index = torch.randint(0, 20, [N, 1])

    output = CasualSelfAttention_(QKV,class_index)

    # print(output)
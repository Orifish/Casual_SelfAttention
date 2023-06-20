import torch
import CasualSelfAttention
import torch.nn.functional as F
import time

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

    input_Query = torch.transpose(input_Query,0,1)[0]
    output = torch.transpose(output,0,1)[0]
    class_index = class_index.int().cuda()
    index_num = index_num.int().cuda()
    input_Query = input_Query.cuda()
    output = output.cuda()
    Origin_index = Origin_index.int().cuda()
    row_sum = row_sum.cuda()
    row_max =row_max.cuda()
    begin = time.time_ns()
    CasualSelfAttention.casualSA_forward_wrapper(class_num,N,C,class_index,index_num,input_Query,output,Origin_index,row_sum,row_max)
    end = time.time_ns()
    print('kernel takes:',(end-begin)/1000000.0,'ms')
    # a = F.softmax(input_Query[class_index[class_index==0].long(),:].matmul(input_Query[class_index[class_index==0].long(),:].transpose(0,1)),dim=1).matmul(input_Query[class_index[class_index==0].long(),:])

    return output


if __name__ == '__main__':
    N = 10000
    C = 100
    QKV = torch.rand([N, C])
    class_index = torch.randint(0, 2, [N, 1])

    output = CasualSelfAttention_(QKV,class_index)

    print(output)
import torch
import CasualSelfAttention


if __name__ == '__main__':
    N = 10000
    C = 100
    QKV = torch.rand([N, C])
    class_index = torch.randint(0, 20, [N, 1])
    _,Origin_index = torch.sort(class_index,dim=0)
    input_Query = QKV[Origin_index]     # 排序后的输入
    output = torch.zeros(input_Query.size())
    class_num = class_index.unique().size()[0]
    _,index_num = torch.unique(torch.sort(class_index)[0],dim=0,return_counts=True)
    row_sum = torch.zeros(N,1)
    CasualSelfAttention.casualSA_forward_wrapper(class_num,N,C,class_index,index_num,input_Query,output,Origin_index,row_sum)
    print(output)
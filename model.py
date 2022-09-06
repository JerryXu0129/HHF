from config import *

class HHF(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, num_bits).to(device))
        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')

    def forward(self, x = None, x_ = None, batch_y = None, batch_y_ = None, reg = None):
        if dataset not in ['cifar10', 'cifar100']:
            P_one_hot = batch_y
            if based_method == 'pair':
                P_one_hot_ = batch_y_
        else:
            P_one_hot = torch.from_numpy(np.eye(num_classes)[batch_y.cpu().numpy()]).to(device) 
            if based_method == 'pair':
                P_one_hot_ = torch.from_numpy(np.eye(num_classes)[batch_y_.cpu().numpy()]).to(device) 

        if based_method == 'proxy':
            P = self.proxies 
            cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(P, p = 2, dim = 1).T)       
        elif based_method == 'pair':
            cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)  

        if method == 'anchor':                       
            if HHF_flag:
                pos_exp = torch.exp(alpha * F.relu(1 - delta - cos)) - 1
                neg_exp = torch.exp(alpha * F.relu(cos - threshold - delta)) - 1
            else:     
                pos_exp = torch.exp(alpha * (0.1 - cos))
                neg_exp = torch.exp(alpha * (0.1 + cos))
            P_sim_sum = torch.where(P_one_hot  ==  1, pos_exp, torch.zeros_like(pos_exp)).sum(dim = 0)         
            N_sim_sum = torch.where(P_one_hot  ==  0, neg_exp, torch.zeros_like(neg_exp)).sum(dim = 0)
            pos_term = torch.log(1 + P_sim_sum).sum() / len(torch.nonzero(P_one_hot.sum(dim = 0) !=  0).squeeze(dim = 1))
            neg_term = torch.log(1 + N_sim_sum).sum() / num_classes                

        elif method == 'baseline': 
            if HHF_flag:
                pos = alpha * (1 - cos - delta)
                neg = alpha * F.relu(cos - threshold - delta)
            else:
                pos = alpha * (1 - cos)
                neg = alpha * (1 + cos)

            P_num = len(P_one_hot.nonzero())
            N_num = len((P_one_hot == 0).nonzero())
            pos_term = torch.where(P_one_hot  ==  1, pos, torch.zeros_like(cos)).sum() / P_num
            neg_term = torch.where(P_one_hot  ==  0, neg, torch.zeros_like(cos)).sum() / N_num

        elif method == 'DHN':
            S_ij = P_one_hot.float().mm(P_one_hot_.float().T)
            S_ij = torch.where(S_ij > 0, torch.ones_like(S_ij), torch.zeros_like(S_ij))
            if HHF_flag:
                pos = F.relu(1 - delta - cos)
                neg = F.relu(cos - threshold - delta)
            else:
                pos = torch.log(1 + torch.exp(cos)) - cos
                neg = torch.log(1 + torch.exp(cos))

            S_0 = len((S_ij == 0).nonzero())
            S_1 = len(S_ij.nonzero())
            if S_1 > 0:
                pos_term = torch.where(S_ij  ==  1, pos, torch.zeros_like(cos)).sum() / S_1
            else:
                pos_term = 0
            neg_term = torch.where(S_ij  ==  0, neg, torch.zeros_like(cos)).sum() / S_0

        elif method == 'NCA':
            if HHF_flag:
                pos_term = torch.where(P_one_hot  ==  1, F.relu(1 - cos - delta), torch.zeros_like(cos)).sum()
                neg_term = torch.log(torch.where(P_one_hot  ==  0, torch.exp(F.relu(cos - threshold - delta)), torch.zeros_like(cos)).sum(dim = 1)).sum()
            else:
                pos_term = torch.where(P_one_hot  ==  1, 1 - cos, torch.zeros_like(cos)).sum()
                neg_term = torch.log(torch.where(P_one_hot  ==  0, torch.exp(cos), torch.zeros_like(cos)).sum(dim = 1)).sum()

        loss1 = pos_term + neg_term

        if not reg:
            return loss1

        if based_method == 'pair':
            loss2 = torch.sum(torch.norm(torch.sign(x) - x, dim = 1).pow(2)) + torch.sum(torch.norm(torch.sign(x_) - x_, dim = 1).pow(2))
        else:
            loss2 = torch.sum(torch.norm(torch.sign(x) - x, dim = 1).pow(2))
            # loss2 =  len(x) - torch.sum(F.normalize(x, p = 2, dim = 1) * F.normalize(torch.sign(x), p = 2, dim = 1))
            # print(loss2.detach())
        return loss1 + beta * loss2

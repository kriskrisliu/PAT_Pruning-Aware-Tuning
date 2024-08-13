import torch.nn as nn
import torch
import math

class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass, if necessary
        ctx.save_for_backward(input)
        
        # Forward pass: convert input to 0 or 1 based on condition
        return (input >= 0).float().to(input)  # .float() converts boolean to float (0.0 or 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input, if necessary (not used here)
        input, = ctx.saved_tensors
        
        # Straight through estimator: pass gradient through unchanged
        grad_input = grad_output.clone()
        
        return grad_input

# Alias for easier usage
binarize = BinarizeFunction.apply

class DimDownMask(nn.Module):
    def __init__(self, global_step, alpha, hidden_size, dimdown_dim, trainable_mask):
        super(DimDownMask, self).__init__()
        self.trainable_mask = trainable_mask
        if trainable_mask:
            self.mask = nn.Parameter(torch.zeros((1, 1, hidden_size)))
        else:
            self.mask = None
        self.scalar = torch.ones((1,1,hidden_size))
        self.alpha = alpha
        self.global_step = global_step
        self.dimdown_dim = dimdown_dim
        self.inference_only = False
        self.current_step = 0
    def init_parameters(self):
        nn.init.zeros_(self.mask)

    def forward(self, x):
        return x * self.scalar.to(x)
    
    def step(self, step):
        self.current_step = step
        if self.trainable_mask:
            if self.inference_only:
                temperature = 1e6

                bias = 0.0
                self.scalar = 1.0/(1.0 + torch.exp((-temperature*self.mask).clamp(max=10))) + bias
                self.scalar = self.scalar.round()
            else:
                if step%100==0:
                    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                        print("="*40)
                        print("step =",step, "global step =",self.global_step)
                        # print("mask value:",self.mask.data.sort()[0])
                        hist, bin_edges = torch.histogram(self.mask.data.clone().detach().flatten().cpu().float(), bins=10)
                        hist, bin_edges = hist.tolist(), bin_edges.tolist()
                        result = []
                        for i in range(len(bin_edges)):
                            result.append(bin_edges[i])  # 首先添加list2的元素
                            if i < len(hist):       # 检查list1是否还有元素可供插入
                                result.append(hist[i])  # 插入list1的元素
                        print("mask:", result)
                        print("# value < 0 :", torch.sum(self.mask.data<0))
                        print("*"*40)
                        print("scalar value:",self.scalar.data.sort()[0])
                        print("="*40)
                    elif not torch.distributed.is_initialized():
                        print("="*40)
                        print("mask value:",self.mask)
                        print("*"*40)
                        print("scalar value:",self.scalar)
                        print("="*40)
                temperature = 1.0 / (1 - torch.log(torch.tensor(max(1,step)))/torch.log(torch.tensor(self.global_step)) + 1e-6) if step < self.global_step else 1e6
                bias = max(0.5 - step/self.global_step, 0.0)
                alpha = 0.8
                if step < int(self.global_step*alpha):
                    self.scalar = 1.0/(1.0 + torch.exp((-temperature*self.mask).clamp(max=10))) + bias
                else:
                    step = int(self.global_step*alpha)
                    temperature = 1.0 / (1 - torch.log(torch.tensor(max(1,step)))/torch.log(torch.tensor(self.global_step)) + 1e-6)
                    self.scalar = 1.0/(1.0 + torch.exp((-temperature*self.mask.detach()).clamp(max=10))) + bias

        else:
            if self.inference_only:
                val = torch.tensor(0.0)
            else:
                if step < self.global_step:
                    val = torch.exp(torch.tensor(-self.alpha * (step / self.global_step)))
                else:
                    val = torch.exp(
                        torch.tensor(-self.alpha*2*((step-self.global_step)/self.global_step))
                    ) * torch.exp(torch.tensor(-self.alpha))
            self.scalar.data[:,:,self.dimdown_dim:] = val

    def __repr__(self):
        return f"{self.__class__.__name__}(trainable_mask={self.trainable_mask}, global_step={self.global_step}, alpha={self.alpha}, hidden_size={self.scalar.size(-1)}, dimdown_dim={self.dimdown_dim})"


class DimDownLayer(nn.Module):
    def __init__(self, dimdown_dim, global_step, hidden_size, trainable_mask, register_mask, identity_loss):
        super(DimDownLayer, self).__init__()
        self.trainable_mask = trainable_mask
        eff = 200
        print(f"eff: {eff}")
        self.dimdown_inout = nn.Sequential(
            nn.Linear(hidden_size, eff, bias=False),
            nn.Linear(eff, hidden_size, bias=False),
        )

        self.dimdown_dim = dimdown_dim
        self.register_mask = register_mask
        self.identity_loss = identity_loss
        if register_mask:
            self.dimdown_mask = DimDownMask(
                global_step, alpha=2.77, 
                hidden_size=hidden_size, dimdown_dim=dimdown_dim,
                trainable_mask=trainable_mask
            )
        if identity_loss:
            self.mse_loss = nn.MSELoss()
            self.dimdown_scale = nn.Linear(hidden_size, 1, bias=False)
            nn.init.constant_(self.dimdown_scale.weight, 1)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.kaiming_uniform_(self.dimdown_inout[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.dimdown_inout[1].weight)

        if self.trainable_mask and self.register_mask:
            self.dimdown_mask.init_parameters()
        if self.identity_loss:
            nn.init.constant_(self.dimdown_scale.weight, 1)

    def forward(self, x, dimdown_mask=None, return_loss=False):
        output = self.dimdown_inout(x)
        output += x

        if dimdown_mask is not None:
            output = dimdown_mask(output)

            if self.identity_loss:
                output = self.dimdown_scale.weight.unsqueeze(0) * output
        else:
            output = self.dimdown_mask(output)
            if self.identity_loss:
                output = self.dimdown_scale.weight.unsqueeze(0) * output
        if return_loss:
            dimdown_loss = 0.0
            if dimdown_mask is None:
                dimdown_loss = 10 * torch.abs(binarize(self.dimdown_mask.mask).sum() - self.dimdown_dim)/self.dimdown_dim
                dimdown_loss += ((self.dimdown_mask.mask[self.dimdown_mask.mask>0.0]-0.5)**2).mean()
                dimdown_loss += ((self.dimdown_mask.mask[self.dimdown_mask.mask<=0.0]+0.5)**2).mean()
            
            if self.identity_loss:
                identity_loss = self.mse_loss(self.dimdown_inout[0].weight@self.dimdown_inout[0].weight.T, torch.eye(self.dimdown_inout[0].out_features, device=self.dimdown_inout[0].weight.device))
            else:
                identity_loss = 0.0
            return (
                output, 
                dimdown_loss + identity_loss
            )
        else:
            return output
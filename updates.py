import torch.optim as opt



class Optimizer:
    """
     Custom Optimizer 
    """
    def __init__(self, optimizer, model_params, **optimizer_params):        
        print("Make sure optimizer parameters are initialized appropriately for the chosen optimizer")
        if(optimizer == 'SGD'):
            self.opt = opt.SGD(model_params, **optimizer_params)
        if(optimizer == 'RMS'):
            self.opt = opt.RMSprop(model_params, **optimizer_params)
        if(optimizer == 'Adam'):
            self.opt = opt.Adam(model_params, **optimizer_params)
    
    def lr_decay(self):
        self.opt.param_groups[0]['lr'] = self.opt.param_groups[0]['lr']/2

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()



import torch
import torch.nn as nn
import math

class Cat(nn.Module):
    def __init__(self, dim=0):
        super(Cat, self).__init__()
        self.dim = dim
    def forward(self, tensors):
        return torch.cat(tensors, self.dim)
class Chunk(nn.Module):
    def __init__(self, chunks, dim=0):
        super(Chunk, self).__init__()
        self.chunks = chunks
        self.dim = dim
    
    def forward(self, input):
        return torch.chunk(input, self.chunks, self.dim)

def xigmoid_alpha(s, g=0.01):
    a2 = 2*(s - 1)
    b = (s - 1)*math.log(s - 1)
    # delta = math.square(b) + 4*(s - 1)*g*math.square(s)
    delta = b*b + 4*(s - 1)*g*(s*s)

    alpha = (-b + math.sqrt(delta))/a2
    if alpha < 1.0:
        alpha = 1.0
    elif alpha > 1.7988:
        alpha = 1.7988
    return alpha
def xigmoid(x, alpha=1.0):
    cond = x > 0
    ax = alpha*x
    if_x = torch.exp(ax)
    else_x = 1.0/if_x

    if_x = if_x - 1.0
    else_x = 1.0 - else_x

    cond_x = torch.where(cond, if_x, else_x)
    return torch.sigmoid(alpha*cond_x)
class Xigmoid(nn.Module):
    def __init__(self, alpha=1.0):
        super(Xigmoid, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return xigmoid(x, self.alpha)


################XLSTM#############################################
class LinearForXLSTM(nn.Module):
    #input.shape = (batch_size, in_features)
    #p.shape = (in_features, hidden_size*3)
    #output.shape = (batch_size, hidden_size*3)
    def __init__(self, in_features, hidden_size, alpha, seq_len):
        super(LinearForXLSTM, self).__init__()

        #forget gate initialization
        b = math.log(1.0 + math.log(seq_len - 1)/alpha)/alpha

        f_w = torch.empty((in_features, hidden_size))
        b = math.sqrt(1.0/in_features)*b
        nn.init.uniform_(f_w, a=-b, b=b)
        f_b = torch.zeros((hidden_size, ))

        u = math.sqrt(1.0/hidden_size)
        #output gate initialization
        o_w = torch.empty((in_features, hidden_size))
        o_b = torch.empty((hidden_size, ))
        nn.init.uniform_(o_w, a=-u, b=u)
        nn.init.uniform_(o_b, a=-u, b=u)

        #cell initialization
        c_w = torch.empty((in_features, hidden_size))
        c_b = torch.empty((hidden_size, ))
        nn.init.uniform_(c_w, a=-u, b=u)
        nn.init.uniform_(c_b, a=-u, b=u)

        #w.shape = (in_features, out_features)#out_features = hidden_size*3
        #b.shape = (out_features, )
        w = torch.cat((f_w, o_w, c_w), 1)
        b = torch.cat((f_b, o_b, c_b), 0)

        self.w = nn.parameter.Parameter(w, requires_grad=True)
        self.b = nn.parameter.Parameter(b, requires_grad=True)
    
    def forward(self, x):
        #x.shape = (batch_size, in_features)
        #w.shape = (in_features, out_features)
        #xw.shape = (batch_size, out_features)
        xw = torch.mm(x, self.w)
        xwb = xw + self.b
        return xwb


class XLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, g=0.01):
        super(XLSTMCell, self).__init__()
        alpha = xigmoid_alpha(seq_len, g=g)
        self.hx = nn.Sequential(
            #(batch_size, input_size), (batch_size, hidden_size)
            Cat(1),
            #(batch_size, input_size + hidden_size)
            LinearForXLSTM(input_size + hidden_size, hidden_size, alpha=alpha, seq_len=seq_len),
            #(batch_size, hidden_size*3)
            Chunk(3, 1),
            #(batch_size, hidden_size)*3
        )#f, o, c
        self.xigmoid = Xigmoid(alpha=alpha)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    
    def forward(self, x_t, hc):
        #x_t.shape = (batch_size, input_size)
        #h_t_1/c_t_1.shape = (batch_size, hidden_size)
        tanh = self.tanh
        h_t_1, c_t_1 = hc
        f_hx_t, o_hx_t, c_hx_t = self.hx((x_t, h_t_1))
        f_t = self.xigmoid(f_hx_t)
        o_t = self.sigmoid(o_hx_t)
        c_hat_t = tanh(c_hx_t)

        c_t = f_t*c_t_1 + (1.0 - f_t)*c_hat_t
        h_t = o_t*tanh(c_t)
        return (h_t, c_t)

class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, g=0.01):
        super(XLSTM, self).__init__()
        self.cell = XLSTMCell(input_size, hidden_size, seq_len, g=g)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.g = g
        
    def forward(self, x, hc=None):
        dtype = x.dtype
        device = x.device
        batch_size, seq_len, _ = x.size()
        hidden_size = self.hidden_size
        cell = self.cell
        if hc is None:
            h_t_1 = torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
            c_t_1 = torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
        else:
            h_t_1, c_t_1 = hc
        
        h_list = []
        for t in range(seq_len):
            x_t = x[:,t,:]
            h_t, c_t = cell(x_t, (h_t_1, c_t_1))

            h_list.append(h_t)
            h_t_1, c_t_1 = h_t, c_t
        #h.shape = (batch_size, seq_len, hidden_size)
        h = torch.stack(h_list, 1)
        return h, (h_t_1, c_t_1)
################XLSTM#############################################

################XGRU##############################################
class LinearForXGRU(nn.Module):
    #input.shape = (batch_size, in_features)
    #p.shape = (in_features, hidden_size*2)
    #output.shape = (batch_size, hidden_size*2)
    def __init__(self, in_features, hidden_size, alpha, seq_len):
        super(LinearForXGRU, self).__init__()

        #update gate initialization
        b = math.log(1.0 + math.log(seq_len - 1)/alpha)/alpha

        f_w = torch.empty((in_features, hidden_size))
        b = math.sqrt(1.0/in_features)*b
        nn.init.uniform_(f_w, a=-b, b=b)
        f_b = torch.zeros((hidden_size, ))

        u = math.sqrt(1.0/hidden_size)
        #reset gate initialization
        o_w = torch.empty((in_features, hidden_size))
        o_b = torch.empty((hidden_size, ))
        nn.init.uniform_(o_w, a=-u, b=u)
        nn.init.uniform_(o_b, a=-u, b=u)


        #w.shape = (in_features, out_features)#out_features = hidden_size*2
        #b.shape = (out_features, )
        w = torch.cat((f_w, o_w), 1)
        b = torch.cat((f_b, o_b), 0)

        self.w = nn.parameter.Parameter(w, requires_grad=True)
        self.b = nn.parameter.Parameter(b, requires_grad=True)
    
    def forward(self, x):
        #x.shape = (batch_size, in_features)
        #w.shape = (in_features, out_features)
        #xw.shape = (batch_size, out_features)
        xw = torch.mm(x, self.w)
        xwb = xw + self.b
        return xwb

class XGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, g=0.01):
        super(XGRUCell, self).__init__()
        alpha = xigmoid_alpha(seq_len, g=g)

        self.rz = nn.Sequential(
            #(batch_size, input_size), (batch_size, hidden_size)
            Cat(1),
            #(batch_size, input_size + hidden_size)
            LinearForXGRU(input_size + hidden_size, hidden_size, alpha, seq_len),
            #(batch_size, hidden_size*2)
            Chunk(2, 1),
            #(batch_size, hidden_size)*2
        )
        self.h_hat = nn.Sequential(
            #(batch_size, input_size), (batch_size, hidden_size)
            Cat(1),
            #(batch_size, input_size + hidden_size)
            nn.Linear(input_size + hidden_size, hidden_size),
            #(batch_size, hidden_size)
        )
        
        self.xigmoid = Xigmoid(alpha=alpha)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.reset_parameters(hidden_size)
    
    def reset_parameters(self, hidden_size):
        std = 1.0/math.sqrt(hidden_size)
        for p in self.h_hat.parameters():
            nn.init.uniform_(p.data, a=-std, b=std)


    
    def forward(self, x_t, h_t_1):
        z_hx_t, r_hx_t = self.rz((x_t, h_t_1))
        z_t = self.xigmoid(z_hx_t)
        r_t = self.sigmoid(r_hx_t)
        h_hx_t = self.h_hat((x_t, r_t*h_t_1))
        h_hat_t = self.tanh(h_hx_t)
        h_t = z_t*h_t_1 + (1.0 - z_t)*h_hat_t
        return h_t

class XGRU(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, g=0.01):
        super(XGRU, self).__init__()
        self.cell = XGRUCell(input_size, hidden_size, seq_len, g=g)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
    
    def forward(self, x, h_t_1=None):
        
        dtype = x.dtype
        device = x.device
        batch_size, seq_len, _ = x.size()
        hidden_size = self.hidden_size
        cell = self.cell
        if h_t_1 is None:
            h_t_1 = torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
        
        h_list = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = cell(x_t, h_t_1)
            h_list.append(h_t)
            h_t_1 = h_t
        
        #h.shape = (batch_size, seq_len, hidden_size)
        h = torch.stack(h_list, 1)
        return h, h_t_1

################XGRU##############################################

if __name__ == '__main__':
    batch_size = 1
    seq_len = 1000
    input_size = 20
    hidden_size = 30
    x = torch.randn((batch_size, seq_len, input_size))
    f = XLSTM(input_size, hidden_size, seq_len)
    h, (h_t_1, c_t_1) = f(x)
    print(h.shape, h.mean(), h.std())

    batch_size = 1
    seq_len = 1000
    input_size = 20
    hidden_size = 30
    x = torch.randn((batch_size, seq_len, input_size))
    f = XGRU(input_size, hidden_size, seq_len)
    h, h_t_1 = f(x)
    print(h.shape, h.mean(), h.std())


import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange
import tqdm


def get_matrix(B,C,D):
    assert B.shape[0] == D.shape[1]
    assert C.shape[0] == D.shape[2]
    assert B.shape[2] == C.shape[1] == D.shape[0]
    b1, p, r = B.shape
    b2, r, q = C.shape
    r, b1, b2 = D.shape
    C = C.unsqueeze(0) # 1,b2,r,q
    D = D.permute(1,2,0).unsqueeze(-1) # b1,b2,r,1
    DC = C*D
    DC = DC.permute(0,1,3,2).reshape(b1, b2*q, r) # b1 n r
    A = torch.bmm(B, DC.transpose(1,2))
    A = A.view(b1*p, b2*q)
    return A


def get_batched_matrix(B,C,D):
    assert B.shape[0] == C.shape[0] == D.shape[0]
    assert B.shape[1] == D.shape[2]
    assert C.shape[1] == D.shape[3]
    assert B.shape[3] == C.shape[2] == D.shape[1]
    batch, b1, p, r = B.shape
    batch, b2, r, q = C.shape
    batch, r, b1, b2 = D.shape
    C = C.unsqueeze(1) # batch,1,b2,r,q
    D = D.permute(0,2,3,1).unsqueeze(-1) # batch,b1,b2,r,1
    DC = C*D
    DC = DC.permute(0,1,2,4,3).reshape(batch*b1, b2*q, r) # (batch b1) n r
    B = B.view(batch*b1, p, r)
    A = torch.bmm(B, DC.transpose(1,2))
    A = A.view(batch,b1*p, b2*q)
    return A


#def q_dq(w, quantizer):
#    quantized_weight, max_abs, shape = quantizer.quantize_block(w)
#    return quantizer.dequantize_block(quantized_weight, max_abs, shape)


def _blast_precond_gd_single(A, B, C, D, 
                             T = 300,
                             print_freq = 1,
                             lr = 1.0,
                             device = None,
                             precondition = True,
                             end_factor = 0.01,
                             enforce_nonzero_D = False,
                             delta = 1.0,
                             verbose = False,
                             quantizer=None,
                             weight_decay=0.0,
                             lambd=0.0,
                             fix_B=False,
                             fix_C=False,
                             fix_D=False,
                             ):

    variables = []
    if not fix_B:
        variables.append(B)
    if not fix_C:
        variables.append(C)
    if not fix_D:
        variables.append(D)
    opt = torch.optim.SGD(variables, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=T)


    pbar = tqdm.tqdm(range(T)) if verbose else range(T)
    for t in pbar:
        for ti in range(3):
            if ti==0 and fix_B:
                continue
            else:
                B.requires_grad_(ti==0)
            if ti==1 and fix_C:
                continue
            else:
                C.requires_grad_(ti==1)
            if ti==2 and fix_D:
                continue
            else:
                D.requires_grad_(ti==2)
            opt.zero_grad()
            Ap = get_matrix(B,C,D)
            loss = torch.norm(A-Ap)**2
            with torch.no_grad():
                eps = torch.sqrt(loss) * delta
            loss.backward()

            if precondition:
                with torch.no_grad():
                    if ti%3==0:
                        r, b1, b2 = D.shape
                        D_ = rearrange(D, 'r b1 b2 -> b1 b2 r 1', r=r, b1=b1, b2=b2)
                        C_ = rearrange(C, 'b2 r q -> 1 b2 r q', b2=b2, r=r)
                        Cbar = C_*D_ # b1 b2 r q
                        Cbar = rearrange(Cbar, 'b1 b2 r q -> b1 r (b2 q)')
                        cov = torch.bmm(Cbar, Cbar.transpose(1,2)) + eps * torch.eye(r, dtype=B.dtype, device=B.device)
                        if r <= 2048:
                            B.grad = torch.linalg.solve(cov, B.grad, left=False) # this is faster than torch.linalg.inv(cov).
                        else:
                            B.grad = torch.stack([torch.linalg.solve(cov[bi,...], B.grad[bi,...], left=False) for bi in range(b1)], dim=0)

                    if ti%3==1:
                        r, b1, b2 = D.shape
                        D_ = rearrange(D, 'r b1 b2 -> b2 b1 1 r', r=r, b1=b1, b2=b2)
                        B_ = rearrange(B, 'b1 p r -> 1 b1 p r', b1=b1, r=r)
                        Bbar = B_ * D_
                        Bbar = rearrange(Bbar, 'b2 b1 p r -> b2 (b1 p) r')
                        cov = torch.bmm(Bbar.transpose(1,2), Bbar) + eps * torch.eye(r, dtype=B.dtype, device=B.device)
                        if r <= 2048:
                            C.grad = torch.linalg.solve(cov, C.grad) # this is faster than torch.linalg.inv(cov).
                        else:
                            C.grad = torch.stack([torch.linalg.solve(cov[bi,...], C.grad[bi,...]) for bi in range(b2)], dim=0) 


                    if ti%3==2:
                        r, b1, b2 = D.shape
                        B_cov = torch.bmm(B.transpose(1,2), B)
                        C_cov = torch.bmm(C, C.transpose(1,2))
                        cov = B_cov.unsqueeze(0) * C_cov.unsqueeze(1) + eps * torch.eye(r, dtype=B.dtype, device=B.device)
                        cov = rearrange(cov, 'b1 b2 r1 r2 -> (b1 b2) r1 r2')
                        D_grad = rearrange(D.grad, 'r b1 b2 -> (b1 b2) r 1')
                        if r <= 2048:
                            D_grad = torch.linalg.solve(cov, D_grad) # this is faster than torch.linalg.inv(cov).
                        else:
                            D_grad = torch.stack([torch.linalg.solve(cov[bi,...], D_grad[bi,...]) for bi in range(b1*b2)], dim=0)
                        D.grad = rearrange(D_grad, '(b1 b2) r 1 -> r b1 b2', b1=b1, b2=b2)


            opt.step()
        if enforce_nonzero_D:
            with torch.no_grad():
                D.data = F.relu(D)
        if lambd>0.0:
            with torch.no_grad():
                lr = opt.param_groups[0]['lr']
                #D.data = F.softshrink(D, lambd=lr*lambd)
                D.data = F.relu(1 - lambd*lr/torch.norm(D, dim=(1,2), keepdim=True))*D



        sched.step()

        if t==0 or t%print_freq==print_freq-1:
            with torch.no_grad():
                n_fro = torch.linalg.matrix_norm(A-Ap) / torch.linalg.matrix_norm(A)
                if verbose:
                    desc = "{:5d} - Fro: {:10.3e}".format(t, n_fro.item())
                    if lambd > 0:
                        desc += " - D Sparsity: {:.3f}".format(1.0 - (D.count_nonzero()/D.numel()).item())
                    pbar.set_description(desc)

    return B,C,D


def _batched_blast_precond_gd(A, B, C, D, 
                              T = 300,
                              print_freq = 1,
                              lr = 1.0,
                              device = None,
                              precondition = True,
                              end_factor = 0.01,
                              enforce_nonzero_D = False,
                              delta = 1.0,
                              normalize = False,
                              lambd=0.0,
                              ):
    opt = torch.optim.SGD([B,C,D], lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=T)


    pbar = tqdm.tqdm(range(T))
    for t in pbar:
        for ti in range(3):
            opt.zero_grad()
            B.requires_grad_(ti==0)
            C.requires_grad_(ti==1)
            D.requires_grad_(ti==2)
            Ap = get_batched_matrix(B,C,D)
            loss = torch.norm(A-Ap)**2
            with torch.no_grad():
                eps = torch.sqrt(loss) * delta
            loss.backward()

            if precondition:
                with torch.no_grad():
                    batch, r, b1, b2 = D.shape
                    _, _, p, _ = B.shape
                    _, _, _, q = C.shape
                    if ti%3==0:
                        D_ = rearrange(D, 'batch r b1 b2 -> batch b1 b2 r 1', r=r, b1=b1, b2=b2)
                        C_ = rearrange(C, 'batch b2 r q -> batch 1 b2 r q', b2=b2, r=r)
                        Cbar = C_*D_ # b1 b2 r q
                        Cbar = rearrange(Cbar, 'batch b1 b2 r q -> (batch b1) r (b2 q)')
                        cov = torch.bmm(Cbar, Cbar.transpose(-2,-1))
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        B_grad_ = B.grad.view(batch*b1, p, r)
                        B_grad_ = torch.bmm(B_grad_, P.transpose(-2,-1))
                        B.grad = B_grad_.view(batch, b1, p, r)

                    if ti%3==1:
                        D_ = rearrange(D, 'batch r b1 b2 -> batch b2 b1 1 r', r=r, b1=b1, b2=b2)
                        B_ = rearrange(B, 'batch b1 p r -> batch 1 b1 p r', b1=b1, r=r)
                        Bbar = B_ * D_
                        Bbar = rearrange(Bbar, 'batch b2 b1 p r -> (batch b2) (b1 p) r')
                        cov = torch.bmm(Bbar.transpose(-2,-1), Bbar)
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        C_grad_ = C.grad.view(batch*b2, r, q)
                        C_grad_ = torch.bmm(P.transpose(-2,-1), C_grad_)
                        C.grad = C_grad_.view(batch, b2, r, q)

                    if ti%3==2:
                        B_ = B.view(batch*b1, p, r)
                        B_cov = torch.bmm(B_.transpose(-2,-1), B_)
                        B_cov = B_cov.view(batch, b1, r, r)
                        C_ = C.view(batch*b2, r, q)
                        C_cov = torch.bmm(C_, C_.transpose(-2,-1))
                        C_cov = C_cov.view(batch, b2, r, r)
                        cov = B_cov.unsqueeze(1) * C_cov.unsqueeze(2)
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        P = rearrange(P, 'batch b1 b2 r1 r2 -> (batch b1 b2) r1 r2')
                        D_grad_ = rearrange(D.grad, 'batch r b1 b2 -> (batch b1 b2) r 1')
                        D_grad_ = torch.bmm(P, D_grad_)
                        D.grad = rearrange(D_grad_, '(batch b1 b2) r 1 -> batch r b1 b2', batch=batch, b1=b1, b2=b2)


            opt.step()
            if normalize:
                with torch.no_grad():
                    B.data = B / (torch.norm(B, dim=1, keepdim=True) + 1e-8)
                    C.data = C / (torch.norm(C, dim=2, keepdim=True) + 1e-8)
            if lambd>0.0:
                with torch.no_grad():
                    lr = opt.param_groups[0]['lr']
                    D.data = F.softshrink(D, lambd=lr*lambd)
        if enforce_nonzero_D:
            with torch.no_grad():
                D.data = F.relu(D)

        sched.step()

        if t==0 or t%print_freq==print_freq-1:
            with torch.no_grad():
                n_fro = torch.linalg.matrix_norm(A-Ap) / torch.linalg.matrix_norm(A)
                pbar.set_description("{:5d} - Fro - avg: {:10.3e}, max: {:10.3e}, min: {:10.3e}".format(t, n_fro.mean().item(), n_fro.max().item(), n_fro.min().item()))

    return B,C,D




def blast_precond_gd(targets, # a 2-D or 3-D tensor, or a list of 2-D tensors
                   num_blocks,
                   r,
                   T = 300,
                   print_freq = 1,
                   lr = 1.0,
                   device=None,
                   precondition=True,
                   end_factor=0.01,
                   enforce_nonzero_D=False,
                   delta=1.0,
                   normalize=False,
                   verbose=False,
                   q_dq=None,
                   weight_decay=0.0,
                   lambd=0.0,
                   #num_iter=1,
              ):

    #
    assert r > 0

    A = targets



    if isinstance(A, list):
        A = torch.stack(A, dim=0)

    if device is None:
        device = A.device
    else:
        A = A.to(device)

    with torch.no_grad():
        if len(A.shape) == 2:
            A_std = A.std()
            A = A / A_std
            M, N = A.shape
            assert M%num_blocks == 0 and N%num_blocks == 0
            p = M//num_blocks
            q = N//num_blocks
            B = torch.empty(num_blocks, p, r, device=device)
            C = torch.empty(num_blocks, r, q, device=device)
            D = torch.empty(r, num_blocks, num_blocks, device=device)
        elif len(A.shape) == 3:
            A_std = A.std((1,2), keepdim=True)
            A = A / A_std
            A_std = A_std.view(-1,1,1,1)
            b, M, N = A.shape
            assert M%num_blocks == 0 and N%num_blocks == 0
            p = M//num_blocks
            q = N//num_blocks
            B = torch.empty(b, num_blocks, p, r, device=device)
            C = torch.empty(b, num_blocks, r, q, device=device)
            D = torch.empty(b, r, num_blocks, num_blocks, device=device)
        else:
            raise ValueError("len(A.shape) should be either 2 or 3, given: {}".format(len(A.shape)))


        B.copy_(torch.randn_like(B)/np.sqrt(r)*0.001)
        C.copy_(torch.randn_like(C)/np.sqrt(r)*0.001)
        D.copy_(torch.randn_like(D))

    if len(A.shape) == 2:
        if q_dq is None:
            B,C,D =  _blast_precond_gd_single(A, B, C, D,
                                 T = T,
                                 print_freq = print_freq,
                                 lr = lr,
                                 device = device,
                                 precondition = precondition,
                                 end_factor = end_factor,
                                 enforce_nonzero_D = enforce_nonzero_D,
                                 delta = delta,
                                 verbose=verbose,
                                 weight_decay=weight_decay,
                                 lambd=lambd,
                             )
        else:
            pass
            #B,C,D =  _blast_precond_gd_single(A, B, C, D,
            #                     T = T,
            #                     print_freq = print_freq,
            #                     lr = lr,
            #                     device = device,
            #                     precondition = precondition,
            #                     end_factor = end_factor,
            #                     enforce_nonzero_D = enforce_nonzero_D,
            #                     delta = delta,
            #                     verbose=verbose,
            #                     weight_decay=weight_decay,
            #                     lambd=lambd,
            #                 )

            #with torch.no_grad():
            #    b, p, r = B.shape
            #    B.data = q_dq(B.flatten(1)).view(*B.shape).detach().clone()
            #    C = C.clone().detach().requires_grad_()
            #    D = D.clone().detach().requires_grad_()

            #    #C.data = q_dq(C.transpose(1,2).reshape(-1,r), quantizer).view(b, -1, r).transpose(1,2).contiguous()
            #    #D.data = q_dq(D.flatten(1), quantizer).view(r,b,b)

            #_,C,D =  _blast_precond_gd_single(A, B, C, D,
            #                     T = T,
            #                     print_freq = print_freq,
            #                     lr = lr,
            #                     device = device,
            #                     precondition = precondition,
            #                     end_factor = end_factor,
            #                     enforce_nonzero_D = enforce_nonzero_D,
            #                     delta = delta,
            #                     verbose=verbose,
            #                     weight_decay=weight_decay,
            #                     lambd=lambd,
            #                     fix_B = True,
            #                 )
            #with torch.no_grad():
            #    b, p, r = B.shape
            #    C.data = q_dq(C.flatten(1)).view(*C.shape).detach().clone()
            #    D = D.clone().detach().requires_grad_()

            #_,_,D =  _blast_precond_gd_single(A, B, C, D,
            #                     T = T,
            #                     print_freq = print_freq,
            #                     lr = lr,
            #                     device = device,
            #                     precondition = precondition,
            #                     end_factor = end_factor,
            #                     enforce_nonzero_D = enforce_nonzero_D,
            #                     delta = delta,
            #                     verbose=verbose,
            #                     weight_decay=weight_decay,
            #                     lambd=lambd,
            #                     fix_B = True,
            #                     fix_C = True,
            #                 )
            #with torch.no_grad():
            #    D = q_dq(D.flatten(1)).view(*D.shape)

            #for _ in range(1, num_iter):
            #    B = B.clone().detach().requires_grad_()
            #    C = C.clone().detach().requires_grad_()
            #    D = D.clone().detach().requires_grad_()

            #    B,_,_ =  _blast_precond_gd_single(A, B, C, D,
            #                         T = T,
            #                         print_freq = print_freq,
            #                         lr = lr,
            #                         device = device,
            #                         precondition = precondition,
            #                         end_factor = end_factor,
            #                         enforce_nonzero_D = enforce_nonzero_D,
            #                         delta = delta,
            #                         verbose=verbose,
            #                         weight_decay=weight_decay,
            #                         lambd=lambd,
            #                         fix_B = False,
            #                         fix_C = True,
            #                         fix_D = True,
            #                     )
            #    B.data = q_dq(B.flatten(1)).view(*B.shape).detach().clone()

            #    _,C,_ =  _blast_precond_gd_single(A, B, C, D,
            #                         T = T,
            #                         print_freq = print_freq,
            #                         lr = lr,
            #                         device = device,
            #                         precondition = precondition,
            #                         end_factor = end_factor,
            #                         enforce_nonzero_D = enforce_nonzero_D,
            #                         delta = delta,
            #                         verbose=verbose,
            #                         weight_decay=weight_decay,
            #                         lambd=lambd,
            #                         fix_B = True,
            #                         fix_C = False,
            #                         fix_D = True,
            #                     )

            #    C.data = q_dq(C.flatten(1)).view(*C.shape).detach().clone()

            #    _,_,D =  _blast_precond_gd_single(A, B, C, D,
            #                         T = T,
            #                         print_freq = print_freq,
            #                         lr = lr,
            #                         device = device,
            #                         precondition = precondition,
            #                         end_factor = end_factor,
            #                         enforce_nonzero_D = enforce_nonzero_D,
            #                         delta = delta,
            #                         verbose=verbose,
            #                         weight_decay=weight_decay,
            #                         lambd=lambd,
            #                         fix_B = True,
            #                         fix_C = True,
            #                         fix_D = False,
            #                     )
            #    D.data = q_dq(D.flatten(1)).view(*D.shape).detach().clone()

    elif len(A.shape) == 3:
        B,C,D = _batched_blast_precond_gd(A, B, C, D,
                             T = T,
                             print_freq = print_freq,
                             lr = lr,
                             device = device,
                             precondition = precondition,
                             end_factor = end_factor,
                             enforce_nonzero_D = enforce_nonzero_D,
                             delta = delta,
                             normalize=normalize,
                             lambd=lambd,
                         )

    else:
        raise ValueError("len(A.shape) should be either 2 or 3, given: {}".format(len(A.shape)))
    return B, C, D*A_std
                    
    

def blast_module_precond_gd(m, A, B_init, C_init, D_init, M, N,
                   T = 300,
                   print_freq = 1,
                   step_1by1 = True,
                   lr = 1.0,
                   device=None,
                   initialize=True,
                   precondition=True,
                   end_factor=0.001,
                   enforce_nonzero_D=False,
                   delta=1.0,
              ):
    if initialize:
        with torch.no_grad():
            m.B.copy_(B_init)
            m.C.copy_(C_init)
            m.D.copy_(D_init)

    opt = torch.optim.SGD([m.B,m.C,m.D], lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=end_factor, total_iters=T)

    steps = []
    n_fro_list = []

    pbar = tqdm.tqdm(range(T))
    for t in pbar:
        for ti in range(3):
            opt.zero_grad()
            m.B.requires_grad_(ti==0)
            m.C.requires_grad_(ti==1)
            m.D.requires_grad_(ti==2)
            Ap = m.get_matrix()
            loss = torch.norm(A-Ap)**2
            with torch.no_grad():
                eps = torch.sqrt(loss) * delta
            loss.backward()

            if precondition:
                with torch.no_grad():
                    if m.B.requires_grad and m.B.grad is not None:
                        r, b1, b2 = m.D.shape
                        D_ = rearrange(m.D, 'r b1 b2 -> b1 b2 r 1', r=r, b1=b1, b2=b2)
                        C_ = rearrange(m.C, 'b2 r q -> 1 b2 r q', b2=b2, r=r)
                        Cbar = C_*D_ # b1 b2 r q
                        Cbar = rearrange(Cbar, 'b1 b2 r q -> b1 r (b2 q)')
                        cov = torch.bmm(Cbar, Cbar.transpose(1,2))
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        m.B.grad = torch.bmm(m.B.grad, P.transpose(1,2))
                    if m.D.requires_grad and m.D.grad is not None:
                        r, b1, b2 = m.D.shape
                        B_cov = torch.bmm(m.B.transpose(1,2), m.B)
                        C_cov = torch.bmm(m.C, m.C.transpose(1,2))
                        cov = B_cov.unsqueeze(0) * C_cov.unsqueeze(1)
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        P = rearrange(P, 'b1 b2 r1 r2 -> (b1 b2) r1 r2')
                        D_grad = rearrange(m.D.grad, 'r b1 b2 -> (b1 b2) r 1')
                        D_grad = torch.bmm(P, D_grad)
                        m.D.grad = rearrange(D_grad, '(b1 b2) r 1 -> r b1 b2', b1=b1, b2=b2)
                    if m.C.requires_grad and m.C.grad is not None:
                        r, b1, b2 = m.D.shape
                        D_ = rearrange(m.D, 'r b1 b2 -> b2 b1 1 r', r=r, b1=b1, b2=b2)
                        B_ = rearrange(m.B, 'b1 p r -> 1 b1 p r', b1=b1, r=r)
                        Bbar = B_ * D_
                        Bbar = rearrange(Bbar, 'b2 b1 p r -> b2 (b1 p) r')
                        cov = torch.bmm(Bbar.transpose(1,2), Bbar)
                        P = torch.linalg.inv(cov + eps * torch.eye(r, dtype=cov.dtype, device=cov.device))
                        m.C.grad = torch.bmm(P.transpose(1,2), m.C.grad)

            opt.step()
        if enforce_nonzero_D:
            with torch.no_grad():
                m.D.data = F.relu(m.D)

        sched.step()

        if t==0 or t%print_freq==print_freq-1:
            with torch.no_grad():
                n_fro = torch.linalg.matrix_norm(A-Ap) / torch.linalg.matrix_norm(A)
                pbar.set_description("{:5d} - Fro: {:10.3e}".format(t//3, n_fro.item()))

                steps.append(t//3)
                n_fro_list.append(n_fro.item())

    return steps, n_fro_list

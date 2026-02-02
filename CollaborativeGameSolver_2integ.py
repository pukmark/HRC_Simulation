#!/usr/bin/env python3

import numpy as np
import casadi as ca
from types import SimpleNamespace
import matplotlib.pyplot as plt

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
jl.using("PyCall")
jl.using("PATHSolver")

"""
To use this solver, install the prerequisites using the following steps
1. Install Julia:
- wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz
- tar zxvf julia-1.10.1-linux-x86_64.tar.gz
- export PATH="$PATH:/path/to/<Julia directory>/bin"
2. Install Julia packages:
- In the Julia REPL package manager: 
-- add PyCall
-- add PATHSolver@1.1.1 (side note, only version 1.1.1 works when called from pyjulia)
3. Install pyjulia:
- python3 -m pip install julia
"""


class CollaborativeGame():
    def __init__(self, N: int = 10, dt: float = 0.1, d: float = 4.0, delta_d: float = 0.01, Obstcles: list = None, verbose: int = 0):
        self.N = N
        self.dt = dt
        self.verbose = verbose
        self.Obstcles = Obstcles
        self.n_dim = 2
        self.d = d
        self.delta_d = delta_d

        self.d_min, self.d_max = d-delta_d, d+delta_d
        self.v1_max = 2.0
        self.a1_max = 10.0
        self.v2_max = 5.0
        self.a2_max = 20.0
        self.xlim = [-10.0, 10.0]
        self.ylim = [-10.0, 10.0]

        # Define the states and controls
        x1, x2 = ca.SX.sym('x1', N+1, 2), ca.SX.sym('x2', N+1, 2)
        v1, v2 = ca.SX.sym('v1', N+1, 2), ca.SX.sym('v2', N+1, 2)
        a1, a2 = ca.SX.sym('a1', N, 2), ca.SX.sym('a2', N, 2)
        x1_0, x2_0 = ca.SX.sym('x1_0', 1, 2), ca.SX.sym('x2_0', 1, 2)
        v1_0, v2_0 = ca.SX.sym('v1_0', 1, 2), ca.SX.sym('v2_0', 1, 2)
        x1_f, x2_f = ca.SX.sym('x1_f', 1, 2), ca.SX.sym('x2_f', 1, 2)
        alpha = ca.SX.sym('alpha')
        beta = ca.SX.sym('beta')
        SlackL = ca.SX.sym('Slack', N)

        J1 = 0.5*(ca.sumsqr(x1[:,0]-x1_f[0]) + ca.sumsqr(x1[:,1]-x1_f[1])) + 0.1*ca.sumsqr(a1)
        J2 = 0.5*(ca.sumsqr(x2[:,0]-x2_f[0]) + ca.sumsqr(x2[:,1]-x2_f[1])) + 0.1*ca.sumsqr(a2) + 1e4*ca.sumsqr(SlackL)
        
        # Obstacle avoidance term
        w_obs = 0.0      # tune this (1–50 typical)
        eps   = 1e-3     # numerical safety
        for k in range(N+1):
            if Obstcles is not None:
                for Obstcle in Obstcles:
                    # d2 = ca.sumsqr(x2[k,:] - Obstcle['Pos'])
                    r2 = (Obstcle['diam']/2)**2
                    factors = np.linspace(0.0, 1.0, 1+int(d/(Obstcle['diam']/4)))
                    for factor in factors:
                        d2 = ca.sumsqr(factor*x1[k,:]+(1-factor)*x2[k,:]-Obstcle['Pos'])
                        J2 += -w_obs * ca.log(d2 - r2 + eps)

        h_vec = []
        pc_vec = []
        sc_vec = []
        # shared constraint: s(x) >= 0
        #### Define Lagrangian of players
        # Cost function
        L1 = J1
        L2 = J2

        # Equality constraints (Dynamics) - player 1:
        h1 = []
        h1.append(x1[0,:] - x1_0)
        h1.append(v1[0,:] - v1_0)
        for k in range(N):
            h1.append(v1[k+1,:]-v1[k,:]-dt*a1[k,:] )
            h1.append(x1[k+1,:]-x1[k,:]-dt*v1[k,:]-0.5*dt**2*a1[k,:])
        h1 = ca.horzcat(*h1).T
        mu1 = ca.SX.sym('mu1', h1.shape[0])
        L1 += ca.dot(mu1, h1)
        h_vec.append(h1)

        # Private constraints (Dynamics) - player 1:
        pc1 = []
        for k in range(N):
            pc1.append(self.v1_max**2 - ca.sumsqr(v1[k+1,:]))
            # pc1.append(self.a1_max**2 - ca.sumsqr(a1[k,:]))
            # pc1.append(x1[k+1,0]-self.xlim[0])
            # pc1.append(self.xlim[1]-x1[k+1,0])
            # pc1.append(x1[k+1,1]-self.ylim[0])
            # pc1.append(self.ylim[1]-x1[k+1,1])
        pc1 = ca.horzcat(*pc1).T
        lam1 = ca.SX.sym('lam1', pc1.shape[0])
        L1 -= ca.dot(lam1, pc1)
        pc_vec.append(pc1)

        # Equality constraints (Dynamics) - player 2:
        h2 = []
        h2.append(x2[0,:] - x2_0)
        h2.append(v2[0,:] - v2_0)
        for k in range(N):
            h2.append(v2[k+1,:]-v2[k,:]-dt*a2[k,:] )
            h2.append(x2[k+1,:]-x2[k,:]-dt*v2[k,:]-0.5*dt**2*a2[k,:] )
        h2 = ca.horzcat(*h2).T
        mu2 = ca.SX.sym('mu2', h2.shape[0])
        L2 += ca.dot(mu2, h2)
        h_vec.append(h2)

        # Private constraints (Dynamics) - player 2:
        pc2 = []
        for k in range(N):
            pc2.append(self.v2_max**2 - ca.sumsqr(v2[k+1,:]))
            pc2.append(self.a2_max**2 - ca.sumsqr(a2[k,:]))
            # pc2.append(x2[k+1,0]-self.xlim[0])
            # pc2.append(self.xlim[1]-x2[k+1,0])
            # pc2.append(x2[k+1,1]-self.ylim[0])
            # pc2.append(self.ylim[1]-x2[k+1,1])
            # pc2.append(SlackL[k])  # Slack variable >=0
        pc2 = ca.horzcat(*pc2).T
        lam2 = ca.SX.sym('lam2', pc2.shape[0])
        L2 -= ca.dot(lam2, pc2)
        pc_vec.append(pc2)

        # Shared Constranints
        scL_vec = []
        scOA_vec = []
        for k in range(N):
            scL_vec.append(self.d_max**2 + SlackL[k]**2 - ca.sumsqr(x1[k+1,:]-x2[k+1,:])) # d_max**2 - (x1-x2)**2 >=0
            scL_vec.append(ca.sumsqr(x1[k+1,:]-x2[k+1,:]) - self.d_min**2 - SlackL[k]**2) # (x1-x2)**2 - d_min**2 >=0
            if Obstcles is not None:
                for iObs, Obstcle in enumerate(Obstcles):
                    factors = np.linspace(0.0, 1.0, 1+int(d/(Obstcle['diam']/5)))
                    for factor in factors:
                        scOA_vec.append(ca.sumsqr(factor*x1[k+1,:]+(1-factor)*x2[k+1,:] - Obstcle['Pos']) - (Obstcle['diam']/2)**2) # (fac*x1+(1-fac)*x2-Obs)**2 - r_Obs**2 >=0
        scL_vec = ca.horzcat(*scL_vec).T
        scOA_vec = ca.horzcat(*scOA_vec).T
        sigL = ca.SX.sym('sc1', scL_vec.shape[0])
        sigOA = ca.SX.sym('sc2', scOA_vec.shape[0])

        L1 -= alpha*ca.dot(sigL, scL_vec)
        L2 -= (1-alpha)*ca.dot(sigL, scL_vec)
        L1 -= beta*ca.dot(sigOA, scOA_vec)
        L2 -= (1-beta)*ca.dot(sigOA, scOA_vec)
        
        sc_vec = ca.vertcat(scL_vec, scOA_vec)
        sig = ca.vertcat(sigL, sigOA)

        self.Z = ca.vertcat(x1[:], v1[:], a1[:], x2[:], v2[:], a2[:], SlackL[:], mu1[:], mu2[:], lam1[:], lam2[:], sig[:])
        self.Z_len = [[2*N+2, 2*N+2, 2*N], [2*N+2, 2*N+2, 2*N, N], mu1.shape[0]+mu2.shape[0], pc1.shape[0]+pc2.shape[0], sig.shape[0]]

        self.indx_x1 = 0
        self.indx_y1 = self.indx_x1 + N+1
        self.indx_vx1 = self.indx_y1 + N+1
        self.indx_vy1 = self.indx_vx1 + N+1
        self.indx_ax1 = self.indx_vy1 + N+1
        self.indx_ay1 = self.indx_ax1 + N
        self.indx_x2 = self.indx_ay1 + N
        self.indx_y2 = self.indx_x2 + N+1
        self.indx_vx2 = self.indx_y2 + N+1
        self.indx_vy2 = self.indx_vx2 + N+1
        self.indx_ax2 = self.indx_vy2 + N+1
        self.indx_ay2 = self.indx_ax2 + N+1
        self.indx_lam1 = self.indx_ay2+ pc1.shape[0]
        self.indx_lam2 = self.indx_lam1 + pc2.shape[0]
        self.indx_sigL = self.indx_lam2 + sigL.shape[0]
        self.indx_sigOA = self.indx_sigL + sigOA.shape[0]
        self.lam1_len = pc1.shape[0]
        self.lam2_len = pc2.shape[0]
        self.sigL_len = sigL.shape[0]
        self.sigOA_len = sigOA.shape[0]
        

        _Dxu_L = []
        _Ch = []
        _Cgp = []
        _Cgs = []

        _Dxu_L.append( ca.jacobian(L1, ca.vertcat(x1[:], v1[:], a1[:])).T )
        _Dxu_L.append( ca.jacobian(L2, ca.vertcat(x2[:], v2[:], a2[:], SlackL[:])).T )
        # _Dxu_L.append( ca.jacobian(L2, ca.vertcat(x2[:], v2[:], a2[:], Slack[:])).T )
        _Ch = h_vec
        _Cgp = pc_vec
        _Cgs = sc_vec

        # Define the F and J functions
        F = ca.vertcat(*_Dxu_L, *_Ch, *_Cgp, _Cgs)
        self.fun_F = ca.Function('F', [self.Z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta], [F])
        J = ca.jacobian(F, self.Z)
        self.fun_J = ca.Function('J', [self.Z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta], [J])

        self.n_l_inf = 0
        self.n_l_inf += np.sum(self.Z_len[0]) + np.sum(self.Z_len[1])
        self.n_l_inf += self.Z_len[2]
        self.n_u_inf = self.n_l_inf + np.sum(self.Z_len[3:])

        self.z0 = np.zeros((self.Z.shape[0],))

        self.p_tol = 1e-4
        
        self.nms = 1

        self.success = False
        
        self.sumJ_fun = ca.Function('sumJ', [self.Z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f], [J1 + J2])
        self.fun_x1 = ca.Function('x1', [self.Z], [x1])
        self.fun_x2 = ca.Function('x2', [self.Z], [x2])
        self.fun_v1 = ca.Function('v1', [self.Z], [v1])
        self.fun_v2 = ca.Function('v2', [self.Z], [v2])
        self.fun_a1 = ca.Function('a1', [self.Z], [a1])
        self.fun_a2 = ca.Function('a2', [self.Z], [a2])
        self.fun_slackL = ca.Function('SlackL', [self.Z], [SlackL])
        self.fun_mu1 = ca.Function('mu1', [self.Z], [mu1])
        self.fun_mu2 = ca.Function('mu2', [self.Z], [mu2])
        self.fun_pc1 = ca.Function('pc1', [self.Z], [pc1])
        self.fun_pc2 = ca.Function('pc2', [self.Z], [pc2])
        self.fun_scL = ca.Function('scL', [self.Z], [scL_vec])
        self.fun_scOA = ca.Function('scOA', [self.Z], [scOA_vec])
        self.fun_lam1 = ca.Function('lam1', [self.Z], [lam1])
        self.fun_lam2 = ca.Function('lam2', [self.Z], [lam2])
        self.fun_sigL = ca.Function('sigL', [self.Z], [sigL])
        self.fun_sigOA = ca.Function('sigOA', [self.Z], [sigOA])

        self.sol = SimpleNamespace()
        self.sol.time = -1.0
        self.sol.x1 = self.fun_x1(self.z0).toarray()
        self.sol.x2 = self.fun_x2(self.z0).toarray()
        self.sol.v1 = self.fun_v1(self.z0).toarray()
        self.sol.v2 = self.fun_v2(self.z0).toarray()
        self.sol.a1 = self.fun_a1(self.z0).toarray()
        self.sol.a2 = self.fun_a2(self.z0).toarray()
        self.sol.Slack = self.fun_slackL(self.z0).toarray().reshape(self.N,-1)
        self.sol.mu1 = self.fun_mu1(self.z0).toarray().reshape(self.N+1,-1)
        self.sol.mu2 = self.fun_mu2(self.z0).toarray().reshape(self.N+1,-1)
        self.sol.lam1 = self.fun_lam1(self.z0).toarray().reshape(self.N,-1)
        self.sol.lam2 = self.fun_lam2(self.z0).toarray().reshape(self.N,-1)
        self.sol.sigL = self.fun_sigL(self.z0).toarray().reshape(self.N,-1)
        self.sol.sigOA = self.fun_sigOA(self.z0).toarray().reshape(self.N,-1)
        self.sol.z = self.z0

        self.MPC_guess_human_init()
        self.MPC_guess_robot_init()
        self.Centralized_MPC_init()


    def Solve(self, time, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta, z0 = None, log: bool = True):

        if z0 is not None:
            self.z0 = z0
        
        Main.z0 = self.z0

        Main.ub = np.inf*np.ones(self.n_u_inf)
        Main.lb = np.concatenate((-np.inf*np.ones(self.n_l_inf), np.zeros(self.n_u_inf-self.n_l_inf)))

        # Main.nnz = self.J.sparsity_out(0).nnz()
        Main.nnz = self.fun_J.numel_out(0)

        Main.F_py = lambda z: np.array(self.fun_F(z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta)).squeeze()
        Main.J_py = lambda z: np.array(self.fun_J(z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta))

        Main.tol = self.p_tol

        F_def = """
        function F(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
            @assert n == length(x)
            f .= F_py(x)
            return Cint(0)
        end
        return(F)
        """
        Main.F = jl.eval(F_def)

        J_def = """
        function J(
            n::Cint,
            nnz::Cint,
            x::Vector{Cdouble},
            col::Vector{Cint},
            len::Vector{Cint},
            row::Vector{Cint},
            data::Vector{Cdouble},
        )
            @assert n == length(x)  == length(col) == length(len)
            @assert nnz == length(row) == length(data)
            j = Array{Float64}(undef, n, n)
            j .= J_py(x)
            i = 1
            for c in 1:n
                col[c], len[c] = i, 0
                for r in 1:n
                    # if !iszero(j[r, c])
                    #     row[i], data[i] = r, j[r, c]
                    #     len[c] += 1
                    #     i += 1
                    # end
                    row[i], data[i] = r, j[r, c]
                    len[c] += 1
                    i += 1
                end
            end
            return Cint(0)
        end
        return(J)
        """
        Main.J = jl.eval(J_def)
        
        if self.verbose:
            output = 'yes'
        else:
            output = 'no'
            
        if self.nms:
            nms = 'yes'
        else:
            nms = 'no'

        solve = f"""
        PATHSolver.c_api_License_SetString("1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
        status, z, info = PATHSolver.solve_mcp(F, 
                                               J,
                                               lb,
                                               ub,
                                               z0,
                                               nnz=nnz,
                                               output="{output}",
                                               convergence_tolerance=tol,
                                               nms="{nms}",
                                               crash_nbchange_limit=3,
                                               major_iteration_limit=10000,
                                               minor_iteration_limit=10000,
                                               cumulative_iteration_limit=50000,
                                               restart_limit=100)
        success = status == PATHSolver.MCP_Solved

        return z, success, info.residual, status
        """
        z, self.success, res, status = jl.eval(solve)
        
        self.status_msg = status.__name__
        if self.success:
            self.z0 = z


        _f = np.array(self.fun_F(z, x1_0, v1_0, x1_f, x2_0, v2_0, x2_f, alpha, beta)).squeeze()
        n_xu = np.sum(self.Z_len[0]) + np.sum(self.Z_len[1])
        _g = -_f[n_xu:]
        _l = z[n_xu:]
        stat = np.linalg.norm(_f[:n_xu], ord=np.inf)
        feas = max(0, np.amax(_g))
        comp = np.linalg.norm(_g * _l, ord=np.inf)

        if self.success:
            self.sol.time = time
            self.sol.alpha = alpha
            self.sol.beta = beta
            self.sol.x1 = self.fun_x1(z).toarray()
            self.sol.x2 = self.fun_x2(z).toarray()
            self.sol.v1 = self.fun_v1(z).toarray()
            self.sol.v2 = self.fun_v2(z).toarray()
            self.sol.a1 = self.fun_a1(z).toarray()
            self.sol.a2 = self.fun_a2(z).toarray()
            self.sol.Slack = self.fun_slackL(z).toarray().reshape(self.N,-1)
            self.sol.mu1 = self.fun_mu1(z).toarray().reshape(self.N+1,-1)
            self.sol.mu2 = self.fun_mu2(z).toarray().reshape(self.N+1,-1)
            self.sol.lam1 = self.fun_lam1(z).toarray().reshape(self.N,-1)
            self.sol.lam2 = self.fun_lam2(z).toarray().reshape(self.N,-1)
            self.sol.sigL = self.fun_sigL(z).toarray().reshape(self.N,-1)
            self.sol.sigOA = self.fun_sigOA(z).toarray().reshape(self.N,-1)
            self.sol.z = z


        if log:
            try:
                print(f'{self.status_msg} - p feas: {feas:.4e} | comp: {comp:.4e} | stat: {stat:.4e} | Slack: {np.max(np.abs(self.sol.Slack)):.4e}')
            except BlockingIOError:
                # Avoid crashing if stdout pipe is saturated (e.g., headless workers).
                pass

        return self.sol.x1, self.sol.x2, self.success


    def MPC_guess_human_init(self):
        opti = ca.Opti()

        x = opti.variable(self.N+1, self.n_dim)
        v = opti.variable(self.N+1, self.n_dim)
        a = opti.variable(self.N, self.n_dim)
        x_0, v_0 = opti.parameter(1, self.n_dim), opti.parameter(1, self.n_dim)
        x_tgt = opti.parameter(1, self.n_dim)

        opti.minimize(0.01*ca.sumsqr(a) + ca.sumsqr(x[self.N,:]-x_tgt))
        opti.subject_to(x[0,:] == x_0)
        opti.subject_to(v[0,:] == v_0)
        for k in range(self.N):
            # Dynamics
            opti.subject_to(v[k+1,:] == v[k,:] + self.dt*a[k,:])
            opti.subject_to(x[k+1,:] == x[k,:] + self.dt*v[k,:]  + 0.5*a[k,:]*self.dt**2)
            # V max
            opti.subject_to(self.v1_max**2 >= ca.sumsqr(v[k+1,:]))
            # A max
            opti.subject_to(self.a1_max**2 >= ca.sumsqr(a[k,:]))
            # Obstcles constraint:
            for iObs, Obstcle in enumerate(self.Obstcles):
                opti.subject_to(ca.sumsqr(x[k+1,:]-Obstcle['Pos']) >= (Obstcle['diam']/2)**2)

        opts = {"print_time": 0,  # Print timing, 
                "ipopt": {
                "mu_strategy": "adaptive",  # "adaptive" or "adaptive" Strategy for updating the barrier parameter
                "tol": 1e-6,  # Convergence tolerance
                # "max_iter": 250,  # Max iterations
                "print_level": 0,  # Verbosity level
                'print_frequency_iter': 5,  # print_frequency_iter
                "timing_statistics": "no", # Enable timing statistics
                # "nlp_scaling_method": "none", # 'none' 'gradient-based', # Scaling method
            }}
        opti.solver('ipopt', opts)

        self.human_mpc = SimpleNamespace()
        self.human_mpc.opti = opti
        self.human_mpc.x = x
        self.human_mpc.v = v
        self.human_mpc.a = a
        self.human_mpc.x_0 = x_0
        self.human_mpc.v_0 = v_0
        self.human_mpc.x_tgt = x_tgt

        return

    def MPC_guess_robot_init(self):
        opti = ca.Opti()

        x = opti.variable(self.N+1, self.n_dim)
        v = opti.variable(self.N+1, self.n_dim)
        a = opti.variable(self.N, self.n_dim)
        Slack = opti.variable(self.N)
        x_0, v_0 = opti.parameter(1, self.n_dim), opti.parameter(1, self.n_dim)
        x_tgt = opti.parameter(1, self.n_dim)
        x_partner = opti.parameter(self.N+1, self.n_dim)


        opti.minimize(0.001*ca.sumsqr(a) + 0.01*ca.sumsqr(v) + ca.sumsqr(x[self.N,:]- x_tgt) + 1e6**ca.sumsqr(Slack))
        opti.subject_to(x[0,:] == x_0)
        opti.subject_to(v[0,:] == v_0)
        for k in range(self.N):
            # Dynamics
            opti.subject_to(v[k+1,:] == v[k,:] + self.dt*a[k,:])
            opti.subject_to(x[k+1,:] == x[k,:] + self.dt*v[k,:]  + 0.5*a[k,:]*self.dt**2)
            # V max
            opti.subject_to(self.v2_max**2 >= ca.sumsqr(v[k+1,:]))
            # A max
            opti.subject_to(self.a2_max**2 >= ca.sumsqr(a[k,:]))
            # distance constraint:
            opti.subject_to(self.d_max**2 + Slack[k]**2 >= ca.sumsqr(x[k+1,:]-x_partner[k+1,:]))
            opti.subject_to(self.d_min**2 - Slack[k]**2 <= ca.sumsqr(x[k+1,:]-x_partner[k+1,:]))
            # Obstcles constraint:
            d = (self.d_max + self.d_min)/2
            for iObs, Obstcle in enumerate(self.Obstcles):
                factors = np.linspace(0.0, 1.0, 1+int(d/(Obstcle['diam']/4)))
                for factor in factors[:-1]:
                    opti.subject_to(ca.sumsqr((1-factor)*x[k+1,:]+factor*x_partner[k+1,:]-Obstcle['Pos']) >= (Obstcle['diam']/2)**2-Slack[k]**2) # (fac*x1+(1-fac)*x2-Obs)**2 - r_Obs**2 >=0

        opts = {"print_time": 0,  # Print timing, 
                "ipopt": {
                "mu_strategy": "adaptive",  # "adaptive" or "adaptive" Strategy for updating the barrier parameter
                "tol": 1e-6,  # Convergence tolerance
                # "max_iter": 250,  # Max iterations
                "print_level": 0,  # Verbosity level
                'print_frequency_iter': 5,  # print_frequency_iter
                "timing_statistics": "no", # Enable timing statistics
                # "nlp_scaling_method": "none", # 'none' 'gradient-based', # Scaling method
            }}
        opti.solver('ipopt', opts)

        self.robot_mpc = SimpleNamespace()
        self.robot_mpc.opti = opti
        self.robot_mpc.x = x
        self.robot_mpc.v = v
        self.robot_mpc.a = a
        self.robot_mpc.Slack = Slack
        self.robot_mpc.x_0 = x_0
        self.robot_mpc.v_0 = v_0
        self.robot_mpc.x_partner = x_partner
        self.robot_mpc.x_tgt = x_tgt

        return

    def MPC_guess_human_calc(self, x_0, v_0, x_tgt):
        self.human_mpc.opti.set_value(self.human_mpc.x_0, x_0)
        self.human_mpc.opti.set_value(self.human_mpc.v_0, v_0)
        self.human_mpc.opti.set_value(self.human_mpc.x_tgt, x_tgt)

        x_guess, v_guess, a_guess = x_0, v_0, np.array([[0.0001, 0.0001]])
        for k in range(self.N+1):
            self.human_mpc.opti.set_initial(self.human_mpc.x[k,:], x_guess)
            self.human_mpc.opti.set_initial(self.human_mpc.v[k,:], v_guess)
            if k<self.N:
                self.human_mpc.opti.set_initial(self.human_mpc.a[k,:], a_guess)
            
            x_guess = x_guess + self.dt*v_guess + 0.5*a_guess*self.dt**2
            v_guess = v_guess + self.dt*a_guess

        try:
            plot_sol = False
            sol = self.human_mpc.opti.solve()
        except:
            plot_sol = False

        x_sol = np.array(self.human_mpc.opti.debug.value(self.human_mpc.x))
        v_sol = np.array(self.human_mpc.opti.debug.value(self.human_mpc.v))
        a_sol = np.array(self.human_mpc.opti.debug.value(self.human_mpc.a))

        if plot_sol:
            plt.figure()
            plt.plot(x_sol[:,0], x_sol[:,1], 'g')
            if self.Obstcles is not None:
                for iObs, Obstcle in enumerate(self.Obstcles):
                    x, y = Obstcle['diam']/2*np.cos(np.linspace(0,2*np.pi,100)), Obstcle['diam']/2*np.sin(np.linspace(0,2*np.pi,100))
                    plt.plot(Obstcle['Pos'][0,0]+x, Obstcle['Pos'][0,1]+y, 'k', linewidth=3)
            plt.xlim([-5,10])
            plt.ylim([-5,10])
            plt.grid()
            plt.show()
            plt.pause(0.1)
            plt.pause(0.1)

        return x_sol, v_sol, a_sol


    def MPC_guess_robot_calc(self, x_0, v_0, x_tgt, x_partner = None, reverse_init = False):

        self.robot_mpc.opti.set_value(self.robot_mpc.x_0, x_0)
        self.robot_mpc.opti.set_value(self.robot_mpc.v_0, v_0)
        self.robot_mpc.opti.set_value(self.robot_mpc.x_partner, x_partner)
        self.robot_mpc.opti.set_value(self.robot_mpc.x_tgt, x_tgt)


        x_guess, v_guess = x_0, v_0
        for k in range(self.N+1):
            self.robot_mpc.opti.set_initial(self.robot_mpc.x[k,:], x_guess)
            self.robot_mpc.opti.set_initial(self.robot_mpc.v[k,:], v_guess)
            
            if k<self.N: 
                if reverse_init == False:
                    a_guess = -0.5*v_guess
                    self.robot_mpc.opti.set_initial(self.robot_mpc.a[k,:], a_guess)
                    x_guess = x_guess + self.dt*v_guess + 0.5*a_guess*self.dt**2
                    v_guess = v_guess + self.dt*a_guess
                else:
                    x_prev = x_guess
                    v_prev = v_guess
                    if k==0:
                        theta = np.atan2(x_0[0,1]-x_partner[0,1], x_0[0,0]-x_partner[0,0])
                        theta_vec = np.linspace(theta, theta+np.pi, self.N+1)
                    x_guess = x_partner[k] + 0.5*(self.d_min+self.d_max)*np.array([np.cos(theta_vec[k]), np.sin(theta_vec[k])]).T
                    v_guess = (x_guess - x_prev)/self.dt
                    a_guess = (v_guess - v_prev)/self.dt

        try:
            plot_sol = False
            sol = self.robot_mpc.opti.solve()
        except:
            plot_sol = False

        x_sol = np.array(self.robot_mpc.opti.debug.value(self.robot_mpc.x))
        v_sol = np.array(self.robot_mpc.opti.debug.value(self.robot_mpc.v))
        a_sol = np.array(self.robot_mpc.opti.debug.value(self.robot_mpc.a))
        Slack_sol = np.array(self.robot_mpc.opti.debug.value(self.robot_mpc.Slack))

        if plot_sol:
            plt.figure()
            plt.plot(x_sol[:,0], x_sol[:,1], 'b', linewidth=3, label="robot")
            plt.plot(x_partner[:,0], x_partner[:,1], 'g', linewidth=3, label="human")
            for k in range(self.N):
                d_cur = np.linalg.norm(x_sol[k+1,:] - x_partner[k+1,:])
                if d_cur-0.01>self.d_max or d_cur+0.01 < self.d_min:
                    plt.plot([x_sol[k+1,0], x_partner[k+1,0]], [x_sol[k+1,1], x_partner[k+1,1]], 'k:', linewidth=2)

                if self.Obstcles is not None:
                    d = (self.d_max + self.d_min)/2
                    for iObs, Obstcle in enumerate(self.Obstcles):
                        x, y = Obstcle['diam']/2*np.cos(np.linspace(0,2*np.pi,100)), Obstcle['diam']/2*np.sin(np.linspace(0,2*np.pi,100))
                        plt.plot(Obstcle['Pos'][0,0]+x, Obstcle['Pos'][0,1]+y, 'k', linewidth=3)
                        factors = np.linspace(0.0, 1.0, 1+int(d/(Obstcle['diam']/4)))
                        for factor in factors[1:-1]:
                            x_cur = factor*x_sol[k+1,:]+(1-factor)*x_partner[k+1,:]
                            d_cur = np.linalg.norm(x_cur - Obstcle['Pos'])
                            if d_cur+0.0001 < Obstcle['diam']/2:
                                plt.plot(x_cur[0],x_cur[1],'r*')
                        

            plt.xlim([-5,10])
            plt.ylim([-5,10])
            plt.grid()
            plt.gca().set_aspect('equal')
            plt.show()
            plt.pause(0.1)
            plt.pause(0.1)

        return x_sol, v_sol, a_sol
    
    def Centralized_MPC_init(self):
        
        opti = ca.Opti()
        
        x = opti.variable(2*(self.N+1), self.n_dim)
        v = opti.variable(2*(self.N+1), self.n_dim)
        a = opti.variable(2*self.N, self.n_dim)
        Slack = opti.variable(self.N)
        x1_0, v1_0 = opti.parameter(1, self.n_dim), opti.parameter(1, self.n_dim)
        x2_0, v2_0 = opti.parameter(1, self.n_dim), opti.parameter(1, self.n_dim)
        x1_tgt = opti.parameter(1, self.n_dim)
        x2_tgt = opti.parameter(1, self.n_dim)
        alpha = opti.parameter(1)
        
        # Slice views for readability
        x1, x2 = x[0:self.N+1, :], x[self.N+1:, :]
        v1, v2 = v[0:self.N+1, :], v[self.N+1:, :]
        a1, a2 = a[0:self.N, :], a[self.N:, :]

        # Objective: terminal accuracy + smooth controls + small velocity penalty + Slack penalty
        opti.minimize((1-alpha)*0.001*ca.sumsqr(a1) + (1-alpha)*ca.sumsqr(x1[self.N, :] - x1_tgt) + 
                      alpha*0.001*ca.sumsqr(a2) + alpha*ca.sumsqr(x2[self.N, :] - x2_tgt) +
                      1e6*ca.sumsqr(Slack) )

        # Initial conditions
        opti.subject_to(x1[0, :] == x1_0)
        opti.subject_to(v1[0, :] == v1_0)
        opti.subject_to(x2[0, :] == x2_0)
        opti.subject_to(v2[0, :] == v2_0)
        opti.subject_to(Slack >= 0)

        for k in range(self.N):
            # Dynamics
            opti.subject_to(v1[k+1, :] == v1[k, :] + self.dt*a1[k, :])
            opti.subject_to(x1[k+1, :] == x1[k, :] + self.dt*v1[k, :] + 0.5*self.dt**2*a1[k, :])
            opti.subject_to(v2[k+1, :] == v2[k, :] + self.dt*a2[k, :])
            opti.subject_to(x2[k+1, :] == x2[k, :] + self.dt*v2[k, :] + 0.5*self.dt**2*a2[k, :])
            # Speed/acc bounds
            opti.subject_to(self.v1_max**2 >= ca.sumsqr(v1[k+1, :]))
            opti.subject_to(self.a1_max**2 >= ca.sumsqr(a1[k, :]))
            opti.subject_to(self.v2_max**2 >= ca.sumsqr(v2[k+1, :]))
            opti.subject_to(self.a2_max**2 >= ca.sumsqr(a2[k, :]))
            # Distance constraints between agents
            opti.subject_to(self.d_max**2 + Slack[k] >= ca.sumsqr(x1[k+1, :] - x2[k+1, :]))
            opti.subject_to(self.d_min**2 - Slack[k] <= ca.sumsqr(x1[k+1, :] - x2[k+1, :]))
            # Obstacle avoidance for each agent and along the connecting segment
            for iObs, Obstcle in enumerate(self.Obstcles):
                opti.subject_to(ca.sumsqr(x1[k+1, :] - Obstcle['Pos']) >= (Obstcle['diam']/2)**2)
                opti.subject_to(ca.sumsqr(x2[k+1, :] - Obstcle['Pos']) >= (Obstcle['diam']/2)**2)
                d = (self.d_max + self.d_min)/2
                factors = np.linspace(0.0, 1.0, 1+int(d/(Obstcle['diam']/4)))
                for factor in factors[1:-1]:
                    opti.subject_to( ca.sumsqr((1-factor)*x1[k+1, :] + factor*x2[k+1, :] - Obstcle['Pos']) >= (Obstcle['diam']/2)**2 )

        opts = {"print_time": 0,
                "ipopt": {"mu_strategy": "adaptive",
                          "tol": 1e-6,
                          "print_level": 0,
                          'print_frequency_iter': 5,
                          "timing_statistics": "no",
                          }}
        opti.solver('ipopt', opts)

        self.centralized_mpc = SimpleNamespace()
        self.centralized_mpc.opti = opti
        self.centralized_mpc.x = x
        self.centralized_mpc.v = v
        self.centralized_mpc.a = a
        self.centralized_mpc.Slack = Slack
        self.centralized_mpc.x1_0 = x1_0
        self.centralized_mpc.v1_0 = v1_0
        self.centralized_mpc.x2_0 = x2_0
        self.centralized_mpc.v2_0 = v2_0
        self.centralized_mpc.x1_tgt = x1_tgt
        self.centralized_mpc.x2_tgt = x2_tgt
        self.centralized_mpc.alpha = alpha

        return

    def Centralized_MPC_calc(self, x1_0, v1_0, x1_tgt, x2_0, v2_0, x2_tgt, alpha):

        self.centralized_mpc.opti.set_value(self.centralized_mpc.x1_0, x1_0)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.v1_0, v1_0)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.x2_0, x2_0)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.v2_0, v2_0)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.x1_tgt, x1_tgt)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.x2_tgt, x2_tgt)
        self.centralized_mpc.opti.set_value(self.centralized_mpc.alpha, alpha)

        # Linear interpolation guess between start and goal for both agents
        T_horizon = self.N * self.dt
        v1_guess = (x1_tgt - x1_0) / max(T_horizon, 1e-6)
        v2_guess = (x2_tgt - x2_0) / max(T_horizon, 1e-6)
        v1_norm = np.linalg.norm(v1_guess)
        v2_norm = np.linalg.norm(v2_guess)
        if v1_norm > self.v1_max:
            v1_guess = v1_guess / v1_norm * self.v1_max
        if v2_norm > self.v2_max:
            v2_guess = v2_guess / v2_norm * self.v2_max

        x1_guess = np.array(x1_0, copy=True)
        x2_guess = np.array(x2_0, copy=True)
        for k in range(self.N+1):
            self.centralized_mpc.opti.set_initial(self.centralized_mpc.x[k, :], x1_guess)
            self.centralized_mpc.opti.set_initial(self.centralized_mpc.v[k, :], v1_guess)
            self.centralized_mpc.opti.set_initial(self.centralized_mpc.x[self.N+1 + k, :], x2_guess)
            self.centralized_mpc.opti.set_initial(self.centralized_mpc.v[self.N+1 + k, :], v2_guess)
            if k < self.N:
                self.centralized_mpc.opti.set_initial(self.centralized_mpc.a[k, :], np.zeros((1, self.n_dim)))
                self.centralized_mpc.opti.set_initial(self.centralized_mpc.a[self.N + k, :], np.zeros((1, self.n_dim)))
                self.centralized_mpc.opti.set_initial(self.centralized_mpc.Slack[k], 0.0)
            x1_guess = x1_guess + self.dt*v1_guess
            x2_guess = x2_guess + self.dt*v2_guess

        self.success = False
        try:
            plot_sol = False
            sol = self.centralized_mpc.opti.solve()
            self.success = True
        except:
            plot_sol = False

        x_sol = np.array(self.centralized_mpc.opti.debug.value(self.centralized_mpc.x))
        v_sol = np.array(self.centralized_mpc.opti.debug.value(self.centralized_mpc.v))
        a_sol = np.array(self.centralized_mpc.opti.debug.value(self.centralized_mpc.a))
        Slack_sol = np.array(self.centralized_mpc.opti.debug.value(self.centralized_mpc.Slack))

        x1_sol, x2_sol = x_sol[0:self.N+1, :], x_sol[self.N+1:, :]
        v1_sol, v2_sol = v_sol[0:self.N+1, :], v_sol[self.N+1:, :]
        a1_sol, a2_sol = a_sol[0:self.N, :], a_sol[self.N:, :]

        if plot_sol:
            plt.figure()
            plt.plot(x1_sol[:,0], x1_sol[:,1], 'g', linewidth=3, label="human")
            plt.plot(x2_sol[:,0], x2_sol[:,1], 'b', linewidth=3, label="robot")
            plt.legend()
            plt.grid(True)
            plt.pause(0.1)
            
        self.Centralized_MPC_sol = SimpleNamespace()
        self.Centralized_MPC_sol.x1 = x1_sol
        self.Centralized_MPC_sol.v1 = v1_sol
        self.Centralized_MPC_sol.a1 = a1_sol
        self.Centralized_MPC_sol.x2 = x2_sol
        self.Centralized_MPC_sol.v2 = v2_sol
        self.Centralized_MPC_sol.a2 = a2_sol
        self.Centralized_MPC_sol.Slack = Slack_sol

        return x1_sol, v1_sol, a1_sol, x2_sol, v2_sol, a2_sol, Slack_sol

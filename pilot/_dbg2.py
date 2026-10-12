import warnings, numpy as np, traceback
warnings.filterwarnings("ignore")
import sim
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py
from scipy.integrate import solve_ivp
last={}
class Dbg:
    @staticmethod
    def integrate(x00, par, t0, t1, dt):
        last['x00']=np.array(x00); last['par']=list(par); last['t']=(t0,t1)
        bad=[]
        def f(t,y):
            if y[6]<=0 or y[7]<=0: bad.append((t, y[6], y[7])); 
            return indpensim_ode_py(t,y,par)
        try:
            sol=solve_ivp(f,(t0,t1),x00,method='RK45',rtol=1e-6,atol=1e-6,first_step=dt)
        except Exception:
            print('bad stage values', bad[:3]); raise
        return list(sol.y[:,-1])
sim.pes.fastodeint=Dbg
from gen_data import one_seed
for seed in [1000,1001]:
    try: one_seed(seed); print('ok',seed)
    except Exception as e:
        x=last['x00']; p=last['par']
        print('seed',seed,'fail t',last['t'],repr(e)); print('H+',x[6],'T',x[7],'V',x[4],'S',x[0],'DO2',x[1],'PAA',x[29],'NH3',x[30],'visc',x[9])
        print('Fs,Fg,Fpaa,Foil,Fw,Fa,Fb', p[6],p[7],p[18],p[19],p[14],p[12],p[11])

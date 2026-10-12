import numpy as np, math
from scipy.integrate import solve_ivp
import pensimpy.peni_env_setup as pes
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py
from pensimpy.constants import STEP_IN_MINUTES, MINUTES_PER_HOUR
from _smoke import combo
info={}
class FakeFast:
    @staticmethod
    def integrate(x00, par, t0, t1, dt):
        info['x00']=list(x00); info['par']=list(par); info['t']=(t0,t1)
        bad=[]
        def f(t,y):
            if y[6]<=0: bad.append((t,y[6]))
            return indpensim_ode_py(t,y,par)
        sol = solve_ivp(f,(t0,t1),x00,method='RK45',rtol=1e-6,atol=1e-6,first_step=dt)
        return list(sol.y[:,-1])
pes.fastodeint=FakeFast
env=pes.PenSimEnv(recipe_combo=combo(),fast=True)
env.random_seed_ref=1
obs,x=env.reset(); env.yield_pre=0
for k in range(1,1151):
    v=env.recipe_combo.get_values_dict_at(time=k*STEP_IN_MINUTES/MINUTES_PER_HOUR)
    try:
        obs,x,r,done=env.step(k,x,v['Fs'],v['Foil'],v['Fg'],v['pressure'],v['discharge'],v['Fw'],v['Fpaa'])
    except Exception as e:
        print('FAIL at k',k,'t',info['t'],repr(e))
        x00=info['x00']; par=info['par']
        print('x00 pH-state y6=',x00[6],' T y7=',x00[7],' V=',x00[4],' S=',x00[0])
        print('par[5:]=',[round(p,5) if isinstance(p,float) else p for p in par[5:]])
        print('prev pH H+ series', [x.pH.y[i] for i in range(max(0,k-6),k-1)])
        break
    if k%100==0: print('k',k,'P',round(x.P.y[k-1],3),'V',round(x.V.y[k-1]),'H+',x.pH.y[k-1],'T',round(x.T.y[k-1],2))

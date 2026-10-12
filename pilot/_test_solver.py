import time, numpy as np
from scipy.integrate import solve_ivp
import pensimpy.peni_env_setup as pes
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py
# monkeypatch: emulate boost::odeint integrate() = controlled dopri5, abs=rel=1e-6
class FakeFast:
    @staticmethod
    def integrate(x00, par, t0, t1, dt):
        sol = solve_ivp(lambda t,y: indpensim_ode_py(t,y,par), (t0,t1), x00, method='RK45',
                        rtol=1e-6, atol=1e-6, first_step=dt)
        return list(sol.y[:,-1])
pes.fastodeint = FakeFast
from _smoke import combo
env = pes.PenSimEnv(recipe_combo=combo(), fast=True)
t=time.time()
df,y = env.get_batches(random_seed=1, include_raman=False)
print('seed1 yield kg', round(y,1), 'time s', round(time.time()-t,1))

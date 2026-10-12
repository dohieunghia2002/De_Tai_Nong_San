import time
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.data.constants import *
def combo():
    return RecipeCombo(recipe_dict={FS: Recipe(FS_DEFAULT_PROFILE, FS), FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG: Recipe(FG_DEFAULT_PROFILE, FG), PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE), WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
        PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)})
if __name__=="__main__":
  t=time.time()
  env=PenSimEnv(recipe_combo=combo())
  df,y=env.get_batches(random_seed=1, include_raman=False)
  print('seed1 yield kg', round(y,1), 'time s', round(time.time()-t,1), 'rows', len(df))
  print(df.columns.tolist()[:60])

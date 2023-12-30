print("hello world")

#%%
import tqdm

n = 100000

sum = 0

for i in tqdm(range(n)):
    sum = sum + i 

print(sum)
# %%

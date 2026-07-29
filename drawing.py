import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reward_type = 'aesthetic'
sampler_type = 'smc'
dup = 10
seed = 10736
cvar_beta = 0.8

if sampler_type == 'svdd':
    tag = ''
elif sampler_type == 'smc':
    tag = '_smc'

eval_rewards_nocvar = np.load(f'/home/liang.1439/SVDD-image/output_monkey/{reward_type}_MC_{dup}_monkey{tag}_{seed}.npy')
eval_rewards_cvar = np.load(f'/home/liang.1439/SVDD-image/output_monkey/{reward_type}_MC_{dup}_monkey{tag}_cvar_{seed}.npy')

print(f"eval_{reward_type}_rewards_mean Vanilla {sampler_type}:", np.mean(eval_rewards_nocvar))
print(f"eval_{reward_type}_rewards_std Vanilla {sampler_type}:", np.std(eval_rewards_nocvar))
print(f"Average of lower {1-cvar_beta} quantile {sampler_type}:", np.mean(eval_rewards_nocvar[eval_rewards_nocvar <= np.quantile(eval_rewards_nocvar, 1-cvar_beta)]))

print("--------------------------------------------------")
print(f"eval_{reward_type}_rewards_mean CVaR {sampler_type}:", np.mean(eval_rewards_cvar))
print(f"eval_{reward_type}_rewards_std CVaR {sampler_type}:", np.std(eval_rewards_cvar))
print(f"Average of lower {1-cvar_beta} quantile {sampler_type}:", np.mean(eval_rewards_cvar[eval_rewards_cvar <= np.quantile(eval_rewards_cvar, 1-cvar_beta)]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.violinplot(data=eval_rewards_nocvar, ax=axes[0], color="skyblue")
axes[0].set_title("Vanilla SVDD")

sns.violinplot(data=eval_rewards_cvar, ax=axes[1], color="salmon")
axes[1].set_title("CVaR SVDD")

plt.tight_layout()
plt.savefig(f'output_monkey/{reward_type}_MC_{dup}_monkey_{sampler_type}_{seed}.png')
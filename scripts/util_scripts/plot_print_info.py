from matplotlib import pyplot as plt
import numpy as np

# After adjusting the plot
sub_params = dict(top=0.92, bottom=0.11, left=0.12, right=0.90, hspace=0.50, wspace=0.20, )
figsize = (5.95, 4.89)
fig, axes = plt.subplots(3,1, figsize=figsize)
fig.subplots_adjust(**sub_params)

# Before manual adjustment
# fig, axes = plt.subplots(3,1)

for i in range(3):
    data = np.sin(np.arange(0, 300)* np.pi*2/300*(i+1)*2)
    axes[i].plot(data)
    axes[i].set_title(f"Joint {i+1}")

plt.show() 

# Get last subplot params
sub_params = fig.subplotpars
dict_str = "sub_params = dict("
for param in ["top", "bottom", "left", "right", "hspace", "wspace"]:
    dict_str = dict_str + f"{param}={getattr(sub_params, param):0.2f}, "
dict_str = dict_str + ")"

# Get figure size 
fig_size = fig.get_size_inches()
fig_str = f"figsize = ({fig_size[0]:0.2f}, {fig_size[1]:0.2f})"

print(dict_str)
print(fig_str)
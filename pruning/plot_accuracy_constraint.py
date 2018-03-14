from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

constraint_50 = dict()
constraint_50['unconstrained'] = [0.1284	,0.4132	,0.4353	,0.4526	,0.4674]
constraint_50['one-step'] = [0.0358	,0.42	,0.4289,	0.44,	0.4895]
constraint_50['linear'] = [0.2026,	0.3074	,0.4026,	0.4247,	0.4926]
constraint_50['exponential'] = [0.4516,	0.5625	,0.5774,	0.5805,	0.5853]

constraint_60 = dict()
constraint_60['unconstrained'] = [0.4016,	0.4074	,0.4132,	0.4411	,0.4642]
constraint_60['one-step'] = [0.4347	,0.5437,	0.5658,	0.57	,0.5816]
constraint_60['linear'] = [0.5716,	0.5737	,0.5742,	0.5816,	0.5911]
constraint_60['exponential'] = [0.5853,	0.5874	,0.5889,	0.5942	,0.5953]

constraint_70 = dict()
constraint_70['unconstrained'] = [0.4568	,0.4616,	0.4663,	0.4781	,0.5037]
constraint_70['one-step'] = [0.0395	,0.5816,	0.5874,	0.5921,	0.5968]
constraint_70['linear'] = [0.5363,	0.5874,	0.6,	0.6021,	0.6053]
constraint_70['exponential'] = [0.5911,	0.5979,	0.6021	,0.6058	,0.6079]

ylim = 1
mfc = None


fig = plt.figure(1)
ax0 = fig.add_subplot(111)
# ax0.spines['top'].set_visible(False)
# ax0.spines['right'].set_visible(False)
ax0.get_xaxis().set_ticks([])
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.xlabel('Latency constraints(ms)', labelpad=20)
plt.ylabel('Top-1 accuracy')

ax1 = fig.add_subplot(131)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_ticks([])
ax1.set_ylim([0, ylim])
ax1.set_xlim([0, 5])
plt.plot([1]*5, constraint_50['unconstrained'], 'C1o', mfc=mfc)
plt.plot([2]*5, constraint_50['one-step'], 'C2o', mfc=mfc)
plt.plot([3]*5, constraint_50['linear'], 'C3o', mfc=mfc)
plt.plot([4]*5, constraint_50['exponential'], 'C4o', mfc=mfc)
plt.xlabel('50')

ax2 = fig.add_subplot(132)
plt.xlabel('60')
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_ticks([])
ax2.set_ylim([0, ylim])
ax2.set_xlim([0, 5])
plt.plot([1]*5, constraint_60['unconstrained'], 'C1o', mfc=mfc)
plt.plot([2]*5, constraint_60['one-step'], 'C2o', mfc=mfc)
plt.plot([3]*5, constraint_60['linear'], 'C3o', mfc=mfc)
plt.plot([4]*5, constraint_60['exponential'], 'C4o', mfc=mfc)
plt.xlabel('60')

ax3 = fig.add_subplot(133)
ax3.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.get_xaxis().set_ticks([])
ax3.set_ylim([0, ylim])
ax3.set_xlim([0, 5])
plt.plot([1]*5, constraint_70['unconstrained'], 'C1o', mfc=mfc)
plt.plot([2]*5, constraint_70['one-step'], 'C2o', mfc=mfc)
plt.plot([3]*5, constraint_70['linear'], 'C3o', mfc=mfc)
plt.plot([4]*5, constraint_70['exponential'], 'C4o', mfc=mfc)
plt.xlabel('70')

unconstrained_patch = mpatches.Patch(color='C1', label='unconstrained')
onestep_patch = mpatches.Patch(color='C2', label='one-step')
linear_patch = mpatches.Patch(color='C3', label='linear')
exponential_patch = mpatches.Patch(color='C4', label='exponential')
ax3.legend(handles=[unconstrained_patch, onestep_patch, linear_patch, exponential_patch], loc='upper right')

plt.show()
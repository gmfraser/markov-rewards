import re
import sys
from collections import defaultdict
import numpy
import matplotlib.pyplot as plt
import math


if (len(sys.argv) < 3):
    print('Usage: ./value-iteration-markov-reward.py [sequence file] [reward file])')
    sys.exit('additional parameters needed')

s_file = sys.argv[1]
r_file = sys.argv[2]

f = open(s_file, 'r')
scan = f.read()
f.close()

f2 = open(r_file, 'r')
scan2 = f2.readlines()
f2.close()

'''
a dictionary containing info about rewards/punishments based on attributes that states possess.
e.g. all PM states have a punishment of -1
'''
pres_rew_dict = {}

'''
a dictionary containing info about rewards/punishments based on attributes that states do not possess.
e.g. all non-PM states have a reward of 1
'''
miss_rew_dict = {}

'''
a list of attributes determining the outcome states of interest
'''
pres_outcome_atts = []  # must be present in the outcome states
miss_outcome_atts = []  # must be missing in the outcome states

for eachline in scan2:
    eachline = re.sub('\n+', '', eachline)
    (att,pres,rew) = eachline.split(',')
    if (rew == 'outcome'):
        if (pres == 'present'):
            pres_outcome_atts.append(att)
        else:
            miss_outcome_atts.append(att)
    else:
        if (pres == 'present'):
            pres_rew_dict[att] = float(rew)
        else:
            miss_rew_dict[att] = float(rew)


alldacts = re.findall('(.+)\n', scan)
uniqdacts = sorted(list(set(alldacts)))
dict = defaultdict(list)

for i in range(0, len(alldacts)-1):
    if alldacts[i] == 'STOP':  # we do not treat STOP followed by START as a transition, 
        continue               # i.e. STOP is absorbing
    if alldacts[i+1] == 'START':
        continue
    dict[alldacts[i]].append(alldacts[i+1])

mat = []
init = [0]*len(uniqdacts)
init[0] = 1.0
initvec = numpy.matrix(init)

start = dict['START']
rewards = {}

'''
calculate the immediate rewards for each unique state.
'''
for d in uniqdacts:

    rew_total = 0
    
    for att in pres_rew_dict:
        if re.search(att, d):
           rew_total = rew_total + pres_rew_dict[att]

    for att in miss_rew_dict:
        if not re.search(att, d):
           rew_total = rew_total + miss_rew_dict[att]

    rewards[d] = rew_total
    
   
    l = []
    look_scores = []
    dentry = dict[d]

    # building the rows of the transition matrix
    if len(dentry) == 0:
        l = [0]*len(uniqdacts)
    else:
        for od in uniqdacts:
            freq = dentry.count(od) / float(len(dentry))
            l.append(freq)
            look_scores.append((od,freq))
    mat.append(l)

# transition matrix
tranmat = numpy.matrix(mat)

rewardarr = []
for d in uniqdacts:
    if d in rewards:
        rewardarr.append(rewards[d])
    else:
        rewardarr.append(0.0)

# reward vector
rewardvec = numpy.matrix(rewardarr)

newrewards = rewardvec.transpose()


# the Value Iteration algorithm
converged = False
thresh = 0.001

step = 0
discount = 0.9
while not converged:
    step += 1
    print('value iteration, step %s' % step)
    saver = newrewards
    newrewards = rewardvec.transpose() + (tranmat * (discount*newrewards))
    change = newrewards - saver
    if sum(change > thresh) == 0:
        print('converged at step %s' % step)
        converged = True
    #print change, '\n'

state_vals = [(uniqdacts[i],newrewards.item(i)) for i in range(0,len(uniqdacts))]
sort_states = sorted(state_vals, key=lambda x: x[1], reverse=True)

xaxis = []
yaxis = []
print('state values (estimated using value iteration)')
for i in sort_states:
    corpus_count = alldacts.count(i[0])
    p_outcome = [True for p in pres_outcome_atts if re.search(p, i[0])]
    m_outcome = [True for m in miss_outcome_atts if not re.search(m, i[0])]
    show = False
    if (sum(p_outcome) == len(pres_outcome_atts) and sum(m_outcome) == len(miss_outcome_atts)):
        show = True
    if show and corpus_count > 10:   # show only the states corresponding to outcomes of interest
        print(i, alldacts.count(i[0]))
        xaxis.append(i[1])
        yaxis.append(math.log(corpus_count))


plt.scatter(xaxis, yaxis)
plt.xlabel('Estimated Value of State')
plt.ylabel('Log Frequency of State')
plt.show()



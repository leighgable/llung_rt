Based on "Self Distillation Enables Continual Learning" and
        "Reinforcement Learning via Self Distillation" by Hubotter et al.

Self-distillation is deceptively simple but powerful. By using one model as
both the student and teacher, but augmenting the teacher model with context
that helps it generate a correct and confident answer, the student can learn
from a teacher that share its same prior, unlocking on-policy learning.
In both structured fine tuning and reinforcement learning settings, this
allows for dense reward signals--the holy grail of reinforcement learning.


Furthermore, in the RL setting, its been observed that the models can learn
new behaviours, but not new information. With self-distillation that has
changed, and now they can both update their knowledge, and acquire reasoning,
tool use, or whatever other skills one would like.  
Incidentally, this also seems to at least partially solve catastrophic
forgetting.   

My idea is to augment self-distillation with power sequential Monte Carlo
sampling which magically gives you a probability density estimate for
**future** tokens in the trajectory!

To do this efficiently on the GPU, there are two "particle" kv
cache implementations in the kv_cache.py file. The first one is for 4 to 8
particle generations. If you go for 8 to 32, then you'll need the second one.

In the explore.py file, are a couple of functions for "surprisal" annealing.
The idea is early on in training, encourage the model to try some unlikely
approaches in the hopes that it will unlock some creativity. And as training
progresses, the model will start to zero in on strategies that "60% percent
of the time work all the time." It does this by measuring surprisal along
sampled trajectory, and favoring sequences from the teacher that surprise
the student, ie. high surprisal and high likelihood instead of purely high
likelihood, or a top-k of high likelihood, which is the usual case.

Potential datasets:  
 * allenai/Dolci-Instruct-SFT-Tool-Use  



# DeepLane: Parallel DQN for Autonomous Lane Keeping
1. Introduction:

Self-driving cars need to stay in their lanes, which is an important skill for them. This requires accurate control, a strong understanding of their surroundings, and quick decision-making in changing road conditions. Standard rule-based controllers often struggle to cope with changes in lane shapes, lighting conditions, and nearby traffic. This has led to the use of deep reinforcement learning (DRL) to create more flexible and responsive behavior. DeepLane uses a Parallel Deep Q-Network (DQN) system to speed up training and make policies more stable. It does this by gathering experiences from several simulated environments at the same time. This parallel approach not only reduces the similarities between samples but also enables the agent to learn from a wide variety of driving situations. DeepLane combines convolutional perception with value-based decision-making to better manage complicated lane changes, curves, and unexpected obstacles. The experimental results show that this model provides smoother control, reaches solutions more quickly, and is more reliable than traditional single-agent DQN methods. DeepLane offers a flexible deep reinforcement learning system that helps make autonomous lane-keeping technology ready for real-world use.

2. Parallel Deep Q-network (DQN)

We developed a method called DeepLane that treats lane keeping as a decision-making problem. We solve this problem using a parallel Deep Q-Network (DQN). 
The custom LaneKeepingEnv simulates how a vehicle moves sideways while it drives at a steady speed, using specific steering commands. "The situation at time t is"

st​=[yt​, ψt​, xt​, k(xt​), k(xt​+dprev​)],

where 𝑦𝑡 is the lateral offset from the lane center, 𝜓𝑡 is the heading error, 𝑥𝑡 is the longitudinal position, k(⋅) is the lane curvature, and 
𝑑_prev = 15m is a preview distance. The action space is a 15-level discretization of the steering angle 
𝛿𝑡 ∈[−0.25,0.25] rad. A simple safety layer clamps and rate-limits the steering command so that 
∣𝛿𝑡∣≤0.25rad and ∣𝛿𝑡−𝛿𝑡_1∣≤0.5, which keeps the policy within physically reasonable bounds.



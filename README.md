# DeepLane: Parallel DQN for Autonomous Lane Keeping
1. Introduction:

Lane-keeping, a crucial skill for self-driving cars, requires precise control, a profound understanding of the environment, and fast decision-making regarding changing road circumstances. Standard controllers based on explicit rules are often unable to keep up with changes in lane shapes, lighting conditions, and local traffic. Recently, several works have focused on using deep reinforcement learning to create more flexible and responsive behavior. DeepLane considers a Parallel DQN system to accelerate the training process and enhance policy stability. This is achieved by leveraging experiences from multiple simulated environments simultaneously. Such parallel exploration reduces the similarities between samples and allows the agent to learn from a wide variety of driving situations. DeepLane proposes to combine convolutional perception with value-based decisions to better handle complex lane changes, curves, and unexpected obstacles. From the experiments, the proposed model produces smoother control, reaches solutions faster, and is more reliable compared to common single-agent DQN methods. DeepLane presents a flexible deep reinforcement learning system to help autonomous lane-keeping technology become ready for real-world applications.

2. Parallel Deep Q-network (DQN)

To understand a Parallel Deep Q-network, we first start with a Deep Q-network. 
A DQN is a type of reinforcement learning that mixes Q-learning with deep neural networks to estimate the Q-value function. 
The neural network makes predictions instead of relying on a Q-table. Q(s, a) shows the expected long-term benefit of choosing action a when in state s. 
The aim is to discover the best action for each situation so that the agent can increase its total rewards over time.
A Parallel DQN builds on this concept by operating multiple environments simultaneously. "Every environment creates sets of experiences." 

(s,a,r,s′)

All these experiences are kept in a shared replay buffer. 
This allows the learner to train the Q-network with a large, varied, and less related group of samples. This results in quicker training, improved exploration, and more reliable learning than a DQN that only works in one environment.

We created a method named DeepLane that treats lane keeping as a decision-making challenge using reinforcement learning. 
DeepLane uses a Parallel Deep Q-Network to figure out how to keep the car in the middle of the lane while driving at a constant speed.
Our custom LaneKeepingEnv simulates how a vehicle moves side to side within its lane and reacts to specific steering instructions. 
By using a parallel DQN, DeepLane can collect different driving experiences from various simulated environments. 
This approach speeds up training, makes it more stable, and helps the system perform better overall compared to using just one environment with a DQN.

The input to the network is a 5-dimensional state vector: 

𝛿t​=[yt​, ψt​, xt​, k(xt​), k(xt​+d_prev​)],

where 
- 𝑦𝑡 is the lateral offset from the lane center
- 𝜓𝑡 is the heading error
- 𝑥𝑡 is the longitudinal position
- k(⋅) is the lane curvature
- 𝑑_prev = 15m is a preview distance.

The Output is a 15-level discretization of the steering angle 
𝛿𝑡 ∈[−0.25,0.25] rad. 
A simple safety layer clamps and rate-limits the steering command so that ∣𝛿𝑡∣ ≤ 0.25 rad and ∣𝛿𝑡−𝛿𝑡_1∣ ≤ 0.5s^-1, which keeps the policy within physically reasonable bounds.

3. Reward function

The reward is carefully shaped to encourage stable, centered, and smooth driving. It includes:
- Lateral error penalty: 𝑦_t^2
- Heading error penalty: 𝜓_𝑡^2
- Lateral velocity penalty: 𝑦_t^2
- ​Steering effort penalty: 𝛿_𝑡^2
- Steering rate penalty: (𝛿𝑡−𝛿𝑡_−1)^2

Soft boundary penalty when the car is close to the lane edges
Progress bonus for moving forward along the lane centerline
This reward design prevents aggressive actions, encourages smooth control, and keeps the car centered.

4. Loss Function 
DeepLane uses the standard DQN temporal difference (TD)

L(θ)=E[(ytarget​−Qθ​(st​,at​))^2]

Where the target Q-value is:
ytarget​=rt​+γa′max​Qθ−​(st+1​,a′)

- 𝑄𝜃​ is the online network.
- 𝑄𝜃− is the target network (updated every 8000 steps).
- 𝛾=0.99 is the discount factor.
The loss encourages the network to match its predictions to a stable TD target.
Experience replay (buffer of 300,000 transitions) is used to decorrelate samples.
This loss function allows DeepLane to learn long-term lane-keeping behavior.



Experiment 

This work investigates whether a Parallel DQN can learn a stable, reliable lane-keeping policy in a noisy, dynamically changing road environment. We seek to establish if parallel training enhances learning speed, policy stability, and generalization over a standard single-environment DQN. The present study also examines how well the agent maintains lane position, tracks the centerline, and generates smooth steering behavior. The robustness of the learned control strategy is assessed by testing the trained policy on unseen lane geometries. Overall, this research aims to confirm that DeepLane offers safe and efficient autonomous lane-keeping viable for real-world deployment.


<img width="687" height="539" alt="image" src="https://github.com/user-attachments/assets/bd452934-8124-41f5-8d32-f93fa2f6b9f6" />




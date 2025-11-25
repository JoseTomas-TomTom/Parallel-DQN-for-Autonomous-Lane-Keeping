

# DeepLane: Parallel DQN for Autonomous Lane Keeping
1. Introduction:

Lane-keeping, a crucial skill for self-driving cars, requires precise control, a profound understanding of the environment, and fast decision-making regarding changing road circumstances. Standard controllers based on explicit rules are often unable to keep up with changes in lane shapes, lighting conditions, and local traffic. Recently, several works have focused on using deep reinforcement learning to create more flexible and responsive behavior. DeepLane considers a Parallel DQN system to accelerate the training process and enhance policy stability. This is achieved by leveraging experiences from multiple simulated environments simultaneously. Such parallel exploration reduces the similarities between samples and allows the agent to learn from a wide variety of driving situations. DeepLane proposes to combine convolutional perception with value-based decisions to better handle complex lane changes, curves, and unexpected obstacles. From the experiments, the proposed model produces smoother control, reaches solutions faster, and is more reliable compared to common single-agent DQN methods. DeepLane presents a flexible deep reinforcement learning system to help autonomous lane-keeping technology become ready for real-world applications.

2. Parallel Deep Q-network (DQN)

We developed a method called DeepLane that treats lane keeping as a decision-making problem. We solve this problem using a parallel Deep Q-Network (DQN). 
The custom LaneKeepingEnv simulates how a vehicle moves sideways while it drives at a steady speed, using specific steering commands. 

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

5. Baseline Architecture Details

The baseline is a standard DQN agent with the same 256-256 MLP policy, replay buffer size and batch size as DeepLane but trained with a single environment, without VecNormalize and without the curvature curriculum. This will isolate the effect of parallel training and normalization.

Experiment 




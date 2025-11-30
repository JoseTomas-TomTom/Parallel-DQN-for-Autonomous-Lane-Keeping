# DeepLane: Parallel DQN for Autonomous Lane Keeping
1. Introduction:

Maintaining proper lane positioning is essential for autonomous vehicles. This skill demands precise control, a thorough understanding of the surroundings, and the ability to swiftly adapt to road changes. Traditional controllers that adhere strictly to predefined rules often struggle with variations in lane configurations, lighting conditions, and the surrounding traffic. Recently, numerous studies have investigated how deep reinforcement learning can foster more adaptable and responsive behavior. DeepLane employs a Parallel DQN system to accelerate training and enhance the reliability of its approaches. By utilizing experiences from several simulated environments simultaneously, this method diversifies the learning samples, enabling the agent to draw insights from a broader spectrum of driving situations. Testing results show that this model delivers smoother handling, quicker problem-solving, and greater dependability compared to conventional single-agent DQN strategies. DeepLane represents a flexible deep reinforcement learning framework that supports the advancement of lane-keeping technology for real-world use.

2. Parallel Deep Q-network (DQN)

To understand a Parallel Deep Q-network, we first start with a Deep Q-network. 
A DQN is a type of reinforcement learning that mixes Q-learning with deep neural networks to estimate the Q-value function. 
The neural network makes predictions instead of relying on a Q-table. Q(s, a) shows the expected long-term benefit of choosing action a when in state s. 
The aim is to discover the best action for each situation so that the agent can increase its total rewards over time.
A Parallel DQN builds on this concept by operating multiple environments simultaneously. "Every environment creates sets of experiences." 

(s,a,r,s′)

All these experiences are kept in a shared replay buffer. 
This allows the learner to train the Q-network with a large, varied, and less related group of samples. 
This results in quicker training, improved exploration, and more reliable learning than a DQN that only works in one environment.

We created a method named DeepLane that treats lane keeping as a decision-making challenge using reinforcement learning. 
DeepLane uses a Parallel Deep Q-Network to figure out how to keep the car in the middle of the lane while driving at a constant speed.
Our custom LaneKeepingEnv simulates how a vehicle moves side to side within its lane and reacts to specific steering instructions. 
By using a parallel DQN, DeepLane can collect different driving experiences from various simulated environments. 
This approach speeds up training, makes it more stable, and helps the system perform better overall compared to using just one environment with a DQN.


The input to the network is a 5-dimensional state vector

s_t = [ y_t, ψ_t, x_t, k(x_t), k(x_t + d_preview) ],

where

- y_t is the lateral offset from the lane center,
- ψ_t is the heading error,
- x_t is the longitudinal position,
- k(·) is the lane curvature,
- d_preview = 15 m is a preview distance.

Although curvature values are included in the observation, the current implementation uses a straight lane; curvature is provided only as a noisy preview feature rather than modifying the lane geometry.


The network output is a discrete steering action δ_t selected from 15 uniformly spaced values in the range [−0.25, 0.25] rad. 
A simple safety layer clamps and rate-limits the steering command so that |δ_t| ≤ 0.25 rad and |δ_t − δ_{t−1}| ≤ 0.5 rad/s, which keeps the policy within physically reasonable bounds.


3. Reward function

The reward is carefully shaped to encourage stable, centered, and smooth driving. At each time step t, we compute

r_t = 1− λ_y y_t^2− λ_ψ ψ_t^2− λ_{ẏ} ẏ_t^2− λ_δ δ_t^2− λ_{Δδ} (δ_t − δ_{t−1})^2− edge_pen(y_t)+ progress_bonus(x_t, ψ_t),

where

- y_t is the lateral offset,
- ψ_t is the heading error,
- ẏ_t = v sin(ψ_t) is the lateral velocity,
- δ_t is the steering angle,
- δ_{t−1} is the previous steering angle,

and the weights are

- λ_y = 0.8,
- λ_ψ = 0.25,
- λ_{ẏ} = 0.08,
- λ_δ = 0.002,
- λ_{Δδ} = 0.05.

The **soft boundary penalty** `edge_pen(y_t)` ramps up as the vehicle approaches the lane edges:
it is zero near the center and increases smoothly within the outer 0.3 m of the lane, up to a maximum penalty of 0.5 when |y_t| is close to 1.8 m.

The **progress bonus** is a small positive term proportional to v cos(ψ_t), which rewards moving forward while aligned with the lane centerline.

This reward design discourages aggressive steering, punishes large lateral and heading errors, and encourages smooth, centered lane keeping.

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

5. Training setup

DeepLane is trained using a Parallel DQN, which runs in 8 different environments at the same time. Each `SubprocVecEnv` runs its own version of the custom `LaneKeepingEnv`. 
The vehicle travels at a constant speed of 15 m/s with a simulation time step of Δt = 0.05 s.
The wheelbase is 2.7 m, matching the parameters used in the LaneKeepingEnv dynamics.
We use `VecNormalize` to adjust our observations while still keeping the original rewards.

Key DQN hyperparameters:

- Network architecture: MLP with hidden layers [256, 256] and ReLU activations,
- Discount factor: γ = 0.99,
- Replay buffer size: 300,000 transitions (shared by all environments),
- Batch size: 256,
- Learning rate: linearly decayed from 3e−4 to 0 over training,
- Target network update interval: 8,000 environment steps,
- Exploration: ε decayed from 1.0 to 0.02 over 35% of training,
- Total training steps: 400,000 environment steps.

Sampling from eight different environments at the same time boosts the variety of experiences. 
The replay buffer improves learning speed and stability when compared to using a single environment in DQN.
 
6. Experiment 

This work investigates whether a Parallel DQN can learn a stable, reliable lane-keeping policy in a noisy, dynamically changing road environment. We seek to establish if parallel training enhances learning speed, policy stability, and generalization over a standard single-environment DQN. The present study also examines how well the agent maintains lane position, tracks the centerline, and generates smooth steering behavior. The robustness of the learned control strategy is assessed by testing the trained policy on unseen lane geometries. Overall, this research aims to confirm that DeepLane offers safe and efficient autonomous lane-keeping viable for real-world deployment.

We evaluate the trained policy on held-out random lane geometries and report two
simple metrics:


**Centerline RMSE**: root-mean-square lateral error √E[y_t^2],
**Lane retention**: percentage of time steps with |y_t| ≤ 1.8 m.


Even in the face of sensor noise and randomized curvature characteristics, the parallel DQN strategy often achieves low lateral RMSE and remains inside the lane borders for the whole 30-second episodes.
The figure below shows one representative evaluation rollout:


<img width="687" height="539" alt="image" src="https://github.com/user-attachments/assets/bd452934-8124-41f5-8d32-f93fa2f6b9f6" />


Trajectory Plot (Top graph)
- This graph displays the vehicle's lateral position 𝑦𝑡 over time:
- The lane center is shown by the dashed line at 𝑦=0.
- The red dotted lines show the lane limits at 𝑦=±1.8 m.
- The blue curve depicts the vehicle's route throughout the episode.
- The vehicle's ultimate position is displayed by the orange triangle.

Interpretation
- The agent starts with a little offset.
- It rapidly centers itself, demonstrating the effective correction of heading and lateral inaccuracy by the learnt controller.
- The agent never exceeds the lane borders during the full 600-step rollout.
- When the controller makes small heading adjustments, the trajectory stays steady but exhibits slight oscillations.

This demonstrates that the policy learned by Parallel DQN maintains safe lane-keeping performance under noise.

Steering Command Plot (bottom Graph)
- The steering angle 𝛿𝑡 is displayed on the bottom graph: There are fifteen distinct ways to steer.
- A safety layer prevents sudden changes by enforcing a rate restriction of 0.5 rad/s.
Interpretation 
- While the agent corrects the first lateral mistake, steering adjustments are more significant early in the episode.
- The agent uses tiny, regulated oscillations to stay on track after stabilizing.
- As the run comes to a close, noise and shifting curvature cause the oscillations to significantly increase, but the rate limiter maintains the motion's smoothness and physical realism.

These steering patterns verify that the safety layer and incentive structure effectively encourage smooth control, not only low lane error.



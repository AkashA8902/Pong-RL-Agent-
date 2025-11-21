#!/usr/bin/env python3
"""
Build and Train Agent for Atari Games - Organized Directory Structure
ECE612 Assignment 5 - Applied Reinforcement Learning with TF-Agents

Usage: python buildAndTrainAgent-GAME.py [GAME_NAME]
where GAME_NAME is either "PongEasy" or "PongHard"
"""

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import time
import gym

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF-Agents imports
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.utils.common import function
from tf_agents.policies.policy_saver import PolicySaver

# Game Configuration
GAMES = {
    "PongEasy": {
        "environment_name": "PongNoFrameskip-v4",
        "training_iterations": 10000,  # Reduced for testing
        "difficulty": "easier",
        "checkpoint_interval": 2500,
        "policy_save_interval": 2500,
        "max_episode_steps": 27000,
        "learning_rate": 2.5e-4,
        "target_update_period": 2000
    },
    "PongHard": {
        "environment_name": "PongNoFrameskip-v4", 
        "training_iterations": 15000,  # More training for harder variant
        "difficulty": "harder",
        "checkpoint_interval": 3000,
        "policy_save_interval": 3000,
        "max_episode_steps": 20000,  # Shorter episodes for harder difficulty
        "learning_rate": 1.5e-4,  # Slower learning for harder variant
        "target_update_period": 1500  # More frequent target updates
    }
}

# Training hyperparameters
MAX_EPISODE_STEPS = 27000  # Will be overridden by game config
INITIAL_COLLECT_STEPS = 5000  # Reduced for faster startup
BATCH_SIZE = 64
REPLAY_BUFFER_MAX_LENGTH = 100000  # Reduced for memory
UPDATE_PERIOD = 4
TARGET_UPDATE_PERIOD = 2000  # Will be overridden by game config
LEARNING_RATE = 2.5e-4  # Will be overridden by game config
GAMMA = 0.99
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_STEPS = 250000 // UPDATE_PERIOD

class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    """Custom Atari preprocessing with auto-fire functionality."""
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # Fire to start the game
        super().step(1)
        return obs

    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        # Auto-fire after losing a life
        if self.ale.lives() < lives_before_action and not done:
            super().step(1)
        return obs, rewards, done, info

import gym

class PongHardWrapper(gym.Wrapper):
    """Wrapper to make Pong more challenging by modifying game dynamics."""
    
    def __init__(self, env):
        super(PongHardWrapper, self).__init__(env)
        self.step_count = 0
        self.consecutive_actions = 0
        self.reward_modifier = 1.0
    
    def reset(self, **kwargs):
        self.step_count = 0
        self.consecutive_actions = 0
        self.reward_modifier = 1.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Make rewards more extreme for harder difficulty
        if reward > 0:  # Player scored
            reward *= 1.5  # Higher reward for scoring
        elif reward < 0:  # Player lost point
            reward *= 1.8  # Higher penalty for losing
        
        # Add slight penalty for long episodes to encourage faster play
        self.step_count += 1
        if self.step_count > 15000:  # After 15k steps, add time pressure
            reward -= 0.01  # Small penalty to encourage decisive play
        
        return obs, reward, done, info
class ProgressTracker:
    """Simple progress tracking."""
    
    def __init__(self, total_steps):
        self.counter = 0
        self.total = total_steps
        self.start_time = time.time()

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
            if self.counter % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = self.counter / elapsed if elapsed > 0 else 0
                print(f"\rProgress: {self.counter}/{self.total} ({rate:.1f} steps/sec)", end="")

def get_game_selection():
    """Get game selection from command line or user input."""
    # Check command line arguments
    if len(sys.argv) > 1:
        game_arg = sys.argv[1].strip()
        # Handle different possible inputs
        if game_arg.lower() in ["pongeasy", "easy", "e"]:
            return "PongEasy"
        elif game_arg.lower() in ["ponghard", "hard", "h"]:
            return "PongHard"
        else:
            print(f"Unknown game '{game_arg}'. Please choose 'PongEasy' or 'PongHard'.")
    
    # Interactive selection
    print("Available Games:")
    print("1. PongEasy (Easier - standard Pong)")
    print("2. PongHard (Harder - enhanced difficulty Pong)")
    
    while True:
        choice = input("\nSelect game (1/2 or PongEasy/PongHard): ").strip()
        if choice in ["1", "pongeasy", "PongEasy", "easy", "e", "E"]:
            return "PongEasy"
        elif choice in ["2", "ponghard", "PongHard", "hard", "h", "H"]:
            return "PongHard"
        else:
            print("Invalid choice. Please enter 1, 2, PongEasy, or PongHard.")
def create_environment(game_name):
    """Create the Atari environment."""
    config = GAMES[game_name]
    environment_name = config["environment_name"]
    max_episode_steps = config["max_episode_steps"]
    
    print(f"Creating {game_name} environment...")
    
    # Create base wrappers
    wrappers = [AtariPreprocessingWithAutoFire, FrameStack4]
    
    # Add PongHard wrapper if needed
    if game_name == "PongHard":
        print("Applying PongHard difficulty modifications...")
        wrappers.insert(-1, PongHardWrapper)  # Insert before FrameStack4
    
    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=wrappers
    )
    
    tf_env = TFPyEnvironment(env)
    return tf_env
def create_q_network(tf_env):
    """Create the Q-Network."""
    print("Creating Q-Network...")
    preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]
    
    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )
    return q_net

def create_agent(tf_env, q_net, game_name):
    """Create the DQN agent with game-specific parameters."""
    print("Creating DQN agent...")
    config = GAMES[game_name]
    
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=config["learning_rate"],
        rho=0.95,
        momentum=0.0,
        epsilon=0.00001,
        centered=True
    )
    
    train_step = tf.Variable(0)
    
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=EPSILON_INITIAL,
        decay_steps=EPSILON_DECAY_STEPS,
        end_learning_rate=EPSILON_FINAL
    )
    
    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=config["target_update_period"],
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=GAMMA,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step)
    )
    
    agent.initialize()
    return agent
def save_model_organized(agent, game_name, iteration=None):
    """Save model in organized directory structure."""
    model_dir = os.path.join("models", game_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # Save with iteration number if provided, otherwise as final
    if iteration is not None:
        model_name = f"model_step_{iteration}"
    else:
        model_name = "final_model"
    
    model_path = os.path.join(model_dir, model_name)
    
    # Try different saving methods in order of preference
    try:
        print(f"Saving model using PolicySaver to {model_path}...")
        policy_saver = PolicySaver(agent.policy)
        policy_saver.save(model_path)
        print(f"Model saved successfully using PolicySaver")
        return model_path
    except Exception as e:
        print(f"PolicySaver failed: {e}")
        
        try:
            print(f"Trying tf.saved_model.save...")
            tf.saved_model.save(agent.policy, model_path + "_savedmodel")
            print(f"Model saved using tf.saved_model.save")
            return model_path + "_savedmodel"
        except Exception as e2:
            print(f"tf.saved_model.save failed: {e2}")
            
            try:
                print(f"Saving Q-network weights as backup...")
                weights_path = model_path + "_weights"
                agent.q_network.save_weights(weights_path)
                print(f"Q-network weights saved to {weights_path}")
                return weights_path
            except Exception as e3:
                print(f"Weight saving also failed: {e3}")
                print("Model could not be saved!")
                return None

def save_checkpoint_organized(agent, game_name, step):
    """Save checkpoint in organized directory."""
    checkpoint_dir = os.path.join("checkpoints", game_name.lower())
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        from tf_agents.utils import common
        checkpoint = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            global_step=step
        )
        checkpoint.save(step)
        print(f"Checkpoint saved to {checkpoint_dir}")
        return True
    except Exception as e:
        print(f"Checkpoint save failed: {e}")
        return False
def train_agent_organized(game_name):
    """Training with organized file structure."""
    config = GAMES[game_name]
    n_iterations = config["training_iterations"]
    checkpoint_interval = config["checkpoint_interval"]
    policy_save_interval = config["policy_save_interval"]
    
    print(f"\nStarting training for {game_name}")
    print(f"Difficulty: {config['difficulty']}")
    print(f"Training iterations: {n_iterations}")
    
    # Create environment
    tf_env = create_environment(game_name)
    
    # Create Q-network and agent
    q_net = create_q_network(tf_env)
    agent = create_agent(tf_env, q_net, game_name)
    
    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=REPLAY_BUFFER_MAX_LENGTH
    )
    
    # Create metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]
    
    # Initial data collection
    print(f"Collecting {INITIAL_COLLECT_STEPS} initial random steps...")
    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ProgressTracker(INITIAL_COLLECT_STEPS)],
        num_steps=INITIAL_COLLECT_STEPS
    )
    
    init_driver.run()
    print("\nInitial data collection completed!")
    
    # Setup training
    print("Setting up training...")
    dataset = replay_buffer.as_dataset(
        sample_batch_size=BATCH_SIZE,
        num_steps=2,
        num_parallel_calls=3
    ).prefetch(3)
    
    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=UPDATE_PERIOD
    )
    
    # Convert to TensorFlow functions
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)
    
    # Training loop
    print(f"Starting main training loop...")
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    
    training_data = {
        'game': game_name,
        'iterations': [],
        'rewards': [],
        'episodes': [],
        'losses': [],
        'steps': [],
        'model_saves': []  # Track when models were saved
    }
    
    log_interval = 1000
    
    try:
        for iteration in range(n_iterations):
            # Collect experience
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            
            # Train the agent
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            
            # Progress update
            if iteration % 100 == 0:
                loss_val = train_loss.loss.numpy()
                print(f"\rIteration {iteration}/{n_iterations}, Loss: {loss_val:.5f}", end="")
            
            # Detailed logging
            if iteration % log_interval == 0:
                print(f"\n--- Training Progress: Iteration {iteration}/{n_iterations} ---")
                
                # Get metrics
                avg_return = train_metrics[2].result().numpy()
                num_episodes = train_metrics[0].result().numpy()
                env_steps = train_metrics[1].result().numpy()
                loss_val = train_loss.loss.numpy()
                training_step = agent.train_step_counter.numpy()
                
                print(f"Average Return: {avg_return:.2f}")
                print(f"Episodes: {num_episodes}")
                print(f"Environment Steps: {env_steps}")
                print(f"Training Loss: {loss_val:.5f}")
                print(f"Training Step: {training_step}")
                
                # Store data
                training_data['iterations'].append(iteration)
                training_data['rewards'].append(avg_return)
                training_data['episodes'].append(num_episodes)
                training_data['losses'].append(loss_val)
                training_data['steps'].append(training_step)
            
            # Save checkpoint periodically
            if iteration % checkpoint_interval == 0 and iteration > 0:
                print(f"\nSaving checkpoint at iteration {iteration}...")
                save_checkpoint_organized(agent, game_name, agent.train_step_counter)
            
            # Save model periodically
            if iteration % policy_save_interval == 0 and iteration > 0:
                print(f"\nSaving model at iteration {iteration}...")
                saved_path = save_model_organized(agent, game_name, iteration)
                if saved_path:
                    training_data['model_saves'].append((iteration, saved_path))
    
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user at iteration {iteration}")
    except Exception as e:
        print(f"\nTraining error: {e}")
    
    # Final results and saves
    print(f"\n\nTraining completed for {game_name}!")
    
    # Save final model
    print("Saving final model...")
    final_model_path = save_model_organized(agent, game_name)
    if final_model_path:
        training_data['model_saves'].append(('final', final_model_path))
        print(f"Final model saved to: {final_model_path}")
    
    # Save final checkpoint
    print("Saving final checkpoint...")
    save_checkpoint_organized(agent, game_name, agent.train_step_counter)
    
    # Save training summary to training_data directory
    training_data_dir = "training_data"
    os.makedirs(training_data_dir, exist_ok=True)
    
    try:
        # Enhanced training summary
        training_summary = {
            'game': game_name,
            'config': config,
            'final_metrics': {
                'avg_return': train_metrics[2].result().numpy(),
                'episodes': train_metrics[0].result().numpy(),
                'env_steps': train_metrics[1].result().numpy(),
                'final_loss': training_data['losses'][-1] if training_data['losses'] else 0
            },
            'training_data': training_data,
            'hyperparameters': {
                'max_episode_steps': config['max_episode_steps'],
                'initial_collect_steps': INITIAL_COLLECT_STEPS,
                'batch_size': BATCH_SIZE,
                'replay_buffer_max_length': REPLAY_BUFFER_MAX_LENGTH,
                'update_period': UPDATE_PERIOD,
                'target_update_period': config['target_update_period'],
                'learning_rate': config['learning_rate'],
                'gamma': GAMMA,
                'epsilon_initial': EPSILON_INITIAL,
                'epsilon_final': EPSILON_FINAL
            },
            'directory_structure': {
                'models': f"models/{game_name.lower()}/",
                'checkpoints': f"checkpoints/{game_name.lower()}/",
                'training_data': "training_data/"
            }
        }
        
        summary_path = os.path.join(training_data_dir, f"{game_name}_training_summary.pkl")
        with open(summary_path, 'wb') as f:
            pickle.dump(training_summary, f)
        print(f"Training summary saved to {summary_path}")
        
    except Exception as e:
        print(f"Error saving training summary: {e}")
    
    # Print organized summary
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETED - {game_name.upper()}")
    print("=" * 60)
    print(f"Game Difficulty: {config['difficulty'].title()}")
    print(f"Total Iterations: {n_iterations}")
    
    if training_data['rewards']:
        final_reward = training_data['rewards'][-1]
        best_reward = max(training_data['rewards'])
        print(f"Final Average Reward: {final_reward:.2f}")
        print(f"Best Average Reward: {best_reward:.2f}")
    
    print(f"\nFiles Created in Organized Structure:")
    print(f"- Models: models/{game_name.lower()}/")
    print(f"- Checkpoints: checkpoints/{game_name.lower()}/")
    print(f"- Training Data: training_data/{game_name}_training_summary.pkl")
    
    print(f"\nNext Steps:")
    print(f"1. Train the other Pong variant if not done yet")
    print(f"2. Run: python createTrainingCurvesAndVideos.py")
    print(f"3. Or test this agent: python deployGamePlayer.py")
    
    return agent

def main():
    """Main function to build and train the agent."""
    print("ECE612 Assignment 5 - Build and Train Agent")
    print("=" * 50)
    
    # Get game selection
    try:
        game_name = get_game_selection()
    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        return
    
    print(f"\nSelected: {game_name}")
    print(f"Difficulty: {GAMES[game_name]['difficulty']}")
    print(f"Training Iterations: {GAMES[game_name]['training_iterations']}")
    
    # Show game-specific details
    if game_name == "PongHard":
        print("\nPongHard Features:")
        print("- Enhanced reward/penalty structure")
        print("- Shorter maximum episode length")
        print("- Time pressure penalties")
        print("- Optimized learning rate for difficulty")
    else:
        print("\nPongEasy Features:")
        print("- Standard Pong difficulty")
        print("- Default reward structure")
        print("- Standard episode length")
    
    confirm = input(f"\nProceed with {game_name} training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    try:
        # Train the agent
        trained_agent = train_agent_organized(game_name)
        print(f"\nTraining and saving completed for {game_name}!")
        
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user.")
        print("Partial progress has been saved.")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Check that all dependencies are installed (run setup_project.py)")

if __name__ == "__main__":
    main()
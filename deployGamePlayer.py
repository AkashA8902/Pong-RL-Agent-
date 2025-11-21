#!/usr/bin/env python3
"""
Deploy Game Player - Load Trained Policy and Record Gameplay
ECE612 Assignment 5 - Applied Reinforcement Learning with TF-Agents

This script loads a trained policy and records gameplay videos,
demonstrating the agent's learned behavior.
"""

import os
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver
import PIL
import pickle
import logging
import glob
import re
import time
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger().setLevel(logging.INFO)

# Game Configuration (matching training script)
GAMES = {
    "PongEasy": {
        "environment_name": "PongNoFrameskip-v4",
        "difficulty": "easier",
        "max_episode_steps": 27000,
        "learning_rate": 2.5e-4,
        "target_update_period": 2000,
        "description": "Standard Pong difficulty"
    },
    "PongHard": {
        "environment_name": "PongNoFrameskip-v4", 
        "difficulty": "harder",
        "max_episode_steps": 20000,
        "learning_rate": 1.5e-4,
        "target_update_period": 1500,
        "description": "Enhanced difficulty with modified rewards"
    }
}

class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    """Custom Atari preprocessing with auto-fire functionality."""
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1)  # FIRE to start
        return obs

    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            super().step(1)  # FIRE to start after life lost
        return obs, rewards, done, info

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

class GameplayRecorder:
    """Records gameplay frames and statistics with proper Pong scoring."""
    
    def __init__(self):
        self.frames = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Proper Pong scoring tracking
        self.agent_score = 0      # Our trained agent's score
        self.opponent_score = 0   # Opponent AI's score
        self.total_points = 0     # Total points in the game
        
        self.episodes_completed = 0
        self.scoring_events = []  # Track all scoring events with timestamps
        
    def reset_episode(self):
        """Reset episode tracking."""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episodes_completed += 1
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Reset scores for new episode
        self.agent_score = 0
        self.opponent_score = 0
    
    def record_step(self, frame, reward):
        """Record a single step with proper scoring."""
        self.frames.append(frame)
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Track Pong scoring properly
        if reward > 0:  # Agent scored
            self.agent_score += 1
            self.total_points += 1
            self.scoring_events.append({
                'frame': len(self.frames),
                'scorer': 'agent',
                'agent_score': self.agent_score,
                'opponent_score': self.opponent_score
            })
        elif reward < 0:  # Opponent scored
            self.opponent_score += 1
            self.total_points += 1
            self.scoring_events.append({
                'frame': len(self.frames),
                'scorer': 'opponent',
                'agent_score': self.agent_score,
                'opponent_score': self.opponent_score
            })
    
    def get_statistics(self):
        """Get gameplay statistics with proper scoring."""
        if not self.episode_rewards:
            return {
                'total_steps': len(self.frames),
                'episodes_completed': 0,
                'average_reward': 0,
                'agent_score': self.agent_score,
                'opponent_score': self.opponent_score,
                'total_points': self.total_points,
                'current_episode_reward': self.current_episode_reward,
                'scoring_events': len(self.scoring_events)
            }
        
        return {
            'total_steps': len(self.frames),
            'episodes_completed': self.episodes_completed,
            'average_reward': np.mean(self.episode_rewards),
            'best_episode_reward': max(self.episode_rewards),
            'average_episode_length': np.mean(self.episode_lengths),
            'agent_score': self.agent_score,
            'opponent_score': self.opponent_score,
            'total_points': self.total_points,
            'current_episode_reward': self.current_episode_reward,
            'scoring_events': len(self.scoring_events)
        }

def create_environment(game_name):
    """Create environment for the specified game."""
    if game_name not in GAMES:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAMES.keys())}")
    
    config = GAMES[game_name]
    
    print(f"Creating {game_name} environment...")
    print(f"Description: {config['description']}")
    
    # Create base wrappers
    wrappers = [AtariPreprocessingWithAutoFire, FrameStack4]
    
    # Add PongHard wrapper if needed
    if game_name == "PongHard":
        print("Applying PongHard difficulty modifications...")
        wrappers.insert(-1, PongHardWrapper)  # Insert before FrameStack4
    
    env = suite_atari.load(
        config["environment_name"],
        max_episode_steps=config["max_episode_steps"],
        gym_env_wrappers=wrappers
    )
    
    return TFPyEnvironment(env)

def scan_available_models():
    """Scan for available trained models and checkpoints."""
    available_policies = []
    
    for game_name in GAMES.keys():
        game_policies = []
        
        # Check for saved models
        models_dir = f"models/{game_name.lower()}"
        if os.path.exists(models_dir):
            try:
                contents = os.listdir(models_dir)
                for item in contents:
                    item_path = os.path.join(models_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "saved_model.pb")):
                        # Extract step information from name
                        step_match = re.search(r'step_(\d+)', item)
                        step = int(step_match.group(1)) if step_match else 999999
                        
                        game_policies.append({
                            'type': 'model',
                            'path': item_path,
                            'name': item,
                            'step': step,
                            'description': f"Saved model at step {step}"
                        })
            except Exception as e:
                print(f"Error scanning models for {game_name}: {e}")
        
        # Check for checkpoints
        checkpoint_dir = f"checkpoints/{game_name.lower()}"
        if os.path.exists(checkpoint_dir):
            try:
                files = os.listdir(checkpoint_dir)
                checkpoint_files = [f for f in files if f.startswith("ckpt-") and f.endswith(".index")]
                if checkpoint_files:
                    # Get the latest checkpoint
                    latest_ckpt = max([int(re.search(r'ckpt-(\d+)', f).group(1)) for f in checkpoint_files])
                    game_policies.append({
                        'type': 'checkpoint',
                        'path': checkpoint_dir,
                        'name': f"Latest checkpoint (step {latest_ckpt})",
                        'step': latest_ckpt,
                        'description': f"Checkpoint at step {latest_ckpt}"
                    })
            except Exception as e:
                print(f"Error scanning checkpoints for {game_name}: {e}")
        
        if game_policies:
            # Sort by step number
            game_policies.sort(key=lambda x: x['step'])
            available_policies.append((game_name, game_policies))
    
    return available_policies

def load_policy_from_saved_model(model_path):
    """Load policy from saved model directory."""
    try:
        print(f"Loading policy from saved model: {model_path}")
        policy = tf.saved_model.load(model_path)
        print("Policy loaded successfully!")
        return policy
    except Exception as e:
        print(f"Error loading policy from saved model: {e}")
        return None

def load_policy_from_checkpoint(checkpoint_dir, game_name):
    """Load policy from checkpoint directory."""
    try:
        print(f"Loading policy from checkpoint: {checkpoint_dir}")
        
        # Create environment and agent
        tf_env = create_environment(game_name)
        
        # Create Q-network
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
        
        # Create agent
        config = GAMES[game_name]
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])
        train_step = tf.Variable(0)
        
        agent = DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            target_update_period=config["target_update_period"],
            td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
            gamma=0.99,
            train_step_counter=train_step,
            epsilon_greedy=0.01  # Low epsilon for evaluation
        )
        
        agent.initialize()
        
        # Load checkpoint
        checkpoint = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            agent=agent,
            policy=agent.policy,
            global_step=train_step
        )
        
        checkpoint.initialize_or_restore()
        print("Checkpoint loaded successfully!")
        
        return agent.policy, tf_env
        
    except Exception as e:
        print(f"Error loading policy from checkpoint: {e}")
        return None, None

def record_gameplay(tf_env, policy, num_steps=2000, render_every=1):
    """Record gameplay using the loaded policy with proper Pong scoring."""
    print(f"Recording gameplay for {num_steps} steps...")
    
    recorder = GameplayRecorder()
    step_count = 0
    
    # Reset environment
    time_step = tf_env.reset()
    recorder.reset_episode()
    
    try:
        while step_count < num_steps and not time_step.is_last():
            # Get action from policy
            action_step = policy.action(time_step)
            
            # Take step in environment
            time_step = tf_env.step(action_step.action)
            
            # Record frame (every nth frame to save memory)
            if step_count % render_every == 0:
                frame = tf_env.pyenv.envs[0].render(mode="rgb_array")
                reward = time_step.reward.numpy()[0]
                recorder.record_step(frame, reward)
            
            # Check for episode end
            if time_step.is_last():
                recorder.reset_episode()
                stats = recorder.get_statistics()
                print(f"  Episode completed at step {step_count} - Agent: {stats['agent_score']} vs Opponent: {stats['opponent_score']}")
                
                # Reset for next episode if we haven't reached step limit
                if step_count < num_steps:
                    time_step = tf_env.reset()
                    recorder.reset_episode()
            
            step_count += 1
            
            # Progress update with proper scoring
            if step_count % 100 == 0:
                stats = recorder.get_statistics()
                print(f"\rStep {step_count}/{num_steps} - Episodes: {stats['episodes_completed']} - Score: Agent {stats['agent_score']} vs Opponent {stats['opponent_score']}", end="")
    
    except Exception as e:
        print(f"\nError during gameplay recording: {e}")
    
    print(f"\nRecording completed!")
    return recorder

def save_video_gif(frames, output_path, fps=30, max_frames=500):
    """Save frames as GIF video."""
    if not frames:
        print("No frames to save!")
        return False
    
    try:
        print(f"Saving {len(frames)} frames as GIF...")
        
        # Sample frames if too many
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step]
            print(f"Sampled to {len(frames)} frames for reasonable file size")
        
        # Convert to PIL Images
        frame_images = [PIL.Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        if frame_images:
            frame_images[0].save(
                output_path,
                format='GIF',
                append_images=frame_images[1:],
                save_all=True,
                duration=1000//fps,
                loop=0
            )
            print(f"Video saved as: {output_path}")
            return True
    except Exception as e:
        print(f"Error saving video: {e}")
    
    return False

def save_video_mp4(frames, output_path, fps=30):
    """Save frames as MP4 video using matplotlib animation."""
    try:
        import matplotlib.animation as animation
        
        print(f"Saving {len(frames)} frames as MP4...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create animation
        def animate(frame_idx):
            ax.clear()
            ax.imshow(frames[frame_idx])
            ax.axis('off')
            ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}")
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, repeat=False)
        
        # Save animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='TF-Agents'), bitrate=1800)
        anim.save(output_path, writer=writer)
        
        plt.close(fig)
        print(f"Video saved as: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving MP4 (trying GIF instead): {e}")
        # Fallback to GIF
        gif_path = output_path.replace('.mp4', '.gif')
        return save_video_gif(frames, gif_path)

def create_gameplay_statistics_plot(recorder, game_name, output_dir):
    """Create and save gameplay statistics plots with proper Pong scoring."""
    stats = recorder.get_statistics()
    
    if stats['episodes_completed'] == 0:
        print("No complete episodes to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{game_name} Gameplay Statistics', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(recorder.episode_rewards, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(recorder.episode_lengths, 'g-s', linewidth=2, markersize=4)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics summary with proper Pong scoring
    axes[1, 0].axis('off')
    stats_text = f"""
Gameplay Summary:
• Total Steps: {stats['total_steps']:,}
• Episodes Completed: {stats['episodes_completed']}
• Average Reward: {stats['average_reward']:.2f}
• Best Episode: {stats.get('best_episode_reward', 0):.2f}

Pong Scoring:
• Agent Score: {stats['agent_score']}
• Opponent Score: {stats['opponent_score']}
• Total Points: {stats['total_points']}
• Scoring Events: {stats['scoring_events']}
    """
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Score comparison chart
    if stats['total_points'] > 0:
        scores = [stats['agent_score'], stats['opponent_score']]
        labels = ['Trained Agent', 'Opponent AI']
        colors = ['green', 'red']
        
        axes[1, 1].pie(scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Score Distribution')
    else:
        # If no scoring occurred, show scoring events timeline
        if recorder.scoring_events:
            frames = [event['frame'] for event in recorder.scoring_events]
            scorers = [1 if event['scorer'] == 'agent' else -1 for event in recorder.scoring_events]
            
            axes[1, 1].scatter(frames, scorers, c=['green' if s > 0 else 'red' for s in scorers], s=50)
            axes[1, 1].set_title('Scoring Events Timeline')
            axes[1, 1].set_xlabel('Frame Number')
            axes[1, 1].set_ylabel('Scorer (1=Agent, -1=Opponent)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Scoring Events\nRecorded', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightyellow'))
            axes[1, 1].set_title('Score Timeline')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{game_name}_gameplay_stats.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistics plot saved as: {plot_path}")

def main():
    """Main function to load policy and record gameplay."""
    print("Deploy Game Player - Policy Demo and Video Recording")
    print("=" * 60)
    
    # Scan for available policies
    available_policies = scan_available_models()
    
    if not available_policies:
        print("No trained policies found!")
        print("\nExpected locations:")
        for game_name in GAMES.keys():
            print(f"  - models/{game_name.lower()}/")
            print(f"  - checkpoints/{game_name.lower()}/")
        print("\nPlease train a model first using buildAndTrainAgent-GAME.py")
        return
    
    # Display available policies
    print("Available Trained Policies:")
    policy_options = []
    option_num = 1
    
    for game_name, policies in available_policies:
        print(f"\n{game_name} ({GAMES[game_name]['difficulty']}):")
        for policy_info in policies:
            print(f"  {option_num}. {policy_info['description']}")
            policy_options.append((game_name, policy_info))
            option_num += 1
    
    # Get user selection
    while True:
        try:
            choice = int(input(f"\nSelect policy to demo (1-{len(policy_options)}): ").strip())
            if 1 <= choice <= len(policy_options):
                game_name, policy_info = policy_options[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(policy_options)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nSelected: {game_name} - {policy_info['description']}")
    
    # Load the selected policy
    policy = None
    tf_env = None
    
    if policy_info['type'] == 'model':
        policy = load_policy_from_saved_model(policy_info['path'])
        if policy:
            tf_env = create_environment(game_name)
    elif policy_info['type'] == 'checkpoint':
        policy, tf_env = load_policy_from_checkpoint(policy_info['path'], game_name)
    
    if policy is None or tf_env is None:
        print("Failed to load policy or create environment!")
        return
    
    # Get recording parameters
    try:
        steps = int(input("Enter number of steps to record (default 2000): ").strip() or "2000")
        fps = int(input("Enter video frame rate (default 30): ").strip() or "30")
        video_format = input("Video format (gif/mp4, default gif): ").strip().lower() or "gif"
    except ValueError:
        steps, fps, video_format = 2000, 30, "gif"
    
    # Record gameplay
    print(f"\nStarting gameplay recording...")
    print(f"Game: {game_name}")
    print(f"Policy: {policy_info['name']}")
    print(f"Steps: {steps}")
    print(f"Format: {video_format.upper()}")
    
    start_time = time.time()
    recorder = record_gameplay(tf_env, policy, num_steps=steps)
    recording_time = time.time() - start_time
    
    # Create output directory
    output_dir = "gameplay_recordings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{game_name}_demo_{timestamp}"
    
    # Save video
    if video_format == "mp4":
        video_path = os.path.join(output_dir, f"{base_filename}.mp4")
        save_video_mp4(recorder.frames, video_path, fps)
    else:
        video_path = os.path.join(output_dir, f"{base_filename}.gif")
        save_video_gif(recorder.frames, video_path, fps)
    
    # Save statistics plot
    create_gameplay_statistics_plot(recorder, game_name, output_dir)
    
    # Print final statistics with proper Pong scoring
    stats = recorder.get_statistics()
    print("\n" + "=" * 60)
    print("GAMEPLAY RECORDING COMPLETED")
    print("=" * 60)
    print(f"Game: {game_name}")
    print(f"Policy: {policy_info['name']}")
    print(f"Recording Time: {recording_time:.1f} seconds")
    print(f"Total Steps Recorded: {stats['total_steps']:,}")
    print(f"Episodes Completed: {stats['episodes_completed']}")
    
    # Detailed Pong scoring breakdown
    print(f"\nPong Scoring Breakdown:")
    print(f"  Agent Score: {stats['agent_score']}")
    print(f"  Opponent Score: {stats['opponent_score']}")
    print(f"  Total Points Played: {stats['total_points']}")
    print(f"  Scoring Events: {stats['scoring_events']}")
    
    if stats['total_points'] > 0:
        agent_win_rate = (stats['agent_score'] / stats['total_points']) * 100
        print(f"  Agent Win Rate: {agent_win_rate:.1f}%")
    
    if stats['episodes_completed'] > 0:
        print(f"\nEpisode Statistics:")
        print(f"  Average Episode Reward: {stats['average_reward']:.2f}")
        print(f"  Best Episode Reward: {stats['best_episode_reward']:.2f}")
        print(f"  Average Episode Length: {stats['average_episode_length']:.1f} steps")
    
    print(f"\nFiles Created:")
    print(f"  - Video: {video_path}")
    print(f"  - Statistics: {output_dir}/{game_name}_gameplay_stats.png")
    
    # Cleanup
    try:
        tf_env.close()
    except:
        pass

if __name__ == "__main__":
    main()
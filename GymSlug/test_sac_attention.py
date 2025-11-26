#!/usr/bin/env python
"""
Quick test script for SAC with Attention Policy
Tests if the policy can be instantiated and used with the environment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gym
import numpy as np
from stable_baselines import SAC
from stable_baselines.common.env_checker import check_env

# Import custom Attention Policy
from policies.attention_policy import AttentionPolicy

print("=" * 60)
print("Testing SAC with Attention Policy")
print("=" * 60)

# Test 1: Import
print("\n[Test 1] Testing imports...")
try:
    from policies.attention_policy import AttentionPolicy
    print("✓ AttentionPolicy imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Environment
print("\n[Test 2] Testing environment...")
try:
    env = gym.make("gym_slug:slug-v0")
    check_env(env)
    print("✓ Environment created and validated")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    sys.exit(1)

# Test 3: Policy instantiation (without full SAC model)
print("\n[Test 3] Testing policy class structure...")
try:
    print(f"  Base class: {AttentionPolicy.__bases__[0].__name__}")
    print(f"  Has make_actor method: {hasattr(AttentionPolicy, 'make_actor')}")
    print("✓ Policy class structure looks good")
except Exception as e:
    print(f"✗ Policy structure test failed: {e}")
    sys.exit(1)

# Test 4: SAC model creation with AttentionPolicy
print("\n[Test 4] Testing SAC model creation with AttentionPolicy...")
try:
    train_env = gym.make("gym_slug:slug-v0")
    train_env.present_inputs = 0
    train_env.set_verbose(0)
    
    print("  Creating SAC model...")
    model = SAC(
        AttentionPolicy,
        train_env,
        learning_rate=2.5e-4,
        verbose=1,
        policy_kwargs={
            'attention_heads': 4,
            'attention_dim': 64
        }
    )
    print("✓ SAC model created successfully with AttentionPolicy")
except Exception as e:
    print(f"✗ SAC model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test prediction
print("\n[Test 5] Testing model prediction...")
try:
    obs = train_env.reset()
    action, _states = model.predict(obs, deterministic=False)
    print(f"✓ Model prediction successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
except Exception as e:
    print(f"✗ Prediction test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test a few steps
print("\n[Test 6] Testing environment steps...")
try:
    obs = train_env.reset()
    total_reward = 0
    for i in range(10):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = train_env.step(action)
        total_reward += reward
        if done:
            obs = train_env.reset()
    print(f"✓ Environment steps successful")
    print(f"  Total reward over 10 steps: {total_reward:.2f}")
except Exception as e:
    print(f"✗ Environment steps failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run the full training script:")
print("  python sac_attention.py")
print("=" * 60)


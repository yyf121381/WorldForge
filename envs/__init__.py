"""WorldForge · Environments"""
from envs.base_env import WorldState, Action, StepResult, Environment
from envs.emotion_depletion import EmotionDepletionEnv
from envs.meaning_crisis import MeaningCrisisEnv

ENV_REGISTRY = {
    "emotion_depletion": EmotionDepletionEnv,
    "meaning_crisis": MeaningCrisisEnv,
}

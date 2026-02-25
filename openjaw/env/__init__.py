"""OpenJaw environment package."""

from gymnasium.envs.registration import register

register(
    id="OpenJaw-Mouth-v0",
    entry_point="openjaw.env.mouth_env:MouthEnv",
)

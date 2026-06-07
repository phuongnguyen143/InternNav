from typing import Any, Dict

from internnav.configs.agent import AgentCfg
from internnav.model.utils.device import resolve_torch_device


class Agent:
    agents = {}

    def __init__(self, config: AgentCfg):
        self.config = config

    def step(self, obs: Dict[str, Any]):
        raise NotImplementedError("This function is not implemented yet.")

    def reset(self):
        raise NotImplementedError("This function is not implemented yet.")

    @classmethod
    def register(cls, agent_type: str):
        """
        Register a agent class.
        """

        def decorator(agent_class):
            if agent_type in cls.agents:
                raise ValueError(f"Agent {agent_type} already registered.")
            cls.agents[agent_type] = agent_class
            return agent_class

        return decorator

    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        if config.model_settings is not None:
            resolved = resolve_torch_device(config.model_settings.get('device'))
            config.model_settings['device'] = str(resolved)
        return cls.agents[config.model_name](config)

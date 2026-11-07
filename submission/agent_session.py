"""One serial inference episode with idempotent reset and last-step replay."""
import hashlib
import json
from threading import RLock

from submission.remote_protocol import PROTOCOL, checked_action, valid_envelope

MAX_RETIRED_EPISODES = 4096


class SessionConflict(ValueError):
    pass


class AgentSession:
    def __init__(self, policy):
        self.policy = policy
        self.lock = RLock()
        self.episode = None
        self.failed = False
        self.next_step = 0
        self.last_response = None
        self.retired = set()

    @staticmethod
    def digest(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).digest()

    def reset(self, payload):
        episode = valid_envelope(payload)
        digest = self.digest(payload)
        with self.lock:
            if episode == self.episode:
                if digest != self.reset_digest:
                    raise SessionConflict("Conflicting reset for the same episode")
                return self.reset_response
            if episode in self.retired:
                raise SessionConflict("Episode has retired; old reset requests cannot reactivate it")
            if self.episode is not None and len(self.retired) >= MAX_RETIRED_EPISODES:
                raise SessionConflict("Episode history is full; restart A service before a new evaluation")
            self.failed = True
            self.policy.reset(episode_id=episode, options=payload.get("options"))
            if self.episode is not None:
                self.retired.add(self.episode)
            self.episode, self.reset_digest = episode, digest
            self.failed, self.next_step, self.last_response = False, 0, None
            self.reset_response = dict(ok=True, protocol=PROTOCOL, episode_id=episode,
                                       model_id=self.policy.status().get("model_id"))
            return self.reset_response

    def act(self, payload, coerce):
        episode = valid_envelope(payload)
        index = payload.get("step_id")
        if type(index) is not int or index < 0 or not isinstance(payload.get("observation"), dict):
            raise ValueError("act requires a nonnegative integer step_id and an observation object")
        digest = self.digest(payload)
        with self.lock:
            if episode != self.episode or self.failed:
                raise SessionConflict("Episode is inactive or failed; start a new episode")
            if self.last_response is not None and index == self.next_step - 1:
                if digest != self.last_digest:
                    raise SessionConflict("Conflicting payload for an already completed step")
                return self.last_response
            if index != self.next_step:
                raise SessionConflict("Out-of-order step_id")
            try:
                action = checked_action(self.policy.act(coerce(payload["observation"])))
            except Exception:
                self.failed = True
                raise
            result = dict(ok=True, protocol=PROTOCOL, episode_id=episode, step_id=index,
                          action=action, model_id=self.policy.status().get("model_id"))
            self.next_step += 1
            self.last_digest, self.last_response = digest, result
            return result

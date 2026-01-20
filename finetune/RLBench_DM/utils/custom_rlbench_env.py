# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/custom_rlbench_env.py
from typing import List
from bridgevla.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from yarr.agents.agent import VideoSummary, TextSummary
from yarr.utils.process_str import change_case
import numpy as np


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict


class CustomMultiTaskRLBenchEnvActionChunk(CustomMultiTaskRLBenchEnv2):
    """
    Environment that supports multi-step actions.
    The step function accepts a list of ActResult objects and executes them sequentially.
    """
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnvActionChunk, self).__init__(*args, **kwargs)

    def step(self, act_result_list: List[ActResult]) -> Transition:
        """
        Execute multiple actions in sequence.

        Args:
            act_result_list: List of ActResult objects to execute sequentially

        Returns:
            Transition object with the final observation, cumulative reward,
            terminal status, and summaries
        """
        if not isinstance(act_result_list, list):
            # If single ActResult is passed, convert to list for compatibility
            act_result_list = [act_result_list]

        success = False
        obs = self._previous_obs_dict  # in case action fails
        cumulative_reward = 0.0
        terminal = False
        summaries = []

        # Execute each action in the list sequentially
        for step_idx, act_result in enumerate(act_result_list):
            action = act_result.action

            try:
                obs, reward, terminal = self._task.step(action)
                if reward >= 1:
                    success = True
                    reward *= self._reward_scale
                else:
                    reward = 0.0

                cumulative_reward += reward
                obs = self.extract_obs(obs)
                self._previous_obs_dict = obs

                # If terminal or success, stop executing remaining actions
                if terminal:
                    break

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                terminal = True
                # Don't add reward for failed action

                if isinstance(e, IKError):
                    self._error_type_counts['IKError'] += 1
                elif isinstance(e, ConfigurationPathError):
                    self._error_type_counts['ConfigurationPathError'] += 1
                elif isinstance(e, InvalidActionError):
                    self._error_type_counts['InvalidActionError'] += 1

                self._last_exception = e
                break  # Stop executing remaining actions on error

        self._i += 1

        # Record video and generate summaries
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}',
                vid, fps=30))

            # Error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))

        return Transition(obs, cumulative_reward, terminal, summaries=summaries)

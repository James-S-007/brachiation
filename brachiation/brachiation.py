import math
import os
import pickle
import types

import gym
import numpy as np
import pybullet

from bottleneck import ss, nansum
from scipy.linalg.blas import sscal as SCAL
import torch

from bullet.objects import VCylinder, VSphere
from brachiation import EnvBase
from brachiation.agents import Gibbon2D


DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)


class Gibbon2DCustomEnv(EnvBase):
    control_step = 1 / 60
    llc_frame_skip = 8
    sim_frame_skip = 1
    max_timestep = 1000

    robot_class = Gibbon2D
    robot_random_start = False
    robot_init_position = [0, 0, 0.6]
    robot_init_velocity = [0, 0, 0]

    num_steps = 30
    dist_range = np.array([0.5, 3])
    pitch_range = np.array([-15 * DEG2RAD, 15 * DEG2RAD])

    grab_threshold = 0.05
    min_grab_duration = 15
    max_grab_duration = 240

    lookahead = 1

    def __init__(self, ref_traj=False, traj_num=None, noise_body_sd=0.0, noise_handholds_sd=0.0, noise_reftraj_sd=0.0, is_eval=False, **kwargs):
        if is_eval:
            camera_params = {
                'img_width': 640,
                'img_height': 360,
                'camera_dist': 4.5
            }
        else:
            camera_params = {
                'img_width': 80,
                'img_height': 80,
                'camera_dist': 1.5
            }

        super().__init__(self.robot_class, remove_ground=True, **camera_params, **kwargs)
        self.robot.set_base_pose(pose="hanging")
        self.ref_traj = ref_traj
        self.noise_body_sd = noise_body_sd
        self.noise_handholds_sd = noise_handholds_sd
        self.noise_reftraj_sd = noise_reftraj_sd
        self.is_eval = is_eval

        basepath = os.path.join(parent_dir, "data", "objects", "misc")
        filename = os.path.join(basepath, "plane_stadium.sdf")
        id_ = self._p.loadSDF(filename, useMaximalCoordinates=True)[0]
        self._p.resetBasePositionAndOrientation(id_, (0, 0.2, 0), (1, 0, 0, 1))
        if self.noisy_img:
            id_ = self._p_dummy.loadSDF(filename, useMaximalCoordinates=True)[0]
            self._p_dummy.resetBasePositionAndOrientation(id_, (0, 0.2, 0), (1, 0, 0, 1))

        # Fix-ordered Curriculum
        self.curriculum = 9
        self.max_curriculum = 9
        self.advance_threshold = 12  # steps_reached

        RO = self.robot.observation_space.shape[0]
        K = (self.lookahead + 1) * 3 + 6 * 3
        high = np.inf * np.ones(RO + K, dtype=np.float32)
        obs_space = {}
        obs_space['state'] = gym.spaces.Box(-high, high, dtype="f4")
        obs_space['noisy_state'] = gym.spaces.Box(-high, high, dtype="f4")
        obs_space['handholds_grabbed'] = gym.spaces.Box(0, 255, shape=(), dtype="f4")
        if self.img_obs:
            obs_space['log_img'] = gym.spaces.Box(0, 255, shape=(self.camera.height, self.camera.width, 3), dtype="uint8")
            if self.noisy_img:
                obs_space['log_img_noisy'] = gym.spaces.Box(0, 255, shape=(self.camera.height, self.camera.width, 3), dtype="uint8")
        self.observation_space = gym.spaces.Dict(obs_space)

        RA = self.robot.action_space.shape[0]
        # Two more action variables for grab
        high = np.ones(RA + 2, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.hand_ids = self.robot.foot_ids
        self.grab_duration = np.zeros(len(self.hand_ids), dtype=np.float32)
        self.grab_constraint_ids = -1 * np.ones(len(self.hand_ids), dtype=np.int32)

        # Set Kp and Kd for PD control
        joint_ranges = -np.subtract(*self.robot.joint_limits.T)
        self._a = 0.5 * np.fromiter(joint_ranges, dtype=np.float32)
        # Shoulder joints are not limited, so hardcode a kp
        self._a[[-6, -3]] = np.pi / 2

        if self.is_rendered or self.img_obs:
            self.target_marker = VSphere(self._p, 0.05, rgba=(0, 0, 1, 1))
            self.free_hand_marker = VSphere(self._p, 0.05, rgba=(0, 0, 1, 1))
            self.grab_hand_marker = VSphere(self._p, 0.06, rgba=(0, 1, 0, 1))
            self.handhold_markers = [
                VSphere(self._p, 0.05) for _ in range(self.num_steps)
            ]
            if self.noisy_img:
                self.dummy_handhold_markers = [
                    VSphere(self._p_dummy, 0.05) for _ in range(self.num_steps)
                ]

        self.ref_xyz = np.zeros((self.max_timestep * 2, 3), dtype=np.float32)
        self.ref_swing = np.zeros(len(self.ref_xyz), dtype=np.float32)

        self.load_traj_data_file()

        self.traj_num = traj_num

    def load_traj_data_file(self):
        filename = f"simple_trajs_{self.curriculum}.pickle"
        datafile = os.path.join(parent_dir, "data", "trajectories", filename)
        with open(datafile, "rb") as f:
            self.traj_data = pickle.load(f)
            self.traj_data_curriculum = self.curriculum

        for i, (xz, grab, hh) in enumerate(self.traj_data):
            x, z = xz.T
            y = np.zeros_like(x)
            ref_xyz = np.stack((x, y, z), axis=-1)

            x, z = hh.T
            y = -0.16 * 2 * (np.arange(len(hh)) % 2 - 0.5)
            handholds = np.stack((x, y, z), axis=-1)

            self.traj_data[i] = (ref_xyz, grab.squeeze(), handholds)

    def get_observation_components(self):
        k = self.next_step_index
        targets = self.handholds[k - 1 : k + self.lookahead]
        noise_handholds = np.random.normal(0.0, self.noise_handholds_sd, self.handholds.shape).astype(self.handholds.dtype)
        targets_noise = noise_handholds[k-1:k+self.lookahead]
        if self.noisy_img:
            self.set_dummy_handholds(self.dummy_handhold_markers, self.dummy_handholds + noise_handholds)
        target_delta = targets - self.robot.body_xyz
        noisy_target_delta = (targets + targets_noise) - self.robot.body_xyz  # with noise

        window = slice(self.ref_timestep + 1, self.ref_timestep + 30, 5)
        noise_reftraj = np.random.normal(0.0, self.noise_reftraj_sd, self.ref_xyz[window].shape)
        ref_delta = self.ref_xyz[window] - self.robot.body_xyz
        noisy_ref_delta = (self.ref_xyz[window] + noise_reftraj) - self.robot.body_xyz  # with noise

        pitch = self.robot.body_rpy[1]
        cos_ = math.cos(-pitch)
        sin_ = math.sin(-pitch)

        # FIXME: This is incorrect; relies network to correct the error
        target_delta[:, 0] = target_delta[:, 0] * cos_ - target_delta[:, 2] * sin_
        target_delta[:, 2] = target_delta[:, 0] * sin_ + target_delta[:, 2] * cos_

        ref_delta[:, 0] = ref_delta[:, 0] * cos_ - ref_delta[:, 2] * sin_
        ref_delta[:, 2] = ref_delta[:, 0] * sin_ + ref_delta[:, 2] * cos_

        obs = {}
        if self.ref_traj:
            obs['state'] = self.robot_state, target_delta.flatten(), ref_delta.flatten()
            obs['noisy_state'] = self.noisy_robot_state, noisy_target_delta.flatten(), noisy_ref_delta.flatten()
        else:
            obs['state'] = self.robot_state, target_delta.flatten(), np.zeros_like(ref_delta.ravel())
            obs['noisy_state'] = self.noisy_robot_state, noisy_target_delta.flatten(), np.zeros_like(noisy_ref_delta.flatten())
        obs['state'] = np.concatenate(obs['state']).astype('float32')
        obs['noisy_state'] = np.concatenate(obs['noisy_state']).astype('float32')
        obs['handholds_grabbed'] = self.next_step_index

        if self.img_obs:
            # update camera pos and get img
            self.camera.wait()

            if not self.is_eval:
                camera_xyz = (  # track monkey
                    *self.robot.body_xyz[0:2],
                    self.handholds[self.next_step_index][2],
                )
            else:
                camera_xyz = self.handholds[0] + (self.handholds[5] - self.handholds[0]) / 2  # constant distant view
            
            self.camera.track(camera_xyz)
            # get x,y positions of first ~8 handholds and set camera to there, increase view distance
            obs['log_img'] = self.camera.dump_rgb_array()
            if self.noisy_img:
                self.camera_noisy.wait()
                self.camera_noisy.track(camera_xyz)
                obs['log_img_noisy'] = self.camera_noisy.dump_rgb_array()
        return obs

    def generate_handholds(self):

        # Check just in case
        self.curriculum = min(self.curriculum, self.max_curriculum)
        ratio = self.curriculum / self.max_curriculum

        dist_mean = self.dist_range.mean()
        dist_range = dist_mean + (self.dist_range - dist_mean) * ratio
        pitch_range = self.pitch_range * ratio + np.pi / 2

        N = self.num_steps
        dr = self.np_random.uniform(*dist_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)

        dr[0] = 0
        dtheta[0] = 0

        dx = dr * np.sin(dtheta)
        dz = dr * np.cos(dtheta)

        # 0.16 is Gibbon2D's shoulder separation
        zigzag = -0.16 * 2 * (np.arange(N) % 2 - 0.5)

        x = np.cumsum(dx) + self.robot.feet_xyz[1, 0]
        y = np.zeros_like(dx) + self.robot.body_xyz[1] + zigzag
        z = np.cumsum(dz) + self.robot.feet_xyz[1, 2]

        xyz = np.stack((x, y, z), axis=1).astype(np.float32)

        if self.is_rendered or self.img_obs:
            for h, pos in zip(self.handhold_markers, xyz):
                h.set_position(pos)

        # Handle out-of-bounds. append first and repeat last
        return np.concatenate(
            (
                xyz,
                np.repeat(xyz[[-1]], self.lookahead, axis=0),
            )
        )

    def reset(self):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)
        self.timestep = 0
        self.ref_timestep = 0
        self.done = False

        self.robot_state, self.noisy_robot_state, noisy_robot_state_raw = self.robot.reset(
            random_pose=self.robot_random_start,
            random_mirror=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
            noise_body_sd=self.noise_body_sd
        )

        # self.set_dummy_robot_state(**noisy_robot_state_raw)
        if self.noisy_img:
            self.dummy_robot.reset(
                random_pose=False,
                random_mirror=False,
                pos=noisy_robot_state_raw['pos'],
                quat=noisy_robot_state_raw['quat'],
                vel=None,
                ang_vel=None,
                noise_body_sd=0.0
            )
            self.dummy_robot.reset_joint_states(noisy_robot_state_raw['joint_angles'], noisy_robot_state_raw['joint_speeds'])

        if len(self.traj_num) < 1:
            traj_id = self.np_random.randint(len(self.traj_data))
        else:
            traj_id = self.traj_num[self.np_random.randint(len(self.traj_num))]

        self.current_traj_id = traj_id
        ref_xyz, ref_swing, handholds = self.traj_data[traj_id]

        self.ref_xyz, self.ref_swing, self.handholds = (
            ref_xyz.copy(),
            ref_swing.copy(),
            handholds.copy(),
        )

        x0, _, z0 = self.robot.feet_xyz[1]
        self.ref_xyz[:, 0] += x0
        self.ref_xyz[:, 2] += z0
        self.handholds[:, 0] += x0
        self.handholds[:, 2] += z0

        # Uncomment if not using reference trajectory
        if not self.ref_traj:
            self.handholds = self.generate_handholds()

        if self.is_rendered or self.img_obs:
            for h, pos in zip(self.handhold_markers, self.handholds):
                h.set_position(pos)

        self.dummy_handholds = self.handholds.copy()

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        for cid in self.grab_constraint_ids:
            if cid > -1:
                self._p.removeConstraint(cid)
        self.grab_duration.fill(0)
        self.grab_constraint_ids.fill(-1)

        hand = 1  # left hand
        hand_id = self.hand_ids[hand]
        hand_xyz = self.robot.feet_xyz[hand]
        id = pybullet.createConstraint(
            self.robot.id,
            hand_id,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_POINT2POINT,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=hand_xyz,
            physicsClientId=self._p._client,
        )
        self.grab_constraint_ids[hand] = id
        self.robot.feet_contact[hand] = 1
        self.robot_state[hand - 2] = 1
        self.noisy_robot_state[hand - 2] = 1

        if self.is_rendered or self.img_obs:
            xyz = self.robot.feet_xyz[hand]
            self.grab_hand_marker.set_position(xyz)

        # Randomize platforms
        self.next_step_index = 1  # start at 1, 0 is already grabbed
        self.prev_next_step_index = 1
        self.prev_grab_cids = self.grab_constraint_ids.copy()

        return self.get_observation_components()

    def step(self, action):
        self.timestep += 1
        self.ref_timestep += 1

        grab_action = action[-2:]  # last two are grab flags
        action = action[:-2]  # others are normal joint actions

        self.grab_arm_id = self.next_step_index % 2

        # Swing arm can be calculated in this simple environment
        self.free_arm_id = (self.next_step_index + 1) % 2
        free_arm_xyz = self.robot.feet_xyz[self.free_arm_id]

        target_pose = action * self._a + self.robot.joint_angles

        for _ in range(self.llc_frame_skip):
            delta = target_pose - self.robot.joint_angles - self.robot.joint_speeds
            delta[delta > +1] = +1
            delta[delta < -1] = -1

            self.robot.apply_action(delta)
            self.scene.global_step()

            # update joint angles and velocities
            pybullet.getJointStates2(
                self.robot.id,
                self.robot.joint_ids,
                self.robot.joint_angles,
                self.robot.joint_speeds,
                physicsClientId=self._p._client,
            )
            SCAL(0.1, self.robot.joint_speeds)

        # Order matters here, calc_state -> grab_action -> set contact
        self.robot_state, self.noisy_robot_state, noisy_robot_state_raw = self.robot.calc_state(noise_body_sd=self.noise_body_sd)
        if self.noisy_img:
            self.dummy_robot.reset(
                random_pose=False,
                random_mirror=False,
                pos=noisy_robot_state_raw['pos'],
                quat=noisy_robot_state_raw['quat'],
                vel=None,
                ang_vel=None,
                noise_body_sd=0.0
            )
            self.dummy_robot.reset_joint_states(noisy_robot_state_raw['joint_angles'], noisy_robot_state_raw['joint_speeds'])

        self.calc_hand_state()
        self.apply_grab_action(grab_action)

        # Contact based on whether hand is grabbing
        is_grabbing = self.grab_constraint_ids >= 0
        self.robot.feet_contact[:] = is_grabbing.astype(np.float32)
        self.robot_state[-2:] = self.robot.feet_contact
        self.noisy_robot_state[-2:] = self.robot.feet_contact

        just_grabbed = is_grabbing * (self.prev_grab_cids == -1)
        # just_released = ~is_grabbing * (self.prev_grab_cids > -1)
        self.prev_grab_cids = self.grab_constraint_ids.copy()

        target = self.ref_xyz[self.ref_timestep]
        hh_xyz = self.handholds[self.next_step_index]

        # In 2D, pitch is better calculated if y-first
        # pitch = R.from_quat(self.robot.body_quat).as_euler("yxz")[0]
        pitch = self.robot.body_rpy[1]

        in_flight = ~is_grabbing[self.grab_arm_id]
        tracking_term = -4.0 * ss(self.robot.body_xyz - target)
        reaching_term = -0.1 * ss(free_arm_xyz - hh_xyz) if in_flight else 0

        posture_term = 0.7 - abs(pitch) if not -0.7 < pitch < 0.7 else 0
        arm_speed_term = -0.1 * abs(self.robot.joint_speeds[[-6, -3]][self.free_arm_id])
        knee_term = -0.1 * abs(self.robot.joint_angles[[2, 5]] - 110 * DEG2RAD).sum()
        action_term = -0.01 * (ss(delta) + nansum(abs(self.robot.joint_speeds)))

        step_reward = 10 if self.next_step_index > self.prev_next_step_index else 0
        self.prev_next_step_index = self.next_step_index

        aux_reward = tracking_term + reaching_term
        task_reward = posture_term + action_term + knee_term + arm_speed_term

        ratio = self.curriculum / self.max_curriculum
        reward = 2.718 ** (aux_reward + task_reward) + step_reward

        L = len(self.ref_xyz)
        max_time_reached = (self.timestep >= L - 1) or (self.ref_timestep >= L - 30)
        max_step_reached = self.next_step_index >= self.num_steps
        unrecoverable = (
            (not is_grabbing.any())
            and self.robot.body_vel[2] < 0
            and self.robot.body_xyz[2] - hh_xyz[2] < -0.6
        )
        done = self.done or max_time_reached or max_step_reached or unrecoverable

        if self.is_rendered or self.img_obs:
            self.target_marker.set_position(target)
            self.free_hand_marker.set_position(free_arm_xyz)

        info = {"just_grabbed": just_grabbed.any()}
        if done or self.timestep == self.max_timestep - 1:
            info["curriculum_metric"] = self.next_step_index

        state = self.get_observation_components()
        return state, reward, done, info

    def apply_grab_action(self, grab_action):

        client_id = self._p._client

        not_grabbing = self.grab_constraint_ids[self.free_arm_id] < 0
        is_close = self.hand_dist_to_target < self.grab_threshold
        can_grab = not_grabbing * is_close
        set_grab = can_grab * (grab_action[self.free_arm_id] > 0)

        if set_grab:
            xyz = self.robot.feet_xyz[self.free_arm_id]
            id = pybullet.createConstraint(
                self.robot.id,
                self.hand_ids[self.free_arm_id],
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_POINT2POINT,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=xyz,
                physicsClientId=client_id,
            )
            self.grab_constraint_ids[self.free_arm_id] = id

            if self.is_rendered or self.img_obs:
                self.grab_hand_marker.set_position(xyz)

        cur_grab_duration = self.grab_duration[self.grab_arm_id]
        is_grabbing = self.grab_constraint_ids[self.grab_arm_id] >= 0
        max_duration_reached = cur_grab_duration >= self.max_grab_duration
        min_duration_reached = cur_grab_duration >= self.min_grab_duration
        action_release = grab_action[self.grab_arm_id] < 0
        release_grab = is_grabbing * action_release + max_duration_reached
        release_grab = min_duration_reached * release_grab

        if release_grab:
            pybullet.removeConstraint(
                self.grab_constraint_ids[self.grab_arm_id],
                physicsClientId=client_id,
            )
            self.grab_constraint_ids[self.grab_arm_id] = -1

            if self.is_rendered or self.img_obs:
                self.grab_hand_marker.set_position((0, -1000, 0))

        # Keep track number of consecutive frames each hand has grabbed
        still_grabbing = self.grab_constraint_ids >= 0
        self.grab_duration = still_grabbing * (self.grab_duration + 1)

    def calc_hand_state(self):

        free_arm_xyz = self.robot.feet_xyz[self.free_arm_id]
        target_xyz = self.handholds[self.next_step_index, 0:3]
        self.hand_dist_to_target = ss(free_arm_xyz - target_xyz) ** 0.5

        # Target reached if any hand is less than 5cm away
        self.target_reached = self.hand_dist_to_target < self.grab_threshold

        # At least one foot is on the plank
        if self.target_reached:
            self.next_step_index += 1

            # Prevent out of bound
            if self.next_step_index > self.num_steps + 1:
                self.next_step_index -= 1

    def get_mirror_indices(self):

        # delta_y needs to be negated
        robot = self.robot
        neg_target_indices = np.array([i * 3 + 1 for i in range(self.lookahead + 1)])

        negation_obs_indices = np.concatenate(
            (
                robot.neg_obs_indices,
                neg_target_indices + robot.observation_space.shape[0],
            )
        )

        A = self.action_space.shape[0]
        right_action_indices = np.concatenate((robot._right_joint_indices, [A - 2]))
        left_action_indices = np.concatenate((robot._left_joint_indices, [A - 1]))

        return (
            negation_obs_indices,
            robot.right_obs_indices,
            robot.left_obs_indices,
            robot._negation_joint_indices,
            right_action_indices,
            left_action_indices,
        )


class Gibbon2DPointMassEnv(gym.Env):
    control_step = 1 / 60
    sim_frame_skip = 8
    sim_dt = control_step / sim_frame_skip
    max_timesteps = 1000

    kp = 9.8 * 20
    kd = kp * 0.3
    grab_threshold = 0.75
    min_arm_length = 0.1
    max_arm_length = 0.75
    init_arm_length = max_arm_length * 0.98

    early_termination = True

    num_steps = 30

    def __init__(self, noise_stdev=0.0, img_obs=True, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.is_rendered = kwargs.get("render", False)
        self.num_parallel = kwargs.get("num_parallel", 4)

        self.lookahead = kwargs.get("lookahead", 1)
        self.min_grab_duration = kwargs.get("min_grab_duration", 15)
        self.max_grab_duration = kwargs.get("max_grab_duration", 240)

        # Fix-ordered Curriculum
        self.max_curriculum = 1
        self.curriculum = kwargs.get("curriculum", self.max_curriculum)
        self.advance_threshold = 15
        self.noise_stdev = noise_stdev
        self.img_obs = img_obs

        print(f"{self.lookahead=}")
        print(f"{self.curriculum=}")
        print(f"{self.min_grab_duration=}")
        print(f"{self.max_grab_duration=}")
        print(f"{self.noise_stdev=}")
        print(f"{self.img_obs=}")

        P = self.num_parallel
        N = self.num_steps
        L = self.lookahead
        D = self.device

        self.start_state_bounds = (195 * DEG2RAD, 195 * DEG2RAD)
        self.dist_range = torch.tensor([1, 2], device=D)
        self.pitch_range = torch.tensor([-15, 15], device=D) * DEG2RAD

        self.body_positions = torch.zeros((P, 2), device=D)
        self.body_velocities = torch.zeros((P, 2), device=D)

        if self.is_rendered or self.img_obs:
            self.enable_rendering()

        # grab + target length
        high = np.ones(2, dtype="f4")
        self.action_space = gym.spaces.Box(-high, high, dtype="f4")

        # body_vel + grab_status + handhold positions (x3)
        high = np.inf * np.ones(3 + (L + 1) * 2, dtype="f4")
        obs_space = {}
        obs_space['state'] = gym.spaces.Box(-high, high, dtype="f4")
        if self.img_obs:
            obs_space['log_img'] = gym.spaces.Box(0, 255, shape=(self.camera.height, self.camera.width, 3), dtype="uint8")
        self.observation_space = gym.spaces.Dict(obs_space)

        self._dr = torch.zeros((P, N + L), device=D)
        self._dtheta = torch.zeros((P, N + L), device=D)
        self._init_angles = torch.zeros(P, device=D)
        self._all = torch.arange(P, device=D).long()
        self._offsets = torch.arange(self.lookahead + 1, device=D).add(-1)[:, None]
        self.handholds = torch.zeros((P, N + L, 2), device=D)

        # Lots of bookkeeping stuff
        shape = (P, 1)
        self.done_flags = torch.zeros(shape, device=D).bool()
        self.timesteps = torch.zeros(shape, device=D).long()
        self.next_step_indices = torch.zeros(shape, device=D).long()
        self.grab_flags = torch.zeros(shape, device=D).bool()
        self.grab_durations = torch.zeros(shape, device=D).long()
        self.episode_rewards = torch.zeros(shape, device=D)
            
    def enable_rendering(self):
        """
            Creates Camera object and sets up pybullet for rendering
            If img_obs is enabled, GUI will be disabled
        """
        from bullet.utils import BulletClient, Camera, OffscreenCamera, StadiumScene
        bc_mode = pybullet.GUI if not self.img_obs else pybullet.DIRECT
        self._p = BulletClient(bc_mode, fps=1 / self.control_step)

        bc = self._p
        bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 0)
        if not self.img_obs:
            bc.configureDebugVisualizer(bc.COV_ENABLE_GUI, 0)
            bc.configureDebugVisualizer(bc.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            bc.configureDebugVisualizer(bc.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            bc.configureDebugVisualizer(bc.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            bc.configureDebugVisualizer(bc.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.scene = StadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.sim_dt,
            frame_skip=1,
        )
        self.scene.initialize(remove_ground=True)
        self.camera = Camera(self._p, fps=1 / self.control_step) if not self.img_obs else OffscreenCamera(self._p, fps=1 / self.control_step)
        self.handhold_markers = [VSphere(bc, 0.05) for _ in range(self.num_steps)]
        colour = (0.6, 0.63, 0.66, 1)
        self.body_marker = VSphere(bc, 0.06, rgba=colour)
        self.target_length_marker = VCylinder(bc, 0.01, 0.2, rgba=colour)
        self.target_handhold_marker = VSphere(bc, 0.055, rgba=(0, 0, 1, 1))

        # For rendering only
        self._last_body_position = (0, 0, 0)

        if not self.img_obs:
            bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 1)

            self._handle_keyboard = types.MethodType(EnvBase._handle_keyboard, self)

    def generate_handholds(self):
        self._dr.uniform_(*self.dist_range)
        self._dtheta.uniform_(*self.pitch_range)

        self._dr[:, 0] = 0
        self._dtheta[:, 0] = 0

        x = (self._dr * self._dtheta.cos()).cumsum(dim=-1)
        z = (self._dr * self._dtheta.sin()).cumsum(dim=-1)
        handholds = torch.stack((x, z), dim=-1)

        return handholds

    def get_observation_components(self):
        indices = self.next_step_indices.squeeze(-1)
        target_indices = indices.repeat(self.lookahead + 1, 1) + self._offsets

        next_handholds = self.handholds[self._all, target_indices].transpose(0, 1)
        noise = torch.from_numpy(np.random.normal(0.0, self.noise_stdev, next_handholds.shape))
        deltas = (next_handholds + noise) - self.body_positions[:, None]

        normalized_durations = self.grab_durations / self.max_grab_duration

        obs = {}
        state = self.body_velocities, normalized_durations, deltas.flatten(1, 2)
        obs['state'] = torch.cat(state, dim=-1)[0].numpy()
        if self.img_obs:
            # update camera pos and get img
            camera_xyz = (
                *self.body_positions[0],
                0
            )
            self.camera.track(camera_xyz)
            obs['log_img'] = self.camera.dump_rgb_array()
        return obs

    def reset(self, indices=None):
        # Used in render mode, trigger by keyboard
        self.done = False

        indices = self._all if indices is None else indices

        self.timesteps[indices] = 0
        self.done_flags[indices] = False
        self.grab_flags[indices] = True
        self.next_step_indices[indices] = 1
        self.grab_durations[indices] = 0
        self.episode_rewards[indices] = 0

        L = self.init_arm_length

        bounds = self.start_state_bounds
        angles = self._init_angles[indices].uniform_(*bounds)
        self._init_angles[indices] = angles  # Don't really need

        self.body_positions[indices, 0] = L * angles.cos()
        self.body_positions[indices, 1] = L * angles.sin()
        self.body_velocities[indices] = 0

        new_handholds = self.generate_handholds()
        self.handholds[indices] = new_handholds[indices]

        if self.is_rendered or self.img_obs:
            self.camera.lookat((0, 0, 0))
            for h, (x, z) in zip(self.handhold_markers, self.handholds[0]):
                h.set_position((x, 0, z))

        return self.get_observation_components()

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions.clamp_(-1, 1)

        ratio = self.curriculum / self.max_curriculum

        # grab + target length
        # grab_actions = actions[:, [0]]
        # target_length_offsets = actions[:, [1]] * self.max_arm_length
        grab_actions = actions[0]
        target_length_offsets = actions[1] * self.max_arm_length

        targets = self.handholds[self._all, self.next_step_indices.squeeze(-1)]
        vec_to_targets = self.body_positions - targets
        dist_to_targets = vec_to_targets.norm(dim=-1, keepdim=True)

        # Set grab action
        max_duration_reached = self.grab_durations >= self.max_grab_duration
        max_duration_reached = (self.max_grab_duration > 0) * max_duration_reached
        min_duration_reached = self.grab_durations >= self.min_grab_duration
        release_grab = (grab_actions < 0) + max_duration_reached
        release_grab = min_duration_reached * release_grab

        is_close = dist_to_targets < self.grab_threshold
        set_grab = ~self.grab_flags * is_close * (grab_actions > 0)

        # set_grab takes precedence if both are true
        self.grab_durations[set_grab] = 0  # reset on new grab
        self.grab_flags.mul_(~release_grab)
        self.grab_flags.add_(set_grab)
        self.grab_durations = self.grab_flags * (self.grab_durations + 1)

        # Target might have changed
        self.next_step_indices = set_grab + self.next_step_indices
        current_targets = self.handholds[self._all, self.next_step_indices.squeeze(-1)]

        # Handle swing and flight
        gsi = self.next_step_indices.squeeze(-1) - 1
        current_handholds = self.handholds[self._all, gsi]

        # Calculate "residual" target length, residual part is in loop
        arm_vectors = current_handholds - self.body_positions
        arm_lengths = arm_vectors.norm(dim=-1, keepdim=True)
        target_lengths = arm_lengths + target_length_offsets
        target_lengths.clamp_(self.min_arm_length, self.max_arm_length)

        for _ in range(self.sim_frame_skip):
            arm_vectors = current_handholds - self.body_positions
            arm_lengths = arm_vectors.norm(dim=-1, keepdim=True)
            arm_directions = arm_vectors / arm_lengths

            errors = arm_lengths - target_lengths
            vns = (self.body_velocities * arm_directions).sum(dim=-1, keepdim=True)

            length_violations = arm_lengths > self.max_arm_length
            apply_corrections = self.grab_flags * length_violations
            # Reset position and kill normal component of velocity
            p_corrections = (arm_lengths - self.max_arm_length) * arm_directions
            self.body_positions.add_(apply_corrections * p_corrections)
            v_corrections = (vns < 0) * -vns * arm_directions
            self.body_velocities.add_(apply_corrections * v_corrections)

            # Spring accelerations only happens when grabbing
            errors.clamp_(max=self.max_arm_length)
            magnitudes = self.kp * errors - self.kd * vns
            accelerations = self.grab_flags * magnitudes * arm_directions
            accelerations.clamp_(-20, 20)  # Based on 300N for normal gibbons
            accelerations[:, 1].add_(-9.8)

            a_projected = (accelerations * arm_directions).sum(dim=-1, keepdim=True)
            a_corrections = (a_projected < 0) * -a_projected * arm_directions
            accelerations.add_(apply_corrections * a_corrections)

            self.body_positions.add_(self.body_velocities * self.sim_dt)
            self.body_velocities.add_(accelerations * self.sim_dt)

        self.timesteps.add_(1)

        n = 2.1 * ratio + 3.0 * (1 - ratio)
        speeds = self.body_velocities.norm(dim=-1, keepdim=True)
        # rewards = set_grab.mul(40) + (n - speeds).clamp(max=0).pow(2).mul(-ratio)
        rewards = set_grab
        self.episode_rewards.add_(rewards)

        vz, z0, z1 = (
            self.body_velocities[:, [1]],
            current_targets[:, [1]],
            self.body_positions[:, [1]],
        )

        max_step_reached = self.next_step_indices >= self.num_steps
        unrecoverable = ~self.grab_flags * (vz < 0) * (z1 - z0 < -self.max_arm_length)
        time_limit_reached = self.timesteps >= self.max_timesteps
        dones = (
            self.done
            + max_step_reached
            + self.early_termination * unrecoverable
            + (not self.early_termination) * time_limit_reached
        )
        dones = dones.bool()

        info = {
            "bad_mask": (~max_step_reached).float(),
            "just_grabbed": set_grab,
        }
        # if dones.any():
        #     average_completed = self.next_step_indices[dones].float().mean()
        #     info["curriculum_metric"] = float(average_completed) - 1
        #     info["episode_rewards"] = self.episode_rewards[dones]

        #     # Reset terminated agents
        #     reset_indices = self._all[dones.squeeze(-1)]
        #     states = self.reset(indices=reset_indices)

        if self.is_rendered or self.img_obs:
            x0, z0 = self.body_positions[0]
            self.body_marker.set_position((x0, 0, z0))

            x, z = current_targets[0]
            self.target_handhold_marker.set_position((x, 0, z))

            x, z = current_handholds[0] - target_lengths[0] * arm_directions[0]
            z = z if self.grab_flags[0] else -1000
            marker_rotation = torch.atan2(*arm_directions[0]) - np.pi / 2
            quat = pybullet.getQuaternionFromEuler([0, marker_rotation, 0])
            self.target_length_marker.set_position((x, 0, z), quat)

            x1, z1 = current_handholds[0]
            x, z = current_handholds[0] - self.max_arm_length * arm_directions[0]
            p1 = (float(x1), 0, float(z1))
            p0 = (float(x), 0, float(z)) if self.grab_flags[0] else p1
            colour = (1, 0.67, 0.06) if gsi[0] % 2 == 0 else (0.02, 0.51, 0.26)
            self._p.addUserDebugLine(p0, p1, colour, lifeTime=0.5)

            x, z = self.body_positions[0]
            cur_body_xyz = (float(x), 0, float(z))
            colour = (1, 0, 0) if self.grab_flags[0] else (0, 0, 1)
            if self.timesteps[0] == 1:
                self._last_body_position = cur_body_xyz
            self._p.addUserDebugLine(
                self._last_body_position,
                cur_body_xyz,
                lifeTime=0,
                lineColorRGB=colour,
            )
            self._last_body_position = cur_body_xyz

            # rewards = rewards[0]
            # dones = dones[0]

        states = self.get_observation_components()
        rewards = rewards.item()
        dones = dones.item()
        return states, rewards, dones, info

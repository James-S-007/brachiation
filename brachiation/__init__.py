import os
import datetime

import gym
import gym.utils.seeding
import numpy as np
import pybullet

from bullet.utils import BulletClient, Camera, OffscreenCamera, StadiumScene


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


for observation_mode in ["FO", "PO", "Asym"]:
    # Multi-Trajectory Base
    register(
        id=f"{observation_mode}MultiTrajGibbon2DCustomEnv-v0",
        entry_point="brachiation.brachiation:Gibbon2DCustomEnv",
        max_episode_steps=1000,
        kwargs={
            'ref_traj': True,
            'traj_num': [10,11,12,13,14,15,16,17,18,19],  # set to [] for random trajectory each time
            'noise_body_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_handholds_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_reftraj_sd': 0.0 if observation_mode == "FO" else 0.05,
            'img_obs': True,
            'is_eval': False,
            'noisy_img': False
        }
    )

    # Multi-Trajectory Eval
    register(
        id=f"{observation_mode}MultiTrajEvalGibbon2DCustomEnv-v0",
        entry_point="brachiation.brachiation:Gibbon2DCustomEnv",
        max_episode_steps=1000,
        kwargs={
            'ref_traj': True,
            'traj_num': [10,11,12,13,14,15,16,17,18,19],  # set to [] for random trajectory each time
            'noise_body_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_handholds_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_reftraj_sd': 0.0 if observation_mode == "FO" else 0.05,
            'img_obs': True,
            'is_eval': True,
            'noisy_img': True
        }
    )

    # Single-Trajectory Base
    register(
        id=f"{observation_mode}SingleTrajGibbon2DCustomEnv-v0",
        entry_point="brachiation.brachiation:Gibbon2DCustomEnv",
        max_episode_steps=1000,
        kwargs={
            'ref_traj': True,
            'traj_num': [10],
            'noise_body_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_handholds_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_reftraj_sd': 0.0 if observation_mode == "FO" else 0.05,
            'img_obs': True,
            'is_eval': False,
            'noisy_img': False
        }
    )

    # Single-Trajectory Eval
    register(
        id=f"{observation_mode}SingleTrajEvalGibbon2DCustomEnv-v0",
        entry_point="brachiation.brachiation:Gibbon2DCustomEnv",
        max_episode_steps=1000,
        kwargs={
            'ref_traj': True,
	    'traj_num': [10],
            # 'traj_num': [10,11,12,13,14,15,16,17,18,19],  # set to [] for random trajectory each time
            'noise_body_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_handholds_sd': 0.0 if observation_mode == "FO" else 0.05,
            'noise_reftraj_sd': 0.0 if observation_mode == "FO" else 0.05,
            'img_obs': True,
            'is_eval': True,
            'noisy_img': True
        }
    )

for observation_mode in ["FO", "PO"]:
    register(
        id=f"{observation_mode}Gibbon2DPointMassEnv-v0",
        entry_point="brachiation.brachiation:Gibbon2DPointMassEnv",
        kwargs={
            'num_parallel': 1,
            'noise_stdev': 0.0 if observation_mode == "FO" else 0.01
        }
    )


class EnvBase(gym.Env):
    def __init__(
        self,
        robot_class,
        render=False,
        remove_ground=False,
        use_egl=False,
        use_ffmpeg=False,
        img_obs=True,
        img_width=80,
        img_height=80,
        camera_dist=1.5,
        noisy_img=True
    ):
        self.robot_class = robot_class

        self.is_rendered = render
        self.remove_ground = remove_ground
        self.use_egl = use_egl
        self.use_ffmpeg = use_ffmpeg
        self.img_obs = img_obs
        self.img_width = img_width
        self.img_height = img_height
        self.camera_dist = camera_dist
        self.noisy_img = noisy_img

        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.keypress_status = {}
        self.force_reset = False

        self.seed()
        self.initialize_scene_and_robot()
        if self.noisy_img:
            self.init_dummy_env()

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        if self.owns_physics_client and self.dummy_physics_client_id is not None and self.dummy_physics_client_id >= 0:
            self._p_dummy.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):

        self.owns_physics_client = True

        bc_mode = pybullet.GUI if self.is_rendered and not self.img_obs else pybullet.DIRECT
        render_fps = 1 / self.control_step
        self._p = BulletClient(bc_mode, use_ffmpeg=self.use_ffmpeg, fps=render_fps)

        if self.is_rendered or self.use_egl or self.img_obs:
            if hasattr(self, "create_target"):
                self.create_target()

        if self.use_egl:
            import pkgutil

            egl = pkgutil.get_loader("eglRenderer")
            self.egl = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.physics_client_id = self._p._client

        pc = self._p
        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SHADOWS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.scene = StadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize(self.remove_ground)

        # Create floor
        if not self.remove_ground:
            self.ground_ids = {(self.scene.id, -1)}

        # Create robot object
        if self.robot_class is not None:
            self.robot = self.robot_class(self._p)
            self.robot.initialize()
            self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        if (self.is_rendered or self.use_egl) and not self.img_obs:
            self.camera = Camera(self._p, render_fps, use_egl=self.use_egl)
        elif self.img_obs:
            self.camera = OffscreenCamera(self._p, render_fps, use_egl=self.use_egl, dist=self.camera_dist, width=self.img_width, height=self.img_height)

        # self.state_id = self._p.saveState()

    def set_env_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_env_param(self, param_name, default):
        return getattr(self, param_name, default)

    def render(self, mode="human"):
        pass

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_mouse(self):
        events = self._p.getMouseEvents()
        for ev in events:
            if (
                ev[0] == 2
                and ev[3] == 0
                and ev[4] == self._p.KEY_WAS_RELEASED
                and self.keypress_status.get(self._p.B3G_SHIFT)
            ):
                # (is mouse click) and (is left click)

                (
                    width,
                    height,
                    _,
                    proj,
                    _,
                    _,
                    _,
                    _,
                    yaw,
                    pitch,
                    dist,
                    target,
                ) = self._p.getDebugVisualizerCamera()

                pitch *= np.pi / 180
                yaw *= np.pi / 180

                R = np.reshape(
                    self._p.getMatrixFromQuaternion(
                        self._p.getQuaternionFromEuler([pitch, 0, yaw])
                    ),
                    (3, 3),
                )

                # Can't use the ones returned by pybullet API, because they are wrong
                camera_up = np.matmul(R, [0, 0, 1])
                camera_forward = np.matmul(R, [0, 1, 0])
                camera_right = np.matmul(R, [1, 0, 0])

                x = ev[1] / width
                y = ev[2] / height

                # calculate from field of view, which should be constant 90 degrees
                # can also get from projection matrix
                # d = 1 / np.tan(np.pi / 2 / 2)
                d = proj[5]

                A = target - camera_forward * dist
                aspect = height / width

                B = (
                    A
                    + camera_forward * d
                    + (x - 0.5) * 2 * camera_right / aspect
                    - (y - 0.5) * 2 * camera_up
                )

                # C = np.array(
                #     [
                #         (B[2] * A[0] - A[2] * B[0]) / (B[2] - A[2]),
                #         (B[2] * A[1] - A[2] * B[1]) / (B[2] - A[2]),
                #         0,
                #     ]
                # )

                if hasattr(self, "target"):
                    for result in self._p.rayTest(A, 2 * dist * (B - A) + A):
                        self.target.set_position(result[3])
                        break

    def _handle_keyboard(self, keys=None, callback=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()

        RELEASED = self._p.KEY_WAS_RELEASED
        self.keypress_status = keys

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif keys.get(ord("r")) == RELEASED:
            self.force_reset = True
            self.done = True
        elif keys.get(ord("f")) == RELEASED:
            self.camera.tracking = not self.camera.tracking
        elif keys.get(self._p.B3G_F1) == RELEASED:
            from imageio import imwrite

            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            imwrite("{}.png".format(now), self.camera.dump_rgb_array())
        elif keys.get(pybullet.B3G_F2) == RELEASED:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._p.startStateLogging(
                self._p.STATE_LOGGING_VIDEO_MP4, "{}.mp4".format(now)
            )
        elif keys.get(ord(" ")) == RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
        elif keys.get(61) == RELEASED:
            # '=' to speed up rendering
            self.camera._fps += 5
            self.camera._target_period = 1 / self.camera._fps
        elif keys.get(45) == RELEASED:
            # '-' to slow down rendering
            self.camera._fps = int(self.camera._fps - 5)
            self.camera._fps = max(self.camera._fps, 5)
            self.camera._target_period = 1 / self.camera._fps
        else:
            if callback is not None:
                callback(keys)

    def get_mirror_indices(self):
        return None


    def init_dummy_env(self):
        """
            Workaround to generate noisy observations without messing with dynamics of true simulator
            Creates an additional simulator client with the same initialization
            Can then update the robot and handhold state to the noisy versions and render images
        """
        if not self.noisy_img:
            self._p_dummy = None
            return
        
        bc_mode = pybullet.GUI if self.is_rendered and not self.img_obs else pybullet.DIRECT
        render_fps = 1 / self.control_step
        self._p_dummy = BulletClient(bc_mode, use_ffmpeg=self.use_ffmpeg, fps=render_fps)

        if self.use_egl:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            self.egl = self._p_dummy.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.dummy_physics_client_id = self._p_dummy._client

        pc = self._p_dummy
        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SHADOWS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.dummy_scene = StadiumScene(
            self._p_dummy,
            gravity=0.0,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.dummy_scene.initialize(self.remove_ground)

        # Create floor
        if not self.remove_ground:
            self.dummy_ground_ids = {(self.dummy_scene.id, -1)}

        # Create robot object
        if self.robot_class is not None:
            self.dummy_robot = self.robot_class(self._p_dummy)
            self.dummy_robot.initialize()
            self.dummy_robot.np_random = self.np_random

        if (self.is_rendered or self.use_egl) and not self.img_obs:
            self.camera_noisy = Camera(self._p_dummy, render_fps, use_egl=self.use_egl)  # only need one camera right?
        elif self.img_obs:
            self.camera_noisy = OffscreenCamera(self._p_dummy, render_fps, use_egl=self.use_egl, dist=self.camera_dist, width=self.img_width, height=self.img_height)

    def set_dummy_robot_state(self, joint_angles, joint_speeds, pos, quat):
        if not self.noisy_img:
            return
        
        pybullet.resetJointStates(
            self.dummy_robot.id,
            self.dummy_robot.joint_ids,
            targetValues=joint_angles,
            targetVelocities=joint_speeds,
            physicsClientId=self.dummy_physics_client_id,
        )

        self._p_dummy.resetBasePositionAndOrientation(self.dummy_robot.id, pos, quat)


    def set_dummy_handholds(self, dummy_handhold_markers, noisy_pos):
        for i, h in enumerate(dummy_handhold_markers):
            h.set_position(noisy_pos[i])
        

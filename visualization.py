import gym
import imageio
from env.brachiation import Gibbon2DCustomEnv
import cv2
import pybullet
import numpy as np
from scipy.spatial.transform import Rotation as R
import moviepy.editor as mpy
import time

NOISE_BODY = 0.05
NOISE_HANDHOLDS = 0.05
NOISE_REFTRAJ = 0.05
NUM_OBSERVATIONS = 100
FPS = 30

DEBUG = False

np.random.seed(seed=int(time.time()))


def main():
    env = Gibbon2DCustomEnv(ref_traj=True, noise_body_sd=NOISE_BODY, noise_handholds_sd=NOISE_HANDHOLDS, noise_reftraj_sd=NOISE_REFTRAJ,\
                            img_obs=True, img_width=720, img_height=720, camera_dist=4.0)

    # Start Env
    imgs = []
    noiseless_img = env.reset()['img']    

    # Remove handmarkers and record noiseless handholds
    handholds_noiseless = [h._pos.copy() for h in env.handhold_markers]
    set_hand_markers_off(env)

    # Generate noisy observations
    for _ in range(NUM_OBSERVATIONS):
        add_noise_robot(env._p, env.robot, env)
        add_noise_handholds(env.handhold_markers, env.target_marker)

        img = env.camera.dump_rgb_array()
        imgs.append(img)

        reset_handholds(env.handhold_markers, handholds_noiseless)
        if DEBUG:
            cv2.imshow(f'Noisy Obs', img)
            cv2.waitKey(0)
    
    clip = mpy.ImageSequenceClip(imgs, fps=FPS)
    clip.write_gif(f'brachiation_noise_{NOISE_BODY}_{NOISE_HANDHOLDS}.gif', fps=FPS)


def add_noise_robot(pybullet_client, robot, env):
    joint_angles = np.copy(robot.base_joint_angles)
    joint_angles += np.random.normal(0.0, NOISE_BODY, joint_angles.shape)
    joint_orientations = np.copy(robot.base_orientation)
    joint_orientations += np.random.normal(0.0, NOISE_BODY, joint_orientations.shape)

    pybullet.resetJointStates(
            robot.id,
            robot.joint_ids,
            targetValues=joint_angles,
            targetVelocities=joint_orientations,
            physicsClientId=pybullet_client._client,
        )
    
    pos = np.array(robot.base_position, dtype='float64')
    # pos_y_i = pos[1]  # don't add noise to y
    pos += np.random.normal(0.0, NOISE_BODY, pos.shape)
    # pos[1] = pos_y_i

    pitch_roll_yaw = np.array(R.from_quat(robot.base_orientation).as_euler("yxz").astype("f4"))
    pitch_roll_yaw += np.random.normal(0.0, NOISE_BODY, pitch_roll_yaw.shape)
    quat = R.from_euler("yzx", pitch_roll_yaw).as_quat()

    pybullet_client.resetBasePositionAndOrientation(robot.id, pos, quat)


def add_noise_handholds(handhold_markers, target_marker):
    for i, h in enumerate(handhold_markers):
        # pos_y_i = h._pos[1]  # don't add noise to y
        pos_n = h._pos + np.random.normal(0.0, NOISE_HANDHOLDS, h._pos.shape)
        # pos_n[1] = pos_y_i
        h.set_position(pos_n)

        if i == 0:
            target_marker.set_position(pos_n)  # set target marker to first handhold


def reset_handholds(handhold_markers, original_pos):
    for i, h in enumerate(handhold_markers):
        h.set_position(original_pos[i])


def set_hand_markers_off(env):
    env.grab_hand_marker.set_position([-10,-10,-10])
    env.free_hand_marker.set_position([-10,-10,-10])
    

def get_joints(pybullet_client, robot_id):
    num_joints = pybullet_client.getNumJoints(robot_id)

    # Retrieve the joint indices and names
    joint_indices = []
    joint_names = []
    for i in range(num_joints):
        joint_info = pybullet_client.getJointInfo(robot_id, i)
        joint_index = joint_info[0]
        joint_name = joint_info[1].decode("utf-8")
        joint_indices.append(joint_index)
        joint_names.append(joint_name)
    
    return joint_indices, joint_names

if __name__ == '__main__':
    main()

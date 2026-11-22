"""SO-101 kinematics. Cartesian positions are in the robot world frame (meters). Arm joints use degrees."""
import numpy as np

def Rx(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    
def Rz(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def make_transform(rotation, displacement):
    return np.block([[rotation, np.array(displacement).reshape(3, 1)],
                     [0, 0, 0, 1]])

def invert_transform(g):
    R = g[:3,:3]
    p = g[:3,3]
    g_inv = np.eye(4)
    g_inv[:3,:3] = R.T
    g_inv[:3,3] = -R.T @ p
    return g_inv

# BASE_HEIGHT = 0.0047625
BASE_HEIGHT = 0.0
# g_ab maps coordinates from frame b into frame a (w = world, t = tool).
def get_gw1(theta1_deg):
    displacement = (0.0388353, 0.0, BASE_HEIGHT + 0.0624)
    rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
    return make_transform(rotation, displacement)

def get_g12(theta2_deg):
    displacement = (-0.0303992, 0.0, -0.0542)
    rotation = Ry(-90) @ Rx(-90) @ Rz(theta2_deg)
    return make_transform(rotation, displacement)
 
def get_g23(theta3_deg):
    displacement = (-0.11257, -0.028, 0.0)
    rotation = Rz(90) @ Rz(theta3_deg)
    return make_transform(rotation, displacement)
 
def get_g34(theta4_deg):
    displacement = (-0.1349, 0.0, 0.0)
    rotation = Rz(-90) @ Rz(theta4_deg)
    return make_transform(rotation, displacement)
 
def get_g45(theta5_deg):
    displacement = (0.0, -0.0611, 0.0)
    rotation = Rz(180) @ Rx(90) @ Rz(theta5_deg)
    return make_transform(rotation, displacement)
 

def get_g5t():
    displacement = (0, 0, -0.1034)
    rotation = np.eye(3)
    return make_transform(rotation, displacement)


def forward_kinematics(joint_positions):
    """Return tool position and rotation in the robot world frame."""
    gw1 = get_gw1(joint_positions['shoulder_pan'])
    g12 = get_g12(joint_positions['shoulder_lift'])
    g23 = get_g23(joint_positions['elbow_flex'])
    g34 = get_g34(joint_positions['wrist_flex'])
    g45 = get_g45(joint_positions['wrist_roll'])
    g5t = get_g5t()
    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation


def get_wrist_flex_pose(target_position):
    target_eef_position = np.array(target_position, dtype=float).copy()
    # target_eef_position[2] += 0.01
    gwt = make_transform(np.eye(3), target_eef_position)
    g4t = get_g45(0) @ get_g5t()
    gw4 = gwt @ np.linalg.inv(g4t)
    wrist_flex_position = gw4[:3,3]
    wrist_flex_rotation = gw4[:3,:3]
    return wrist_flex_position, wrist_flex_rotation

BLOCK_WIDTH = 0.0285  # meters (1.125 inches)
FIXED_JAW_OFFSET = 0.0079  # meters, from the wrist roll axis to the inner face of the fixed jaw

def get_gripper_target(block_position):
    "Offset from the center of the block so that the jaws surround the block"
    gripper_target = list(block_position)
    gripper_target[0] -= BLOCK_WIDTH / 2 - FIXED_JAW_OFFSET
    gripper_target[2] -= BLOCK_WIDTH / 2 
    return gripper_target

def solve_theta4(theta1_deg, theta2_deg, theta3_deg):
    gw4 = (get_gw1(theta1_deg)
           @ get_g12(theta2_deg)
           @ get_g23(theta3_deg)
           @ get_g34(0))

    g4t = get_g45(0) @ get_g5t()

    R_w4 = gw4[:3, :3]
    R_wt = R_w4 @ g4t[:3, :3]

    current_z = R_wt[:, 2]
    desired_z = np.array([0.0, 0.0, 1.0])
    rotation_axis = R_w4[:, 2]

    theta4_rad = np.arctan2(
        np.dot(rotation_axis, np.cross(current_z, desired_z)),
        np.dot(current_z, desired_z)
    )

    return np.rad2deg(theta4_rad)

def get_inverse_kinematics(target_position, target_orientation=np.eye(3)):
    """Solve for a world-frame tool position. target_orientation is currently unused."""
    joint_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }

    target_x, target_y, target_z = target_position
    theta1_deg = -np.rad2deg(np.arctan2(target_y, target_x-0.0388353))
    wrist_flex_position, _ = get_wrist_flex_pose(target_position)
    joint2_pose = get_gw1(theta1_deg) @ get_g12(0)
    joint2_position = joint2_pose[:3,3] 

    l1 = 0.11257
    l2 = 0.028
    l3 = 0.1349
    x, y= np.sqrt(wrist_flex_position[0]**2+wrist_flex_position[1]**2)-np.sqrt(joint2_position[0]**2+joint2_position[1]**2), wrist_flex_position[2]-joint2_position[2]

    alpha = np.rad2deg(np.arctan2(y, x))
    beta = np.rad2deg(np.arctan2(l2, l1))
    phi = beta + 90
    epsilon = np.rad2deg(np.arccos((l1**2 + l2**2 + l3**2 - x**2 - y**2) / (2*np.sqrt(l1**2+l2**2)*l3)))
    ksi = np.rad2deg(np.arccos((l1**2 + l2**2 + x**2 + y**2 - l3**2) / (2*np.sqrt(l1**2+l2**2)*np.sqrt(x**2+y**2))))

    # print(f"alpha={alpha} beta={beta} phi={phi} epsilon={epsilon} ksi={ksi}")
    theta2_deg = (90 - beta - alpha - ksi)
    theta3_deg = -(epsilon - phi)
    theta4_deg = solve_theta4(theta1_deg, theta2_deg, theta3_deg)
    # print(f"theta2={theta2_deg} theta3={theta3_deg}")

    joint_positions['shoulder_pan'] = theta1_deg
    joint_positions["shoulder_lift"] = theta2_deg
    joint_positions["elbow_flex"] = theta3_deg
    joint_positions["wrist_flex"] = theta4_deg
    joint_positions["wrist_roll"] = theta1_deg
    return joint_positions


# Compatibility with earlier lab scripts.
make_pose = make_transform
get_wrist_flex_position = get_wrist_flex_pose


def get_forward_kinematics(position_dict):
    return forward_kinematics(position_dict)

from frankapy import FrankaArm
# 如果禁用了gripper，需要明确指定 with_gripper=False
fa = FrankaArm(with_gripper=False)
joints=fa.get_joints()
print(joints)
# fa.open_gripper()
# fa.reset_joints()


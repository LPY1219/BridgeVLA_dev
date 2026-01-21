from frankapy import FrankaArm
# 如果禁用了gripper，需要明确指定 with_gripper=False
fa = FrankaArm(with_gripper=False)
joints=fa.get_joints()
print(joints)
joints_comma=[ float(joint) for joint in joints]
print(joints_comma)
target_joints_cook =[0.0784198723430451, 0.1658338288898725, 0.37200969253987426, -1.7786963640681988, -0.07054511707834935, 1.9461087191172164, 2.8966576721175743]

# fa.open_gripper()
# fa.reset_joints()
fa.goto_joints(
    target_joints_cook,
    duration=5.0,
    ignore_virtual_walls=True
)

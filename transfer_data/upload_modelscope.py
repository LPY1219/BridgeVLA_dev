from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

YOUR_ACCESS_TOKEN = 'ms-848bce6f-4798-407e-98be-efe5180f7cc2'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)


owner_name = 'LPY1219'
model_name = 'Franka_data_3zed_5'
model_id = f"{owner_name}/{model_name}"


# 创建仓库
# api.create_model(
#     model_id,
#     visibility=ModelVisibility.PUBLIC,
#     license=Licenses.APACHE_V2,
#     chinese_name="我的测试模型"
# )

# 上传文件夹
# api.upload_folder(
#     repo_id=f"{owner_name}/{model_name}",
#     folder_path='/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter',
#     commit_message='upload pour_filter to repo',
# )

#上传文件
api.upload_file(
    path_or_fileobj='/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter.zip',
    path_in_repo='pour_filter.zip',
    repo_id=f"{owner_name}/{model_name}",
    commit_message='upload pour_filter to repo',
)
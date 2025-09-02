from huggingface_hub import HfApi

api = HfApi(token="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")
api.upload_folder(
    folder_path="/home/lpy/BridgeVLA_dev/finetune/Real/logs/update_model",
    repo_id="XiangnanW/Bridgevla_test",
    repo_type="model",
)

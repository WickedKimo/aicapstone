from safetensors.torch import load_file
# 改成你檔案的絕對路徑
try:
    weights = load_file("advanced_policy_v1/pretrained_model/model.safetensors", device="cpu")
    print("【成功】底層讀取完全正常，代表模型沒壞！")
except Exception as e:
    print(f"【失敗】底層也讀不到，錯誤為: {e}")
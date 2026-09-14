import os
import getpass
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("請先安裝 huggingface_hub： pip install huggingface_hub")
    exit(1)

def main():
    print("=== Hugging Face 模型上傳工具 ===")
    # 為了安全，密碼輸入時不會顯示在畫面上
    token = getpass.getpass("請輸入您的 Hugging Face Write Token (輸入時不會顯示，貼上後按 Enter 即可): ")
    if not token.strip():
        print("Token 不可為空，請重新執行。")
        return

    repo_id = input("請輸入您的 Repo ID (例如：你的帳號/你的儲存庫名稱): ").strip()
    if not repo_id:
        print("Repo ID 不可為空。")
        return
        
    api = HfApi(token=token)
    
    # 建立 Repo (如果已存在會略過)
    print("\n正在確認 Repo 狀態...")
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"✅ 成功確認/建立 Repo: {repo_id}")
    except Exception as e:
        print(f"⚠️ 建立/確認 Repo 時發生問題 (若確定 Repo 存在可忽略此訊息): {e}")

    file_paths_str = input("\n請輸入要上傳的檔案「絕對路徑」(若有多個檔案，請用逗號 ',' 分隔):\n> ").strip()
    if not file_paths_str:
        print("未輸入檔案路徑。")
        return
        
    files_to_upload = [path.strip() for path in file_paths_str.split(',') if path.strip()]
    
    valid_files = []
    for path in files_to_upload:
        if not os.path.isfile(path):
            print(f"❌ 找不到檔案或該路徑不是檔案: {path}")
        else:
            valid_files.append(path)
            
    if not valid_files:
        print("沒有有效的檔案可以上傳。")
        return
            
    print("\n準備上傳下列檔案：")
    for path in valid_files:
        print(f" - {path}")
        
    confirm = input("\n確定要上傳嗎？ (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消上傳。")
        return
        
    print("\n開始上傳，請稍候...(檔案較大可能需要一些時間)")
    for file_path in valid_files:
        # 擷取檔案名稱作為 Repo 內的檔名
        file_name = os.path.basename(file_path)
        print(f"正在上傳 {file_name} ...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ {file_name} 上傳成功！")
        except Exception as e:
            print(f"❌ {file_name} 上傳失敗: {e}")

    print(f"\n🎉 恭喜！檔案已全數上傳完成。請前往 https://huggingface.co/{repo_id} 查看。")

if __name__ == "__main__":
    main()

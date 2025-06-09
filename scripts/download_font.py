import os
import requests

def download_font():
    # 创建多个字体目录
    font_dirs = [
        'fonts',
        'static/fonts',
    ]
    
    for font_dir in font_dirs:
        os.makedirs(font_dir, exist_ok=True)
    
    # 主字体文件路径
    main_font_path = os.path.join('fonts', 'SimHei.ttf')
    
    # 如果字体文件已存在，则跳过下载
    if os.path.exists(main_font_path):
        print("字体文件已存在")
        return
    
    # 使用 GitHub 上的开源中文字体
    font_url = 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf'
    
    try:
        print("正在下载字体文件...")
        response = requests.get(font_url, timeout=30)
        response.raise_for_status()
        
        # 保存字体文件到所有目录
        for font_dir in font_dirs:
            font_path = os.path.join(font_dir, 'SimHei.ttf')
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"字体文件已保存到: {font_path}")
        
        print("字体文件下载和复制完成")
        
    except Exception as e:
        print(f"下载字体文件失败: {str(e)}")
        print("\n请尝试以下方法之一：")
        print("1. 手动下载字体文件并放置到 static/fonts/SimHei.ttf")
        print("2. 从 Windows 系统复制：C:\\Windows\\Fonts\\SimHei.ttf")
        print("3. 从 Linux 系统安装：sudo apt-get install fonts-wqy-zenhei")
        print("4. 从 macOS 系统复制：/System/Library/Fonts/PingFang.ttc")
        
        # 尝试从系统字体目录复制
        system_font_paths = [
            "C:\\Windows\\Fonts\\SimHei.ttf",  # Windows
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
            "/System/Library/Fonts/PingFang.ttc"  # macOS
        ]
        
        for system_font in system_font_paths:
            if os.path.exists(system_font):
                print(f"\n找到系统字体：{system_font}")
                print("正在复制到项目目录...")
                try:
                    import shutil
                    shutil.copy2(system_font, main_font_path)
                    print("字体文件复制成功！")
                    return
                except Exception as copy_error:
                    print(f"复制失败: {str(copy_error)}")

if __name__ == "__main__":
    download_font() 
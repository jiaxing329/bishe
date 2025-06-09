import os
import csv
from mutagen.mp3 import MP3
from datetime import datetime

def scan_mp3_directory():
    # 获取当前目录下的static/mp3路径
    mp3_dir = os.path.join('static', 'mp3')
    songs_data = []
    
    # 确保目录存在
    if not os.path.exists(mp3_dir):
        print(f"目录不存在: {mp3_dir}")
        return songs_data
    
    # 扫描所有mp3文件
    for file in os.listdir(mp3_dir):
        if file.endswith('.mp3'):
            file_path = os.path.join(mp3_dir, file)
            try:
                # 读取MP3文件信息
                audio = MP3(file_path)
                
                # 获取文件基本信息
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size / (1024 * 1024)  # 转换为MB
                create_time = datetime.fromtimestamp(file_stats.st_ctime)
                
                # 计算时长
                duration = int(audio.info.length)
                minutes = duration // 60
                seconds = duration % 60
                duration_str = f"{minutes}:{seconds:02d}"
                
                # 获取文件名作为标题（去除.mp3后缀）
                title = os.path.splitext(file)[0]
                
                # 收集歌曲信息
                song_info = {
                    'title': title,
                    'artist': '未知艺术家',
                    'duration': duration_str,
                    'file_size': f"{file_size:.2f}MB",
                    'create_time': create_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'file_path': f'/static/mp3/{file}'
                }
                
                songs_data.append(song_info)
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
    
    return songs_data

def save_to_csv(songs_data, output_file='songs.csv'):
    """将歌曲信息保存到CSV文件"""
    if not songs_data:
        print("没有找到音乐文件")
        return
    
    # CSV文件表头
    fieldnames = ['title', 'artist', 'duration', 'file_size', 'create_time', 'file_path']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(songs_data)
        print(f"数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")

def main():
    # 扫描MP3文件
    print("开始扫描MP3文件...")
    songs_data = scan_mp3_directory()
    
    if songs_data:
        print(f"找到 {len(songs_data)} 个音乐文件")
        # 保存到CSV
        save_to_csv(songs_data)
    else:
        print("未找到任何音乐文件")

if __name__ == "__main__":
    main()

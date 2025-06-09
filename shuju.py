def remove_duplicates_from_sql():
    # 读取SQL文件
    with open('dj舞曲.sql', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 用于存储已见过的歌曲信息
    seen_songs = set()
    # 用于存储不重复的SQL语句
    unique_lines = []
    
    for line in lines:
        if line.strip().startswith('INSERT INTO'):
            # 提取歌曲名和歌手名作为唯一标识
            # 使用split来分割字符串,获取VALUES中的值
            try:
                values = line.split('VALUES (')[1].strip()[:-2]  # 去掉结尾的 ');'
                parts = values.split(',')
                # 获取歌曲名(第2个值)和歌手名(第3个值)
                song_name = parts[1].strip()
                artist_name = parts[2].strip()
                # 创建唯一标识
                song_id = (song_name, artist_name)
                
                # 如果这首歌还没见过，就保留这行
                if song_id not in seen_songs:
                    seen_songs.add(song_id)
                    unique_lines.append(line)
            except:
                # 如果解析失败，保留这行
                unique_lines.append(line)
        else:
            # 非INSERT语句直接保留
            unique_lines.append(line)
    
    # 写回文件
    with open('dj舞曲.sql', 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

if __name__ == "__main__":
    remove_duplicates_from_sql()
    print("重复数据清除完成！")

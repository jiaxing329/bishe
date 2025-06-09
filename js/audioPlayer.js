class AudioPlayer {
    constructor() {
        this.audio = new Audio();
        this.currentMusic = null;
        this.isPlaying = false;
        this.volume = 1;
        this.setupEventListeners();
    }

    setupEventListeners() {
        // 音频播放结束事件
        this.audio.addEventListener('ended', () => {
            this.isPlaying = false;
            // 可以在这里添加播放完成后的逻辑
        });

        // 音频加载错误事件
        this.audio.addEventListener('error', () => {
            console.error('音频加载失败');
            alert('音频加载失败，请稍后重试');
        });
    }

    playMusic(music) {
        if (!music || !music.music_url) {
            console.error('无效的音乐数据');
            return;
        }

        // 如果是新的音乐，则加载并播放
        if (!this.currentMusic || this.currentMusic.id !== music.id) {
            this.currentMusic = music;
            this.audio.src = music.music_url;
            this.audio.load();
            
            // 记录播放历史
            this.recordPlayHistory(music.id);
        }

        // 播放音频
        this.audio.play()
            .then(() => {
                this.isPlaying = true;
            })
            .catch(error => {
                console.error('播放失败:', error);
                alert('播放失败，请稍后重试');
            });
    }

    pauseMusic() {
        if (this.isPlaying) {
            this.audio.pause();
            this.isPlaying = false;
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pauseMusic();
        } else if (this.currentMusic) {
            this.playMusic(this.currentMusic);
        }
    }

    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        this.audio.volume = this.volume;
    }

    getCurrentTime() {
        return this.audio.currentTime;
    }

    getDuration() {
        return this.audio.duration;
    }

    seekTo(time) {
        if (time >= 0 && time <= this.audio.duration) {
            this.audio.currentTime = time;
        }
    }

    // 记录播放历史
    recordPlayHistory(musicId) {
        // 从localStorage获取用户信息
        const userInfo = JSON.parse(localStorage.getItem('userInfo') || '{}');
        if (!userInfo.id) {
            console.log('用户未登录，不记录播放历史');
            return;
        }

        // 发送请求记录播放历史
        fetch('/api/add_play_history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                music_id: musicId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.code !== 200) {
                console.error('记录播放历史失败:', data.msg);
            }
        })
        .catch(error => {
            console.error('记录播放历史请求失败:', error);
        });
    }
}

// 创建全局播放器实例
window.globalAudioPlayer = new AudioPlayer(); 
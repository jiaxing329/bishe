const audioPlayer = document.getElementById('audio-player');
const playPauseBtn = document.getElementById('play-pause');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');
const progress = document.querySelector('.progress');
const progressBar = document.querySelector('.progress-bar');

// 示例歌曲列表
const songs = [
    {
        title: '示例歌曲 1',
        artist: '艺术家 1',
        url: 'path/to/song1.mp3'
    },
    {
        title: '示例歌曲 2',
        artist: '艺术家 2',
        url: 'path/to/song2.mp3'
    }
];

let currentSongIndex = 0;

// 播放/暂停功能
playPauseBtn.addEventListener('click', () => {
    if (audioPlayer.paused) {
        audioPlayer.play();
        playPauseBtn.textContent = '暂停';
    } else {
        audioPlayer.pause();
        playPauseBtn.textContent = '播放';
    }
});

// 更新进度条
audioPlayer.addEventListener('timeupdate', () => {
    const progressPercent = (audioPlayer.currentTime / audioPlayer.duration) * 100;
    progress.style.width = `${progressPercent}%`;
});

// 点击进度条跳转
progressBar.addEventListener('click', (e) => {
    const progressWidth = progressBar.clientWidth;
    const clickX = e.offsetX;
    const duration = audioPlayer.duration;
    
    audioPlayer.currentTime = (clickX / progressWidth) * duration;
});

// 加载歌曲
function loadSong(song) {
    document.getElementById('song-title').textContent = song.title;
    document.getElementById('artist').textContent = song.artist;
    audioPlayer.src = song.url;
}

// 下一首
nextBtn.addEventListener('click', () => {
    currentSongIndex++;
    if (currentSongIndex > songs.length - 1) {
        currentSongIndex = 0;
    }
    loadSong(songs[currentSongIndex]);
    audioPlayer.play();
    playPauseBtn.textContent = '暂停';
});

// 上一首
prevBtn.addEventListener('click', () => {
    currentSongIndex--;
    if (currentSongIndex < 0) {
        currentSongIndex = songs.length - 1;
    }
    loadSong(songs[currentSongIndex]);
    audioPlayer.play();
    playPauseBtn.textContent = '暂停';
});

// 初始加载第一首歌
loadSong(songs[currentSongIndex]); 
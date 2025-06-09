// 处理用户登录
function handleLogin(username, password) {
    return fetch('/user/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.code === 200) {
            // 登录成功，将用户信息存储到本地存储
            localStorage.setItem('userInfo', JSON.stringify(data.data));
            return true;
        } else {
            throw new Error(data.msg);
        }
    });
}

// 获取当前登录用户信息
function getCurrentUser() {
    const userStr = localStorage.getItem('userInfo');
    return userStr ? JSON.parse(userStr) : null;
}

// 获取当前用户ID
function getCurrentUserId() {
    const user = getCurrentUser();
    return user ? user.id : null;
}

// 退出登录
function logout() {
    return fetch('/user/logout', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.code === 200) {
            // 清除本地存储的用户信息
            localStorage.removeItem('userInfo');
            return true;
        }
        throw new Error(data.msg);
    });
} 
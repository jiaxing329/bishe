import psutil
import time

class SystemInfo:
    @staticmethod
    def get_system_info():
        try:
            # 获取CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_info = {
                'name': f"CPU ({psutil.cpu_count()} cores)",  # CPU名称和核心数
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'freq_current': round(cpu_freq.current if cpu_freq else 0, 2),
                'freq_max': round(cpu_freq.max if cpu_freq else 0, 2),
                'percent': cpu_percent
            }
            
            # 获取内存信息
            memory = psutil.virtual_memory()
            memory_info = {
                'total': round(memory.total / (1024.0 * 1024.0 * 1024.0), 2),  # GB
                'used': round(memory.used / (1024.0 * 1024.0 * 1024.0), 2),    # GB
                'free': round(memory.available / (1024.0 * 1024.0 * 1024.0), 2),  # GB
                'percent': memory.percent
            }
            
            # 获取磁盘信息
            disk = psutil.disk_usage('/')
            disk_usage = {
                'total': round(disk.total / (1024.0 * 1024.0 * 1024.0), 2),      # GB
                'used': round(disk.used / (1024.0 * 1024.0 * 1024.0), 2),        # GB
                'free': round(disk.free / (1024.0 * 1024.0 * 1024.0), 2),        # GB
                'percent': disk.percent
            }
            
            # 获取网络信息
            net_io = psutil.net_io_counters()
            network_info = {
                'sent': round(net_io.bytes_sent / (1024.0 * 1024.0), 2),        # MB
                'received': round(net_io.bytes_recv / (1024.0 * 1024.0), 2)      # MB
            }
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_usage,
                'network': network_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"获取系统信息时发生错误: {str(e)}")
            # 返回基本信息
            return {
                'cpu': {
                    'name': 'Unknown',
                    'cores': 0,
                    'threads': 0,
                    'freq_current': 0,
                    'freq_max': 0,
                    'percent': 0
                },
                'memory': {
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'percent': 0
                },
                'disk': {
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'percent': 0
                },
                'network': {
                    'sent': 0,
                    'received': 0
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            } 
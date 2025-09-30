"""
記憶體監控工具
用於追蹤和診斷記憶體使用情況
"""
import torch
import psutil
import gc
from typing import Dict, Any

class MemoryMonitor:
    def __init__(self):
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """獲取當前記憶體使用情況"""
        # 系統記憶體
        system_memory = psutil.virtual_memory()
        
        # CUDA 記憶體
        cuda_memory = {}
        if torch.cuda.is_available():
            cuda_memory = {
                'cuda_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cuda_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'cuda_max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        
        return {
            'system_used': system_memory.used / 1024**3,      # GB
            'system_available': system_memory.available / 1024**3,  # GB
            'system_percent': system_memory.percent,
            **cuda_memory
        }
    
    def print_memory_status(self, label: str = ""):
        """打印記憶體狀態"""
        memory = self.get_memory_usage()
        print(f"\n=== 記憶體狀態 {label} ===")
        print(f"系統記憶體: {memory['system_used']:.2f}GB / {memory['system_available']:.2f}GB ({memory['system_percent']:.1f}%)")
        
        if 'cuda_allocated' in memory:
            print(f"CUDA 已分配: {memory['cuda_allocated']:.2f}GB")
            print(f"CUDA 已保留: {memory['cuda_reserved']:.2f}GB")
            print(f"CUDA 最大分配: {memory['cuda_max_allocated']:.2f}GB")
        
        # 計算記憶體增長
        if hasattr(self, 'last_memory'):
            system_growth = memory['system_used'] - self.last_memory['system_used']
            if 'cuda_allocated' in memory:
                cuda_growth = memory['cuda_allocated'] - self.last_memory.get('cuda_allocated', 0)
                print(f"CUDA 記憶體增長: {cuda_growth:.2f}GB")
            print(f"系統記憶體增長: {system_growth:.2f}GB")
        
        self.last_memory = memory
    
    def force_cleanup(self):
        """強制清理記憶體"""
        print("執行強制記憶體清理...")
        
        # Python 垃圾回收
        collected = gc.collect()
        print(f"垃圾回收: 清理了 {collected} 個對象")
        
        # CUDA 記憶體清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA 快取已清理")
        
        # 打印清理後的狀態
        self.print_memory_status("(清理後)")
    
    def check_memory_leak(self, threshold_gb: float = 1.0):
        """檢查是否有記憶體洩漏"""
        current = self.get_memory_usage()
        if hasattr(self, 'last_memory'):
            system_growth = current['system_used'] - self.last_memory['system_used']
            if system_growth > threshold_gb:
                print(f"⚠️  警告: 系統記憶體增長 {system_growth:.2f}GB，可能發生記憶體洩漏！")
                return True
        return False

# 使用範例
if __name__ == "__main__":
    monitor = MemoryMonitor()
    monitor.print_memory_status("(初始)")
    
    # 模擬一些操作
    x = torch.randn(1000, 1000).cuda()
    monitor.print_memory_status("(創建張量後)")
    
    # 清理
    del x
    monitor.force_cleanup()

# MouseScope Desktop — 打包分发指南

## 快速打包（Windows）

### 前提条件
- Python 3.9 ~ 3.11（推荐 3.10）
- Windows 10/11

### 步骤

1. **双击运行** `build_exe.bat`，等待打包完成（约 3-5 分钟）。

2. 打包完成后，`dist/MouseScope/` 文件夹就是最终产物。

3. **分发方式**：将 `dist/MouseScope/` 整个文件夹压缩为 ZIP 发给用户。

### 用户使用
1. 解压 ZIP
2. 双击 `MouseScope.exe`
3. 自动弹出浏览器访问 `http://localhost:8080`

---

## 手动打包命令

```bash
# 安装依赖
pip install -r requirements.txt
pip install pyinstaller

# 打包
pyinstaller MouseScope.spec

# 或者不用 spec 文件，直接命令行打包
pyinstaller --name MouseScope \
    --add-data "templates;templates" \
    --add-data "static;static" \
    --add-data "core;core" \
    --add-data "LICENSE;." \
    --hidden-import flask \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import pandas \
    --hidden-import vision_tracker \
    --hidden-import performance_utils \
    --exclude-module tkinter \
    --exclude-module matplotlib \
    --exclude-module torch \
    --console \
    app.py
```

> ⚠ **注意**：`--add-data` 分隔符在 Windows 上是 `;`，Linux/Mac 上是 `:`。

---

## 高级选项

### 1. 单文件模式（--onefile）
可以打成单个 exe，但启动较慢（需解压临时文件）：
```bash
pyinstaller --onefile --name MouseScope ...
```

### 2. 隐藏控制台窗口
在 `MouseScope.spec` 中将 `console=True` 改为 `console=False`，
或命令行加 `--noconsole`。

### 3. 添加程序图标
准备一个 `.ico` 文件放到 `static/img/icon.ico`，
然后在 spec 文件中取消 `icon=` 那行的注释。

### 4. 包含 YOLO 模型
如果需要 YOLO 功能，在 spec 文件中：
- 从 `excludes` 列表中移除 `'ultralytics'` 和 `'torch'`
- 在 `datas` 中添加模型文件路径
- ⚠ 这会让打包体积从 ~200MB 增大到 ~2GB+

---

## 常见问题

### Q: 打包后的 exe 有多大？
不含 YOLO/PyTorch 约 **150-250 MB**（主要是 OpenCV + NumPy）。

### Q: 打包后运行报 "找不到模板" 错误？
确保 `app.py` 中已有 `sys._MEIPASS` 路径兼容代码（已添加）。

### Q: 杀毒软件误报？
PyInstaller 打包的 exe 可能被误报。可以：
- 对 exe 进行代码签名
- 提交到杀毒厂商的白名单
- 使用 `--key` 参数加密字节码

### Q: 用户电脑需要装 Python 吗？
**不需要**。PyInstaller 会把 Python 解释器和所有依赖一起打包。
